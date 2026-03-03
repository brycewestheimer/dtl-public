// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_reduce_correctness.cpp
/// @brief Phase 06 correctness tests for reduce (T01: silent fallback, T02: product init)

#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <vector>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;
    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// T02: Product reduction uses correct identity element (1, not 0)
// =============================================================================

TEST(ReduceProductTest, ProductIdentityIsOne) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 3; lv[2] = 4;

    // product() should use init=1
    int result = dtl::product(vec);
    EXPECT_EQ(result, 24);  // 2 * 3 * 4 = 24, NOT 0
}

TEST(ReduceProductTest, ProductAllOnes) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = 1;

    int result = dtl::product(vec);
    EXPECT_EQ(result, 1);
}

TEST(ReduceProductTest, ProductWithExplicitInitOne) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 5; lv[2] = 7;

    int result = dtl::reduce(dtl::seq{}, vec, 1, std::multiplies<>{});
    EXPECT_EQ(result, 70);  // 2 * 5 * 7 = 70
}

TEST(ReduceProductTest, ProductSingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;

    int result = dtl::product(vec);
    EXPECT_EQ(result, 42);
}

TEST(ReduceProductTest, ProductEmpty) {
    distributed_vector<int> vec(0, test_context{0, 1});
    // Product of empty range should return identity (1)
    int result = dtl::product(vec);
    EXPECT_EQ(result, 1);
}

// =============================================================================
// Sum reduction remains correct (regression tests)
// =============================================================================

TEST(ReduceRegressionTest, SumStillWorks) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    int result = dtl::sum(vec);
    EXPECT_EQ(result, 15);
}

TEST(ReduceRegressionTest, MinStillWorks) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto result = dtl::min_element(vec);
    EXPECT_EQ(result, 1);
}

TEST(ReduceRegressionTest, MaxStillWorks) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 9; lv[3] = 4; lv[4] = 2;

    auto result = dtl::max_element(vec);
    EXPECT_EQ(result, 9);
}

// =============================================================================
// Local reduce with custom ops still works (no error for non-distributed)
// =============================================================================

TEST(LocalReduceTest, CustomOpWorksLocally) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    // local_reduce with custom op should still work since no MPI dispatch
    int result = dtl::local_reduce(vec, 0, [](int a, int b) { return a + b * b; });
    EXPECT_EQ(result, 55);  // 1 + 4 + 9 + 16 + 25
}

// =============================================================================
// T01: Verify standalone reduce (no communicator) works for all standard ops
// =============================================================================

TEST(ReduceStandaloneTest, StandalonePlusWorks) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 10; lv[1] = 20; lv[2] = 30;

    int result = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(result, 60);
}

TEST(ReduceStandaloneTest, StandaloneMultipliesWorks) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 3; lv[2] = 4;

    int result = dtl::reduce(dtl::seq{}, vec, 1, std::multiplies<>{});
    EXPECT_EQ(result, 24);
}

TEST(ReduceStandaloneTest, StandaloneReduceMinWorks) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 1; lv[2] = 3;

    int result = dtl::reduce(dtl::seq{}, vec,
                             reduce_min<int>::identity(), reduce_min<int>{});
    EXPECT_EQ(result, 1);
}

TEST(ReduceStandaloneTest, StandaloneReduceMaxWorks) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 1; lv[2] = 3;

    int result = dtl::reduce(dtl::seq{}, vec,
                             reduce_max<int>::identity(), reduce_max<int>{});
    EXPECT_EQ(result, 5);
}

}  // namespace dtl::test
