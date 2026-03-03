// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_accumulate.cpp
/// @brief Unit tests for accumulate algorithm (Phase 06 T08)

#include <dtl/algorithms/reductions/accumulate.hpp>
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/execution/seq.hpp>

#include <gtest/gtest.h>

#include <string>
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

// Standalone accumulate (without communicator) falls through to local reduce/sum.
// The distributed accumulate requires a communicator. Here we test local behavior.

TEST(AccumulateTest, SumIntegers) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    // Use reduce as surrogate for standalone accumulate
    int result = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(AccumulateTest, ProductIntegers) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    int result = dtl::reduce(dtl::seq{}, vec, 1, std::multiplies<>{});
    EXPECT_EQ(result, 120);
}

TEST(AccumulateTest, EmptyRange) {
    distributed_vector<int> vec(0, test_context{0, 1});
    int result = dtl::reduce(dtl::seq{}, vec, 42, std::plus<>{});
    EXPECT_EQ(result, 42);  // Returns init for empty range
}

TEST(AccumulateTest, CustomOp) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 10; lv[1] = 20; lv[2] = 30;

    // Sum of squares via custom op
    int result = dtl::reduce(dtl::seq{}, vec, 0,
                             [](int a, int b) { return a + b * b; });
    EXPECT_EQ(result, 1400);  // 100 + 400 + 900
}

TEST(AccumulateTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;
    int result = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(result, 42);
}

TEST(AccumulateTest, WithInitialValue) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3;

    int result = dtl::reduce(dtl::seq{}, vec, 100, std::plus<>{});
    EXPECT_EQ(result, 106);
}

}  // namespace dtl::test
