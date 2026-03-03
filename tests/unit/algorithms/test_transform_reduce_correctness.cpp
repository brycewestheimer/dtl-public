// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_transform_reduce_correctness.cpp
/// @brief Phase 06 correctness tests for transform_reduce (T05)

#include <dtl/algorithms/reductions/transform_reduce.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <gtest/gtest.h>

#include <functional>
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
// Unary transform_reduce tests (standalone, no communicator)
// =============================================================================

TEST(TransformReduceTest, SumOfSquares) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    int result = dtl::transform_reduce(dtl::seq{}, vec, 0, std::plus<>{},
                                        [](int x) { return x * x; });
    EXPECT_EQ(result, 55);  // 1 + 4 + 9 + 16 + 25
}

TEST(TransformReduceTest, SumOfAbsoluteValues) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = -3; lv[1] = -2; lv[2] = 1; lv[3] = 4;

    int result = dtl::transform_reduce(dtl::seq{}, vec, 0, std::plus<>{},
                                        [](int x) { return x < 0 ? -x : x; });
    EXPECT_EQ(result, 10);  // 3 + 2 + 1 + 4
}

TEST(TransformReduceTest, IdentityTransform) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 10; lv[1] = 20; lv[2] = 30;

    int result = dtl::transform_reduce(dtl::seq{}, vec, 0, std::plus<>{},
                                        [](int x) { return x; });
    EXPECT_EQ(result, 60);
}

TEST(TransformReduceTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    int result = dtl::transform_reduce(dtl::seq{}, vec, 42, std::plus<>{},
                                        [](int x) { return x * x; });
    EXPECT_EQ(result, 42);  // Returns init
}

TEST(TransformReduceTest, ProductTransformReduce) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 3; lv[2] = 4;

    // Product of doubled values (standalone)
    int result = dtl::transform_reduce(dtl::seq{}, vec, 1, std::multiplies<>{},
                                        [](int x) { return x * 2; });
    EXPECT_EQ(result, 4 * 6 * 8);  // 192
}

// =============================================================================
// Binary transform_reduce tests (inner product pattern)
// =============================================================================

TEST(BinaryTransformReduceTest, DotProduct) {
    distributed_vector<int> a(4, test_context{0, 1});
    distributed_vector<int> b(4, test_context{0, 1});
    auto la = a.local_view();
    auto lb = b.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3; la[3] = 4;
    lb[0] = 5; lb[1] = 6; lb[2] = 7; lb[3] = 8;

    int result =
        dtl::transform_reduce(dtl::seq{}, a, b, 0, std::plus<>{}, std::multiplies<>{});
    EXPECT_EQ(result, 70);  // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
}

TEST(BinaryTransformReduceTest, InnerProduct) {
    distributed_vector<double> a(3, test_context{0, 1});
    distributed_vector<double> b(3, test_context{0, 1});
    auto la = a.local_view();
    auto lb = b.local_view();
    la[0] = 1.0; la[1] = 0.0; la[2] = 0.0;
    lb[0] = 0.0; lb[1] = 1.0; lb[2] = 0.0;

    double result = dtl::inner_product(dtl::seq{}, a, b, 0.0);
    EXPECT_DOUBLE_EQ(result, 0.0);  // Orthogonal vectors
}

TEST(BinaryTransformReduceTest, SelfDotProduct) {
    distributed_vector<double> a(3, test_context{0, 1});
    auto la = a.local_view();
    la[0] = 3.0; la[1] = 4.0; la[2] = 0.0;

    double result = dtl::inner_product(dtl::seq{}, a, a, 0.0);
    EXPECT_DOUBLE_EQ(result, 25.0);  // 9 + 16 + 0 = 25
}

// =============================================================================
// Common patterns
// =============================================================================

TEST(TransformReducePatternTest, SumOfSquaresPattern) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    auto result = dtl::sum_of_squares(dtl::seq{}, vec);
    EXPECT_EQ(result, 55);
}

TEST(TransformReducePatternTest, SumOfAbsPattern) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = -3; lv[1] = -2; lv[2] = 1; lv[3] = 4;

    auto result = dtl::sum_of_abs(dtl::seq{}, vec);
    EXPECT_EQ(result, 10);
}

}  // namespace dtl::test
