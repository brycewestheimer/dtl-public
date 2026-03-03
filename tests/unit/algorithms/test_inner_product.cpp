// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_inner_product.cpp
/// @brief Unit tests for distributed inner_product / dot algorithm (R6.2)

#include <dtl/algorithms/reductions/inner_product.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

#include <cmath>
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
// Basic Inner Product Tests
// =============================================================================

TEST(InnerProductTest, BasicIntegerProduct) {
    // a = [1, 2, 3], b = [4, 5, 6]
    // inner_product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3;

    auto lb = b.local_view();
    lb[0] = 4; lb[1] = 5; lb[2] = 6;

    int result = dtl::inner_product(seq{}, a, b, 0);
    EXPECT_EQ(result, 32);
}

TEST(InnerProductTest, WithInitValue) {
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3;

    auto lb = b.local_view();
    lb[0] = 4; lb[1] = 5; lb[2] = 6;

    // Same as above but with init = 100
    int result = dtl::inner_product(seq{}, a, b, 100);
    EXPECT_EQ(result, 132);  // 100 + 32
}

TEST(InnerProductTest, ZeroVectors) {
    test_context ctx{0, 1};
    distributed_vector<int> a(5, 0, ctx);
    distributed_vector<int> b(5, 0, ctx);

    int result = dtl::inner_product(seq{}, a, b, 0);
    EXPECT_EQ(result, 0);
}

TEST(InnerProductTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> a(1, 7, ctx);
    distributed_vector<int> b(1, 3, ctx);

    int result = dtl::inner_product(seq{}, a, b, 0);
    EXPECT_EQ(result, 21);  // 7 * 3
}

TEST(InnerProductTest, EmptyContainers) {
    test_context ctx{0, 1};
    distributed_vector<int> a(0, ctx);
    distributed_vector<int> b(0, ctx);

    int result = dtl::inner_product(seq{}, a, b, 42);
    EXPECT_EQ(result, 42);  // Just the init value
}

TEST(InnerProductTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3;

    auto lb = b.local_view();
    lb[0] = 4; lb[1] = 5; lb[2] = 6;

    // Default execution (no policy)
    int result = dtl::inner_product(seq{}, a, b, 0);
    EXPECT_EQ(result, 32);
}

// =============================================================================
// Double Precision Tests
// =============================================================================

TEST(InnerProductTest, DoubleValues) {
    test_context ctx{0, 1};
    distributed_vector<double> a(3, 0.0, ctx);
    distributed_vector<double> b(3, 0.0, ctx);

    auto la = a.local_view();
    la[0] = 1.0; la[1] = 2.0; la[2] = 3.0;

    auto lb = b.local_view();
    lb[0] = 0.5; lb[1] = 0.5; lb[2] = 0.5;

    double result = dtl::inner_product(seq{}, a, b, 0.0);
    EXPECT_DOUBLE_EQ(result, 3.0);  // 0.5 + 1.0 + 1.5
}

TEST(InnerProductTest, UnitVectorDot) {
    // Dot product of unit vectors: should be 1.0
    test_context ctx{0, 1};
    distributed_vector<double> a(3, 0.0, ctx);
    distributed_vector<double> b(3, 0.0, ctx);

    auto la = a.local_view();
    la[0] = 1.0; la[1] = 0.0; la[2] = 0.0;

    auto lb = b.local_view();
    lb[0] = 1.0; lb[1] = 0.0; lb[2] = 0.0;

    double result = dtl::inner_product(seq{}, a, b, 0.0);
    EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST(InnerProductTest, OrthogonalVectors) {
    // Orthogonal vectors: dot product should be 0
    test_context ctx{0, 1};
    distributed_vector<double> a(3, 0.0, ctx);
    distributed_vector<double> b(3, 0.0, ctx);

    auto la = a.local_view();
    la[0] = 1.0; la[1] = 0.0; la[2] = 0.0;

    auto lb = b.local_view();
    lb[0] = 0.0; lb[1] = 1.0; lb[2] = 0.0;

    double result = dtl::inner_product(seq{}, a, b, 0.0);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

// =============================================================================
// Custom Operations Tests
// =============================================================================

TEST(InnerProductTest, CustomOpsMinPlus) {
    // Custom "min-plus" inner product: op1=min, op2=plus
    // result = min(init, min(a[0]+b[0], min(a[1]+b[1], ...)))
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 10; la[1] = 20; la[2] = 30;

    auto lb = b.local_view();
    lb[0] = 5; lb[1] = 3; lb[2] = 1;

    // op1 = min, op2 = plus
    // Computes: min(1000, min(10+5, min(20+3, 30+1))) = min(1000, 15, 23, 31) = 15
    int result = dtl::inner_product(seq{}, a, b, 1000,
                                    [](int acc, int val) { return std::min(acc, val); },
                                    std::plus<int>{});
    EXPECT_EQ(result, 15);
}

TEST(InnerProductTest, CustomOpsSumMax) {
    // op1 = sum, op2 = max
    // result = init + max(a[0],b[0]) + max(a[1],b[1]) + ...
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 1; la[1] = 5; la[2] = 3;

    auto lb = b.local_view();
    lb[0] = 4; lb[1] = 2; lb[2] = 6;

    // result = 0 + max(1,4) + max(5,2) + max(3,6) = 0 + 4 + 5 + 6 = 15
    int result = dtl::inner_product(seq{}, a, b, 0,
                                    std::plus<int>{},
                                    [](int x, int y) { return std::max(x, y); });
    EXPECT_EQ(result, 15);
}

TEST(InnerProductTest, CustomOpsDefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> a(2, 0, ctx);
    distributed_vector<int> b(2, 0, ctx);

    auto la = a.local_view();
    la[0] = 3; la[1] = 7;

    auto lb = b.local_view();
    lb[0] = 2; lb[1] = 4;

    // Default execution with custom ops
    int result = dtl::inner_product(seq{}, a, b, 0,
                                    std::plus<int>{}, std::multiplies<int>{});
    EXPECT_EQ(result, 34);  // 3*2 + 7*4 = 6 + 28
}

// =============================================================================
// Dot Product Named Algorithm Tests
// =============================================================================

TEST(DotProductTest, BasicDot) {
    test_context ctx{0, 1};
    distributed_vector<int> a(3, 0, ctx);
    distributed_vector<int> b(3, 0, ctx);

    auto la = a.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3;

    auto lb = b.local_view();
    lb[0] = 4; lb[1] = 5; lb[2] = 6;

    int result = dtl::dot(seq{}, a, b);
    EXPECT_EQ(result, 32);
}

TEST(DotProductTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<double> a(2, 0.0, ctx);
    distributed_vector<double> b(2, 0.0, ctx);

    auto la = a.local_view();
    la[0] = 3.0; la[1] = 4.0;

    auto lb = b.local_view();
    lb[0] = 3.0; lb[1] = 4.0;

    double result = dtl::dot(seq{}, a, b);
    EXPECT_DOUBLE_EQ(result, 25.0);  // 9 + 16
}

// =============================================================================
// Larger Container Tests
// =============================================================================

TEST(InnerProductTest, LargerContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> a(100, 1, ctx);  // All ones
    distributed_vector<int> b(100, 2, ctx);  // All twos

    int result = dtl::inner_product(seq{}, a, b, 0);
    EXPECT_EQ(result, 200);  // 100 * (1*2)
}

TEST(InnerProductTest, MultiRankLocalOnly) {
    // Multi-rank without communicator is intentionally unsupported.
    test_context ctx{2, 4};  // Rank 2 of 4
    distributed_vector<int> a(100, 3, ctx);  // 25 elements per rank, all 3
    distributed_vector<int> b(100, 4, ctx);  // 25 elements per rank, all 4

    EXPECT_THROW((void)dtl::inner_product(seq{}, a, b, 0), std::runtime_error);
}

}  // namespace dtl::test
