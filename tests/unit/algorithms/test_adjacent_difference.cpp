// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_adjacent_difference.cpp
/// @brief Unit tests for distributed adjacent_difference algorithm (R6.1)

#include <dtl/algorithms/modifying/adjacent_difference.hpp>
#include <dtl/algorithms/modifying/fill.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

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
// Basic Adjacent Difference Tests
// =============================================================================

TEST(AdjacentDifferenceTest, BasicSequence) {
    // Input: [1, 4, 9, 16, 25]
    // Expected: [1, 3, 5, 7, 9]
    test_context ctx{0, 1};
    distributed_vector<int> input(5, 0, ctx);
    distributed_vector<int> output(5, 0, ctx);

    auto local_in = input.local_view();
    local_in[0] = 1;
    local_in[1] = 4;
    local_in[2] = 9;
    local_in[3] = 16;
    local_in[4] = 25;

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 5u);
    EXPECT_TRUE(res.value().success);

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);   // First element is copied
    EXPECT_EQ(local_out[1], 3);   // 4 - 1
    EXPECT_EQ(local_out[2], 5);   // 9 - 4
    EXPECT_EQ(local_out[3], 7);   // 16 - 9
    EXPECT_EQ(local_out[4], 9);   // 25 - 16
}

TEST(AdjacentDifferenceTest, ConstantInput) {
    test_context ctx{0, 1};
    distributed_vector<int> input(5, 42, ctx);
    distributed_vector<int> output(5, 0, ctx);

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 42);  // First element copied
    for (size_type i = 1; i < 5; ++i) {
        EXPECT_EQ(local_out[i], 0);  // 42 - 42 = 0
    }
}

TEST(AdjacentDifferenceTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> input(1, 99, ctx);
    distributed_vector<int> output(1, 0, ctx);

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 1u);

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 99);
}

TEST(AdjacentDifferenceTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> input(0, ctx);
    distributed_vector<int> output(0, ctx);

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 0u);
    EXPECT_TRUE(res.value().success);
}

TEST(AdjacentDifferenceTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> input(5, 0, ctx);
    distributed_vector<int> output(5, 0, ctx);

    auto local_in = input.local_view();
    for (size_type i = 0; i < 5; ++i) {
        local_in[i] = static_cast<int>(i * i);
    }

    // Uses default seq execution
    auto res = dtl::adjacent_difference(input, output);
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 0);   // 0
    EXPECT_EQ(local_out[1], 1);   // 1 - 0
    EXPECT_EQ(local_out[2], 3);   // 4 - 1
    EXPECT_EQ(local_out[3], 5);   // 9 - 4
    EXPECT_EQ(local_out[4], 7);   // 16 - 9
}

// =============================================================================
// Custom Binary Operation Tests
// =============================================================================

TEST(AdjacentDifferenceTest, CustomBinaryOpPlus) {
    // Instead of subtraction, use addition: output[i] = input[i] + input[i-1]
    test_context ctx{0, 1};
    distributed_vector<int> input(4, 0, ctx);
    distributed_vector<int> output(4, 0, ctx);

    auto local_in = input.local_view();
    local_in[0] = 1;
    local_in[1] = 2;
    local_in[2] = 3;
    local_in[3] = 4;

    auto res = dtl::adjacent_difference(seq{}, input, output,
                                        [](int a, int b) { return a + b; });
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);   // First element copied
    EXPECT_EQ(local_out[1], 3);   // 2 + 1
    EXPECT_EQ(local_out[2], 5);   // 3 + 2
    EXPECT_EQ(local_out[3], 7);   // 4 + 3
}

TEST(AdjacentDifferenceTest, CustomBinaryOpMultiply) {
    test_context ctx{0, 1};
    distributed_vector<int> input(4, 0, ctx);
    distributed_vector<int> output(4, 0, ctx);

    auto local_in = input.local_view();
    local_in[0] = 2;
    local_in[1] = 3;
    local_in[2] = 4;
    local_in[3] = 5;

    auto res = dtl::adjacent_difference(seq{}, input, output,
                                        std::multiplies<int>{});
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 2);   // First element copied
    EXPECT_EQ(local_out[1], 6);   // 3 * 2
    EXPECT_EQ(local_out[2], 12);  // 4 * 3
    EXPECT_EQ(local_out[3], 20);  // 5 * 4
}

TEST(AdjacentDifferenceTest, CustomBinaryOpDefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> input(3, 0, ctx);
    distributed_vector<int> output(3, 0, ctx);

    auto local_in = input.local_view();
    local_in[0] = 10;
    local_in[1] = 20;
    local_in[2] = 30;

    // Uses default seq execution with custom op
    auto res = dtl::adjacent_difference(input, output,
                                        [](int a, int b) { return a - b; });
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 10);
    EXPECT_EQ(local_out[1], 10);  // 20 - 10
    EXPECT_EQ(local_out[2], 10);  // 30 - 20
}

// =============================================================================
// Multi-Rank Local Tests (each rank operates independently)
// =============================================================================

TEST(AdjacentDifferenceTest, MultiRankLocalOperation) {
    test_context ctx{1, 4};  // Rank 1 of 4
    distributed_vector<int> input(100, 0, ctx);
    distributed_vector<int> output(100, 0, ctx);

    // Fill local partition with increasing values
    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>((i + 1) * 10);  // 10, 20, 30, ...
    }

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, local_in.size());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 10);  // First element copied
    for (size_type i = 1; i < local_out.size(); ++i) {
        EXPECT_EQ(local_out[i], 10);  // Constant difference
    }
}

// =============================================================================
// Double Precision Tests
// =============================================================================

TEST(AdjacentDifferenceTest, DoubleValues) {
    test_context ctx{0, 1};
    distributed_vector<double> input(4, 0.0, ctx);
    distributed_vector<double> output(4, 0.0, ctx);

    auto local_in = input.local_view();
    local_in[0] = 1.0;
    local_in[1] = 1.5;
    local_in[2] = 3.0;
    local_in[3] = 6.5;

    auto res = dtl::adjacent_difference(seq{}, input, output);
    ASSERT_TRUE(res.has_value());

    auto local_out = output.local_view();
    EXPECT_DOUBLE_EQ(local_out[0], 1.0);
    EXPECT_DOUBLE_EQ(local_out[1], 0.5);   // 1.5 - 1.0
    EXPECT_DOUBLE_EQ(local_out[2], 1.5);   // 3.0 - 1.5
    EXPECT_DOUBLE_EQ(local_out[3], 3.5);   // 6.5 - 3.0
}

// =============================================================================
// Result Type Tests
// =============================================================================

TEST(AdjacentDifferenceResultTest, DefaultConstruction) {
    adjacent_difference_result result;
    EXPECT_EQ(result.count, 0u);
    EXPECT_TRUE(result.success);
}

TEST(AdjacentDifferenceResultTest, CustomValues) {
    adjacent_difference_result result{42, true};
    EXPECT_EQ(result.count, 42u);
    EXPECT_TRUE(result.success);
}

}  // namespace dtl::test
