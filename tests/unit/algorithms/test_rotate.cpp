// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rotate.cpp
/// @brief Unit tests for distributed rotate/shift algorithm (R6.4)

#include <dtl/algorithms/modifying/rotate.hpp>
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
// Local Rotate Tests
// =============================================================================

TEST(LocalRotateTest, BasicLeftRotation) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    auto res = dtl::local_rotate(seq{}, vec, 2);
    ASSERT_TRUE(res.has_value());

    // After rotating left by 2: [3, 4, 5, 1, 2]
    EXPECT_EQ(local[0], 3);
    EXPECT_EQ(local[1], 4);
    EXPECT_EQ(local[2], 5);
    EXPECT_EQ(local[3], 1);
    EXPECT_EQ(local[4], 2);
}

TEST(LocalRotateTest, RotateByOne) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 30; local[3] = 40;

    auto res = dtl::local_rotate(seq{}, vec, 1);
    ASSERT_TRUE(res.has_value());

    // After rotating left by 1: [20, 30, 40, 10]
    EXPECT_EQ(local[0], 20);
    EXPECT_EQ(local[1], 30);
    EXPECT_EQ(local[2], 40);
    EXPECT_EQ(local[3], 10);
}

TEST(LocalRotateTest, RotateBySize) {
    // Rotating by size should be a no-op
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    auto res = dtl::local_rotate(seq{}, vec, 4);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 1);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 3);
    EXPECT_EQ(local[3], 4);
}

TEST(LocalRotateTest, RotateByZero) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    auto res = dtl::local_rotate(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 1);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 3);
    EXPECT_EQ(local[3], 4);
}

TEST(LocalRotateTest, NegativeRotation) {
    // Negative rotation = rotate right
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    auto res = dtl::local_rotate(seq{}, vec, -1);
    ASSERT_TRUE(res.has_value());

    // Rotate right by 1: [5, 1, 2, 3, 4]
    // normalized: -1 mod 5 = 4 (rotate left by 4 = rotate right by 1)
    EXPECT_EQ(local[0], 5);
    EXPECT_EQ(local[1], 1);
    EXPECT_EQ(local[2], 2);
    EXPECT_EQ(local[3], 3);
    EXPECT_EQ(local[4], 4);
}

TEST(LocalRotateTest, LargeRotation) {
    // Rotation larger than size should wrap around
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    // Rotate by 6 = rotate by 2 (6 mod 4 = 2)
    auto res = dtl::local_rotate(seq{}, vec, 6);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 3);
    EXPECT_EQ(local[1], 4);
    EXPECT_EQ(local[2], 1);
    EXPECT_EQ(local[3], 2);
}

TEST(LocalRotateTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    auto res = dtl::local_rotate(seq{}, vec, 5);
    ASSERT_TRUE(res.has_value());
}

TEST(LocalRotateTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(1, 42, ctx);

    auto res = dtl::local_rotate(seq{}, vec, 1);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 42);
}

TEST(LocalRotateTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 30;

    auto res = dtl::local_rotate(vec, 1);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 20);
    EXPECT_EQ(local[1], 30);
    EXPECT_EQ(local[2], 10);
}

// =============================================================================
// Local Shift Left Tests
// =============================================================================

TEST(LocalShiftLeftTest, BasicShift) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    size_type remaining = dtl::local_shift_left(seq{}, vec, 2, 0);
    EXPECT_EQ(remaining, 3u);

    // After shift left by 2 with fill 0: [3, 4, 5, 0, 0]
    EXPECT_EQ(local[0], 3);
    EXPECT_EQ(local[1], 4);
    EXPECT_EQ(local[2], 5);
    EXPECT_EQ(local[3], 0);
    EXPECT_EQ(local[4], 0);
}

TEST(LocalShiftLeftTest, ShiftByMoreThanSize) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3;

    size_type remaining = dtl::local_shift_left(seq{}, vec, 5, -1);
    EXPECT_EQ(remaining, 0u);

    // All filled with -1
    EXPECT_EQ(local[0], -1);
    EXPECT_EQ(local[1], -1);
    EXPECT_EQ(local[2], -1);
}

TEST(LocalShiftLeftTest, ShiftByZero) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3;

    size_type remaining = dtl::local_shift_left(seq{}, vec, 0, 0);
    EXPECT_EQ(remaining, 3u);

    EXPECT_EQ(local[0], 1);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 3);
}

TEST(LocalShiftLeftTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 30; local[3] = 40;

    size_type remaining = dtl::local_shift_left(vec, 1, 99);
    EXPECT_EQ(remaining, 3u);

    EXPECT_EQ(local[0], 20);
    EXPECT_EQ(local[1], 30);
    EXPECT_EQ(local[2], 40);
    EXPECT_EQ(local[3], 99);
}

// =============================================================================
// Local Shift Right Tests
// =============================================================================

TEST(LocalShiftRightTest, BasicShift) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    size_type remaining = dtl::local_shift_right(seq{}, vec, 2, 0);
    EXPECT_EQ(remaining, 3u);

    // After shift right by 2 with fill 0: [0, 0, 1, 2, 3]
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 0);
    EXPECT_EQ(local[2], 1);
    EXPECT_EQ(local[3], 2);
    EXPECT_EQ(local[4], 3);
}

TEST(LocalShiftRightTest, ShiftByMoreThanSize) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3;

    size_type remaining = dtl::local_shift_right(seq{}, vec, 10, -1);
    EXPECT_EQ(remaining, 0u);

    EXPECT_EQ(local[0], -1);
    EXPECT_EQ(local[1], -1);
    EXPECT_EQ(local[2], -1);
}

TEST(LocalShiftRightTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 30; local[3] = 40;

    size_type remaining = dtl::local_shift_right(vec, 1, 99);
    EXPECT_EQ(remaining, 3u);

    EXPECT_EQ(local[0], 99);
    EXPECT_EQ(local[1], 10);
    EXPECT_EQ(local[2], 20);
    EXPECT_EQ(local[3], 30);
}

// =============================================================================
// Multi-Rank Tests
// =============================================================================

TEST(LocalRotateTest, MultiRankLocal) {
    test_context ctx{1, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i);
    }

    auto res = dtl::local_rotate(seq{}, vec, 5);
    ASSERT_TRUE(res.has_value());

    // First 5 elements rotated to end
    EXPECT_EQ(local[0], 5);
    EXPECT_EQ(local[local.size() - 1], 4);
}

}  // namespace dtl::test
