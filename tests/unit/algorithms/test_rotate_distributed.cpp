// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rotate_distributed.cpp
/// @brief Unit tests for distributed rotate algorithm (Phase 22, Task 22.1)
/// @details Tests the cross-rank distributed rotate that uses alltoallv
///          communication. Single-rank tests exercise the algorithm logic
///          without requiring real MPI.

#include <dtl/algorithms/modifying/rotate.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include "mock_single_rank_comm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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
// Distributed Rotate Tests (single-rank via mock communicator)
// =============================================================================

TEST(DistributedRotateTest, RotateLeftByTwo) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 2, comm);
    ASSERT_TRUE(res.has_value());

    // After rotating left by 2: [3, 4, 5, 1, 2]
    EXPECT_EQ(local[0], 3);
    EXPECT_EQ(local[1], 4);
    EXPECT_EQ(local[2], 5);
    EXPECT_EQ(local[3], 1);
    EXPECT_EQ(local[4], 2);
}

TEST(DistributedRotateTest, RotateLeftByOne) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 30; local[3] = 40;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 1, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 20);
    EXPECT_EQ(local[1], 30);
    EXPECT_EQ(local[2], 40);
    EXPECT_EQ(local[3], 10);
}

TEST(DistributedRotateTest, RotateByZero) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 0, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 1);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 3);
    EXPECT_EQ(local[3], 4);
}

TEST(DistributedRotateTest, RotateBySize) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 4, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 1);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 3);
    EXPECT_EQ(local[3], 4);
}

TEST(DistributedRotateTest, NegativeRotation) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, -1, comm);
    ASSERT_TRUE(res.has_value());

    // Rotate right by 1: [5, 1, 2, 3, 4]
    EXPECT_EQ(local[0], 5);
    EXPECT_EQ(local[1], 1);
    EXPECT_EQ(local[2], 2);
    EXPECT_EQ(local[3], 3);
    EXPECT_EQ(local[4], 4);
}

TEST(DistributedRotateTest, LargeRotation) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4;

    mock_single_rank_comm comm;
    // Rotate by 6 = rotate by 2 (6 mod 4 = 2)
    auto res = dtl::rotate(seq{}, vec, 6, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 3);
    EXPECT_EQ(local[1], 4);
    EXPECT_EQ(local[2], 1);
    EXPECT_EQ(local[3], 2);
}

TEST(DistributedRotateTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 5, comm);
    ASSERT_TRUE(res.has_value());
}

TEST(DistributedRotateTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(1, 42, ctx);

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 1, comm);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 42);
}

TEST(DistributedRotateTest, RotateHalfSize) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(6, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < 6; ++i) {
        local[i] = static_cast<int>(i + 1);
    }

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 3, comm);
    ASSERT_TRUE(res.has_value());

    // [4, 5, 6, 1, 2, 3]
    EXPECT_EQ(local[0], 4);
    EXPECT_EQ(local[1], 5);
    EXPECT_EQ(local[2], 6);
    EXPECT_EQ(local[3], 1);
    EXPECT_EQ(local[4], 2);
    EXPECT_EQ(local[5], 3);
}

TEST(DistributedRotateTest, LargeNegativeRotation) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = 2; local[2] = 3; local[3] = 4; local[4] = 5;

    mock_single_rank_comm comm;
    // -7 mod 5 = -2, so ((-2 % 5) + 5) % 5 = 3 => rotate left by 3
    auto res = dtl::rotate(seq{}, vec, -7, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(local[0], 4);
    EXPECT_EQ(local[1], 5);
    EXPECT_EQ(local[2], 1);
    EXPECT_EQ(local[3], 2);
    EXPECT_EQ(local[4], 3);
}

TEST(DistributedRotateTest, DoubleValues) {
    test_context ctx{0, 1};
    distributed_vector<double> vec(4, 0.0, ctx);

    auto local = vec.local_view();
    local[0] = 1.1; local[1] = 2.2; local[2] = 3.3; local[3] = 4.4;

    mock_single_rank_comm comm;
    auto res = dtl::rotate(seq{}, vec, 1, comm);
    ASSERT_TRUE(res.has_value());

    EXPECT_DOUBLE_EQ(local[0], 2.2);
    EXPECT_DOUBLE_EQ(local[1], 3.3);
    EXPECT_DOUBLE_EQ(local[2], 4.4);
    EXPECT_DOUBLE_EQ(local[3], 1.1);
}

TEST(DistributedRotateTest, RotateIsInverseOfReverseRotation) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(8, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < 8; ++i) {
        local[i] = static_cast<int>(i * 10);
    }

    mock_single_rank_comm comm;

    auto res1 = dtl::rotate(seq{}, vec, 3, comm);
    ASSERT_TRUE(res1.has_value());

    auto res2 = dtl::rotate(seq{}, vec, -3, comm);
    ASSERT_TRUE(res2.has_value());

    for (size_type i = 0; i < 8; ++i) {
        EXPECT_EQ(local[i], static_cast<int>(i * 10));
    }
}

}  // namespace dtl::test
