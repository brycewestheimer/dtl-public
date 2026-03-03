// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_reduce_min_max_multielement.cpp
/// @brief Unit tests for multi-element reduce min/max (Phase 01 / CR-P01-T01)
/// @details Verifies that reduce() with min/max writes all elements correctly.

#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace dtl::test {

// =============================================================================
// Multi-Element Reduce Min Tests
// =============================================================================

TEST(ReduceMinMaxTest, MultiElementMin_AllElementsWritten) {
    null_communicator comm;

    std::vector<int> send = {5, 3, 8, 1, 9};
    std::vector<int> recv(5, -1);  // Initialize to sentinel

    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_min<int>{}, 0);

    ASSERT_TRUE(result.has_value());

    // With null_communicator (single rank), each element reduction is identity
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i])
            << "Element " << i << " not correctly written by reduce min";
    }
}

TEST(ReduceMinMaxTest, MultiElementMax_AllElementsWritten) {
    null_communicator comm;

    std::vector<int> send = {5, 3, 8, 1, 9};
    std::vector<int> recv(5, -1);  // Initialize to sentinel

    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_max<int>{}, 0);

    ASSERT_TRUE(result.has_value());

    // With null_communicator (single rank), each element reduction is identity
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i])
            << "Element " << i << " not correctly written by reduce max";
    }
}

TEST(ReduceMinMaxTest, MultiElementMin_DoubleType) {
    null_communicator comm;

    std::vector<double> send = {1.5, 2.5, 0.5, 3.5};
    std::vector<double> recv(4, -1.0);

    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_min<double>{}, 0);

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST(ReduceMinMaxTest, MultiElementMax_DoubleType) {
    null_communicator comm;

    std::vector<double> send = {1.5, 2.5, 0.5, 3.5};
    std::vector<double> recv(4, -1.0);

    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_max<double>{}, 0);

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST(ReduceMinMaxTest, SingleElementMin_StillWorks) {
    null_communicator comm;

    std::vector<int> send = {42};
    std::vector<int> recv(1, 0);

    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_min<int>{}, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 42);
}

TEST(ReduceMinMaxTest, SingleElementMax_StillWorks) {
    null_communicator comm;

    std::vector<int> send = {42};
    std::vector<int> recv(1, 0);

    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_max<int>{}, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 42);
}

// =============================================================================
// Reduce Sum Type Dispatch Tests (CR-P01-T09)
// =============================================================================

TEST(ReduceSumTypeTest, ReduceSum_IntType) {
    null_communicator comm;

    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);

    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_sum<int>{}, 0);

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST(ReduceSumTypeTest, ReduceSum_FloatType) {
    null_communicator comm;

    std::vector<float> send = {1.0f, 2.0f, 3.0f};
    std::vector<float> recv(3, 0.0f);

    auto result = reduce(comm, std::span<const float>(send),
                         std::span<float>(recv), reduce_sum<float>{}, 0);

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_FLOAT_EQ(recv[i], send[i]);
    }
}

TEST(ReduceSumTypeTest, ReduceSum_LongType) {
    null_communicator comm;

    std::vector<long> send = {100L, 200L, 300L};
    std::vector<long> recv(3, 0L);

    auto result = reduce(comm, std::span<const long>(send),
                         std::span<long>(recv), reduce_sum<long>{}, 0);

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST(ReduceSumTypeTest, AllreduceSum_IntType) {
    null_communicator comm;

    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);

    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_sum<int>{});

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST(ReduceSumTypeTest, AllreduceSum_LongType) {
    null_communicator comm;

    std::vector<long> send = {100L, 200L, 300L};
    std::vector<long> recv(3, 0L);

    auto result = allreduce(comm, std::span<const long>(send),
                            std::span<long>(recv), reduce_sum<long>{});

    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

}  // namespace dtl::test
