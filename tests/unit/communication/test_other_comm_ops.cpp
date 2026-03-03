// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_other_comm_ops.cpp
/// @brief Unit tests for waitany and sendrecv_replace communication operations

#include <gtest/gtest.h>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/point_to_point.hpp>
#include <dtl/core/types.hpp>
#include <array>
#include <vector>

using namespace dtl;

// ============================================================================
// Null Communicator Tests
// ============================================================================

TEST(OtherCommOps, NullCommunicator_Waitany_ReturnsZero) {
    null_communicator comm;

    // Create some dummy request handles
    std::array<request_handle, 3> requests = {
        request_handle{},
        request_handle{},
        request_handle{}
    };

    // waitany should return 0 (first index) for null communicator
    auto result = comm.waitany(requests.data(), requests.size());
    EXPECT_EQ(result, 0u);
}

TEST(OtherCommOps, NullCommunicator_Waitany_EmptyRequests) {
    null_communicator comm;

    // Empty request array should return 0
    auto result = comm.waitany(nullptr, 0);
    EXPECT_EQ(result, 0u);
}

TEST(OtherCommOps, NullCommunicator_SendrecvReplace_NoOp) {
    null_communicator comm;

    // For null communicator, sendrecv_replace should be no-op
    // (data should remain unchanged)
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> original = data;

    comm.sendrecv_replace(data.data(), data.size() * sizeof(int),
                           0, 100, 0, 200);

    // Data should be unchanged
    EXPECT_EQ(data, original);
}

// ============================================================================
// Free Function Tests
// ============================================================================

TEST(OtherCommOps, WaitAny_EmptyRequests_ReturnsError) {
    null_communicator comm;
    std::span<request_handle> empty_requests;

    auto result = wait_any(comm, empty_requests);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_argument);
}

TEST(OtherCommOps, WaitAny_WithNullCommunicator) {
    null_communicator comm;

    std::array<request_handle, 3> requests = {
        request_handle{},
        request_handle{},
        request_handle{}
    };

    auto result = wait_any(comm, std::span(requests));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
}

TEST(OtherCommOps, SendrecvReplace_WithNullCommunicator) {
    null_communicator comm;

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> original = data;

    auto result = sendrecv_replace(comm, std::span(data), 0, 100, 0, 200);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 200);
    EXPECT_EQ(result->count, 4u);

    // Data should be unchanged for null communicator
    EXPECT_EQ(data, original);
}

TEST(OtherCommOps, SendrecvReplace_EmptyData) {
    null_communicator comm;

    std::vector<int> data;
    auto result = sendrecv_replace(comm, std::span(data), 0, 100, 0, 200);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->count, 0u);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(OtherCommOps, SendrecvReplace_DifferentTypes) {
    null_communicator comm;

    // Test with different element types
    {
        std::vector<int> int_data = {1, 2, 3};
        auto result = sendrecv_replace(comm, std::span(int_data), 0, 100, 0, 200);
        EXPECT_TRUE(result.has_value());
    }

    {
        std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
        auto result = sendrecv_replace(comm, std::span(float_data), 0, 100, 0, 200);
        EXPECT_TRUE(result.has_value());
    }

    {
        std::vector<char> char_data = {'a', 'b', 'c'};
        auto result = sendrecv_replace(comm, std::span(char_data), 0, 100, 0, 200);
        EXPECT_TRUE(result.has_value());
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(OtherCommOps, Waitany_SingleRequest) {
    null_communicator comm;

    std::array<request_handle, 1> requests = {request_handle{}};

    auto result = wait_any(comm, std::span(requests));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
}

TEST(OtherCommOps, SendrecvReplace_SelfCommunication) {
    null_communicator comm;

    // Send to self (same source and dest)
    std::vector<int> data = {10, 20, 30};
    std::vector<int> original = data;

    auto result = sendrecv_replace(comm, std::span(data), 0, 42, 0, 42);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 42);

    // For null communicator, data remains unchanged
    EXPECT_EQ(data, original);
}
