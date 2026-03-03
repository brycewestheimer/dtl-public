// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_dynamic_handler_remote.cpp
/// @brief MPI integration tests for dynamic RPC handler remote invocation
/// @details Tests that dynamic handlers work correctly across MPI ranks.

#include <dtl/remote/action.hpp>
#include <dtl/remote/action_registry.hpp>
#include <dtl/remote/dynamic_handler.hpp>
#include <dtl/remote/rpc_serialization.hpp>
#include <dtl/remote/rpc_request.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/serialization/serializer.hpp>

#include <gtest/gtest.h>

#include <mpi.h>
#include <array>
#include <vector>
#include <thread>
#include <chrono>

namespace dtl::remote::test {

// =============================================================================
// Test Functions (must be visible to all ranks)
// =============================================================================

int add_remote(int a, int b) { return a + b; }
int echo_rank(int rank_hint) { 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank_hint + rank * 1000;  // Return hint + rank*1000 for verification
}
void store_value(int value) {
    // Side effect - store in static for testing
    static int stored = 0;
    stored = value;
}
int get_stored() {
    static int stored = 0;
    return stored;
}

DTL_REGISTER_ACTION(add_remote);
DTL_REGISTER_ACTION(echo_rank);
DTL_REGISTER_ACTION(store_value);
DTL_REGISTER_ACTION(get_stored);

// =============================================================================
// Test Fixture
// =============================================================================

class DynamicHandlerRemoteTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        // Build registry with all actions
        registry_ = registry_builder<16>{}
            .add<action<&add_remote>>()
            .add<action<&echo_rank>>()
            .add<action<&store_value>>()
            .add<action<&get_stored>>()
            .build();
    }
    
    void TearDown() override {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int rank_;
    int size_;
    action_registry<16> registry_;
};

// =============================================================================
// Remote Invocation Tests
// =============================================================================

TEST_F(DynamicHandlerRemoteTest, SendRequestReceiveResponse) {
    // Skip if single rank
    if (size_ < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }
    
    const int TAG_REQUEST = 100;
    const int TAG_RESPONSE = 101;
    
    if (rank_ == 0) {
        // Rank 0 sends request to rank 1
        int target = 1;
        
        // Serialize request: add_remote(17, 25)
        auto request = serialize_request<action<&add_remote>>(
            rank_, 1 /* request_id */, 17, 25);
        
        // Send request
        MPI_Send(request.data(), static_cast<int>(request.size()),
                MPI_BYTE, target, TAG_REQUEST, MPI_COMM_WORLD);
        
        // Receive response
        MPI_Status status;
        MPI_Probe(target, TAG_RESPONSE, MPI_COMM_WORLD, &status);
        
        int response_size;
        MPI_Get_count(&status, MPI_BYTE, &response_size);
        
        std::vector<std::byte> response(response_size);
        MPI_Recv(response.data(), response_size, MPI_BYTE,
                target, TAG_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Parse response
        auto header = message_header::deserialize(response.data());
        EXPECT_EQ(header.msg_type, message_header::response_type);
        EXPECT_EQ(header.request, 1u);
        
        // Deserialize result
        const std::byte* payload = response.data() + message_header::serialized_size();
        int result = deserialize<int>(payload, header.payload_size);
        EXPECT_EQ(result, 42);  // 17 + 25 = 42
        
    } else if (rank_ == 1) {
        // Rank 1 receives request, invokes handler, sends response
        MPI_Status status;
        MPI_Probe(0, TAG_REQUEST, MPI_COMM_WORLD, &status);
        
        int request_size;
        MPI_Get_count(&status, MPI_BYTE, &request_size);
        
        std::vector<std::byte> request(request_size);
        MPI_Recv(request.data(), request_size, MPI_BYTE,
                0, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Parse request
        auto header = message_header::deserialize(request.data());
        EXPECT_EQ(header.msg_type, message_header::request_type);
        
        // Look up handler
        auto handler_opt = registry_.find(header.action);
        ASSERT_TRUE(handler_opt.has_value());
        ASSERT_TRUE(handler_opt->valid());
        
        // Invoke handler
        const std::byte* payload = request.data() + message_header::serialized_size();
        std::vector<std::byte> result_buf(256);
        
        size_type result_size = handler_opt->invoke(
            payload, header.payload_size,
            result_buf.data(), result_buf.size());
        
        EXPECT_GT(result_size, 0u);
        
        // Build and send response
        std::vector<std::byte> response;
        response.resize(message_header::serialized_size() + result_size);
        
        auto resp_header = message_header::make_response(
            header.request, result_size, rank_);
        message_header::serialize(resp_header, response.data());
        std::memcpy(response.data() + message_header::serialized_size(),
                   result_buf.data(), result_size);
        
        MPI_Send(response.data(), static_cast<int>(response.size()),
                MPI_BYTE, 0, TAG_RESPONSE, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(DynamicHandlerRemoteTest, RoundRobinInvocation) {
    // All ranks invoke add_remote on the next rank
    // Skip if single rank
    if (size_ < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }
    
    const int TAG_REQUEST = 200;
    const int TAG_RESPONSE = 201;
    
    int next_rank = (rank_ + 1) % size_;
    int prev_rank = (rank_ + size_ - 1) % size_;
    
    // Send request to next rank
    int a = rank_ * 10;
    int b = rank_ * 10 + 1;
    auto request = serialize_request<action<&add_remote>>(
        rank_, rank_ + 1000 /* request_id */, a, b);
    
    MPI_Request send_req, recv_req;
    MPI_Isend(request.data(), static_cast<int>(request.size()),
             MPI_BYTE, next_rank, TAG_REQUEST, MPI_COMM_WORLD, &send_req);
    
    // Receive request from previous rank
    std::vector<std::byte> incoming_request(256);
    MPI_Irecv(incoming_request.data(), static_cast<int>(incoming_request.size()),
             MPI_BYTE, prev_rank, TAG_REQUEST, MPI_COMM_WORLD, &recv_req);
    
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    
    // Parse and invoke
    auto header = message_header::deserialize(incoming_request.data());
    auto handler_opt = registry_.find(header.action);
    ASSERT_TRUE(handler_opt.has_value());
    
    const std::byte* payload = incoming_request.data() + message_header::serialized_size();
    std::vector<std::byte> result_buf(64);
    
    size_type result_size = handler_opt->invoke(
        payload, header.payload_size,
        result_buf.data(), result_buf.size());
    
    // Build and send response
    std::vector<std::byte> response;
    response.resize(message_header::serialized_size() + result_size);
    
    auto resp_header = message_header::make_response(
        header.request, result_size, rank_);
    message_header::serialize(resp_header, response.data());
    std::memcpy(response.data() + message_header::serialized_size(),
               result_buf.data(), result_size);
    
    MPI_Send(response.data(), static_cast<int>(response.size()),
            MPI_BYTE, prev_rank, TAG_RESPONSE, MPI_COMM_WORLD);
    
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    
    // Receive response from next rank
    MPI_Status status;
    MPI_Probe(next_rank, TAG_RESPONSE, MPI_COMM_WORLD, &status);
    
    int response_size;
    MPI_Get_count(&status, MPI_BYTE, &response_size);
    
    std::vector<std::byte> our_response(response_size);
    MPI_Recv(our_response.data(), response_size, MPI_BYTE,
            next_rank, TAG_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Verify result
    auto our_resp_header = message_header::deserialize(our_response.data());
    const std::byte* our_payload = our_response.data() + message_header::serialized_size();
    int result = deserialize<int>(our_payload, our_resp_header.payload_size);
    
    EXPECT_EQ(result, a + b);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(DynamicHandlerRemoteTest, UnknownActionReturnsError) {
    // Skip if single rank
    if (size_ < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }
    
    const int TAG_REQUEST = 300;
    const int TAG_RESPONSE = 301;
    
    if (rank_ == 0) {
        // Build a request with an unknown action ID
        std::vector<std::byte> request;
        request.resize(message_header::serialized_size());
        
        auto header = message_header::make_request(
            999999, // Unknown action ID
            1,      // request_id
            0,      // payload_size
            rank_);
        message_header::serialize(header, request.data());
        
        MPI_Send(request.data(), static_cast<int>(request.size()),
                MPI_BYTE, 1, TAG_REQUEST, MPI_COMM_WORLD);
        
        // Receive error response
        MPI_Status status;
        MPI_Probe(1, TAG_RESPONSE, MPI_COMM_WORLD, &status);
        
        int response_size;
        MPI_Get_count(&status, MPI_BYTE, &response_size);
        
        std::vector<std::byte> response(response_size);
        MPI_Recv(response.data(), response_size, MPI_BYTE,
                1, TAG_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto resp_header = message_header::deserialize(response.data());
        EXPECT_EQ(resp_header.msg_type, message_header::error_type);
        
    } else if (rank_ == 1) {
        // Receive and process request
        MPI_Status status;
        MPI_Probe(0, TAG_REQUEST, MPI_COMM_WORLD, &status);
        
        int request_size;
        MPI_Get_count(&status, MPI_BYTE, &request_size);
        
        std::vector<std::byte> request(request_size);
        MPI_Recv(request.data(), request_size, MPI_BYTE,
                0, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto header = message_header::deserialize(request.data());
        
        // Look up handler - should not be found
        auto handler_opt = registry_.find(header.action);
        EXPECT_FALSE(handler_opt.has_value());
        
        // Send error response
        auto error_response = serialize_error(
            header.request, rank_,
            static_cast<int32_t>(status_code::not_found),
            "Unknown action");
        
        MPI_Send(error_response.data(), static_cast<int>(error_response.size()),
                MPI_BYTE, 0, TAG_RESPONSE, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(DynamicHandlerRemoteTest, VoidFunctionRemoteInvocation) {
    // Skip if single rank
    if (size_ < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }
    
    const int TAG_REQUEST = 400;
    const int TAG_RESPONSE = 401;
    
    if (rank_ == 0) {
        // Send store_value(123) to rank 1
        auto request = serialize_request<action<&store_value>>(
            rank_, 1 /* request_id */, 123);
        
        MPI_Send(request.data(), static_cast<int>(request.size()),
                MPI_BYTE, 1, TAG_REQUEST, MPI_COMM_WORLD);
        
        // Receive void response
        MPI_Status status;
        MPI_Probe(1, TAG_RESPONSE, MPI_COMM_WORLD, &status);
        
        int response_size;
        MPI_Get_count(&status, MPI_BYTE, &response_size);
        
        std::vector<std::byte> response(response_size);
        MPI_Recv(response.data(), response_size, MPI_BYTE,
                1, TAG_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto resp_header = message_header::deserialize(response.data());
        EXPECT_EQ(resp_header.msg_type, message_header::response_type);
        EXPECT_EQ(resp_header.payload_size, 0u);  // Void return
        
    } else if (rank_ == 1) {
        // Receive and process request
        MPI_Status status;
        MPI_Probe(0, TAG_REQUEST, MPI_COMM_WORLD, &status);
        
        int request_size;
        MPI_Get_count(&status, MPI_BYTE, &request_size);
        
        std::vector<std::byte> request(request_size);
        MPI_Recv(request.data(), request_size, MPI_BYTE,
                0, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto header = message_header::deserialize(request.data());
        auto handler_opt = registry_.find(header.action);
        ASSERT_TRUE(handler_opt.has_value());
        
        const std::byte* payload = request.data() + message_header::serialized_size();
        std::vector<std::byte> result_buf(64);
        
        size_type result_size = handler_opt->invoke(
            payload, header.payload_size,
            result_buf.data(), result_buf.size());
        
        EXPECT_EQ(result_size, 0u);  // Void return
        
        // Send void response
        auto response = serialize_void_response(header.request, rank_);
        MPI_Send(response.data(), static_cast<int>(response.size()),
                MPI_BYTE, 0, TAG_RESPONSE, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(DynamicHandlerRemoteTest, EchoRankVerifiesRemoteExecution) {
    // Skip if single rank
    if (size_ < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }
    
    const int TAG_REQUEST = 500;
    const int TAG_RESPONSE = 501;
    
    if (rank_ == 0) {
        // Send echo_rank(42) to rank 1
        // Should return 42 + 1*1000 = 1042
        auto request = serialize_request<action<&echo_rank>>(
            rank_, 1 /* request_id */, 42);
        
        MPI_Send(request.data(), static_cast<int>(request.size()),
                MPI_BYTE, 1, TAG_REQUEST, MPI_COMM_WORLD);
        
        // Receive response
        MPI_Status status;
        MPI_Probe(1, TAG_RESPONSE, MPI_COMM_WORLD, &status);
        
        int response_size;
        MPI_Get_count(&status, MPI_BYTE, &response_size);
        
        std::vector<std::byte> response(response_size);
        MPI_Recv(response.data(), response_size, MPI_BYTE,
                1, TAG_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto resp_header = message_header::deserialize(response.data());
        const std::byte* payload = response.data() + message_header::serialized_size();
        int result = deserialize<int>(payload, resp_header.payload_size);
        
        // Verify the function ran on rank 1
        EXPECT_EQ(result, 42 + 1 * 1000);
        
    } else if (rank_ == 1) {
        // Receive and process request
        MPI_Status status;
        MPI_Probe(0, TAG_REQUEST, MPI_COMM_WORLD, &status);
        
        int request_size;
        MPI_Get_count(&status, MPI_BYTE, &request_size);
        
        std::vector<std::byte> request(request_size);
        MPI_Recv(request.data(), request_size, MPI_BYTE,
                0, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        auto header = message_header::deserialize(request.data());
        auto handler_opt = registry_.find(header.action);
        ASSERT_TRUE(handler_opt.has_value());
        
        const std::byte* payload = request.data() + message_header::serialized_size();
        std::vector<std::byte> result_buf(64);
        
        size_type result_size = handler_opt->invoke(
            payload, header.payload_size,
            result_buf.data(), result_buf.size());
        
        // Build and send response
        std::vector<std::byte> response;
        response.resize(message_header::serialized_size() + result_size);
        
        auto resp_header = message_header::make_response(
            header.request, result_size, rank_);
        message_header::serialize(resp_header, response.data());
        std::memcpy(response.data() + message_header::serialized_size(),
                   result_buf.data(), result_size);
        
        MPI_Send(response.data(), static_cast<int>(response.size()),
                MPI_BYTE, 0, TAG_RESPONSE, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace dtl::remote::test
