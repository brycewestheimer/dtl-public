// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_send_mode_wiring.cpp
/// @brief Unit tests for send mode wiring (ssend, rsend, issend, irsend)
/// @details Verifies that ssend/rsend operations are properly wired through
///          the three-layer DTL communication architecture:
///          1. mpi_communicator (low-level MPI wrapper)
///          2. mpi_comm_adapter (concept-compliant adapter)
///          3. point_to_point.hpp (free functions)
/// @since Phase 12B (V1.2)

#include <dtl/communication/point_to_point.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/default_communicator.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>

namespace dtl::tests {

// =============================================================================
// Concept Verification (compile-time)
// =============================================================================

// Verify that null_communicator still satisfies Communicator concept
// after adding send mode methods
static_assert(Communicator<null_communicator>,
              "null_communicator must satisfy Communicator concept");

// =============================================================================
// null_communicator Send Mode Operations
// =============================================================================

TEST(NullCommunicatorSendModeTest, SsendIsNoOp) {
    null_communicator comm;
    std::vector<int> data = {1, 2, 3, 4, 5};

    // Should not throw or crash
    comm.ssend(data.data(), data.size() * sizeof(int), 0, 0);
}

TEST(NullCommunicatorSendModeTest, RsendIsNoOp) {
    null_communicator comm;
    std::vector<double> data = {1.1, 2.2, 3.3};

    // Should not throw or crash
    comm.rsend(data.data(), data.size() * sizeof(double), 0, 0);
}

TEST(NullCommunicatorSendModeTest, IssendReturnsValidHandle) {
    null_communicator comm;
    std::vector<int> data = {10, 20, 30};

    auto req = comm.issend(data.data(), data.size() * sizeof(int), 0, 0);

    // In null_communicator, request is immediately complete
    EXPECT_TRUE(comm.test(req));
}

TEST(NullCommunicatorSendModeTest, IrsendReturnsValidHandle) {
    null_communicator comm;
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto req = comm.irsend(data.data(), data.size() * sizeof(float), 0, 0);

    // In null_communicator, request is immediately complete
    EXPECT_TRUE(comm.test(req));
}

TEST(NullCommunicatorSendModeTest, MultipleSendModesDoNotConflict) {
    null_communicator comm;
    std::vector<int> data1 = {1, 2, 3};
    std::vector<int> data2 = {4, 5, 6};
    std::vector<int> data3 = {7, 8, 9};
    std::vector<int> data4 = {10, 11, 12};

    // Should be able to call all send modes without issues
    comm.send(data1.data(), data1.size() * sizeof(int), 0, 0);
    comm.ssend(data2.data(), data2.size() * sizeof(int), 0, 0);
    comm.rsend(data3.data(), data3.size() * sizeof(int), 0, 0);

    auto req1 = comm.isend(data1.data(), data1.size() * sizeof(int), 0, 0);
    auto req2 = comm.issend(data2.data(), data2.size() * sizeof(int), 0, 0);
    auto req3 = comm.irsend(data3.data(), data3.size() * sizeof(int), 0, 0);

    EXPECT_TRUE(comm.test(req1));
    EXPECT_TRUE(comm.test(req2));
    EXPECT_TRUE(comm.test(req3));
}

// =============================================================================
// point_to_point.hpp Free Function Wiring
// =============================================================================

TEST(PointToPointSendModeTest, SsendFreeFunctionCompiles) {
    null_communicator comm;
    std::vector<int> data = {1, 2, 3, 4};

    // Verify that ssend() free function compiles with null_communicator
    auto result = ssend(comm, std::span<const int>(data), 0, 0);
    EXPECT_TRUE(result.has_value());
}

TEST(PointToPointSendModeTest, RsendFreeFunctionCompiles) {
    null_communicator comm;
    std::vector<double> data = {1.5, 2.5, 3.5};

    // Verify that rsend() free function compiles with null_communicator
    auto result = rsend(comm, std::span<const double>(data), 0, 0);
    EXPECT_TRUE(result.has_value());
}

TEST(PointToPointSendModeTest, SsendWithDifferentTypes) {
    null_communicator comm;

    // Test with various types
    std::vector<int32_t> ints = {1, 2, 3};
    std::vector<int64_t> longs = {100L, 200L, 300L};
    std::vector<float> floats = {1.0f, 2.0f, 3.0f};
    std::vector<double> doubles = {10.0, 20.0, 30.0};

    EXPECT_TRUE(ssend(comm, std::span<const int32_t>(ints), 0, 0).has_value());
    EXPECT_TRUE(ssend(comm, std::span<const int64_t>(longs), 0, 0).has_value());
    EXPECT_TRUE(ssend(comm, std::span<const float>(floats), 0, 0).has_value());
    EXPECT_TRUE(ssend(comm, std::span<const double>(doubles), 0, 0).has_value());
}

TEST(PointToPointSendModeTest, RsendWithDifferentTypes) {
    null_communicator comm;

    // Test with various types
    std::vector<int32_t> ints = {1, 2, 3};
    std::vector<int64_t> longs = {100L, 200L, 300L};
    std::vector<float> floats = {1.0f, 2.0f, 3.0f};
    std::vector<double> doubles = {10.0, 20.0, 30.0};

    EXPECT_TRUE(rsend(comm, std::span<const int32_t>(ints), 0, 0).has_value());
    EXPECT_TRUE(rsend(comm, std::span<const int64_t>(longs), 0, 0).has_value());
    EXPECT_TRUE(rsend(comm, std::span<const float>(floats), 0, 0).has_value());
    EXPECT_TRUE(rsend(comm, std::span<const double>(doubles), 0, 0).has_value());
}

TEST(PointToPointSendModeTest, SsendWithEmptyData) {
    null_communicator comm;
    std::vector<int> empty_data;

    // Empty send should succeed
    auto result = ssend(comm, std::span<const int>(empty_data), 0, 0);
    EXPECT_TRUE(result.has_value());
}

TEST(PointToPointSendModeTest, RsendWithEmptyData) {
    null_communicator comm;
    std::vector<double> empty_data;

    // Empty send should succeed
    auto result = rsend(comm, std::span<const double>(empty_data), 0, 0);
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// default_communicator Integration
// =============================================================================

TEST(DefaultCommunicatorSendModeTest, SsendCompilesWithDefaultCommunicator) {
    default_communicator comm{};
    std::vector<int> data = {1, 2, 3};

    // Verify that ssend works with default_communicator
    // In single-process mode, this should still work (no actual send)
    if (comm.size() == 1) {
        // Single process: ssend is a no-op in null_communicator
        auto result = ssend(comm, std::span<const int>(data), 0, 0);
        EXPECT_TRUE(result.has_value());
    }
}

TEST(DefaultCommunicatorSendModeTest, RsendCompilesWithDefaultCommunicator) {
    default_communicator comm{};
    std::vector<double> data = {1.0, 2.0, 3.0};

    // Verify that rsend works with default_communicator
    if (comm.size() == 1) {
        // Single process: rsend is a no-op in null_communicator
        auto result = rsend(comm, std::span<const double>(data), 0, 0);
        EXPECT_TRUE(result.has_value());
    }
}

// =============================================================================
// Behavioral Verification
// =============================================================================

TEST(SendModeWiringTest, SendModesDoNotThrowWithNullCommunicator) {
    null_communicator comm;
    std::vector<int> data = {1, 2, 3, 4, 5};

    // None of these should throw
    EXPECT_NO_THROW({
        comm.send(data.data(), data.size() * sizeof(int), 0, 0);
        comm.ssend(data.data(), data.size() * sizeof(int), 0, 0);
        comm.rsend(data.data(), data.size() * sizeof(int), 0, 0);

        auto req1 = comm.isend(data.data(), data.size() * sizeof(int), 0, 0);
        auto req2 = comm.issend(data.data(), data.size() * sizeof(int), 0, 0);
        auto req3 = comm.irsend(data.data(), data.size() * sizeof(int), 0, 0);

        comm.wait(req1);
        comm.wait(req2);
        comm.wait(req3);
    });
}

TEST(SendModeWiringTest, FreeFunctionsDoNotThrowWithNullCommunicator) {
    null_communicator comm;
    std::vector<int> data = {1, 2, 3};

    // None of these should throw
    EXPECT_NO_THROW({
        auto r1 = send(comm, std::span<const int>(data), 0, 0);
        auto r2 = ssend(comm, std::span<const int>(data), 0, 0);
        auto r3 = rsend(comm, std::span<const int>(data), 0, 0);
    });
}

} // namespace dtl::tests
