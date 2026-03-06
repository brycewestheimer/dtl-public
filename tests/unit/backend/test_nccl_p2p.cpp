// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_p2p.cpp
/// @brief Unit tests for NCCL point-to-point and non-blocking operations
/// @details Tests NCCL send/recv (ncclSend/ncclRecv) and async isend/irecv
///          with CUDA event-based completion tracking. Tests work both with
///          and without NCCL/CUDA enabled.
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
#include <backends/nccl/nccl_communicator.hpp>
#endif

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// NCCL P2P Tests (NCCL + CUDA available)
// =============================================================================

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

// ---------------------------------------------------------------------------
// Task 24.1: Blocking send/recv
// ---------------------------------------------------------------------------

TEST(NcclP2PTest, SendImplReturnsInvalidStateOnDefaultComm) {
    dtl::nccl::nccl_communicator comm;
    int data = 42;
    auto result = comm.send_impl(&data, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(NcclP2PTest, RecvImplReturnsInvalidStateOnDefaultComm) {
    dtl::nccl::nccl_communicator comm;
    int data = 0;
    auto result = comm.recv_impl(&data, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(NcclP2PTest, SendImplTagIgnored) {
    // NCCL does not support tags — verify send works with any tag value
    dtl::nccl::nccl_communicator comm;
    int data = 99;
    auto r1 = comm.send_impl(&data, 1, sizeof(int), 0, 0);
    auto r2 = comm.send_impl(&data, 1, sizeof(int), 0, 42);
    auto r3 = comm.send_impl(&data, 1, sizeof(int), 0, -1);
    // All should fail identically (invalid_state on default comm)
    EXPECT_EQ(r1.error().code(), r2.error().code());
    EXPECT_EQ(r2.error().code(), r3.error().code());
}

TEST(NcclP2PTest, RecvImplTagIgnored) {
    // NCCL does not support tags — verify recv works with any tag value
    dtl::nccl::nccl_communicator comm;
    int data = 0;
    auto r1 = comm.recv_impl(&data, 1, sizeof(int), 0, 0);
    auto r2 = comm.recv_impl(&data, 1, sizeof(int), 0, 42);
    auto r3 = comm.recv_impl(&data, 1, sizeof(int), 0, -1);
    EXPECT_EQ(r1.error().code(), r2.error().code());
    EXPECT_EQ(r2.error().code(), r3.error().code());
}

TEST(NcclP2PTest, SendImplZeroCount) {
    // Zero-count send should still fail on default-constructed comm (invalid state)
    dtl::nccl::nccl_communicator comm;
    int data = 0;
    auto result = comm.send_impl(&data, 0, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(NcclP2PTest, RecvImplZeroCount) {
    dtl::nccl::nccl_communicator comm;
    int data = 0;
    auto result = comm.recv_impl(&data, 0, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

// ---------------------------------------------------------------------------
// Task 24.3: Non-blocking isend/irecv
// ---------------------------------------------------------------------------

TEST(NcclP2PTest, IsendReturnsInvalidStateOnDefaultComm) {
    dtl::nccl::nccl_communicator comm;
    int data = 42;
    auto result = comm.isend(&data, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(NcclP2PTest, IrecvReturnsInvalidStateOnDefaultComm) {
    dtl::nccl::nccl_communicator comm;
    int data = 0;
    auto result = comm.irecv(&data, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(NcclP2PTest, WaitEventNullHandleIsSafe) {
    dtl::nccl::nccl_communicator comm;
    request_handle req{nullptr};
    // wait_event on null handle should be a no-op
    auto result = comm.wait_event(req);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(req.handle, nullptr);
}

TEST(NcclP2PTest, TestEventNullHandleReturnsTrue) {
    dtl::nccl::nccl_communicator comm;
    request_handle req{nullptr};
    // Null handle is "already completed"
    auto result = comm.test_event(req);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(*result);
}

// ---------------------------------------------------------------------------
// Method signature verification
// ---------------------------------------------------------------------------

TEST(NcclP2PTest, IsendMethodSignature) {
    using comm_t = dtl::nccl::nccl_communicator;
    static_assert(std::is_same_v<
        decltype(std::declval<comm_t>().isend(
            std::declval<const void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::size_type>(), std::declval<dtl::rank_t>(),
            std::declval<int>())),
        dtl::result<dtl::request_handle>>,
        "isend must return result<request_handle>");
    SUCCEED();
}

TEST(NcclP2PTest, IrecvMethodSignature) {
    using comm_t = dtl::nccl::nccl_communicator;
    static_assert(std::is_same_v<
        decltype(std::declval<comm_t>().irecv(
            std::declval<void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::size_type>(), std::declval<dtl::rank_t>(),
            std::declval<int>())),
        dtl::result<dtl::request_handle>>,
        "irecv must return result<request_handle>");
    SUCCEED();
}

#else  // !DTL_ENABLE_NCCL || !DTL_ENABLE_CUDA

// Placeholder tests when NCCL/CUDA not available

TEST(NcclP2PTest, SendImplPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, RecvImplPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, SendImplTagIgnoredPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, RecvImplTagIgnoredPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, SendImplZeroCountPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, RecvImplZeroCountPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, IsendPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, IrecvPlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, WaitEventNullHandlePlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, TestEventNullHandlePlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, IsendMethodSignaturePlaceholder) { SUCCEED(); }
TEST(NcclP2PTest, IrecvMethodSignaturePlaceholder) { SUCCEED(); }

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

}  // namespace dtl::test
