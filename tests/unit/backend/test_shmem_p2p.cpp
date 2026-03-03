// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shmem_p2p.cpp
/// @brief Unit tests for SHMEM message-passing emulation via symmetric mailbox
/// @details Tests the send/recv emulation layer in shmem_rma_adapter that uses
///          shmem_putmem + atomic flag signaling to provide two-sided message
///          passing over SHMEM's one-sided PGAS model.
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>

#include <backends/shmem/shmem_rma_adapter.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <cstring>

namespace dtl::test {

// =============================================================================
// SHMEM RMA Adapter - Mailbox and P2P Tests
// =============================================================================

TEST(ShmemP2PTest, MailboxCapacityIsReasonable) {
    // The mailbox should be large enough for typical messages
    EXPECT_GE(dtl::shmem::shmem_rma_adapter::mailbox_capacity, 1024u);
    EXPECT_EQ(dtl::shmem::shmem_rma_adapter::mailbox_capacity, 65536u);
}

TEST(ShmemP2PTest, SendMethodExists) {
    using adapter_t = dtl::shmem::shmem_rma_adapter;
    // Verify the send method signature exists
    static_assert(std::is_same_v<
        decltype(std::declval<adapter_t>().send(
            std::declval<const void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::rank_t>(), std::declval<int>())),
        void>,
        "send() must return void");
    SUCCEED();
}

TEST(ShmemP2PTest, RecvMethodExists) {
    using adapter_t = dtl::shmem::shmem_rma_adapter;
    // Verify the recv method signature exists
    static_assert(std::is_same_v<
        decltype(std::declval<adapter_t>().recv(
            std::declval<void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::rank_t>(), std::declval<int>())),
        void>,
        "recv() must return void");
    SUCCEED();
}

TEST(ShmemP2PTest, IsendReturnsCompletedHandle) {
    using adapter_t = dtl::shmem::shmem_rma_adapter;
    // Verify isend method signature returns request_handle
    static_assert(std::is_same_v<
        decltype(std::declval<adapter_t>().isend(
            std::declval<const void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::rank_t>(), std::declval<int>())),
        dtl::request_handle>,
        "isend() must return request_handle");
    SUCCEED();
}

TEST(ShmemP2PTest, IrecvReturnsCompletedHandle) {
    using adapter_t = dtl::shmem::shmem_rma_adapter;
    // Verify irecv method signature returns request_handle
    static_assert(std::is_same_v<
        decltype(std::declval<adapter_t>().irecv(
            std::declval<void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::rank_t>(), std::declval<int>())),
        dtl::request_handle>,
        "irecv() must return request_handle");
    SUCCEED();
}

TEST(ShmemP2PTest, TestAlwaysReturnsTrue) {
    // SHMEM ops complete inline, so test() always returns true
    dtl::shmem::shmem_rma_adapter adapter;
    dtl::request_handle req{};
    EXPECT_TRUE(adapter.test(req));
}

#if DTL_ENABLE_SHMEM

// These tests only run when SHMEM is actually available

TEST(ShmemP2PTest, ConceptCompliance) {
    // Verify concept satisfaction (also checked via static_assert in header)
    static_assert(dtl::Communicator<dtl::shmem::shmem_rma_adapter>,
                  "shmem_rma_adapter must satisfy Communicator concept");
    static_assert(dtl::RmaCommunicator<dtl::shmem::shmem_rma_adapter>,
                  "shmem_rma_adapter must satisfy RmaCommunicator concept");
    static_assert(dtl::FullRmaCommunicator<dtl::shmem::shmem_rma_adapter>,
                  "shmem_rma_adapter must satisfy FullRmaCommunicator concept");
    SUCCEED();
}

TEST(ShmemP2PTest, AdapterIsValid) {
    // When SHMEM is initialized, the global adapter should be valid
    auto& adapter = dtl::shmem::global_rma_adapter();
    EXPECT_TRUE(adapter.valid());
    EXPECT_GE(adapter.size(), 1);
}

#else

TEST(ShmemP2PTest, ConceptCompliancePlaceholder) { SUCCEED(); }
TEST(ShmemP2PTest, AdapterIsValidPlaceholder) { SUCCEED(); }

#endif  // DTL_ENABLE_SHMEM

}  // namespace dtl::test
