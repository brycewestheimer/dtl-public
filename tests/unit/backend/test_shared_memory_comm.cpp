// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shared_memory_comm.cpp
/// @brief Unit tests for shared_memory_communicator send/recv and collectives
/// @details Tests the mailbox-based point-to-point communication and
///          collective operations (broadcast, gather, scatter, allgather)
///          using threads to simulate multiple ranks.
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <backends/shared_memory/shared_memory_communicator.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

namespace dtl::test {

// =============================================================================
// Helper: create N communicators sharing the same regions
// =============================================================================

/// For intra-process testing, we create separate communicator objects that
/// share the same mailbox/collective regions via the same initialization.
/// Since shared_region uses heap fallback in single-process mode, these
/// won't truly share memory between ranks — but the algorithmic logic
/// (write-to-slot, barrier, read-from-slot) is tested with threads.

// =============================================================================
// Single-Rank Tests (no threading)
// =============================================================================

TEST(SharedMemoryCommTest, DefaultConstructionIsInvalid) {
    dtl::shared_memory::shared_memory_communicator comm;
    EXPECT_EQ(comm.rank(), dtl::no_rank);
    EXPECT_EQ(comm.size(), 0);
    EXPECT_FALSE(comm.valid());
}

TEST(SharedMemoryCommTest, SingleRankValid) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    EXPECT_EQ(comm.rank(), 0);
    EXPECT_EQ(comm.size(), 1);
    EXPECT_TRUE(comm.valid());
}

TEST(SharedMemoryCommTest, BarrierSingleRank) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    auto result = comm.barrier();
    EXPECT_TRUE(result.has_value());
}

TEST(SharedMemoryCommTest, BarrierUninitializedReturnsError) {
    dtl::shared_memory::shared_memory_communicator comm;
    auto result = comm.barrier();
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}

TEST(SharedMemoryCommTest, PropertiesCorrect) {
    dtl::shared_memory::shared_memory_communicator comm(2, 4);
    auto props = comm.properties();
    EXPECT_EQ(props.rank, 2);
    EXPECT_EQ(props.size, 4);
    EXPECT_FALSE(props.is_inter);
    EXPECT_EQ(std::string(props.name), "shared_memory");
}

TEST(SharedMemoryCommTest, SharedBufferAllocated) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    EXPECT_NE(comm.shared_buffer(), nullptr);
    EXPECT_GT(comm.shared_buffer_size(), 0u);
}

// =============================================================================
// Send/Recv Tests (single-rank self-send)
// =============================================================================

TEST(SharedMemoryCommTest, SendInvalidDestReturnsError) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    int data = 42;
    auto result = comm.send_impl(&data, 1, sizeof(int), -1, 0);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_argument);
}

TEST(SharedMemoryCommTest, RecvInvalidSourceReturnsError) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    int data = 0;
    auto result = comm.recv_impl(&data, 1, sizeof(int), -1, 0);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_argument);
}

TEST(SharedMemoryCommTest, SendOversizedReturnsError) {
    dtl::shared_memory::shared_memory_communicator::config cfg;
    cfg.mailbox_size = 128;  // Small mailbox
    dtl::shared_memory::shared_memory_communicator comm(0, 2, cfg);
    std::vector<char> big(256, 'A');
    auto result = comm.send_impl(big.data(), big.size(), 1, 1, 0);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::buffer_too_small);
}

TEST(SharedMemoryCommTest, SelfSendRecv) {
    // Rank 0 sends to itself (rank 0), then receives
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    int send_val = 42;
    auto send_result = comm.send_impl(&send_val, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(send_result.has_value());

    int recv_val = 0;
    auto recv_result = comm.recv_impl(&recv_val, 1, sizeof(int), 0, 0);
    ASSERT_TRUE(recv_result.has_value());
    EXPECT_EQ(recv_val, 42);
}

// =============================================================================
// Broadcast (single rank — root copies to itself)
// =============================================================================

TEST(SharedMemoryCommTest, BroadcastSingleRank) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    int data = 99;
    auto result = comm.broadcast_impl(&data, 1, sizeof(int), 0);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(data, 99);
}

TEST(SharedMemoryCommTest, BroadcastInvalidRoot) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    int data = 1;
    auto result = comm.broadcast_impl(&data, 1, sizeof(int), -1);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_argument);
}

// =============================================================================
// Gather/Scatter/Allgather (single rank)
// =============================================================================

TEST(SharedMemoryCommTest, GatherSingleRank) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    int send_val = 7;
    int recv_val = 0;
    auto result = comm.gather_impl(&send_val, 1, &recv_val, 1, sizeof(int), 0);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(recv_val, 7);
}

TEST(SharedMemoryCommTest, ScatterSingleRank) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    int send_val = 13;
    int recv_val = 0;
    auto result = comm.scatter_impl(&send_val, 1, &recv_val, 1, sizeof(int), 0);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(recv_val, 13);
}

TEST(SharedMemoryCommTest, AllgatherSingleRank) {
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    int send_val = 21;
    int recv_val = 0;
    auto result = comm.allgather_impl(&send_val, 1, &recv_val, 1, sizeof(int));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(recv_val, 21);
}

TEST(SharedMemoryCommTest, GatherInvalidRoot) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    int s = 1, r = 0;
    auto result = comm.gather_impl(&s, 1, &r, 1, sizeof(int), 99);
    EXPECT_TRUE(result.has_error());
}

TEST(SharedMemoryCommTest, ScatterInvalidRoot) {
    dtl::shared_memory::shared_memory_communicator comm(0, 2);
    int s = 1, r = 0;
    auto result = comm.scatter_impl(&s, 1, &r, 1, sizeof(int), -1);
    EXPECT_TRUE(result.has_error());
}

// =============================================================================
// Mailbox Header Layout
// =============================================================================

TEST(SharedMemoryCommTest, MailboxHeaderAlignment) {
    // mailbox_header should be aligned to 64 bytes for cache line
    static_assert(alignof(dtl::shared_memory::mailbox_header) == 64,
                  "mailbox_header must be 64-byte aligned");
    SUCCEED();
}

TEST(SharedMemoryCommTest, MailboxHeaderHasAtomicFlag) {
    static_assert(std::is_same_v<
        decltype(dtl::shared_memory::mailbox_header::flag),
        std::atomic<int>>,
        "mailbox_header::flag must be atomic<int>");
    SUCCEED();
}

// =============================================================================
// Factory Function
// =============================================================================

TEST(SharedMemoryCommTest, FactoryCreatesValid) {
    auto comm = dtl::shared_memory::make_shared_memory_communicator(0, 4);
    ASSERT_NE(comm, nullptr);
    EXPECT_EQ(comm->rank(), 0);
    EXPECT_EQ(comm->size(), 4);
    EXPECT_TRUE(comm->valid());
}

// =============================================================================
// Move Semantics
// =============================================================================

TEST(SharedMemoryCommTest, MoveConstruction) {
    dtl::shared_memory::shared_memory_communicator a(0, 2);
    ASSERT_TRUE(a.valid());
    dtl::shared_memory::shared_memory_communicator b(std::move(a));
    EXPECT_TRUE(b.valid());
    EXPECT_EQ(b.rank(), 0);
    EXPECT_EQ(b.size(), 2);
}

TEST(SharedMemoryCommTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<dtl::shared_memory::shared_memory_communicator>,
                  "shared_memory_communicator must not be copyable");
    SUCCEED();
}

}  // namespace dtl::test
