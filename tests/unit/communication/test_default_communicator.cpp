// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_default_communicator.cpp
/// @brief Unit tests for default_communicator type alias and world_comm() factory
/// @details Verifies compile-time dispatch, concept satisfaction, and
///          null_communicator convenience method behavior.

#include <dtl/communication/default_communicator.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <backends/mpi/mpi_lifecycle.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cstring>

namespace dtl::test {

// =============================================================================
// Concept Verification (compile-time)
// =============================================================================

static_assert(Communicator<null_communicator>,
              "null_communicator must satisfy Communicator concept");
static_assert(Communicator<default_communicator>,
              "default_communicator must satisfy Communicator concept");

// =============================================================================
// world_comm() Factory
// =============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

TEST(DefaultCommunicatorTest, WorldCommReturnsValidCommunicator) {
    auto comm = world_comm();
#if DTL_ENABLE_MPI
    if (dtl::mpi::is_initialized()) {
        EXPECT_GE(comm.rank(), 0);
        EXPECT_GE(comm.size(), 1);
    } else {
        // Pre-init access yields sentinel values for MPI communicator.
        EXPECT_EQ(comm.rank(), dtl::no_rank);
        EXPECT_EQ(comm.size(), dtl::rank_t{0});
    }
#else
    EXPECT_GE(comm.rank(), 0);
    EXPECT_GE(comm.size(), 1);
#endif
}

TEST(DefaultCommunicatorTest, WorldCommRankLessThanSize) {
    auto comm = world_comm();
    EXPECT_LT(comm.rank(), comm.size());
}

#pragma GCC diagnostic pop

// =============================================================================
// null_communicator Basic Operations
// =============================================================================

TEST(NullCommunicatorTest, RankIsZero) {
    null_communicator comm;
    EXPECT_EQ(comm.rank(), 0);
}

TEST(NullCommunicatorTest, SizeIsOne) {
    null_communicator comm;
    EXPECT_EQ(comm.size(), 1);
}

TEST(NullCommunicatorTest, IsRootReturnsTrue) {
    null_communicator comm;
    EXPECT_TRUE(comm.is_root());
}

TEST(NullCommunicatorTest, BarrierIsNoOp) {
    null_communicator comm;
    comm.barrier();  // Should not throw or crash
}

// =============================================================================
// null_communicator Template Convenience Methods
// =============================================================================

TEST(NullCommunicatorTest, AllreduceSumValueReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.allreduce_sum_value<int>(42), 42);
    EXPECT_EQ(comm.allreduce_sum_value<long>(100L), 100L);
    EXPECT_DOUBLE_EQ(comm.allreduce_sum_value<double>(3.14), 3.14);
}

TEST(NullCommunicatorTest, AllreduceMinValueReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.allreduce_min_value<int>(7), 7);
    EXPECT_EQ(comm.allreduce_min_value<long>(-5L), -5L);
}

TEST(NullCommunicatorTest, AllreduceMaxValueReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.allreduce_max_value<int>(99), 99);
    EXPECT_DOUBLE_EQ(comm.allreduce_max_value<double>(2.718), 2.718);
}

TEST(NullCommunicatorTest, AllreduceProdValueReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.allreduce_prod_value<int>(6), 6);
}

TEST(NullCommunicatorTest, AllreduceLandValueReturnsInput) {
    null_communicator comm;
    EXPECT_TRUE(comm.allreduce_land_value(true));
    EXPECT_FALSE(comm.allreduce_land_value(false));
}

TEST(NullCommunicatorTest, AllreduceLorValueReturnsInput) {
    null_communicator comm;
    EXPECT_TRUE(comm.allreduce_lor_value(true));
    EXPECT_FALSE(comm.allreduce_lor_value(false));
}

TEST(NullCommunicatorTest, ReduceSumToRootReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.reduce_sum_to_root<int>(42, 0), 42);
    EXPECT_EQ(comm.reduce_sum_to_root<long>(99L, 0), 99L);
}

// =============================================================================
// null_communicator Scan Operations
// =============================================================================

TEST(NullCommunicatorTest, ScanSumValueReturnsInput) {
    null_communicator comm;
    EXPECT_EQ(comm.scan_sum_value<int>(7), 7);
    EXPECT_EQ(comm.scan_sum_value<long>(100L), 100L);
}

TEST(NullCommunicatorTest, ExscanSumValueReturnsZero) {
    null_communicator comm;
    EXPECT_EQ(comm.exscan_sum_value<int>(7), 0);
    EXPECT_EQ(comm.exscan_sum_value<long>(100L), 0L);
    EXPECT_DOUBLE_EQ(comm.exscan_sum_value<double>(3.14), 0.0);
}

// =============================================================================
// null_communicator Buffer-Level Operations
// =============================================================================

TEST(NullCommunicatorTest, BroadcastIsNoOp) {
    null_communicator comm;
    int value = 42;
    comm.broadcast(&value, sizeof(int), 0);
    EXPECT_EQ(value, 42);
}

TEST(NullCommunicatorTest, GatherCopiesData) {
    null_communicator comm;
    int send = 42;
    int recv = 0;
    comm.gather(&send, &recv, sizeof(int), 0);
    EXPECT_EQ(recv, 42);
}

TEST(NullCommunicatorTest, ScatterCopiesData) {
    null_communicator comm;
    int send = 99;
    int recv = 0;
    comm.scatter(&send, &recv, sizeof(int), 0);
    EXPECT_EQ(recv, 99);
}

TEST(NullCommunicatorTest, AllgatherCopiesData) {
    null_communicator comm;
    int send = 77;
    int recv = 0;
    comm.allgather(&send, &recv, sizeof(int));
    EXPECT_EQ(recv, 77);
}

TEST(NullCommunicatorTest, AllreduceSumIntCopiesData) {
    null_communicator comm;
    int send = 42;
    int recv = 0;
    comm.allreduce_sum_int(&send, &recv, 1);
    EXPECT_EQ(recv, 42);
}

TEST(NullCommunicatorTest, ScanSumIntCopiesData) {
    null_communicator comm;
    int send = 10;
    int recv = 0;
    comm.scan_sum_int(&send, &recv, 1);
    EXPECT_EQ(recv, 10);
}

TEST(NullCommunicatorTest, ExscanSumIntZerosOutput) {
    null_communicator comm;
    int send = 10;
    int recv = 99;
    comm.exscan_sum_int(&send, &recv, 1);
    EXPECT_EQ(recv, 0);
}

TEST(NullCommunicatorTest, TestAlwaysReturnsTrue) {
    null_communicator comm;
    request_handle req{};
    EXPECT_TRUE(comm.test(req));
}

TEST(NullCommunicatorTest, WaitIsNoOp) {
    null_communicator comm;
    request_handle req{};
    comm.wait(req);  // Should not throw
}

}  // namespace dtl::test
