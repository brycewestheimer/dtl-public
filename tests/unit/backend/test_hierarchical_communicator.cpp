// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hierarchical_communicator.cpp
/// @brief Unit tests for hierarchical communicator (MPI + NCCL two-level)
/// @details Tests the hierarchical_communicator's topology detection, level
///          determination, and API structure without requiring actual MPI/NCCL
///          hardware (tests work in both enabled and disabled configurations).
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <backends/hybrid/hierarchical_communicator.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Topology Struct Tests
// =============================================================================

TEST(HierarchicalCommTest, NodeTopologyDefaults) {
    dtl::hybrid::node_topology topo;
    EXPECT_EQ(topo.world_rank, dtl::no_rank);
    EXPECT_EQ(topo.world_size, 0);
    EXPECT_EQ(topo.node_index, 0u);
    EXPECT_EQ(topo.num_nodes, 0u);
    EXPECT_EQ(topo.local_rank, 0);
    EXPECT_EQ(topo.local_size, 0);
}

TEST(HierarchicalCommTest, NodeTopologyIsLeader) {
    dtl::hybrid::node_topology topo;
    topo.local_rank = 0;
    EXPECT_TRUE(topo.is_node_leader());

    topo.local_rank = 1;
    EXPECT_FALSE(topo.is_node_leader());
}

// =============================================================================
// comm_level Enum Tests
// =============================================================================

TEST(HierarchicalCommTest, CommLevelValues) {
    EXPECT_NE(dtl::hybrid::comm_level::intra_gpu, dtl::hybrid::comm_level::intra_node);
    EXPECT_NE(dtl::hybrid::comm_level::intra_node, dtl::hybrid::comm_level::inter_node);
    EXPECT_NE(dtl::hybrid::comm_level::inter_node, dtl::hybrid::comm_level::automatic);
}

// =============================================================================
// Default-Constructed Communicator Tests
// =============================================================================

TEST(HierarchicalCommTest, DefaultConstructionValid) {
    dtl::hybrid::hierarchical_communicator comm;
    EXPECT_EQ(comm.rank(), dtl::no_rank);
    EXPECT_EQ(comm.size(), 0);
#if DTL_ENABLE_MPI
    EXPECT_FALSE(comm.valid());
#endif
}

TEST(HierarchicalCommTest, PropertiesDefaultComm) {
    dtl::hybrid::hierarchical_communicator comm;
    auto props = comm.properties();
    EXPECT_EQ(props.rank, dtl::no_rank);
    EXPECT_EQ(props.size, 0);
    EXPECT_FALSE(props.is_inter);
    EXPECT_EQ(std::string(props.name), "hierarchical");
}

TEST(HierarchicalCommTest, DetermineLevel) {
    dtl::hybrid::hierarchical_communicator comm;
    // Same rank → intra_gpu
    EXPECT_EQ(comm.determine_level(0, 0), dtl::hybrid::comm_level::intra_gpu);
    EXPECT_EQ(comm.determine_level(5, 5), dtl::hybrid::comm_level::intra_gpu);
}

TEST(HierarchicalCommTest, TopologyAccessible) {
    dtl::hybrid::hierarchical_communicator comm;
    const auto& topo = comm.topology();
    EXPECT_EQ(topo.world_rank, dtl::no_rank);
    EXPECT_EQ(topo.world_size, 0);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST(HierarchicalCommTest, ConfigDefaults) {
    dtl::hybrid::hierarchical_communicator::config cfg;
    EXPECT_TRUE(cfg.enable_gpu_direct);
    EXPECT_TRUE(cfg.enable_host_staging);
    EXPECT_EQ(cfg.staging_buffer_size, 64u * 1024u * 1024u);
}

// =============================================================================
// Move Semantics
// =============================================================================

TEST(HierarchicalCommTest, MoveConstructor) {
    dtl::hybrid::hierarchical_communicator a;
    dtl::hybrid::hierarchical_communicator b(std::move(a));
    EXPECT_EQ(b.rank(), dtl::no_rank);
    EXPECT_EQ(b.size(), 0);
}

TEST(HierarchicalCommTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<dtl::hybrid::hierarchical_communicator>,
                  "hierarchical_communicator must not be copyable");
    static_assert(!std::is_copy_assignable_v<dtl::hybrid::hierarchical_communicator>,
                  "hierarchical_communicator must not be copy-assignable");
    SUCCEED();
}

// =============================================================================
// Barrier / Broadcast on default comm (stubs — should not crash)
// =============================================================================

TEST(HierarchicalCommTest, BarrierDefaultComm) {
    dtl::hybrid::hierarchical_communicator comm;
    auto result = comm.barrier();
    // Default comm has no MPI/NCCL → barrier is a no-op
    EXPECT_TRUE(result.has_value());
}

TEST(HierarchicalCommTest, BroadcastDefaultComm) {
    dtl::hybrid::hierarchical_communicator comm;
    int data = 42;
    auto result = comm.broadcast_impl(&data, 1, sizeof(int), 0);
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Method Signature Verification
// =============================================================================

TEST(HierarchicalCommTest, SendImplSignature) {
    using comm_t = dtl::hybrid::hierarchical_communicator;
    static_assert(std::is_same_v<
        decltype(std::declval<comm_t>().send_impl(
            std::declval<const void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::size_type>(), std::declval<dtl::rank_t>(),
            std::declval<int>())),
        dtl::result<void>>,
        "send_impl must return result<void>");
    SUCCEED();
}

TEST(HierarchicalCommTest, RecvImplSignature) {
    using comm_t = dtl::hybrid::hierarchical_communicator;
    static_assert(std::is_same_v<
        decltype(std::declval<comm_t>().recv_impl(
            std::declval<void*>(), std::declval<dtl::size_type>(),
            std::declval<dtl::size_type>(), std::declval<dtl::rank_t>(),
            std::declval<int>())),
        dtl::result<void>>,
        "recv_impl must return result<void>");
    SUCCEED();
}

#if DTL_ENABLE_MPI
// =============================================================================
// MPI-Dependent Tests (only when MPI is enabled)
// =============================================================================

TEST(HierarchicalCommTest, NativeHandleAccessCompiles) {
    dtl::hybrid::hierarchical_communicator comm;
    EXPECT_EQ(comm.world_comm(), MPI_COMM_NULL);
    EXPECT_EQ(comm.node_comm(), MPI_COMM_NULL);
    EXPECT_EQ(comm.leader_comm(), MPI_COMM_NULL);
}

#endif  // DTL_ENABLE_MPI

#if DTL_ENABLE_NCCL
TEST(HierarchicalCommTest, NcclHandleAccessCompiles) {
    dtl::hybrid::hierarchical_communicator comm;
    EXPECT_EQ(comm.nccl_comm(), nullptr);
}
#endif  // DTL_ENABLE_NCCL

}  // namespace dtl::test
