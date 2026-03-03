// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_topology_coverage.cpp
/// @brief Expanded unit tests for the DTL topology module
/// @details Phase 14 T08: hardware_topology, proximity_matrix,
///          cpu_set/affinity, rank_host_map/network topology.

#include <dtl/topology/topology.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// hardware_topology Tests
// =============================================================================

TEST(HardwareTopologyTest, DiscoverLocalNonEmpty) {
    const auto& topo = dtl::topology::local_topology();
    EXPECT_GT(topo.total_cpus, 0u);
    EXPECT_FALSE(topo.empty());
}

TEST(HardwareTopologyTest, HostnameNonEmpty) {
    const auto& topo = dtl::topology::local_topology();
    EXPECT_FALSE(topo.hostname.empty());
}

TEST(HardwareTopologyTest, NumaNodesPresent) {
    const auto& topo = dtl::topology::local_topology();
    EXPECT_GE(topo.num_numa_nodes(), 1u);
}

TEST(HardwareTopologyTest, CpuCount) {
    auto count = dtl::topology::cpu_count();
    EXPECT_GT(count, 0u);
}

TEST(HardwareTopologyTest, NumaNodeCount) {
    auto count = dtl::topology::numa_node_count();
    EXPECT_GE(count, 1u);
}

TEST(HardwareTopologyTest, GpuCountNoCrash) {
    // May be 0 in CI without GPUs, but shouldn't crash
    auto count = dtl::topology::gpu_count();
    EXPECT_GE(count, 0u);
}

TEST(HardwareTopologyTest, HasNumaCheck) {
    const auto& topo = dtl::topology::local_topology();
    if (topo.numa_nodes.size() > 1) {
        EXPECT_TRUE(topo.has_numa());
    } else {
        EXPECT_FALSE(topo.has_numa());
    }
}

TEST(HardwareTopologyTest, CpusOnNumaNode) {
    auto cpus = dtl::topology::cpus_on_numa_node(0);
    EXPECT_FALSE(cpus.empty());
}

TEST(HardwareTopologyTest, CpusOnInvalidNumaNode) {
    auto cpus = dtl::topology::cpus_on_numa_node(9999);
    EXPECT_TRUE(cpus.empty());
}

TEST(HardwareTopologyTest, NumaNodeForCpu) {
    auto node = dtl::topology::numa_node_for_cpu(0);
    // Should return some valid node
    EXPECT_GE(node, 0u);
}

TEST(HardwareTopologyTest, EmptyTopology) {
    dtl::topology::hardware_topology empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.num_numa_nodes(), 0u);
    EXPECT_EQ(empty.num_gpus(), 0u);
    EXPECT_FALSE(empty.has_numa());
    EXPECT_FALSE(empty.has_gpus());
}

// =============================================================================
// proximity_matrix Tests
// =============================================================================

TEST(ProximityMatrixTest, DefaultConstruction) {
    dtl::topology::proximity_matrix pm;
    EXPECT_EQ(pm.size(), 0u);
    EXPECT_TRUE(pm.empty());
}

TEST(ProximityMatrixTest, ConstructWithSize) {
    dtl::topology::proximity_matrix pm(4);
    EXPECT_EQ(pm.size(), 4u);
    EXPECT_FALSE(pm.empty());
}

TEST(ProximityMatrixTest, DiagonalIsZero) {
    dtl::topology::proximity_matrix pm(3);
    EXPECT_EQ(pm.distance(0, 0), dtl::topology::local_distance);
    EXPECT_EQ(pm.distance(1, 1), dtl::topology::local_distance);
    EXPECT_EQ(pm.distance(2, 2), dtl::topology::local_distance);
}

TEST(ProximityMatrixTest, OffDiagonalIsMaxByDefault) {
    dtl::topology::proximity_matrix pm(3);
    EXPECT_EQ(pm.distance(0, 1), dtl::topology::max_distance);
    EXPECT_EQ(pm.distance(1, 2), dtl::topology::max_distance);
}

TEST(ProximityMatrixTest, SetDistanceSymmetric) {
    dtl::topology::proximity_matrix pm(3);
    pm.set_distance(0, 2, dtl::topology::adjacent_distance);
    EXPECT_EQ(pm.distance(0, 2), dtl::topology::adjacent_distance);
    EXPECT_EQ(pm.distance(2, 0), dtl::topology::adjacent_distance);
}

TEST(ProximityMatrixTest, OutOfBoundsReturnsMax) {
    dtl::topology::proximity_matrix pm(2);
    EXPECT_EQ(pm.distance(5, 0), dtl::topology::max_distance);
    EXPECT_EQ(pm.distance(0, 5), dtl::topology::max_distance);
}

TEST(ProximityMatrixTest, SetDistanceOutOfBoundsNoOp) {
    dtl::topology::proximity_matrix pm(2);
    pm.set_distance(5, 0, 10);  // Should not crash
    EXPECT_EQ(pm.distance(0, 1), dtl::topology::max_distance);
}

TEST(ProximityMatrixTest, NearestNeighbor) {
    dtl::topology::proximity_matrix pm(3);
    pm.set_distance(0, 1, dtl::topology::adjacent_distance);
    pm.set_distance(0, 2, dtl::topology::remote_distance);

    auto nearest = pm.nearest(0);
    EXPECT_EQ(nearest, 1u);  // 1 is closer than 2
}

TEST(ProximityMatrixTest, NearestSingleElement) {
    dtl::topology::proximity_matrix pm(1);
    auto nearest = pm.nearest(0);
    EXPECT_EQ(nearest, 0u);  // Only self
}

TEST(ProximityMatrixTest, KNearestNeighbors) {
    dtl::topology::proximity_matrix pm(4);
    pm.set_distance(0, 1, 10);
    pm.set_distance(0, 2, 20);
    pm.set_distance(0, 3, 30);

    auto k2 = pm.k_nearest(0, 2);
    ASSERT_EQ(k2.size(), 2u);
    EXPECT_EQ(k2[0], 1u);  // closest
    EXPECT_EQ(k2[1], 2u);  // second closest
}

TEST(ProximityMatrixTest, KNearestMoreThanAvailable) {
    dtl::topology::proximity_matrix pm(3);
    pm.set_distance(0, 1, 10);
    pm.set_distance(0, 2, 20);

    auto k5 = pm.k_nearest(0, 5);
    EXPECT_EQ(k5.size(), 2u);  // Only 2 neighbors available
}

TEST(ProximityMatrixTest, WithinDistance) {
    dtl::topology::proximity_matrix pm(4);
    pm.set_distance(0, 1, 10);
    pm.set_distance(0, 2, 20);
    pm.set_distance(0, 3, 100);

    auto within = pm.within_distance(0, 20);
    EXPECT_GE(within.size(), 3u);  // self (0), 1, 2
    EXPECT_TRUE(std::find(within.begin(), within.end(), 0u) != within.end());
    EXPECT_TRUE(std::find(within.begin(), within.end(), 1u) != within.end());
    EXPECT_TRUE(std::find(within.begin(), within.end(), 2u) != within.end());
}

TEST(ProximityMatrixTest, KNearestOutOfBounds) {
    dtl::topology::proximity_matrix pm(3);
    auto result = pm.k_nearest(10, 2);
    EXPECT_TRUE(result.empty());
}

// =============================================================================
// Distance Constants Tests
// =============================================================================

TEST(DistanceConstantsTest, Ordering) {
    EXPECT_EQ(dtl::topology::local_distance, 0u);
    EXPECT_LT(dtl::topology::adjacent_distance, dtl::topology::remote_distance);
    EXPECT_LT(dtl::topology::remote_distance, dtl::topology::network_distance);
    EXPECT_LT(dtl::topology::network_distance, dtl::topology::max_distance);
}

// =============================================================================
// cpu_set Tests
// =============================================================================

TEST(CpuSetTest, DefaultConstruction) {
    dtl::topology::cpu_set cs;
    EXPECT_TRUE(cs.empty());
    EXPECT_EQ(cs.count(), 0u);
}

TEST(CpuSetTest, ConstructFromSingleCpu) {
    dtl::topology::cpu_set cs(3);
    EXPECT_FALSE(cs.empty());
    EXPECT_EQ(cs.count(), 1u);
    EXPECT_TRUE(cs.contains(3));
    EXPECT_FALSE(cs.contains(0));
}

TEST(CpuSetTest, ConstructFromRange) {
    dtl::topology::cpu_set cs(2, 5);
    EXPECT_EQ(cs.count(), 4u);
    EXPECT_TRUE(cs.contains(2));
    EXPECT_TRUE(cs.contains(3));
    EXPECT_TRUE(cs.contains(4));
    EXPECT_TRUE(cs.contains(5));
    EXPECT_FALSE(cs.contains(1));
    EXPECT_FALSE(cs.contains(6));
}

TEST(CpuSetTest, ConstructFromVector) {
    std::vector<std::uint32_t> cpus = {0, 4, 8};
    dtl::topology::cpu_set cs(cpus);
    EXPECT_EQ(cs.count(), 3u);
    EXPECT_TRUE(cs.contains(0));
    EXPECT_TRUE(cs.contains(4));
    EXPECT_TRUE(cs.contains(8));
}

TEST(CpuSetTest, AddRemove) {
    dtl::topology::cpu_set cs;
    cs.add(5);
    EXPECT_TRUE(cs.contains(5));
    cs.remove(5);
    EXPECT_FALSE(cs.contains(5));
}

TEST(CpuSetTest, Clear) {
    dtl::topology::cpu_set cs(0, 7);
    EXPECT_EQ(cs.count(), 8u);
    cs.clear();
    EXPECT_TRUE(cs.empty());
}

TEST(CpuSetTest, First) {
    dtl::topology::cpu_set cs;
    cs.add(3);
    cs.add(10);
    EXPECT_EQ(cs.first(), 3u);
}

TEST(CpuSetTest, FirstEmpty) {
    dtl::topology::cpu_set cs;
    EXPECT_EQ(cs.first(), dtl::topology::cpu_set::max_cpus);
}

TEST(CpuSetTest, ToVector) {
    dtl::topology::cpu_set cs;
    cs.add(1);
    cs.add(3);
    cs.add(5);
    auto vec = cs.to_vector();
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 1u);
    EXPECT_EQ(vec[1], 3u);
    EXPECT_EQ(vec[2], 5u);
}

TEST(CpuSetTest, Union) {
    dtl::topology::cpu_set a(0, 3);
    dtl::topology::cpu_set b(2, 5);
    auto result = a | b;
    EXPECT_EQ(result.count(), 6u);  // 0,1,2,3,4,5
}

TEST(CpuSetTest, Intersection) {
    dtl::topology::cpu_set a(0, 3);
    dtl::topology::cpu_set b(2, 5);
    auto result = a & b;
    EXPECT_EQ(result.count(), 2u);  // 2,3
    EXPECT_TRUE(result.contains(2));
    EXPECT_TRUE(result.contains(3));
}

TEST(CpuSetTest, Difference) {
    dtl::topology::cpu_set a(0, 3);
    dtl::topology::cpu_set b(2, 5);
    auto result = a - b;
    EXPECT_EQ(result.count(), 2u);  // 0,1
    EXPECT_TRUE(result.contains(0));
    EXPECT_TRUE(result.contains(1));
}

TEST(CpuSetTest, Equality) {
    dtl::topology::cpu_set a(0, 3);
    dtl::topology::cpu_set b(0, 3);
    dtl::topology::cpu_set c(1, 4);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(CpuSetTest, ContainsOutOfRange) {
    dtl::topology::cpu_set cs;
    EXPECT_FALSE(cs.contains(dtl::topology::cpu_set::max_cpus + 1));
}

// =============================================================================
// rank_host_map Tests
// =============================================================================

TEST(RankHostMapTest, DefaultConstruction) {
    dtl::topology::rank_host_map rhm;
    EXPECT_EQ(rhm.size(), 0u);
}

TEST(RankHostMapTest, ConstructFromHostnames) {
    std::vector<std::string> hosts = {"node0", "node0", "node1", "node1"};
    dtl::topology::rank_host_map rhm(hosts);
    EXPECT_EQ(rhm.size(), 4u);
}

TEST(RankHostMapTest, HostnameForRank) {
    std::vector<std::string> hosts = {"alpha", "beta", "gamma"};
    dtl::topology::rank_host_map rhm(hosts);
    EXPECT_EQ(rhm.hostname_for_rank(0), "alpha");
    EXPECT_EQ(rhm.hostname_for_rank(1), "beta");
    EXPECT_EQ(rhm.hostname_for_rank(2), "gamma");
}

TEST(RankHostMapTest, HostnameForInvalidRank) {
    std::vector<std::string> hosts = {"a"};
    dtl::topology::rank_host_map rhm(hosts);
    EXPECT_EQ(rhm.hostname_for_rank(99), "");
    EXPECT_EQ(rhm.hostname_for_rank(-1), "");
}

TEST(RankHostMapTest, SameNode) {
    std::vector<std::string> hosts = {"node0", "node0", "node1", "node1"};
    dtl::topology::rank_host_map rhm(hosts);
    EXPECT_TRUE(rhm.same_node(0, 1));
    EXPECT_TRUE(rhm.same_node(2, 3));
    EXPECT_FALSE(rhm.same_node(0, 2));
    EXPECT_FALSE(rhm.same_node(1, 3));
}

TEST(RankHostMapTest, RanksOnSameNode) {
    std::vector<std::string> hosts = {"n0", "n0", "n1", "n0"};
    dtl::topology::rank_host_map rhm(hosts);
    auto same = rhm.ranks_on_same_node(0);
    EXPECT_EQ(same.size(), 3u);  // ranks 0, 1, 3
}

TEST(RankHostMapTest, RanksByProximity) {
    std::vector<std::string> hosts = {"local", "remote", "local"};
    dtl::topology::rank_host_map rhm(hosts);
    auto proximity = rhm.ranks_by_proximity(0);
    ASSERT_EQ(proximity.size(), 3u);
    // Same-node ranks should come first
    EXPECT_TRUE(rhm.same_node(0, proximity[0]));
}

TEST(RankHostMapTest, UniqueHosts) {
    std::vector<std::string> hosts = {"a", "b", "a", "c", "b"};
    dtl::topology::rank_host_map rhm(hosts);
    auto unique = rhm.unique_hosts();
    EXPECT_EQ(unique.size(), 3u);
    EXPECT_EQ(rhm.num_hosts(), 3u);
}

// =============================================================================
// build_rank_proximity Tests
// =============================================================================

TEST(BuildRankProximityTest, SameNodeAdjacent) {
    std::vector<std::string> hosts = {"n0", "n0", "n1", "n1"};
    dtl::topology::rank_host_map rhm(hosts);
    auto prox = dtl::topology::build_rank_proximity(rhm);

    EXPECT_EQ(prox.size(), 4u);
    EXPECT_EQ(prox.distance(0, 1), dtl::topology::adjacent_distance);
    EXPECT_EQ(prox.distance(2, 3), dtl::topology::adjacent_distance);
}

TEST(BuildRankProximityTest, DifferentNodeNetwork) {
    std::vector<std::string> hosts = {"n0", "n1"};
    dtl::topology::rank_host_map rhm(hosts);
    auto prox = dtl::topology::build_rank_proximity(rhm);

    EXPECT_EQ(prox.distance(0, 1), dtl::topology::network_distance);
}

TEST(BuildRankProximityTest, EmptyMap) {
    dtl::topology::rank_host_map rhm;
    auto prox = dtl::topology::build_rank_proximity(rhm);
    EXPECT_TRUE(prox.empty());
}

// =============================================================================
// build_cpu_proximity Tests
// =============================================================================

TEST(BuildCpuProximityTest, WithLocalTopology) {
    const auto& topo = dtl::topology::local_topology();
    if (topo.total_cpus > 0) {
        auto prox = dtl::topology::build_cpu_proximity(topo);
        EXPECT_EQ(prox.size(), static_cast<dtl::size_type>(topo.total_cpus));
        // Diagonal is 0
        EXPECT_EQ(prox.distance(0, 0), dtl::topology::local_distance);
    }
}

TEST(BuildCpuProximityTest, EmptyTopology) {
    dtl::topology::hardware_topology empty;
    auto prox = dtl::topology::build_cpu_proximity(empty);
    EXPECT_TRUE(prox.empty());
}

// =============================================================================
// get_hostname Tests
// =============================================================================

TEST(NetworkTest, GetHostname) {
    auto hostname = dtl::topology::get_hostname();
    EXPECT_FALSE(hostname.empty());
}

}  // namespace dtl::test
