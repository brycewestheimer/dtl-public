// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hardware.cpp
/// @brief Unit tests for dtl/topology/hardware.hpp
/// @details Tests hardware discovery and query functions.

#include <dtl/topology/topology.hpp>

#include <gtest/gtest.h>

namespace dtl::topology::test {

// =============================================================================
// Hardware Discovery Tests
// =============================================================================

TEST(HardwareDiscoveryTest, DiscoverRuns) {
    // Should not throw
    auto topo = discover_local();

    // Should have at least 1 CPU
    EXPECT_GT(topo.total_cpus, 0u);
}

TEST(HardwareDiscoveryTest, LocalTopologySingleton) {
    const auto& topo1 = local_topology();
    const auto& topo2 = local_topology();

    // Should be same instance
    EXPECT_EQ(&topo1, &topo2);
}

TEST(HardwareDiscoveryTest, CpuCountValid) {
    auto count = cpu_count();
    EXPECT_GT(count, 0u);
    EXPECT_LE(count, 65536u);  // Reasonable upper bound
}

TEST(HardwareDiscoveryTest, NumaDetected) {
    auto count = numa_node_count();
    EXPECT_GE(count, 1u);  // At least 1 NUMA node
}

TEST(HardwareDiscoveryTest, TopologyNotEmpty) {
    const auto& topo = local_topology();
    EXPECT_FALSE(topo.empty());
}

TEST(HardwareDiscoveryTest, HasHostname) {
    const auto& topo = local_topology();
    EXPECT_FALSE(topo.hostname.empty());
}

// =============================================================================
// NUMA Query Tests
// =============================================================================

TEST(NumaQueryTest, CpusOnNumaNode) {
    auto cpus = cpus_on_numa_node(0);
    EXPECT_FALSE(cpus.empty());
}

TEST(NumaQueryTest, NumaNodeForCpu) {
    // CPU 0 should be on some NUMA node
    auto node = numa_node_for_cpu(0);
    EXPECT_GE(node, 0u);
}

TEST(NumaQueryTest, InvalidNumaNodeReturnsEmpty) {
    auto cpus = cpus_on_numa_node(9999);
    EXPECT_TRUE(cpus.empty());
}

// =============================================================================
// GPU Query Tests
// =============================================================================

TEST(GpuQueryTest, GpuCountNonNegative) {
    auto count = gpu_count();
    EXPECT_GE(count, 0u);
}

TEST(GpuQueryTest, GpuDevicesConsistent) {
    const auto& topo = local_topology();
    EXPECT_EQ(topo.num_gpus(), gpu_count());
}

// =============================================================================
// Hardware Topology Struct Tests
// =============================================================================

TEST(HardwareTopologyTest, HasNumaMethod) {
    const auto& topo = local_topology();
    // If more than 1 NUMA node, has_numa should be true
    bool expected = topo.numa_nodes.size() > 1;
    EXPECT_EQ(topo.has_numa(), expected);
}

TEST(HardwareTopologyTest, HasGpusMethod) {
    const auto& topo = local_topology();
    bool expected = !topo.gpus.empty();
    EXPECT_EQ(topo.has_gpus(), expected);
}

TEST(HardwareTopologyTest, TotalCpusMatchesCores) {
    const auto& topo = local_topology();
    // total_cpus should be >= total_cores (due to SMT)
    EXPECT_GE(topo.total_cpus, topo.total_cores);
}

}  // namespace dtl::topology::test
