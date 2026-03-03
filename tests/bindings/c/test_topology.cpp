// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_topology.cpp
 * @brief Unit tests for DTL C bindings topology operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>

// ============================================================================
// CPU Topology Tests
// ============================================================================

TEST(CBindingsTopology, NumCpusSucceeds) {
    int count = -1;
    dtl_status status = dtl_topology_num_cpus(&count);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_GE(count, 0);
}

TEST(CBindingsTopology, NumCpusWithNullFails) {
    dtl_status status = dtl_topology_num_cpus(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsTopology, CpuAffinitySucceeds) {
    int cpu_id = -1;
    dtl_status status = dtl_topology_cpu_affinity(0, &cpu_id);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_GE(cpu_id, 0);
}

TEST(CBindingsTopology, CpuAffinityWithNullFails) {
    dtl_status status = dtl_topology_cpu_affinity(0, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// GPU Topology Tests
// ============================================================================

TEST(CBindingsTopology, NumGpusSucceeds) {
    int count = -1;
    dtl_status status = dtl_topology_num_gpus(&count);
    EXPECT_EQ(status, DTL_SUCCESS);
    // Count may be 0 if no GPU backend enabled
    EXPECT_GE(count, 0);
}

TEST(CBindingsTopology, NumGpusWithNullFails) {
    dtl_status status = dtl_topology_num_gpus(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsTopology, GpuIdSucceeds) {
    int gpu_id = -2;
    dtl_status status = dtl_topology_gpu_id(0, &gpu_id);
    EXPECT_EQ(status, DTL_SUCCESS);
    // gpu_id is -1 if no GPUs available, >=0 otherwise
    EXPECT_GE(gpu_id, -1);
}

TEST(CBindingsTopology, GpuIdWithNullFails) {
    dtl_status status = dtl_topology_gpu_id(0, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Node Locality Tests
// ============================================================================

TEST(CBindingsTopology, IsLocalSameRankIsLocal) {
    int is_local = 0;
    dtl_status status = dtl_topology_is_local(0, 0, &is_local);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(is_local, 1);
}

TEST(CBindingsTopology, IsLocalWithNullFails) {
    dtl_status status = dtl_topology_is_local(0, 0, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsTopology, NodeIdSucceeds) {
    int node_id = -1;
    dtl_status status = dtl_topology_node_id(0, &node_id);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_GE(node_id, 0);
}

TEST(CBindingsTopology, NodeIdWithNullFails) {
    dtl_status status = dtl_topology_node_id(0, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}
