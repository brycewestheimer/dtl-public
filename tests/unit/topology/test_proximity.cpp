// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_proximity.cpp
/// @brief Unit tests for dtl/topology/proximity.hpp
/// @details Tests proximity matrix operations.

#include <dtl/topology/topology.hpp>

#include <gtest/gtest.h>

namespace dtl::topology::test {

// =============================================================================
// Proximity Matrix Basic Tests
// =============================================================================

TEST(ProximityMatrixTest, EmptyMatrix) {
    proximity_matrix prox;
    EXPECT_TRUE(prox.empty());
    EXPECT_EQ(prox.size(), 0u);
}

TEST(ProximityMatrixTest, SingleElement) {
    proximity_matrix prox(1);
    EXPECT_EQ(prox.size(), 1u);
    EXPECT_EQ(prox.distance(0, 0), local_distance);
}

TEST(ProximityMatrixTest, DiagonalIsZero) {
    proximity_matrix prox(5);

    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(prox.distance(i, i), local_distance);
    }
}

TEST(ProximityMatrixTest, SetDistance) {
    proximity_matrix prox(3);

    prox.set_distance(0, 1, 10);
    prox.set_distance(0, 2, 20);
    prox.set_distance(1, 2, 30);

    EXPECT_EQ(prox.distance(0, 1), 10u);
    EXPECT_EQ(prox.distance(0, 2), 20u);
    EXPECT_EQ(prox.distance(1, 2), 30u);
}

TEST(ProximityMatrixTest, Symmetric) {
    proximity_matrix prox(3);

    prox.set_distance(0, 1, 10);

    EXPECT_EQ(prox.distance(0, 1), prox.distance(1, 0));
}

// =============================================================================
// Nearest Neighbor Tests
// =============================================================================

TEST(ProximityMatrixTest, NearestSelf) {
    proximity_matrix prox(1);
    EXPECT_EQ(prox.nearest(0), 0u);
}

TEST(ProximityMatrixTest, NearestBasic) {
    proximity_matrix prox(3);

    prox.set_distance(0, 1, 100);
    prox.set_distance(0, 2, 10);  // 2 is nearest to 0

    EXPECT_EQ(prox.nearest(0), 2u);
}

TEST(ProximityMatrixTest, KNearest) {
    proximity_matrix prox(5);

    // Set distances from node 0
    prox.set_distance(0, 1, 10);  // Nearest
    prox.set_distance(0, 2, 30);  // Third
    prox.set_distance(0, 3, 20);  // Second
    prox.set_distance(0, 4, 40);  // Fourth

    auto nearest = prox.k_nearest(0, 3);

    ASSERT_EQ(nearest.size(), 3u);
    EXPECT_EQ(nearest[0], 1u);  // Distance 10
    EXPECT_EQ(nearest[1], 3u);  // Distance 20
    EXPECT_EQ(nearest[2], 2u);  // Distance 30
}

TEST(ProximityMatrixTest, KNearestOrdered) {
    proximity_matrix prox(4);

    prox.set_distance(0, 1, 30);
    prox.set_distance(0, 2, 10);
    prox.set_distance(0, 3, 20);

    auto nearest = prox.k_nearest(0, 3);

    // Verify ordering by checking distances are non-decreasing
    for (size_type i = 1; i < nearest.size(); ++i) {
        EXPECT_LE(prox.distance(0, nearest[i-1]),
                  prox.distance(0, nearest[i]));
    }
}

// =============================================================================
// Within Distance Tests
// =============================================================================

TEST(ProximityMatrixTest, WithinDistance) {
    proximity_matrix prox(5);

    prox.set_distance(0, 1, 5);
    prox.set_distance(0, 2, 15);
    prox.set_distance(0, 3, 5);
    prox.set_distance(0, 4, 25);

    auto within = prox.within_distance(0, 10);

    // Should include 0 (self), 1, 3
    EXPECT_EQ(within.size(), 3u);

    bool has_self = std::find(within.begin(), within.end(), 0) != within.end();
    bool has_1 = std::find(within.begin(), within.end(), 1) != within.end();
    bool has_3 = std::find(within.begin(), within.end(), 3) != within.end();

    EXPECT_TRUE(has_self);
    EXPECT_TRUE(has_1);
    EXPECT_TRUE(has_3);
}

// =============================================================================
// Boundary Tests
// =============================================================================

TEST(ProximityMatrixTest, OutOfBoundsDistance) {
    proximity_matrix prox(3);

    EXPECT_EQ(prox.distance(10, 0), max_distance);
    EXPECT_EQ(prox.distance(0, 10), max_distance);
}

// =============================================================================
// CPU Proximity Builder Tests
// =============================================================================

TEST(CpuProximityTest, BuildFromTopology) {
    const auto& topo = local_topology();
    auto prox = build_cpu_proximity(topo);

    EXPECT_EQ(prox.size(), topo.total_cpus);
}

TEST(CpuProximityTest, SameCpuIsLocal) {
    const auto& topo = local_topology();
    auto prox = build_cpu_proximity(topo);

    if (!prox.empty()) {
        EXPECT_EQ(prox.distance(0, 0), local_distance);
    }
}

// =============================================================================
// NUMA Proximity Builder Tests
// =============================================================================

TEST(NumaProximityTest, BuildFromTopology) {
    const auto& topo = local_topology();
    auto prox = build_numa_proximity(topo);

    EXPECT_EQ(prox.size(), topo.numa_nodes.size());
}

}  // namespace dtl::topology::test
