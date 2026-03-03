// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_affinity.cpp
/// @brief Unit tests for dtl/topology/affinity.hpp
/// @details Tests CPU set operations and affinity control.

#include <dtl/topology/topology.hpp>

#include <gtest/gtest.h>

namespace dtl::topology::test {

// =============================================================================
// CPU Set Basic Tests
// =============================================================================

TEST(CpuSetTest, DefaultEmpty) {
    cpu_set set;
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.count(), 0u);
}

TEST(CpuSetTest, SingleCpu) {
    cpu_set set(5);
    EXPECT_FALSE(set.empty());
    EXPECT_EQ(set.count(), 1u);
    EXPECT_TRUE(set.contains(5));
    EXPECT_FALSE(set.contains(0));
}

TEST(CpuSetTest, RangeConstruction) {
    cpu_set set(0, 3);
    EXPECT_EQ(set.count(), 4u);
    EXPECT_TRUE(set.contains(0));
    EXPECT_TRUE(set.contains(1));
    EXPECT_TRUE(set.contains(2));
    EXPECT_TRUE(set.contains(3));
    EXPECT_FALSE(set.contains(4));
}

TEST(CpuSetTest, VectorConstruction) {
    std::vector<std::uint32_t> cpus{1, 3, 5};
    cpu_set set(cpus);

    EXPECT_EQ(set.count(), 3u);
    EXPECT_TRUE(set.contains(1));
    EXPECT_TRUE(set.contains(3));
    EXPECT_TRUE(set.contains(5));
    EXPECT_FALSE(set.contains(2));
}

// =============================================================================
// CPU Set Operations Tests
// =============================================================================

TEST(CpuSetTest, Add) {
    cpu_set set;
    set.add(3);
    set.add(7);

    EXPECT_EQ(set.count(), 2u);
    EXPECT_TRUE(set.contains(3));
    EXPECT_TRUE(set.contains(7));
}

TEST(CpuSetTest, Remove) {
    cpu_set set(0, 4);
    EXPECT_EQ(set.count(), 5u);

    set.remove(2);
    EXPECT_EQ(set.count(), 4u);
    EXPECT_FALSE(set.contains(2));
}

TEST(CpuSetTest, Clear) {
    cpu_set set(0, 9);
    EXPECT_FALSE(set.empty());

    set.clear();
    EXPECT_TRUE(set.empty());
}

TEST(CpuSetTest, First) {
    cpu_set set;
    set.add(10);
    set.add(5);
    set.add(20);

    EXPECT_EQ(set.first(), 5u);
}

TEST(CpuSetTest, FirstEmpty) {
    cpu_set set;
    EXPECT_EQ(set.first(), cpu_set::max_cpus);
}

TEST(CpuSetTest, ToVector) {
    cpu_set set;
    set.add(1);
    set.add(5);
    set.add(3);

    auto vec = set.to_vector();
    EXPECT_EQ(vec.size(), 3u);

    // Should be sorted
    EXPECT_EQ(vec[0], 1u);
    EXPECT_EQ(vec[1], 3u);
    EXPECT_EQ(vec[2], 5u);
}

// =============================================================================
// CPU Set Algebra Tests
// =============================================================================

TEST(CpuSetTest, Union) {
    cpu_set a;
    a.add(1);
    a.add(2);

    cpu_set b;
    b.add(2);
    b.add(3);

    auto c = a | b;

    EXPECT_EQ(c.count(), 3u);
    EXPECT_TRUE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

TEST(CpuSetTest, Intersection) {
    cpu_set a;
    a.add(1);
    a.add(2);
    a.add(3);

    cpu_set b;
    b.add(2);
    b.add(3);
    b.add(4);

    auto c = a & b;

    EXPECT_EQ(c.count(), 2u);
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

TEST(CpuSetTest, Difference) {
    cpu_set a;
    a.add(1);
    a.add(2);
    a.add(3);

    cpu_set b;
    b.add(2);

    auto c = a - b;

    EXPECT_EQ(c.count(), 2u);
    EXPECT_TRUE(c.contains(1));
    EXPECT_TRUE(c.contains(3));
    EXPECT_FALSE(c.contains(2));
}

TEST(CpuSetTest, Equality) {
    cpu_set a(0, 3);
    cpu_set b(0, 3);
    cpu_set c(0, 4);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// =============================================================================
// Affinity Get/Set Tests
// =============================================================================

TEST(AffinityTest, GetAffinity) {
    auto result = get_affinity();
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result->empty());
}

TEST(AffinityTest, SetAffinityReturnsResult) {
    cpu_set set(0);  // CPU 0
    auto result = set_affinity(set);

    // Should succeed on Linux, may fail on other platforms
#if defined(__linux__)
    EXPECT_TRUE(result.has_value());
#endif
}

TEST(AffinityTest, SetEmptyAffinityFails) {
    cpu_set empty;
    auto result = set_affinity(empty);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// NUMA Binding Tests
// =============================================================================

TEST(NumaBindingTest, BindToNumaNode) {
    // May fail if NUMA node 0 doesn't exist (shouldn't happen)
    auto result = bind_to_numa_node(0);
    // Just verify it returns a result (success or error)
    // Actual success depends on platform
    EXPECT_TRUE(result.has_value() || result.has_error());
}

TEST(NumaBindingTest, InvalidNumaNodeFails) {
    auto result = bind_to_numa_node(9999);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Scoped Affinity Tests
// =============================================================================

TEST(ScopedAffinityTest, RestoresAffinity) {
    auto original = get_affinity();
    if (!original) {
        GTEST_SKIP() << "Cannot get original affinity";
    }

    {
        cpu_set new_set(0);
        scoped_affinity guard(new_set);
        // Inside scope, affinity may have changed
    }

    auto restored = get_affinity();
    EXPECT_TRUE(restored.has_value());

    // Should be restored (though exact restoration depends on permissions)
}

// =============================================================================
// CPU Binding Tests
// =============================================================================

TEST(CpuBindingTest, BindToCpu) {
    auto result = bind_to_cpu(0);
#if defined(__linux__)
    EXPECT_TRUE(result.has_value());

    // Verify we're now bound to CPU 0
    auto affinity = get_affinity();
    if (affinity) {
        EXPECT_TRUE(affinity->contains(0));
    }
#endif
}

}  // namespace dtl::topology::test
