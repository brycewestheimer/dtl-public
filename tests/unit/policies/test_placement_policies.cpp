// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_placement_policies.cpp
/// @brief Unit tests for placement policies
/// @details Tests host_only, device_only, unified_memory, and device_preferred.

#include <dtl/policies/placement/placement_policy.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/unified_memory.hpp>
#include <dtl/policies/placement/device_preferred.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Host Only Tests
// =============================================================================

TEST(HostOnlyTest, ConceptSatisfaction) {
    static_assert(PlacementPolicyConcept<host_only>);
}

TEST(HostOnlyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename host_only::policy_category, placement_policy_tag>);
}

TEST(HostOnlyTest, PreferredLocation) {
    EXPECT_EQ(host_only::preferred_location(), memory_location::host);
}

TEST(HostOnlyTest, AccessibilityFlags) {
    EXPECT_TRUE(host_only::is_host_accessible());
    EXPECT_FALSE(host_only::is_device_accessible());
    EXPECT_TRUE(host_only::requires_device_copy());
}

TEST(HostOnlyTest, DeviceId) {
    EXPECT_EQ(host_only::device_id(), -1);
}

// =============================================================================
// Device Only Tests
// =============================================================================

TEST(DeviceOnlyTest, ConceptSatisfaction) {
    static_assert(PlacementPolicyConcept<device_only<0>>);
    static_assert(PlacementPolicyConcept<device_only<1>>);
}

TEST(DeviceOnlyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename device_only<0>::policy_category, placement_policy_tag>);
}

TEST(DeviceOnlyTest, PreferredLocation) {
    EXPECT_EQ(device_only<0>::preferred_location(), memory_location::device);
}

TEST(DeviceOnlyTest, AccessibilityFlags) {
    EXPECT_FALSE(device_only<0>::is_host_accessible());
    EXPECT_TRUE(device_only<0>::is_device_accessible());
    EXPECT_TRUE(device_only<0>::requires_host_copy());
}

TEST(DeviceOnlyTest, DeviceId) {
    EXPECT_EQ(device_only<0>::device_id(), 0);
    EXPECT_EQ(device_only<1>::device_id(), 1);
    EXPECT_EQ(device_only<7>::device_id(), 7);
}

// =============================================================================
// Unified Memory Tests
// =============================================================================

TEST(UnifiedMemoryTest, ConceptSatisfaction) {
    static_assert(PlacementPolicyConcept<unified_memory>);
}

TEST(UnifiedMemoryTest, PolicyCategory) {
    static_assert(std::is_same_v<typename unified_memory::policy_category, placement_policy_tag>);
}

TEST(UnifiedMemoryTest, PreferredLocation) {
    EXPECT_EQ(unified_memory::preferred_location(), memory_location::unified);
}

TEST(UnifiedMemoryTest, AccessibilityFlags) {
    // Unified memory is accessible from both host and device
    EXPECT_TRUE(unified_memory::is_host_accessible());
    EXPECT_TRUE(unified_memory::is_device_accessible());
    // No explicit copy required (but may have page migration)
    EXPECT_FALSE(unified_memory::requires_host_copy());
    EXPECT_FALSE(unified_memory::requires_device_copy());
}

// =============================================================================
// Device Preferred Tests
// =============================================================================

TEST(DevicePreferredTest, ConceptSatisfaction) {
    static_assert(PlacementPolicyConcept<device_preferred>);
}

TEST(DevicePreferredTest, PolicyCategory) {
    static_assert(std::is_same_v<typename device_preferred::policy_category, placement_policy_tag>);
}

TEST(DevicePreferredTest, PreferredLocation) {
    EXPECT_EQ(device_preferred::preferred_location(), memory_location::device);
}

TEST(DevicePreferredTest, AccessibilityFlags) {
    // Device preferred means we prefer device but can fallback to host
    EXPECT_TRUE(device_preferred::is_device_accessible());
    // Whether host accessible depends on runtime fallback
}

// =============================================================================
// Memory Location Enum Tests
// =============================================================================

TEST(MemoryLocationTest, EnumValues) {
    // Verify all enum values are distinct
    EXPECT_NE(memory_location::host, memory_location::device);
    EXPECT_NE(memory_location::host, memory_location::unified);
    EXPECT_NE(memory_location::host, memory_location::remote);
    EXPECT_NE(memory_location::device, memory_location::unified);
    EXPECT_NE(memory_location::device, memory_location::remote);
    EXPECT_NE(memory_location::unified, memory_location::remote);
}

// =============================================================================
// Default Policy Tests
// =============================================================================

TEST(DefaultPlacementTest, IsHostOnly) {
    static_assert(std::is_same_v<default_placement_policy, host_only>);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(PlacementConstexprTest, HostOnlyCompileTime) {
    constexpr auto loc = host_only::preferred_location();
    constexpr bool host_acc = host_only::is_host_accessible();
    constexpr bool dev_acc = host_only::is_device_accessible();

    static_assert(loc == memory_location::host);
    static_assert(host_acc == true);
    static_assert(dev_acc == false);
}

TEST(PlacementConstexprTest, DeviceOnlyCompileTime) {
    constexpr auto loc = device_only<2>::preferred_location();
    constexpr int dev_id = device_only<2>::device_id();

    static_assert(loc == memory_location::device);
    static_assert(dev_id == 2);
}

}  // namespace dtl::test
