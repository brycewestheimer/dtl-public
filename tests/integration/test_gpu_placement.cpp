// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_gpu_placement.cpp
/// @brief GPU placement policy integration tests
/// @details See test_placement_policies.cpp for comprehensive placement policy tests.
///          This file provides basic sanity checks for GPU placement.

#include <dtl/core/config.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/unified_memory.hpp>
#include <dtl/policies/placement/device_preferred.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Placement Policy Compile-Time Properties
// =============================================================================

TEST(GpuPlacementTest, DeviceOnlyProperties) {
    using policy = device_only<0>;

    EXPECT_EQ(policy::device, 0);
    EXPECT_FALSE(policy::is_host_accessible());
    EXPECT_TRUE(policy::is_device_accessible());
    EXPECT_TRUE(policy::requires_host_copy());
    EXPECT_FALSE(policy::requires_device_copy());
}

TEST(GpuPlacementTest, DeviceOnly1Properties) {
    using policy = device_only<1>;

    EXPECT_EQ(policy::device, 1);
    EXPECT_FALSE(policy::is_host_accessible());
    EXPECT_TRUE(policy::is_device_accessible());
}

TEST(GpuPlacementTest, UnifiedMemoryProperties) {
    EXPECT_TRUE(unified_memory::is_host_accessible());
    EXPECT_TRUE(unified_memory::is_device_accessible());
    EXPECT_FALSE(unified_memory::requires_explicit_copy());
    EXPECT_TRUE(unified_memory::supports_prefetch());
}

TEST(GpuPlacementTest, DevicePreferredProperties) {
    EXPECT_FALSE(device_preferred::is_host_accessible());
    EXPECT_TRUE(device_preferred::is_device_accessible());
    EXPECT_TRUE(device_preferred::allows_fallback());
}

}  // namespace dtl::test
