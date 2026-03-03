// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_only_semantics.cpp
/// @brief Unit tests for device_only placement policy semantics
/// @details Verifies that device_only<N> correctly enforces device selection.

#include <dtl/core/config.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/memory/default_allocator.hpp>

#include <gtest/gtest.h>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// Policy Type Properties
// =============================================================================

TEST(DeviceOnlyPolicyTest, PolicyCategoryTag) {
    using policy_t = device_only<0>;

    static_assert(std::is_same_v<typename policy_t::policy_category, placement_policy_tag>);
    SUCCEED();
}

TEST(DeviceOnlyPolicyTest, StaticDeviceId) {
    static_assert(device_only<0>::device_id() == 0);
    static_assert(device_only<1>::device_id() == 1);
    static_assert(device_only<7>::device_id() == 7);
    SUCCEED();
}

TEST(DeviceOnlyPolicyTest, StaticDeviceConstant) {
    static_assert(device_only<0>::device == 0);
    static_assert(device_only<1>::device == 1);
    static_assert(device_only<3>::device == 3);
    SUCCEED();
}

TEST(DeviceOnlyPolicyTest, MemoryLocation) {
    EXPECT_EQ(device_only<0>::preferred_location(), memory_location::device);
    EXPECT_FALSE(device_only<0>::is_host_accessible());
    EXPECT_TRUE(device_only<0>::is_device_accessible());
    EXPECT_TRUE(device_only<0>::requires_host_copy());
    EXPECT_FALSE(device_only<0>::requires_device_copy());
}

TEST(DeviceOnlyPolicyTest, IsCompileTimeDevice) {
    static_assert(device_only<0>::is_compile_time_device());
    static_assert(device_only<1>::is_compile_time_device());
    SUCCEED();
}

// =============================================================================
// Different Device IDs Are Different Types
// =============================================================================

TEST(DeviceOnlyPolicyTest, DifferentDeviceIdsDifferentTypes) {
    static_assert(!std::is_same_v<device_only<0>, device_only<1>>,
                  "device_only<0> and device_only<1> must be different types");
    static_assert(!std::is_same_v<device_only<0>, device_only<2>>,
                  "device_only<0> and device_only<2> must be different types");
    static_assert(!std::is_same_v<device_only<1>, device_only<2>>,
                  "device_only<1> and device_only<2> must be different types");
    SUCCEED();
}

TEST(DeviceOnlyPolicyTest, SameDeviceIdSameType) {
    static_assert(std::is_same_v<device_only<0>, device_only<0>>,
                  "device_only<0> should be same as device_only<0>");
    static_assert(std::is_same_v<device_only<1>, device_only<1>>,
                  "device_only<1> should be same as device_only<1>");
    SUCCEED();
}

// =============================================================================
// Allocator Selection Produces Different Types
// =============================================================================

#if DTL_ENABLE_CUDA

TEST(DeviceOnlyAllocatorTest, DifferentDeviceIdsDifferentAllocators) {
    using alloc0_t = select_allocator_t<float, device_only<0>>;
    using alloc1_t = select_allocator_t<float, device_only<1>>;
    using alloc2_t = select_allocator_t<float, device_only<2>>;

    static_assert(!std::is_same_v<alloc0_t, alloc1_t>,
                  "device_only<0> and device_only<1> must select different allocators");
    static_assert(!std::is_same_v<alloc0_t, alloc2_t>,
                  "device_only<0> and device_only<2> must select different allocators");
    static_assert(!std::is_same_v<alloc1_t, alloc2_t>,
                  "device_only<1> and device_only<2> must select different allocators");
    SUCCEED();
}

TEST(DeviceOnlyAllocatorTest, SameDeviceIdSameAllocator) {
    using alloc0a_t = select_allocator_t<float, device_only<0>>;
    using alloc0b_t = select_allocator_t<float, device_only<0>>;

    static_assert(std::is_same_v<alloc0a_t, alloc0b_t>,
                  "Same device_only<N> should produce same allocator type");
    SUCCEED();
}

TEST(DeviceOnlyAllocatorTest, DifferentElementTypesSameDevice) {
    // Same device, different element types should produce different allocator instances
    // but the underlying memory space should be the same template instantiation
    using alloc_float_t = select_allocator_t<float, device_only<0>>;
    using alloc_double_t = select_allocator_t<double, device_only<0>>;

    // Different value types
    static_assert(!std::is_same_v<alloc_float_t, alloc_double_t>);

    // But rebind should work
    using rebound_t = typename alloc_float_t::template rebind<double>::other;
    static_assert(std::is_same_v<rebound_t, alloc_double_t>);
    SUCCEED();
}

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// Default Device Alias
// =============================================================================

TEST(DeviceOnlyPolicyTest, DefaultDeviceAlias) {
    static_assert(std::is_same_v<device_only_default, device_only<0>>,
                  "device_only_default should be device_only<0>");
    SUCCEED();
}

// =============================================================================
// Default Template Parameter
// =============================================================================

TEST(DeviceOnlyPolicyTest, DefaultTemplateParameterIsZero) {
    // device_only<> should default to device 0
    static_assert(device_only<>::device_id() == 0);
    static_assert(std::is_same_v<device_only<>, device_only<0>>);
    SUCCEED();
}

}  // namespace dtl::test
