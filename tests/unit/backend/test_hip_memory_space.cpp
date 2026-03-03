// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hip_memory_space.cpp
/// @brief Unit tests for HIP memory space implementations
/// @details Tests API structure and behavior. The hip_memory_space class
///          uses concept-based static dispatch (no virtual) matching CUDA's
///          pattern with std::atomic allocation tracking. When HIP is not
///          available, placeholder tests document the expected API.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_HIP
#include <backends/hip/hip_memory_space.hpp>
#endif

#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <string_view>

namespace dtl::test {

// =============================================================================
// Tests available with HIP backend
// =============================================================================

#if DTL_ENABLE_HIP

TEST(HipMemorySpaceTest, MemorySpaceName) {
    EXPECT_STREQ(dtl::hip::hip_memory_space::name(), "hip_device");
}

TEST(HipMemorySpaceTest, ManagedSpaceName) {
    EXPECT_STREQ(dtl::hip::hip_managed_memory_space::name(), "hip_managed");
}

TEST(HipMemorySpaceTest, HostAccessible) {
    EXPECT_FALSE(dtl::hip::hip_memory_space::host_accessible);
}

TEST(HipMemorySpaceTest, DeviceAccessible) {
    EXPECT_TRUE(dtl::hip::hip_memory_space::device_accessible);
}

TEST(HipMemorySpaceTest, ManagedHostAccessible) {
    EXPECT_TRUE(dtl::hip::hip_managed_memory_space::host_accessible);
}

TEST(HipMemorySpaceTest, ManagedDeviceAccessible) {
    EXPECT_TRUE(dtl::hip::hip_managed_memory_space::device_accessible);
}

TEST(HipMemorySpaceTest, AllocateDevice) {
    dtl::hip::hip_memory_space space;
    void* ptr = space.allocate(1024);
    if (ptr) {
        space.deallocate(ptr, 1024);
    }
    SUCCEED();
}

TEST(HipMemorySpaceTest, DeallocateNullptr) {
    dtl::hip::hip_memory_space space;
    space.deallocate(nullptr, 0);
    SUCCEED();
}

TEST(HipMemorySpaceTest, ContainsHostPointer) {
    dtl::hip::hip_memory_space space;
    int x = 42;
    EXPECT_FALSE(space.contains(&x));
}

TEST(HipMemorySpaceTest, DefaultDevice) {
    auto dev = dtl::hip::current_device();
    SUCCEED();
}

TEST(HipMemorySpaceTest, DeviceCount) {
    int count = dtl::hip::device_count();
    EXPECT_GE(count, 0);
}

TEST(HipMemorySpaceTest, AvailableMemory) {
    dtl::hip::hip_memory_space space;
    size_type avail = space.available_memory();
    EXPECT_GE(avail, 0u);
}

TEST(HipMemorySpaceTest, MemsetDevice) {
    dtl::hip::hip_memory_space space;
    void* ptr = space.allocate(256);
    if (ptr) {
        space.memset(ptr, 0, 256);
        space.deallocate(ptr, 256);
    }
    SUCCEED();
}

TEST(HipMemorySpaceTest, AllocationTracking) {
    dtl::hip::hip_memory_space space;
    EXPECT_EQ(space.total_allocated(), 0u);
    void* ptr = space.allocate(1024);
    if (ptr) {
        EXPECT_EQ(space.total_allocated(), 1024u);
        EXPECT_GE(space.peak_allocated(), 1024u);
        space.deallocate(ptr, 1024);
        EXPECT_EQ(space.total_allocated(), 0u);
    }
}

TEST(HipMemorySpaceTest, Properties) {
    dtl::hip::hip_memory_space space;
    auto props = space.properties();
    EXPECT_FALSE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_EQ(props.alignment, 256u);
}

TEST(HipMemorySpaceTest, ManagedProperties) {
    dtl::hip::hip_managed_memory_space space;
    auto props = space.properties();
    EXPECT_TRUE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_TRUE(props.unified);
}

TEST(HipMemorySpaceTest, DefaultFactory) {
    auto& space = dtl::hip::default_hip_memory_space();
    EXPECT_STREQ(decltype(space)::name(), "hip_device");
}

TEST(HipMemorySpaceTest, ManagedFactory) {
    auto& space = dtl::hip::default_hip_managed_memory_space();
    EXPECT_STREQ(decltype(space)::name(), "hip_managed");
}

#else

// =============================================================================
// Placeholder tests when HIP backend is not available
// =============================================================================

TEST(HipMemorySpaceTest, MemorySpaceNamePlaceholder) {
    // hip_memory_space::name() == "hip_device"
    SUCCEED();
}

TEST(HipMemorySpaceTest, ManagedSpaceNamePlaceholder) {
    // hip_managed_memory_space::name() == "hip_managed"
    SUCCEED();
}

TEST(HipMemorySpaceTest, HostAccessiblePlaceholder) {
    // hip_memory_space::host_accessible == false
    SUCCEED();
}

TEST(HipMemorySpaceTest, DeviceAccessiblePlaceholder) {
    // hip_memory_space::device_accessible == true
    SUCCEED();
}

TEST(HipMemorySpaceTest, ManagedHostAccessiblePlaceholder) {
    // hip_managed_memory_space::host_accessible == true
    SUCCEED();
}

TEST(HipMemorySpaceTest, ManagedDeviceAccessiblePlaceholder) {
    // hip_managed_memory_space::device_accessible == true
    SUCCEED();
}

TEST(HipMemorySpaceTest, AllocateWithoutHIPPlaceholder) {
    // allocate returns nullptr without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, DeallocateWithoutHIPPlaceholder) {
    // deallocate is noexcept no-op without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, ContainsWithoutHIPPlaceholder) {
    // contains returns false without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, DefaultDevicePlaceholder) {
    // current_device returns invalid_device without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, DeviceCountPlaceholder) {
    // device_count returns 0 without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, AvailableMemoryPlaceholder) {
    // available_memory returns 0 without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, MemsetWithoutHIPPlaceholder) {
    // memset is noexcept no-op without HIP
    SUCCEED();
}

TEST(HipMemorySpaceTest, AllocationTrackingPlaceholder) {
    // total_allocated/peak_allocated use std::atomic
    SUCCEED();
}

TEST(HipMemorySpaceTest, PropertiesPlaceholder) {
    // properties() returns memory_space_properties struct
    SUCCEED();
}

TEST(HipMemorySpaceTest, ManagedPropertiesPlaceholder) {
    // managed space has host_accessible=true, unified=true
    SUCCEED();
}

TEST(HipMemorySpaceTest, DefaultFactoryPlaceholder) {
    // default_hip_memory_space works
    SUCCEED();
}

TEST(HipMemorySpaceTest, ManagedFactoryPlaceholder) {
    // default_hip_managed_memory_space works
    SUCCEED();
}

#endif  // DTL_ENABLE_HIP

}  // namespace dtl::test
