// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_select_allocator_device_only_device_id.cpp
/// @brief Unit tests for allocator selection with device_only<DeviceId>
/// @details Verifies that different device IDs produce different allocator types.

#include <dtl/core/config.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/unified_memory.hpp>
#include <dtl/policies/placement/device_preferred.hpp>

#include <gtest/gtest.h>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// Host Allocator Selection
// =============================================================================

TEST(SelectAllocatorTest, HostOnlySelectsHostSpace) {
    using alloc_t = select_allocator_t<float, host_only>;
    using expected_t = memory_space_allocator<float, host_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

// =============================================================================
// Device Allocator Selection (CUDA)
// =============================================================================

#if DTL_ENABLE_CUDA

TEST(SelectAllocatorTest, DeviceOnly0SelectsDeviceSpace0) {
    using alloc_t = select_allocator_t<float, device_only<0>>;

    // Should use cuda_device_memory_space_for<0>
    using expected_space_t = cuda::cuda_device_memory_space_for<0>;
    using expected_t = memory_space_allocator<float, expected_space_t>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

TEST(SelectAllocatorTest, DeviceOnly1SelectsDeviceSpace1) {
    using alloc_t = select_allocator_t<float, device_only<1>>;

    using expected_space_t = cuda::cuda_device_memory_space_for<1>;
    using expected_t = memory_space_allocator<float, expected_space_t>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

TEST(SelectAllocatorTest, DifferentDevicesDifferentSpaces) {
    using alloc0_t = select_allocator_t<double, device_only<0>>;
    using alloc1_t = select_allocator_t<double, device_only<1>>;
    using alloc2_t = select_allocator_t<double, device_only<2>>;
    using alloc7_t = select_allocator_t<double, device_only<7>>;

    // All must be different types
    static_assert(!std::is_same_v<alloc0_t, alloc1_t>);
    static_assert(!std::is_same_v<alloc0_t, alloc2_t>);
    static_assert(!std::is_same_v<alloc0_t, alloc7_t>);
    static_assert(!std::is_same_v<alloc1_t, alloc2_t>);
    static_assert(!std::is_same_v<alloc1_t, alloc7_t>);
    static_assert(!std::is_same_v<alloc2_t, alloc7_t>);

    SUCCEED();
}

TEST(SelectAllocatorTest, UnifiedMemorySelectsUnifiedSpace) {
    using alloc_t = select_allocator_t<float, unified_memory>;
    using expected_t = memory_space_allocator<float, cuda::cuda_unified_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

TEST(SelectAllocatorTest, DevicePreferredSelectsDeviceSpace) {
    using alloc_t = select_allocator_t<float, device_preferred>;
    using expected_t = memory_space_allocator<float, cuda::cuda_device_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

#else  // !DTL_ENABLE_CUDA

TEST(SelectAllocatorTest, DevicePreferredFallsBackToHost) {
    using alloc_t = select_allocator_t<float, device_preferred>;
    using expected_t = memory_space_allocator<float, host_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>);
    SUCCEED();
}

// Note: device_only<N> and unified_memory would static_assert when instantiated
// without CUDA, so we don't test them here.

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// Allocator Properties
// =============================================================================

TEST(SelectAllocatorTest, HostAllocatorIsHostAllocator) {
    EXPECT_TRUE(is_host_allocator<default_allocator<float>>());
    EXPECT_FALSE(is_device_allocator<default_allocator<float>>());
}

#if DTL_ENABLE_CUDA

TEST(SelectAllocatorTest, DeviceAllocatorIsDeviceAllocator) {
    using alloc_t = select_allocator_t<float, device_only<0>>;

    EXPECT_FALSE(is_host_allocator<alloc_t>());
    EXPECT_TRUE(is_device_allocator<alloc_t>());
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::test
