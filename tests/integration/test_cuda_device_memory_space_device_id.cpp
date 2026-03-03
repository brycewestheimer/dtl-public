// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_memory_space_device_id.cpp
/// @brief Integration tests for device-specific CUDA memory spaces
/// @details Verifies that allocations on different devices land on the correct GPU.

#include <dtl/core/config.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/policies/placement/device_only.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <type_traits>

namespace dtl::cuda::test {

// =============================================================================
// Compile-Time Allocator Selection Tests
// =============================================================================

#if DTL_ENABLE_CUDA

TEST(CudaDeviceMemorySpaceTest, DifferentDeviceIdsDifferentAllocatorTypes) {
    // Verify that device_only<0> and device_only<1> produce different allocator types
    using alloc0_t = select_allocator_t<float, device_only<0>>;
    using alloc1_t = select_allocator_t<float, device_only<1>>;

    static_assert(!std::is_same_v<alloc0_t, alloc1_t>,
                  "device_only<0> and device_only<1> must select different allocator types");
    SUCCEED();
}

TEST(CudaDeviceMemorySpaceTest, AllocatorUsesCorrectMemorySpace) {
    using alloc_t = select_allocator_t<float, device_only<0>>;
    using space_t = cuda_device_memory_space_for<0>;

    // Verify the allocator uses the device-specific memory space
    // (memory_space_allocator<T, cuda_device_memory_space_for<0>>)
    using expected_t = memory_space_allocator<float, space_t>;
    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "Allocator should use cuda_device_memory_space_for<0>");
    SUCCEED();
}

// =============================================================================
// Memory Space Static Properties
// =============================================================================

TEST(CudaDeviceMemorySpaceTest, StaticProperties) {
    using space0_t = cuda_device_memory_space_for<0>;
    using space1_t = cuda_device_memory_space_for<1>;

    // Compile-time device ID
    static_assert(space0_t::device_id == 0);
    static_assert(space1_t::device_id == 1);

    // Memory properties
    EXPECT_FALSE(space0_t::host_accessible);
    EXPECT_TRUE(space0_t::device_accessible);

    auto props = space0_t::properties();
    EXPECT_FALSE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_EQ(props.alignment, 256);
}

// =============================================================================
// Single Device Allocation Tests
// =============================================================================

TEST(CudaDeviceMemorySpaceTest, AllocateOnDevice0) {
    if (device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    using space_t = cuda_device_memory_space_for<0>;
    constexpr size_t size = 1024;

    void* ptr = space_t::allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Verify it's device memory
    EXPECT_TRUE(is_device_pointer(ptr));

    // Verify it's on device 0
    int device = get_pointer_device(ptr);
    EXPECT_EQ(device, 0);

    space_t::deallocate(ptr, size);
}

// =============================================================================
// Multi-Device Allocation Tests
// =============================================================================

TEST(CudaDeviceMemorySpaceTest, AllocateOnDifferentDevices) {
    if (device_count() < 2) {
        GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
    }

    using space0_t = cuda_device_memory_space_for<0>;
    using space1_t = cuda_device_memory_space_for<1>;
    constexpr size_t size = 1024;

    // Record current device
    int original_device = current_device_id();

    // Allocate on device 0
    void* ptr0 = space0_t::allocate(size);
    ASSERT_NE(ptr0, nullptr);

    // Allocate on device 1
    void* ptr1 = space1_t::allocate(size);
    ASSERT_NE(ptr1, nullptr);

    // Verify allocations are on correct devices
    EXPECT_EQ(get_pointer_device(ptr0), 0);
    EXPECT_EQ(get_pointer_device(ptr1), 1);

    // Verify current device was restored after each allocation
    EXPECT_EQ(current_device_id(), original_device);

    // Clean up
    space0_t::deallocate(ptr0, size);
    space1_t::deallocate(ptr1, size);

    // Verify current device is still restored
    EXPECT_EQ(current_device_id(), original_device);
}

TEST(CudaDeviceMemorySpaceTest, InterleavedAllocationsRestoreDevice) {
    if (device_count() < 2) {
        GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
    }

    using space0_t = cuda_device_memory_space_for<0>;
    using space1_t = cuda_device_memory_space_for<1>;
    constexpr size_t size = 512;

    // Start on device 1
    cudaSetDevice(1);
    EXPECT_EQ(current_device_id(), 1);

    // Multiple interleaved allocations
    for (int i = 0; i < 5; ++i) {
        void* ptr0 = space0_t::allocate(size);
        ASSERT_NE(ptr0, nullptr);
        EXPECT_EQ(current_device_id(), 1);  // Should be restored

        void* ptr1 = space1_t::allocate(size);
        ASSERT_NE(ptr1, nullptr);
        EXPECT_EQ(current_device_id(), 1);  // Should be restored

        EXPECT_EQ(get_pointer_device(ptr0), 0);
        EXPECT_EQ(get_pointer_device(ptr1), 1);

        space0_t::deallocate(ptr0, size);
        EXPECT_EQ(current_device_id(), 1);  // Should be restored

        space1_t::deallocate(ptr1, size);
        EXPECT_EQ(current_device_id(), 1);  // Should be restored
    }
}

// =============================================================================
// Runtime Memory Space Tests
// =============================================================================

TEST(CudaDeviceMemorySpaceRuntimeTest, AllocateOnSpecifiedDevice) {
    if (device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    cuda_device_memory_space_runtime space(0);
    constexpr size_t size = 1024;

    void* ptr = space.allocate(size);
    ASSERT_NE(ptr, nullptr);

    EXPECT_TRUE(is_device_pointer(ptr));
    EXPECT_EQ(get_pointer_device(ptr), 0);
    EXPECT_EQ(space.get_device_id(), 0);

    space.deallocate(ptr, size);
}

TEST(CudaDeviceMemorySpaceRuntimeTest, DifferentInstancesDifferentDevices) {
    if (device_count() < 2) {
        GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
    }

    cuda_device_memory_space_runtime space0(0);
    cuda_device_memory_space_runtime space1(1);
    constexpr size_t size = 1024;

    EXPECT_NE(space0, space1);  // Different devices

    void* ptr0 = space0.allocate(size);
    void* ptr1 = space1.allocate(size);

    ASSERT_NE(ptr0, nullptr);
    ASSERT_NE(ptr1, nullptr);

    EXPECT_EQ(get_pointer_device(ptr0), 0);
    EXPECT_EQ(get_pointer_device(ptr1), 1);

    space0.deallocate(ptr0, size);
    space1.deallocate(ptr1, size);
}

// =============================================================================
// Memory Info Tests
// =============================================================================

TEST(CudaDeviceMemorySpaceTest, MemoryInfo) {
    if (device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    using space_t = cuda_device_memory_space_for<0>;

    size_type free_bytes = 0;
    size_type total_bytes = 0;

    bool success = space_t::memory_info(free_bytes, total_bytes);
    EXPECT_TRUE(success);
    EXPECT_GT(total_bytes, 0);
    EXPECT_GT(free_bytes, 0);
    EXPECT_LE(free_bytes, total_bytes);
}

#else  // !DTL_ENABLE_CUDA

TEST(CudaDeviceMemorySpaceTest, AllocateReturnsNullWhenCudaDisabled) {
    using space_t = cuda_device_memory_space_for<0>;

    void* ptr = space_t::allocate(1024);
    EXPECT_EQ(ptr, nullptr);
}

TEST(CudaDeviceMemorySpaceTest, PointerQueriesReturnDefaultsWhenCudaDisabled) {
    EXPECT_EQ(get_pointer_device(nullptr), invalid_device_id);
    EXPECT_FALSE(is_device_pointer(nullptr));
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::cuda::test
