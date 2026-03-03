// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_runtime_device_selection.cpp
/// @brief Integration tests for runtime device selection
/// @details Tests the device_only_runtime placement policy and context-based
///          device selection for containers.
/// @since 0.1.0

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <dtl/dtl.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/core/runtime_device_context.hpp>
#include <dtl/cuda/device_guard.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

namespace {

// Test fixture for runtime device selection
class RuntimeDeviceSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_count_ = dtl::cuda::device_count();
        original_device_ = dtl::cuda::current_device_id();
    }

    void TearDown() override {
        // Restore original device
        if (device_count_ > 0 && original_device_ >= 0) {
            cudaSetDevice(original_device_);
        }
    }

    int device_count_{0};
    int original_device_{-1};
};

// Skip helper
#define SKIP_IF_NO_CUDA() \
    if (device_count_ == 0) { \
        GTEST_SKIP() << "No CUDA devices available"; \
    }

#define SKIP_IF_SINGLE_GPU() \
    if (device_count_ < 2) { \
        GTEST_SKIP() << "Test requires at least 2 GPUs"; \
    }

// ============================================================================
// device_only_runtime Policy Tests
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, PolicyTraits) {
    // Verify policy traits
    EXPECT_EQ(dtl::device_only_runtime::preferred_location(), dtl::memory_location::device);
    EXPECT_FALSE(dtl::device_only_runtime::is_host_accessible());
    EXPECT_TRUE(dtl::device_only_runtime::is_device_accessible());
    EXPECT_TRUE(dtl::device_only_runtime::requires_host_copy());
    EXPECT_FALSE(dtl::device_only_runtime::requires_device_copy());
    EXPECT_FALSE(dtl::device_only_runtime::is_compile_time_device());
    EXPECT_TRUE(dtl::device_only_runtime::is_runtime_device());
}

TEST_F(RuntimeDeviceSelectionTest, PolicyTypeTraits) {
    // Type traits
    EXPECT_TRUE(dtl::is_runtime_device_policy_v<dtl::device_only_runtime>);
    EXPECT_FALSE(dtl::is_compile_time_device_policy_v<dtl::device_only_runtime>);
    EXPECT_TRUE(dtl::is_device_placement_policy_v<dtl::device_only_runtime>);
    
    // Compile-time policy comparison
    EXPECT_FALSE(dtl::is_runtime_device_policy_v<dtl::device_only<0>>);
    EXPECT_TRUE(dtl::is_compile_time_device_policy_v<dtl::device_only<0>>);
    EXPECT_TRUE(dtl::is_device_placement_policy_v<dtl::device_only<0>>);
}

// ============================================================================
// Context Device Extraction Tests
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, ContextDeviceIdExtraction) {
    SKIP_IF_NO_CUDA();
    
    auto base_ctx = dtl::make_cpu_context();
    auto cuda_ctx = base_ctx.with_cuda(0);
    
    // Extract device ID from context
    auto device_id = dtl::detail::ctx_gpu_device_id(cuda_ctx);
    ASSERT_TRUE(device_id.has_value());
    EXPECT_EQ(*device_id, 0);
}

TEST_F(RuntimeDeviceSelectionTest, ContextWithDifferentDevices) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto base_ctx = dtl::make_cpu_context();
    
    auto ctx0 = base_ctx.with_cuda(0);
    auto ctx1 = base_ctx.with_cuda(1);
    
    auto id0 = dtl::detail::ctx_gpu_device_id(ctx0);
    auto id1 = dtl::detail::ctx_gpu_device_id(ctx1);
    
    ASSERT_TRUE(id0.has_value());
    ASSERT_TRUE(id1.has_value());
    EXPECT_EQ(*id0, 0);
    EXPECT_EQ(*id1, 1);
}

TEST_F(RuntimeDeviceSelectionTest, CpuOnlyContextHasNoGpuDomain) {
    auto cpu_ctx = dtl::make_cpu_context();
    
    auto device_id = dtl::detail::ctx_gpu_device_id(cpu_ctx);
    EXPECT_FALSE(device_id.has_value());
}

// ============================================================================
// Container with Runtime Device Selection
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, ContainerOnDevice0) {
    SKIP_IF_NO_CUDA();
    
    auto ctx = dtl::make_cpu_context().with_cuda(0);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
    
    EXPECT_EQ(vec.device_id(), 0);
    EXPECT_TRUE(vec.has_device_affinity());
    EXPECT_TRUE(vec.is_device_accessible());
    EXPECT_FALSE(vec.is_host_accessible());
}

TEST_F(RuntimeDeviceSelectionTest, ContainerOnDevice1) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto ctx = dtl::make_cpu_context().with_cuda(1);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
    
    EXPECT_EQ(vec.device_id(), 1);
    EXPECT_TRUE(vec.has_device_affinity());
}

TEST_F(RuntimeDeviceSelectionTest, ContainersOnDifferentDevices) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto base = dtl::make_cpu_context();
    auto ctx0 = base.with_cuda(0);
    auto ctx1 = base.with_cuda(1);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec0(100, ctx0);
    dtl::distributed_vector<float, dtl::device_only_runtime> vec1(100, ctx1);
    
    EXPECT_EQ(vec0.device_id(), 0);
    EXPECT_EQ(vec1.device_id(), 1);
    EXPECT_NE(vec0.device_id(), vec1.device_id());
}

// ============================================================================
// Device Context Preservation Tests
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, ContainerCreationPreservesDevice) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    // Set to device 1
    cudaSetDevice(1);
    int before = dtl::cuda::current_device_id();
    EXPECT_EQ(before, 1);
    
    // Create container on device 0
    auto ctx = dtl::make_cpu_context().with_cuda(0);
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
    
    // Device 1 should still be current
    int after = dtl::cuda::current_device_id();
    EXPECT_EQ(after, before);
}

TEST_F(RuntimeDeviceSelectionTest, MultipleContainersPreserveDevice) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    // Set to device 0
    cudaSetDevice(0);
    int original = dtl::cuda::current_device_id();
    
    auto base = dtl::make_cpu_context();
    
    // Create containers on different devices
    {
        auto ctx1 = base.with_cuda(1);
        dtl::distributed_vector<float, dtl::device_only_runtime> vec1(100, ctx1);
        EXPECT_EQ(vec1.device_id(), 1);
    }
    
    // Device should still be 0
    EXPECT_EQ(dtl::cuda::current_device_id(), original);
    
    {
        auto ctx0 = base.with_cuda(0);
        dtl::distributed_vector<float, dtl::device_only_runtime> vec0(100, ctx0);
        EXPECT_EQ(vec0.device_id(), 0);
    }
    
    // Still device 0
    EXPECT_EQ(dtl::cuda::current_device_id(), original);
}

// ============================================================================
// Memory Allocation Verification
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, AllocationOnCorrectDevice) {
    SKIP_IF_NO_CUDA();
    
    auto ctx = dtl::make_cpu_context().with_cuda(0);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
    
    // Query the pointer's device
    const void* ptr = vec.local_data();
    int ptr_device = dtl::cuda::get_pointer_device(ptr);
    
    EXPECT_EQ(ptr_device, 0);
}

TEST_F(RuntimeDeviceSelectionTest, AllocationOnSecondDevice) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto ctx = dtl::make_cpu_context().with_cuda(1);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
    
    const void* ptr = vec.local_data();
    int ptr_device = dtl::cuda::get_pointer_device(ptr);
    
    EXPECT_EQ(ptr_device, 1);
}

TEST_F(RuntimeDeviceSelectionTest, TwoVectorsOnDifferentDevicesHaveCorrectPointers) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto base = dtl::make_cpu_context();
    auto ctx0 = base.with_cuda(0);
    auto ctx1 = base.with_cuda(1);
    
    dtl::distributed_vector<float, dtl::device_only_runtime> vec0(100, ctx0);
    dtl::distributed_vector<float, dtl::device_only_runtime> vec1(100, ctx1);
    
    int ptr0_device = dtl::cuda::get_pointer_device(vec0.local_data());
    int ptr1_device = dtl::cuda::get_pointer_device(vec1.local_data());
    
    EXPECT_EQ(ptr0_device, 0);
    EXPECT_EQ(ptr1_device, 1);
}

// ============================================================================
// Runtime Memory Space Tests
// ============================================================================

TEST_F(RuntimeDeviceSelectionTest, RuntimeMemorySpaceAllocatesOnCorrectDevice) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    // Allocate using runtime memory space on device 1
    dtl::cuda::cuda_device_memory_space_runtime space(1);
    
    void* ptr = space.allocate(1024);
    ASSERT_NE(ptr, nullptr);
    
    int ptr_device = dtl::cuda::get_pointer_device(ptr);
    EXPECT_EQ(ptr_device, 1);
    
    // Current device should be unchanged
    int current = dtl::cuda::current_device_id();
    EXPECT_EQ(current, original_device_);
    
    space.deallocate(ptr, 1024);
}

}  // namespace

#else  // !DTL_ENABLE_CUDA

#include <gtest/gtest.h>

TEST(RuntimeDeviceSelectionTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled in this build";
}

#endif  // DTL_ENABLE_CUDA
