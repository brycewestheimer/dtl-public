// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_device_pool_domain.cpp
/// @brief Integration tests for cuda_device_pool_domain
/// @details Tests the multi-device domain for managing multiple GPUs
///          within a single context.
/// @since 0.1.0

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <dtl/core/cuda_device_pool_domain.hpp>
#include <dtl/cuda/device_guard.hpp>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

namespace {

class CudaDevicePoolDomainTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDeviceCount(&device_count_);
    }

    int device_count_{0};
};

#define SKIP_IF_NO_CUDA() \
    if (device_count_ == 0) { \
        GTEST_SKIP() << "No CUDA devices available"; \
    }

#define SKIP_IF_SINGLE_GPU() \
    if (device_count_ < 2) { \
        GTEST_SKIP() << "Test requires at least 2 GPUs"; \
    }

// ============================================================================
// Pool Creation Tests
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, CreateWithSingleDevice) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    auto& pool = *result;
    EXPECT_TRUE(pool.valid());
    EXPECT_EQ(pool.device_count(), 1u);
    EXPECT_TRUE(pool.contains(0));
}

TEST_F(CudaDevicePoolDomainTest, CreateWithMultipleDevices) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto result = dtl::cuda_device_pool_domain::create({0, 1});
    ASSERT_TRUE(result.has_value());
    
    auto& pool = *result;
    EXPECT_TRUE(pool.valid());
    EXPECT_EQ(pool.device_count(), 2u);
    EXPECT_TRUE(pool.contains(0));
    EXPECT_TRUE(pool.contains(1));
}

TEST_F(CudaDevicePoolDomainTest, CreateAllDevices) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create_all();
    ASSERT_TRUE(result.has_value());
    
    auto& pool = *result;
    EXPECT_TRUE(pool.valid());
    EXPECT_EQ(pool.device_count(), static_cast<dtl::size_type>(device_count_));
    
    for (int i = 0; i < device_count_; ++i) {
        EXPECT_TRUE(pool.contains(i));
    }
}

TEST_F(CudaDevicePoolDomainTest, CreateWithEmptyListFails) {
    auto result = dtl::cuda_device_pool_domain::create({});
    EXPECT_FALSE(result.has_value());
}

TEST_F(CudaDevicePoolDomainTest, CreateWithInvalidDeviceFails) {
    auto result = dtl::cuda_device_pool_domain::create({999});
    EXPECT_FALSE(result.has_value());
}

TEST_F(CudaDevicePoolDomainTest, CreateDeduplicatesDevices) {
    SKIP_IF_NO_CUDA();
    
    // Duplicate device IDs should be deduplicated
    auto result = dtl::cuda_device_pool_domain::create({0, 0, 0});
    ASSERT_TRUE(result.has_value());
    
    EXPECT_EQ(result->device_count(), 1u);
}

// ============================================================================
// Device Access Tests
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, GetDeviceIdByIndex) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto result = dtl::cuda_device_pool_domain::create({1, 0});  // Note: sorted internally
    ASSERT_TRUE(result.has_value());
    
    // After sorting, device 0 should be first
    EXPECT_EQ(result->device_id_at(0), 0);
    EXPECT_EQ(result->device_id_at(1), 1);
}

TEST_F(CudaDevicePoolDomainTest, GetStreamForDevice) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    cudaStream_t stream = result->stream(0);
    EXPECT_NE(stream, nullptr);
}

TEST_F(CudaDevicePoolDomainTest, GetStreamByIndex) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    cudaStream_t stream = result->stream_at(0);
    EXPECT_NE(stream, nullptr);
}

TEST_F(CudaDevicePoolDomainTest, StreamForInvalidDeviceThrows) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    EXPECT_THROW(result->stream(999), std::out_of_range);
}

TEST_F(CudaDevicePoolDomainTest, PrimaryDeviceId) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto result = dtl::cuda_device_pool_domain::create({1, 0});
    ASSERT_TRUE(result.has_value());
    
    // Primary device is first in sorted list
    EXPECT_EQ(result->primary_device_id(), 0);
}

// ============================================================================
// Synchronization Tests
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, SynchronizeSingleDevice) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    // Should not throw
    EXPECT_NO_THROW(result->synchronize(0));
}

TEST_F(CudaDevicePoolDomainTest, SynchronizeAllDevices) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create_all();
    ASSERT_TRUE(result.has_value());
    
    // Should not throw
    EXPECT_NO_THROW(result->synchronize_all());
}

// ============================================================================
// Device Guard Factory Tests
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, MakeDeviceGuard) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto result = dtl::cuda_device_pool_domain::create({0, 1});
    ASSERT_TRUE(result.has_value());
    
    int original = dtl::cuda::current_device_id();
    
    {
        auto guard = result->make_device_guard(1);
        EXPECT_EQ(dtl::cuda::current_device_id(), 1);
    }
    
    EXPECT_EQ(dtl::cuda::current_device_id(), original);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, MoveConstruction) {
    SKIP_IF_NO_CUDA();
    
    auto result = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result.has_value());
    
    dtl::cuda_device_pool_domain pool = std::move(*result);
    EXPECT_TRUE(pool.valid());
    EXPECT_EQ(pool.device_count(), 1u);
}

TEST_F(CudaDevicePoolDomainTest, MoveAssignment) {
    SKIP_IF_NO_CUDA();
    
    auto result1 = dtl::cuda_device_pool_domain::create({0});
    ASSERT_TRUE(result1.has_value());
    
    dtl::cuda_device_pool_domain pool;
    EXPECT_FALSE(pool.valid());
    
    pool = std::move(*result1);
    EXPECT_TRUE(pool.valid());
}

// ============================================================================
// Device IDs Query Test
// ============================================================================

TEST_F(CudaDevicePoolDomainTest, GetDeviceIds) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    auto result = dtl::cuda_device_pool_domain::create({1, 0});
    ASSERT_TRUE(result.has_value());
    
    std::vector<int> ids = result->device_ids();
    ASSERT_EQ(ids.size(), 2u);
    
    // Should be sorted
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 1);
}

}  // namespace

#else  // !DTL_ENABLE_CUDA

#include <gtest/gtest.h>

TEST(CudaDevicePoolDomainTest, CudaNotEnabled) {
    // Test stub behavior
    auto result = dtl::cuda_device_pool_domain::create({0});
    EXPECT_FALSE(result.has_value());
    
    dtl::cuda_device_pool_domain pool;
    EXPECT_FALSE(pool.valid());
    EXPECT_EQ(pool.device_count(), 0u);
}

#endif  // DTL_ENABLE_CUDA
