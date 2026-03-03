// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_memory_space.cpp
/// @brief Integration tests for CUDA memory space
/// @details Tests CUDA memory allocation and transfers.

#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/concepts/executor.hpp>

#if DTL_ENABLE_CUDA
#include <backends/cuda/cuda_memory_space.hpp>
#include <backends/cuda/cuda_executor.hpp>
#include <backends/cuda/cuda_memory_transfer.hpp>
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_CUDA

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(CudaConceptTest, MemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<cuda::cuda_memory_space>,
                  "cuda_memory_space must satisfy MemorySpace concept");
}

TEST(CudaConceptTest, ExecutorSatisfiesConcept) {
    static_assert(Executor<cuda::cuda_executor>,
                  "cuda_executor must satisfy Executor concept");
}

// =============================================================================
// Test Fixture
// =============================================================================

class CudaMemorySpaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = cuda::device_count();
        if (device_count <= 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        space_ = std::make_unique<cuda::cuda_memory_space>();
    }

    void TearDown() override {
        // Ensure all CUDA operations complete
        cudaDeviceSynchronize();
    }

    std::unique_ptr<cuda::cuda_memory_space> space_;
};

// =============================================================================
// Memory Space Properties Tests
// =============================================================================

TEST_F(CudaMemorySpaceTest, NameIsCudaDevice) {
    EXPECT_STREQ(space_->name(), "cuda_device");
}

TEST_F(CudaMemorySpaceTest, PropertiesAreCorrect) {
    auto props = space_->properties();
    EXPECT_FALSE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_FALSE(props.pageable);
    EXPECT_EQ(props.alignment, 256);
}

TEST_F(CudaMemorySpaceTest, DeviceIdValid) {
    EXPECT_GE(space_->device_id(), 0);
}

// =============================================================================
// Allocation Tests
// =============================================================================

TEST_F(CudaMemorySpaceTest, BasicAllocation) {
    constexpr size_type size = 1024;

    void* ptr = space_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(space_->contains(ptr));

    space_->deallocate(ptr, size);
}

TEST_F(CudaMemorySpaceTest, AlignedAllocation) {
    constexpr size_type size = 1024;
    constexpr size_type alignment = 512;

    void* ptr = space_->allocate(size, alignment);
    ASSERT_NE(ptr, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0);

    space_->deallocate(ptr, size);
}

TEST_F(CudaMemorySpaceTest, LargeAllocation) {
    constexpr size_type size = 1024 * 1024 * 100;  // 100 MB

    void* ptr = space_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Verify it's device memory
    EXPECT_TRUE(space_->contains(ptr));

    space_->deallocate(ptr, size);
}

TEST_F(CudaMemorySpaceTest, MultipleAllocations) {
    constexpr size_type num_allocs = 10;
    constexpr size_type size = 4096;

    std::vector<void*> ptrs(num_allocs);

    for (size_type i = 0; i < num_allocs; ++i) {
        ptrs[i] = space_->allocate(size);
        ASSERT_NE(ptrs[i], nullptr);
        EXPECT_TRUE(space_->contains(ptrs[i]));
    }

    // All allocations should be distinct
    for (size_type i = 0; i < num_allocs; ++i) {
        for (size_type j = i + 1; j < num_allocs; ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]);
        }
    }

    for (size_type i = 0; i < num_allocs; ++i) {
        space_->deallocate(ptrs[i], size);
    }
}

TEST_F(CudaMemorySpaceTest, AllocationTracking) {
    constexpr size_type size = 1024;

    size_type before = space_->total_allocated();
    void* ptr = space_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(space_->total_allocated(), before + size);

    space_->deallocate(ptr, size);
    EXPECT_EQ(space_->total_allocated(), before);
}

TEST_F(CudaMemorySpaceTest, Memset) {
    constexpr size_type size = 1024;

    void* ptr = space_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Memset to 0
    space_->memset(ptr, 0, size);

    // Copy to host and verify
    std::vector<char> host_data(size);
    cudaMemcpy(host_data.data(), ptr, size, cudaMemcpyDeviceToHost);

    for (size_type i = 0; i < size; ++i) {
        EXPECT_EQ(host_data[i], 0);
    }

    space_->deallocate(ptr, size);
}

TEST_F(CudaMemorySpaceTest, HostPointerNotContained) {
    int host_var = 42;
    EXPECT_FALSE(space_->contains(&host_var));
}

// =============================================================================
// Memory Transfer Tests
// =============================================================================

TEST_F(CudaMemorySpaceTest, HostToDeviceTransfer) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);

    void* device_ptr = space_->allocate(count * sizeof(int));
    ASSERT_NE(device_ptr, nullptr);

    // Copy host to device
    cudaError_t err = cudaMemcpy(device_ptr, host_data.data(),
                                  count * sizeof(int), cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    // Copy back and verify
    std::vector<int> result(count);
    err = cudaMemcpy(result.data(), device_ptr,
                     count * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }

    space_->deallocate(device_ptr, count * sizeof(int));
}

TEST_F(CudaMemorySpaceTest, DeviceToDeviceTransfer) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);

    void* src_ptr = space_->allocate(count * sizeof(int));
    void* dst_ptr = space_->allocate(count * sizeof(int));
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);

    // Initialize source
    cudaMemcpy(src_ptr, host_data.data(), count * sizeof(int), cudaMemcpyHostToDevice);

    // Device to device copy
    cudaError_t err = cudaMemcpy(dst_ptr, src_ptr,
                                  count * sizeof(int), cudaMemcpyDeviceToDevice);
    EXPECT_EQ(err, cudaSuccess);

    // Copy back and verify
    std::vector<int> result(count);
    cudaMemcpy(result.data(), dst_ptr, count * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }

    space_->deallocate(src_ptr, count * sizeof(int));
    space_->deallocate(dst_ptr, count * sizeof(int));
}

// =============================================================================
// Executor Tests
// =============================================================================

class CudaExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = cuda::device_count();
        if (device_count <= 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaExecutorTest, NameIsCuda) {
    cuda::cuda_executor exec;
    EXPECT_STREQ(exec.name(), "cuda");
}

TEST_F(CudaExecutorTest, DefaultExecutorValid) {
    auto& exec = cuda::default_cuda_executor();
    EXPECT_TRUE(exec.valid());
}

TEST_F(CudaExecutorTest, ExecuteCallback) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    bool executed = false;
    exec.execute([&executed]() {
        executed = true;
    });

    exec.synchronize();
    EXPECT_TRUE(executed);
}

TEST_F(CudaExecutorTest, SyncExecute) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    bool executed = false;
    exec.sync_execute([&executed]() {
        executed = true;
    });

    // Should be executed by the time sync_execute returns
    EXPECT_TRUE(executed);
}

TEST_F(CudaExecutorTest, StreamSynchronization) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    int counter = 0;

    // Submit multiple callbacks
    for (int i = 0; i < 10; ++i) {
        exec.execute([&counter, i]() {
            counter = i + 1;
        });
    }

    exec.synchronize();

    // All callbacks should have executed
    EXPECT_EQ(counter, 10);
}

TEST_F(CudaExecutorTest, IsIdle) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    // Initially should be idle
    EXPECT_TRUE(exec.is_idle());

    // After synchronize should be idle
    exec.synchronize();
    EXPECT_TRUE(exec.is_idle());
}

TEST_F(CudaExecutorTest, MultiStreamExecutor) {
    auto multi_exec = cuda::make_multi_stream_executor(4);
    EXPECT_EQ(multi_exec->num_streams(), 4);

    int counters[4] = {0, 0, 0, 0};

    // Submit to each stream
    for (size_type i = 0; i < 4; ++i) {
        (*multi_exec)[i].execute([&counters, i]() {
            counters[i] = static_cast<int>(i + 1);
        });
    }

    multi_exec->synchronize_all();

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(counters[i], i + 1);
    }
}

// =============================================================================
// Device Query Tests
// =============================================================================

TEST(CudaDeviceTest, DeviceCount) {
    int count = cuda::device_count();
    EXPECT_GE(count, 0);
}

TEST(CudaDeviceTest, CurrentDevice) {
    if (cuda::device_count() <= 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    cuda::device_id_t device = cuda::current_device();
    EXPECT_GE(device, 0);
}

#else  // !DTL_ENABLE_CUDA

TEST(CudaTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled - skipping CUDA tests";
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::test
