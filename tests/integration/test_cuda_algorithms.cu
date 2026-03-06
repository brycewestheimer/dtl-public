// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_algorithms.cpp
/// @brief Integration tests for CUDA GPU-accelerated algorithms
/// @details Tests Thrust-based algorithm implementations for GPU execution.

#include <dtl/core/config.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/cuda_exec.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/cuda_algorithms.hpp>
#include <backends/cuda/cuda_memory_space.hpp>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_CUDA

// =============================================================================
// Test Fixture
// =============================================================================

class CudaAlgorithmsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = cuda::device_count();
        if (device_count <= 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Keep algorithm tests deterministic by pinning to device 0.
        // Other CUDA suites may leave a different active device in thread-local state.
        cudaError_t set_err = cudaSetDevice(0);
        if (set_err != cudaSuccess) {
            GTEST_SKIP() << "Unable to set CUDA device 0: " << cudaGetErrorString(set_err);
        }
        (void)cudaGetLastError();  // Clear any stale async error state.
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        (void)cudaGetLastError();  // Prevent cross-test contamination from async failures.
    }

    /// @brief Allocate device memory for type T
    template <typename T>
    T* allocate_device(size_type count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    /// @brief Free device memory
    void free_device(void* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    /// @brief Copy data from host to device
    template <typename T>
    void copy_to_device(T* device, const T* host, size_type count) {
        cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    /// @brief Copy data from device to host
    template <typename T>
    void copy_to_host(T* host, const T* device, size_type count) {
        cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

// =============================================================================
// Execution Policy Tests
// =============================================================================

TEST(CudaExecPolicyTest, IsCudaPolicy) {
    static_assert(is_cuda_policy_v<cuda_exec>, "cuda_exec should be recognized as CUDA policy");
    static_assert(!is_cuda_policy_v<seq>, "seq should not be CUDA policy");
    static_assert(!is_cuda_policy_v<par>, "par should not be CUDA policy");
}

TEST(CudaExecPolicyTest, ExecutionTraits) {
    static_assert(!execution_traits<cuda_exec>::is_blocking,
                  "cuda_exec should be non-blocking");
    static_assert(execution_traits<cuda_exec>::is_parallel,
                  "cuda_exec should be parallel");
    static_assert(execution_traits<cuda_exec>::mode == execution_mode::asynchronous,
                  "cuda_exec should be asynchronous");
    static_assert(execution_traits<cuda_exec>::parallelism == parallelism_level::heterogeneous,
                  "cuda_exec should be heterogeneous");
}

TEST(CudaExecPolicyTest, DeviceExecutionConcept) {
    static_assert(DeviceExecutionPolicy<cuda_exec>,
                  "cuda_exec must satisfy DeviceExecutionPolicy concept");
}

// =============================================================================
// Reduce Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, ReduceSumBasic) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 1);

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    // GPU reduce sum
    int gpu_result = cuda::reduce_sum_device(device_data, count);

    // Expected result: 1 + 2 + ... + 1000 = 1000 * 1001 / 2 = 500500
    EXPECT_EQ(gpu_result, 500500);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, ReduceWithBinaryOp) {
    constexpr size_type count = 100;
    std::vector<double> host_data(count, 2.0);

    double* device_data = allocate_device<double>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    // GPU reduce with multiply (product of 100 elements = 2^100, too big, use smaller)
    double gpu_result = cuda::reduce_device(device_data, count, 0.0, thrust::plus<double>{});

    EXPECT_DOUBLE_EQ(gpu_result, 200.0);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, ReduceMinElement) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 100);
    host_data[500] = -42;  // Put minimum in the middle

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    int gpu_result = cuda::reduce_min_device(device_data, count);
    EXPECT_EQ(gpu_result, -42);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, ReduceMaxElement) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);
    host_data[750] = 9999;  // Put maximum in the middle

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    int gpu_result = cuda::reduce_max_device(device_data, count);
    EXPECT_EQ(gpu_result, 9999);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, ReduceEmpty) {
    // Empty reduce should return init value
    int* device_data = nullptr;
    int result = cuda::reduce_device(device_data, 0, 42, thrust::plus<int>{});
    EXPECT_EQ(result, 42);
}

// =============================================================================
// Sort Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, SortBasic) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);

    // Fill with descending values
    for (size_type i = 0; i < count; ++i) {
        host_data[i] = static_cast<int>(count - i);
    }

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    cuda::sort_device(device_data, count);

    copy_to_host(host_data.data(), device_data, count);

    // Verify sorted in ascending order
    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(host_data[i], static_cast<int>(i + 1));
    }

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, SortWithComparator) {
    constexpr size_type count = 100;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 1);

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    // Sort in descending order
    cuda::sort_device(device_data, count, thrust::greater<int>{});

    copy_to_host(host_data.data(), device_data, count);

    // Verify sorted in descending order
    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(host_data[i], static_cast<int>(count - i));
    }

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, SortEmpty) {
    // Sort of empty or single element should not crash
    cuda::sort_device<int>(nullptr, 0);

    int single = 42;
    int* device_data = allocate_device<int>(1);
    copy_to_device(device_data, &single, 1);

    cuda::sort_device(device_data, 1);

    int result;
    copy_to_host(&result, device_data, 1);
    EXPECT_EQ(result, 42);

    free_device(device_data);
}

// =============================================================================
// Fill Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, FillBasic) {
    constexpr size_type count = 1000;

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    cuda::fill_device(device_data, count, 42);

    std::vector<int> host_data(count);
    copy_to_host(host_data.data(), device_data, count);

    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(host_data[i], 42);
    }

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, FillDouble) {
    constexpr size_type count = 500;

    double* device_data = allocate_device<double>(count);
    ASSERT_NE(device_data, nullptr);

    cuda::fill_device(device_data, count, 3.14159);

    std::vector<double> host_data(count);
    copy_to_host(host_data.data(), device_data, count);

    for (size_type i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(host_data[i], 3.14159);
    }

    free_device(device_data);
}

// =============================================================================
// Copy Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, CopyDeviceToDevice) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);

    int* src_device = allocate_device<int>(count);
    int* dst_device = allocate_device<int>(count);
    ASSERT_NE(src_device, nullptr);
    ASSERT_NE(dst_device, nullptr);

    copy_to_device(src_device, host_data.data(), count);

    cuda::copy_device(src_device, dst_device, count);

    std::vector<int> result(count);
    copy_to_host(result.data(), dst_device, count);

    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }

    free_device(src_device);
    free_device(dst_device);
}

// =============================================================================
// Count Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, CountBasic) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);

    // Put some specific values
    std::fill(host_data.begin(), host_data.end(), 0);
    for (size_type i = 0; i < count; i += 10) {
        host_data[i] = 42;
    }

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    size_type result = cuda::count_device(device_data, count, 42);
    EXPECT_EQ(result, 100);  // 1000/10 = 100

    free_device(device_data);
}

// Note: Device lambdas require CUDA compilation (.cu file)
// These tests use Thrust functors which work with standard C++ compilation

// =============================================================================
// Thrust Functor Tests
// =============================================================================

/// @brief Custom positive predicate for testing (works without __device__)
struct is_positive_predicate {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
};

/// @brief Custom non-zero predicate for testing
struct is_non_zero_predicate {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

/// @brief Custom negative predicate for testing
struct is_negative_predicate {
    __host__ __device__ bool operator()(int x) const { return x < 0; }
};

// =============================================================================
// All/Any/None Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, AllOfTrue) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 1);  // All positive

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    bool result = cuda::all_of_device(device_data, count, is_positive_predicate{});

    EXPECT_TRUE(result);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, AllOfFalse) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 1);
    host_data[500] = -1;  // One negative

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    bool result = cuda::all_of_device(device_data, count, is_positive_predicate{});

    EXPECT_FALSE(result);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, AnyOfTrue) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count, 0);
    host_data[999] = 42;  // One non-zero

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    bool result = cuda::any_of_device(device_data, count, is_non_zero_predicate{});

    EXPECT_TRUE(result);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, NoneOfTrue) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count, 1);  // All positive

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    bool result = cuda::none_of_device(device_data, count, is_negative_predicate{});

    EXPECT_TRUE(result);

    free_device(device_data);
}

// =============================================================================
// Find Algorithm Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, FindBasic) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    size_type result = cuda::find_device(device_data, count, 500);
    EXPECT_EQ(result, 500);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, FindNotFound) {
    constexpr size_type count = 1000;
    std::vector<int> host_data(count);
    std::iota(host_data.begin(), host_data.end(), 0);

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    copy_to_device(device_data, host_data.data(), count);

    size_type result = cuda::find_device(device_data, count, 9999);
    EXPECT_EQ(result, count);  // Not found returns n

    free_device(device_data);
}

// =============================================================================
// Stream Execution Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, StreamExecutionOrder) {
    constexpr size_type count = 1000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int* device_data = allocate_device<int>(count);
    ASSERT_NE(device_data, nullptr);

    // Fill, then reduce on same stream - should be ordered
    cuda::fill_device(device_data, count, 5, stream);
    int result = cuda::reduce_sum_device(device_data, count, stream);

    cudaStreamSynchronize(stream);

    EXPECT_EQ(result, 5000);  // 1000 * 5

    cudaStreamDestroy(stream);
    free_device(device_data);
}

// =============================================================================
// Unified Memory Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, UnifiedMemoryReduce) {
    constexpr size_type count = 1000;

    // Allocate unified memory
    int* data = nullptr;
    cudaError_t err = cudaMallocManaged(&data, count * sizeof(int));
    ASSERT_EQ(err, cudaSuccess);

    // Initialize from host
    for (size_type i = 0; i < count; ++i) {
        data[i] = 1;
    }

    // Reduce on device
    int result = cuda::reduce_sum_device(data, count);

    EXPECT_EQ(result, 1000);

    cudaFree(data);
}

// =============================================================================
// Large Data Tests
// =============================================================================

TEST_F(CudaAlgorithmsTest, LargeDataReduce) {
    constexpr size_type count = 10'000'000;  // 10 million elements

    int* device_data = allocate_device<int>(count);
    if (device_data == nullptr) {
        GTEST_SKIP() << "Not enough device memory for large data test";
    }

    // Fill with 1s
    cuda::fill_device(device_data, count, 1);

    int result = cuda::reduce_sum_device(device_data, count);

    EXPECT_EQ(result, 10'000'000);

    free_device(device_data);
}

TEST_F(CudaAlgorithmsTest, LargeDataSort) {
    constexpr size_type count = 1'000'000;  // 1 million elements

    int* device_data = allocate_device<int>(count);
    if (device_data == nullptr) {
        GTEST_SKIP() << "Not enough device memory for large data test";
    }

    // Initialize with random-ish values (descending)
    std::vector<int> host_data(count);
    for (size_type i = 0; i < count; ++i) {
        host_data[i] = static_cast<int>(count - i);
    }
    copy_to_device(device_data, host_data.data(), count);

    cuda::sort_device(device_data, count);

    // Verify first and last elements
    int first, last;
    cudaMemcpy(&first, device_data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last, device_data + count - 1, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(first, 1);
    EXPECT_EQ(last, static_cast<int>(count));

    free_device(device_data);
}

#else  // !DTL_ENABLE_CUDA

TEST(CudaAlgorithmsTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled - skipping CUDA algorithm tests";
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::test
