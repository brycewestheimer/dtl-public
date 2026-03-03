// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_device_buffer.cpp
/// @brief Integration tests for cuda::device_buffer
/// @since 0.1.0

#include <dtl/cuda/device_buffer.hpp>
#include <dtl/core/device_concepts.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>
#include <cstdio>
#include <cstdlib>

// Helper to check CUDA errors
#if DTL_ENABLE_CUDA
#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            return 1;                                                        \
        }                                                                    \
    } while (0)
#endif

int main() {
#if DTL_ENABLE_CUDA
    // Check if CUDA device is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("No CUDA devices available. Skipping test.\n");
        return 0;  // Skip test
    }

    printf("Testing device_buffer on device 0...\n");

    // Test 1: Default construction
    {
        dtl::cuda::device_buffer<int> buf;
        if (buf.data() != nullptr) {
            fprintf(stderr, "Test 1 FAILED: default buffer should have null data\n");
            return 1;
        }
        if (buf.size() != 0) {
            fprintf(stderr, "Test 1 FAILED: default buffer should have size 0\n");
            return 1;
        }
        printf("Test 1 PASSED: default construction\n");
    }

    // Test 2: Construction with size
    {
        constexpr size_t N = 1024;
        dtl::cuda::device_buffer<float> buf(N, 0);

        if (buf.data() == nullptr) {
            fprintf(stderr, "Test 2 FAILED: buffer should have non-null data\n");
            return 1;
        }
        if (buf.size() != N) {
            fprintf(stderr, "Test 2 FAILED: buffer size should be %zu, got %zu\n",
                    N, static_cast<size_t>(buf.size()));
            return 1;
        }
        if (buf.device_id() != 0) {
            fprintf(stderr, "Test 2 FAILED: device_id should be 0\n");
            return 1;
        }
        printf("Test 2 PASSED: construction with size\n");
    }

    // Test 3: Memset and copy back
    {
        constexpr size_t N = 256;
        dtl::cuda::device_buffer<int> buf(N, 0);

        // Zero-fill
        buf.memset(0);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy back to host and verify
        std::vector<int> host_data(N);
        CHECK_CUDA(cudaMemcpy(host_data.data(), buf.data(), N * sizeof(int),
                              cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < N; ++i) {
            if (host_data[i] != 0) {
                fprintf(stderr, "Test 3 FAILED: memset did not zero memory at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 3 PASSED: memset and copy back\n");
    }

    // Test 4: Resize (grow)
    {
        constexpr size_t N1 = 128;
        constexpr size_t N2 = 512;
        dtl::cuda::device_buffer<double> buf(N1, 0);

        // Write some data
        std::vector<double> host_data(N1, 3.14);
        CHECK_CUDA(cudaMemcpy(buf.data(), host_data.data(), N1 * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Resize (should preserve existing data)
        buf.resize(N2);

        if (buf.size() != N2) {
            fprintf(stderr, "Test 4 FAILED: resized buffer should have size %zu\n", N2);
            return 1;
        }

        // Verify old data preserved
        std::vector<double> verify_data(N2);
        CHECK_CUDA(cudaMemcpy(verify_data.data(), buf.data(), N2 * sizeof(double),
                              cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < N1; ++i) {
            if (verify_data[i] != 3.14) {
                fprintf(stderr, "Test 4 FAILED: data not preserved at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 4 PASSED: resize (grow)\n");
    }

    // Test 5: Move semantics
    {
        constexpr size_t N = 64;
        dtl::cuda::device_buffer<int> buf1(N, 0);
        int* original_ptr = buf1.data();

        // Move construct
        dtl::cuda::device_buffer<int> buf2(std::move(buf1));

        if (buf2.data() != original_ptr) {
            fprintf(stderr, "Test 5 FAILED: move should transfer pointer\n");
            return 1;
        }
        if (buf2.size() != N) {
            fprintf(stderr, "Test 5 FAILED: move should transfer size\n");
            return 1;
        }
        if (buf1.data() != nullptr || buf1.size() != 0) {
            fprintf(stderr, "Test 5 FAILED: moved-from buffer should be empty\n");
            return 1;
        }

        // Move assign
        dtl::cuda::device_buffer<int> buf3;
        buf3 = std::move(buf2);

        if (buf3.data() != original_ptr) {
            fprintf(stderr, "Test 5 FAILED: move assign should transfer pointer\n");
            return 1;
        }

        printf("Test 5 PASSED: move semantics\n");
    }

    // Test 6: Clear
    {
        constexpr size_t N = 100;
        dtl::cuda::device_buffer<int> buf(N, 0);

        buf.clear();

        if (buf.size() != 0) {
            fprintf(stderr, "Test 6 FAILED: clear should set size to 0\n");
            return 1;
        }
        // Capacity may still be N (memory retained)
        if (buf.capacity() < N) {
            fprintf(stderr, "Test 6 FAILED: clear should retain capacity\n");
            return 1;
        }
        printf("Test 6 PASSED: clear\n");
    }

    // Test 7: Device-storable constraint (compile-time, validated by compilation)
    {
        // This should compile: int is DeviceStorable
        dtl::cuda::device_buffer<int> int_buf;

        // This would NOT compile (correctly):
        // dtl::cuda::device_buffer<std::string> string_buf;  // Error!

        printf("Test 7 PASSED: DeviceStorable constraint\n");
    }

    printf("\nAll device_buffer tests PASSED!\n");
    return 0;

#else
    printf("CUDA not enabled. Skipping device_buffer tests.\n");
    return 0;
#endif
}
