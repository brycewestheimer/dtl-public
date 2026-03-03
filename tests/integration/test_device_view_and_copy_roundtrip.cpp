// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_view_and_copy_roundtrip.cpp
/// @brief Integration tests for device_view and copy helpers
/// @since 0.1.0

#include <dtl/views/device_view.hpp>
#include <dtl/memory/copy.hpp>
#include <dtl/cuda/device_buffer.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#endif

#include <vector>
#include <numeric>
#include <cstdio>

int main() {
#if DTL_ENABLE_CUDA
    // Check if CUDA device is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("No CUDA devices available. Skipping test.\n");
        return 0;
    }

    printf("Testing device_view and copy helpers...\n");

    // Test 1: Create device buffer and wrap in device_view
    {
        constexpr size_t N = 1024;
        dtl::cuda::device_buffer<int> buffer(N, 0);

        dtl::device_view<int> view(buffer.data(), buffer.size(), buffer.device_id());

        if (view.size() != N) {
            fprintf(stderr, "Test 1 FAILED: view size mismatch\n");
            return 1;
        }
        if (view.data() != buffer.data()) {
            fprintf(stderr, "Test 1 FAILED: view data pointer mismatch\n");
            return 1;
        }
        printf("Test 1 PASSED: device_view from buffer\n");
    }

    // Test 2: Host-to-device copy and verification
    {
        constexpr size_t N = 512;
        dtl::cuda::device_buffer<float> buffer(N, 0);
        dtl::device_view<float> view(buffer.data(), buffer.size(), buffer.device_id());

        // Create host data
        std::vector<float> host_data(N);
        std::iota(host_data.begin(), host_data.end(), 0.0f);  // 0, 1, 2, ...

        // Copy to device
        auto result = dtl::copy_from_host(host_data, view);
        if (!result) {
            fprintf(stderr, "Test 2 FAILED: copy_from_host failed\n");
            return 1;
        }

        // Copy back and verify
        auto verify_data = dtl::copy_to_host(view);
        if (verify_data.size() != N) {
            fprintf(stderr, "Test 2 FAILED: copy_to_host returned wrong size\n");
            return 1;
        }

        for (size_t i = 0; i < N; ++i) {
            if (verify_data[i] != static_cast<float>(i)) {
                fprintf(stderr, "Test 2 FAILED: data mismatch at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 2 PASSED: host-to-device-to-host roundtrip\n");
    }

    // Test 3: Use Thrust with device_view
    {
        constexpr size_t N = 256;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        dtl::device_view<int> view(buffer.data(), buffer.size(), buffer.device_id());

        // Use Thrust to fill the device memory
        thrust::device_ptr<int> begin(view.data());
        thrust::device_ptr<int> end = begin + view.size();
        thrust::fill(begin, end, 42);

        cudaDeviceSynchronize();

        // Copy back and verify
        auto host_data = dtl::copy_to_host(view);

        for (size_t i = 0; i < N; ++i) {
            if (host_data[i] != 42) {
                fprintf(stderr, "Test 3 FAILED: thrust fill did not work at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 3 PASSED: Thrust integration with device_view\n");
    }

    // Test 4: Subview operations
    {
        constexpr size_t N = 100;
        dtl::cuda::device_buffer<double> buffer(N, 0);
        dtl::device_view<double> full_view(buffer.data(), buffer.size(), buffer.device_id());

        // Create subview
        auto sub = full_view.subview(10, 20);  // Elements 10-29

        if (sub.size() != 20) {
            fprintf(stderr, "Test 4 FAILED: subview size wrong\n");
            return 1;
        }
        if (sub.data() != buffer.data() + 10) {
            fprintf(stderr, "Test 4 FAILED: subview data pointer wrong\n");
            return 1;
        }

        // Test first() and last()
        auto first10 = full_view.first(10);
        auto last10 = full_view.last(10);

        if (first10.size() != 10 || first10.data() != buffer.data()) {
            fprintf(stderr, "Test 4 FAILED: first() wrong\n");
            return 1;
        }
        if (last10.size() != 10 || last10.data() != buffer.data() + 90) {
            fprintf(stderr, "Test 4 FAILED: last() wrong\n");
            return 1;
        }

        printf("Test 4 PASSED: subview operations\n");
    }

    // Test 5: Device-to-device copy
    {
        constexpr size_t N = 128;
        dtl::cuda::device_buffer<int> src_buf(N, 0);
        dtl::cuda::device_buffer<int> dst_buf(N, 0);

        // Fill source with data from host
        std::vector<int> host_data(N, 99);
        cudaMemcpy(src_buf.data(), host_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // Device-to-device copy
        auto result = dtl::copy_device_to_device(src_buf.data(), dst_buf.data(), N, 0);
        if (!result) {
            fprintf(stderr, "Test 5 FAILED: device-to-device copy failed\n");
            return 1;
        }

        // Verify
        std::vector<int> verify(N);
        cudaMemcpy(verify.data(), dst_buf.data(), N * sizeof(int), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < N; ++i) {
            if (verify[i] != 99) {
                fprintf(stderr, "Test 5 FAILED: device-to-device copy incorrect at %zu\n", i);
                return 1;
            }
        }
        printf("Test 5 PASSED: device-to-device copy\n");
    }

    // Test 6: copy_result structure
    {
        constexpr size_t N = 64;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        std::vector<int> host_data(N, 123);

        auto result = dtl::copy_from_host(host_data.data(), buffer.data(), N, 0);

        if (!result.success) {
            fprintf(stderr, "Test 6 FAILED: copy_result.success should be true\n");
            return 1;
        }
        if (result.bytes_copied != N * sizeof(int)) {
            fprintf(stderr, "Test 6 FAILED: bytes_copied incorrect\n");
            return 1;
        }
        if (result.error_code != 0) {
            fprintf(stderr, "Test 6 FAILED: error_code should be 0\n");
            return 1;
        }

        printf("Test 6 PASSED: copy_result structure\n");
    }

    printf("\nAll device_view and copy helper tests PASSED!\n");
    return 0;

#else
    printf("CUDA not enabled. Skipping device_view and copy tests.\n");
    return 0;
#endif
}
