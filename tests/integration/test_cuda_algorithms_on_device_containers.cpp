// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_algorithms_on_device_containers.cpp
/// @brief Integration tests for GPU algorithms on device containers
/// @since 0.1.0

#include <dtl/algorithms/gpu_algorithms.hpp>
#include <dtl/cuda/device_buffer.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/memory/copy.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/functional.h>
#endif

#include <vector>
#include <numeric>
#include <cstdio>
#include <cmath>

int main() {
#if DTL_ENABLE_CUDA
    // Check if CUDA device is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("No CUDA devices available. Skipping test.\n");
        return 0;
    }

    printf("Testing GPU algorithms on device containers...\n");

    // Test 1: fill_device with raw pointers
    {
        constexpr size_t N = 1024;
        dtl::cuda::device_buffer<int> buffer(N, 0);

        auto result = dtl::fill_device(buffer.data(), buffer.size(), 42, 0, 0);
        if (!result) {
            fprintf(stderr, "Test 1 FAILED: fill_device returned error\n");
            return 1;
        }

        cudaDeviceSynchronize();

        auto host_data = dtl::copy_to_host(buffer.data(), buffer.size(), 0);

        for (size_t i = 0; i < N; ++i) {
            if (host_data[i] != 42) {
                fprintf(stderr, "Test 1 FAILED: fill_device incorrect at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 1 PASSED: fill_device with raw pointers\n");
    }

    // Test 2: fill_device with device_view
    {
        constexpr size_t N = 512;
        dtl::cuda::device_buffer<float> buffer(N, 0);
        dtl::device_view<float> view(buffer.data(), buffer.size(), 0);

        auto result = dtl::fill_device(view, 3.14f);
        if (!result) {
            fprintf(stderr, "Test 2 FAILED: fill_device(view) returned error\n");
            return 1;
        }

        cudaDeviceSynchronize();

        auto host_data = dtl::copy_to_host(view);

        for (size_t i = 0; i < N; ++i) {
            if (std::abs(host_data[i] - 3.14f) > 0.001f) {
                fprintf(stderr, "Test 2 FAILED: fill_device(view) incorrect at index %zu\n", i);
                return 1;
            }
        }
        printf("Test 2 PASSED: fill_device with device_view\n");
    }

    // Test 3: transform_device (in-place)
    {
        constexpr size_t N = 256;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        dtl::device_view<int> view(buffer.data(), buffer.size(), 0);

        // Fill with 1, 2, 3, ...
        std::vector<int> host_data(N);
        std::iota(host_data.begin(), host_data.end(), 1);
        dtl::copy_from_host(host_data, view);

        // Transform: square each element
        auto square = [] __device__ (int x) { return x * x; };
        auto result = dtl::transform_device(view.data(), view.data(), view.size(),
                                             square, 0, 0);
        if (!result) {
            fprintf(stderr, "Test 3 FAILED: transform_device returned error\n");
            return 1;
        }

        cudaDeviceSynchronize();

        auto verify = dtl::copy_to_host(view);

        for (size_t i = 0; i < N; ++i) {
            int expected = static_cast<int>((i + 1) * (i + 1));
            if (verify[i] != expected) {
                fprintf(stderr, "Test 3 FAILED: transform at index %zu: expected %d, got %d\n",
                        i, expected, verify[i]);
                return 1;
            }
        }
        printf("Test 3 PASSED: transform_device (in-place square)\n");
    }

    // Test 4: sum_device
    {
        constexpr size_t N = 1000;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        dtl::device_view<int> view(buffer.data(), buffer.size(), 0);

        // Fill with 1, 2, 3, ... N
        std::vector<int> host_data(N);
        std::iota(host_data.begin(), host_data.end(), 1);
        dtl::copy_from_host(host_data, view);

        auto result = dtl::sum_device(view);
        if (!result) {
            fprintf(stderr, "Test 4 FAILED: sum_device returned error\n");
            return 1;
        }

        // Sum of 1..N = N*(N+1)/2
        int expected = static_cast<int>(N * (N + 1) / 2);
        if (result.value() != expected) {
            fprintf(stderr, "Test 4 FAILED: expected sum %d, got %d\n",
                    expected, result.value());
            return 1;
        }
        printf("Test 4 PASSED: sum_device\n");
    }

    // Test 5: reduce_device with custom operation
    {
        constexpr size_t N = 128;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        dtl::device_view<int> view(buffer.data(), buffer.size(), 0);

        // Fill with 2, 2, 2, ...
        std::vector<int> host_data(N, 2);
        dtl::copy_from_host(host_data, view);

        // Reduce with max
        auto result = dtl::reduce_device(view, 0, thrust::maximum<int>());
        if (!result) {
            fprintf(stderr, "Test 5 FAILED: reduce_device returned error\n");
            return 1;
        }

        if (result.value() != 2) {
            fprintf(stderr, "Test 5 FAILED: expected max 2, got %d\n", result.value());
            return 1;
        }
        printf("Test 5 PASSED: reduce_device with maximum\n");
    }

    // Test 6: sort_device
    {
        constexpr size_t N = 500;
        dtl::cuda::device_buffer<int> buffer(N, 0);
        dtl::device_view<int> view(buffer.data(), buffer.size(), 0);

        // Fill with random-ish data (descending)
        std::vector<int> host_data(N);
        for (size_t i = 0; i < N; ++i) {
            host_data[i] = static_cast<int>(N - i);
        }
        dtl::copy_from_host(host_data, view);

        auto result = dtl::sort_device(view);
        if (!result) {
            fprintf(stderr, "Test 6 FAILED: sort_device returned error\n");
            return 1;
        }

        cudaDeviceSynchronize();

        auto sorted = dtl::copy_to_host(view);

        // Verify sorted in ascending order
        for (size_t i = 1; i < N; ++i) {
            if (sorted[i] < sorted[i - 1]) {
                fprintf(stderr, "Test 6 FAILED: sort_device produced unsorted output at %zu\n", i);
                return 1;
            }
        }

        // Verify all elements present (should be 1..N sorted)
        for (size_t i = 0; i < N; ++i) {
            if (sorted[i] != static_cast<int>(i + 1)) {
                fprintf(stderr, "Test 6 FAILED: sorted[%zu] = %d, expected %zu\n",
                        i, sorted[i], i + 1);
                return 1;
            }
        }
        printf("Test 6 PASSED: sort_device\n");
    }

    // Test 7: Dispatch traits (compile-time)
    {
        // These are compile-time checks validated by the fact this compiles

        // DeviceStorable check
        static_assert(dtl::DeviceStorable<int>, "int should be DeviceStorable");
        static_assert(dtl::DeviceStorable<float>, "float should be DeviceStorable");
        static_assert(!dtl::DeviceStorable<std::string>, "string should not be DeviceStorable");

        printf("Test 7 PASSED: dispatch traits (compile-time)\n");
    }

    printf("\nAll GPU algorithm tests PASSED!\n");
    return 0;

#else
    printf("CUDA not enabled. Skipping GPU algorithm tests.\n");
    return 0;
#endif
}
