// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_on_stream_cuda_handle.cpp
/// @brief Unit tests for on_stream execution policy with CUDA stream handles
/// @since 0.1.0

#include <dtl/policies/execution/on_stream.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/stream_handle.hpp>
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <type_traits>

int main() {
    printf("Testing on_stream execution policy...\n");

    // Test 1: Default stream handle construction
    {
        dtl::default_stream_handle handle;
        if (!handle.is_default()) {
            fprintf(stderr, "Test 1 FAILED: default handle should be default stream\n");
            return 1;
        }
        printf("Test 1 PASSED: default stream handle\n");
    }

    // Test 2: on_stream with default stream
    {
        dtl::on_stream<> policy;

        if (!policy.is_device_execution()) {
            fprintf(stderr, "Test 2 FAILED: on_stream should be device execution\n");
            return 1;
        }
        if (policy.is_blocking()) {
            fprintf(stderr, "Test 2 FAILED: on_stream should not be blocking\n");
            return 1;
        }
        if (!policy.is_parallel()) {
            fprintf(stderr, "Test 2 FAILED: on_stream should be parallel\n");
            return 1;
        }

        printf("Test 2 PASSED: on_stream basic properties\n");
    }

    // Test 3: execution_traits for on_stream
    {
        using traits = dtl::execution_traits<dtl::on_stream<>>;

        static_assert(!traits::is_blocking, "on_stream should not be blocking");
        static_assert(traits::is_parallel, "on_stream should be parallel");
        static_assert(traits::mode == dtl::execution_mode::asynchronous,
            "on_stream should be asynchronous");
        static_assert(traits::parallelism == dtl::parallelism_level::heterogeneous,
            "on_stream should be heterogeneous parallelism");

        printf("Test 3 PASSED: execution_traits for on_stream\n");
    }

    // Test 4: make_on_stream factory function
    {
        dtl::default_stream_handle handle;
        auto policy = dtl::make_on_stream(handle);

        using policy_type = decltype(policy);
        static_assert(std::is_same_v<typename policy_type::stream_type, dtl::default_stream_handle>,
            "make_on_stream should preserve stream type");

        printf("Test 4 PASSED: make_on_stream factory\n");
    }

#if DTL_ENABLE_CUDA
    printf("\nTesting CUDA-specific stream handle features...\n");

    // Test 5: CUDA stream_handle creation
    {
        dtl::cuda::stream_handle owned_stream(true);  // Create new stream

        if (owned_stream.is_default()) {
            fprintf(stderr, "Test 5 FAILED: created stream should not be default\n");
            return 1;
        }
        if (!owned_stream.owns()) {
            fprintf(stderr, "Test 5 FAILED: created stream should own handle\n");
            return 1;
        }

        printf("Test 5 PASSED: CUDA stream_handle creation\n");
    }

    // Test 6: CUDA stream_handle from raw stream
    {
        cudaStream_t raw_stream;
        cudaStreamCreate(&raw_stream);

        dtl::cuda::stream_handle wrapper(raw_stream, false);  // Don't own

        if (wrapper.native_handle() != raw_stream) {
            fprintf(stderr, "Test 6 FAILED: native_handle should match\n");
            return 1;
        }
        if (wrapper.owns()) {
            fprintf(stderr, "Test 6 FAILED: wrapper should not own\n");
            return 1;
        }

        cudaStreamDestroy(raw_stream);
        printf("Test 6 PASSED: CUDA stream_handle from raw stream\n");
    }

    // Test 7: stream_handle factory methods
    {
        auto stream = dtl::cuda::stream_handle::create();
        if (stream.is_default()) {
            fprintf(stderr, "Test 7 FAILED: created stream should not be default\n");
            return 1;
        }
        if (!stream.owns()) {
            fprintf(stderr, "Test 7 FAILED: created stream should own\n");
            return 1;
        }

        auto non_blocking = dtl::cuda::stream_handle::create_non_blocking();
        if (non_blocking.is_default()) {
            fprintf(stderr, "Test 7 FAILED: non-blocking stream should not be default\n");
            return 1;
        }

        auto default_stream = dtl::cuda::stream_handle::default_stream();
        if (!default_stream.is_default()) {
            fprintf(stderr, "Test 7 FAILED: default_stream() should be default\n");
            return 1;
        }

        printf("Test 7 PASSED: stream_handle factory methods\n");
    }

    // Test 8: stream synchronization
    {
        auto stream = dtl::cuda::stream_handle::create();

        // Launch a simple operation
        int* d_data;
        cudaMalloc(&d_data, sizeof(int));
        cudaMemsetAsync(d_data, 0, sizeof(int), stream.get());

        // Synchronize
        if (!stream.synchronize()) {
            fprintf(stderr, "Test 8 FAILED: synchronize should succeed\n");
            cudaFree(d_data);
            return 1;
        }

        cudaFree(d_data);
        printf("Test 8 PASSED: stream synchronization\n");
    }

    // Test 9: stream query
    {
        auto stream = dtl::cuda::stream_handle::create();

        // Empty stream should be complete
        if (!stream.query()) {
            fprintf(stderr, "Test 9 FAILED: empty stream should be complete\n");
            return 1;
        }

        printf("Test 9 PASSED: stream query\n");
    }

    // Test 10: Move semantics
    {
        auto stream1 = dtl::cuda::stream_handle::create();
        cudaStream_t raw = stream1.native_handle();

        // Move construct
        dtl::cuda::stream_handle stream2(std::move(stream1));

        if (stream2.native_handle() != raw) {
            fprintf(stderr, "Test 10 FAILED: move should transfer handle\n");
            return 1;
        }
        if (stream2.is_default()) {
            fprintf(stderr, "Test 10 FAILED: moved stream should not be default\n");
            return 1;
        }

        printf("Test 10 PASSED: stream_handle move semantics\n");
    }

    // Test 11: on_stream with CUDA stream_handle
    {
        auto stream = dtl::cuda::stream_handle::create();
        dtl::on_stream policy(stream);

        // Get native handle through policy
        auto native = policy.native_handle();
        if (native == nullptr) {
            fprintf(stderr, "Test 11 FAILED: native_handle should not be null\n");
            return 1;
        }

        printf("Test 11 PASSED: on_stream with CUDA stream_handle\n");
    }

#endif  // DTL_ENABLE_CUDA

    printf("\nAll on_stream tests PASSED!\n");
    return 0;
}
