// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file gpu_accelerated_transform.cpp
/// @brief GPU-accelerated transform operations using unified memory
/// @details Demonstrates DTL's unified_memory placement policy which provides
///          host+device accessible memory with automatic page migration.
///          Shows how unified memory simplifies GPU programming by eliminating
///          explicit data transfers.
///
/// Build (requires CUDA):
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON -DDTL_ENABLE_CUDA=ON
///   make
///
/// Run:
///   ./gpu_accelerated_transform
///
/// Expected output:
///   DTL GPU Accelerated Transform Example (Unified Memory)
///   ======================================================
///
///   Configuration:
///     Vector size: 1000000
///     CUDA enabled: true
///     Placement: unified_memory (host+device accessible)
///
///   --- Creating Unified Memory Vector ---
///   Creating distributed_vector with unified_memory placement...
///   Host accessible: true, Device accessible: true
///
///   --- Initializing on Host ---
///   Writing directly to unified memory from host...
///   Initialization complete (no explicit transfer needed).
///
///   --- GPU Computation ---
///   Computing f(x) = x^2 + 2*x + 1 on GPU...
///   GPU kernel complete.
///
///   --- Reading Results on Host ---
///   Reading directly from unified memory on host...
///   First 10 results: 1 4 9 16 25 36 49 64 81 100
///
///   --- Verification ---
///   Results correct: true
///
///   SUCCESS: Unified memory example completed!

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>

// CUDA kernel for element-wise transform: f(x) = x^2 + 2*x + 1 = (x+1)^2
__global__ void transform_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x * x + 2.0f * x + 1.0f;
    }
}
#endif

// CPU implementation for reference
void cpu_transform(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = data[i];
        data[i] = x * x + 2.0f * x + 1.0f;
    }
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    const auto ctx = dtl::make_cpu_context();

    const size_t N = 1000000;

    std::cout << "DTL GPU Accelerated Transform Example (Unified Memory)\n";
    std::cout << "======================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Vector size: " << N << "\n";

#if DTL_ENABLE_CUDA
    std::cout << "  CUDA enabled: true\n";

    // =========================================================================
    // Create Unified Memory Vector using DTL's unified_memory placement
    // =========================================================================
    std::cout << "  Placement: unified_memory (host+device accessible)\n\n";

    std::cout << "--- Creating Unified Memory Vector ---\n";
    std::cout << "Creating distributed_vector with unified_memory placement...\n";

    // Use DTL's unified_memory placement policy
    // This allocates memory via cudaMallocManaged - accessible from both host and device
    using unified_vector_t = dtl::distributed_vector<float, dtl::unified_memory>;
    unified_vector_t vec(N, ctx);  // N elements, 1 rank, rank 0

    // Verify placement policy properties
    std::cout << "Host accessible: " << std::boolalpha << unified_vector_t::is_host_accessible()
              << ", Device accessible: " << unified_vector_t::is_device_accessible() << "\n\n";

    // =========================================================================
    // Initialize on Host (Direct Access)
    // =========================================================================
    std::cout << "--- Initializing on Host ---\n";
    std::cout << "Writing directly to unified memory from host...\n";

    // With unified memory, we can directly access from host
    auto local = vec.local_view();
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] = static_cast<float>(i);
    }

    std::cout << "Initialization complete (no explicit transfer needed).\n\n";

    // =========================================================================
    // GPU Computation
    // =========================================================================
    std::cout << "--- GPU Computation ---\n";
    std::cout << "Computing f(x) = x^2 + 2*x + 1 on GPU...\n";

    // Get pointer for kernel (same pointer works on host and device!)
    float* ptr = vec.local_data();
    int n = static_cast<int>(N);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    auto gpu_start = std::chrono::high_resolution_clock::now();
    transform_kernel<<<numBlocks, blockSize>>>(ptr, n);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
        gpu_end - gpu_start).count() / 1000.0;

    std::cout << "GPU kernel complete (" << gpu_time << "ms).\n\n";

    // =========================================================================
    // Read Results on Host (Direct Access)
    // =========================================================================
    std::cout << "--- Reading Results on Host ---\n";
    std::cout << "Reading directly from unified memory on host...\n";

    // With unified memory, results are automatically visible on host
    // (no explicit cudaMemcpy needed!)
    std::cout << "First 10 results: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    // =========================================================================
    // Verification
    // =========================================================================
    std::cout << "--- Verification ---\n";

    bool correct = true;
    for (size_t i = 0; i < 1000; ++i) {
        float expected = static_cast<float>(i);
        expected = expected * expected + 2.0f * expected + 1.0f;  // (i+1)^2
        if (std::abs(local[i] - expected) > 1e-5f) {
            correct = false;
            std::cout << "Mismatch at " << i << ": got " << local[i]
                      << ", expected " << expected << "\n";
        }
    }

    std::cout << "Results correct: " << std::boolalpha << correct << "\n\n";

    if (correct) {
        std::cout << "SUCCESS: Unified memory example completed!\n";
        return 0;
    } else {
        std::cout << "FAILURE: Verification failed!\n";
        return 1;
    }

#else
    std::cout << "  CUDA enabled: false\n\n";
    std::cout << "This example demonstrates unified memory, which requires CUDA.\n";
    std::cout << "Rebuild with: cmake .. -DDTL_ENABLE_CUDA=ON\n\n";

    // CPU-only demonstration
    std::cout << "--- CPU-Only Demo ---\n";
    std::cout << "Running CPU transform for demonstration...\n";

    // Use default (host_only) placement
    dtl::distributed_vector<float> vec(N, ctx);
    auto local = vec.local_view();

    // Initialize
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] = static_cast<float>(i);
    }

    // Transform on CPU
    auto start = std::chrono::high_resolution_clock::now();
    cpu_transform(vec.local_data(), local.size());
    auto end = std::chrono::high_resolution_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count() / 1000.0;

    std::cout << "CPU transform complete (" << time << "ms).\n";
    std::cout << "First 10 results: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "DTL unified_memory placement benefits:\n";
    std::cout << "  - No explicit cudaMemcpy calls needed\n";
    std::cout << "  - Same pointer works on host and device\n";
    std::cout << "  - Automatic page migration handles data movement\n";
    std::cout << "  - Simpler code, easier debugging\n\n";

    std::cout << "SUCCESS: CPU transform example completed!\n";
    std::cout << "(Enable CUDA to see unified memory benefits)\n";
    return 0;
#endif
}
