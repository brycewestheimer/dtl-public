// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file gpu_resident_vector.cpp
/// @brief Device-only placement for GPU-resident data
/// @details Demonstrates DTL's device_only placement policy where data
///          lives primarily on the GPU. Shows how to minimize host-device
///          transfers by keeping data on the device across multiple operations.
///
/// @note Device Selection: The device_only<N> policy allocates
///       on device N specifically. Different N values produce different types.
///       Allocations are guarded to preserve the caller's current CUDA device.
///
/// Build (requires CUDA):
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON -DDTL_ENABLE_CUDA=ON
///   make
///
/// Run:
///   ./gpu_resident_vector
///
/// Expected output:
///   DTL GPU Resident Vector Example
///   ================================
///
///   Configuration:
///     Vector size: 1000000
///     CUDA enabled: true
///     Target device: 0
///
///   --- Creating GPU-Resident Vector ---
///   Creating distributed_vector with device_only<0> placement...
///   Placement: device-only (not host accessible)
///   Container device affinity: 0
///   Memory allocated successfully on GPU.
///
///   --- Initializing on GPU ---
///   Running initialization kernel...
///   Data initialized on GPU (no host transfer).
///
///   --- Multiple GPU Operations (No Transfers) ---
///   Operation 1: Scale by 2.0
///   Operation 2: Add 100.0
///   Operation 3: Square each element
///   All operations completed on GPU.
///
///   --- Final Transfer to Host ---
///   Copying results to host for verification...
///   Transfer complete.
///
///   --- Verification ---
///   Sample results (first 5): 10201 10404 10609 10816 11025
///   Expected formula: ((i * 2.0) + 100.0)^2
///   Results correct: true
///
///   SUCCESS: GPU resident vector example completed!

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>

// Kernel: Initialize with indices
__global__ void init_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<float>(idx);
    }
}

// Kernel: Scale by factor
__global__ void scale_kernel(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

// Kernel: Add constant
__global__ void add_kernel(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += value;
    }
}

// Kernel: Square each element
__global__ void square_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}
#endif

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    const auto ctx = dtl::make_cpu_context();

    const size_t N = 1000000;

    std::cout << "DTL GPU Resident Vector Example\n";
    std::cout << "================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Vector size: " << N << "\n";

#if DTL_ENABLE_CUDA
    std::cout << "  CUDA enabled: true\n";
    std::cout << "  Target device: 0\n\n";

    // =========================================================================
    // Create GPU-Resident Vector using DTL's device_only placement
    // =========================================================================
    std::cout << "--- Creating GPU-Resident Vector ---\n";
    std::cout << "Creating distributed_vector with device_only<0> placement...\n";

    // Use DTL's device_only<0> placement policy
    // This allocates memory on GPU device 0 specifically
    // Different device_only<N> values produce different types that allocate
    // on device N regardless of the current CUDA context device.
    using gpu_vector_t = dtl::distributed_vector<float, dtl::device_only<0>>;
    gpu_vector_t vec(N, ctx);  // N elements using context

    // Verify placement policy properties
    std::cout << "Placement: device-only "
              << "(host accessible: " << std::boolalpha << gpu_vector_t::is_host_accessible()
              << ", device accessible: " << gpu_vector_t::is_device_accessible() << ")\n";

    // Container carries device affinity
    std::cout << "Container device affinity: " << vec.device_id() << "\n";

    std::cout << "Memory allocated successfully on GPU.\n\n";

    // Get device pointer for kernels
    float* d_ptr = vec.local_data();
    int n = static_cast<int>(N);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // =========================================================================
    // Initialize on GPU
    // =========================================================================
    std::cout << "--- Initializing on GPU ---\n";
    std::cout << "Running initialization kernel...\n";

    init_kernel<<<numBlocks, blockSize>>>(d_ptr, n);
    cudaDeviceSynchronize();

    std::cout << "Data initialized on GPU (no host transfer).\n\n";

    // =========================================================================
    // Multiple GPU Operations (No Host Transfers)
    // =========================================================================
    std::cout << "--- Multiple GPU Operations (No Transfers) ---\n";

    std::cout << "Operation 1: Scale by 2.0\n";
    scale_kernel<<<numBlocks, blockSize>>>(d_ptr, 2.0f, n);

    std::cout << "Operation 2: Add 100.0\n";
    add_kernel<<<numBlocks, blockSize>>>(d_ptr, 100.0f, n);

    std::cout << "Operation 3: Square each element\n";
    square_kernel<<<numBlocks, blockSize>>>(d_ptr, n);

    cudaDeviceSynchronize();
    std::cout << "All operations completed on GPU.\n\n";

    // =========================================================================
    // Final Transfer to Host
    // =========================================================================
    std::cout << "--- Final Transfer to Host ---\n";
    std::cout << "Copying results to host for verification...\n";

    std::vector<float> host_data(N);
    cudaMemcpy(host_data.data(), d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Transfer complete.\n\n";

    // =========================================================================
    // Verification
    // =========================================================================
    std::cout << "--- Verification ---\n";
    std::cout << "Sample results (first 5): ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Expected formula: ((i * 2.0) + 100.0)^2\n";

    // Verify a few values
    bool correct = true;
    for (size_t i = 0; i < 100; ++i) {
        float expected = (static_cast<float>(i) * 2.0f + 100.0f);
        expected = expected * expected;
        if (std::abs(host_data[i] - expected) > 1e-3f) {
            correct = false;
            std::cout << "Mismatch at " << i << ": got " << host_data[i]
                      << ", expected " << expected << "\n";
        }
    }

    std::cout << "Results correct: " << std::boolalpha << correct << "\n\n";

    if (correct) {
        std::cout << "SUCCESS: GPU resident vector example completed!\n";
        return 0;
    } else {
        std::cout << "FAILURE: Verification failed!\n";
        return 1;
    }

#else
    std::cout << "  CUDA enabled: false\n\n";
    std::cout << "This example requires CUDA support.\n";
    std::cout << "Rebuild with: cmake .. -DDTL_ENABLE_CUDA=ON\n\n";

    // Show conceptual example without CUDA
    std::cout << "--- Conceptual Example (CPU simulation) ---\n";
    std::cout << "With GPU-resident vectors using DTL's device_only<0> placement:\n";
    std::cout << "  1. Allocate on device via cuda_device_memory_space\n";
    std::cout << "  2. Initialize via kernel (no transfer)\n";
    std::cout << "  3. Multiple operations via local_data() pointer (no transfers)\n";
    std::cout << "  4. Final transfer to host only when needed\n\n";

    std::cout << "DTL placement policies:\n";
    std::cout << "  - device_only<DeviceId>: GPU-only memory\n";
    std::cout << "  - unified_memory: Host+device accessible (managed memory)\n";
    std::cout << "  - device_preferred: GPU with host fallback\n";
    std::cout << "  - host_only: CPU memory (default)\n\n";

    std::cout << "This minimizes PCIe bandwidth usage and maximizes GPU utilization.\n\n";
    std::cout << "SUCCESS: Example completed (CUDA disabled)\n";
    return 0;
#endif
}
