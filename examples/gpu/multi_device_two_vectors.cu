// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file multi_device_two_vectors.cu
/// @brief Example: Multi-device usage with two vectors
/// @details Demonstrates creating containers on different GPU devices
///          and running operations on each without device context leakage.
/// @since 1.0.2

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <dtl/dtl.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/cuda/device_guard.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Kernel to fill with device-specific pattern
__global__ void fill_with_pattern(float* data, size_t n, int device_id) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Pattern: device_id * 1000 + index
        data[idx] = static_cast<float>(device_id * 1000 + idx % 100);
    }
}

// Kernel to compute sum reduction (simplified)
__global__ void partial_sum(const float* data, size_t n, float* result) {
    __shared__ float sdata[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Simple reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

void copy_to_host(int device_id, const float* device_ptr, float* host_ptr, size_t n) {
    dtl::cuda::device_guard guard(device_id);
    cudaMemcpy(host_ptr, device_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    // Initialize DTL environment
    dtl::environment env(argc, argv);

    // Check for multiple GPUs
    int device_count = dtl::cuda::device_count();
    
    std::cout << "=== DTL Multi-Device Example ===\n";
    std::cout << "CUDA devices available: " << device_count << "\n\n";

    if (device_count < 2) {
        std::cout << "This example requires at least 2 CUDA devices.\n";
        std::cout << "Skipping multi-device demonstration.\n";
        std::cout << "\n=== Example skipped (insufficient GPUs) ===\n";
        return 0;  // Not an error, just skip
    }

    // Create base context
    auto base_ctx = dtl::make_cpu_context();

    // Create contexts for two different devices
    auto ctx0 = base_ctx.with_cuda(0);
    auto ctx1 = base_ctx.with_cuda(1);

    std::cout << "Created contexts for device 0 and device 1\n";

    // Record original device
    int original_device = dtl::cuda::current_device_id();
    std::cout << "Original current device: " << original_device << "\n\n";

    // Create vectors on different devices
    constexpr size_t N = 1024;
    
    std::cout << "Creating vectors...\n";
    dtl::distributed_vector<float, dtl::device_only_runtime> vec0(N, ctx0);
    dtl::distributed_vector<float, dtl::device_only_runtime> vec1(N, ctx1);

    std::cout << "  vec0: device_id = " << vec0.device_id() << "\n";
    std::cout << "  vec1: device_id = " << vec1.device_id() << "\n";

    // Verify current device is unchanged
    int after_create = dtl::cuda::current_device_id();
    std::cout << "\nCurrent device after creation: " << after_create << "\n";
    
    if (after_create != original_device) {
        std::cerr << "ERROR: Device context leaked during container creation!\n";
        return 1;
    }

    // Launch kernels on each device
    std::cout << "\nLaunching kernels on each device...\n";

    int block_size = 256;
    int grid_size = (static_cast<int>(N) + block_size - 1) / block_size;

    // Kernel on device 0
    {
        dtl::cuda::device_guard guard(vec0.device_id());
        fill_with_pattern<<<grid_size, block_size>>>(vec0.local_data(), N, 0);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch on device 0 failed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        std::cout << "  Launched kernel on device 0\n";
    }

    // Kernel on device 1
    {
        dtl::cuda::device_guard guard(vec1.device_id());
        fill_with_pattern<<<grid_size, block_size>>>(vec1.local_data(), N, 1);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch on device 1 failed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        std::cout << "  Launched kernel on device 1\n";
    }

    // Synchronize both devices
    {
        dtl::cuda::device_guard guard(0);
        cudaDeviceSynchronize();
    }
    {
        dtl::cuda::device_guard guard(1);
        cudaDeviceSynchronize();
    }

    // Verify current device is still unchanged
    int after_kernels = dtl::cuda::current_device_id();
    std::cout << "\nCurrent device after kernels: " << after_kernels << "\n";

    if (after_kernels != original_device) {
        std::cerr << "ERROR: Device context leaked during kernel execution!\n";
        return 1;
    }

    // Copy results to host and verify patterns
    std::cout << "\nVerifying data patterns...\n";
    
    std::vector<float> host0(N), host1(N);
    copy_to_host(0, vec0.local_data(), host0.data(), N);
    copy_to_host(1, vec1.local_data(), host1.data(), N);

    // Check patterns
    bool vec0_correct = true;
    bool vec1_correct = true;

    for (size_t i = 0; i < std::min(N, size_t(10)); ++i) {
        float expected0 = static_cast<float>(0 * 1000 + i % 100);
        float expected1 = static_cast<float>(1 * 1000 + i % 100);
        
        if (std::fabs(host0[i] - expected0) > 0.001f) vec0_correct = false;
        if (std::fabs(host1[i] - expected1) > 0.001f) vec1_correct = false;
    }

    std::cout << "  vec0 pattern correct: " << (vec0_correct ? "YES" : "NO") << "\n";
    std::cout << "  vec1 pattern correct: " << (vec1_correct ? "YES" : "NO") << "\n";

    if (!vec0_correct || !vec1_correct) {
        std::cerr << "ERROR: Data pattern verification failed!\n";
        return 1;
    }

    // Sample values
    std::cout << "\nSample values:\n";
    std::cout << "  vec0[0] = " << host0[0] << " (expected: 0)\n";
    std::cout << "  vec0[50] = " << host0[50] << " (expected: 50)\n";
    std::cout << "  vec1[0] = " << host1[0] << " (expected: 1000)\n";
    std::cout << "  vec1[50] = " << host1[50] << " (expected: 1050)\n";

    // Final device check
    int final_device = dtl::cuda::current_device_id();
    std::cout << "\nFinal current device: " << final_device << "\n";

    if (final_device != original_device) {
        std::cerr << "ERROR: Device context leaked!\n";
        return 1;
    }

    std::cout << "\n=== Example completed successfully ===\n";
    std::cout << "Multi-device usage verified: no device context leakage.\n";
    return 0;
}

#else  // !DTL_ENABLE_CUDA

#include <iostream>

int main() {
    std::cerr << "This example requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON\n";
    return 1;
}

#endif  // DTL_ENABLE_CUDA
