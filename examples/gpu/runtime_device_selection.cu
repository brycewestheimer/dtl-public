// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file runtime_device_selection.cu
/// @brief Example: Runtime device selection with device_only_runtime
/// @details Demonstrates how to select a GPU device at runtime using the
///          device_only_runtime placement policy and context with_cuda().
/// @since 1.0.2

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <dtl/dtl.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/cuda/device_guard.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <string>

// Simple CUDA kernel to verify device is correct
__global__ void verify_device_kernel(int* result, int expected_device) {
    int device;
    cudaGetDevice(&device);  // This is the device where the kernel is running
    // Note: Inside kernel, use threadIdx to avoid CUDA API in kernel
    // Instead, we just write a known value
    *result = 42;  // Marker value to verify kernel ran
}

// Fill kernel
__global__ void fill_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

int get_device_from_args(int argc, char* argv[], int default_device = 0) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--device-id" && i + 1 < argc) {
            return std::atoi(argv[i + 1]);
        }
        if (std::string(argv[i]).substr(0, 12) == "--device-id=") {
            return std::atoi(argv[i] + 12);
        }
    }
    return default_device;
}

int main(int argc, char* argv[]) {
    // Initialize DTL environment
    dtl::environment env(argc, argv);

    // Get device count
    int device_count = dtl::cuda::device_count();
    if (device_count == 0) {
        std::cerr << "No CUDA devices available.\n";
        return 1;
    }

    // Get device ID from command line (runtime value!)
    int requested_device = get_device_from_args(argc, argv, 0);
    
    // Validate device ID
    if (requested_device < 0 || requested_device >= device_count) {
        std::cerr << "Invalid device ID: " << requested_device 
                  << ". Available devices: 0-" << (device_count - 1) << "\n";
        return 1;
    }

    std::cout << "=== DTL Runtime Device Selection Example ===\n";
    std::cout << "Total CUDA devices: " << device_count << "\n";
    std::cout << "Selected device: " << requested_device << "\n\n";

    // Create base context
    auto base_ctx = dtl::make_cpu_context();

    // Add CUDA domain with runtime-selected device
    // This is the key: device ID is a runtime value!
    auto ctx = base_ctx.with_cuda(requested_device);

    std::cout << "Context created with CUDA device " << requested_device << "\n";

    // Verify the context has the correct device
    if (ctx.has_cuda()) {
        const auto& cuda_dom = ctx.get<dtl::cuda_domain>();
        std::cout << "CUDA domain device ID: " << cuda_dom.device_id() << "\n";
    }

    // Create a distributed vector with runtime device selection
    // The device_only_runtime policy gets device ID from context
    constexpr size_t N = 1024;
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(N, ctx);

    // Verify container device affinity
    std::cout << "\nContainer device affinity:\n";
    std::cout << "  device_id(): " << vec.device_id() << "\n";
    std::cout << "  has_device_affinity(): " << (vec.has_device_affinity() ? "yes" : "no") << "\n";
    std::cout << "  is_device_accessible(): " << (vec.is_device_accessible() ? "yes" : "no") << "\n";
    std::cout << "  is_host_accessible(): " << (vec.is_host_accessible() ? "yes" : "no") << "\n";

    // Launch a kernel on the container's data
    // Note: We need to use device guard for kernel launch
    {
        dtl::cuda::device_guard guard(vec.device_id());
        
        float* data_ptr = vec.local_data();
        size_t n = vec.local_size();
        
        // Calculate grid/block dimensions
        int block_size = 256;
        int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;
        
        // Fill with value
        float fill_value = 3.14159f;
        fill_kernel<<<grid_size, block_size>>>(data_ptr, n, fill_value);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        std::cout << "\nSuccessfully launched kernel on device " << vec.device_id() << "\n";
    }

    // Verify current device is unchanged after container operations
    int current_device = dtl::cuda::current_device_id();
    std::cout << "\nCurrent CUDA device after operations: " << current_device << "\n";

    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}

#else  // !DTL_ENABLE_CUDA

#include <iostream>

int main() {
    std::cerr << "This example requires CUDA. Rebuild with -DDTL_ENABLE_CUDA=ON\n";
    return 1;
}

#endif  // DTL_ENABLE_CUDA
