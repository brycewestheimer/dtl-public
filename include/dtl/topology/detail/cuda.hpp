// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file detail/cuda.hpp
/// @brief CUDA-specific hardware topology discovery
/// @details Uses CUDA runtime API to query GPU device properties.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/topology/hardware.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::topology::detail {

/// @brief Discover CUDA GPU devices
/// @param topo Topology to populate with GPU info
inline void discover_cuda_gpus([[maybe_unused]] hardware_topology& topo) {
#if DTL_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return;  // No CUDA devices
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, i);
        if (err != cudaSuccess) continue;

        gpu_device gpu;
        gpu.id = static_cast<std::uint32_t>(i);
        gpu.name = props.name;
        gpu.memory_bytes = static_cast<std::uint64_t>(props.totalGlobalMem);
        gpu.compute_capability_major = static_cast<std::uint32_t>(props.major);
        gpu.compute_capability_minor = static_cast<std::uint32_t>(props.minor);

        // Try to determine NUMA affinity (CUDA 11.0+)
#if CUDART_VERSION >= 11000
        int numa_id = -1;
        cudaDeviceGetAttribute(&numa_id, cudaDevAttrNumaId, i);
        if (numa_id >= 0) {
            gpu.numa_node = static_cast<std::uint32_t>(numa_id);

            // Update NUMA node's GPU list
            for (auto& node : topo.numa_nodes) {
                if (node.id == static_cast<std::uint32_t>(numa_id)) {
                    node.gpu_ids.push_back(gpu.id);
                    break;
                }
            }
        }
#endif

        topo.gpus.push_back(std::move(gpu));
    }
#endif  // DTL_ENABLE_CUDA
}

/// @brief Check if CUDA is available
/// @return true if CUDA runtime is available and has devices
[[nodiscard]] inline bool cuda_available() {
#if DTL_ENABLE_CUDA
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

/// @brief Get CUDA device count
/// @return Number of CUDA devices
[[nodiscard]] inline std::uint32_t cuda_device_count() {
#if DTL_ENABLE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        return static_cast<std::uint32_t>(count);
    }
#endif
    return 0;
}

}  // namespace dtl::topology::detail
