// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_topology.cpp
 * @brief DTL C bindings - Topology query implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_topology.h>

#include "dtl_internal.hpp"

#include <thread>
#include <vector>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

// ============================================================================
// CPU Topology
// ============================================================================

extern "C" {

dtl_status dtl_topology_num_cpus(int* count) {
    if (!count) {
        return DTL_ERROR_NULL_POINTER;
    }

    unsigned int hw = std::thread::hardware_concurrency();
    *count = static_cast<int>(hw);
    return DTL_SUCCESS;
}

dtl_status dtl_topology_cpu_affinity(dtl_rank_t rank, int* cpu_id) {
    if (!cpu_id) {
        return DTL_ERROR_NULL_POINTER;
    }

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        // Cannot determine CPU count; default to CPU 0
        *cpu_id = 0;
        return DTL_SUCCESS;
    }

    *cpu_id = static_cast<int>(rank % static_cast<dtl_rank_t>(hw));
    return DTL_SUCCESS;
}

// ============================================================================
// GPU Topology
// ============================================================================

dtl_status dtl_topology_num_gpus(int* count) {
    if (!count) {
        return DTL_ERROR_NULL_POINTER;
    }

#if DTL_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        *count = 0;
        return DTL_SUCCESS;
    }
    *count = device_count;
    return DTL_SUCCESS;
#elif DTL_ENABLE_HIP
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        *count = 0;
        return DTL_SUCCESS;
    }
    *count = device_count;
    return DTL_SUCCESS;
#else
    *count = 0;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_topology_gpu_id(dtl_rank_t rank, int* gpu_id) {
    if (!gpu_id) {
        return DTL_ERROR_NULL_POINTER;
    }

    int num_gpus = 0;
    dtl_status status = dtl_topology_num_gpus(&num_gpus);
    if (status != DTL_SUCCESS) {
        *gpu_id = -1;
        return status;
    }

    if (num_gpus <= 0) {
        *gpu_id = -1;
        return DTL_SUCCESS;
    }

    *gpu_id = static_cast<int>(rank % static_cast<dtl_rank_t>(num_gpus));
    return DTL_SUCCESS;
}

// ============================================================================
// Node Locality
// ============================================================================

dtl_status dtl_topology_is_local(dtl_rank_t rank_a, dtl_rank_t rank_b,
                                  int* is_local) {
    if (!is_local) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    // Split communicator by shared memory to determine node locality
    MPI_Comm shared_comm;
    int err = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  0, MPI_INFO_NULL, &shared_comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COMMUNICATION;
    }

    // Get the group of ranks sharing memory with us
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int shared_size;
    MPI_Comm_size(shared_comm, &shared_size);

    // Gather all world ranks in this shared communicator
    int shared_rank;
    MPI_Comm_rank(shared_comm, &shared_rank);

    std::vector<int> world_ranks(shared_size);
    MPI_Allgather(&world_rank, 1, MPI_INT,
                  world_ranks.data(), 1, MPI_INT, shared_comm);

    // Check if both rank_a and rank_b are in the same shared group
    bool found_a = false;
    bool found_b = false;
    for (int i = 0; i < shared_size; ++i) {
        if (world_ranks[i] == rank_a) found_a = true;
        if (world_ranks[i] == rank_b) found_b = true;
    }

    *is_local = (found_a && found_b) ? 1 : 0;

    MPI_Comm_free(&shared_comm);
    return DTL_SUCCESS;
#else
    (void)rank_a;
    (void)rank_b;
    // Without MPI, all ranks are local (single process)
    *is_local = 1;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_topology_node_id(dtl_rank_t rank, int* node_id) {
    if (!node_id) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    // Split communicator by shared memory to determine node grouping
    MPI_Comm shared_comm;
    int err = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  0, MPI_INFO_NULL, &shared_comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COMMUNICATION;
    }

    // The lowest world rank in each shared communicator serves as the node ID
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int shared_size;
    MPI_Comm_size(shared_comm, &shared_size);

    // Gather all world ranks in this shared communicator
    std::vector<int> world_ranks(shared_size);
    MPI_Allgather(&world_rank, 1, MPI_INT,
                  world_ranks.data(), 1, MPI_INT, shared_comm);

    // Find the minimum world rank in the shared group (node leader)
    int min_rank = world_ranks[0];
    for (int i = 1; i < shared_size; ++i) {
        if (world_ranks[i] < min_rank) {
            min_rank = world_ranks[i];
        }
    }

    // Check if the queried rank is in our shared group
    bool found = false;
    for (int i = 0; i < shared_size; ++i) {
        if (world_ranks[i] == rank) {
            found = true;
            break;
        }
    }

    if (found) {
        *node_id = min_rank;
    } else {
        // The queried rank is not on our node; we cannot determine its
        // node ID from this process alone. Return -1 to indicate unknown.
        *node_id = -1;
    }

    MPI_Comm_free(&shared_comm);
    return DTL_SUCCESS;
#else
    (void)rank;
    // Without MPI, single process: node 0
    *node_id = 0;
    return DTL_SUCCESS;
#endif
}

}  // extern "C"
