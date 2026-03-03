// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_topology.h
 * @brief DTL C bindings - Topology query operations
 * @since 0.1.0
 *
 * This header provides C bindings for querying hardware topology
 * information, including CPU/GPU counts, affinity mappings,
 * and node locality.
 */

#ifndef DTL_TOPOLOGY_H
#define DTL_TOPOLOGY_H

#include "dtl_config.h"
#include "dtl_types.h"
#include "dtl_status.h"

DTL_C_BEGIN

/* ==========================================================================
 * CPU Topology
 * ========================================================================== */

/**
 * @brief Get the number of available CPUs/cores
 *
 * Returns the number of hardware threads (logical CPUs) available
 * on the current node. This uses std::thread::hardware_concurrency()
 * internally.
 *
 * @param[out] count Pointer to receive the CPU count
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre count must not be NULL
 * @post On success, *count contains the number of CPUs (may be 0 if
 *       the value cannot be determined)
 */
DTL_API dtl_status dtl_topology_num_cpus(int* count);

/**
 * @brief Get CPU affinity for a rank
 *
 * Returns a CPU identifier that the given rank should be bound to,
 * using a simple round-robin mapping: cpu_id = rank % num_cpus.
 *
 * @param rank The MPI rank to query
 * @param[out] cpu_id Pointer to receive the CPU identifier
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre cpu_id must not be NULL
 * @post On success, *cpu_id contains the CPU identifier (0-based)
 */
DTL_API dtl_status dtl_topology_cpu_affinity(dtl_rank_t rank, int* cpu_id);

/* ==========================================================================
 * GPU Topology
 * ========================================================================== */

/**
 * @brief Get the number of available GPU devices
 *
 * Returns the number of GPU devices visible on the current node.
 * Uses cudaGetDeviceCount when CUDA is enabled, hipGetDeviceCount
 * when HIP is enabled, or returns 0 when no GPU backend is available.
 *
 * @param[out] count Pointer to receive the GPU count
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre count must not be NULL
 * @post On success, *count contains the number of GPUs (0 if no
 *       GPU backend is enabled or no devices are found)
 */
DTL_API dtl_status dtl_topology_num_gpus(int* count);

/**
 * @brief Get GPU device for a rank
 *
 * Returns a GPU device identifier for the given rank, using a
 * simple round-robin mapping: gpu_id = rank % num_gpus. If no
 * GPU devices are available, returns -1.
 *
 * @param rank The MPI rank to query
 * @param[out] gpu_id Pointer to receive the GPU device identifier
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre gpu_id must not be NULL
 * @post On success, *gpu_id contains the GPU device identifier
 *       (0-based), or -1 if no GPUs are available
 */
DTL_API dtl_status dtl_topology_gpu_id(dtl_rank_t rank, int* gpu_id);

/* ==========================================================================
 * Node Locality
 * ========================================================================== */

/**
 * @brief Check if two ranks share a node
 *
 * Determines whether two MPI ranks are located on the same physical
 * node. When MPI is available, this uses MPI_Comm_split_type with
 * MPI_COMM_TYPE_SHARED to determine locality. Without MPI, all
 * ranks are assumed to be local (returns 1).
 *
 * @param rank_a First rank
 * @param rank_b Second rank
 * @param[out] is_local Pointer to receive the locality result
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre is_local must not be NULL
 * @post On success, *is_local is 1 if both ranks are on the same
 *       node, 0 otherwise
 *
 * @note This is a collective operation when MPI is enabled. All
 *       ranks in MPI_COMM_WORLD must call this function.
 */
DTL_API dtl_status dtl_topology_is_local(dtl_rank_t rank_a,
                                          dtl_rank_t rank_b,
                                          int* is_local);

/**
 * @brief Get node identifier for a rank
 *
 * Returns a node identifier for the given rank. Ranks on the same
 * physical node will have the same node identifier. When MPI is
 * available, this uses MPI_Comm_split_type with MPI_COMM_TYPE_SHARED
 * to determine the node grouping. Without MPI, all ranks are on
 * node 0.
 *
 * @param rank The MPI rank to query
 * @param[out] node_id Pointer to receive the node identifier
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre node_id must not be NULL
 * @post On success, *node_id contains the node identifier (0-based)
 *
 * @note This is a collective operation when MPI is enabled. All
 *       ranks in MPI_COMM_WORLD must call this function.
 */
DTL_API dtl_status dtl_topology_node_id(dtl_rank_t rank, int* node_id);

DTL_C_END

#endif /* DTL_TOPOLOGY_H */
