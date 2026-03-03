// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file backend_skip.hpp
 * @brief Test utilities for skipping tests based on backend availability
 * 
 * This header provides macros and functions for conditionally skipping tests
 * when required backends are not available. This ensures tests fail gracefully
 * with informative messages rather than crashing or hanging.
 * 
 * Usage:
 *   #include "dtl/test_util/backend_skip.hpp"
 * 
 *   TEST(CudaTest, Allocation) {
 *       DTL_SKIP_IF_NO_CUDA();
 *       // ... test code that requires CUDA
 *   }
 * 
 * @see docs/testing/test_matrix.md for full documentation
 */

#pragma once

#include <gtest/gtest.h>
#include <dtl/core/config.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

// =============================================================================
// Backend Availability Skip Macros
// =============================================================================

/**
 * @brief Skip test if CUDA is not available
 * 
 * Checks both compile-time flag and runtime device availability.
 */
#define DTL_SKIP_IF_NO_CUDA() \
    do { \
        if (!::dtl::test::cuda_available()) { \
            GTEST_SKIP() << "CUDA not available (not compiled with DTL_ENABLE_CUDA or no device found)"; \
        } \
    } while(0)

/**
 * @brief Skip test if HIP is not available
 * 
 * Checks both compile-time flag and runtime device availability.
 */
#define DTL_SKIP_IF_NO_HIP() \
    do { \
        if (!::dtl::test::hip_available()) { \
            GTEST_SKIP() << "HIP not available (not compiled with DTL_ENABLE_HIP or no device found)"; \
        } \
    } while(0)

/**
 * @brief Skip test if MPI is not available
 * 
 * Checks compile-time flag.
 */
#define DTL_SKIP_IF_NO_MPI() \
    do { \
        if (!::dtl::test::mpi_available()) { \
            GTEST_SKIP() << "MPI not available (not compiled with DTL_ENABLE_MPI)"; \
        } \
    } while(0)

/**
 * @brief Skip test if NCCL is not available
 */
#define DTL_SKIP_IF_NO_NCCL() \
    do { \
        if (!::dtl::test::nccl_available()) { \
            GTEST_SKIP() << "NCCL not available (not compiled with DTL_ENABLE_NCCL)"; \
        } \
    } while(0)

/**
 * @brief Skip test if OpenSHMEM is not available
 */
#define DTL_SKIP_IF_NO_SHMEM() \
    do { \
        if (!::dtl::test::shmem_available()) { \
            GTEST_SKIP() << "OpenSHMEM not available (not compiled with DTL_ENABLE_SHMEM)"; \
        } \
    } while(0)

// =============================================================================
// MPI Rank Skip Macros
// =============================================================================

/**
 * @brief Skip test if running with only a single rank
 * 
 * Use for tests that require inter-rank communication.
 */
#define DTL_SKIP_IF_SINGLE_RANK() \
    do { \
        if (::dtl::test::world_size() < 2) { \
            GTEST_SKIP() << "Test requires multiple ranks (n >= 2), got n=" \
                         << ::dtl::test::world_size(); \
        } \
    } while(0)

/**
 * @brief Skip test if fewer than n ranks are available
 * 
 * @param n Minimum number of ranks required
 */
#define DTL_REQUIRE_RANKS(n) \
    do { \
        if (::dtl::test::world_size() < (n)) { \
            GTEST_SKIP() << "Test requires at least " << (n) << " ranks, got " \
                         << ::dtl::test::world_size(); \
        } \
    } while(0)

/**
 * @brief Skip test if not running with exactly n ranks
 * 
 * @param n Exact number of ranks required
 */
#define DTL_REQUIRE_EXACT_RANKS(n) \
    do { \
        if (::dtl::test::world_size() != (n)) { \
            GTEST_SKIP() << "Test requires exactly " << (n) << " ranks, got " \
                         << ::dtl::test::world_size(); \
        } \
    } while(0)

// =============================================================================
// Resource Skip Macros
// =============================================================================

/**
 * @brief Skip test if insufficient GPU memory
 * 
 * @param bytes Minimum bytes of free GPU memory required
 */
#define DTL_SKIP_IF_INSUFFICIENT_GPU_MEMORY(bytes) \
    do { \
        if (!::dtl::test::has_gpu_memory(bytes)) { \
            GTEST_SKIP() << "Insufficient GPU memory: need " << (bytes) \
                         << " bytes"; \
        } \
    } while(0)

// =============================================================================
// Implementation
// =============================================================================

namespace dtl::test {

/**
 * @brief Check if CUDA is available at runtime
 * 
 * Returns true if:
 * - DTL was compiled with DTL_ENABLE_CUDA
 * - CUDA runtime is functional
 * - At least one CUDA device exists
 */
inline bool cuda_available() {
#if DTL_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

/**
 * @brief Check if HIP is available at runtime
 */
inline bool hip_available() {
#if DTL_ENABLE_HIP
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return err == hipSuccess && device_count > 0;
#else
    return false;
#endif
}

/**
 * @brief Check if MPI is available
 */
inline bool mpi_available() {
#if DTL_ENABLE_MPI
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if NCCL is available
 */
inline bool nccl_available() {
#if DTL_ENABLE_NCCL
    // NCCL requires CUDA
    return cuda_available();
#else
    return false;
#endif
}

/**
 * @brief Check if OpenSHMEM is available
 */
inline bool shmem_available() {
#if DTL_ENABLE_SHMEM
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get MPI world size
 * 
 * Returns 1 if MPI is not available or not initialized.
 */
inline int world_size() {
#if DTL_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return 1;
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
}

/**
 * @brief Get MPI world rank
 * 
 * Returns 0 if MPI is not available or not initialized.
 */
inline int world_rank() {
#if DTL_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return 0;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

/**
 * @brief Check if sufficient GPU memory is available
 * 
 * @param bytes Minimum bytes required
 * @return true if at least 'bytes' of GPU memory is free
 */
inline bool has_gpu_memory([[maybe_unused]] size_t bytes) {
#if DTL_ENABLE_CUDA
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    return err == cudaSuccess && free_bytes >= bytes;
#elif DTL_ENABLE_HIP
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
    return err == hipSuccess && free_bytes >= bytes;
#else
    return false;
#endif
}

/**
 * @brief Get number of available GPUs
 */
inline int gpu_count() {
#if DTL_ENABLE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        return count;
    }
    return 0;
#elif DTL_ENABLE_HIP
    int count = 0;
    if (hipGetDeviceCount(&count) == hipSuccess) {
        return count;
    }
    return 0;
#else
    return 0;
#endif
}

}  // namespace dtl::test
