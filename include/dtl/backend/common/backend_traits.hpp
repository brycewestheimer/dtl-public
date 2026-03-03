// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file backend_traits.hpp
/// @brief Traits for backend capability detection
/// @details Provides compile-time queries for backend features.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <cstdint>
#include <type_traits>

namespace dtl {

/// @brief Backend implementation maturity classification
enum class backend_maturity : std::uint8_t {
    stub = 0,
    partial = 1,
    production = 2,
};

// ============================================================================
// Backend Feature Tags
// ============================================================================

/// @brief Tag indicating MPI support
struct mpi_backend_tag {};

/// @brief Tag indicating CUDA support
struct cuda_backend_tag {};

/// @brief Tag indicating HIP/ROCm support
struct hip_backend_tag {};

/// @brief Tag indicating SYCL support
struct sycl_backend_tag {};

/// @brief Tag indicating NCCL support
struct nccl_backend_tag {};

/// @brief Tag indicating shared memory support
struct shared_memory_backend_tag {};

/// @brief Tag indicating OpenSHMEM support
struct shmem_backend_tag {};

/// @brief Tag indicating UCX support
struct ucx_backend_tag {};

/// @brief Tag indicating GASNet-EX support
struct gasnet_backend_tag {};

// ============================================================================
// Backend Capability Traits
// ============================================================================

/// @brief Primary traits template for backend capabilities
/// @tparam Backend Backend type or tag
template <typename Backend>
struct backend_traits {
    /// @brief Whether backend supports point-to-point communication
    static constexpr bool supports_point_to_point = false;

    /// @brief Whether backend supports collective communication
    static constexpr bool supports_collectives = false;

    /// @brief Whether backend supports one-sided (RMA) communication
    static constexpr bool supports_rma = false;

    /// @brief Whether backend supports GPU-aware communication
    static constexpr bool supports_gpu_aware = false;

    /// @brief Whether backend supports asynchronous operations
    static constexpr bool supports_async = false;

    /// @brief Whether backend supports thread-multiple
    static constexpr bool supports_thread_multiple = false;

    /// @brief Whether backend supports RDMA
    static constexpr bool supports_rdma = false;

    /// @brief Backend name
    static constexpr const char* name = "unknown";

    /// @brief Backend implementation maturity
    static constexpr backend_maturity maturity = backend_maturity::stub;
};

// ============================================================================
// MPI Backend Traits
// ============================================================================

/// @brief Traits for MPI backend
template <>
struct backend_traits<mpi_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = true;
    static constexpr bool supports_gpu_aware = false;  // Depends on build
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = false;  // Depends on implementation
    static constexpr const char* name = "MPI";
    static constexpr backend_maturity maturity = backend_maturity::production;
};

// ============================================================================
// CUDA Backend Traits
// ============================================================================

/// @brief Traits for CUDA backend
template <>
struct backend_traits<cuda_backend_tag> {
    static constexpr bool supports_point_to_point = false;
    static constexpr bool supports_collectives = false;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = true;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "CUDA";
    static constexpr backend_maturity maturity = backend_maturity::partial;
};

// ============================================================================
// NCCL Backend Traits
// ============================================================================

/// @brief Traits for NCCL backend
template <>
struct backend_traits<nccl_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = true;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = false;
    static constexpr bool supports_rdma = true;  // NVLink, InfiniBand
    static constexpr const char* name = "NCCL";
    static constexpr backend_maturity maturity = backend_maturity::partial;
};

// ============================================================================
// HIP Backend Traits
// ============================================================================

/// @brief Traits for HIP/ROCm backend
template <>
struct backend_traits<hip_backend_tag> {
    static constexpr bool supports_point_to_point = false;
    static constexpr bool supports_collectives = false;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = true;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "HIP";
    static constexpr backend_maturity maturity = backend_maturity::partial;
};

// ============================================================================
// OpenSHMEM Backend Traits
// ============================================================================

/// @brief Traits for OpenSHMEM backend
template <>
struct backend_traits<shmem_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = true;
    static constexpr bool supports_gpu_aware = false;
    static constexpr bool supports_async = false;
    static constexpr bool supports_thread_multiple = false;
    static constexpr bool supports_rdma = true;  // Native RMA
    static constexpr const char* name = "SHMEM";
    static constexpr backend_maturity maturity = backend_maturity::production;
};

// ============================================================================
// Shared Memory Backend Traits
// ============================================================================

/// @brief Traits for shared memory backend
template <>
struct backend_traits<shared_memory_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = true;
    static constexpr bool supports_gpu_aware = false;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "SharedMemory";
    static constexpr backend_maturity maturity = backend_maturity::production;
};

// ============================================================================
// UCX Backend Traits
// ============================================================================

/// @brief Traits for UCX transport backend
template <>
struct backend_traits<ucx_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = true;
    static constexpr bool supports_gpu_aware = true;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = true;
    static constexpr const char* name = "UCX";
    static constexpr backend_maturity maturity = backend_maturity::stub;
};

// ============================================================================
// GASNet-EX Backend Traits
// ============================================================================

/// @brief Traits for GASNet-EX PGAS backend
template <>
struct backend_traits<gasnet_backend_tag> {
    static constexpr bool supports_point_to_point = true;
    static constexpr bool supports_collectives = true;
    static constexpr bool supports_rma = true;
    static constexpr bool supports_gpu_aware = false;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = false;
    static constexpr bool supports_rdma = true;
    static constexpr const char* name = "GASNet";
    static constexpr backend_maturity maturity = backend_maturity::stub;
};

// ============================================================================
// SYCL Backend Traits
// ============================================================================

/// @brief Traits for SYCL compute backend
template <>
struct backend_traits<sycl_backend_tag> {
    static constexpr bool supports_point_to_point = false;
    static constexpr bool supports_collectives = false;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = true;
    static constexpr bool supports_async = true;
    static constexpr bool supports_thread_multiple = true;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "SYCL";
    static constexpr backend_maturity maturity = backend_maturity::stub;
};

// ============================================================================
// Backend Detection
// ============================================================================

/// @brief Check if a backend supports a specific feature
/// @tparam Backend Backend type
/// @tparam Feature Feature to check for
template <typename Backend, typename Feature>
struct backend_supports : std::false_type {};

/// @brief Check if backend supports GPU-aware operations
template <typename Backend>
inline constexpr bool supports_gpu_aware_v = backend_traits<Backend>::supports_gpu_aware;

/// @brief Check if backend supports async operations
template <typename Backend>
inline constexpr bool supports_async_v = backend_traits<Backend>::supports_async;

/// @brief Check if backend supports collectives
template <typename Backend>
inline constexpr bool supports_collectives_v = backend_traits<Backend>::supports_collectives;

// ============================================================================
// Backend Combination Traits
// ============================================================================

/// @brief Traits for combined backends (e.g., MPI + CUDA)
/// @tparam Backends Backend types
template <typename... Backends>
struct combined_backend_traits {
    /// @brief All backends support point-to-point
    static constexpr bool supports_point_to_point =
        (backend_traits<Backends>::supports_point_to_point && ...);

    /// @brief All backends support collectives
    static constexpr bool supports_collectives =
        (backend_traits<Backends>::supports_collectives && ...);

    /// @brief Any backend supports GPU-aware
    static constexpr bool supports_gpu_aware =
        (backend_traits<Backends>::supports_gpu_aware || ...);

    /// @brief All backends support async
    static constexpr bool supports_async =
        (backend_traits<Backends>::supports_async && ...);
};

// ============================================================================
// Backend Selection Utilities
// ============================================================================

/// @brief Select best available backend for an operation
/// @details Primary template yields void. Specialize for concrete backend
///          combinations to provide meaningful selection logic.
/// @tparam PreferGpu Whether to prefer GPU backends
/// @tparam Backends Available backends
template <bool PreferGpu, typename... Backends>
struct select_backend {
    static_assert(sizeof...(Backends) == 0,
                  "No select_backend specialization found for the given backend set. "
                  "Provide a specialization of select_backend for your backend combination.");
    using type = void;
};

// ============================================================================
// Concept Verification — Backend Traits
// ============================================================================

// Verify all backend_traits specializations provide expected fields
static_assert(backend_traits<mpi_backend_tag>::supports_collectives,
              "MPI should support collectives");
static_assert(backend_traits<cuda_backend_tag>::supports_gpu_aware,
              "CUDA should support GPU-aware operations");
static_assert(backend_traits<hip_backend_tag>::supports_gpu_aware,
              "HIP should support GPU-aware operations");
static_assert(backend_traits<shmem_backend_tag>::supports_rma,
              "SHMEM should support RMA operations");
static_assert(backend_traits<nccl_backend_tag>::supports_collectives,
              "NCCL should support collectives");

/// @brief Default communicator backend
#if defined(DTL_HAS_MPI)
using default_communicator_backend = mpi_backend_tag;
#else
using default_communicator_backend = shared_memory_backend_tag;
#endif

/// @brief Default executor backend
#if defined(DTL_HAS_CUDA)
using default_gpu_backend = cuda_backend_tag;
#elif defined(DTL_HAS_HIP)
using default_gpu_backend = hip_backend_tag;
#else
using default_gpu_backend = void;
#endif

}  // namespace dtl
