// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_exec.hpp
/// @brief CUDA GPU execution policy
/// @details Defines execution policy for GPU-accelerated algorithm execution
///          using CUDA streams. Works with Thrust-based algorithm implementations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {

// ============================================================================
// CUDA Execution Policy
// ============================================================================

/// @brief CUDA GPU execution policy
/// @details Executes algorithms on GPU using CUDA streams.
///          Operations are asynchronous by default with respect to host code,
///          but ordered within the same stream.
///
/// @par Usage:
/// @code
/// dtl::cuda_exec policy{stream};
/// dtl::for_each(policy, container, [](auto& x) { x *= 2; });
/// policy.synchronize();  // Wait for GPU completion
/// @endcode
///
/// @par Memory Requirements:
/// Data must be accessible from GPU. Use placement policies:
/// - `device_only<DeviceId>` - Optimal GPU performance
/// - `unified_memory` - Automatic page migration between CPU/GPU
/// - `device_preferred` - GPU with CPU fallback
///
/// @par Functor Requirements:
/// Functors passed to GPU algorithms must be `__device__` callable:
/// - Lambda functions: Use `__device__` or `__host__ __device__` specifiers
/// - Functor objects: Member `operator()` must be `__device__` callable
/// - Thrust functors: `thrust::plus<T>`, `thrust::negate<T>`, etc.
struct cuda_exec {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

#if DTL_ENABLE_CUDA
    /// @brief CUDA stream for execution ordering
    cudaStream_t stream = 0;

    /// @brief Default constructor (uses default stream)
    cuda_exec() = default;

    /// @brief Construct with specific stream
    /// @param s CUDA stream for ordered execution
    explicit cuda_exec(cudaStream_t s) : stream(s) {}

    /// @brief Synchronize this policy's stream
    /// @details Blocks until all operations on the stream complete
    void synchronize() const {
        cudaStreamSynchronize(stream);
    }

    /// @brief Query if stream is idle
    /// @return true if all operations have completed
    [[nodiscard]] bool is_idle() const {
        return cudaStreamQuery(stream) == cudaSuccess;
    }

    /// @brief Get the CUDA stream
    [[nodiscard]] cudaStream_t get_stream() const noexcept { return stream; }
#else
    /// @brief Default constructor
    cuda_exec() = default;

    /// @brief Synchronize (no-op when CUDA disabled)
    void synchronize() const {}

    /// @brief Query if idle (always true when CUDA disabled)
    [[nodiscard]] bool is_idle() const { return true; }
#endif

    /// @brief Get execution mode (asynchronous for GPU)
    [[nodiscard]] static constexpr execution_mode mode() noexcept {
        return execution_mode::asynchronous;
    }

    /// @brief Check if execution blocks the host
    [[nodiscard]] static constexpr bool is_blocking() noexcept {
        return false;  // GPU execution is non-blocking
    }

    /// @brief Check if this is device execution
    [[nodiscard]] static constexpr bool is_device_execution() noexcept {
        return true;
    }
};

// ============================================================================
// CUDA Execution Traits
// ============================================================================

/// @brief Execution traits for cuda_exec
template <>
struct execution_traits<cuda_exec> {
    /// @brief CUDA execution is non-blocking
    static constexpr bool is_blocking = false;

    /// @brief CUDA execution is parallel
    static constexpr bool is_parallel = true;

    /// @brief Execution mode is asynchronous
    static constexpr execution_mode mode = execution_mode::asynchronous;

    /// @brief Parallelism includes GPU (heterogeneous)
    static constexpr parallelism_level parallelism = parallelism_level::heterogeneous;
};

// ============================================================================
// Policy Type Detection
// ============================================================================

/// @brief Type trait to detect cuda_exec policy
template <typename T>
struct is_cuda_exec_policy : std::false_type {};

template <>
struct is_cuda_exec_policy<cuda_exec> : std::true_type {};

/// @brief Helper variable template
template <typename T>
inline constexpr bool is_cuda_exec_policy_v = is_cuda_exec_policy<std::decay_t<T>>::value;

// ============================================================================
// Device Execution Concept
// ============================================================================

/// @brief Concept for device execution policies
template <typename Policy>
concept DeviceExecutionPolicy = requires {
    { Policy::is_device_execution() } -> std::same_as<bool>;
    requires Policy::is_device_execution();
};

// Verify cuda_exec satisfies the concept
static_assert(DeviceExecutionPolicy<cuda_exec>,
              "cuda_exec must satisfy DeviceExecutionPolicy concept");

}  // namespace dtl
