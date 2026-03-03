// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file par.hpp
/// @brief Parallel execution policy
/// @details Multi-threaded execution with blocking completion.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

namespace dtl {

/// @brief Parallel execution policy
/// @details Operations execute in parallel across multiple threads,
///          but the call blocks until all parallel work completes.
///
/// @par Characteristics:
/// - Multi-threaded execution within a rank
/// - Blocking (synchronous) completion
/// - Non-deterministic order of element processing
/// - Utilizes available CPU cores
///
/// @par Thread Safety:
/// - Element access is thread-safe (different elements)
/// - User-provided functions must be thread-safe
/// - Reductions use thread-local accumulation
///
/// @par Use Cases:
/// - CPU-bound operations on large data
/// - Independent element operations (map, filter)
/// - Parallel reductions
struct par {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

    /// @brief Get the execution mode
    [[nodiscard]] static constexpr execution_mode mode() noexcept {
        return execution_mode::synchronous;  // Blocks until complete
    }

    /// @brief Check if execution is blocking
    [[nodiscard]] static constexpr bool is_blocking() noexcept {
        return true;
    }

    /// @brief Check if execution is parallel
    [[nodiscard]] static constexpr bool is_parallel() noexcept {
        return true;
    }

    /// @brief Get the parallelism level
    [[nodiscard]] static constexpr parallelism_level parallelism() noexcept {
        return parallelism_level::parallel;
    }

    /// @brief Check if vectorization is allowed
    [[nodiscard]] static constexpr bool allows_vectorization() noexcept {
        return true;
    }

    /// @brief Check if order is deterministic
    [[nodiscard]] static constexpr bool is_deterministic() noexcept {
        return false;  // Order of element processing is not guaranteed
    }

    /// @brief Get the number of threads to use
    /// @return Number of hardware threads (0 = auto-detect)
    [[nodiscard]] static unsigned int num_threads() noexcept {
        return 0;  // 0 means use hardware concurrency
    }
};

/// @brief Parallel policy with explicit thread count
/// @tparam N Number of threads (0 = auto)
template <unsigned int N = 0>
struct par_n {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

    /// @brief Number of threads
    static constexpr unsigned int thread_count = N;

    /// @brief Get the execution mode
    [[nodiscard]] static constexpr execution_mode mode() noexcept {
        return execution_mode::synchronous;
    }

    /// @brief Check if execution is blocking
    [[nodiscard]] static constexpr bool is_blocking() noexcept {
        return true;
    }

    /// @brief Check if execution is parallel
    [[nodiscard]] static constexpr bool is_parallel() noexcept {
        return true;
    }

    /// @brief Get the number of threads
    [[nodiscard]] static constexpr unsigned int num_threads() noexcept {
        return N;
    }
};

/// @brief Specialization of execution_traits for par
template <>
struct execution_traits<par> {
    static constexpr bool is_blocking = true;
    static constexpr bool is_parallel = true;
    static constexpr execution_mode mode = execution_mode::synchronous;
    static constexpr parallelism_level parallelism = parallelism_level::parallel;
};

}  // namespace dtl
