// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file execution_policy.hpp
/// @brief Base execution policy concept and interface
/// @details Defines how operations are dispatched and executed.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

namespace dtl {

// Note: execution_policy_tag is defined in dtl/core/traits.hpp
// Note: is_execution_policy_v is defined in dtl/core/traits.hpp

/// @brief Execution mode for operations
enum class execution_mode {
    synchronous,     ///< Blocking execution
    asynchronous,    ///< Non-blocking execution
    deferred         ///< Lazy evaluation
};

/// @brief Parallelism level for execution
enum class parallelism_level {
    sequential,      ///< Single-threaded
    parallel,        ///< Multi-threaded (within rank)
    distributed,     ///< Across ranks
    heterogeneous    ///< CPU + GPU
};

// Note: ExecutionPolicy concept is defined in dtl/core/concepts.hpp
// using the is_execution_policy_v trait from dtl/core/traits.hpp.
// All execution policy types must:
// - Define policy_category as execution_policy_tag
// - Optionally provide mode() and is_blocking() static methods

/// @brief Traits for execution policy inspection
template <typename Policy>
struct execution_traits {
    /// @brief Check if execution is blocking
    static constexpr bool is_blocking = true;

    /// @brief Check if execution is parallel
    static constexpr bool is_parallel = false;

    /// @brief Get the execution mode
    static constexpr execution_mode mode = execution_mode::synchronous;

    /// @brief Get the parallelism level
    static constexpr parallelism_level parallelism = parallelism_level::sequential;
};

/// @brief Determine return type based on execution policy
/// @tparam Policy The execution policy
/// @tparam T The result type for synchronous execution
/// @tparam Future The future type for async execution
template <typename Policy, typename T, typename Future>
struct execution_return_type {
    using type = T;  // Default to synchronous return
};

}  // namespace dtl
