// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file seq.hpp
/// @brief Sequential (synchronous) execution policy
/// @details Single-threaded, blocking execution for deterministic behavior.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

namespace dtl {

/// @brief Sequential execution policy
/// @details Operations execute synchronously on a single thread.
///          This is the simplest execution policy and provides
///          deterministic, predictable behavior.
///
/// @par Characteristics:
/// - Single-threaded execution
/// - Blocking (synchronous) completion
/// - Deterministic order of operations
/// - Easiest to debug and reason about
///
/// @par Use Cases:
/// - Debugging and development
/// - Small data sizes where parallelism overhead isn't worth it
/// - Operations with complex dependencies
struct seq {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

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
        return false;
    }

    /// @brief Get the parallelism level
    [[nodiscard]] static constexpr parallelism_level parallelism() noexcept {
        return parallelism_level::sequential;
    }

    /// @brief Check if vectorization is allowed
    [[nodiscard]] static constexpr bool allows_vectorization() noexcept {
        return true;  // SIMD within single thread is fine
    }

    /// @brief Check if order is deterministic
    [[nodiscard]] static constexpr bool is_deterministic() noexcept {
        return true;
    }
};

/// @brief Specialization of execution_traits for seq
template <>
struct execution_traits<seq> {
    static constexpr bool is_blocking = true;
    static constexpr bool is_parallel = false;
    static constexpr execution_mode mode = execution_mode::synchronous;
    static constexpr parallelism_level parallelism = parallelism_level::sequential;
};

}  // namespace dtl
