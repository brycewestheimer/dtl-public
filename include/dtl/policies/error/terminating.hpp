// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file terminating.hpp
/// @brief Terminating error policy
/// @details Call std::terminate on errors (HPC fail-fast style).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/error/status.hpp>
#include <dtl/policies/error/error_policy.hpp>

#include <cstdlib>
#include <iostream>

namespace dtl {

/// @brief Terminating error policy
/// @details Operations call std::terminate on failure. This is the
///          traditional HPC "fail-fast" approach where errors indicate
///          unrecoverable situations.
///
/// @par Characteristics:
/// - No exceptions
/// - No error return values to check
/// - Program terminates on any error
/// - Simplest error "handling"
///
/// @par Rationale:
/// In HPC environments, errors often indicate fundamental problems
/// (network failure, memory exhaustion) where recovery is impossible
/// or not worth the complexity. Failing fast provides:
/// - Clear point of failure for debugging
/// - No hidden error propagation
/// - Deterministic behavior
///
/// @warning This policy terminates the entire program, including all
///          MPI ranks. Not suitable for applications requiring graceful
///          error recovery.
///
/// @par Use Cases:
/// - HPC batch jobs
/// - Development/debugging
/// - When errors are truly unrecoverable
struct terminating_policy {
    /// @brief Policy category tag
    using policy_category = error_policy_tag;

    /// @brief Get the error handling strategy
    [[nodiscard]] static constexpr error_strategy strategy() noexcept {
        return error_strategy::terminate;
    }

    /// @brief Check if this policy uses exceptions
    [[nodiscard]] static constexpr bool uses_exceptions() noexcept {
        return false;
    }

    /// @brief Check if errors can be safely ignored
    [[nodiscard]] static constexpr bool can_ignore_errors() noexcept {
        return false;  // Errors terminate the program
    }

    /// @brief Check if errors are propagated automatically
    [[nodiscard]] static constexpr bool auto_propagates() noexcept {
        return false;  // Terminates instead of propagating
    }

    /// @brief Handle an error by terminating the program
    /// @param s The error status
    /// @note Prints error message to stderr before terminating
    [[noreturn]] static void handle_error(status s) {
        std::cerr << "DTL FATAL ERROR: " << s.message() << "\n";
        std::cerr << "  Code: " << static_cast<int>(s.code()) << "\n";
        std::terminate();
    }

    /// @brief Handle success by returning the value directly
    /// @tparam T The success type
    /// @param value The success value
    /// @return The value directly
    template <typename T>
    [[nodiscard]] static T handle_success(T value) {
        return value;
    }
};

/// @brief Specialization of error_policy_traits for terminating_policy
template <>
struct error_policy_traits<terminating_policy> {
    static constexpr error_strategy strategy = error_strategy::terminate;
    static constexpr bool uses_exceptions = false;
    static constexpr bool can_ignore = false;
};

/// @brief Specialization of return type for terminating_policy
template <typename T, typename E>
struct error_return_type<terminating_policy, T, E> {
    using type = T;  // Return T directly (terminates on error)
};

}  // namespace dtl
