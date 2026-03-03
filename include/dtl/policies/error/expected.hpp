// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file expected.hpp
/// @brief Expected (result) error policy
/// @details Return result<T> containing either value or error.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/error/status.hpp>
#include <dtl/error/result.hpp>
#include <dtl/policies/error/error_policy.hpp>

namespace dtl {

/// @brief Expected (result-based) error policy
/// @details Operations return result<T> containing either the value on
///          success or an error status on failure. This enables monadic
///          error handling without exceptions.
///
/// @par Characteristics:
/// - No exceptions thrown
/// - Caller must check for errors
/// - Composable via map/and_then/or_else
/// - Zero overhead in success path
///
/// @par Usage Pattern:
/// @code
/// auto res = operation();
/// if (res.has_value()) {
///     use(res.value());
/// } else {
///     handle_error(res.error());
/// }
/// // Or using monadic operations:
/// operation()
///     .map([](auto v) { return transform(v); })
///     .or_else([](auto e) { return fallback(e); });
/// @endcode
///
/// @par Use Cases:
/// - Library code (caller decides error handling)
/// - Recoverable errors
/// - Functional programming style
struct expected_policy {
    /// @brief Policy category tag
    using policy_category = error_policy_tag;

    /// @brief Get the error handling strategy
    [[nodiscard]] static constexpr error_strategy strategy() noexcept {
        return error_strategy::return_result;
    }

    /// @brief Check if this policy uses exceptions
    [[nodiscard]] static constexpr bool uses_exceptions() noexcept {
        return false;
    }

    /// @brief Check if errors can be safely ignored
    [[nodiscard]] static constexpr bool can_ignore_errors() noexcept {
        return false;  // Errors should be checked
    }

    /// @brief Check if errors are propagated automatically
    [[nodiscard]] static constexpr bool auto_propagates() noexcept {
        return false;  // Caller must handle
    }

    /// @brief Handle an error by returning it in result
    /// @tparam T The expected success type
    /// @param s The error status
    /// @return result<T> containing the error
    template <typename T>
    [[nodiscard]] static result<T> handle_error(status s) {
        return result<T>::failure(std::move(s));
    }

    /// @brief Handle success by returning value in result
    /// @tparam T The success type
    /// @param value The success value
    /// @return result<T> containing the value
    template <typename T>
    [[nodiscard]] static result<T> handle_success(T value) {
        return result<T>::success(std::move(value));
    }
};

/// @brief Specialization of error_policy_traits for expected_policy
template <>
struct error_policy_traits<expected_policy> {
    static constexpr error_strategy strategy = error_strategy::return_result;
    static constexpr bool uses_exceptions = false;
    static constexpr bool can_ignore = false;
};

/// @brief Specialization of return type for expected_policy
template <typename T, typename E>
struct error_return_type<expected_policy, T, E> {
    using type = result<T>;  // Return result<T> instead of T
};

}  // namespace dtl
