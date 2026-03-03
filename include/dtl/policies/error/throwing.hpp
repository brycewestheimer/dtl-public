// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file throwing.hpp
/// @brief Throwing error policy
/// @details Throw exceptions on errors.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/error/status.hpp>
#include <dtl/policies/error/error_policy.hpp>

#include <stdexcept>
#include <string>

namespace dtl {

/// @brief Exception thrown by DTL operations
/// @details Contains the error status and provides a descriptive message.
class dtl_exception : public std::runtime_error {
public:
    /// @brief Construct from status
    /// @param s The error status
    explicit dtl_exception(status s)
        : std::runtime_error(s.message())
        , status_{std::move(s)} {}

    /// @brief Get the error status
    [[nodiscard]] const status& error_status() const noexcept {
        return status_;
    }

    /// @brief Get the status code
    [[nodiscard]] status_code code() const noexcept {
        return status_.code();
    }

private:
    status status_;
};

/// @brief Throwing error policy
/// @details Operations throw exceptions on failure. Success returns
///          the value directly without wrapping.
///
/// @par Characteristics:
/// - Exceptions thrown on error
/// - Clean success path (direct value return)
/// - Automatic stack unwinding on error
/// - Natural control flow for error handling
///
/// @par Usage Pattern:
/// @code
/// try {
///     auto value = operation();
///     use(value);
/// } catch (const dtl::dtl_exception& e) {
///     handle_error(e.error_status());
/// }
/// @endcode
///
/// @warning Not suitable for GPU code paths or performance-critical
///          error handling in tight loops.
///
/// @par Use Cases:
/// - Application code
/// - Simple error handling
/// - When errors are truly exceptional
struct throwing_policy {
    /// @brief Policy category tag
    using policy_category = error_policy_tag;

    /// @brief Get the error handling strategy
    [[nodiscard]] static constexpr error_strategy strategy() noexcept {
        return error_strategy::throw_exception;
    }

    /// @brief Check if this policy uses exceptions
    [[nodiscard]] static constexpr bool uses_exceptions() noexcept {
        return true;
    }

    /// @brief Check if errors can be safely ignored
    [[nodiscard]] static constexpr bool can_ignore_errors() noexcept {
        return false;  // Exceptions propagate automatically
    }

    /// @brief Check if errors are propagated automatically
    [[nodiscard]] static constexpr bool auto_propagates() noexcept {
        return true;  // Exceptions propagate up the stack
    }

    /// @brief Handle an error by throwing an exception
    /// @param s The error status
    /// @throws dtl_exception Always throws
    [[noreturn]] static void handle_error(status s) {
        throw dtl_exception{std::move(s)};
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

/// @brief Specialization of error_policy_traits for throwing_policy
template <>
struct error_policy_traits<throwing_policy> {
    static constexpr error_strategy strategy = error_strategy::throw_exception;
    static constexpr bool uses_exceptions = true;
    static constexpr bool can_ignore = false;
};

/// @brief Specialization of return type for throwing_policy
template <typename T, typename E>
struct error_return_type<throwing_policy, T, E> {
    using type = T;  // Return T directly (throws on error)
};

}  // namespace dtl
