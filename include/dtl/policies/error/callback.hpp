// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file callback.hpp
/// @brief Callback error policy
/// @details Invoke user-provided callback on errors.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/error/status.hpp>
#include <dtl/error/result.hpp>
#include <dtl/policies/error/error_policy.hpp>

#include <functional>
#include <utility>

namespace dtl {

/// @brief Action to take after callback handles error
enum class error_action {
    continue_execution,  ///< Continue with default/fallback value
    propagate_error,     ///< Return/throw the error
    terminate            ///< Terminate the program
};

/// @brief Callback error policy
/// @tparam Handler Callback type: (status) -> error_action
/// @details Operations invoke user-provided callback on failure,
///          allowing flexible, context-specific error handling.
///
/// @par Characteristics:
/// - User-controlled error handling
/// - Can log, transform, or recover from errors
/// - Callback determines next action
/// - Flexible integration with existing error systems
///
/// @par Callback Contract:
/// The callback receives a status and returns an error_action:
/// - continue_execution: Use fallback value, continue
/// - propagate_error: Return error to caller
/// - terminate: Terminate the program
///
/// @par Usage Pattern:
/// @code
/// auto policy = dtl::callback_policy{[](dtl::status s) {
///     log_error(s);
///     if (can_recover(s)) {
///         return dtl::error_action::continue_execution;
///     }
///     return dtl::error_action::propagate_error;
/// }};
/// @endcode
///
/// @par Use Cases:
/// - Custom logging
/// - Error transformation
/// - Selective error recovery
/// - Integration with existing error frameworks
template <typename Handler>
struct callback_policy {
    /// @brief Policy category tag
    using policy_category = error_policy_tag;

    /// @brief The callback handler type
    using handler_type = Handler;

    /// @brief The error callback
    Handler handler;

    /// @brief Construct with handler
    /// @param h The error handler callback
    explicit callback_policy(Handler h) : handler{std::move(h)} {}

    /// @brief Get the error handling strategy
    [[nodiscard]] static constexpr error_strategy strategy() noexcept {
        return error_strategy::callback;
    }

    /// @brief Check if this policy uses exceptions
    [[nodiscard]] static constexpr bool uses_exceptions() noexcept {
        return false;  // Depends on callback implementation
    }

    /// @brief Check if errors can be safely ignored
    [[nodiscard]] static constexpr bool can_ignore_errors() noexcept {
        return true;  // Callback may choose to ignore
    }

    /// @brief Handle an error by invoking the callback
    /// @tparam T The expected success type
    /// @param s The error status
    /// @return result<T> based on callback's decision
    template <typename T>
    [[nodiscard]] result<T> handle_error(status s) const {
        error_action action = handler(s);

        switch (action) {
            case error_action::continue_execution:
                return result<T>::success(T{});  // Default value
            case error_action::propagate_error:
                return result<T>::failure(std::move(s));
            case error_action::terminate:
                std::terminate();
        }
        return result<T>::failure(std::move(s));  // Fallback
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

/// @brief Deduction guide for callback_policy
template <typename Handler>
callback_policy(Handler) -> callback_policy<Handler>;

/// @brief Factory function to create callback policy
/// @tparam Handler Callback type
/// @param h The error handler
/// @return callback_policy<Handler>
template <typename Handler>
[[nodiscard]] auto make_callback_policy(Handler&& h) {
    return callback_policy<std::decay_t<Handler>>{std::forward<Handler>(h)};
}

/// @brief Type-erased callback policy using std::function
using dynamic_callback_policy = callback_policy<std::function<error_action(status)>>;

/// @brief Specialization of error_policy_traits for callback_policy
template <typename Handler>
struct error_policy_traits<callback_policy<Handler>> {
    static constexpr error_strategy strategy = error_strategy::callback;
    static constexpr bool uses_exceptions = false;
    static constexpr bool can_ignore = true;
};

/// @brief Specialization of return type for callback_policy
template <typename Handler, typename T, typename E>
struct error_return_type<callback_policy<Handler>, T, E> {
    using type = result<T>;  // Return result<T>
};

}  // namespace dtl
