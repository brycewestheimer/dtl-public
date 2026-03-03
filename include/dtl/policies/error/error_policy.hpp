// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file error_policy.hpp
/// @brief Base error policy concept and interface
/// @details Defines how errors are reported and handled.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/status.hpp>

namespace dtl {

// Note: error_policy_tag is defined in dtl/core/traits.hpp
// Note: is_error_policy_v is defined in dtl/core/traits.hpp
// Note: ErrorPolicy concept is defined in dtl/core/concepts.hpp

/// @brief Error handling strategy
enum class error_strategy {
    return_result,   ///< Return result<T> with error
    throw_exception, ///< Throw exception on error
    terminate,       ///< Call std::terminate on error
    callback,        ///< Invoke user-provided callback
    ignore           ///< Ignore errors (unsafe)
};

/// @brief Traits for error policy inspection
template <typename Policy>
struct error_policy_traits {
    /// @brief Get the error handling strategy
    static constexpr error_strategy strategy = error_strategy::return_result;

    /// @brief Check if policy uses exceptions
    static constexpr bool uses_exceptions = false;

    /// @brief Check if errors can be ignored
    static constexpr bool can_ignore = false;
};

/// @brief Determine the return type based on error policy
/// @tparam Policy The error policy
/// @tparam T The success type
/// @tparam E The error type
template <typename Policy, typename T, typename E = status>
struct error_return_type {
    using type = T;  // Default: return T directly (throwing or terminating)
};

}  // namespace dtl
