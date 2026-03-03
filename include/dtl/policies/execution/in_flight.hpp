// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file in_flight.hpp
/// @brief Concurrency limit hint for execution control
/// @details Limits the number of work units that may be in-flight simultaneously.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <limits>
#include <thread>

namespace dtl {

/// @brief Hint for maximum concurrent operations
/// @details Limits the number of work units that may be in-flight
///          simultaneously. Useful for:
///          - Controlling memory pressure from buffered operations
///          - Rate limiting for external resources
///          - Preventing thread oversubscription
///
/// @par Usage:
/// @code
/// // Limit to 4 concurrent operations
/// dtl::async_for_each(data, work_fn, dtl::in_flight{4});
/// @endcode
///
/// @par Memory Pressure Control:
/// @code
/// // Limit concurrent I/O to avoid buffer exhaustion
/// dtl::parallel_io(files, process_fn, dtl::in_flight{8});
/// @endcode
struct in_flight {
    /// @brief Maximum concurrent work units
    size_type max_concurrent;

    /// @brief Construct with specific limit
    constexpr explicit in_flight(size_type limit) noexcept
        : max_concurrent{limit} {}

    /// @brief Unlimited concurrency (default)
    [[nodiscard]] static constexpr in_flight unlimited() noexcept {
        return in_flight{std::numeric_limits<size_type>::max()};
    }

    /// @brief Single work unit at a time (sequential-like)
    [[nodiscard]] static constexpr in_flight sequential() noexcept {
        return in_flight{1};
    }

    /// @brief Match hardware thread count
    [[nodiscard]] static in_flight hardware_concurrency() noexcept {
        auto hw = std::thread::hardware_concurrency();
        return in_flight{hw > 0 ? static_cast<size_type>(hw) : 1};
    }

    /// @brief Conservative limit (half hardware threads)
    [[nodiscard]] static in_flight conservative() noexcept {
        auto hw = std::thread::hardware_concurrency();
        return in_flight{hw > 1 ? static_cast<size_type>(hw / 2) : 1};
    }

    /// @brief Aggressive limit (double hardware threads for I/O-bound work)
    [[nodiscard]] static in_flight aggressive() noexcept {
        auto hw = std::thread::hardware_concurrency();
        return in_flight{hw > 0 ? static_cast<size_type>(hw * 2) : 2};
    }

    /// @brief Check if this is unlimited
    [[nodiscard]] constexpr bool is_unlimited() const noexcept {
        return max_concurrent == std::numeric_limits<size_type>::max();
    }

    /// @brief Check if this is sequential (1)
    [[nodiscard]] constexpr bool is_sequential() const noexcept {
        return max_concurrent == 1;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const in_flight& other) const noexcept {
        return max_concurrent == other.max_concurrent;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const in_flight& other) const noexcept {
        return max_concurrent != other.max_concurrent;
    }
};

/// @brief Concept for types that can provide an in-flight hint
template <typename T>
concept InFlightHint = requires(T t) {
    { t.max_concurrent } -> std::convertible_to<size_type>;
    { t.is_unlimited() } -> std::convertible_to<bool>;
};

// Verify in_flight satisfies its own concept
static_assert(InFlightHint<in_flight>, "in_flight must satisfy InFlightHint");

}  // namespace dtl
