// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file affinity.hpp
/// @brief Placement affinity hint for execution
/// @details Suggests where work should preferentially execute (NUMA, GPU, etc.).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>

namespace dtl {

/// @brief Hint for work placement affinity
/// @details Suggests where work should preferentially execute.
///          Backends may ignore this hint if not applicable.
///
/// @par Design Rationale:
/// Affinity hints enable:
/// - NUMA-aware execution
/// - GPU placement control
/// - Data locality optimization
/// - Cross-platform portable hints
///
/// @par Usage:
/// @code
/// // Prefer executing near the data
/// dtl::for_each(dtl::par, data, work_fn, dtl::affinity::data_local());
///
/// // Execute on specific GPU
/// dtl::for_each(dtl::par, data, work_fn, dtl::affinity::device(0));
/// @endcode
struct affinity {
    /// @brief Affinity mode
    enum class mode {
        none,           ///< No affinity preference
        data_local,     ///< Execute near data (NUMA-aware)
        compute_local,  ///< Execute near compute resources (GPU-aware)
        balanced        ///< Balance between data and compute locality
    };

    /// @brief The affinity mode
    mode preference;

    /// @brief Optional specific placement target (device ID, NUMA node, etc.)
    /// -1 means no specific target
    int target;

    /// @brief Construct with mode only
    constexpr explicit affinity(mode m) noexcept
        : preference{m}
        , target{-1} {}

    /// @brief Construct with mode and specific target
    constexpr affinity(mode m, int t) noexcept
        : preference{m}
        , target{t} {}

    // =========================================================================
    // Factory Functions
    // =========================================================================

    /// @brief No affinity preference
    [[nodiscard]] static constexpr affinity none() noexcept {
        return affinity{mode::none};
    }

    /// @brief Prefer data locality (NUMA-aware)
    [[nodiscard]] static constexpr affinity data_local() noexcept {
        return affinity{mode::data_local};
    }

    /// @brief Prefer compute locality (GPU-aware)
    [[nodiscard]] static constexpr affinity compute_local() noexcept {
        return affinity{mode::compute_local};
    }

    /// @brief Balanced locality between data and compute
    [[nodiscard]] static constexpr affinity balanced() noexcept {
        return affinity{mode::balanced};
    }

    /// @brief Specific GPU device target
    /// @param device_id The GPU device ID to target
    [[nodiscard]] static constexpr affinity device(int device_id) noexcept {
        return affinity{mode::compute_local, device_id};
    }

    /// @brief Specific NUMA node target
    /// @param node_id The NUMA node ID to target
    [[nodiscard]] static constexpr affinity numa_node(int node_id) noexcept {
        return affinity{mode::data_local, node_id};
    }

    /// @brief Specific CPU socket target
    /// @param socket_id The CPU socket ID to target
    [[nodiscard]] static constexpr affinity socket(int socket_id) noexcept {
        return affinity{mode::compute_local, socket_id};
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// @brief Check if a specific target is set
    [[nodiscard]] constexpr bool has_target() const noexcept {
        return target >= 0;
    }

    /// @brief Check if this is the "no preference" mode
    [[nodiscard]] constexpr bool is_none() const noexcept {
        return preference == mode::none;
    }

    /// @brief Check if this prefers data locality
    [[nodiscard]] constexpr bool is_data_local() const noexcept {
        return preference == mode::data_local;
    }

    /// @brief Check if this prefers compute locality
    [[nodiscard]] constexpr bool is_compute_local() const noexcept {
        return preference == mode::compute_local;
    }

    /// @brief Check if this is balanced
    [[nodiscard]] constexpr bool is_balanced() const noexcept {
        return preference == mode::balanced;
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const affinity& other) const noexcept {
        return preference == other.preference && target == other.target;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const affinity& other) const noexcept {
        return !(*this == other);
    }
};

/// @brief Concept for types that can provide an affinity hint
template <typename T>
concept AffinityHint = requires(T t) {
    { t.preference } -> std::convertible_to<affinity::mode>;
    { t.target } -> std::convertible_to<int>;
    { t.has_target() } -> std::convertible_to<bool>;
};

// Verify affinity satisfies its own concept
static_assert(AffinityHint<affinity>, "affinity must satisfy AffinityHint");

}  // namespace dtl
