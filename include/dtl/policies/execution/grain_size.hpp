// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file grain_size.hpp
/// @brief Work unit size hint for throughput-oriented execution
/// @details Suggests the minimum number of elements to process per work unit.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>

namespace dtl {

/// @brief Hint for preferred work unit size
/// @details Suggests the minimum number of elements to process per work unit.
///          Backends may ignore this hint if not applicable.
///
/// @par Design Rationale:
/// - Larger grain sizes reduce scheduling overhead
/// - Smaller grain sizes improve load balancing
/// - Optimal size depends on work complexity and hardware
///
/// @par Usage:
/// @code
/// dtl::for_each(dtl::par, data, [](auto& x) { x *= 2; }, dtl::grain_size{1024});
/// @endcode
///
/// @par Integration with Batching Views:
/// @code
/// // Process in chunks matching grain size for optimal throughput
/// for (auto chunk : chunk_view(local_data, hint.value)) {
///     vectorized_process(chunk);
/// }
/// @endcode
struct grain_size {
    /// @brief The suggested work unit size
    size_type value;

    /// @brief Construct with specific grain size
    constexpr explicit grain_size(size_type size) noexcept : value{size} {}

    /// @brief Default grain size (auto-selected by backend)
    [[nodiscard]] static constexpr grain_size automatic() noexcept {
        return grain_size{0};  // 0 = auto
    }

    /// @brief Small grain (good for complex work per element)
    /// @details Recommended for operations with high per-element cost
    [[nodiscard]] static constexpr grain_size small() noexcept {
        return grain_size{64};
    }

    /// @brief Medium grain (balanced)
    /// @details Good default for moderate per-element work
    [[nodiscard]] static constexpr grain_size medium() noexcept {
        return grain_size{1024};
    }

    /// @brief Large grain (good for simple work per element)
    /// @details Recommended for trivial operations to minimize overhead
    [[nodiscard]] static constexpr grain_size large() noexcept {
        return grain_size{16384};
    }

    /// @brief GPU-friendly grain (warp-sized multiples)
    /// @details Multiple of typical GPU warp size (32)
    [[nodiscard]] static constexpr grain_size gpu_friendly() noexcept {
        return grain_size{256};  // 8 warps
    }

    /// @brief Cache-line friendly grain
    /// @details Good for memory-bound operations
    [[nodiscard]] static constexpr grain_size cache_friendly() noexcept {
        return grain_size{64};  // Typical cache line elements
    }

    /// @brief Check if this is automatic grain selection
    [[nodiscard]] constexpr bool is_automatic() const noexcept {
        return value == 0;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const grain_size& other) const noexcept {
        return value == other.value;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const grain_size& other) const noexcept {
        return value != other.value;
    }
};

/// @brief Concept for types that can provide a grain size hint
template <typename T>
concept GrainSizeHint = requires(T t) {
    { t.value } -> std::convertible_to<size_type>;
    { t.is_automatic() } -> std::convertible_to<bool>;
};

// Verify grain_size satisfies its own concept
static_assert(GrainSizeHint<grain_size>, "grain_size must satisfy GrainSizeHint");

}  // namespace dtl
