// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file reduction_ops.hpp
/// @brief Reduction operation types for collective operations
/// @details Defines standard reduction operations (plus, min, max, etc.).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <functional>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace dtl {

// ============================================================================
// Reduction Operation Tags
// ============================================================================

/// @brief Tag for sum reduction
struct reduce_sum_tag {};

/// @brief Tag for product reduction
struct reduce_product_tag {};

/// @brief Tag for minimum reduction
struct reduce_min_tag {};

/// @brief Tag for maximum reduction
struct reduce_max_tag {};

/// @brief Tag for logical AND reduction
struct reduce_land_tag {};

/// @brief Tag for logical OR reduction
struct reduce_lor_tag {};

/// @brief Tag for bitwise AND reduction
struct reduce_band_tag {};

/// @brief Tag for bitwise OR reduction
struct reduce_bor_tag {};

/// @brief Tag for bitwise XOR reduction
struct reduce_bxor_tag {};

/// @brief Tag for min with location reduction
struct reduce_minloc_tag {};

/// @brief Tag for max with location reduction
struct reduce_maxloc_tag {};

// ============================================================================
// Reduction Operation Concept
// ============================================================================

/// @brief Concept for reduction operations
template <typename Op, typename T>
concept ReductionOp = requires(Op op, T a, T b) {
    { op(a, b) } -> std::convertible_to<T>;
};

// ============================================================================
// Standard Reduction Operations
// ============================================================================

/// @brief Sum reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_sum {
    using tag_type = reduce_sum_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a + b;
    }

    /// @brief Identity element for sum (0)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{0};
    }
};

/// @brief Transparent sum reduction
template <>
struct reduce_sum<void> {
    using tag_type = reduce_sum_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr auto operator()(T&& a, U&& b) const noexcept
        -> decltype(std::forward<T>(a) + std::forward<U>(b)) {
        return std::forward<T>(a) + std::forward<U>(b);
    }
};

/// @brief Product reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_product {
    using tag_type = reduce_product_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a * b;
    }

    /// @brief Identity element for product (1)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{1};
    }
};

/// @brief Transparent product reduction
template <>
struct reduce_product<void> {
    using tag_type = reduce_product_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr auto operator()(T&& a, U&& b) const noexcept
        -> decltype(std::forward<T>(a) * std::forward<U>(b)) {
        return std::forward<T>(a) * std::forward<U>(b);
    }
};

/// @brief Minimum reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_min {
    using tag_type = reduce_min_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return (std::min)(a, b);
    }

    /// @brief Identity element for min (max value)
    [[nodiscard]] static constexpr T identity() noexcept {
        if constexpr (std::numeric_limits<T>::has_infinity) {
            return std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::max();
        }
    }
};

/// @brief Transparent min reduction
template <>
struct reduce_min<void> {
    using tag_type = reduce_min_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr auto operator()(T&& a, U&& b) const noexcept
        -> std::common_type_t<T, U> {
        return (std::min)(std::forward<T>(a), std::forward<U>(b));
    }
};

/// @brief Maximum reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_max {
    using tag_type = reduce_max_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return (std::max)(a, b);
    }

    /// @brief Identity element for max (min value)
    [[nodiscard]] static constexpr T identity() noexcept {
        if constexpr (std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }
};

/// @brief Transparent max reduction
template <>
struct reduce_max<void> {
    using tag_type = reduce_max_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr auto operator()(T&& a, U&& b) const noexcept
        -> std::common_type_t<T, U> {
        return (std::max)(std::forward<T>(a), std::forward<U>(b));
    }
};

/// @brief Logical AND reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_land {
    using tag_type = reduce_land_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a && b;
    }

    /// @brief Identity element for logical AND (true)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{true};
    }
};

/// @brief Transparent logical AND reduction
template <>
struct reduce_land<void> {
    using tag_type = reduce_land_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr bool operator()(T&& a, U&& b) const noexcept {
        return std::forward<T>(a) && std::forward<U>(b);
    }
};

/// @brief Logical OR reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_lor {
    using tag_type = reduce_lor_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a || b;
    }

    /// @brief Identity element for logical OR (false)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{false};
    }
};

/// @brief Transparent logical OR reduction
template <>
struct reduce_lor<void> {
    using tag_type = reduce_lor_tag;
    using is_transparent = void;

    template <typename T, typename U>
    [[nodiscard]] constexpr bool operator()(T&& a, U&& b) const noexcept {
        return std::forward<T>(a) || std::forward<U>(b);
    }
};

/// @brief Bitwise AND reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_band {
    using tag_type = reduce_band_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a & b;
    }

    /// @brief Identity element for bitwise AND (all 1s)
    [[nodiscard]] static constexpr T identity() noexcept {
        return ~T{0};
    }
};

/// @brief Bitwise OR reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_bor {
    using tag_type = reduce_bor_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a | b;
    }

    /// @brief Identity element for bitwise OR (all 0s)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{0};
    }
};

/// @brief Bitwise XOR reduction operation
/// @tparam T Value type
template <typename T = void>
struct reduce_bxor {
    using tag_type = reduce_bxor_tag;

    /// @brief Apply the reduction
    [[nodiscard]] constexpr T operator()(const T& a, const T& b) const noexcept {
        return a ^ b;
    }

    /// @brief Identity element for bitwise XOR (all 0s)
    [[nodiscard]] static constexpr T identity() noexcept {
        return T{0};
    }
};

// ============================================================================
// Value-Location Pairs for Min/Max with Location
// ============================================================================

/// @brief Value with location for minloc/maxloc reductions
/// @tparam T Value type
/// @tparam L Location type
template <typename T, typename L = index_t>
struct value_location {
    /// @brief The value
    T value;

    /// @brief The location (index or rank)
    L location;

    /// @brief Default constructor
    value_location() = default;

    /// @brief Construct with value and location
    constexpr value_location(T v, L loc) noexcept : value(v), location(loc) {}
};

/// @brief Min-location reduction
/// @tparam T Value type
/// @tparam L Location type
template <typename T, typename L = index_t>
struct reduce_minloc {
    using tag_type = reduce_minloc_tag;
    using result_type = value_location<T, L>;

    /// @brief Apply the reduction (keep value with smaller location on tie)
    [[nodiscard]] constexpr result_type operator()(
        const result_type& a, const result_type& b) const noexcept {
        if (a.value < b.value) return a;
        if (b.value < a.value) return b;
        // Equal values: keep smaller location
        return (a.location < b.location) ? a : b;
    }
};

/// @brief Max-location reduction
/// @tparam T Value type
/// @tparam L Location type
template <typename T, typename L = index_t>
struct reduce_maxloc {
    using tag_type = reduce_maxloc_tag;
    using result_type = value_location<T, L>;

    /// @brief Apply the reduction (keep value with smaller location on tie)
    [[nodiscard]] constexpr result_type operator()(
        const result_type& a, const result_type& b) const noexcept {
        if (a.value > b.value) return a;
        if (b.value > a.value) return b;
        // Equal values: keep smaller location
        return (a.location < b.location) ? a : b;
    }
};

// ============================================================================
// Reduction Operation Traits
// ============================================================================

/// @brief Traits for reduction operations
template <typename Op>
struct reduction_op_traits {
    /// @brief Whether the operation is commutative
    static constexpr bool is_commutative = true;

    /// @brief Whether the operation is associative
    static constexpr bool is_associative = true;

    /// @brief Whether the operation has an identity element
    static constexpr bool has_identity = false;
};

/// @brief Traits specialization for sum
template <typename T>
struct reduction_op_traits<reduce_sum<T>> {
    static constexpr bool is_commutative = true;
    static constexpr bool is_associative = true;
    static constexpr bool has_identity = true;
};

/// @brief Traits specialization for product
template <typename T>
struct reduction_op_traits<reduce_product<T>> {
    static constexpr bool is_commutative = true;
    static constexpr bool is_associative = true;
    static constexpr bool has_identity = true;
};

/// @brief Traits specialization for min
template <typename T>
struct reduction_op_traits<reduce_min<T>> {
    static constexpr bool is_commutative = true;
    static constexpr bool is_associative = true;
    static constexpr bool has_identity = true;
};

/// @brief Traits specialization for max
template <typename T>
struct reduction_op_traits<reduce_max<T>> {
    static constexpr bool is_commutative = true;
    static constexpr bool is_associative = true;
    static constexpr bool has_identity = true;
};

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

/// @brief Alias for reduce_sum
using plus = reduce_sum<void>;

/// @brief Alias for reduce_product
using multiplies = reduce_product<void>;

/// @brief Alias for reduce_min
using minimum = reduce_min<void>;

/// @brief Alias for reduce_max
using maximum = reduce_max<void>;

}  // namespace dtl
