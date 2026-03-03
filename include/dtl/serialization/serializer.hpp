// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file serializer.hpp
/// @brief Primary serializer trait template for custom type serialization
/// @details Defines the serializer trait that must be specialized for
///          non-trivially-serializable types that need to be transported.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/concepts.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

namespace dtl {

// =============================================================================
// Forward Declarations for SFINAE
// =============================================================================

// Helper to detect std::array
template <typename T>
struct is_std_array : std::false_type {};

template <typename T, size_type N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;

// Helper to detect std::pair
template <typename T>
struct is_std_pair : std::false_type {};

template <typename T1, typename T2>
struct is_std_pair<std::pair<T1, T2>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_pair_v = is_std_pair<T>::value;

// Forward declaration for std::optional detection
template <typename T>
struct is_std_optional : std::false_type {};

template <typename T>
inline constexpr bool is_std_optional_v = is_std_optional<T>::value;

// =============================================================================
// Serializer Primary Template
// =============================================================================

/// @brief Primary serializer trait template
/// @tparam T The type to serialize
/// @tparam Enable SFINAE enabler for specializations
/// @details To make a custom type transportable, specialize this template
///          with the required static member functions.
///
/// @par Required interface for specializations:
/// @code
/// template <>
/// struct serializer<MyType> {
///     // Return serialized size in bytes (may depend on value)
///     static size_type serialized_size(const MyType& value);
///
///     // Serialize value to buffer, return bytes written
///     static size_type serialize(const MyType& value, std::byte* buffer);
///
///     // Deserialize from buffer, return value
///     static MyType deserialize(const std::byte* buffer, size_type size);
/// };
/// @endcode
template <typename T, typename Enable = void>
struct serializer {
    // Primary template is not defined - must be specialized
    static_assert(sizeof(T) == 0,
                  "No serializer defined for type T. "
                  "Either T must be trivially serializable, or you must "
                  "provide a specialization of dtl::serializer<T>.");
};

// =============================================================================
// Trivially Serializable Specialization
// =============================================================================

/// @brief Serializer specialization for trivially serializable types
/// @tparam T A trivially copyable, standard layout type
/// @details Trivially serializable types can be memcpy'd directly.
///          std::array and std::pair are handled by separate specializations.
template <typename T>
struct serializer<T, std::enable_if_t<is_trivially_serializable_v<T> &&
                                       !is_std_array_v<T> &&
                                       !is_std_pair_v<T> &&
                                       !is_std_optional_v<T>>> {
    /// @brief Get serialized size (always sizeof(T) for trivial types)
    /// @param value The value (ignored for trivial types)
    /// @return Size in bytes
    [[nodiscard]] static constexpr size_type serialized_size(const T& value) noexcept {
        (void)value;
        return sizeof(T);
    }

    /// @brief Get serialized size without an instance
    /// @return Size in bytes
    [[nodiscard]] static constexpr size_type serialized_size() noexcept {
        return sizeof(T);
    }

    /// @brief Serialize value to buffer
    /// @param value The value to serialize
    /// @param buffer Destination buffer (must have sizeof(T) bytes)
    /// @return Number of bytes written
    static size_type serialize(const T& value, std::byte* buffer) noexcept {
        std::memcpy(buffer, &value, sizeof(T));
        return sizeof(T);
    }

    /// @brief Deserialize value from buffer
    /// @param buffer Source buffer
    /// @param size Buffer size (must be >= sizeof(T))
    /// @return The deserialized value
    [[nodiscard]] static T deserialize(const std::byte* buffer, size_type size) noexcept {
        DTL_ASSERT(size >= sizeof(T));
        (void)size;
        T value;
        std::memcpy(&value, buffer, sizeof(T));
        return value;
    }

    /// @brief Check if serialization is in-place (no copy needed)
    /// @return true for trivial types
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return true;
    }
};

// =============================================================================
// std::array Specialization (for trivially serializable element types)
// =============================================================================

/// @brief Serializer specialization for std::array with trivially serializable elements
/// @tparam T Element type (must be trivially serializable)
/// @tparam N Array size
template <typename T, size_type N>
struct serializer<std::array<T, N>, std::enable_if_t<is_trivially_serializable_v<T>>> {
    using array_type = std::array<T, N>;

    /// @brief Get serialized size (fixed for arrays)
    [[nodiscard]] static constexpr size_type serialized_size(const array_type& /*value*/) noexcept {
        return sizeof(array_type);
    }

    /// @brief Get serialized size without an instance
    [[nodiscard]] static constexpr size_type serialized_size() noexcept {
        return sizeof(array_type);
    }

    /// @brief Serialize array to buffer
    static size_type serialize(const array_type& value, std::byte* buffer) noexcept {
        std::memcpy(buffer, value.data(), sizeof(array_type));
        return sizeof(array_type);
    }

    /// @brief Deserialize array from buffer
    [[nodiscard]] static array_type deserialize(const std::byte* buffer, size_type size) noexcept {
        DTL_ASSERT(size >= sizeof(array_type));
        (void)size;
        array_type value;
        std::memcpy(value.data(), buffer, sizeof(array_type));
        return value;
    }

    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return true;
    }
};

// =============================================================================
// std::pair Specialization (for trivially serializable types)
// =============================================================================

/// @brief Serializer specialization for std::pair with trivially serializable elements
/// @tparam T1 First element type (must be trivially serializable)
/// @tparam T2 Second element type (must be trivially serializable)
template <typename T1, typename T2>
struct serializer<std::pair<T1, T2>,
                  std::enable_if_t<is_trivially_serializable_v<T1> && is_trivially_serializable_v<T2>>> {
    using pair_type = std::pair<T1, T2>;

    /// @brief Get serialized size
    [[nodiscard]] static constexpr size_type serialized_size(const pair_type& /*value*/) noexcept {
        return sizeof(T1) + sizeof(T2);
    }

    /// @brief Get serialized size without an instance
    [[nodiscard]] static constexpr size_type serialized_size() noexcept {
        return sizeof(T1) + sizeof(T2);
    }

    /// @brief Serialize pair to buffer
    static size_type serialize(const pair_type& value, std::byte* buffer) noexcept {
        std::memcpy(buffer, &value.first, sizeof(T1));
        std::memcpy(buffer + sizeof(T1), &value.second, sizeof(T2));
        return sizeof(T1) + sizeof(T2);
    }

    /// @brief Deserialize pair from buffer
    [[nodiscard]] static pair_type deserialize(const std::byte* buffer, size_type size) noexcept {
        DTL_ASSERT(size >= sizeof(T1) + sizeof(T2));
        (void)size;
        pair_type value;
        std::memcpy(&value.first, buffer, sizeof(T1));
        std::memcpy(&value.second, buffer + sizeof(T1), sizeof(T2));
        return value;
    }

    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return true;
    }
};

// =============================================================================
// Serialization Helper Functions
// =============================================================================

/// @brief Get serialized size for a value
/// @tparam T The value type
/// @param value The value to measure
/// @return Size in bytes when serialized
template <typename T>
[[nodiscard]] inline size_type serialized_size(const T& value) {
    return serializer<T>::serialized_size(value);
}

/// @brief Serialize a value to a buffer
/// @tparam T The value type
/// @param value The value to serialize
/// @param buffer Destination buffer
/// @return Number of bytes written
template <typename T>
inline size_type serialize(const T& value, std::byte* buffer) {
    return serializer<T>::serialize(value, buffer);
}

/// @brief Deserialize a value from a buffer
/// @tparam T The value type
/// @param buffer Source buffer
/// @param size Buffer size
/// @return The deserialized value
template <typename T>
[[nodiscard]] inline T deserialize(const std::byte* buffer, size_type size) {
    return serializer<T>::deserialize(buffer, size);
}

// =============================================================================
// Serialization Detection Traits
// =============================================================================

/// @brief Detect if a type has a serializer specialization
/// @tparam T The type to check
template <typename T, typename = void>
struct has_serializer : std::false_type {};

/// @brief Specialization for types with a valid serializer
template <typename T>
struct has_serializer<T,
    std::void_t<decltype(serializer<T>::serialized_size(std::declval<const T&>())),
                decltype(serializer<T>::serialize(std::declval<const T&>(), std::declval<std::byte*>())),
                decltype(serializer<T>::deserialize(std::declval<const std::byte*>(), std::declval<size_type>()))>>
    : std::true_type {};

/// @brief Helper variable for has_serializer
template <typename T>
inline constexpr bool has_serializer_v = has_serializer<T>::value;

// =============================================================================
// Serialization Concepts
// =============================================================================

// Note: TriviallySerializable is defined in core/concepts.hpp

/// @brief Concept for types with a valid serializer
template <typename T>
concept Serializable = requires(const T& value, std::byte* buf, const std::byte* cbuf) {
    { serializer<T>::serialized_size(value) } -> std::convertible_to<size_type>;
    { serializer<T>::serialize(value, buf) } -> std::convertible_to<size_type>;
    { serializer<T>::deserialize(cbuf, size_type{}) } -> std::same_as<T>;
};

/// @brief Concept for types with fixed-size serialization
template <typename T>
concept FixedSizeSerializer = Serializable<T> && requires {
    { serializer<T>::serialized_size() } -> std::convertible_to<size_type>;
};

}  // namespace dtl
