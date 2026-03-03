// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file serialization_traits.hpp
/// @brief Traits for detecting serialization capabilities
/// @details Provides compile-time detection of serialization methods.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <type_traits>

namespace dtl {

// =============================================================================
// Serialization Method Detection
// =============================================================================

/// @brief Detect if type has member serialize function
/// @tparam T The type to check
template <typename T, typename = void>
struct has_member_serialize : std::false_type {};

template <typename T>
struct has_member_serialize<T,
    std::void_t<decltype(std::declval<const T&>().serialize(std::declval<std::byte*>()))>>
    : std::true_type {};

/// @brief Helper variable for has_member_serialize
template <typename T>
inline constexpr bool has_member_serialize_v = has_member_serialize<T>::value;

/// @brief Detect if type has member deserialize function
/// @tparam T The type to check
template <typename T, typename = void>
struct has_member_deserialize : std::false_type {};

template <typename T>
struct has_member_deserialize<T,
    std::void_t<decltype(T::deserialize(std::declval<const std::byte*>(), std::declval<size_type>()))>>
    : std::true_type {};

/// @brief Helper variable for has_member_deserialize
template <typename T>
inline constexpr bool has_member_deserialize_v = has_member_deserialize<T>::value;

/// @brief Detect if type has member serialized_size function
/// @tparam T The type to check
template <typename T, typename = void>
struct has_member_serialized_size : std::false_type {};

template <typename T>
struct has_member_serialized_size<T,
    std::void_t<decltype(std::declval<const T&>().serialized_size())>>
    : std::true_type {};

/// @brief Helper variable for has_member_serialized_size
template <typename T>
inline constexpr bool has_member_serialized_size_v = has_member_serialized_size<T>::value;

/// @brief Check if type has complete member serialization interface
/// @tparam T The type to check
template <typename T>
struct has_complete_member_serialization
    : std::bool_constant<has_member_serialize_v<T> &&
                         has_member_deserialize_v<T> &&
                         has_member_serialized_size_v<T>> {};

/// @brief Helper variable for has_complete_member_serialization
template <typename T>
inline constexpr bool has_complete_member_serialization_v =
    has_complete_member_serialization<T>::value;

// =============================================================================
// Serializer Specialization Detection
// =============================================================================

/// @brief Detect if type has a serializer specialization
/// @tparam T The type to check
template <typename T, typename = void>
struct has_serializer_specialization : std::false_type {};

template <typename T>
struct has_serializer_specialization<T,
    std::void_t<decltype(serializer<T>::serialized_size(std::declval<const T&>()))>>
    : std::true_type {};

/// @brief Helper variable for has_serializer_specialization
template <typename T>
inline constexpr bool has_serializer_specialization_v =
    has_serializer_specialization<T>::value;

// =============================================================================
// Serialization Strategy Selection
// =============================================================================

/// @brief Enumeration of serialization strategies
enum class serialization_strategy {
    trivial,          ///< Direct memcpy (for trivially serializable types)
    member_functions, ///< Use member serialize/deserialize functions
    specialization,   ///< Use serializer<T> specialization
    not_serializable  ///< Type cannot be serialized
};

/// @brief Determine the serialization strategy for a type
/// @tparam T The type to analyze
/// @return The appropriate serialization strategy
template <typename T>
[[nodiscard]] constexpr serialization_strategy get_serialization_strategy() noexcept {
    if constexpr (is_trivially_serializable_v<T>) {
        return serialization_strategy::trivial;
    } else if constexpr (has_complete_member_serialization_v<T>) {
        return serialization_strategy::member_functions;
    } else if constexpr (has_serializer_specialization_v<T>) {
        return serialization_strategy::specialization;
    } else {
        return serialization_strategy::not_serializable;
    }
}

/// @brief Check if a type is serializable through any mechanism
/// @tparam T The type to check
template <typename T>
struct is_serializable
    : std::bool_constant<get_serialization_strategy<T>() != serialization_strategy::not_serializable> {};

/// @brief Helper variable for is_serializable
template <typename T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;

}  // namespace dtl
