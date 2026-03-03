// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file serializer.hpp
/// @brief Serializer concept for type serialization
/// @details Defines requirements for serializing types for communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>  // For TriviallySerializable

#include <concepts>
#include <cstring>
#include <span>
#include <type_traits>

namespace dtl {

// ============================================================================
// Serializer Concept
// ============================================================================

/// @brief Core serializer concept for type serialization
/// @details Defines minimum requirements for a serializer S that can serialize type T.
///
/// @par Required Operations:
/// - serialized_size(): Get size in bytes for serialization
/// - serialize(): Write object to byte buffer
/// - deserialize(): Read object from byte buffer
///
/// @note Named "SerializerFor" to distinguish from type-trait based concepts
template <typename S, typename T>
concept SerializerFor = requires(const T& obj, T& out_obj,
                              std::byte* buffer, const std::byte* cbuffer,
                              size_type size) {
    // Size query
    { S::serialized_size(obj) } -> std::same_as<size_type>;

    // Serialize to buffer (returns bytes written)
    { S::serialize(obj, buffer) } -> std::same_as<size_type>;

    // Deserialize from buffer (returns bytes read)
    { S::deserialize(cbuffer, out_obj) } -> std::same_as<size_type>;
};

// ============================================================================
// Fixed Size Serializer Concept
// ============================================================================

/// @brief Serializer for types with compile-time known size
/// @details Enables more efficient buffer allocation.
///
/// @note Named "FixedSizeSerializerFor" to distinguish from type-trait based concepts
template <typename S, typename T>
concept FixedSizeSerializerFor = SerializerFor<S, T> &&
    requires {
    // Compile-time size
    { S::fixed_size } -> std::convertible_to<size_type>;
};

// Note: TriviallySerializable is defined in core/concepts.hpp

// ============================================================================
// Contiguous Serializable Concept
// ============================================================================

/// @brief Type whose contiguous range can be serialized efficiently
template <typename T>
concept ContiguousSerializable = TriviallySerializable<T> ||
    requires {
    typename T::value_type;
    requires TriviallySerializable<typename T::value_type>;
};

// ============================================================================
// Serializer Traits
// ============================================================================

/// @brief Traits for serializer types
template <typename S, typename T>
struct serializer_traits {
    /// @brief Whether serialization is trivial (memcpy)
    static constexpr bool is_trivial = false;

    /// @brief Whether size is known at compile time
    static constexpr bool is_fixed_size = false;

    /// @brief Fixed size (only valid if is_fixed_size)
    static constexpr size_type fixed_size = 0;
};

/// @brief Specialization for trivially serializable types
template <typename S, TriviallySerializable T>
struct serializer_traits<S, T> {
    static constexpr bool is_trivial = true;
    static constexpr bool is_fixed_size = true;
    static constexpr size_type fixed_size = sizeof(T);
};

// ============================================================================
// Default Trivial Serializer
// ============================================================================

/// @brief Default serializer for trivially copyable types
/// @tparam T Trivially serializable type
template <TriviallySerializable T>
struct trivial_serializer {
    /// @brief Fixed size for this type
    static constexpr size_type fixed_size = sizeof(T);

    /// @brief Get serialized size
    /// @param obj The object (unused for trivial types)
    /// @return Size in bytes
    [[nodiscard]] static constexpr size_type serialized_size(const T& obj) noexcept {
        (void)obj;
        return sizeof(T);
    }

    /// @brief Serialize object to buffer
    /// @param obj The object to serialize
    /// @param buffer Destination buffer
    /// @return Bytes written
    static size_type serialize(const T& obj, std::byte* buffer) noexcept {
        std::memcpy(buffer, &obj, sizeof(T));
        return sizeof(T);
    }

    /// @brief Deserialize object from buffer
    /// @param buffer Source buffer
    /// @param obj Output object
    /// @return Bytes read
    static size_type deserialize(const std::byte* buffer, T& obj) noexcept {
        std::memcpy(&obj, buffer, sizeof(T));
        return sizeof(T);
    }
};

// ============================================================================
// Serialization Utilities
// ============================================================================

/// @brief Serialize an object using its default serializer
/// @tparam T Object type
/// @param obj The object to serialize
/// @param buffer Destination buffer
/// @return Bytes written
template <typename T>
    requires TriviallySerializable<T>
size_type serialize_trivial(const T& obj, std::byte* buffer) {
    return trivial_serializer<T>::serialize(obj, buffer);
}

/// @brief Deserialize an object using its default serializer
/// @tparam T Object type
/// @param buffer Source buffer
/// @param obj Output object
/// @return Bytes read
template <typename T>
    requires TriviallySerializable<T>
size_type deserialize_trivial(const std::byte* buffer, T& obj) {
    return trivial_serializer<T>::deserialize(buffer, obj);
}

/// @brief Get serialized size of an object
/// @tparam T Object type
/// @param obj The object
/// @return Serialized size in bytes
template <typename T>
    requires TriviallySerializable<T>
[[nodiscard]] constexpr size_type serialized_size_trivial(const T& obj) noexcept {
    return trivial_serializer<T>::serialized_size(obj);
}

}  // namespace dtl
