// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file member_serialization.hpp
/// @brief Member function extension point for serialization
/// @details Provides serializer specialization that delegates to member functions.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/serialization/serializer.hpp>
#include <dtl/serialization/serialization_traits.hpp>

namespace dtl {

// =============================================================================
// Member Function Based Serializer
// =============================================================================

/// @brief Serializer specialization for types with member serialization functions
/// @tparam T A type with serialize(), deserialize(), and serialized_size() members
/// @details This specialization is used when a type provides its own
///          serialization interface through member functions.
///
/// @par Expected member function signatures:
/// @code
/// class MyType {
/// public:
///     // Return serialized size in bytes
///     size_type serialized_size() const;
///
///     // Serialize to buffer, return bytes written
///     size_type serialize(std::byte* buffer) const;
///
///     // Static deserialize from buffer
///     static MyType deserialize(const std::byte* buffer, size_type size);
/// };
/// @endcode
template <typename T>
    requires has_complete_member_serialization_v<T> && (!is_trivially_serializable_v<T>)
struct serializer<T, void> {
    /// @brief Get serialized size from member function
    /// @param value The value to measure
    /// @return Size in bytes
    [[nodiscard]] static size_type serialized_size(const T& value) {
        return value.serialized_size();
    }

    /// @brief Serialize using member function
    /// @param value The value to serialize
    /// @param buffer Destination buffer
    /// @return Number of bytes written
    static size_type serialize(const T& value, std::byte* buffer) {
        return value.serialize(buffer);
    }

    /// @brief Deserialize using static member function
    /// @param buffer Source buffer
    /// @param size Buffer size
    /// @return The deserialized value
    [[nodiscard]] static T deserialize(const std::byte* buffer, size_type size) {
        return T::deserialize(buffer, size);
    }

    /// @brief Member-based serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

// =============================================================================
// CRTP Base for Serializable Types
// =============================================================================

/// @brief CRTP base class for types with serialization support
/// @tparam Derived The derived type
/// @details Inherit from this to get serializer compatibility automatically.
///          Derived class must implement:
///          - size_type do_serialized_size() const
///          - size_type do_serialize(std::byte*) const
///          - static Derived do_deserialize(const std::byte*, size_type)
template <typename Derived>
class serializable_base {
public:
    /// @brief Get serialized size
    [[nodiscard]] size_type serialized_size() const {
        return static_cast<const Derived*>(this)->do_serialized_size();
    }

    /// @brief Serialize to buffer
    size_type serialize(std::byte* buffer) const {
        return static_cast<const Derived*>(this)->do_serialize(buffer);
    }

    /// @brief Deserialize from buffer
    [[nodiscard]] static Derived deserialize(const std::byte* buffer, size_type size) {
        return Derived::do_deserialize(buffer, size);
    }

protected:
    ~serializable_base() = default;
};

// =============================================================================
// Serialization Helpers for Member Functions
// =============================================================================

/// @brief Write a field to a serialization buffer
/// @tparam T Field type
/// @param value The field value
/// @param buffer Destination buffer
/// @return Number of bytes written
template <typename T>
inline size_type serialize_field(const T& value, std::byte* buffer) {
    return serializer<T>::serialize(value, buffer);
}

/// @brief Read a field from a serialization buffer
/// @tparam T Field type
/// @param buffer Source buffer
/// @param size Available bytes
/// @return The deserialized field value
template <typename T>
[[nodiscard]] inline T deserialize_field(const std::byte* buffer, size_type size) {
    return serializer<T>::deserialize(buffer, size);
}

/// @brief Get serialized size of a field
/// @tparam T Field type
/// @param value The field value
/// @return Size in bytes
template <typename T>
[[nodiscard]] inline size_type field_serialized_size(const T& value) {
    return serializer<T>::serialized_size(value);
}

}  // namespace dtl
