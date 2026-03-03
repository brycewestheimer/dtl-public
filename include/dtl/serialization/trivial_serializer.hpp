// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file trivial_serializer.hpp
/// @brief Trivially-copyable type serialization utilities
/// @details Provides optimized serialization for trivially-copyable types
///          using direct memory operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/serialization/serializer.hpp>

#include <cstring>
#include <vector>

namespace dtl {

// =============================================================================
// Bulk Serialization for Trivial Types
// =============================================================================

/// @brief Serialize a contiguous range of trivially serializable elements
/// @tparam T Element type (must be trivially serializable)
/// @param data Pointer to first element
/// @param count Number of elements
/// @param buffer Destination buffer
/// @return Number of bytes written
template <typename T>
    requires TriviallySerializable<T>
inline size_type serialize_trivial_range(const T* data, size_type count, std::byte* buffer) noexcept {
    const size_type bytes = count * sizeof(T);
    std::memcpy(buffer, data, bytes);
    return bytes;
}

/// @brief Deserialize a contiguous range of trivially serializable elements
/// @tparam T Element type (must be trivially serializable)
/// @param buffer Source buffer
/// @param count Number of elements to deserialize
/// @param data Destination array
/// @return Number of bytes read
template <typename T>
    requires TriviallySerializable<T>
inline size_type deserialize_trivial_range(const std::byte* buffer, size_type count, T* data) noexcept {
    const size_type bytes = count * sizeof(T);
    std::memcpy(data, buffer, bytes);
    return bytes;
}

/// @brief Deserialize into a new vector of trivially serializable elements
/// @tparam T Element type (must be trivially serializable)
/// @param buffer Source buffer
/// @param count Number of elements to deserialize
/// @return Vector containing deserialized elements
template <typename T>
    requires TriviallySerializable<T>
[[nodiscard]] inline std::vector<T> deserialize_trivial_vector(const std::byte* buffer, size_type count) {
    std::vector<T> result(count);
    deserialize_trivial_range(buffer, count, result.data());
    return result;
}

// =============================================================================
// Trivial Serialization Helpers
// =============================================================================

/// @brief Check if a type can use trivial (memcpy) serialization
/// @tparam T The type to check
/// @return true if T is trivially serializable
template <typename T>
[[nodiscard]] inline constexpr bool can_use_trivial_serialization() noexcept {
    return is_trivially_serializable_v<T>;
}

/// @brief Get aligned size for serialization (for performance)
/// @param size The actual size in bytes
/// @param alignment Desired alignment (default: 8 bytes)
/// @return Aligned size
[[nodiscard]] inline constexpr size_type aligned_size(size_type size, size_type alignment = 8) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

/// @brief Serialization buffer with trivial type optimization
/// @details Manages a byte buffer for serialization with special
///          handling for trivially-copyable types.
class trivial_buffer {
public:
    /// @brief Default constructor with optional initial capacity
    /// @param initial_capacity Initial buffer capacity in bytes
    explicit trivial_buffer(size_type initial_capacity = 1024)
        : data_(initial_capacity) {}

    /// @brief Reserve capacity
    /// @param capacity Minimum capacity in bytes
    void reserve(size_type capacity) {
        if (capacity > data_.size()) {
            data_.resize(capacity);
        }
    }

    /// @brief Get current write position
    [[nodiscard]] size_type position() const noexcept { return position_; }

    /// @brief Get buffer capacity
    [[nodiscard]] size_type capacity() const noexcept { return data_.size(); }

    /// @brief Get pointer to buffer data
    [[nodiscard]] std::byte* data() noexcept { return data_.data(); }
    [[nodiscard]] const std::byte* data() const noexcept { return data_.data(); }

    /// @brief Write a trivially serializable value
    /// @tparam T Value type
    /// @param value The value to write
    template <typename T>
        requires TriviallySerializable<T>
    void write(const T& value) {
        ensure_capacity(position_ + sizeof(T));
        std::memcpy(data_.data() + position_, &value, sizeof(T));
        position_ += sizeof(T);
    }

    /// @brief Write a range of trivially serializable values
    /// @tparam T Element type
    /// @param data Pointer to first element
    /// @param count Number of elements
    template <typename T>
        requires TriviallySerializable<T>
    void write_range(const T* data, size_type count) {
        const size_type bytes = count * sizeof(T);
        ensure_capacity(position_ + bytes);
        std::memcpy(data_.data() + position_, data, bytes);
        position_ += bytes;
    }

    /// @brief Reset write position
    void reset() noexcept { position_ = 0; }

    /// @brief Clear buffer and reset position
    void clear() {
        data_.clear();
        position_ = 0;
    }

private:
    void ensure_capacity(size_type required) {
        if (required > data_.size()) {
            data_.resize(std::max(required, data_.size() * 2));
        }
    }

    std::vector<std::byte> data_;
    size_type position_ = 0;
};

}  // namespace dtl
