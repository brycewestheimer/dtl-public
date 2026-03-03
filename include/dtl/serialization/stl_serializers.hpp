// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stl_serializers.hpp
/// @brief Built-in serializers for common STL types
/// @details Provides serializer specializations for std::string,
///          std::vector<T>, and std::optional<T>.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/serialization/serializer.hpp>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace dtl {

// Complete the std::optional detection trait (forward-declared in serializer.hpp)
template <typename T>
struct is_std_optional<std::optional<T>> : std::true_type {};

// =============================================================================
// std::string Serializer
// =============================================================================

/// @brief Serializer specialization for std::string
/// @details Format: [length: sizeof(size_type)] [char_data: length bytes]
template <>
struct serializer<std::string, void> {
    /// @brief Get serialized size
    [[nodiscard]] static size_type serialized_size(const std::string& value) noexcept {
        return sizeof(size_type) + value.size();
    }

    /// @brief Serialize string to buffer
    static size_type serialize(const std::string& value, std::byte* buffer) noexcept {
        const size_type len = value.size();
        std::memcpy(buffer, &len, sizeof(size_type));
        if (len > 0) {
            std::memcpy(buffer + sizeof(size_type), value.data(), len);
        }
        return sizeof(size_type) + len;
    }

    /// @brief Deserialize string from buffer
    [[nodiscard]] static std::string deserialize(const std::byte* buffer, size_type /*size*/) {
        size_type len = 0;
        std::memcpy(&len, buffer, sizeof(size_type));
        return std::string(reinterpret_cast<const char*>(buffer + sizeof(size_type)), len);
    }

    /// @brief String serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

// =============================================================================
// std::vector<T> Serializer
// =============================================================================

/// @brief Serializer specialization for std::vector<T>
/// @tparam T Element type (must have a serializer)
/// @details Trivial elements use optimized memcpy path.
///          Non-trivial elements are serialized individually with size prefixes.
///          Format (trivial):     [count: sizeof(size_type)] [data: count * sizeof(T)]
///          Format (non-trivial): [count: sizeof(size_type)] [elem_size + elem_data]...
template <typename T>
struct serializer<std::vector<T>, void> {
    /// @brief Get serialized size
    [[nodiscard]] static size_type serialized_size(const std::vector<T>& value) {
        size_type total = sizeof(size_type);  // element count
        if constexpr (is_trivially_serializable_v<T>) {
            total += value.size() * sizeof(T);
        } else {
            for (const auto& elem : value) {
                total += sizeof(size_type);  // per-element size prefix
                total += serializer<T>::serialized_size(elem);
            }
        }
        return total;
    }

    /// @brief Serialize vector to buffer
    static size_type serialize(const std::vector<T>& value, std::byte* buffer) {
        const size_type count = value.size();
        std::memcpy(buffer, &count, sizeof(size_type));
        size_type offset = sizeof(size_type);

        if constexpr (is_trivially_serializable_v<T>) {
            const size_type data_bytes = count * sizeof(T);
            if (data_bytes > 0) {
                std::memcpy(buffer + offset, value.data(), data_bytes);
            }
            offset += data_bytes;
        } else {
            for (const auto& elem : value) {
                const size_type elem_size = serializer<T>::serialized_size(elem);
                std::memcpy(buffer + offset, &elem_size, sizeof(size_type));
                offset += sizeof(size_type);
                offset += serializer<T>::serialize(elem, buffer + offset);
            }
        }
        return offset;
    }

    /// @brief Deserialize vector from buffer
    [[nodiscard]] static std::vector<T> deserialize(const std::byte* buffer, size_type /*size*/) {
        size_type count = 0;
        std::memcpy(&count, buffer, sizeof(size_type));

        std::vector<T> result;
        result.reserve(count);
        size_type offset = sizeof(size_type);

        if constexpr (is_trivially_serializable_v<T>) {
            result.resize(count);
            if (count > 0) {
                std::memcpy(result.data(), buffer + offset, count * sizeof(T));
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                size_type elem_size = 0;
                std::memcpy(&elem_size, buffer + offset, sizeof(size_type));
                offset += sizeof(size_type);
                result.push_back(serializer<T>::deserialize(buffer + offset, elem_size));
                offset += elem_size;
            }
        }
        return result;
    }

    /// @brief Vector serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

// =============================================================================
// std::optional<T> Serializer
// =============================================================================

/// @brief Serializer specialization for std::optional<T>
/// @tparam T Value type (must have a serializer)
/// @details Format: [has_value: 1 byte] [value_data] (if has_value)
template <typename T>
struct serializer<std::optional<T>, void> {
    /// @brief Get serialized size
    [[nodiscard]] static size_type serialized_size(const std::optional<T>& value) {
        size_type total = 1;  // has_value flag
        if (value.has_value()) {
            total += serializer<T>::serialized_size(*value);
        }
        return total;
    }

    /// @brief Serialize optional to buffer
    static size_type serialize(const std::optional<T>& value, std::byte* buffer) {
        buffer[0] = value.has_value() ? std::byte{1} : std::byte{0};
        size_type offset = 1;
        if (value.has_value()) {
            offset += serializer<T>::serialize(*value, buffer + offset);
        }
        return offset;
    }

    /// @brief Deserialize optional from buffer
    [[nodiscard]] static std::optional<T> deserialize(const std::byte* buffer, size_type size) {
        if (buffer[0] == std::byte{0}) {
            return std::nullopt;
        }
        return serializer<T>::deserialize(buffer + 1, size - 1);
    }

    /// @brief Optional serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

}  // namespace dtl
