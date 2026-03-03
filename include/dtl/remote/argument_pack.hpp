// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file argument_pack.hpp
/// @brief Argument pack serialization for RPC
/// @details Provides serialization for variadic argument tuples using
///          the existing dtl::serializer trait.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/serialization/serializer.hpp>

#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtl::remote {

// ============================================================================
// Argument Pack
// ============================================================================

/// @brief Serializable pack of function arguments
/// @tparam Args Argument types
/// @details Provides static methods for serializing and deserializing
///          tuples of arguments for RPC transport.
template <typename... Args>
class argument_pack {
public:
    /// @brief The tuple type for these arguments
    using tuple_type = std::tuple<std::decay_t<Args>...>;

    /// @brief Number of arguments
    static constexpr size_type arity = sizeof...(Args);

    // ========================================================================
    // Size Calculation
    // ========================================================================

    /// @brief Calculate serialized size for argument values
    /// @param args The arguments to measure
    /// @return Total size in bytes when serialized
    static size_type serialized_size(const Args&... args) {
        if constexpr (arity == 0) {
            return 0;
        } else {
            return (dtl::serialized_size(args) + ...);
        }
    }

    /// @brief Calculate serialized size for a tuple
    /// @param args_tuple Tuple of arguments
    /// @return Total size in bytes
    static size_type serialized_size(const tuple_type& args_tuple) {
        return std::apply([](const auto&... args) {
            return serialized_size(args...);
        }, args_tuple);
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// @brief Serialize arguments to a buffer
    /// @param args The arguments to serialize
    /// @param buffer Destination buffer (must have sufficient space)
    /// @return Number of bytes written
    static size_type serialize(const Args&... args, std::byte* buffer) {
        if constexpr (arity == 0) {
            return 0;
        } else {
            size_type offset = 0;
            ((offset += dtl::serialize(args, buffer + offset)), ...);
            return offset;
        }
    }

    /// @brief Serialize a tuple to a buffer
    /// @param args_tuple Tuple of arguments
    /// @param buffer Destination buffer
    /// @return Number of bytes written
    static size_type serialize(const tuple_type& args_tuple, std::byte* buffer) {
        return std::apply([buffer](const auto&... args) {
            return serialize(args..., buffer);
        }, args_tuple);
    }

    /// @brief Serialize arguments to a vector
    /// @param args The arguments to serialize
    /// @return Vector containing serialized data
    static std::vector<std::byte> serialize_to_vector(const Args&... args) {
        size_type size = serialized_size(args...);
        std::vector<std::byte> buffer(size);
        if (size > 0) {
            serialize(args..., buffer.data());
        }
        return buffer;
    }

    /// @brief Serialize a tuple to a vector
    /// @param args_tuple Tuple of arguments
    /// @return Vector containing serialized data
    static std::vector<std::byte> serialize_to_vector(const tuple_type& args_tuple) {
        return std::apply([](const auto&... args) {
            return serialize_to_vector(args...);
        }, args_tuple);
    }

    // ========================================================================
    // Deserialization
    // ========================================================================

    /// @brief Deserialize arguments from a buffer
    /// @param buffer Source buffer
    /// @param size Buffer size
    /// @return Tuple containing deserialized arguments
    static tuple_type deserialize(const std::byte* buffer, size_type size) {
        if constexpr (arity == 0) {
            (void)buffer;
            (void)size;
            return tuple_type{};
        } else {
            return deserialize_impl(buffer, size, std::index_sequence_for<Args...>{});
        }
    }

private:
    /// @brief Helper to deserialize each argument in sequence
    template <size_type... Is>
    static tuple_type deserialize_impl(
        const std::byte* buffer,
        size_type total_size,
        std::index_sequence<Is...>) {

        // Track offset through buffer
        size_type offset = 0;

        // Helper lambda to deserialize one argument
        auto deserialize_one = [&]<typename T>(std::type_identity<T>) -> T {
            // Calculate remaining size
            [[maybe_unused]] size_type remaining = total_size - offset;

            // For fixed-size types, we know the size
            size_type arg_size = dtl::serializer<T>::serialized_size();

            DTL_ASSERT(remaining >= arg_size);

            T value = dtl::deserialize<T>(buffer + offset, arg_size);
            offset += arg_size;
            return value;
        };

        // Deserialize all arguments in order
        // Note: Order of evaluation in braced-init-list is left-to-right (C++11)
        return tuple_type{
            deserialize_one(std::type_identity<std::tuple_element_t<Is, tuple_type>>{})...
        };
    }
};

// ============================================================================
// Specialization for Empty Pack
// ============================================================================

template <>
class argument_pack<> {
public:
    using tuple_type = std::tuple<>;
    static constexpr size_type arity = 0;

    static size_type serialized_size() { return 0; }
    static size_type serialize(std::byte*) { return 0; }
    static std::vector<std::byte> serialize_to_vector() { return {}; }
    static tuple_type deserialize(const std::byte*, size_type) { return {}; }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Create an argument pack from types
template <typename... Args>
argument_pack<Args...> make_argument_pack_type(std::tuple<Args...>*);

/// @brief Get argument pack type from a tuple type
template <typename Tuple>
using argument_pack_for = decltype(
    make_argument_pack_type(static_cast<Tuple*>(nullptr)));

// ============================================================================
// Pack Traits
// ============================================================================

/// @brief Check if all types in a pack are serializable
template <typename... Args>
struct all_serializable : std::bool_constant<(Serializable<std::decay_t<Args>> && ...)> {};

template <typename... Args>
inline constexpr bool all_serializable_v = all_serializable<Args...>::value;

/// @brief Concept for serializable argument packs
template <typename... Args>
concept SerializableArgs = all_serializable_v<Args...>;

}  // namespace dtl::remote
