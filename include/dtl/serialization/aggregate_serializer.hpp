// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file aggregate_serializer.hpp
/// @brief Macro-based serializer generation for aggregate types
/// @details Provides the DTL_SERIALIZABLE macro that generates a
///          dtl::serializer<T> specialization for struct/class types
///          using field-by-field serialization via the existing
///          serialize_field/deserialize_field/field_serialized_size helpers.
/// @since 0.1.0

#pragma once

#include <dtl/serialization/serializer.hpp>
#include <dtl/serialization/member_serialization.hpp>

#include <cstddef>
#include <type_traits>

// =============================================================================
// Implementation Helpers
// =============================================================================

/// @cond INTERNAL

#define DTL_DETAIL_SER_SIZE(field) \
    + ::dtl::field_serialized_size(value.field)

#define DTL_DETAIL_SER_WRITE(field) \
    offset += ::dtl::serialize_field(value.field, buffer + offset);

#define DTL_DETAIL_SER_READ(field) \
    result.field = ::dtl::deserialize_field<std::remove_cvref_t<decltype(result.field)>>(buffer + offset, size - offset); \
    offset += ::dtl::field_serialized_size(result.field);

// Helper to apply a macro to each variadic argument (up to 16 fields)
#define DTL_DETAIL_FOR_EACH_1(M, a)          M(a)
#define DTL_DETAIL_FOR_EACH_2(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_1(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_3(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_2(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_4(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_3(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_5(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_4(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_6(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_5(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_7(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_6(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_8(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_7(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_9(M, a, ...)     M(a) DTL_DETAIL_FOR_EACH_8(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_10(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_9(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_11(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_10(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_12(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_11(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_13(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_12(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_14(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_13(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_15(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_14(M, __VA_ARGS__)
#define DTL_DETAIL_FOR_EACH_16(M, a, ...)    M(a) DTL_DETAIL_FOR_EACH_15(M, __VA_ARGS__)

#define DTL_DETAIL_GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,NAME,...) NAME

#define DTL_DETAIL_FOR_EACH(M, ...) \
    DTL_DETAIL_GET_MACRO(__VA_ARGS__, \
        DTL_DETAIL_FOR_EACH_16, DTL_DETAIL_FOR_EACH_15, DTL_DETAIL_FOR_EACH_14, \
        DTL_DETAIL_FOR_EACH_13, DTL_DETAIL_FOR_EACH_12, DTL_DETAIL_FOR_EACH_11, \
        DTL_DETAIL_FOR_EACH_10, DTL_DETAIL_FOR_EACH_9, DTL_DETAIL_FOR_EACH_8,  \
        DTL_DETAIL_FOR_EACH_7, DTL_DETAIL_FOR_EACH_6, DTL_DETAIL_FOR_EACH_5,   \
        DTL_DETAIL_FOR_EACH_4, DTL_DETAIL_FOR_EACH_3, DTL_DETAIL_FOR_EACH_2,   \
        DTL_DETAIL_FOR_EACH_1)(M, __VA_ARGS__)

/// @endcond

// =============================================================================
// Public Macro
// =============================================================================

/// @brief Generate a dtl::serializer specialization for an aggregate type
/// @param Type The struct/class name
/// @param ... Comma-separated list of field names (1 to 16 fields)
///
/// @par Example:
/// @code
/// struct my_message {
///     int id;
///     std::string name;
///     std::vector<double> data;
/// };
/// DTL_SERIALIZABLE(my_message, id, name, data)
/// @endcode
///
/// @details The generated specialization uses dtl::serialize_field,
///          dtl::deserialize_field, and dtl::field_serialized_size to handle
///          each field. Field types must themselves be Serializable.
#define DTL_SERIALIZABLE(Type, ...)                                             \
    template <>                                                                 \
    struct dtl::serializer<Type, void> {                                        \
        [[nodiscard]] static ::dtl::size_type serialized_size(                  \
                const Type& value) {                                            \
            (void)value;                                                        \
            ::dtl::size_type total = 0                                          \
                DTL_DETAIL_FOR_EACH(DTL_DETAIL_SER_SIZE, __VA_ARGS__);          \
            return total;                                                       \
        }                                                                       \
                                                                                \
        static ::dtl::size_type serialize(                                      \
                const Type& value, std::byte* buffer) {                         \
            ::dtl::size_type offset = 0;                                        \
            DTL_DETAIL_FOR_EACH(DTL_DETAIL_SER_WRITE, __VA_ARGS__)              \
            return offset;                                                      \
        }                                                                       \
                                                                                \
        [[nodiscard]] static Type deserialize(                                  \
                const std::byte* buffer, ::dtl::size_type size) {               \
            Type result{};                                                      \
            ::dtl::size_type offset = 0;                                        \
            (void)size;                                                         \
            DTL_DETAIL_FOR_EACH(DTL_DETAIL_SER_READ, __VA_ARGS__)               \
            return result;                                                      \
        }                                                                       \
                                                                                \
        [[nodiscard]] static constexpr bool is_trivial() noexcept {             \
            return false;                                                       \
        }                                                                       \
    }
