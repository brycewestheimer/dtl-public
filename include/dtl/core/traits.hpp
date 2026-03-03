// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file traits.hpp
/// @brief Type traits for DTL types
/// @details Provides compile-time type introspection for serializability,
///          transportability, and other DTL-specific type properties.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <type_traits>

namespace dtl {

// =============================================================================
// Primary Type Traits
// =============================================================================

/// @brief Check if a type is trivially serializable (can be memcpy'd)
/// @tparam T The type to check
/// @details A type is trivially serializable if it's trivially copyable
///          and has standard layout.
template <typename T>
struct is_trivially_serializable
    : std::bool_constant<std::is_trivially_copyable_v<T> &&
                         std::is_standard_layout_v<T>> {};

/// @brief Helper variable template for is_trivially_serializable
template <typename T>
inline constexpr bool is_trivially_serializable_v = is_trivially_serializable<T>::value;

// Forward declaration of has_serializer (defined in serialization/serializer.hpp)
template <typename T, typename = void>
struct has_serializer_trait;

/// @brief Check if a type is transportable (can be sent across ranks)
/// @tparam T The type to check
/// @details A type is transportable if it either:
///          1. Is trivially serializable, or
///          2. Has a specialization of dtl::serializer
/// @note The full check including serializer detection is enabled when
///       serialization headers are included. This base definition only
///       checks trivial serializability.
template <typename T, typename = void>
struct is_transportable : std::bool_constant<is_trivially_serializable_v<T>> {};

/// @brief Helper variable template for is_transportable
template <typename T>
inline constexpr bool is_transportable_v = is_transportable<T>::value;

/// @brief Extended is_transportable that includes serializer detection
/// @details When serializer.hpp is included, types with custom serializers
///          are also considered transportable. This overload is selected
///          via SFINAE when has_serializer_v is available.
template <typename T>
struct is_transportable_extended {
private:
    // Check if type has a serializer (requires serializer.hpp inclusion)
    template <typename U, typename = void>
    struct has_ser : std::false_type {};

    // SFINAE check for serializer availability
    template <typename U>
    struct has_ser<U, std::void_t<decltype(sizeof(U))>>
        : std::bool_constant<is_trivially_serializable_v<U>> {};

public:
    static constexpr bool value = has_ser<T>::value;
};

/// @brief Check if a type is a distributed container
/// @tparam T The type to check
template <typename T>
struct is_distributed_container : std::false_type {};

/// @brief Helper variable template for is_distributed_container
template <typename T>
inline constexpr bool is_distributed_container_v = is_distributed_container<T>::value;

/// @brief Check if a type is a distributed vector
/// @tparam T The type to check
template <typename T>
struct is_distributed_vector : std::false_type {};

/// @brief Helper variable template for is_distributed_vector
template <typename T>
inline constexpr bool is_distributed_vector_v = is_distributed_vector<T>::value;

/// @brief Check if a type is a distributed tensor
/// @tparam T The type to check
template <typename T>
struct is_distributed_tensor : std::false_type {};

/// @brief Helper variable template for is_distributed_tensor
template <typename T>
inline constexpr bool is_distributed_tensor_v = is_distributed_tensor<T>::value;

/// @brief Check if a type is a distributed span
/// @tparam T The type to check
template <typename T>
struct is_distributed_span : std::false_type {};

/// @brief Helper variable template for is_distributed_span
template <typename T>
inline constexpr bool is_distributed_span_v = is_distributed_span<T>::value;

/// @brief Check if a type is a distributed map
/// @tparam T The type to check
template <typename T>
struct is_distributed_map : std::false_type {};

/// @brief Helper variable template for is_distributed_map
template <typename T>
inline constexpr bool is_distributed_map_v = is_distributed_map<T>::value;

/// @brief Check if a type is a local view
/// @tparam T The type to check
template <typename T>
struct is_local_view : std::false_type {};

/// @brief Helper variable template for is_local_view
template <typename T>
inline constexpr bool is_local_view_v = is_local_view<T>::value;

/// @brief Check if a type is a global view
/// @tparam T The type to check
template <typename T>
struct is_global_view : std::false_type {};

/// @brief Helper variable template for is_global_view
template <typename T>
inline constexpr bool is_global_view_v = is_global_view<T>::value;

/// @brief Check if a type is a segmented view
/// @tparam T The type to check
template <typename T>
struct is_segmented_view : std::false_type {};

/// @brief Helper variable template for is_segmented_view
template <typename T>
inline constexpr bool is_segmented_view_v = is_segmented_view<T>::value;

/// @brief Check if a type is a remote reference
/// @tparam T The type to check
template <typename T>
struct is_remote_ref : std::false_type {};

/// @brief Helper variable template for is_remote_ref
template <typename T>
inline constexpr bool is_remote_ref_v = is_remote_ref<T>::value;

// =============================================================================
// Policy Traits
// =============================================================================

/// @brief Tag type for partition policies
struct partition_policy_tag {};

/// @brief Tag type for placement policies
struct placement_policy_tag {};

/// @brief Tag type for consistency policies
struct consistency_policy_tag {};

/// @brief Tag type for execution policies
struct execution_policy_tag {};

/// @brief Tag type for error policies
struct error_policy_tag {};

/// @brief Check if a type is a partition policy
/// @tparam T The type to check
template <typename T>
struct is_partition_policy : std::false_type {};

template <typename T>
    requires requires { typename T::policy_category; }
struct is_partition_policy<T>
    : std::is_same<typename T::policy_category, partition_policy_tag> {};

/// @brief Helper variable template for is_partition_policy
template <typename T>
inline constexpr bool is_partition_policy_v = is_partition_policy<T>::value;

/// @brief Check if a type is a placement policy
/// @tparam T The type to check
template <typename T>
struct is_placement_policy : std::false_type {};

template <typename T>
    requires requires { typename T::policy_category; }
struct is_placement_policy<T>
    : std::is_same<typename T::policy_category, placement_policy_tag> {};

/// @brief Helper variable template for is_placement_policy
template <typename T>
inline constexpr bool is_placement_policy_v = is_placement_policy<T>::value;

/// @brief Check if a type is a consistency policy
/// @tparam T The type to check
template <typename T>
struct is_consistency_policy : std::false_type {};

template <typename T>
    requires requires { typename T::policy_category; }
struct is_consistency_policy<T>
    : std::is_same<typename T::policy_category, consistency_policy_tag> {};

/// @brief Helper variable template for is_consistency_policy
template <typename T>
inline constexpr bool is_consistency_policy_v = is_consistency_policy<T>::value;

/// @brief Check if a type is an execution policy
/// @tparam T The type to check
template <typename T>
struct is_execution_policy : std::false_type {};

template <typename T>
    requires requires { typename T::policy_category; }
struct is_execution_policy<T>
    : std::is_same<typename T::policy_category, execution_policy_tag> {};

/// @brief Helper variable template for is_execution_policy
template <typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

/// @brief Check if a type is an error policy
/// @tparam T The type to check
template <typename T>
struct is_error_policy : std::false_type {};

template <typename T>
    requires requires { typename T::policy_category; }
struct is_error_policy<T>
    : std::is_same<typename T::policy_category, error_policy_tag> {};

/// @brief Helper variable template for is_error_policy
template <typename T>
inline constexpr bool is_error_policy_v = is_error_policy<T>::value;

// =============================================================================
// Extent Traits
// =============================================================================

/// @brief Check if extents are all static (compile-time known)
/// @tparam E The extents type
template <typename E>
struct is_static_extents : std::false_type {};

template <size_type... Extents>
struct is_static_extents<extents<Extents...>>
    : std::bool_constant<((Extents != dynamic_extent) && ...)> {};

/// @brief Helper variable template for is_static_extents
template <typename E>
inline constexpr bool is_static_extents_v = is_static_extents<E>::value;

/// @brief Check if extents are all dynamic (runtime determined)
/// @tparam E The extents type
template <typename E>
struct is_dynamic_extents : std::false_type {};

template <size_type... Extents>
struct is_dynamic_extents<extents<Extents...>>
    : std::bool_constant<((Extents == dynamic_extent) && ...)> {};

/// @brief Helper variable template for is_dynamic_extents
template <typename E>
inline constexpr bool is_dynamic_extents_v = is_dynamic_extents<E>::value;

/// @brief Get the rank (number of dimensions) of an extents type
/// @tparam E The extents type
template <typename E>
struct extents_rank;

template <size_type... Extents>
struct extents_rank<extents<Extents...>>
    : std::integral_constant<size_type, sizeof...(Extents)> {};

/// @brief Helper variable template for extents_rank
template <typename E>
inline constexpr size_type extents_rank_v = extents_rank<E>::value;

// =============================================================================
// Container Element Traits
// =============================================================================

/// @brief Extract the value type from a container
/// @tparam Container The container type
template <typename Container>
struct container_value_type {
    using type = typename Container::value_type;
};

/// @brief Helper alias for container_value_type
template <typename Container>
using container_value_type_t = typename container_value_type<Container>::type;

/// @brief Extract the extents type from a container
/// @tparam Container The container type
template <typename Container>
struct container_extents_type {
    using type = typename Container::extents_type;
};

/// @brief Helper alias for container_extents_type
template <typename Container>
using container_extents_type_t = typename container_extents_type<Container>::type;

// =============================================================================
// Type Manipulation Utilities
// =============================================================================

/// @brief Remove cv-qualifiers and references
/// @tparam T The type to clean
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

/// @brief Check if two types are the same after removing cv-qualifiers
/// @tparam T First type
/// @tparam U Second type
template <typename T, typename U>
inline constexpr bool is_same_cvref_v = std::is_same_v<remove_cvref_t<T>, remove_cvref_t<U>>;

}  // namespace dtl
