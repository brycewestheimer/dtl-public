// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_concepts.hpp
/// @brief Concepts and type constraints for device-compatible types
/// @details Provides compile-time constraints to prevent unsupported types
///          from being used with device-only placement policies.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <type_traits>

namespace dtl {

// ============================================================================
// Device Storable Concept
// ============================================================================

/// @brief Concept for types that can be stored in device memory
/// @details A DeviceStorable type must be trivially copyable, allowing
///          safe memcpy-style transfers between host and device without
///          requiring constructor/destructor calls on the device.
///
/// @par Requirements
/// - `std::is_trivially_copyable_v<T>` must be true
///
/// @par Rationale
/// Device memory (CUDA/HIP) cannot execute arbitrary C++ constructors or
/// destructors. Non-trivially copyable types like `std::string` or `std::vector`
/// would require host-side construction, which would dereference device pointers
/// and cause undefined behavior.
///
/// @par Future Extensions
/// In future versions, we may allow types with `__device__` constructors
/// when using CUDA's device-side new/placement-new. For now, trivial
/// copyability is the safe baseline.
///
/// @par Example
/// @code
/// static_assert(DeviceStorable<int>);          // OK
/// static_assert(DeviceStorable<float>);        // OK
/// static_assert(DeviceStorable<double>);       // OK
/// static_assert(!DeviceStorable<std::string>); // Fails: not trivially copyable
/// static_assert(!DeviceStorable<std::vector<int>>); // Fails
/// @endcode
template <typename T>
concept DeviceStorable = std::is_trivially_copyable_v<T>;

/// @brief Type trait version of DeviceStorable concept
template <typename T>
struct is_device_storable : std::bool_constant<DeviceStorable<T>> {};

/// @brief Helper variable template for is_device_storable
template <typename T>
inline constexpr bool is_device_storable_v = is_device_storable<T>::value;

// ============================================================================
// Device Constructible Concept (Future Extension)
// ============================================================================

/// @brief Concept for types that can be constructed on device
/// @details Currently equivalent to DeviceStorable. In the future, this may
///          also allow types with __device__ constructors.
template <typename T>
concept DeviceConstructible = DeviceStorable<T>;

// ============================================================================
// Placement Constraint Helpers
// ============================================================================

/// @brief Check if a placement policy requires device-storable elements
/// @tparam Placement Placement policy type
template <typename Placement>
struct requires_device_storable : std::false_type {};

// Forward declarations for placement policies
template <int DeviceId>
struct device_only;

struct device_only_runtime;
struct unified_memory;

/// @brief device_only<N> requires device-storable types
template <int DeviceId>
struct requires_device_storable<device_only<DeviceId>> : std::true_type {};

/// @brief device_only_runtime requires device-storable types
template <>
struct requires_device_storable<device_only_runtime> : std::true_type {};

/// @brief unified_memory also requires device-storable types (for GPU access)
template <>
struct requires_device_storable<unified_memory> : std::true_type {};

/// @brief Helper variable template
template <typename Placement>
inline constexpr bool requires_device_storable_v = requires_device_storable<Placement>::value;

// ============================================================================
// Static Assert Message Helpers
// ============================================================================

/// @brief Generate a compile-time error message for non-device-storable types
/// @details This helper provides a clear error message when a type cannot
///          be used with device placement.
///
/// @par Usage
/// @code
/// template <typename T, typename Placement>
/// class container {
///     static_assert(
///         !requires_device_storable_v<Placement> || DeviceStorable<T>,
///         device_storable_error_message<T>::value
///     );
/// };
/// @endcode
template <typename T>
struct device_storable_error_message {
    static constexpr const char* value =
        "Type T is not DeviceStorable. Device-only placements require trivially "
        "copyable types. std::string, std::vector, and other non-trivial types "
        "cannot be used with device_only<N>, device_only_runtime, or unified_memory "
        "placement policies. Use host_only placement instead.";
};

// ============================================================================
// Compound Constraint Check
// ============================================================================

/// @brief Check if T is valid for Placement
/// @tparam T Element type
/// @tparam Placement Placement policy type
/// @return true if T is valid for Placement
template <typename T, typename Placement>
inline constexpr bool is_valid_element_for_placement_v =
    !requires_device_storable_v<Placement> || DeviceStorable<T>;

/// @brief Concept combining element type with placement policy
template <typename T, typename Placement>
concept ValidElementForPlacement =
    !requires_device_storable_v<Placement> || DeviceStorable<T>;

}  // namespace dtl
