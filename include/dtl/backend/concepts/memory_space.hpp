// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory_space.hpp
/// @brief Memory space concept for memory allocation
/// @details Defines requirements for memory spaces (host, device, unified).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <memory>

namespace dtl {

// ============================================================================
// Memory Space Properties
// ============================================================================

/// @brief Properties describing a memory space
struct memory_space_properties {
    /// @brief Whether memory is accessible from host CPU
    bool host_accessible = true;

    /// @brief Whether memory is accessible from device (GPU)
    bool device_accessible = false;

    /// @brief Whether memory is coherent across host and device
    bool unified = false;

    /// @brief Whether memory supports atomic operations
    bool supports_atomics = true;

    /// @brief Whether memory is pageable (vs pinned)
    bool pageable = true;

    /// @brief Alignment requirement in bytes
    size_type alignment = alignof(std::max_align_t);
};

// ============================================================================
// Memory Space Concept
// ============================================================================

/// @brief Core memory space concept
/// @details Defines minimum requirements for a memory space.
///
/// @par Required Types:
/// - pointer: Raw pointer type (void*)
/// - size_type: Size type
///
/// @par Required Operations:
/// - allocate(): Allocate memory
/// - deallocate(): Free memory
/// - properties(): Get space properties
template <typename T>
concept MemorySpace = requires(T& space, const T& cspace,
                               size_type size, size_type alignment,
                               void* ptr) {
    // Type aliases
    typename T::pointer;
    typename T::size_type;

    // Allocation
    { space.allocate(size) } -> std::same_as<void*>;
    { space.allocate(size, alignment) } -> std::same_as<void*>;
    { space.deallocate(ptr, size) } -> std::same_as<void>;

    // Properties (const)
    { cspace.properties() } -> std::same_as<memory_space_properties>;
    { cspace.name() } -> std::convertible_to<const char*>;
};

// ============================================================================
// Typed Memory Space Concept
// ============================================================================

/// @brief Memory space with typed allocation support
template <typename Space, typename T>
concept TypedMemorySpace = MemorySpace<Space> &&
    requires(Space& space, size_type count, T* ptr) {
    // Typed allocation
    { space.template allocate_typed<T>(count) } -> std::same_as<T*>;
    { space.deallocate_typed(ptr, count) } -> std::same_as<void>;

    // Construction (if supported)
    { space.template construct<T>(ptr) } -> std::same_as<void>;
    { space.destroy(ptr) } -> std::same_as<void>;
};

// ============================================================================
// Accessible Memory Space Concept
// ============================================================================

/// @brief Memory space with accessibility query
template <typename Space>
concept AccessibleMemorySpace = MemorySpace<Space> &&
    requires(const Space& space) {
    // Accessibility checks
    { space.is_host_accessible() } -> std::same_as<bool>;
    { space.is_device_accessible() } -> std::same_as<bool>;
    { space.is_accessible_from_host() } -> std::same_as<bool>;
    { space.is_accessible_from_device() } -> std::same_as<bool>;
};

// ============================================================================
// Memory Space Tag Types
// ============================================================================

/// @brief Tag for host (CPU) memory space
struct host_memory_space_tag {};

/// @brief Tag for device (GPU) memory space
struct device_memory_space_tag {};

/// @brief Tag for unified (managed) memory space
struct unified_memory_space_tag {};

/// @brief Tag for pinned (page-locked) host memory
struct pinned_memory_space_tag {};

// ============================================================================
// Memory Space Traits
// ============================================================================

/// @brief Traits for memory space types
template <typename Space>
struct memory_space_traits {
    /// @brief Whether this is a host memory space
    static constexpr bool is_host_space = false;

    /// @brief Whether this is a device memory space
    static constexpr bool is_device_space = false;

    /// @brief Whether this is a unified memory space
    static constexpr bool is_unified_space = false;

    /// @brief Whether allocation is thread-safe
    static constexpr bool is_thread_safe = true;
};

// ============================================================================
// Memory Space Utilities
// ============================================================================

/// @brief Check if two memory spaces are compatible
/// @details Two memory spaces are considered compatible if memory allocated
///          in one can be directly accessed from the other without explicit
///          copies. Unified memory spaces are compatible with everything.
///          Spaces in the same domain (both host or both device) are
///          compatible. Cross-domain spaces (host vs device) are not.
/// @tparam Space1 First memory space type
/// @tparam Space2 Second memory space type
/// @return true if memory from Space1 can be accessed from Space2
template <MemorySpace Space1, MemorySpace Space2>
[[nodiscard]] constexpr bool spaces_compatible() noexcept {
    using traits1 = memory_space_traits<Space1>;
    using traits2 = memory_space_traits<Space2>;

    // Unified memory spaces are compatible with everything
    if constexpr (traits1::is_unified_space || traits2::is_unified_space) {
        return true;
    }
    // Same-domain spaces are compatible (both host or both device)
    else if constexpr (traits1::is_host_space == traits2::is_host_space &&
                       traits1::is_device_space == traits2::is_device_space) {
        return true;
    }
    // Cross-domain: host vs device are incompatible
    else if constexpr ((traits1::is_host_space && traits2::is_device_space) ||
                       (traits1::is_device_space && traits2::is_host_space)) {
        return false;
    }
    // Default: unknown space types are treated as compatible to avoid
    // false negatives for user-defined or mock spaces.
    // Strict unknown-space rejection is deferred until call-site trait
    // coverage is fully audited for external/custom spaces.
    else {
        return true;
    }
}

/// @brief Get alignment for a type in a memory space
/// @tparam Space Memory space type
/// @tparam T Data type
/// @return Required alignment
template <MemorySpace Space, typename T>
[[nodiscard]] constexpr size_type space_alignment() noexcept {
    return alignof(T);
}

}  // namespace dtl
