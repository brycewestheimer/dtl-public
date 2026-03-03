// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_affinity.hpp
/// @brief Device affinity utilities for containers
/// @details Provides helpers for managing container device affinity,
///          including device ID storage and scope guard creation.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/device_guard.hpp>
#endif

#if DTL_ENABLE_HIP
#include <dtl/hip/device_guard.hpp>
#endif

namespace dtl {
namespace detail {

// ============================================================================
// Device Affinity Constants
// ============================================================================

/// @brief Sentinel value indicating no device affinity (host memory)
inline constexpr int no_device_affinity = -1;

/// @brief Sentinel value indicating any device (unified memory)
inline constexpr int any_device_affinity = -2;

// ============================================================================
// Device Affinity Extraction from Placement Policies
// ============================================================================

/// @brief Extract device ID from a placement policy
/// @tparam Placement Placement policy type
/// @return Device ID or no_device_affinity for non-device policies
template <typename Placement>
struct placement_device_id {
    static constexpr int value = no_device_affinity;
};

// Specialization for device_only<DeviceId>
template <template <int> class DeviceOnlyPolicy, int DeviceId>
    requires requires { DeviceOnlyPolicy<DeviceId>::device_id(); }
struct placement_device_id<DeviceOnlyPolicy<DeviceId>> {
    static constexpr int value = DeviceId;
};

/// @brief Helper alias for placement_device_id
template <typename Placement>
inline constexpr int placement_device_id_v = placement_device_id<Placement>::value;

// ============================================================================
// Device Affinity Storage
// ============================================================================

/// @brief Mixin providing device affinity storage and accessors
/// @details Used by containers to store their device affinity at runtime.
///          For compile-time device policies, this stores the known device ID.
///          For runtime device selection, this stores the configured device ID.
class device_affinity_storage {
public:
    /// @brief Default constructor (no device affinity)
    constexpr device_affinity_storage() noexcept
        : device_id_(no_device_affinity) {}

    /// @brief Construct with specific device ID
    /// @param device_id The device ID (-1 for host, -2 for any/unified)
    explicit constexpr device_affinity_storage(int device_id) noexcept
        : device_id_(device_id) {}

    /// @brief Get the device ID
    /// @return Device ID or no_device_affinity for host memory
    [[nodiscard]] constexpr int device_id() const noexcept {
        return device_id_;
    }

    /// @brief Check if container has device affinity
    [[nodiscard]] constexpr bool has_device_affinity() const noexcept {
        return device_id_ >= 0;
    }

    /// @brief Check if container is on host memory
    [[nodiscard]] constexpr bool is_host_memory() const noexcept {
        return device_id_ == no_device_affinity;
    }

    /// @brief Check if container uses unified memory
    [[nodiscard]] constexpr bool is_unified_memory() const noexcept {
        return device_id_ == any_device_affinity;
    }

protected:
    int device_id_;
};

// ============================================================================
// Scoped Device Operations
// ============================================================================

/// @brief Create a device scope guard for the container's device
/// @details Returns an RAII guard that sets the device and restores
///          the previous device on destruction.
///
/// @par Backend Selection
/// Uses CUDA guard if CUDA is enabled, HIP guard if HIP is enabled,
/// or no-op otherwise.
template <typename Container>
[[nodiscard]] auto make_device_scope(const Container& container) noexcept {
    [[maybe_unused]] int device_id = container.device_id();

#if DTL_ENABLE_CUDA
    return cuda::device_guard(device_id);
#elif DTL_ENABLE_HIP
    return hip::device_guard(device_id);
#else
    // No GPU backend, return a no-op object
    struct noop_guard {
        constexpr noop_guard() noexcept = default;
        constexpr int previous_device() const noexcept { return -1; }
        constexpr int target_device() const noexcept { return -1; }
        constexpr bool switched() const noexcept { return false; }
    };
    return noop_guard{};
#endif
}

/// @brief Create a device scope guard for a specific device ID
/// @param device_id Target device ID
/// @return RAII device guard
[[nodiscard]] inline auto make_device_scope(int device_id) noexcept {
#if DTL_ENABLE_CUDA
    return cuda::device_guard(device_id);
#elif DTL_ENABLE_HIP
    return hip::device_guard(device_id);
#else
    (void)device_id;
    struct noop_guard {
        constexpr noop_guard() noexcept = default;
        constexpr int previous_device() const noexcept { return -1; }
        constexpr int target_device() const noexcept { return -1; }
        constexpr bool switched() const noexcept { return false; }
    };
    return noop_guard{};
#endif
}

// ============================================================================
// Memory Location Query
// ============================================================================

/// @brief Represents the actual memory location of a container
enum class actual_memory_location {
    host,           ///< Host (CPU) memory
    device,         ///< GPU device memory
    unified,        ///< Unified/managed memory
    unknown         ///< Unknown or mixed
};

/// @brief Get a string representation of memory location
[[nodiscard]] inline constexpr const char* to_string(actual_memory_location loc) noexcept {
    switch (loc) {
        case actual_memory_location::host:    return "host";
        case actual_memory_location::device:  return "device";
        case actual_memory_location::unified: return "unified";
        case actual_memory_location::unknown: return "unknown";
    }
    return "unknown";
}

}  // namespace detail
}  // namespace dtl
