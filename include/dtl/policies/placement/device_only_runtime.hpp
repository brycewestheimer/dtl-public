// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_only_runtime.hpp
/// @brief Runtime device-only (GPU) memory placement policy
/// @details Memory is allocated exclusively on a GPU device, with the device ID
///          selected at runtime rather than compile-time. The device ID is
///          determined from the context's cuda_domain at container construction.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

namespace dtl {

/// @brief Runtime device-only placement policy
/// @details Unlike `device_only<N>` which encodes the device ID at compile time,
///          `device_only_runtime` allows the device ID to be selected at runtime.
///          The actual device ID is determined from the context's `cuda_domain`
///          or `hip_domain` at container construction time.
///
/// @par Runtime Device Selection
/// When creating a container with this placement policy:
/// 1. The container constructor extracts the device ID from `ctx.get<cuda_domain>().device_id()`
/// 2. The container stores the device ID in its device affinity state
/// 3. All allocations/deallocations use device guards to target the stored device
///
/// @par Requirements
/// - The context must have a `cuda_domain` (or `hip_domain`) present
/// - The domain's `device_id()` must return a valid device ID
/// - If no GPU domain is present, construction will fail with a clear error
///
/// @par Example Usage
/// @code
/// // Create a context with runtime-selected device
/// int gpu_id = get_available_gpu();  // Runtime value!
/// auto ctx = base_ctx.with_cuda(gpu_id);
///
/// // Create container - device ID comes from context
/// dtl::distributed_vector<float, dtl::device_only_runtime> vec(1000, ctx);
///
/// // Verify device affinity
/// assert(vec.device_id() == gpu_id);
/// @endcode
///
/// @par Thread Safety
/// Device operations are guarded with RAII device guards, ensuring thread-safe
/// operation even when multiple threads use different devices.
///
/// @see device_only<N> for compile-time device selection
/// @see cuda_device_memory_space_runtime for the underlying memory space
struct device_only_runtime {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief Sentinel value indicating runtime device selection
    /// @note This is not used as an actual device ID; the real ID is stored in the container
    static constexpr int device = -1;

    /// @brief Get the preferred memory location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::device;
    }

    /// @brief Check if memory is host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return false;  // Requires explicit copy to host
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return true;
    }

    /// @brief Check if data needs to be copied to host for CPU operations
    [[nodiscard]] static constexpr bool requires_host_copy() noexcept {
        return true;
    }

    /// @brief Check if data needs to be copied to device for GPU operations
    [[nodiscard]] static constexpr bool requires_device_copy() noexcept {
        return false;  // Already on device
    }

    /// @brief Check if this is a compile-time device selection policy
    [[nodiscard]] static constexpr bool is_compile_time_device() noexcept {
        return false;
    }

    /// @brief Check if this is a runtime device selection policy
    [[nodiscard]] static constexpr bool is_runtime_device() noexcept {
        return true;
    }

    /// @brief Validate that the provided context has a CUDA domain
    /// @tparam Ctx Context type
    /// @param ctx The context to validate
    /// @return true if context has a valid cuda_domain
    template <typename Ctx>
    [[nodiscard]] static constexpr bool validate_context([[maybe_unused]] const Ctx& ctx) noexcept {
        if constexpr (requires { ctx.template get<cuda_domain>().device_id(); }) {
            return ctx.template get<cuda_domain>().valid();
        } else if constexpr (requires { ctx.template get<hip_domain>().device_id(); }) {
            return ctx.template get<hip_domain>().valid();
        }
        return false;  // No GPU domain present
    }

    /// @brief Extract device ID from context
    /// @tparam Ctx Context type
    /// @param ctx The context to extract device ID from
    /// @return Device ID or -1 if not available
    template <typename Ctx>
    [[nodiscard]] static constexpr int extract_device_id([[maybe_unused]] const Ctx& ctx) noexcept {
        if constexpr (requires { ctx.template get<cuda_domain>().device_id(); }) {
            return ctx.template get<cuda_domain>().device_id();
        } else if constexpr (requires { ctx.template get<hip_domain>().device_id(); }) {
            return ctx.template get<hip_domain>().device_id();
        }
        return -1;  // No GPU domain present
    }
};

// ============================================================================
// Type Traits for Runtime Device Policies
// ============================================================================

/// @brief Check if a placement policy uses runtime device selection
template <typename P>
struct is_runtime_device_policy : std::false_type {};

template <>
struct is_runtime_device_policy<device_only_runtime> : std::true_type {};

/// @brief Helper variable template for is_runtime_device_policy
template <typename P>
inline constexpr bool is_runtime_device_policy_v = is_runtime_device_policy<P>::value;

/// @brief Check if a placement policy uses compile-time device selection
template <typename P>
struct is_compile_time_device_policy : std::false_type {};

template <int DeviceId>
struct is_compile_time_device_policy<device_only<DeviceId>> : std::true_type {};

/// @brief Helper variable template for is_compile_time_device_policy
template <typename P>
inline constexpr bool is_compile_time_device_policy_v = is_compile_time_device_policy<P>::value;

/// @brief Check if a placement policy targets device memory
template <typename P>
struct is_device_placement_policy : std::bool_constant<
    is_runtime_device_policy_v<P> || is_compile_time_device_policy_v<P>
> {};

/// @brief Helper variable template for is_device_placement_policy
template <typename P>
inline constexpr bool is_device_placement_policy_v = is_device_placement_policy<P>::value;

}  // namespace dtl
