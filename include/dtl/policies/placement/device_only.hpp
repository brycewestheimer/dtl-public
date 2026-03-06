// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_only.hpp
/// @brief Device-only (GPU) memory placement policy
/// @details Memory is allocated exclusively on a specific GPU device.
///          The device ID is enforced at allocation time using device guards.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

namespace dtl {

/// @brief Device-only placement allocates memory on a specific GPU
/// @tparam DeviceId The GPU device ID (default: 0)
/// @details Memory is allocated in GPU device memory on the specified device
///          and is not directly accessible from the host CPU. Data must be
///          copied to access from host.
///
/// @par Device Selection
/// The DeviceId template parameter specifies which GPU device will be used
/// for allocations. Allocations are guarded to ensure they occur on the
/// correct device regardless of the current CUDA context state.
///
/// @par Compile-Time vs Runtime Device Selection
/// This policy uses compile-time device selection. For runtime device
/// selection, use `device_only_runtime` (planned for Phase 02).
///
/// @warning Using the same device_only<N> type with different runtime contexts
///          may lead to unexpected behavior. Ensure the context's CUDA domain
///          (if present) is configured for device N.
///
/// @par Example
/// @code
/// // Allocates on device 0
/// dtl::distributed_vector<float, dtl::device_only<0>> vec0(1000, ctx);
///
/// // Allocates on device 1 (different type)
/// dtl::distributed_vector<float, dtl::device_only<1>> vec1(1000, ctx);
///
/// // Verify: these are different types
/// static_assert(!std::is_same_v<decltype(vec0), decltype(vec1)>);
/// @endcode
template <int DeviceId = 0>
struct device_only {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief The device ID for this placement
    static constexpr int device = DeviceId;

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

    /// @brief Get device ID (compile-time)
    [[nodiscard]] static constexpr int device_id() noexcept {
        return DeviceId;
    }

    /// @brief Check if this is a compile-time device selection policy
    [[nodiscard]] static constexpr bool is_compile_time_device() noexcept {
        return true;
    }

    /// @brief Validate that the provided context device matches this policy
    /// @tparam Ctx Context type
    /// @param ctx The context to validate
    /// @return true if context device matches or context has no cuda_domain
    /// @note Returns true if context doesn't have a cuda_domain (permissive)
    template <typename Ctx>
    [[nodiscard]] static constexpr bool validate_context([[maybe_unused]] const Ctx& ctx) noexcept {
        // If context has a cuda_domain, validate device ID matches
        if constexpr (requires { ctx.template get<cuda_domain>().device_id(); }) {
            return ctx.template get<cuda_domain>().device_id() == DeviceId;
        }
        return true;  // No cuda_domain, permissive
    }
};

/// @brief Type alias for default device (device 0)
using device_only_default = device_only<0>;

}  // namespace dtl
