// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file runtime_device_context.hpp
/// @brief Context utilities for runtime device selection
/// @details Provides helper functions for extracting device IDs from contexts
///          and validating contexts for runtime device placement policies.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/error/result.hpp>

#include <optional>
#include <stdexcept>

namespace dtl {
namespace detail {

// ============================================================================
// Context Device ID Extraction
// ============================================================================

/// @brief Extract CUDA/HIP device ID from a context
/// @tparam Ctx Context type
/// @param ctx The context to query
/// @return Device ID if GPU domain is present and valid, nullopt otherwise
template <typename Ctx>
[[nodiscard]] constexpr std::optional<int> ctx_gpu_device_id(const Ctx& ctx) noexcept {
#if DTL_ENABLE_CUDA
    if constexpr (requires { ctx.template get<cuda_domain>(); }) {
        if constexpr (requires { ctx.template has<cuda_domain>(); }) {
            if (ctx.template has<cuda_domain>()) {
                const auto& domain = ctx.template get<cuda_domain>();
                if (domain.valid()) {
                    return domain.device_id();
                }
            }
        } else {
            // Static check only
            const auto& domain = ctx.template get<cuda_domain>();
            if (domain.valid()) {
                return domain.device_id();
            }
        }
    }
#endif

#if DTL_ENABLE_HIP
    if constexpr (requires { ctx.template get<hip_domain>(); }) {
        if constexpr (requires { ctx.template has<hip_domain>(); }) {
            if (ctx.template has<hip_domain>()) {
                const auto& domain = ctx.template get<hip_domain>();
                if (domain.valid()) {
                    return domain.device_id();
                }
            }
        } else {
            const auto& domain = ctx.template get<hip_domain>();
            if (domain.valid()) {
                return domain.device_id();
            }
        }
    }
#endif

    return std::nullopt;
}

/// @brief Check if context has a valid GPU domain
/// @tparam Ctx Context type
/// @param ctx The context to check
/// @return true if context has cuda_domain or hip_domain with valid device
template <typename Ctx>
[[nodiscard]] constexpr bool ctx_has_gpu_domain([[maybe_unused]] const Ctx& ctx) noexcept {
    return ctx_gpu_device_id(ctx).has_value();
}

/// @brief Get device ID from context, throwing if not available
/// @tparam Ctx Context type
/// @param ctx The context to query
/// @return Device ID
/// @throws std::runtime_error if no GPU domain is present or invalid
template <typename Ctx>
[[nodiscard]] int ctx_require_gpu_device_id(const Ctx& ctx) {
    auto device_id = ctx_gpu_device_id(ctx);
    if (!device_id.has_value()) {
        throw std::runtime_error(
            "Runtime device placement requires a context with a valid cuda_domain or hip_domain. "
            "Use ctx.with_cuda(device_id) to add a CUDA domain to your context.");
    }
    return *device_id;
}

/// @brief Get device ID from context, returning result
/// @tparam Ctx Context type
/// @param ctx The context to query
/// @return Result containing device ID or error
template <typename Ctx>
[[nodiscard]] result<int> ctx_try_gpu_device_id(const Ctx& ctx) noexcept {
    auto device_id = ctx_gpu_device_id(ctx);
    if (!device_id.has_value()) {
        return result<int>::failure(
            status{status_code::invalid_argument, no_rank,
                   "Runtime device placement requires context with cuda_domain or hip_domain"});
    }
    return result<int>::success(*device_id);
}

// ============================================================================
// Compile-Time Context Validation
// ============================================================================

/// @brief Concept for contexts with GPU domain
template <typename Ctx>
concept ContextWithGpuDomain = requires(const Ctx& ctx) {
#if DTL_ENABLE_CUDA
    { ctx.template get<cuda_domain>().device_id() } -> std::convertible_to<int>;
#elif DTL_ENABLE_HIP
    { ctx.template get<hip_domain>().device_id() } -> std::convertible_to<int>;
#else
    requires false;  // No GPU backend available
#endif
};

/// @brief Concept for contexts that may have a GPU domain (runtime check needed)
template <typename Ctx>
concept ContextMayHaveGpuDomain =
#if DTL_ENABLE_CUDA
    requires(const Ctx& ctx) {
        { ctx.template has<cuda_domain>() } -> std::convertible_to<bool>;
    } ||
#endif
#if DTL_ENABLE_HIP
    requires(const Ctx& ctx) {
        { ctx.template has<hip_domain>() } -> std::convertible_to<bool>;
    } ||
#endif
    false;

}  // namespace detail
}  // namespace dtl
