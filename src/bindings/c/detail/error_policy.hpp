// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file error_policy.hpp
 * @brief Internal helper for applying C ABI error policies
 * @since 0.1.0
 *
 * This header centralizes the logic for DTL_ERROR_POLICY_* handling for the
 * C ABI. When a container is created with an error policy, all subsequent
 * operations on that container should respect it:
 * - RETURN_STATUS: return the dtl_status as-is
 * - CALLBACK: invoke the per-context handler (if registered), then return status
 * - TERMINATE: abort the process on any error
 */

#ifndef DTL_C_DETAIL_ERROR_POLICY_HPP
#define DTL_C_DETAIL_ERROR_POLICY_HPP

#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>

#include "../dtl_internal.hpp"

#include <cstdlib>

namespace dtl::c::detail {

inline dtl_status apply_error_policy(
    dtl_context_t ctx,
    dtl_status status,
    dtl_error_policy policy,
    const char* message = nullptr) noexcept {

    if (status == DTL_SUCCESS) {
        return status;
    }

    switch (policy) {
        case DTL_ERROR_POLICY_RETURN_STATUS:
            break;

        case DTL_ERROR_POLICY_CALLBACK:
            if (is_valid_context(ctx) && ctx->error_handler) {
                const char* msg = message ? message : dtl_status_message(status);
                ctx->error_handler(status, msg, ctx->error_handler_user_data);
            }
            break;

        case DTL_ERROR_POLICY_TERMINATE:
            std::abort();
            break;

        default:
            break;
    }

    return status;
}

template <typename Handle>
inline dtl_status apply_error_policy(
    const Handle* h,
    dtl_status status,
    const char* message = nullptr) noexcept {

    if (!h) {
        return status;
    }
    return apply_error_policy(h->base.ctx, status, h->base.options.error, message);
}

}  // namespace dtl::c::detail

#endif /* DTL_C_DETAIL_ERROR_POLICY_HPP */

