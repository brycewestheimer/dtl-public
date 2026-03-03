// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_array_v2.cpp
 * @brief DTL C bindings - Distributed array with vtable-based policy dispatch
 * @since 0.1.0
 *
 * This is the new vtable-based array implementation that supports
 * runtime policy dispatch.
 */

#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_communicator.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"
#include "detail/container_vtable.hpp"
#include "detail/error_policy.hpp"

#include <cstring>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

using namespace dtl::c::detail;

extern "C" {

// ============================================================================
// Validation Helper
// ============================================================================

static array_handle* get_handle(dtl_array_t arr) {
    auto* h = reinterpret_cast<array_handle*>(arr);
    return is_valid_array_handle(h) ? h : nullptr;
}

static const array_handle* get_handle_const(dtl_array_t arr) {
    auto* h = reinterpret_cast<const array_handle*>(arr);
    return is_valid_array_handle(h) ? h : nullptr;
}

// ============================================================================
// Array Lifecycle (using default options)
// ============================================================================

dtl_status dtl_array_create(dtl_context_t ctx, dtl_dtype dtype,
                             dtl_size_t size,
                             dtl_array_t* arr) {
    // Delegate to create_with_options with NULL opts (defaults)
    return dtl_array_create_with_options(ctx, dtype, size, nullptr, arr);
}

dtl_status dtl_array_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                  dtl_size_t size,
                                  const void* value,
                                  dtl_array_t* arr) {
    // First create the array
    dtl_status status = dtl_array_create(ctx, dtype, size, arr);
    if (status != DTL_SUCCESS) {
        return status;
    }

    // Then fill if value is provided
    if (value) {
        status = dtl_array_fill_local(*arr, value);
        if (status != DTL_SUCCESS) {
            dtl_array_destroy(*arr);
            *arr = nullptr;
            return status;
        }
    }

    return DTL_SUCCESS;
}

void dtl_array_destroy(dtl_array_t arr) {
    auto* h = get_handle(arr);
    if (!h) return;

    // Destroy implementation
    h->vtable->destroy(h->impl);
    h->impl = nullptr;
    h->vtable = nullptr;

    // Invalidate and free handle
    h->base.magic = 0;
    delete h;
}

// ============================================================================
// Size Queries
// ============================================================================

dtl_size_t dtl_array_global_size(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->global_size(h->impl));
}

dtl_size_t dtl_array_local_size(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->local_size(h->impl));
}

dtl_index_t dtl_array_local_offset(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return 0;
    return static_cast<dtl_index_t>(h->vtable->local_offset(h->impl));
}

int dtl_array_empty(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return 1;
    return h->vtable->global_size(h->impl) == 0 ? 1 : 0;
}

dtl_dtype dtl_array_dtype(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return static_cast<dtl_dtype>(-1);
    return h->base.dtype;
}

// ============================================================================
// Local Data Access
// ============================================================================

const void* dtl_array_local_data(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return nullptr;

    // For device-only placement, return nullptr (use copy functions instead)
    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;
    }

    return h->vtable->local_data(h->impl);
}

void* dtl_array_local_data_mut(dtl_array_t arr) {
    auto* h = get_handle(arr);
    if (!h) return nullptr;

    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;
    }

    return h->vtable->local_data_mut(h->impl);
}

dtl_status dtl_array_get_local(dtl_array_t arr, dtl_size_t local_idx, void* value) {
    const auto* h = get_handle_const(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_idx >= local_size) return apply_error_policy(h, DTL_ERROR_OUT_OF_BOUNDS);

    // For device-only, return not supported (must use copy functions)
    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    const char* src = static_cast<const char*>(h->vtable->local_data(h->impl));
    std::memcpy(value, src + local_idx * elem_size, elem_size);
    return DTL_SUCCESS;
}

dtl_status dtl_array_set_local(dtl_array_t arr, dtl_size_t local_idx, const void* value) {
    auto* h = get_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_idx >= local_size) return apply_error_policy(h, DTL_ERROR_OUT_OF_BOUNDS);

    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    char* dst = static_cast<char*>(h->vtable->local_data_mut(h->impl));
    std::memcpy(dst + local_idx * elem_size, value, elem_size);
    return DTL_SUCCESS;
}

// ============================================================================
// Distribution Queries
// ============================================================================

dtl_rank_t dtl_array_num_ranks(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return 0;
    return h->vtable->num_ranks(h->impl);
}

dtl_rank_t dtl_array_rank(dtl_array_t arr) {
    const auto* h = get_handle_const(arr);
    if (!h) return DTL_NO_RANK;
    return h->vtable->rank(h->impl);
}

int dtl_array_is_local(dtl_array_t arr, dtl_index_t global_idx) {
    const auto* h = get_handle_const(arr);
    if (!h) return 0;
    return h->vtable->is_local(h->impl, global_idx);
}

dtl_rank_t dtl_array_owner(dtl_array_t arr, dtl_index_t global_idx) {
    const auto* h = get_handle_const(arr);
    if (!h) return DTL_NO_RANK;
    return h->vtable->owner(h->impl, global_idx);
}

dtl_index_t dtl_array_to_local(dtl_array_t arr, dtl_index_t global_idx) {
    const auto* h = get_handle_const(arr);
    if (!h) return -1;
    if (!h->vtable->is_local(h->impl, global_idx)) return -1;
    return h->vtable->to_local(h->impl, global_idx);
}

dtl_index_t dtl_array_to_global(dtl_array_t arr, dtl_index_t local_idx) {
    const auto* h = get_handle_const(arr);
    if (!h) return -1;
    return h->vtable->to_global(h->impl, local_idx);
}

// ============================================================================
// Local Operations
// ============================================================================

dtl_status dtl_array_fill_local(dtl_array_t arr, const void* value) {
    auto* h = get_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->fill(h->impl, value));
}

dtl_status dtl_array_barrier(dtl_array_t arr) {
    auto* h = get_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, dtl_barrier(h->base.ctx));
}

// ============================================================================
// Validation
// ============================================================================

int dtl_array_is_valid(dtl_array_t arr) {
    return get_handle_const(arr) != nullptr ? 1 : 0;
}

}  // extern "C"
