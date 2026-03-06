// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_vector_v2.cpp
 * @brief DTL C bindings - Distributed vector with vtable-based policy dispatch
 * @since 0.1.0
 *
 * This is the new vtable-based vector implementation that supports
 * runtime policy dispatch.
 */

#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_communicator.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"
#include "detail/container_vtable.hpp"
#include "detail/error_policy.hpp"
#include "detail/vector_dispatch.hpp"

#include <cstring>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

using namespace dtl::c::detail;

extern "C" {

// ============================================================================
// Validation Helper
// ============================================================================

static vector_handle* get_handle(dtl_vector_t vec) {
    auto* h = reinterpret_cast<vector_handle*>(vec);
    return is_valid_vector_handle(h) ? h : nullptr;
}

static const vector_handle* get_handle_const(dtl_vector_t vec) {
    auto* h = reinterpret_cast<const vector_handle*>(vec);
    return is_valid_vector_handle(h) ? h : nullptr;
}

// ============================================================================
// Vector Lifecycle (using default options)
// ============================================================================

dtl_status dtl_vector_create(dtl_context_t ctx, dtl_dtype dtype,
                              dtl_size_t global_size,
                              dtl_vector_t* vec) {
    // Delegate to create_with_options with NULL opts (defaults)
    return dtl_vector_create_with_options(ctx, dtype, global_size, nullptr, vec);
}

dtl_status dtl_vector_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                   dtl_size_t global_size,
                                   const void* value,
                                   dtl_vector_t* vec) {
    // First create the vector
    dtl_status status = dtl_vector_create(ctx, dtype, global_size, vec);
    if (status != DTL_SUCCESS) {
        return status;
    }

    // Then fill if value is provided
    if (value) {
        status = dtl_vector_fill_local(*vec, value);
        if (status != DTL_SUCCESS) {
            dtl_vector_destroy(*vec);
            *vec = nullptr;
            return status;
        }
    }

    return DTL_SUCCESS;
}

void dtl_vector_destroy(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return;

    // Destroy the implementation
    if (h->vtable && h->impl) {
        h->vtable->destroy(h->impl);
    }

    // Clear magic and delete handle
    h->base.magic = 0;
    delete h;
}

// ============================================================================
// Size Queries
// ============================================================================

dtl_size_t dtl_vector_global_size(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->global_size(h->impl));
}

dtl_size_t dtl_vector_local_size(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->local_size(h->impl));
}

dtl_index_t dtl_vector_local_offset(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return static_cast<dtl_index_t>(h->vtable->local_offset(h->impl));
}

int dtl_vector_empty(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 1;
    return h->vtable->global_size(h->impl) == 0 ? 1 : 0;
}

dtl_dtype dtl_vector_dtype(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return static_cast<dtl_dtype>(-1);
    return h->base.dtype;
}

// ============================================================================
// Local Data Access
// ============================================================================

const void* dtl_vector_local_data(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return nullptr;

    // For device-only placement, return nullptr (use copy functions instead)
    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;  // Cannot access device memory directly from host
    }

    return h->vtable->local_data(h->impl);
}

void* dtl_vector_local_data_mut(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return nullptr;

    // For device-only placement, return nullptr
    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;
    }

    return h->vtable->local_data_mut(h->impl);
}

const void* dtl_vector_device_data(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return nullptr;
    return h->vtable->device_data(h->impl);
}

void* dtl_vector_device_data_mut(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return nullptr;
    return h->vtable->device_data_mut(h->impl);
}

dtl_status dtl_vector_get_local(dtl_vector_t vec, dtl_size_t local_idx, void* value) {
    const auto* h = get_handle_const(vec);
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

dtl_status dtl_vector_set_local(dtl_vector_t vec, dtl_size_t local_idx, const void* value) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_idx >= local_size) return apply_error_policy(h, DTL_ERROR_OUT_OF_BOUNDS);

    // For device-only, return not supported
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

dtl_rank_t dtl_vector_num_ranks(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return h->vtable->num_ranks(h->impl);
}

dtl_rank_t dtl_vector_rank(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return DTL_NO_RANK;
    return h->vtable->rank(h->impl);
}

int dtl_vector_is_local(dtl_vector_t vec, dtl_index_t global_idx) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return h->vtable->is_local(h->impl, global_idx);
}

dtl_rank_t dtl_vector_owner(dtl_vector_t vec, dtl_index_t global_idx) {
    const auto* h = get_handle_const(vec);
    if (!h) return DTL_NO_RANK;
    return h->vtable->owner(h->impl, global_idx);
}

dtl_index_t dtl_vector_to_local(dtl_vector_t vec, dtl_index_t global_idx) {
    const auto* h = get_handle_const(vec);
    if (!h) return -1;
    return h->vtable->to_local(h->impl, global_idx);
}

dtl_index_t dtl_vector_to_global(dtl_vector_t vec, dtl_index_t local_idx) {
    const auto* h = get_handle_const(vec);
    if (!h) return -1;
    return h->vtable->to_global(h->impl, local_idx);
}

// ============================================================================
// Collective Operations
// ============================================================================

dtl_status dtl_vector_resize(dtl_vector_t vec, dtl_size_t new_size) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, h->vtable->resize(h->impl, new_size));
}

dtl_status dtl_vector_barrier(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, dtl_barrier(h->base.ctx));
}

dtl_status dtl_vector_fill_local(dtl_vector_t vec, const void* value) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->fill(h->impl, value));
}

dtl_status dtl_vector_reduce_sum(dtl_vector_t vec, void* result) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->reduce_sum(h->impl, result));
}

dtl_status dtl_vector_reduce_min(dtl_vector_t vec, void* result) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->reduce_min(h->impl, result));
}

dtl_status dtl_vector_reduce_max(dtl_vector_t vec, void* result) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->reduce_max(h->impl, result));
}

dtl_status dtl_vector_sort_ascending(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, h->vtable->sort_ascending(h->impl));
}

// ============================================================================
// Validation
// ============================================================================

int dtl_vector_is_valid(dtl_vector_t vec) {
    return get_handle_const(vec) != nullptr ? 1 : 0;
}

// ============================================================================
// Dirty State (Stub for future implementation)
// ============================================================================

int dtl_vector_is_dirty(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    // Future: track dirty state
    return 0;
}

int dtl_vector_is_clean(dtl_vector_t vec) {
    const auto* h = get_handle_const(vec);
    if (!h) return 0;
    return 1;
}

dtl_status dtl_vector_sync(dtl_vector_t vec) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    // Future: synchronize dirty state
    return DTL_SUCCESS;
}

// ============================================================================
// Redistribution
// ============================================================================

dtl_status dtl_vector_redistribute(dtl_vector_t vec, dtl_partition_type /*new_partition*/) {
    auto* h = get_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (dtl_context_size(h->base.ctx) <= 1) {
        return DTL_SUCCESS;
    }
    // Redistribution requires communication - not yet implemented
    return DTL_ERROR_NOT_IMPLEMENTED;
}

}  // extern "C"
