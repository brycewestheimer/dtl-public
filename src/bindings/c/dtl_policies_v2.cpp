// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_policies.cpp
 * @brief DTL C bindings - Policy implementation with real dispatch
 * @since 0.1.0
 *
 * Implements policy-aware container creation and query functions.
 * Policies are now actually dispatched to different implementations,
 * not silently ignored.
 *
 */

#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"
#include "detail/container_vtable.hpp"
#include "detail/error_policy.hpp"
#include "detail/policy_matrix.hpp"
#include "detail/placement_mapping.hpp"
#include "detail/array_dispatch.hpp"
#include "detail/vector_dispatch.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

using namespace dtl::c::detail;

extern "C" {

// ============================================================================
// Policy Names
// ============================================================================

const char* dtl_partition_policy_name(dtl_partition_policy policy) {
    switch (policy) {
        case DTL_PARTITION_BLOCK:       return "block";
        case DTL_PARTITION_CYCLIC:      return "cyclic";
        case DTL_PARTITION_BLOCK_CYCLIC: return "block_cyclic";
        case DTL_PARTITION_HASH:        return "hash";
        case DTL_PARTITION_REPLICATED:  return "replicated";
        default:                        return "unknown";
    }
}

const char* dtl_placement_policy_name(dtl_placement_policy policy) {
    switch (policy) {
        case DTL_PLACEMENT_HOST:            return "host";
        case DTL_PLACEMENT_DEVICE:          return "device";
        case DTL_PLACEMENT_UNIFIED:         return "unified";
        case DTL_PLACEMENT_DEVICE_PREFERRED: return "device_preferred";
        default:                            return "unknown";
    }
}

const char* dtl_execution_policy_name(dtl_execution_policy policy) {
    switch (policy) {
        case DTL_EXEC_SEQ:   return "seq";
        case DTL_EXEC_PAR:   return "par";
        case DTL_EXEC_ASYNC: return "async";
        default:             return "unknown";
    }
}

const char* dtl_consistency_policy_name(dtl_consistency_policy policy) {
    switch (policy) {
        case DTL_CONSISTENCY_BULK_SYNCHRONOUS: return "bulk_synchronous";
        case DTL_CONSISTENCY_RELAXED:          return "relaxed";
        case DTL_CONSISTENCY_RELEASE_ACQUIRE:  return "release_acquire";
        case DTL_CONSISTENCY_SEQUENTIAL:       return "sequential";
        default:                               return "unknown";
    }
}

const char* dtl_error_policy_name(dtl_error_policy policy) {
    switch (policy) {
        case DTL_ERROR_POLICY_RETURN_STATUS: return "return_status";
        case DTL_ERROR_POLICY_CALLBACK:      return "callback";
        case DTL_ERROR_POLICY_TERMINATE:     return "terminate";
        default:                             return "unknown";
    }
}

// ============================================================================
// Placement Availability
// ============================================================================

int dtl_placement_available(dtl_placement_policy policy) {
    return is_placement_available(policy) ? 1 : 0;
}

// ============================================================================
// Error Handler Registration
// ============================================================================

dtl_status dtl_context_set_error_handler(
    dtl_context_t ctx,
    dtl_error_handler_t handler,
    void* user_data) {

    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    ctx->error_handler = handler;
    ctx->error_handler_user_data = user_data;

    return DTL_SUCCESS;
}

// ============================================================================
// Container Options
// ============================================================================

void dtl_container_options_init(dtl_container_options* opts) {
    if (!opts) return;

    opts->partition = DTL_PARTITION_BLOCK;
    opts->placement = DTL_PLACEMENT_HOST;
    opts->execution = DTL_EXEC_SEQ;
    opts->device_id = 0;
    opts->block_size = 1;
    std::memset(opts->reserved, 0, sizeof(opts->reserved));
    // reserved[0] = DTL_CONSISTENCY_BULK_SYNCHRONOUS (= 0)
    // reserved[1] = DTL_ERROR_POLICY_RETURN_STATUS (= 0)
    // reserved[2] = 0 (future use)
}

// ============================================================================
// Internal: Validate and Normalize Options
// ============================================================================

static dtl_status validate_and_normalize(
    const dtl_container_options* opts,
    stored_options* out) {

    // Apply defaults if NULL
    if (!opts) {
        out->partition = DTL_PARTITION_BLOCK;
        out->placement = DTL_PLACEMENT_HOST;
        out->execution = DTL_EXEC_SEQ;
        out->consistency = DTL_CONSISTENCY_BULK_SYNCHRONOUS;
        out->error = DTL_ERROR_POLICY_RETURN_STATUS;
        out->device_id = 0;
        out->block_size = 1;
        return DTL_SUCCESS;
    }

    // Copy basic fields
    out->partition = opts->partition;
    out->placement = opts->placement;
    out->execution = opts->execution;
    out->device_id = opts->device_id;
    out->block_size = opts->block_size;

    // Extract consistency and error from reserved fields
    out->consistency = static_cast<dtl_consistency_policy>(opts->reserved[0]);
    out->error = static_cast<dtl_error_policy>(opts->reserved[1]);

    // Validate range
    if (out->partition < 0 || out->partition >= DTL_PARTITION_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (out->placement < 0 || out->placement >= DTL_PLACEMENT_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (out->execution < 0 || out->execution >= DTL_EXEC_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (out->consistency < 0 || out->consistency >= DTL_CONSISTENCY_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (out->error < 0 || out->error >= DTL_ERROR_POLICY_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Validate policy combination is supported
    dtl_status policy_status = validate_policy_combination(
        out->partition, out->placement, out->execution,
        out->consistency, out->error);
    if (policy_status != DTL_SUCCESS) {
        return policy_status;
    }

    // Validate device_id
    dtl_status device_status = validate_device_id(out->placement, out->device_id);
    if (device_status != DTL_SUCCESS) {
        return device_status;
    }

    // Validate block_size
    dtl_status block_status = validate_block_size(out->partition, out->block_size);
    if (block_status != DTL_SUCCESS) {
        return block_status;
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Policy-Aware Vector Creation
// ============================================================================

dtl_status dtl_vector_create_with_options(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t global_size,
    const dtl_container_options* opts,
    dtl_vector_t* vec) {

    if (!vec) {
        return DTL_ERROR_NULL_POINTER;
    }
    *vec = nullptr;

    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Validate and normalize options
    stored_options stored;
    dtl_status status = validate_and_normalize(opts, &stored);
    if (status != DTL_SUCCESS) {
        return apply_error_policy(ctx, status, stored.error, "Invalid container options");
    }

    // Validate CUDA context for device placements
    if (placement_requires_cuda(stored.placement)) {
        if (!(ctx->domain_flags & dtl_context_s::HAS_CUDA)) {
            return apply_error_policy(ctx, DTL_ERROR_BACKEND_UNAVAILABLE, stored.error,
                                      "CUDA context required for device/unified placement");
        }
    }

    // Dispatch to create implementation
    const vector_vtable* vtable = nullptr;
    void* impl = nullptr;

    status = dispatch_create_vector(
        dtype, global_size, ctx->rank, ctx->size,
        stored, &vtable, &impl);

    if (status != DTL_SUCCESS) {
        return apply_error_policy(ctx, status, stored.error, "Vector creation failed");
    }

    // Allocate handle
    auto handle = std::unique_ptr<vector_handle>(new (std::nothrow) vector_handle{});
    if (!handle) {
        vtable->destroy(impl);
        return apply_error_policy(ctx, DTL_ERROR_ALLOCATION_FAILED, stored.error, "Handle allocation failed");
    }

    // Initialize handle
    handle->base.magic = vector_handle::VALID_MAGIC;
    handle->base.dtype = dtype;
    handle->base.ctx = ctx;
    handle->base.options = stored;
    handle->vtable = vtable;
    handle->impl = impl;

    *vec = reinterpret_cast<dtl_vector_t>(handle.release());
    return DTL_SUCCESS;
}

// ============================================================================
// Policy-Aware Array Creation
// ============================================================================

dtl_status dtl_array_create_with_options(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t size,
    const dtl_container_options* opts,
    dtl_array_t* arr) {

    if (!arr) {
        return DTL_ERROR_NULL_POINTER;
    }
    *arr = nullptr;

    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Validate and normalize options
    stored_options stored;
    dtl_status status = validate_and_normalize(opts, &stored);
    if (status != DTL_SUCCESS) {
        return apply_error_policy(ctx, status, stored.error, "Invalid container options");
    }

    // Validate CUDA context for device placements
    if (placement_requires_cuda(stored.placement)) {
        if (!(ctx->domain_flags & dtl_context_s::HAS_CUDA)) {
            return apply_error_policy(ctx, DTL_ERROR_BACKEND_UNAVAILABLE, stored.error,
                                      "CUDA context required for device/unified placement");
        }
    }

    // Dispatch to create implementation
    const array_vtable* vtable = nullptr;
    void* impl = nullptr;

    status = dispatch_create_array(
        dtype, size, ctx->rank, ctx->size,
        stored, &vtable, &impl);

    if (status != DTL_SUCCESS) {
        return apply_error_policy(ctx, status, stored.error, "Array creation failed");
    }

    // Allocate handle (using array_handle structure)
    auto handle = std::unique_ptr<array_handle>(new (std::nothrow) array_handle{});
    if (!handle) {
        vtable->destroy(impl);
        return apply_error_policy(ctx, DTL_ERROR_ALLOCATION_FAILED, stored.error, "Handle allocation failed");
    }

    // Initialize handle
    handle->base.magic = array_handle::VALID_MAGIC;
    handle->base.dtype = dtype;
    handle->base.ctx = ctx;
    handle->base.options = stored;
    handle->vtable = vtable;
    handle->impl = impl;

    *arr = reinterpret_cast<dtl_array_t>(handle.release());
    return DTL_SUCCESS;
}

// ============================================================================
// Policy Query Functions - Vector
// ============================================================================

// Helper to get vector handle
static const vector_handle* get_vector_handle(dtl_vector_t vec) {
    auto* h = reinterpret_cast<const vector_handle*>(vec);
    return is_valid_vector_handle(h) ? h : nullptr;
}

dtl_partition_policy dtl_vector_partition_policy(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return static_cast<dtl_partition_policy>(-1);
    return h->base.options.partition;
}

dtl_placement_policy dtl_vector_placement_policy(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return static_cast<dtl_placement_policy>(-1);
    return h->base.options.placement;
}

dtl_execution_policy dtl_vector_execution_policy(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return static_cast<dtl_execution_policy>(-1);
    return h->base.options.execution;
}

int dtl_vector_device_id(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return -1;
    return h->base.options.device_id;
}

dtl_consistency_policy dtl_vector_consistency_policy(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return static_cast<dtl_consistency_policy>(-1);
    return h->base.options.consistency;
}

dtl_error_policy dtl_vector_error_policy(dtl_vector_t vec) {
    const auto* h = get_vector_handle(vec);
    if (!h) return static_cast<dtl_error_policy>(-1);
    return h->base.options.error;
}

// ============================================================================
// Policy Query Functions - Array
// ============================================================================

// Helper to get array handle
static const array_handle* get_array_handle(dtl_array_t arr) {
    auto* h = reinterpret_cast<const array_handle*>(arr);
    return is_valid_array_handle(h) ? h : nullptr;
}

dtl_partition_policy dtl_array_partition_policy(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return static_cast<dtl_partition_policy>(-1);
    return h->base.options.partition;
}

dtl_placement_policy dtl_array_placement_policy(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return static_cast<dtl_placement_policy>(-1);
    return h->base.options.placement;
}

dtl_execution_policy dtl_array_execution_policy(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return static_cast<dtl_execution_policy>(-1);
    return h->base.options.execution;
}

int dtl_array_device_id(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return -1;
    return h->base.options.device_id;
}

dtl_consistency_policy dtl_array_consistency_policy(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return static_cast<dtl_consistency_policy>(-1);
    return h->base.options.consistency;
}

dtl_error_policy dtl_array_error_policy(dtl_array_t arr) {
    const auto* h = get_array_handle(arr);
    if (!h) return static_cast<dtl_error_policy>(-1);
    return h->base.options.error;
}

// ============================================================================
// Copy Helpers
// ============================================================================

dtl_status dtl_vector_copy_to_host(
    dtl_vector_t vec,
    void* host_buffer,
    dtl_size_t count) {

    auto* h = reinterpret_cast<vector_handle*>(vec);
    if (!is_valid_vector_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!host_buffer) {
        return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    }

    std::size_t actual_count = count == 0 ? h->vtable->local_size(h->impl) : count;
    return apply_error_policy(h, h->vtable->copy_to_host(h->impl, host_buffer, actual_count));
}

dtl_status dtl_vector_copy_from_host(
    dtl_vector_t vec,
    const void* host_buffer,
    dtl_size_t count) {

    auto* h = reinterpret_cast<vector_handle*>(vec);
    if (!is_valid_vector_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!host_buffer) {
        return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    }

    std::size_t actual_count = count == 0 ? h->vtable->local_size(h->impl) : count;
    return apply_error_policy(h, h->vtable->copy_from_host(h->impl, host_buffer, actual_count));
}

dtl_status dtl_array_copy_to_host(
    dtl_array_t arr,
    void* host_buffer,
    dtl_size_t count) {

    auto* h = reinterpret_cast<array_handle*>(arr);
    if (!is_valid_array_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!host_buffer) {
        return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    }

    std::size_t actual_count = count == 0 ? h->vtable->local_size(h->impl) : count;
    return apply_error_policy(h, h->vtable->copy_to_host(h->impl, host_buffer, actual_count));
}

dtl_status dtl_array_copy_from_host(
    dtl_array_t arr,
    const void* host_buffer,
    dtl_size_t count) {

    auto* h = reinterpret_cast<array_handle*>(arr);
    if (!is_valid_array_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!host_buffer) {
        return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    }

    std::size_t actual_count = count == 0 ? h->vtable->local_size(h->impl) : count;
    return apply_error_policy(h, h->vtable->copy_from_host(h->impl, host_buffer, actual_count));
}

}  // extern "C"
