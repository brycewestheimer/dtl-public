// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_tensor_v2.cpp
 * @brief DTL C bindings - Distributed tensor with vtable-based policy dispatch
 * @since 0.1.0
 *
 * This is the new vtable-based tensor implementation that supports
 * runtime policy dispatch.
 */

#include <dtl/bindings/c/dtl_tensor.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_communicator.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"
#include "detail/container_vtable.hpp"
#include "detail/tensor_dispatch.hpp"
#include "detail/error_policy.hpp"

#include <cstring>
#include <memory>
#include <new>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

using namespace dtl::c::detail;

extern "C" {

// ============================================================================
// Validation Helper
// ============================================================================

static tensor_handle* get_handle(dtl_tensor_t t) {
    auto* h = reinterpret_cast<tensor_handle*>(t);
    return is_valid_tensor_handle(h) ? h : nullptr;
}

static const tensor_handle* get_handle_const(dtl_tensor_t t) {
    auto* h = reinterpret_cast<const tensor_handle*>(t);
    return is_valid_tensor_handle(h) ? h : nullptr;
}

// ============================================================================
// Tensor Lifecycle
// ============================================================================

dtl_status dtl_tensor_create(dtl_context_t ctx, dtl_dtype dtype,
                              dtl_shape shape, dtl_tensor_t* tensor) {
    if (!is_valid_context(ctx)) return DTL_ERROR_INVALID_ARGUMENT;
    if (!tensor) return DTL_ERROR_NULL_POINTER;
    if (shape.ndim < 1 || shape.ndim > DTL_MAX_TENSOR_RANK) return DTL_ERROR_INVALID_ARGUMENT;
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) return DTL_ERROR_INVALID_ARGUMENT;

    stored_options opts{};
    opts.partition = DTL_PARTITION_BLOCK;
    opts.placement = DTL_PLACEMENT_HOST;
    opts.execution = DTL_EXEC_SEQ;
    opts.consistency = DTL_CONSISTENCY_BULK_SYNCHRONOUS;
    opts.error = DTL_ERROR_POLICY_RETURN_STATUS;
    opts.device_id = 0;
    opts.block_size = 1;

    const tensor_vtable* vt = nullptr;
    void* impl = nullptr;
    dtl_status status = dispatch_create_tensor(dtype, shape, ctx->rank, ctx->size, opts, &vt, &impl);
    if (status != DTL_SUCCESS) return status;

    auto h = std::unique_ptr<tensor_handle>(new (std::nothrow) tensor_handle{});
    if (!h) {
        vt->destroy(impl);
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    h->base.magic = tensor_handle::VALID_MAGIC;
    h->base.dtype = dtype;
    h->base.ctx = ctx;
    h->base.options = opts;
    h->vtable = vt;
    h->impl = impl;

    *tensor = reinterpret_cast<dtl_tensor_t>(h.release());
    return DTL_SUCCESS;
}

dtl_status dtl_tensor_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                   dtl_shape shape, const void* value,
                                   dtl_tensor_t* tensor) {
    dtl_status status = dtl_tensor_create(ctx, dtype, shape, tensor);
    if (status != DTL_SUCCESS) return status;

    if (value) {
        status = dtl_tensor_fill_local(*tensor, value);
        if (status != DTL_SUCCESS) {
            dtl_tensor_destroy(*tensor);
            *tensor = nullptr;
            return status;
        }
    }

    return DTL_SUCCESS;
}

void dtl_tensor_destroy(dtl_tensor_t tensor) {
    auto* h = get_handle(tensor);
    if (!h) return;

    h->vtable->destroy(h->impl);
    h->impl = nullptr;
    h->vtable = nullptr;

    h->base.magic = 0;
    delete h;
}

// ============================================================================
// Shape Queries
// ============================================================================

dtl_shape dtl_tensor_shape(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) {
        dtl_shape empty = {};
        return empty;
    }
    dtl_shape out;
    h->vtable->shape(h->impl, &out);
    return out;
}

int dtl_tensor_ndim(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return h->vtable->ndim(h->impl);
}

dtl_size_t dtl_tensor_dim(dtl_tensor_t tensor, int dim) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return h->vtable->dim(h->impl, dim);
}

dtl_size_t dtl_tensor_global_size(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->global_size(h->impl));
}

dtl_size_t dtl_tensor_local_size(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return static_cast<dtl_size_t>(h->vtable->local_size(h->impl));
}

dtl_shape dtl_tensor_local_shape(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) {
        dtl_shape empty = {};
        return empty;
    }
    dtl_shape out;
    h->vtable->local_shape(h->impl, &out);
    return out;
}

dtl_dtype dtl_tensor_dtype(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return static_cast<dtl_dtype>(-1);
    return h->base.dtype;
}

// ============================================================================
// Local Data Access
// ============================================================================

const void* dtl_tensor_local_data(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return nullptr;

    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;
    }

    return h->vtable->local_data(h->impl);
}

void* dtl_tensor_local_data_mut(dtl_tensor_t tensor) {
    auto* h = get_handle(tensor);
    if (!h) return nullptr;

    if (h->base.options.placement == DTL_PLACEMENT_DEVICE) {
        return nullptr;
    }

    return h->vtable->local_data_mut(h->impl);
}

dtl_size_t dtl_tensor_stride(dtl_tensor_t tensor, int dim) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return h->vtable->stride(h->impl, dim);
}

// ============================================================================
// N-D Access
// ============================================================================

dtl_status dtl_tensor_get_local_nd(dtl_tensor_t tensor,
                                    const dtl_index_t* indices,
                                    void* value) {
    const auto* h = get_handle_const(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!indices || !value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->get_local_nd(h->impl, indices, value));
}

dtl_status dtl_tensor_set_local_nd(dtl_tensor_t tensor,
                                    const dtl_index_t* indices,
                                    const void* value) {
    auto* h = get_handle(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!indices || !value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->set_local_nd(h->impl, indices, value));
}

dtl_status dtl_tensor_get_local(dtl_tensor_t tensor, dtl_size_t linear_idx, void* value) {
    const auto* h = get_handle_const(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->get_local(h->impl, linear_idx, value));
}

dtl_status dtl_tensor_set_local(dtl_tensor_t tensor, dtl_size_t linear_idx, const void* value) {
    auto* h = get_handle(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->set_local(h->impl, linear_idx, value));
}

// ============================================================================
// Distribution Queries
// ============================================================================

dtl_rank_t dtl_tensor_num_ranks(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return h->vtable->num_ranks(h->impl);
}

dtl_rank_t dtl_tensor_rank(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return DTL_NO_RANK;
    return h->vtable->rank(h->impl);
}

int dtl_tensor_distributed_dim(dtl_tensor_t tensor) {
    const auto* h = get_handle_const(tensor);
    if (!h) return 0;
    return h->vtable->distributed_dim(h->impl);
}

// ============================================================================
// Collective Operations
// ============================================================================

dtl_status dtl_tensor_reshape(dtl_tensor_t tensor, dtl_shape new_shape) {
    auto* h = get_handle(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, h->vtable->reshape(h->impl, &new_shape));
}

dtl_status dtl_tensor_fill_local(dtl_tensor_t tensor, const void* value) {
    auto* h = get_handle(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!value) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    return apply_error_policy(h, h->vtable->fill(h->impl, value));
}

dtl_status dtl_tensor_barrier(dtl_tensor_t tensor) {
    auto* h = get_handle(tensor);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, dtl_barrier(h->base.ctx));
}

// ============================================================================
// Validation
// ============================================================================

int dtl_tensor_is_valid(dtl_tensor_t tensor) {
    return get_handle_const(tensor) != nullptr ? 1 : 0;
}

}  // extern "C"
