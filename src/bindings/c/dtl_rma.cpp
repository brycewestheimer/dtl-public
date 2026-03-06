// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_rma.cpp
 * @brief DTL C bindings - RMA operations implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_rma.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_types.h>

#include "dtl_internal.hpp"

#include <cstring>
#include <memory>
#include <vector>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// Internal Structures
// ============================================================================

/**
 * Memory window wrapper
 *
 * In single-process mode, RMA operations are simulated via local memory.
 * In multi-process mode (with MPI), would use MPI_Win.
 */
struct dtl_window_s {
    // Base pointer (user-provided or allocated)
    void* base;

    // Size in bytes
    dtl_size_t size;

    // Whether we own the memory
    bool owns_memory;

    // Context info for multi-rank operations
    dtl_rank_t rank;
    dtl_rank_t num_ranks;

    // Lock state tracking (for passive-target)
    bool locked_all;
    std::vector<bool> locked_targets;
    std::vector<dtl_lock_mode> lock_modes;

    // Fence epoch state
    bool in_fence_epoch;

#ifdef DTL_HAS_MPI
    MPI_Win mpi_win;
#endif

    // Validation magic
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xDEADC0DE;
};

// ============================================================================
// Validation Helpers
// ============================================================================

static bool is_valid_window(dtl_window_t win) {
    return win && win->magic == dtl_window_s::VALID_MAGIC;
}

// ============================================================================
// MPI Type/Op Conversion Helpers
// ============================================================================

#ifdef DTL_HAS_MPI

static MPI_Datatype dtl_to_mpi_datatype(dtl_dtype dtype) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    return MPI_INT8_T;
        case DTL_DTYPE_INT16:   return MPI_INT16_T;
        case DTL_DTYPE_INT32:   return MPI_INT32_T;
        case DTL_DTYPE_INT64:   return MPI_INT64_T;
        case DTL_DTYPE_UINT8:   return MPI_UINT8_T;
        case DTL_DTYPE_UINT16:  return MPI_UINT16_T;
        case DTL_DTYPE_UINT32:  return MPI_UINT32_T;
        case DTL_DTYPE_UINT64:  return MPI_UINT64_T;
        case DTL_DTYPE_FLOAT32: return MPI_FLOAT;
        case DTL_DTYPE_FLOAT64: return MPI_DOUBLE;
        default:                return MPI_DATATYPE_NULL;
    }
}

static MPI_Op dtl_to_mpi_op(dtl_reduce_op op) {
    switch (op) {
        case DTL_OP_SUM:  return MPI_SUM;
        case DTL_OP_PROD: return MPI_PROD;
        case DTL_OP_MIN:  return MPI_MIN;
        case DTL_OP_MAX:  return MPI_MAX;
        case DTL_OP_BAND: return MPI_BAND;
        case DTL_OP_BOR:  return MPI_BOR;
        case DTL_OP_BXOR: return MPI_BXOR;
        default:          return MPI_OP_NULL;
    }
}

#endif  // DTL_HAS_MPI

// ============================================================================
// Window Lifecycle
// ============================================================================

extern "C" {

dtl_status dtl_window_create(dtl_context_t ctx, void* base,
                              dtl_size_t size, dtl_window_t* win) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!win) {
        return DTL_ERROR_NULL_POINTER;
    }

    dtl_window_s* impl = nullptr;
    try {
        impl = new dtl_window_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->base = base;
    impl->size = size;
    impl->owns_memory = false;
    impl->rank = ctx->rank;
    impl->num_ranks = ctx->size;
    impl->locked_all = false;
    impl->in_fence_epoch = false;
    impl->magic = dtl_window_s::VALID_MAGIC;

    // Initialize lock tracking
    try {
        impl->locked_targets.resize(ctx->size, false);
        impl->lock_modes.resize(ctx->size, DTL_LOCK_SHARED);
    } catch (...) {
        delete impl;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

#ifdef DTL_HAS_MPI
    impl->mpi_win = MPI_WIN_NULL;
    if ((ctx->domain_flags & dtl_context_s::HAS_MPI) && ctx->size > 1) {
        MPI_Aint local_size = (base != nullptr) ? static_cast<MPI_Aint>(size) : 0;
        int err = MPI_Win_create(base, local_size, 1 /* disp_unit */,
                                 MPI_INFO_NULL, ctx->comm, &impl->mpi_win);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Win_set_errhandler(impl->mpi_win, MPI_ERRORS_RETURN);
    }
#endif

    *win = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_window_allocate(dtl_context_t ctx,
                                dtl_size_t size, dtl_window_t* win) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!win) {
        return DTL_ERROR_NULL_POINTER;
    }

    dtl_window_s* impl = nullptr;
    try {
        impl = new dtl_window_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Allocate memory
    try {
        impl->base = new char[size]();  // Zero-initialized
    } catch (...) {
        delete impl;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->size = size;
    impl->owns_memory = true;
    impl->rank = ctx->rank;
    impl->num_ranks = ctx->size;
    impl->locked_all = false;
    impl->in_fence_epoch = false;
    impl->magic = dtl_window_s::VALID_MAGIC;

    try {
        impl->locked_targets.resize(ctx->size, false);
        impl->lock_modes.resize(ctx->size, DTL_LOCK_SHARED);
    } catch (...) {
        delete[] static_cast<char*>(impl->base);
        delete impl;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

#ifdef DTL_HAS_MPI
    impl->mpi_win = MPI_WIN_NULL;
    if ((ctx->domain_flags & dtl_context_s::HAS_MPI) && ctx->size > 1) {
        void* alloc_base = nullptr;
        int err = MPI_Win_allocate(static_cast<MPI_Aint>(size), 1 /* disp_unit */,
                                   MPI_INFO_NULL, ctx->comm, &alloc_base, &impl->mpi_win);
        if (err == MPI_SUCCESS) {
            // Replace the locally-allocated base with MPI-allocated base
            delete[] static_cast<char*>(impl->base);
            impl->base = alloc_base;
            impl->owns_memory = false;  // Freed by MPI_Win_free
        } else {
            delete[] static_cast<char*>(impl->base);
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Win_set_errhandler(impl->mpi_win, MPI_ERRORS_RETURN);
    }
#endif

    *win = impl;
    return DTL_SUCCESS;
}

void dtl_window_destroy(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_free(&win->mpi_win);
        win->mpi_win = MPI_WIN_NULL;
    }
#endif

    if (win->owns_memory && win->base) {
        delete[] static_cast<char*>(win->base);
    }

    win->magic = 0;
    delete win;
}

// ============================================================================
// Window Queries
// ============================================================================

void* dtl_window_base(dtl_window_t win) {
    if (!is_valid_window(win)) return nullptr;
    return win->base;
}

dtl_size_t dtl_window_size(dtl_window_t win) {
    if (!is_valid_window(win)) return 0;
    return win->size;
}

int dtl_window_is_valid(dtl_window_t win) {
    return is_valid_window(win) ? 1 : 0;
}

// ============================================================================
// Active-Target Synchronization (Fence)
// ============================================================================

dtl_status dtl_window_fence(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Toggle fence epoch state
    win->in_fence_epoch = !win->in_fence_epoch;

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_fence(0, win->mpi_win);
    }
#endif

    // In single-process mode, fence is a no-op (all operations are local)
    return DTL_SUCCESS;
}

// ============================================================================
// Passive-Target Synchronization (Lock/Unlock)
// ============================================================================

dtl_status dtl_window_lock(dtl_window_t win, dtl_rank_t target,
                            dtl_lock_mode mode) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (win->locked_targets[target]) {
        // Already locked
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    win->locked_targets[target] = true;
    win->lock_modes[target] = mode;

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        int mpi_mode = (mode == DTL_LOCK_EXCLUSIVE) ? MPI_LOCK_EXCLUSIVE : MPI_LOCK_SHARED;
        MPI_Win_lock(mpi_mode, target, 0, win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_window_unlock(dtl_window_t win, dtl_rank_t target) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!win->locked_targets[target]) {
        // Not locked
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    win->locked_targets[target] = false;

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_unlock(target, win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_window_lock_all(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (win->locked_all) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    win->locked_all = true;

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_lock_all(0, win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_window_unlock_all(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!win->locked_all) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    win->locked_all = false;

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_unlock_all(win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

// ============================================================================
// Flush Operations
// ============================================================================

dtl_status dtl_window_flush(dtl_window_t win, dtl_rank_t target) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_flush(target, win->mpi_win);
    }
#endif

    // In single-process mode, operations are always complete
    return DTL_SUCCESS;
}

dtl_status dtl_window_flush_all(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_flush_all(win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_window_flush_local(dtl_window_t win, dtl_rank_t target) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_flush_local(target, win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_window_flush_local_all(dtl_window_t win) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Win_flush_local_all(win->mpi_win);
    }
#endif

    return DTL_SUCCESS;
}

// ============================================================================
// Data Transfer Operations
// ============================================================================

dtl_status dtl_rma_put(dtl_window_t win, dtl_rank_t target,
                        dtl_size_t target_offset, const void* origin,
                        dtl_size_t size) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!origin) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target_offset + size > win->size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    // In single-process mode: if target is self, do local copy
    if (target == win->rank && win->base) {
        char* dst = static_cast<char*>(win->base) + target_offset;
        std::memcpy(dst, origin, size);
        return DTL_SUCCESS;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        int mpi_size = 0;
        if (!checked_size_to_int(size, &mpi_size)) {
            return DTL_ERROR_OUT_OF_RANGE;
        }
        MPI_Put(origin, mpi_size, MPI_BYTE,
                target, static_cast<MPI_Aint>(target_offset),
                mpi_size, MPI_BYTE, win->mpi_win);
        return DTL_SUCCESS;
    }
#endif

    // In single-process mode, cannot put to other ranks
    if (win->num_ranks == 1) {
        return DTL_SUCCESS;  // No-op for single rank
    }

    return DTL_ERROR_NOT_SUPPORTED;
}

dtl_status dtl_rma_get(dtl_window_t win, dtl_rank_t target,
                        dtl_size_t target_offset, void* buffer,
                        dtl_size_t size) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buffer) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (target_offset + size > win->size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    // In single-process mode: if target is self, do local copy
    if (target == win->rank && win->base) {
        const char* src = static_cast<const char*>(win->base) + target_offset;
        std::memcpy(buffer, src, size);
        return DTL_SUCCESS;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        int mpi_size = 0;
        if (!checked_size_to_int(size, &mpi_size)) {
            return DTL_ERROR_OUT_OF_RANGE;
        }
        MPI_Get(buffer, mpi_size, MPI_BYTE,
                target, static_cast<MPI_Aint>(target_offset),
                mpi_size, MPI_BYTE, win->mpi_win);
        return DTL_SUCCESS;
    }
#endif

    if (win->num_ranks == 1) {
        return DTL_SUCCESS;
    }

    return DTL_ERROR_NOT_SUPPORTED;
}

// ============================================================================
// Asynchronous Operations
// ============================================================================

dtl_status dtl_rma_put_async(dtl_window_t win, dtl_rank_t target,
                              dtl_size_t target_offset,
                              const void* origin, dtl_size_t size,
                              dtl_request_t* req) {
    if (!req) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    // In MPI mode, use true async Rput for remote targets.
    if (is_valid_window(win) &&
        win->mpi_win != MPI_WIN_NULL &&
        target != win->rank) {
        if (!origin) {
            return DTL_ERROR_NULL_POINTER;
        }
        if (target < 0 || target >= win->num_ranks) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }
        if (target_offset + size > win->size) {
            return DTL_ERROR_OUT_OF_BOUNDS;
        }

        int mpi_size = 0;
        if (!checked_size_to_int(size, &mpi_size)) {
            return DTL_ERROR_OUT_OF_RANGE;
        }

        dtl_request_s* impl = nullptr;
        try {
            impl = new dtl_request_s();
        } catch (...) {
            return DTL_ERROR_ALLOCATION_FAILED;
        }

        impl->magic = dtl_request_s::VALID_MAGIC;
        impl->state = std::make_shared<dtl_request_s::async_state>();
        impl->state->completed.store(false, std::memory_order_release);
        impl->is_mpi_request = true;

        int err = MPI_Rput(origin, mpi_size, MPI_BYTE,
                           target, static_cast<MPI_Aint>(target_offset),
                           mpi_size, MPI_BYTE, win->mpi_win, &impl->mpi_request);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_SEND_FAILED;
        }

        *req = impl;
        return DTL_SUCCESS;
    }
#endif

    // Fallback path: perform operation synchronously and return completed request.
    dtl_status status = dtl_rma_put(win, target, target_offset, origin, size);
    if (status != DTL_SUCCESS) {
        return status;
    }

    dtl_request_s* impl = nullptr;
    try {
        impl = new dtl_request_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->magic = dtl_request_s::VALID_MAGIC;
    impl->state = std::make_shared<dtl_request_s::async_state>();
    impl->state->completed.store(true, std::memory_order_release);

#ifdef DTL_HAS_MPI
    impl->mpi_request = MPI_REQUEST_NULL;
    impl->is_mpi_request = false;
#endif

    *req = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_rma_get_async(dtl_window_t win, dtl_rank_t target,
                              dtl_size_t target_offset,
                              void* buffer, dtl_size_t size,
                              dtl_request_t* req) {
    if (!req) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    // In MPI mode, use true async Rget for remote targets.
    if (is_valid_window(win) &&
        win->mpi_win != MPI_WIN_NULL &&
        target != win->rank) {
        if (!buffer) {
            return DTL_ERROR_NULL_POINTER;
        }
        if (target < 0 || target >= win->num_ranks) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }
        if (target_offset + size > win->size) {
            return DTL_ERROR_OUT_OF_BOUNDS;
        }

        int mpi_size = 0;
        if (!checked_size_to_int(size, &mpi_size)) {
            return DTL_ERROR_OUT_OF_RANGE;
        }

        dtl_request_s* impl = nullptr;
        try {
            impl = new dtl_request_s();
        } catch (...) {
            return DTL_ERROR_ALLOCATION_FAILED;
        }

        impl->magic = dtl_request_s::VALID_MAGIC;
        impl->state = std::make_shared<dtl_request_s::async_state>();
        impl->state->completed.store(false, std::memory_order_release);
        impl->is_mpi_request = true;

        int err = MPI_Rget(buffer, mpi_size, MPI_BYTE,
                           target, static_cast<MPI_Aint>(target_offset),
                           mpi_size, MPI_BYTE, win->mpi_win, &impl->mpi_request);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_RECV_FAILED;
        }

        *req = impl;
        return DTL_SUCCESS;
    }
#endif

    // Fallback path: perform operation synchronously and return completed request.
    dtl_status status = dtl_rma_get(win, target, target_offset, buffer, size);
    if (status != DTL_SUCCESS) {
        return status;
    }

    dtl_request_s* impl = nullptr;
    try {
        impl = new dtl_request_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->magic = dtl_request_s::VALID_MAGIC;
    impl->state = std::make_shared<dtl_request_s::async_state>();
    impl->state->completed.store(true, std::memory_order_release);

#ifdef DTL_HAS_MPI
    impl->mpi_request = MPI_REQUEST_NULL;
    impl->is_mpi_request = false;
#endif

    *req = impl;
    return DTL_SUCCESS;
}

// ============================================================================
// Atomic Operations
// ============================================================================

dtl_status dtl_rma_accumulate(dtl_window_t win, dtl_rank_t target,
                               dtl_size_t target_offset,
                               const void* origin, dtl_size_t size,
                               dtl_dtype dtype, dtl_reduce_op op) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!origin) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    dtl_size_t elem_size = dtl_dtype_size(dtype);
    if (elem_size == 0 || size % elem_size != 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    dtl_size_t num_elements = size / elem_size;
    if (target_offset + size > win->size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    // Single-process mode: local accumulation
    if (target == win->rank && win->base) {
        char* dst = static_cast<char*>(win->base) + target_offset;
        const char* src = static_cast<const char*>(origin);

        // Apply reduction operation element by element
        for (dtl_size_t i = 0; i < num_elements; ++i) {
            switch (dtype) {
                case DTL_DTYPE_FLOAT64: {
                    double* d = reinterpret_cast<double*>(dst + i * elem_size);
                    double s = *reinterpret_cast<const double*>(src + i * elem_size);
                    switch (op) {
                        case DTL_OP_SUM:  *d += s; break;
                        case DTL_OP_PROD: *d *= s; break;
                        case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                        case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                        default: return DTL_ERROR_NOT_SUPPORTED;
                    }
                    break;
                }
                case DTL_DTYPE_FLOAT32: {
                    float* d = reinterpret_cast<float*>(dst + i * elem_size);
                    float s = *reinterpret_cast<const float*>(src + i * elem_size);
                    switch (op) {
                        case DTL_OP_SUM:  *d += s; break;
                        case DTL_OP_PROD: *d *= s; break;
                        case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                        case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                        default: return DTL_ERROR_NOT_SUPPORTED;
                    }
                    break;
                }
                case DTL_DTYPE_INT32: {
                    int32_t* d = reinterpret_cast<int32_t*>(dst + i * elem_size);
                    int32_t s = *reinterpret_cast<const int32_t*>(src + i * elem_size);
                    switch (op) {
                        case DTL_OP_SUM:  *d += s; break;
                        case DTL_OP_PROD: *d *= s; break;
                        case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                        case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                        case DTL_OP_BAND: *d &= s; break;
                        case DTL_OP_BOR:  *d |= s; break;
                        case DTL_OP_BXOR: *d ^= s; break;
                        default: return DTL_ERROR_NOT_SUPPORTED;
                    }
                    break;
                }
                case DTL_DTYPE_INT64: {
                    int64_t* d = reinterpret_cast<int64_t*>(dst + i * elem_size);
                    int64_t s = *reinterpret_cast<const int64_t*>(src + i * elem_size);
                    switch (op) {
                        case DTL_OP_SUM:  *d += s; break;
                        case DTL_OP_PROD: *d *= s; break;
                        case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                        case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                        case DTL_OP_BAND: *d &= s; break;
                        case DTL_OP_BOR:  *d |= s; break;
                        case DTL_OP_BXOR: *d ^= s; break;
                        default: return DTL_ERROR_NOT_SUPPORTED;
                    }
                    break;
                }
                default:
                    // For other types, just do sum
                    std::memcpy(dst + i * elem_size, src + i * elem_size, elem_size);
                    break;
            }
        }
        return DTL_SUCCESS;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Datatype mpi_type = dtl_to_mpi_datatype(dtype);
        MPI_Op mpi_op = dtl_to_mpi_op(op);
        if (mpi_type != MPI_DATATYPE_NULL && mpi_op != MPI_OP_NULL) {
            int mpi_count = 0;
            if (checked_size_to_int(num_elements, &mpi_count) != DTL_SUCCESS) {
                return DTL_ERROR_OUT_OF_RANGE;
            }
            MPI_Accumulate(origin, mpi_count, mpi_type,
                          target, static_cast<MPI_Aint>(target_offset),
                          mpi_count, mpi_type,
                          mpi_op, win->mpi_win);
            return DTL_SUCCESS;
        }
    }
#endif

    if (win->num_ranks == 1) {
        return DTL_SUCCESS;
    }

    return DTL_ERROR_NOT_SUPPORTED;
}

dtl_status dtl_rma_fetch_and_op(dtl_window_t win, dtl_rank_t target,
                                 dtl_size_t target_offset,
                                 const void* origin, void* result,
                                 dtl_dtype dtype, dtl_reduce_op op) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!origin || !result) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    dtl_size_t elem_size = dtl_dtype_size(dtype);
    if (target_offset + elem_size > win->size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    // Single-process mode: local fetch-and-op
    if (target == win->rank && win->base) {
        char* dst = static_cast<char*>(win->base) + target_offset;

        // First, copy old value to result
        std::memcpy(result, dst, elem_size);

        // Then apply operation
        switch (dtype) {
            case DTL_DTYPE_FLOAT64: {
                double* d = reinterpret_cast<double*>(dst);
                double s = *static_cast<const double*>(origin);
                switch (op) {
                    case DTL_OP_SUM:  *d += s; break;
                    case DTL_OP_PROD: *d *= s; break;
                    case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                    case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                    default: return DTL_ERROR_NOT_SUPPORTED;
                }
                break;
            }
            case DTL_DTYPE_FLOAT32: {
                float* d = reinterpret_cast<float*>(dst);
                float s = *static_cast<const float*>(origin);
                switch (op) {
                    case DTL_OP_SUM:  *d += s; break;
                    case DTL_OP_PROD: *d *= s; break;
                    case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                    case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                    default: return DTL_ERROR_NOT_SUPPORTED;
                }
                break;
            }
            case DTL_DTYPE_INT32: {
                int32_t* d = reinterpret_cast<int32_t*>(dst);
                int32_t s = *static_cast<const int32_t*>(origin);
                switch (op) {
                    case DTL_OP_SUM:  *d += s; break;
                    case DTL_OP_PROD: *d *= s; break;
                    case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                    case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                    case DTL_OP_BAND: *d &= s; break;
                    case DTL_OP_BOR:  *d |= s; break;
                    case DTL_OP_BXOR: *d ^= s; break;
                    default: return DTL_ERROR_NOT_SUPPORTED;
                }
                break;
            }
            case DTL_DTYPE_INT64: {
                int64_t* d = reinterpret_cast<int64_t*>(dst);
                int64_t s = *static_cast<const int64_t*>(origin);
                switch (op) {
                    case DTL_OP_SUM:  *d += s; break;
                    case DTL_OP_PROD: *d *= s; break;
                    case DTL_OP_MIN:  *d = (*d < s) ? *d : s; break;
                    case DTL_OP_MAX:  *d = (*d > s) ? *d : s; break;
                    case DTL_OP_BAND: *d &= s; break;
                    case DTL_OP_BOR:  *d |= s; break;
                    case DTL_OP_BXOR: *d ^= s; break;
                    default: return DTL_ERROR_NOT_SUPPORTED;
                }
                break;
            }
            default:
                return DTL_ERROR_NOT_SUPPORTED;
        }
        return DTL_SUCCESS;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Datatype mpi_type = dtl_to_mpi_datatype(dtype);
        MPI_Op mpi_op = dtl_to_mpi_op(op);
        if (mpi_type != MPI_DATATYPE_NULL && mpi_op != MPI_OP_NULL) {
            MPI_Fetch_and_op(origin, result, mpi_type, target,
                            static_cast<MPI_Aint>(target_offset),
                            mpi_op, win->mpi_win);
            return DTL_SUCCESS;
        }
    }
#endif

    if (win->num_ranks == 1) {
        return DTL_SUCCESS;
    }

    return DTL_ERROR_NOT_SUPPORTED;
}

dtl_status dtl_rma_compare_and_swap(dtl_window_t win, dtl_rank_t target,
                                     dtl_size_t target_offset,
                                     const void* compare,
                                     const void* swap, void* result,
                                     dtl_dtype dtype) {
    if (!is_valid_window(win)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!compare || !swap || !result) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (target < 0 || target >= win->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (dtype < 0 || dtype >= DTL_DTYPE_COUNT) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    dtl_size_t elem_size = dtl_dtype_size(dtype);
    if (target_offset + elem_size > win->size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    // Single-process mode: local compare-and-swap
    if (target == win->rank && win->base) {
        char* dst = static_cast<char*>(win->base) + target_offset;

        // Copy old value to result
        std::memcpy(result, dst, elem_size);

        // Compare and conditionally swap
        if (std::memcmp(dst, compare, elem_size) == 0) {
            std::memcpy(dst, swap, elem_size);
        }
        return DTL_SUCCESS;
    }

#ifdef DTL_HAS_MPI
    if (win->mpi_win != MPI_WIN_NULL) {
        MPI_Datatype mpi_type = dtl_to_mpi_datatype(dtype);
        if (mpi_type != MPI_DATATYPE_NULL) {
            MPI_Compare_and_swap(swap, compare, result, mpi_type,
                                target, static_cast<MPI_Aint>(target_offset),
                                win->mpi_win);
            return DTL_SUCCESS;
        }
    }
#endif

    if (win->num_ranks == 1) {
        return DTL_SUCCESS;
    }

    return DTL_ERROR_NOT_SUPPORTED;
}

}  // extern "C"

// Note: Request management functions (dtl_request_wait, dtl_request_test,
// dtl_request_free) are defined in dtl_communicator.cpp and handle both
// point-to-point and RMA requests.
