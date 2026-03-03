// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_context.cpp
 * @brief DTL C bindings - Context implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <cstring>

#ifdef DTL_HAS_MPI
#include <mpi.h>
#endif

#ifdef DTL_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef DTL_HAS_NCCL
#include <nccl.h>
#endif

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// Track whether we initialized MPI
static std::once_flag mpi_init_flag;
static bool mpi_was_initialized_by_us = false;

// ============================================================================
// MPI Initialization Helpers
// ============================================================================

#ifdef DTL_HAS_MPI

static dtl_status ensure_mpi_initialized(bool init_if_needed) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized) {
        return DTL_SUCCESS;
    }

    if (!init_if_needed) {
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }

    // Initialize MPI
    int provided;
    int err = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_MPI;
    }

    mpi_was_initialized_by_us = true;
    return DTL_SUCCESS;
}

#endif  // DTL_HAS_MPI

// ============================================================================
// Context Lifecycle
// ============================================================================

extern "C" {

dtl_status dtl_context_create(dtl_context_t* ctx, const dtl_context_options* opts) {
    if (!ctx) {
        return DTL_ERROR_NULL_POINTER;
    }

    // Use default options if none provided
    dtl_context_options default_opts;
    dtl_context_options_init(&default_opts);
    if (!opts) {
        opts = &default_opts;
    }

    // Allocate context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Initialize fields
    impl->device_id = opts->device_id;
    impl->determinism_mode = opts->reserved[0];
    impl->reduction_schedule_policy = opts->reserved[1];
    impl->progress_ordering_policy = opts->reserved[2];
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = dtl_context_s::HAS_CPU;  // Always have CPU domain
    impl->error_handler = nullptr;
    impl->error_handler_user_data = nullptr;

#ifdef DTL_HAS_MPI
    // Ensure MPI is initialized
    dtl_status status = ensure_mpi_initialized(opts->init_mpi != 0);
    if (status != DTL_SUCCESS) {
        delete impl;
        return status;
    }

    impl->initialized_mpi = mpi_was_initialized_by_us;
    impl->finalize_mpi = (opts->finalize_mpi != 0);

    // Duplicate MPI_COMM_WORLD for this context
    int err = MPI_Comm_dup(MPI_COMM_WORLD, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    impl->owns_comm = true;

    // Get rank and size
    MPI_Comm_rank(impl->comm, &impl->rank);
    MPI_Comm_size(impl->comm, &impl->size);

    impl->domain_flags |= dtl_context_s::HAS_MPI;

#else
    // Non-MPI build: single process
    impl->rank = 0;
    impl->size = 1;
#endif

    // Set CUDA flag if device is specified
    if (impl->device_id >= 0) {
        impl->domain_flags |= dtl_context_s::HAS_CUDA;
    }

    *ctx = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_context_create_default(dtl_context_t* ctx) {
    return dtl_context_create(ctx, nullptr);
}

void dtl_context_destroy(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return;
    }

#ifdef DTL_HAS_NCCL
    // Destroy NCCL communicator
    if (ctx->domain_flags & dtl_context_s::HAS_NCCL) {
        if (ctx->nccl_comm != nullptr) {
            ncclCommDestroy(static_cast<ncclComm_t>(ctx->nccl_comm));
            ctx->nccl_comm = nullptr;
        }
        if (ctx->cuda_stream != nullptr) {
            cudaStreamDestroy(static_cast<cudaStream_t>(ctx->cuda_stream));
            ctx->cuda_stream = nullptr;
        }
    }
#endif

#ifdef DTL_HAS_MPI
    // Free our communicator
    if (ctx->owns_comm && ctx->comm != MPI_COMM_NULL) {
        int mpi_initialized = 0;
        int mpi_finalized = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);
        if (mpi_initialized && !mpi_finalized) {
            MPI_Comm_free(&ctx->comm);
        }
        ctx->comm = MPI_COMM_NULL;
    }

    // Finalize MPI if requested and we initialized it
    if (ctx->finalize_mpi && ctx->initialized_mpi) {
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
    }
#endif

    // Invalidate magic before deletion
    ctx->magic = 0;
    delete ctx;
}

// ============================================================================
// Context Queries
// ============================================================================

dtl_rank_t dtl_context_rank(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_NO_RANK;
    }
    return ctx->rank;
}

dtl_rank_t dtl_context_size(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return ctx->size;
}

int dtl_context_is_root(dtl_context_t ctx) {
    return (dtl_context_rank(ctx) == 0) ? 1 : 0;
}

int dtl_context_device_id(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return -1;
    }
    return ctx->device_id;
}

int dtl_context_has_device(dtl_context_t ctx) {
    return (dtl_context_device_id(ctx) >= 0) ? 1 : 0;
}

int dtl_context_determinism_mode(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_DETERMINISM_THROUGHPUT;
    }
    return ctx->determinism_mode;
}

int dtl_context_reduction_schedule_policy(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_REDUCTION_SCHEDULE_IMPLEMENTATION_DEFINED;
    }
    return ctx->reduction_schedule_policy;
}

int dtl_context_progress_ordering_policy(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_PROGRESS_ORDERING_IMPLEMENTATION_DEFINED;
    }
    return ctx->progress_ordering_policy;
}

// ============================================================================
// Synchronization
// ============================================================================

dtl_status dtl_context_barrier(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);

    if (!mpi_initialized || mpi_finalized || ctx->comm == MPI_COMM_NULL) {
        return DTL_ERROR_INVALID_STATE;
    }

    int err = MPI_Barrier(ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_BARRIER_FAILED;
    }
#endif

    return DTL_SUCCESS;
}

dtl_status dtl_context_fence(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Memory fence (no inter-process synchronization)
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return DTL_SUCCESS;
}

// ============================================================================
// Context Validation
// ============================================================================

int dtl_context_is_valid(dtl_context_t ctx) {
    return (ctx && ctx->magic == dtl_context_s::VALID_MAGIC) ? 1 : 0;
}

// ============================================================================
// Context Duplication
// ============================================================================

dtl_status dtl_context_dup(dtl_context_t src, dtl_context_t* dst) {
    if (!dst) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!src || src->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Allocate new context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Copy fields
    impl->device_id = src->device_id;
    impl->determinism_mode = src->determinism_mode;
    impl->reduction_schedule_policy = src->reduction_schedule_policy;
    impl->progress_ordering_policy = src->progress_ordering_policy;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->rank = src->rank;
    impl->size = src->size;
    impl->domain_flags = src->domain_flags;
    impl->error_handler = src->error_handler;
    impl->error_handler_user_data = src->error_handler_user_data;

#ifdef DTL_HAS_MPI
    // Duplicate the communicator
    int err = MPI_Comm_dup(src->comm, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    impl->owns_comm = true;
    impl->initialized_mpi = false;  // Don't track MPI init for duplicates
    impl->finalize_mpi = false;
#endif

    *dst = impl;
    return DTL_SUCCESS;
}

// ============================================================================
// Domain Queries (V1.3.0)
// ============================================================================

int dtl_context_has_mpi(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return (ctx->domain_flags & dtl_context_s::HAS_MPI) ? 1 : 0;
}

int dtl_context_has_cuda(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return (ctx->domain_flags & dtl_context_s::HAS_CUDA) ? 1 : 0;
}

int dtl_context_has_nccl(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return (ctx->domain_flags & dtl_context_s::HAS_NCCL) ? 1 : 0;
}

int dtl_context_has_shmem(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return (ctx->domain_flags & dtl_context_s::HAS_SHMEM) ? 1 : 0;
}

// ============================================================================
// Context Splitting (V1.3.0)
// ============================================================================

dtl_status dtl_context_split(dtl_context_t ctx, int color, int key,
                              dtl_context_t* out) {
    if (!out) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    if (!(ctx->domain_flags & dtl_context_s::HAS_MPI)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    // Allocate new context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Split the communicator
    int err = MPI_Comm_split(ctx->comm, color, key, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }

    // Get new rank and size
    MPI_Comm_rank(impl->comm, &impl->rank);
    MPI_Comm_size(impl->comm, &impl->size);

    // Copy other fields
    impl->device_id = ctx->device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags;
    impl->owns_comm = true;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

    *out = impl;
    return DTL_SUCCESS;
#else
    return DTL_ERROR_NOT_SUPPORTED;
#endif
}

dtl_status dtl_context_with_cuda(dtl_context_t ctx, int device_id,
                                  dtl_context_t* out) {
    if (!out) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Allocate new context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Copy fields from source
    impl->rank = ctx->rank;
    impl->size = ctx->size;
    impl->device_id = device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags | dtl_context_s::HAS_CUDA;
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

#ifdef DTL_HAS_MPI
    if (ctx->domain_flags & dtl_context_s::HAS_MPI) {
        int err = MPI_Comm_dup(ctx->comm, &impl->comm);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        impl->owns_comm = true;
        impl->initialized_mpi = false;
        impl->finalize_mpi = false;
    }
#endif

    *out = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_context_with_nccl(dtl_context_t ctx, int device_id,
                                  dtl_context_t* out) {
    if (!out) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // NCCL requires MPI domain
    if (!(ctx->domain_flags & dtl_context_s::HAS_MPI)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

#if defined(DTL_HAS_NCCL) && defined(DTL_HAS_MPI)
    // Verify CUDA device
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }
    if (device_id < 0 || device_id >= device_count) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Set the device
    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }

    // Allocate new context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Copy fields from source
    impl->rank = ctx->rank;
    impl->size = ctx->size;
    impl->device_id = device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags | dtl_context_s::HAS_CUDA | dtl_context_s::HAS_NCCL;
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

    // Duplicate MPI communicator
    int mpi_err = MPI_Comm_dup(ctx->comm, &impl->comm);
    if (mpi_err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    impl->owns_comm = true;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;

    // Step 1: Rank 0 generates NCCL unique ID
    ncclUniqueId unique_id;
    if (impl->rank == 0) {
        ncclResult_t nccl_err = ncclGetUniqueId(&unique_id);
        if (nccl_err != ncclSuccess) {
            int failure_flag = -1;
            MPI_Bcast(&failure_flag, 1, MPI_INT, 0, impl->comm);
            MPI_Comm_free(&impl->comm);
            delete impl;
            return DTL_ERROR_BACKEND_INIT_FAILED;
        }
    }

    // Step 2: Broadcast success flag and unique ID
    int success_flag = (impl->rank == 0) ? 1 : 0;
    MPI_Bcast(&success_flag, 1, MPI_INT, 0, impl->comm);
    if (success_flag < 0) {
        MPI_Comm_free(&impl->comm);
        delete impl;
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }

    MPI_Bcast(&unique_id, sizeof(ncclUniqueId), MPI_BYTE, 0, impl->comm);

    // Step 3: Create CUDA stream for NCCL operations
    cudaStream_t stream = nullptr;
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
        MPI_Comm_free(&impl->comm);
        delete impl;
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }

    // Step 4: Initialize NCCL communicator
    ncclComm_t nccl_comm = nullptr;
    ncclResult_t nccl_err = ncclCommInitRank(&nccl_comm, impl->size, unique_id, impl->rank);
    if (nccl_err != ncclSuccess) {
        cudaStreamDestroy(stream);
        MPI_Comm_free(&impl->comm);
        delete impl;
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }

    impl->nccl_comm = nccl_comm;
    impl->cuda_stream = stream;

    *out = impl;
    return DTL_SUCCESS;
#else
    (void)device_id;
    return DTL_ERROR_NOT_SUPPORTED;
#endif
}

}  // extern "C"
