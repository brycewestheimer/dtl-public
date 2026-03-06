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

static dtl_status validate_cuda_device_id(int device_id) {
    if (device_id < 0) {
        return DTL_SUCCESS;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return DTL_ERROR_BACKEND_INIT_FAILED;
    }
    if (device_count <= 0) {
        return DTL_ERROR_BACKEND_UNAVAILABLE;
    }
    if (device_id >= device_count) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return DTL_SUCCESS;
}
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

#ifdef DTL_HAS_NCCL

static bool nccl_mode_valid(int mode) {
    return mode == DTL_NCCL_MODE_NATIVE_ONLY || mode == DTL_NCCL_MODE_HYBRID_PARITY;
}

static bool nccl_op_supported_native(dtl_nccl_operation op) {
    switch (op) {
        case DTL_NCCL_OP_POINT_TO_POINT:
        case DTL_NCCL_OP_BARRIER:
        case DTL_NCCL_OP_BROADCAST:
        case DTL_NCCL_OP_REDUCE:
        case DTL_NCCL_OP_ALLREDUCE:
        case DTL_NCCL_OP_GATHER:
        case DTL_NCCL_OP_SCATTER:
        case DTL_NCCL_OP_ALLGATHER:
        case DTL_NCCL_OP_ALLTOALL:
            return true;
        case DTL_NCCL_OP_GATHERV:
        case DTL_NCCL_OP_SCATTERV:
        case DTL_NCCL_OP_ALLGATHERV:
        case DTL_NCCL_OP_ALLTOALLV:
        case DTL_NCCL_OP_SCAN:
        case DTL_NCCL_OP_EXSCAN:
        case DTL_NCCL_OP_LOGICAL_REDUCTION:
            return false;
    }
    return false;
}

static bool nccl_op_supported_hybrid(dtl_nccl_operation op) {
    switch (op) {
        case DTL_NCCL_OP_POINT_TO_POINT:
        case DTL_NCCL_OP_BARRIER:
        case DTL_NCCL_OP_BROADCAST:
        case DTL_NCCL_OP_REDUCE:
        case DTL_NCCL_OP_ALLREDUCE:
        case DTL_NCCL_OP_GATHER:
        case DTL_NCCL_OP_SCATTER:
        case DTL_NCCL_OP_ALLGATHER:
        case DTL_NCCL_OP_ALLTOALL:
        case DTL_NCCL_OP_GATHERV:
        case DTL_NCCL_OP_SCATTERV:
        case DTL_NCCL_OP_ALLGATHERV:
        case DTL_NCCL_OP_ALLTOALLV:
        case DTL_NCCL_OP_SCAN:
        case DTL_NCCL_OP_EXSCAN:
        case DTL_NCCL_OP_LOGICAL_REDUCTION:
            return true;
    }
    return false;
}

static void clear_nccl_resources(dtl_context_t ctx) {
    if (!ctx) {
        return;
    }
#ifdef DTL_HAS_CUDA
    if (ctx->barrier_scratch != nullptr) {
        cudaFree(ctx->barrier_scratch);
        ctx->barrier_scratch = nullptr;
    }
    if (ctx->cuda_stream != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(ctx->cuda_stream));
        ctx->cuda_stream = nullptr;
    }
#endif
    if (ctx->nccl_comm != nullptr) {
        ncclCommDestroy(static_cast<ncclComm_t>(ctx->nccl_comm));
        ctx->nccl_comm = nullptr;
    }
}

static dtl_status init_nccl_for_context(dtl_context_t ctx, int device_id) {
    if (!ctx) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
#if !defined(DTL_HAS_MPI) || !defined(DTL_HAS_CUDA)
    (void)device_id;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#else
    if (!(ctx->domain_flags & dtl_context_s::HAS_MPI)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (device_id < 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    dtl_status cuda_status = validate_cuda_device_id(device_id);
    if (cuda_status != DTL_SUCCESS) {
        return cuda_status;
    }

    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        return DTL_ERROR_CUDA;
    }

    cudaStream_t stream = nullptr;
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
        return DTL_ERROR_CUDA;
    }

    int* barrier_scratch = nullptr;
    cuda_err = cudaMalloc(&barrier_scratch, sizeof(int));
    if (cuda_err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return DTL_ERROR_CUDA;
    }
    int zero = 0;
    cuda_err = cudaMemcpy(barrier_scratch, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(barrier_scratch);
        cudaStreamDestroy(stream);
        return DTL_ERROR_CUDA;
    }

    ncclUniqueId id{};
    if (ctx->rank == 0) {
        ncclResult_t get_id_err = ncclGetUniqueId(&id);
        if (get_id_err != ncclSuccess) {
            cudaFree(barrier_scratch);
            cudaStreamDestroy(stream);
            return DTL_ERROR_NCCL;
        }
    }

    int mpi_err = MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, ctx->comm);
    if (mpi_err != MPI_SUCCESS) {
        cudaFree(barrier_scratch);
        cudaStreamDestroy(stream);
        return DTL_ERROR_MPI;
    }

    ncclComm_t comm = nullptr;
    ncclResult_t init_err = ncclCommInitRank(&comm, ctx->size, id, ctx->rank);
    if (init_err != ncclSuccess) {
        cudaFree(barrier_scratch);
        cudaStreamDestroy(stream);
        return DTL_ERROR_NCCL;
    }

    ctx->device_id = device_id;
    ctx->nccl_comm = static_cast<void*>(comm);
    ctx->cuda_stream = static_cast<void*>(stream);
    ctx->barrier_scratch = static_cast<void*>(barrier_scratch);
    ctx->domain_flags |= dtl_context_s::HAS_NCCL;
    ctx->domain_flags |= dtl_context_s::HAS_CUDA;
    return DTL_SUCCESS;
#endif
}

#endif  // DTL_HAS_NCCL

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
    impl->nccl_mode = DTL_NCCL_MODE_HYBRID_PARITY;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = dtl_context_s::HAS_CPU;  // Always have CPU domain
    impl->error_handler = nullptr;
    impl->error_handler_user_data = nullptr;

#ifdef DTL_HAS_NCCL
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;
#endif

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
    (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
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

#ifdef DTL_HAS_CUDA
    dtl_status cuda_status = validate_cuda_device_id(impl->device_id);
    if (cuda_status != DTL_SUCCESS) {
        dtl_context_destroy(impl);
        return cuda_status;
    }
    if (impl->device_id >= 0) {
        impl->domain_flags |= dtl_context_s::HAS_CUDA;
    }
#else
    if (impl->device_id >= 0) {
        dtl_context_destroy(impl);
        return DTL_ERROR_BACKEND_UNAVAILABLE;
    }
#endif

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
    // Destroy NCCL communicator and associated resources
    if (ctx->domain_flags & dtl_context_s::HAS_NCCL) {
        clear_nccl_resources(ctx);
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
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return ((ctx->domain_flags & dtl_context_s::HAS_CUDA) && ctx->device_id >= 0) ? 1 : 0;
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

    // Copy fields — clear NCCL flag since dup doesn't create a new NCCL communicator
    impl->device_id = src->device_id;
    impl->determinism_mode = src->determinism_mode;
    impl->reduction_schedule_policy = src->reduction_schedule_policy;
    impl->progress_ordering_policy = src->progress_ordering_policy;
    impl->nccl_mode = src->nccl_mode;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->rank = src->rank;
    impl->size = src->size;
    impl->domain_flags = src->domain_flags & ~dtl_context_s::HAS_NCCL;
    impl->error_handler = src->error_handler;
    impl->error_handler_user_data = src->error_handler_user_data;

#ifdef DTL_HAS_NCCL
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;
#endif

#ifdef DTL_HAS_MPI
    // Duplicate the communicator
    impl->comm = MPI_COMM_NULL;
    impl->owns_comm = false;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    int err = MPI_Comm_dup(src->comm, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
    impl->owns_comm = true;
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
    return ((ctx->domain_flags & dtl_context_s::HAS_CUDA) && ctx->device_id >= 0) ? 1 : 0;
}

int dtl_context_has_nccl(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    return (ctx->domain_flags & dtl_context_s::HAS_NCCL) ? 1 : 0;
}

int dtl_context_nccl_mode(dtl_context_t ctx) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return -1;
    }
    if (!(ctx->domain_flags & dtl_context_s::HAS_NCCL)) {
        return -1;
    }
    if (!nccl_mode_valid(ctx->nccl_mode)) {
        return -1;
    }
    return ctx->nccl_mode;
}

int dtl_context_nccl_supports_native(dtl_context_t ctx, dtl_nccl_operation op) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    if (!(ctx->domain_flags & dtl_context_s::HAS_NCCL)) {
        return 0;
    }
    return nccl_op_supported_native(op) ? 1 : 0;
}

int dtl_context_nccl_supports_hybrid(dtl_context_t ctx, dtl_nccl_operation op) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return 0;
    }
    if (!(ctx->domain_flags & dtl_context_s::HAS_NCCL)) {
        return 0;
    }
    if (ctx->nccl_mode != DTL_NCCL_MODE_HYBRID_PARITY) {
        return 0;
    }
    return nccl_op_supported_hybrid(op) ? 1 : 0;
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
    impl->comm = MPI_COMM_NULL;
    impl->owns_comm = false;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    int err = MPI_Comm_split(ctx->comm, color, key, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);

    // Get new rank and size
    MPI_Comm_rank(impl->comm, &impl->rank);
    MPI_Comm_size(impl->comm, &impl->size);

    // Copy other fields — clear NCCL flag since split doesn't create a new NCCL communicator
    impl->device_id = ctx->device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->nccl_mode = ctx->nccl_mode;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags & ~dtl_context_s::HAS_NCCL;
    impl->owns_comm = true;
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

#ifdef DTL_HAS_NCCL
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;
#endif

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

#ifdef DTL_HAS_CUDA
    dtl_status cuda_status = validate_cuda_device_id(device_id);
    if (cuda_status != DTL_SUCCESS) {
        return cuda_status;
    }
#else
    if (device_id >= 0) {
        return DTL_ERROR_BACKEND_UNAVAILABLE;
    }
#endif

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
    impl->nccl_mode = ctx->nccl_mode;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags & ~dtl_context_s::HAS_NCCL;
    if (device_id >= 0) {
        impl->domain_flags |= dtl_context_s::HAS_CUDA;
    } else {
        impl->domain_flags &= ~dtl_context_s::HAS_CUDA;
    }
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

#ifdef DTL_HAS_NCCL
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;
#endif

#ifdef DTL_HAS_MPI
    impl->comm = MPI_COMM_NULL;
    impl->owns_comm = false;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    if (ctx->domain_flags & dtl_context_s::HAS_MPI) {
        int err = MPI_Comm_dup(ctx->comm, &impl->comm);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
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
    return dtl_context_with_nccl_ex(
        ctx, device_id, DTL_NCCL_MODE_HYBRID_PARITY, out);
}

dtl_status dtl_context_with_nccl_ex(dtl_context_t ctx, int device_id,
                                     dtl_nccl_operation_mode mode,
                                     dtl_context_t* out) {
    if (!out) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    if (!nccl_mode_valid(mode)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#if !defined(DTL_HAS_NCCL) || !defined(DTL_HAS_CUDA)
    (void)device_id;
    (void)mode;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#else
    // NCCL requires MPI domain
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

    impl->rank = ctx->rank;
    impl->size = ctx->size;
    impl->device_id = device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->nccl_mode = mode;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = ctx->domain_flags | dtl_context_s::HAS_NCCL | dtl_context_s::HAS_CUDA;
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;

#ifdef DTL_HAS_MPI
    impl->comm = MPI_COMM_NULL;
    impl->owns_comm = false;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    if (ctx->domain_flags & dtl_context_s::HAS_MPI) {
        int err = MPI_Comm_dup(ctx->comm, &impl->comm);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
        impl->owns_comm = true;
    }
#endif

    dtl_status init_status = init_nccl_for_context(impl, device_id);
    if (init_status != DTL_SUCCESS) {
        dtl_context_destroy(impl);
        return init_status;
    }

    *out = impl;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_context_split_nccl(dtl_context_t ctx,
                                    int color, int key,
                                    dtl_context_t* out) {
    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    dtl_nccl_operation_mode mode = nccl_mode_valid(ctx->nccl_mode)
        ? static_cast<dtl_nccl_operation_mode>(ctx->nccl_mode)
        : DTL_NCCL_MODE_HYBRID_PARITY;
    return dtl_context_split_nccl_ex(
        ctx, color, key, ctx->device_id, mode, out);
}

dtl_status dtl_context_split_nccl_ex(dtl_context_t ctx,
                                      int color, int key,
                                      int device_id,
                                      dtl_nccl_operation_mode mode,
                                      dtl_context_t* out) {
    if (!out) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!ctx || ctx->magic != dtl_context_s::VALID_MAGIC) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    if (!nccl_mode_valid(mode)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#if !defined(DTL_HAS_NCCL) || !defined(DTL_HAS_CUDA) || !defined(DTL_HAS_MPI)
    (void)color;
    (void)key;
    (void)device_id;
    (void)mode;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#else
    // Requires both MPI and NCCL domains for legacy split behavior parity
    if (!(ctx->domain_flags & dtl_context_s::HAS_MPI)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!(ctx->domain_flags & dtl_context_s::HAS_NCCL)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    // Allocate new context
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->comm = MPI_COMM_NULL;
    impl->owns_comm = false;
    impl->initialized_mpi = false;
    impl->finalize_mpi = false;
    impl->nccl_comm = nullptr;
    impl->cuda_stream = nullptr;
    impl->barrier_scratch = nullptr;

    int err = MPI_Comm_split(ctx->comm, color, key, &impl->comm);
    if (err != MPI_SUCCESS) {
        delete impl;
        return DTL_ERROR_MPI;
    }
    (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
    impl->owns_comm = true;

    MPI_Comm_rank(impl->comm, &impl->rank);
    MPI_Comm_size(impl->comm, &impl->size);
    impl->device_id = device_id;
    impl->determinism_mode = ctx->determinism_mode;
    impl->reduction_schedule_policy = ctx->reduction_schedule_policy;
    impl->progress_ordering_policy = ctx->progress_ordering_policy;
    impl->nccl_mode = mode;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = (ctx->domain_flags | dtl_context_s::HAS_NCCL | dtl_context_s::HAS_CUDA);
    impl->error_handler = ctx->error_handler;
    impl->error_handler_user_data = ctx->error_handler_user_data;

    dtl_status init_status = init_nccl_for_context(impl, device_id);
    if (init_status != DTL_SUCCESS) {
        dtl_context_destroy(impl);
        return init_status;
    }

    *out = impl;
    return DTL_SUCCESS;
#endif
}

}  // extern "C"
