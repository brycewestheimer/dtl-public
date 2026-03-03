// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_context.h
 * @brief DTL C bindings - Context operations
 * @since 0.1.0
 *
 * This header defines the DTL context, which encapsulates the
 * MPI communicator, device selection, and execution environment.
 *
 * @thread_safety A single dtl_context_t is **not** safe for concurrent use
 *   from multiple threads. Each thread should create its own context (via
 *   dtl_context_dup() or dtl_context_create()) or use external synchronization.
 */

#ifndef DTL_CONTEXT_H
#define DTL_CONTEXT_H

#include "dtl_types.h"
#include "dtl_status.h"

DTL_C_BEGIN

/* ==========================================================================
 * Context Lifecycle
 * ========================================================================== */

/**
 * @brief Create a new DTL context with options
 *
 * Creates a context that manages the communication backend, device
 * selection, and other execution environment state. The context
 * will initialize MPI if needed (controlled by opts->init_mpi).
 *
 * @param[out] ctx Pointer to receive the created context handle
 * @param[in] opts Context options (may be NULL for defaults)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must not be NULL
 * @post On success, *ctx contains a valid context handle
 * @post On failure, *ctx is unchanged
 *
 * @note The caller must call dtl_context_destroy() when done.
 *
 * @code
 * dtl_context_t ctx;
 * dtl_context_options opts;
 * dtl_context_options_init(&opts);
 * opts.device_id = 0;  // Use GPU 0
 *
 * dtl_status status = dtl_context_create(&ctx, &opts);
 * if (status != DTL_SUCCESS) {
 *     fprintf(stderr, "Failed: %s\n", dtl_status_message(status));
 *     return 1;
 * }
 * // Use context...
 * dtl_context_destroy(ctx);
 * @endcode
 */
DTL_API dtl_status dtl_context_create(dtl_context_t* ctx,
                                       const dtl_context_options* opts);

/**
 * @brief Create a new DTL context with default options
 *
 * Convenience function equivalent to dtl_context_create(ctx, NULL).
 * Uses CPU-only mode and initializes MPI if needed.
 *
 * @param[out] ctx Pointer to receive the created context handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must not be NULL
 */
DTL_API dtl_status dtl_context_create_default(dtl_context_t* ctx);

/**
 * @brief Destroy a DTL context
 *
 * Releases all resources associated with the context. If the context
 * initialized MPI and opts->finalize_mpi was set, MPI will be finalized.
 *
 * @param ctx The context to destroy (may be NULL, which is a no-op)
 *
 * @post ctx is invalid and must not be used
 *
 * @note It is safe to call with NULL.
 * @note All containers/vectors created with this context must be
 *       destroyed before destroying the context.
 */
DTL_API void dtl_context_destroy(dtl_context_t ctx);

/* ==========================================================================
 * Context Queries
 * ========================================================================== */

/**
 * @brief Get the rank of this process in the communicator
 *
 * @param ctx The context
 * @return This process's rank (0 to size-1), or DTL_NO_RANK on error
 *
 * @pre ctx must be a valid context
 */
DTL_API dtl_rank_t dtl_context_rank(dtl_context_t ctx);

/**
 * @brief Get the total number of ranks in the communicator
 *
 * @param ctx The context
 * @return Number of ranks, or 0 on error
 *
 * @pre ctx must be a valid context
 */
DTL_API dtl_rank_t dtl_context_size(dtl_context_t ctx);

/**
 * @brief Check if this is rank 0 (root)
 *
 * Convenience function for checking if this is the root rank.
 *
 * @param ctx The context
 * @return 1 if rank is 0, 0 otherwise
 *
 * @pre ctx must be a valid context
 */
DTL_API int dtl_context_is_root(dtl_context_t ctx);

/**
 * @brief Get the device ID associated with this context
 *
 * @param ctx The context
 * @return Device ID (>= 0) if GPU context, -1 for CPU-only context
 *
 * @pre ctx must be a valid context
 */
DTL_API int dtl_context_device_id(dtl_context_t ctx);

/**
 * @brief Check if context has GPU support
 *
 * @param ctx The context
 * @return 1 if GPU context (device_id >= 0), 0 for CPU-only
 *
 * @pre ctx must be a valid context
 */
DTL_API int dtl_context_has_device(dtl_context_t ctx);

/**
 * @brief Get configured determinism mode for this context
 *
 * @param ctx The context
 * @return dtl_determinism_mode value, or DTL_DETERMINISM_THROUGHPUT on error
 */
DTL_API int dtl_context_determinism_mode(dtl_context_t ctx);

/**
 * @brief Get configured reduction schedule policy for deterministic mode
 *
 * @param ctx The context
 * @return dtl_reduction_schedule_policy value, or implementation-defined on error
 */
DTL_API int dtl_context_reduction_schedule_policy(dtl_context_t ctx);

/**
 * @brief Get configured progress ordering policy for deterministic mode
 *
 * @param ctx The context
 * @return dtl_progress_ordering_policy value, or implementation-defined on error
 */
DTL_API int dtl_context_progress_ordering_policy(dtl_context_t ctx);

/* ==========================================================================
 * Synchronization
 * ========================================================================== */

/**
 * @brief Perform a barrier synchronization
 *
 * Blocks until all ranks in the communicator have called this function.
 * This is a collective operation - all ranks must call it.
 *
 * @param ctx The context
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre All ranks in the communicator must call this function
 */
DTL_API dtl_status dtl_context_barrier(dtl_context_t ctx);

/**
 * @brief Perform a memory fence
 *
 * Ensures all memory operations are visible to other processes.
 * Does not synchronize with other ranks (unlike barrier).
 *
 * @param ctx The context
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 */
DTL_API dtl_status dtl_context_fence(dtl_context_t ctx);

/* ==========================================================================
 * Context Validation
 * ========================================================================== */

/**
 * @brief Check if a context handle is valid
 *
 * @param ctx The context to check (may be NULL)
 * @return 1 if ctx is a valid context handle, 0 if NULL or invalid
 */
DTL_API int dtl_context_is_valid(dtl_context_t ctx);

/* ==========================================================================
 * Context Duplication (Advanced)
 * ========================================================================== */

/**
 * @brief Duplicate a context
 *
 * Creates a new context with a duplicated MPI communicator.
 * This allows independent communication on the new context.
 *
 * @param[in] src Source context to duplicate
 * @param[out] dst Pointer to receive the new context handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre src must be a valid context
 * @pre dst must not be NULL
 * @post On success, *dst contains a new valid context
 */
DTL_API dtl_status dtl_context_dup(dtl_context_t src, dtl_context_t* dst);

/* ==========================================================================
 * Domain Queries (V1.3.0)
 * ========================================================================== */

/**
 * @brief Check if context has MPI domain
 *
 * @param ctx The context
 * @return 1 if MPI domain is available, 0 otherwise
 *
 * @pre ctx must be a valid context
 * @since 0.1.0
 */
DTL_API int dtl_context_has_mpi(dtl_context_t ctx);

/**
 * @brief Check if context has CUDA domain
 *
 * @param ctx The context
 * @return 1 if CUDA domain is available, 0 otherwise
 *
 * @pre ctx must be a valid context
 * @since 0.1.0
 */
DTL_API int dtl_context_has_cuda(dtl_context_t ctx);

/**
 * @brief Check if context has NCCL domain
 *
 * @param ctx The context
 * @return 1 if NCCL domain is available, 0 otherwise
 *
 * @pre ctx must be a valid context
 * @since 0.1.0
 */
DTL_API int dtl_context_has_nccl(dtl_context_t ctx);

/**
 * @brief Check if context has SHMEM domain
 *
 * @param ctx The context
 * @return 1 if SHMEM domain is available, 0 otherwise
 *
 * @pre ctx must be a valid context
 * @since 0.1.0
 */
DTL_API int dtl_context_has_shmem(dtl_context_t ctx);

/* ==========================================================================
 * Context Splitting (V1.3.0)
 * ========================================================================== */

/**
 * @brief Split context by color
 *
 * Creates a new context with a split MPI communicator. Ranks with the
 * same color will be in the same group.
 *
 * @param[in] ctx Source context to split
 * @param[in] color Color for grouping (ranks with same color in same group)
 * @param[in] key Ordering key within color group
 * @param[out] out Pointer to receive the new context handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context with MPI domain
 * @pre out must not be NULL
 * @post On success, *out contains a new valid context
 * @note This is a collective operation - all ranks must call
 * @since 0.1.0
 */
DTL_API dtl_status dtl_context_split(dtl_context_t ctx, int color, int key,
                                       dtl_context_t* out);

/**
 * @brief Add CUDA domain to context
 *
 * Creates a new context with an additional CUDA domain.
 *
 * @param[in] ctx Source context
 * @param[in] device_id CUDA device ID to use
 * @param[out] out Pointer to receive the new context handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre out must not be NULL
 * @post On success, *out contains a new context with CUDA domain
 * @since 0.1.0
 */
DTL_API dtl_status dtl_context_with_cuda(dtl_context_t ctx, int device_id,
                                           dtl_context_t* out);

/**
 * @brief Add NCCL domain to context
 *
 * Creates a new context with an additional NCCL domain. Requires
 * both MPI and CUDA domains to be present.
 *
 * @param[in] ctx Source context (must have MPI domain)
 * @param[in] device_id CUDA device ID to use for NCCL
 * @param[out] out Pointer to receive the new context handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context with MPI domain
 * @pre out must not be NULL
 * @post On success, *out contains a new context with NCCL domain
 * @note This is a collective operation - all ranks must call
 * @since 0.1.0
 */
DTL_API dtl_status dtl_context_with_nccl(dtl_context_t ctx, int device_id,
                                           dtl_context_t* out);

DTL_C_END

#endif /* DTL_CONTEXT_H */
