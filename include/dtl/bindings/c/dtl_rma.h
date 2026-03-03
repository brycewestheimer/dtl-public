// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_rma.h
 * @brief DTL C bindings - RMA (Remote Memory Access) operations
 * @since 0.1.0
 *
 * This header provides C bindings for RMA one-sided communication,
 * including memory windows, put/get operations, atomic operations,
 * and synchronization primitives.
 */

#ifndef DTL_RMA_H
#define DTL_RMA_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"

DTL_C_BEGIN

/* ==========================================================================
 * RMA Lock Mode
 * ========================================================================== */

/**
 * @brief RMA lock modes for passive-target synchronization
 */
typedef enum dtl_lock_mode {
    DTL_LOCK_EXCLUSIVE = 0,  /**< Exclusive lock (no concurrent access) */
    DTL_LOCK_SHARED    = 1   /**< Shared lock (concurrent reads allowed) */
} dtl_lock_mode;

/* ==========================================================================
 * Window Lifecycle
 * ========================================================================== */

/**
 * @brief Create an RMA memory window from existing memory
 *
 * Creates a window that exposes existing memory for remote access.
 * The caller retains ownership of the memory and must ensure it
 * outlives the window.
 *
 * @param ctx The context
 * @param base Pointer to the memory to expose (may be NULL for remote-only)
 * @param size Size of the memory region in bytes
 * @param[out] win Pointer to receive the window handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre size >= 0
 * @pre win must not be NULL
 * @post On success, *win contains a valid window handle
 *
 * @warning This is a collective operation - all ranks must call.
 * @warning The memory must remain valid until the window is destroyed.
 */
DTL_API dtl_status dtl_window_create(dtl_context_t ctx, void* base,
                                      dtl_size_t size, dtl_window_t* win);

/**
 * @brief Allocate an RMA memory window
 *
 * Creates a window with library-allocated memory. The memory is
 * automatically freed when the window is destroyed.
 *
 * @param ctx The context
 * @param size Size of memory to allocate in bytes
 * @param[out] win Pointer to receive the window handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning This is a collective operation - all ranks must call.
 */
DTL_API dtl_status dtl_window_allocate(dtl_context_t ctx,
                                        dtl_size_t size, dtl_window_t* win);

/**
 * @brief Destroy an RMA memory window
 *
 * Releases all resources associated with the window.
 * If the window was created with dtl_window_allocate, the memory is freed.
 *
 * @param win The window to destroy (may be NULL)
 *
 * @warning This is a collective operation - all ranks must call.
 * @note It is safe to call with NULL.
 */
DTL_API void dtl_window_destroy(dtl_window_t win);

/* ==========================================================================
 * Window Queries
 * ========================================================================== */

/**
 * @brief Get the base pointer of a window
 *
 * @param win The window
 * @return Base pointer, or NULL if invalid or remote-only
 */
DTL_API void* dtl_window_base(dtl_window_t win);

/**
 * @brief Get the size of a window
 *
 * @param win The window
 * @return Size in bytes, or 0 on error
 */
DTL_API dtl_size_t dtl_window_size(dtl_window_t win);

/**
 * @brief Check if a window handle is valid
 *
 * @param win The window to check (may be NULL)
 * @return 1 if valid, 0 if NULL or invalid
 */
DTL_API int dtl_window_is_valid(dtl_window_t win);

/* ==========================================================================
 * Active-Target Synchronization (Fence)
 * ========================================================================== */

/**
 * @brief Synchronize RMA operations with a fence
 *
 * A fence is a collective operation that completes all pending RMA
 * operations on the window. It opens and closes an access epoch.
 *
 * @param win The window
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning This is a collective operation - all ranks must call.
 *
 * @code
 * // Active-target synchronization pattern
 * dtl_window_fence(win);  // Open epoch
 * dtl_rma_put(win, 1, 0, data, 100);
 * dtl_rma_get(win, 2, 0, buffer, 100);
 * dtl_window_fence(win);  // Complete operations
 * @endcode
 */
DTL_API dtl_status dtl_window_fence(dtl_window_t win);

/* ==========================================================================
 * Passive-Target Synchronization (Lock/Unlock)
 * ========================================================================== */

/**
 * @brief Lock a target rank's window for exclusive or shared access
 *
 * Starts a passive-target access epoch to the specified target rank.
 * The lock must be released with dtl_window_unlock.
 *
 * @param win The window
 * @param target Target rank to lock
 * @param mode Lock mode (DTL_LOCK_EXCLUSIVE or DTL_LOCK_SHARED)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note This is NOT a collective operation.
 * @note Multiple shared locks to the same target are allowed.
 */
DTL_API dtl_status dtl_window_lock(dtl_window_t win, dtl_rank_t target,
                                    dtl_lock_mode mode);

/**
 * @brief Unlock a target rank's window
 *
 * Ends the passive-target access epoch to the specified target.
 * All pending RMA operations to the target are completed.
 *
 * @param win The window
 * @param target Target rank to unlock
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_window_unlock(dtl_window_t win, dtl_rank_t target);

/**
 * @brief Lock all target ranks' windows
 *
 * Starts a passive-target access epoch to all ranks simultaneously.
 * This is more efficient than locking each rank individually.
 *
 * @param win The window
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note All locks are in shared mode.
 */
DTL_API dtl_status dtl_window_lock_all(dtl_window_t win);

/**
 * @brief Unlock all target ranks' windows
 *
 * Ends the passive-target access epoch to all ranks.
 *
 * @param win The window
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_window_unlock_all(dtl_window_t win);

/* ==========================================================================
 * Flush Operations (Remote Completion)
 * ========================================================================== */

/**
 * @brief Flush pending operations to a target rank
 *
 * Ensures all pending RMA operations to the target rank have completed
 * at the target (remote completion). The operations are visible at the target.
 *
 * @param win The window
 * @param target Target rank to flush
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note Only valid within a passive-target epoch.
 */
DTL_API dtl_status dtl_window_flush(dtl_window_t win, dtl_rank_t target);

/**
 * @brief Flush pending operations to all target ranks
 *
 * Ensures all pending RMA operations have completed at all targets.
 *
 * @param win The window
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_window_flush_all(dtl_window_t win);

/**
 * @brief Flush local completion for a target rank
 *
 * Ensures the local buffers used in RMA operations to the target can
 * be reused. Does not guarantee remote visibility.
 *
 * @param win The window
 * @param target Target rank
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_window_flush_local(dtl_window_t win, dtl_rank_t target);

/**
 * @brief Flush local completion for all target ranks
 *
 * Ensures all local buffers can be reused.
 *
 * @param win The window
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_window_flush_local_all(dtl_window_t win);

/* ==========================================================================
 * Data Transfer Operations
 * ========================================================================== */

/**
 * @brief Put data into a remote window
 *
 * Transfers data from the local origin buffer to the target rank's window.
 * The operation is non-blocking; use synchronization to ensure completion.
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param origin Pointer to local data to send
 * @param size Size of data in bytes
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre Must be within an access epoch (fence or lock).
 * @pre target_offset + size <= target's window size
 */
DTL_API dtl_status dtl_rma_put(dtl_window_t win, dtl_rank_t target,
                                dtl_size_t target_offset, const void* origin,
                                dtl_size_t size);

/**
 * @brief Get data from a remote window
 *
 * Transfers data from the target rank's window to the local buffer.
 * The operation is non-blocking; use synchronization to ensure completion.
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param buffer Pointer to local buffer to receive data
 * @param size Size of data in bytes
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre Must be within an access epoch (fence or lock).
 * @pre target_offset + size <= target's window size
 */
DTL_API dtl_status dtl_rma_get(dtl_window_t win, dtl_rank_t target,
                                dtl_size_t target_offset, void* buffer,
                                dtl_size_t size);

/* ==========================================================================
 * Asynchronous Operations (with Request)
 * ========================================================================== */

/**
 * @brief Asynchronous put with request handle
 *
 * Like dtl_rma_put, but returns a request handle for completion testing.
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param origin Pointer to local data to send
 * @param size Size of data in bytes
 * @param[out] req Pointer to receive request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note Use dtl_request_wait or dtl_request_test to check completion.
 */
DTL_API dtl_status dtl_rma_put_async(dtl_window_t win, dtl_rank_t target,
                                      dtl_size_t target_offset,
                                      const void* origin, dtl_size_t size,
                                      dtl_request_t* req);

/**
 * @brief Asynchronous get with request handle
 *
 * Like dtl_rma_get, but returns a request handle for completion testing.
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param buffer Pointer to local buffer to receive data
 * @param size Size of data in bytes
 * @param[out] req Pointer to receive request handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_rma_get_async(dtl_window_t win, dtl_rank_t target,
                                      dtl_size_t target_offset,
                                      void* buffer, dtl_size_t size,
                                      dtl_request_t* req);

/* ==========================================================================
 * Atomic Operations
 * ========================================================================== */

/**
 * @brief Atomic accumulate operation
 *
 * Atomically updates the target location using the specified reduction
 * operation: target[offset] = op(target[offset], origin_value)
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param origin Pointer to origin value
 * @param size Size of data in bytes
 * @param dtype Data type (required for reduction operation)
 * @param op Reduction operation (DTL_OP_SUM, DTL_OP_MAX, etc.)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_rma_accumulate(dtl_window_t win, dtl_rank_t target,
                                       dtl_size_t target_offset,
                                       const void* origin, dtl_size_t size,
                                       dtl_dtype dtype, dtl_reduce_op op);

/**
 * @brief Fetch and perform atomic operation
 *
 * Atomically fetches the old value and updates with a reduction:
 *   *result = target[offset]
 *   target[offset] = op(target[offset], origin_value)
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param origin Pointer to origin value
 * @param result Pointer to receive the old value
 * @param dtype Data type
 * @param op Reduction operation
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note For single-element operations (size determined by dtype).
 */
DTL_API dtl_status dtl_rma_fetch_and_op(dtl_window_t win, dtl_rank_t target,
                                         dtl_size_t target_offset,
                                         const void* origin, void* result,
                                         dtl_dtype dtype, dtl_reduce_op op);

/**
 * @brief Atomic compare and swap
 *
 * Atomically compares the target value with compare_value and if equal,
 * replaces it with swap_value. Returns the original value.
 *
 *   *result = target[offset]
 *   if (target[offset] == *compare) target[offset] = *swap
 *
 * @param win The window
 * @param target Target rank
 * @param target_offset Byte offset in target's window
 * @param compare Pointer to comparison value
 * @param swap Pointer to swap value
 * @param result Pointer to receive the original value
 * @param dtype Data type
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note For single-element operations.
 */
DTL_API dtl_status dtl_rma_compare_and_swap(dtl_window_t win, dtl_rank_t target,
                                             dtl_size_t target_offset,
                                             const void* compare,
                                             const void* swap, void* result,
                                             dtl_dtype dtype);

/* ==========================================================================
 * Request Management
 * ========================================================================== */

/*
 * Request management functions are declared in dtl_communicator.h:
 * - dtl_wait(request) - Wait for async operation to complete
 * - dtl_test(request, completed) - Test if operation is complete
 * - dtl_request_free(request) - Free a request handle
 *
 * These functions work with both point-to-point and RMA async operations.
 */

DTL_C_END

/* Mark header as available for master include */
#define DTL_RMA_H_AVAILABLE

#endif /* DTL_RMA_H */
