// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_futures.h
 * @brief DTL C bindings - Futures and asynchronous completion
 * @since 0.1.0
 *
 * This header provides C bindings for futures-based asynchronous
 * programming, including future creation, completion waiting,
 * value transfer, and combinators (when_all, when_any).
 */

/* WARNING: Futures API is experimental. The progress engine has known
   stability issues (see KNOWN_ISSUES.md). */

#ifndef DTL_FUTURES_H
#define DTL_FUTURES_H

#include "dtl_config.h"
#include "dtl_types.h"
#include "dtl_status.h"

DTL_C_BEGIN

/* ==========================================================================
 * Opaque Handle Types
 * ========================================================================== */

/** @brief Forward declaration for future implementation */
struct dtl_future_s;

/**
 * @brief Opaque handle to a DTL future
 *
 * A future represents an asynchronous value that may not yet be
 * available. Futures can be waited on (blocking), tested (non-blocking),
 * and composed using when_all/when_any combinators.
 */
typedef struct dtl_future_s* dtl_future_t;

/* ==========================================================================
 * Future Lifecycle
 * ========================================================================== */

/**
 * @brief Create an incomplete future
 *
 * Creates a new future that is initially in the incomplete state.
 * The future must later be completed via dtl_future_set() and
 * eventually destroyed via dtl_future_destroy().
 *
 * @param[out] fut Pointer to receive the created future handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Future created successfully
 * @retval DTL_ERROR_NULL_POINTER   fut is NULL
 * @retval DTL_ERROR_ALLOCATION_FAILED  Memory allocation failed
 *
 * @pre fut must not be NULL
 * @post On success, *fut contains a valid, incomplete future handle
 *
 * @code
 * dtl_future_t fut;
 * dtl_status status = dtl_future_create(&fut);
 * if (status == DTL_SUCCESS) {
 *     // Use the future...
 *     dtl_future_destroy(fut);
 * }
 * @endcode
 */
DTL_API dtl_status dtl_future_create(dtl_future_t* fut);

/**
 * @brief Destroy a future and release its resources
 *
 * Releases all resources associated with the future. It is safe
 * to call with NULL. After this call, the handle is invalid.
 *
 * @param fut The future to destroy (may be NULL)
 *
 * @note It is safe to call with NULL.
 * @warning Destroying a future that is being waited on by another
 *          thread (e.g., via when_all/when_any) results in undefined behavior.
 */
DTL_API void dtl_future_destroy(dtl_future_t fut);

/* ==========================================================================
 * Future Synchronization
 * ========================================================================== */

/**
 * @brief Block until the future is complete
 *
 * Blocks the calling thread until the future has been completed
 * via dtl_future_set(). If the future is already complete, returns
 * immediately.
 *
 * @param fut The future to wait on
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Future is complete
 * @retval DTL_ERROR_INVALID_ARGUMENT  fut is NULL or invalid
 *
 * @pre fut must be a valid future handle
 * @post The future is in the completed state
 */
DTL_API dtl_status dtl_future_wait(dtl_future_t fut);

/**
 * @brief Non-blocking test for future completion
 *
 * Checks whether the future has been completed without blocking.
 * Sets *completed to 1 if complete, 0 otherwise.
 *
 * @param fut The future to test
 * @param[out] completed Pointer to receive completion status (1 = complete, 0 = pending)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Test performed successfully
 * @retval DTL_ERROR_INVALID_ARGUMENT  fut is NULL or invalid
 * @retval DTL_ERROR_NULL_POINTER   completed is NULL
 *
 * @pre fut must be a valid future handle
 * @pre completed must not be NULL
 */
DTL_API dtl_status dtl_future_test(dtl_future_t fut, int* completed);

/* ==========================================================================
 * Future Value Access
 * ========================================================================== */

/**
 * @brief Get the result value from a completed future
 *
 * Copies the stored result value into the provided buffer. The future
 * must be in the completed state (i.e., dtl_future_set() has been called).
 *
 * @param fut The future to get the value from
 * @param[out] buffer Pointer to buffer to receive the value
 * @param size Size of the buffer in bytes
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Value copied successfully
 * @retval DTL_ERROR_INVALID_ARGUMENT  fut is NULL or invalid
 * @retval DTL_ERROR_NULL_POINTER   buffer is NULL
 * @retval DTL_ERROR_INVALID_STATE  Future is not yet complete
 * @retval DTL_ERROR_BUFFER_TOO_SMALL  buffer size is smaller than stored value
 *
 * @pre fut must be a valid, completed future handle
 * @pre buffer must not be NULL
 * @pre size must be >= the size of the stored value
 */
DTL_API dtl_status dtl_future_get(dtl_future_t fut, void* buffer,
                                   dtl_size_t size);

/**
 * @brief Set the result value and mark the future as complete
 *
 * Stores a copy of the value and transitions the future to the
 * completed state, waking any threads blocked in dtl_future_wait().
 *
 * @param fut The future to complete
 * @param[in] value Pointer to the value to store (may be NULL for zero-size)
 * @param size Size of the value in bytes (may be 0 for signal-only futures)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Value set and future completed
 * @retval DTL_ERROR_INVALID_ARGUMENT  fut is NULL or invalid
 * @retval DTL_ERROR_INVALID_STATE  Future is already complete
 *
 * @pre fut must be a valid, incomplete future handle
 * @post The future is in the completed state
 * @post Any threads blocked in dtl_future_wait() are woken
 */
DTL_API dtl_status dtl_future_set(dtl_future_t fut, const void* value,
                                   dtl_size_t size);

/* ==========================================================================
 * Future Combinators
 * ========================================================================== */

/**
 * @brief Create a future that completes when all input futures complete
 *
 * Creates a new future that transitions to the completed state once
 * all of the input futures have been completed. The result future
 * carries no value (zero-size completion signal).
 *
 * Internally, this registers a progress-engine callback that polls inputs.
 * Completion is driven by calls to dtl_future_test(), dtl_future_wait(), or
 * other APIs that make progress.
 *
 * @param[in] futures Array of futures to wait on
 * @param count Number of futures in the array
 * @param[out] result Pointer to receive the combined future handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Combined future created
 * @retval DTL_ERROR_NULL_POINTER   futures or result is NULL
 * @retval DTL_ERROR_INVALID_ARGUMENT  count is 0
 * @retval DTL_ERROR_ALLOCATION_FAILED  Memory allocation failed
 *
 * @pre futures must not be NULL
 * @pre count must be > 0
 * @pre result must not be NULL
 * @post On success, *result contains a valid future that completes
 *       when all input futures are complete
 *
 * @warning The input futures must remain valid until the result future
 *          completes. Destroying an input future while when_all is
 *          pending results in undefined behavior.
 *
 * @code
 * dtl_future_t futures[3];
 * // ... create and submit work for futures[0..2] ...
 * dtl_future_t all_done;
 * dtl_when_all(futures, 3, &all_done);
 * dtl_future_wait(all_done);
 * dtl_future_destroy(all_done);
 * @endcode
 */
DTL_API dtl_status dtl_when_all(const dtl_future_t* futures,
                                 dtl_size_t count,
                                 dtl_future_t* result);

/**
 * @brief Create a future that completes when any input future completes
 *
 * Creates a new future that transitions to the completed state once
 * any one of the input futures has been completed. The index of the
 * first completed future is stored in the result future payload
 * (`sizeof(dtl_size_t)` bytes retrievable via dtl_future_get()).
 *
 * Internally, this registers a progress-engine callback that polls input futures.
 * Completion is driven by calls to dtl_future_test(), dtl_future_wait(), or
 * other APIs that make progress.
 *
 * @param[in] futures Array of futures to monitor
 * @param count Number of futures in the array
 * @param[out] result Pointer to receive the combined future handle
 * @param[out] completed_index Pointer to receive an immediate sentinel.
 *             On success this is set to count and is not updated asynchronously.
 *             Read the completed index via dtl_future_get(result, ...).
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @retval DTL_SUCCESS              Combined future created
 * @retval DTL_ERROR_NULL_POINTER   futures, result, or completed_index is NULL
 * @retval DTL_ERROR_INVALID_ARGUMENT  count is 0
 * @retval DTL_ERROR_ALLOCATION_FAILED  Memory allocation failed
 *
 * @pre futures must not be NULL
 * @pre count must be > 0
 * @pre result must not be NULL
 * @pre completed_index must not be NULL
 * @post On success, *result contains a valid future; *completed_index is set
 *       to count. After completion, dtl_future_get(result, ...) returns the
 *       winning index payload.
 *
 * @warning The input futures must remain valid until the result future
 *          completes. Destroying an input future while when_any is
 *          pending results in undefined behavior.
 *
 * @code
 * dtl_future_t futures[3];
 * // ... create and submit work for futures[0..2] ...
 * dtl_future_t any_done;
 * dtl_size_t which;
 * dtl_when_any(futures, 3, &any_done, &which);
 * dtl_future_wait(any_done);
 * dtl_future_get(any_done, &which, sizeof(which));
 * // 'which' now contains the index of the first completed future
 * dtl_future_destroy(any_done);
 * @endcode
 */
DTL_API dtl_status dtl_when_any(const dtl_future_t* futures,
                                 dtl_size_t count,
                                 dtl_future_t* result,
                                 dtl_size_t* completed_index);

DTL_C_END

/* Mark header as available for master include */
#define DTL_FUTURES_H_AVAILABLE

#endif /* DTL_FUTURES_H */
