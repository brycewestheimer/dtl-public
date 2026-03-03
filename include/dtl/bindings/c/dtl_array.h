// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_array.h
 * @brief DTL C bindings - Distributed array operations
 * @since 0.1.0
 *
 * This header provides C bindings for the distributed_array container,
 * a fixed-size 1D distributed array similar to std::array.
 *
 * Unlike distributed_vector, distributed_array has a fixed size that
 * cannot be changed after creation. There is no resize() operation.
 */

#ifndef DTL_ARRAY_H
#define DTL_ARRAY_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"

DTL_C_BEGIN

/* ==========================================================================
 * Array Lifecycle
 * ========================================================================== */

/**
 * @brief Create a distributed array with fixed size
 *
 * Creates a new distributed array with the specified size.
 * The data is distributed across ranks using block partitioning.
 * Unlike vectors, arrays cannot be resized after creation.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param size Total number of elements (fixed, cannot change)
 * @param[out] arr Pointer to receive the array handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre size >= 0
 * @pre arr must not be NULL
 * @post On success, *arr contains a valid array handle
 *
 * @code
 * dtl_array_t arr;
 * dtl_status status = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 1000, &arr);
 * if (status != DTL_SUCCESS) {
 *     // Handle error
 * }
 * // Use array... (note: cannot resize)
 * dtl_array_destroy(arr);
 * @endcode
 */
DTL_API dtl_status dtl_array_create(dtl_context_t ctx, dtl_dtype dtype,
                                     dtl_size_t size,
                                     dtl_array_t* arr);

/**
 * @brief Create a distributed array with initial value
 *
 * Like dtl_array_create, but initializes all elements to a value.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param size Total number of elements (fixed)
 * @param value Pointer to the initial value (must match dtype)
 * @param[out] arr Pointer to receive the array handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_array_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                          dtl_size_t size,
                                          const void* value,
                                          dtl_array_t* arr);

/**
 * @brief Destroy a distributed array
 *
 * Releases all resources associated with the array.
 *
 * @param arr The array to destroy (may be NULL)
 *
 * @post arr is invalid and must not be used
 * @note It is safe to call with NULL.
 */
DTL_API void dtl_array_destroy(dtl_array_t arr);

/* ==========================================================================
 * Array Size Queries
 * ========================================================================== */

/**
 * @brief Get the global size of the array
 *
 * Unlike vectors, this size is fixed and cannot change.
 *
 * @param arr The array
 * @return Total number of elements across all ranks, or 0 on error
 */
DTL_API dtl_size_t dtl_array_global_size(dtl_array_t arr);

/**
 * @brief Get the local size on this rank
 *
 * @param arr The array
 * @return Number of elements stored locally, or 0 on error
 */
DTL_API dtl_size_t dtl_array_local_size(dtl_array_t arr);

/**
 * @brief Get the local offset (start index in global space)
 *
 * @param arr The array
 * @return Global index of the first local element, or 0 on error
 */
DTL_API dtl_index_t dtl_array_local_offset(dtl_array_t arr);

/**
 * @brief Check if the array is empty
 *
 * @param arr The array
 * @return 1 if global size is 0, 0 otherwise
 */
DTL_API int dtl_array_empty(dtl_array_t arr);

/**
 * @brief Get the data type of elements
 *
 * @param arr The array
 * @return The element dtype, or -1 on error
 */
DTL_API dtl_dtype dtl_array_dtype(dtl_array_t arr);

/* ==========================================================================
 * Local Data Access
 * ========================================================================== */

/**
 * @brief Get pointer to local data (read-only)
 *
 * Returns a pointer to the local data buffer. The pointer is valid
 * until the array is destroyed.
 *
 * @param arr The array
 * @return Pointer to local data, or NULL on error
 *
 * @note The data type matches the array's dtype.
 * @note This operation does not communicate.
 */
DTL_API const void* dtl_array_local_data(dtl_array_t arr);

/**
 * @brief Get pointer to local data (mutable)
 *
 * Returns a mutable pointer to the local data buffer.
 *
 * @param arr The array
 * @return Pointer to local data, or NULL on error
 *
 * @note Modifications are local only; no communication occurs.
 */
DTL_API void* dtl_array_local_data_mut(dtl_array_t arr);

/**
 * @brief Get element at local index (type-safe)
 *
 * Copies the element at the local index to the output buffer.
 *
 * @param arr The array
 * @param local_idx Index within the local partition
 * @param[out] value Pointer to receive the element value
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre local_idx < local_size
 * @pre value must point to memory of size >= dtype_size(dtype)
 */
DTL_API dtl_status dtl_array_get_local(dtl_array_t arr,
                                        dtl_size_t local_idx,
                                        void* value);

/**
 * @brief Set element at local index (type-safe)
 *
 * Sets the element at the local index from the input buffer.
 *
 * @param arr The array
 * @param local_idx Index within the local partition
 * @param value Pointer to the element value
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre local_idx < local_size
 * @pre value must point to memory of size >= dtype_size(dtype)
 */
DTL_API dtl_status dtl_array_set_local(dtl_array_t arr,
                                        dtl_size_t local_idx,
                                        const void* value);

/* ==========================================================================
 * Distribution Queries
 * ========================================================================== */

/**
 * @brief Get the number of ranks
 *
 * @param arr The array
 * @return Number of ranks the array is distributed across
 */
DTL_API dtl_rank_t dtl_array_num_ranks(dtl_array_t arr);

/**
 * @brief Get the owning rank of this array handle
 *
 * @param arr The array
 * @return The rank that owns this handle
 */
DTL_API dtl_rank_t dtl_array_rank(dtl_array_t arr);

/**
 * @brief Check if a global index is local
 *
 * @param arr The array
 * @param global_idx Global index to check
 * @return 1 if the index is local, 0 otherwise
 */
DTL_API int dtl_array_is_local(dtl_array_t arr, dtl_index_t global_idx);

/**
 * @brief Get the owner rank for a global index
 *
 * @param arr The array
 * @param global_idx Global index to query
 * @return Rank that owns the index, or DTL_NO_RANK on error
 */
DTL_API dtl_rank_t dtl_array_owner(dtl_array_t arr, dtl_index_t global_idx);

/**
 * @brief Convert global index to local index
 *
 * @param arr The array
 * @param global_idx Global index to convert
 * @return Local index, or -1 if not local
 *
 * @pre is_local(global_idx) must be true
 */
DTL_API dtl_index_t dtl_array_to_local(dtl_array_t arr, dtl_index_t global_idx);

/**
 * @brief Convert local index to global index
 *
 * @param arr The array
 * @param local_idx Local index to convert
 * @return Global index
 */
DTL_API dtl_index_t dtl_array_to_global(dtl_array_t arr, dtl_index_t local_idx);

/* ==========================================================================
 * Local Operations
 * ========================================================================== */

/**
 * @brief Fill all local elements with a value
 *
 * Sets all local elements to the specified value.
 * This is a local operation - no communication.
 *
 * @param arr The array
 * @param value Pointer to the fill value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_array_fill_local(dtl_array_t arr, const void* value);

/**
 * @brief Barrier synchronization on array
 *
 * Ensures all ranks have reached this point.
 *
 * @param arr The array
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_array_barrier(dtl_array_t arr);

/* ==========================================================================
 * Array Validation
 * ========================================================================== */

/**
 * @brief Check if an array handle is valid
 *
 * @param arr The array to check (may be NULL)
 * @return 1 if valid, 0 if NULL or invalid
 */
DTL_API int dtl_array_is_valid(dtl_array_t arr);

/* ==========================================================================
 * NOTE: No resize operation
 * ==========================================================================
 * Unlike dtl_vector, there is intentionally NO dtl_array_resize() function.
 * distributed_array is analogous to std::array - it has a fixed size that
 * is determined at creation time and cannot be changed.
 * ========================================================================== */

DTL_C_END

/* Mark header as available for master include */
#define DTL_ARRAY_H_AVAILABLE

#endif /* DTL_ARRAY_H */
