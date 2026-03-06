// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_vector.h
 * @brief DTL C bindings - Distributed vector operations
 * @since 0.1.0
 *
 * This header provides C bindings for the distributed_vector container,
 * a 1D distributed array similar to std::vector.
 */

#ifndef DTL_VECTOR_H
#define DTL_VECTOR_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"
#include "dtl_policies.h"

DTL_C_BEGIN

/* ==========================================================================
 * Vector Lifecycle
 * ========================================================================== */

/**
 * @brief Create a distributed vector
 *
 * Creates a new distributed vector with the specified global size.
 * The data is distributed across ranks using block partitioning.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param global_size Total number of elements across all ranks
 * @param[out] vec Pointer to receive the vector handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre global_size >= 0
 * @pre vec must not be NULL
 * @post On success, *vec contains a valid vector handle
 *
 * @code
 * dtl_vector_t vec;
 * dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1000, &vec);
 * if (status != DTL_SUCCESS) {
 *     // Handle error
 * }
 * // Use vector...
 * dtl_vector_destroy(vec);
 * @endcode
 */
DTL_API dtl_status dtl_vector_create(dtl_context_t ctx, dtl_dtype dtype,
                                      dtl_size_t global_size,
                                      dtl_vector_t* vec);

/**
 * @brief Create a distributed vector with initial value
 *
 * Like dtl_vector_create, but initializes all elements to a value.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param global_size Total number of elements
 * @param value Pointer to the initial value (must match dtype)
 * @param[out] vec Pointer to receive the vector handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                           dtl_size_t global_size,
                                           const void* value,
                                           dtl_vector_t* vec);

/**
 * @brief Destroy a distributed vector
 *
 * Releases all resources associated with the vector.
 *
 * @param vec The vector to destroy (may be NULL)
 *
 * @post vec is invalid and must not be used
 * @note It is safe to call with NULL.
 */
DTL_API void dtl_vector_destroy(dtl_vector_t vec);

/* ==========================================================================
 * Vector Size Queries
 * ========================================================================== */

/**
 * @brief Get the global size of the vector
 *
 * @param vec The vector
 * @return Total number of elements across all ranks, or 0 on error
 */
DTL_API dtl_size_t dtl_vector_global_size(dtl_vector_t vec);

/**
 * @brief Get the local size on this rank
 *
 * @param vec The vector
 * @return Number of elements stored locally, or 0 on error
 */
DTL_API dtl_size_t dtl_vector_local_size(dtl_vector_t vec);

/**
 * @brief Get the local offset (start index in global space)
 *
 * @param vec The vector
 * @return Global index of the first local element, or 0 on error
 */
DTL_API dtl_index_t dtl_vector_local_offset(dtl_vector_t vec);

/**
 * @brief Check if the vector is empty
 *
 * @param vec The vector
 * @return 1 if global size is 0, 0 otherwise
 */
DTL_API int dtl_vector_empty(dtl_vector_t vec);

/**
 * @brief Get the data type of elements
 *
 * @param vec The vector
 * @return The element dtype, or -1 on error
 */
DTL_API dtl_dtype dtl_vector_dtype(dtl_vector_t vec);

/* ==========================================================================
 * Local Data Access
 * ========================================================================== */

/**
 * @brief Get pointer to local data (read-only)
 *
 * Returns a pointer to the local data buffer. The pointer is valid
 * until the vector is resized or destroyed.
 *
 * @param vec The vector
 * @return Pointer to local data, or NULL on error
 *
 * @note The data type matches the vector's dtype.
 * @note This operation does not communicate.
 */
DTL_API const void* dtl_vector_local_data(dtl_vector_t vec);

/**
 * @brief Get pointer to local data (mutable)
 *
 * Returns a mutable pointer to the local data buffer.
 *
 * @param vec The vector
 * @return Pointer to local data, or NULL on error
 *
 * @note Modifications are local only; no communication occurs.
 */
DTL_API void* dtl_vector_local_data_mut(dtl_vector_t vec);

/**
 * @brief Get pointer to device-accessible local data (read-only)
 *
 * Returns a device pointer for device/unified placements.
 * Returns NULL for host-only placement or on error.
 *
 * @param vec The vector
 * @return Device pointer, or NULL if unavailable
 */
DTL_API const void* dtl_vector_device_data(dtl_vector_t vec);

/**
 * @brief Get pointer to device-accessible local data (mutable)
 *
 * Returns a mutable device pointer for device/unified placements.
 * Returns NULL for host-only placement or on error.
 *
 * @param vec The vector
 * @return Mutable device pointer, or NULL if unavailable
 */
DTL_API void* dtl_vector_device_data_mut(dtl_vector_t vec);

/**
 * @brief Get element at local index (type-safe)
 *
 * Copies the element at the local index to the output buffer.
 *
 * @param vec The vector
 * @param local_idx Index within the local partition
 * @param[out] value Pointer to receive the element value
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre local_idx < local_size
 * @pre value must point to memory of size >= dtype_size(dtype)
 */
DTL_API dtl_status dtl_vector_get_local(dtl_vector_t vec,
                                         dtl_size_t local_idx,
                                         void* value);

/**
 * @brief Set element at local index (type-safe)
 *
 * Sets the element at the local index from the input buffer.
 *
 * @param vec The vector
 * @param local_idx Index within the local partition
 * @param value Pointer to the element value
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre local_idx < local_size
 * @pre value must point to memory of size >= dtype_size(dtype)
 */
DTL_API dtl_status dtl_vector_set_local(dtl_vector_t vec,
                                         dtl_size_t local_idx,
                                         const void* value);

/* ==========================================================================
 * Distribution Queries
 * ========================================================================== */

/**
 * @brief Get the number of ranks
 *
 * @param vec The vector
 * @return Number of ranks the vector is distributed across
 */
DTL_API dtl_rank_t dtl_vector_num_ranks(dtl_vector_t vec);

/**
 * @brief Get the owning rank of this vector handle
 *
 * @param vec The vector
 * @return The rank that owns this handle
 */
DTL_API dtl_rank_t dtl_vector_rank(dtl_vector_t vec);

/**
 * @brief Check if a global index is local
 *
 * @param vec The vector
 * @param global_idx Global index to check
 * @return 1 if the index is local, 0 otherwise
 */
DTL_API int dtl_vector_is_local(dtl_vector_t vec, dtl_index_t global_idx);

/**
 * @brief Get the owner rank for a global index
 *
 * @param vec The vector
 * @param global_idx Global index to query
 * @return Rank that owns the index, or DTL_NO_RANK on error
 */
DTL_API dtl_rank_t dtl_vector_owner(dtl_vector_t vec, dtl_index_t global_idx);

/**
 * @brief Convert global index to local index
 *
 * @param vec The vector
 * @param global_idx Global index to convert
 * @return Local index, or -1 if not local
 *
 * @pre is_local(global_idx) must be true
 */
DTL_API dtl_index_t dtl_vector_to_local(dtl_vector_t vec, dtl_index_t global_idx);

/**
 * @brief Convert local index to global index
 *
 * @param vec The vector
 * @param local_idx Local index to convert
 * @return Global index
 */
DTL_API dtl_index_t dtl_vector_to_global(dtl_vector_t vec, dtl_index_t local_idx);

/* ==========================================================================
 * Collective Operations
 * ========================================================================== */

/**
 * @brief Resize the vector (collective)
 *
 * Changes the global size of the vector. Data may be redistributed.
 * All ranks must call this with the same new_size.
 *
 * @param vec The vector
 * @param new_size New global size
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning This is a collective operation - all ranks must call.
 * @warning Invalidates any pointers from local_data().
 */
DTL_API dtl_status dtl_vector_resize(dtl_vector_t vec, dtl_size_t new_size);

/**
 * @brief Barrier synchronization on vector
 *
 * Ensures all ranks have reached this point.
 *
 * @param vec The vector
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_barrier(dtl_vector_t vec);

/**
 * @brief Fill all local elements with a value
 *
 * Sets all local elements to the specified value.
 * This is a local operation - no communication.
 *
 * @param vec The vector
 * @param value Pointer to the fill value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_fill_local(dtl_vector_t vec, const void* value);

/**
 * @brief Reduce local vector values using sum
 *
 * @param vec The vector
 * @param[out] result Pointer to sum result (dtype-matched)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_reduce_sum(dtl_vector_t vec, void* result);

/**
 * @brief Reduce local vector values using min
 *
 * @param vec The vector
 * @param[out] result Pointer to min result (dtype-matched)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_reduce_min(dtl_vector_t vec, void* result);

/**
 * @brief Reduce local vector values using max
 *
 * @param vec The vector
 * @param[out] result Pointer to max result (dtype-matched)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_reduce_max(dtl_vector_t vec, void* result);

/**
 * @brief Sort local vector values in ascending order
 *
 * @param vec The vector
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_sort_ascending(dtl_vector_t vec);

/* ==========================================================================
 * Redistribution (V1.1)
 * ========================================================================== */

/**
 * @brief Partition type for redistribution
 *
 * Alias for dtl_partition_policy (defined in dtl_policies.h).
 * When included via dtl.h, dtl_policies.h is included before this header.
 */
typedef dtl_partition_policy dtl_partition_type;

/**
 * @brief Redistribute the vector with a new partition (collective)
 *
 * Current v1 behavior is a compatibility stub:
 * - `size()==1`: no-op success
 * - `size()>1`: returns `DTL_ERROR_NOT_IMPLEMENTED`
 *
 * Full cross-rank redistribution is tracked in parity completion plans.
 *
 * @param vec The vector
 * @param new_partition The new partition type
 * @return DTL_SUCCESS on no-op single-rank success, error code otherwise
 *
 * @warning This is a collective operation - all ranks must call.
 * @warning Invalidates any pointers from local_data().
 * @warning May be expensive due to data movement.
 *
 * @code
 * // Change from block to cyclic partition
 * dtl_status status = dtl_vector_redistribute(vec, DTL_PARTITION_CYCLIC);
 * @endcode
 */
DTL_API dtl_status dtl_vector_redistribute(dtl_vector_t vec,
                                            dtl_partition_type new_partition);

/* ==========================================================================
 * Sync State (V1.1)
 * ========================================================================== */

/**
 * @brief Check if the vector has uncommitted modifications
 *
 * Current v1 behavior is a stub and reports `0` (clean) for valid handles.
 *
 * @param vec The vector
 * @return 1 if dirty, 0 if clean
 */
DTL_API int dtl_vector_is_dirty(dtl_vector_t vec);

/**
 * @brief Check if the vector is clean (fully synchronized)
 *
 * Current v1 behavior is a stub and reports `1` (clean) for valid handles.
 *
 * @param vec The vector
 * @return 1 if clean, 0 if dirty
 */
DTL_API int dtl_vector_is_clean(dtl_vector_t vec);

/**
 * @brief Synchronize vector state
 *
 * Current v1 behavior is a compatibility stub and returns success without
 * changing dirty-state metadata.
 *
 * @param vec The vector
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_vector_sync(dtl_vector_t vec);

/* ==========================================================================
 * Vector Validation
 * ========================================================================== */

/**
 * @brief Check if a vector handle is valid
 *
 * @param vec The vector to check (may be NULL)
 * @return 1 if valid, 0 if NULL or invalid
 */
DTL_API int dtl_vector_is_valid(dtl_vector_t vec);

DTL_C_END

/* Mark header as available for master include */
#define DTL_VECTOR_H_AVAILABLE

#endif /* DTL_VECTOR_H */
