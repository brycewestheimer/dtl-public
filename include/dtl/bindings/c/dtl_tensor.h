// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_tensor.h
 * @brief DTL C bindings - Distributed tensor operations
 * @since 0.1.0
 *
 * This header provides C bindings for the distributed_tensor container,
 * an N-dimensional distributed array.
 */

#ifndef DTL_TENSOR_H
#define DTL_TENSOR_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"

DTL_C_BEGIN

/* ==========================================================================
 * Tensor Lifecycle
 * ========================================================================== */

/**
 * @brief Create a distributed tensor
 *
 * Creates a new distributed tensor with the specified shape.
 * The tensor is distributed along the first dimension by default.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param shape Shape descriptor (dimensions)
 * @param[out] tensor Pointer to receive the tensor handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre shape.ndim >= 1 and <= DTL_MAX_TENSOR_RANK
 * @pre tensor must not be NULL
 * @post On success, *tensor contains a valid tensor handle
 *
 * @code
 * dtl_tensor_t tensor;
 * dtl_shape shape = dtl_shape_3d(100, 64, 64);  // 100x64x64 tensor
 * dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor);
 * @endcode
 */
DTL_API dtl_status dtl_tensor_create(dtl_context_t ctx, dtl_dtype dtype,
                                      dtl_shape shape,
                                      dtl_tensor_t* tensor);

/**
 * @brief Create a tensor with initial value
 *
 * Like dtl_tensor_create, but initializes all elements to a value.
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param shape Shape descriptor
 * @param value Pointer to the initial value (must match dtype)
 * @param[out] tensor Pointer to receive the tensor handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                           dtl_shape shape, const void* value,
                                           dtl_tensor_t* tensor);

/**
 * @brief Destroy a distributed tensor
 *
 * Releases all resources associated with the tensor.
 *
 * @param tensor The tensor to destroy (may be NULL)
 *
 * @post tensor is invalid and must not be used
 */
DTL_API void dtl_tensor_destroy(dtl_tensor_t tensor);

/* ==========================================================================
 * Tensor Shape Queries
 * ========================================================================== */

/**
 * @brief Get the tensor shape
 *
 * @param tensor The tensor
 * @return Shape descriptor (ndim=0 on error)
 */
DTL_API dtl_shape dtl_tensor_shape(dtl_tensor_t tensor);

/**
 * @brief Get the number of dimensions
 *
 * @param tensor The tensor
 * @return Number of dimensions (rank), or 0 on error
 */
DTL_API int dtl_tensor_ndim(dtl_tensor_t tensor);

/**
 * @brief Get the size along a dimension
 *
 * @param tensor The tensor
 * @param dim Dimension index (0 to ndim-1)
 * @return Size along the dimension, or 0 on error
 */
DTL_API dtl_size_t dtl_tensor_dim(dtl_tensor_t tensor, int dim);

/**
 * @brief Get the total global element count
 *
 * @param tensor The tensor
 * @return Total number of elements (product of dimensions), or 0 on error
 */
DTL_API dtl_size_t dtl_tensor_global_size(dtl_tensor_t tensor);

/**
 * @brief Get the local element count
 *
 * @param tensor The tensor
 * @return Number of elements stored locally, or 0 on error
 */
DTL_API dtl_size_t dtl_tensor_local_size(dtl_tensor_t tensor);

/**
 * @brief Get the local shape
 *
 * The local shape differs from global shape only in the distributed
 * dimension (typically dimension 0).
 *
 * @param tensor The tensor
 * @return Local shape descriptor
 */
DTL_API dtl_shape dtl_tensor_local_shape(dtl_tensor_t tensor);

/**
 * @brief Get the data type of elements
 *
 * @param tensor The tensor
 * @return The element dtype, or -1 on error
 */
DTL_API dtl_dtype dtl_tensor_dtype(dtl_tensor_t tensor);

/* ==========================================================================
 * Local Data Access
 * ========================================================================== */

/**
 * @brief Get pointer to local data (read-only)
 *
 * Returns a pointer to the contiguous local data buffer.
 * Data is stored in row-major (C) order.
 *
 * @param tensor The tensor
 * @return Pointer to local data, or NULL on error
 */
DTL_API const void* dtl_tensor_local_data(dtl_tensor_t tensor);

/**
 * @brief Get pointer to local data (mutable)
 *
 * @param tensor The tensor
 * @return Pointer to local data, or NULL on error
 */
DTL_API void* dtl_tensor_local_data_mut(dtl_tensor_t tensor);

/**
 * @brief Get stride for a dimension
 *
 * The stride is the number of elements between consecutive
 * entries along the specified dimension.
 *
 * @param tensor The tensor
 * @param dim Dimension index
 * @return Stride in elements, or 0 on error
 */
DTL_API dtl_size_t dtl_tensor_stride(dtl_tensor_t tensor, int dim);

/* ==========================================================================
 * N-Dimensional Access
 * ========================================================================== */

/**
 * @brief Get element at N-D local indices
 *
 * @param tensor The tensor
 * @param indices Array of local indices (length = ndim)
 * @param[out] value Pointer to receive the element value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_get_local_nd(dtl_tensor_t tensor,
                                            const dtl_index_t* indices,
                                            void* value);

/**
 * @brief Set element at N-D local indices
 *
 * @param tensor The tensor
 * @param indices Array of local indices (length = ndim)
 * @param value Pointer to the element value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_set_local_nd(dtl_tensor_t tensor,
                                            const dtl_index_t* indices,
                                            const void* value);

/**
 * @brief Get element at linear local index
 *
 * @param tensor The tensor
 * @param linear_idx Linear index into local storage
 * @param[out] value Pointer to receive the element value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_get_local(dtl_tensor_t tensor,
                                         dtl_size_t linear_idx,
                                         void* value);

/**
 * @brief Set element at linear local index
 *
 * @param tensor The tensor
 * @param linear_idx Linear index into local storage
 * @param value Pointer to the element value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_set_local(dtl_tensor_t tensor,
                                         dtl_size_t linear_idx,
                                         const void* value);

/* ==========================================================================
 * Distribution Queries
 * ========================================================================== */

/**
 * @brief Get the number of ranks
 *
 * @param tensor The tensor
 * @return Number of ranks
 */
DTL_API dtl_rank_t dtl_tensor_num_ranks(dtl_tensor_t tensor);

/**
 * @brief Get the rank of this handle
 *
 * @param tensor The tensor
 * @return The rank
 */
DTL_API dtl_rank_t dtl_tensor_rank(dtl_tensor_t tensor);

/**
 * @brief Get the distributed dimension
 *
 * Returns the dimension along which the tensor is partitioned.
 * Typically 0 (first dimension).
 *
 * @param tensor The tensor
 * @return Distributed dimension index
 */
DTL_API int dtl_tensor_distributed_dim(dtl_tensor_t tensor);

/* ==========================================================================
 * Collective Operations
 * ========================================================================== */

/**
 * @brief Reshape the tensor (collective)
 *
 * Changes the shape of the tensor. The total element count must match.
 *
 * Current v1 behavior supports reshapes only when each rank keeps the same
 * local element count after repartitioning. If reshape would require data
 * redistribution across ranks, the function returns `DTL_ERROR_NOT_IMPLEMENTED`.
 *
 * @param tensor The tensor
 * @param new_shape New shape descriptor
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning This is a collective operation.
 * @warning May require redistribution.
 */
DTL_API dtl_status dtl_tensor_reshape(dtl_tensor_t tensor, dtl_shape new_shape);

/**
 * @brief Fill all local elements with a value
 *
 * @param tensor The tensor
 * @param value Pointer to the fill value
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_fill_local(dtl_tensor_t tensor, const void* value);

/**
 * @brief Barrier synchronization
 *
 * @param tensor The tensor
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_tensor_barrier(dtl_tensor_t tensor);

/* ==========================================================================
 * Tensor Validation
 * ========================================================================== */

/**
 * @brief Check if a tensor handle is valid
 *
 * @param tensor The tensor to check (may be NULL)
 * @return 1 if valid, 0 if NULL or invalid
 */
DTL_API int dtl_tensor_is_valid(dtl_tensor_t tensor);

DTL_C_END

/* Mark header as available for master include */
#define DTL_TENSOR_H_AVAILABLE

#endif /* DTL_TENSOR_H */
