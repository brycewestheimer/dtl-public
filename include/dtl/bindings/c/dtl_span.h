// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_span.h
 * @brief DTL C bindings - Distributed span operations
 * @since 0.1.0
 *
 * This header provides C bindings for a non-owning distributed span view.
 * A span borrows contiguous rank-local storage and carries distributed
 * metadata (global size, rank, number of ranks).
 *
 * Spans do not own memory. The backing container/storage must outlive
 * any span handle created from it.
 */

#ifndef DTL_SPAN_H
#define DTL_SPAN_H

#include "dtl_types.h"
#include "dtl_status.h"

DTL_C_BEGIN

/** @brief Sentinel count for subspan-to-end operations */
#define DTL_SPAN_NPOS ((dtl_size_t)-1)

/* ==========================================================================
 * Span Lifecycle
 * ========================================================================== */

/**
 * @brief Create a distributed span from raw local storage and metadata
 *
 * @param dtype Element dtype
 * @param local_data Pointer to local contiguous storage (may be NULL if local_size is 0)
 * @param local_size Number of local elements
 * @param global_size Number of global elements represented by the span
 * @param rank Current rank id
 * @param num_ranks Total rank count
 * @param[out] span Pointer to receive span handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_create(
    dtl_dtype dtype,
    void* local_data,
    dtl_size_t local_size,
    dtl_size_t global_size,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    dtl_span_t* span);

/**
 * @brief Create a span borrowing local storage from a distributed vector
 *
 * @param vec Source vector
 * @param[out] span Pointer to receive span handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_from_vector(dtl_vector_t vec, dtl_span_t* span);

/**
 * @brief Create a span borrowing local storage from a distributed array
 *
 * @param arr Source array
 * @param[out] span Pointer to receive span handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_from_array(dtl_array_t arr, dtl_span_t* span);

/**
 * @brief Create a span borrowing local storage from a distributed tensor
 *
 * Tensor storage is exposed as a flattened local 1D span.
 *
 * @param tensor Source tensor
 * @param[out] span Pointer to receive span handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_from_tensor(dtl_tensor_t tensor, dtl_span_t* span);

/**
 * @brief Destroy a span handle
 *
 * @param span Span to destroy (may be NULL)
 */
DTL_API void dtl_span_destroy(dtl_span_t span);

/* ==========================================================================
 * Span Queries
 * ========================================================================== */

/**
 * @brief Get global size represented by the span
 * @param span The span
 * @return Global size, or 0 on error
 */
DTL_API dtl_size_t dtl_span_size(dtl_span_t span);

/**
 * @brief Get local size represented by the span
 * @param span The span
 * @return Local size, or 0 on error
 */
DTL_API dtl_size_t dtl_span_local_size(dtl_span_t span);

/**
 * @brief Get local size in bytes
 * @param span The span
 * @return Local bytes, or 0 on error
 */
DTL_API dtl_size_t dtl_span_size_bytes(dtl_span_t span);

/**
 * @brief Check whether global size is zero
 * @param span The span
 * @return 1 if empty, 0 otherwise
 */
DTL_API int dtl_span_empty(dtl_span_t span);

/**
 * @brief Get span dtype
 * @param span The span
 * @return Dtype, or -1 on error
 */
DTL_API dtl_dtype dtl_span_dtype(dtl_span_t span);

/**
 * @brief Get local data pointer (read-only)
 * @param span The span
 * @return Local data pointer, or NULL on error
 */
DTL_API const void* dtl_span_data(dtl_span_t span);

/**
 * @brief Get local data pointer (mutable)
 * @param span The span
 * @return Mutable local data pointer, or NULL on error
 */
DTL_API void* dtl_span_data_mut(dtl_span_t span);

/**
 * @brief Get owning rank metadata
 * @param span The span
 * @return Rank id, or DTL_NO_RANK on error
 */
DTL_API dtl_rank_t dtl_span_rank(dtl_span_t span);

/**
 * @brief Get number of ranks metadata
 * @param span The span
 * @return Number of ranks, or 0 on error
 */
DTL_API dtl_rank_t dtl_span_num_ranks(dtl_span_t span);

/* ==========================================================================
 * Span Element Access (Local)
 * ========================================================================== */

/**
 * @brief Read a local element by local index
 *
 * @param span The span
 * @param local_idx Local index
 * @param[out] value Pointer to output value buffer (dtype-sized)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_get_local(
    dtl_span_t span,
    dtl_size_t local_idx,
    void* value);

/**
 * @brief Write a local element by local index
 *
 * @param span The span
 * @param local_idx Local index
 * @param value Pointer to input value buffer (dtype-sized)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_set_local(
    dtl_span_t span,
    dtl_size_t local_idx,
    const void* value);

/* ==========================================================================
 * Subspan Views
 * ========================================================================== */

/**
 * @brief Create a span view of the first @p count local elements
 *
 * @param span Source span
 * @param count Number of local elements
 * @param[out] out_span Output span
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_first(
    dtl_span_t span,
    dtl_size_t count,
    dtl_span_t* out_span);

/**
 * @brief Create a span view of the last @p count local elements
 *
 * @param span Source span
 * @param count Number of local elements
 * @param[out] out_span Output span
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_last(
    dtl_span_t span,
    dtl_size_t count,
    dtl_span_t* out_span);

/**
 * @brief Create a local subspan view
 *
 * If @p count is DTL_SPAN_NPOS, the subspan runs from @p offset to end.
 *
 * @param span Source span
 * @param offset Local start index
 * @param count Local element count or DTL_SPAN_NPOS
 * @param[out] out_span Output span
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_span_subspan(
    dtl_span_t span,
    dtl_size_t offset,
    dtl_size_t count,
    dtl_span_t* out_span);

/* ==========================================================================
 * Validation
 * ========================================================================== */

/**
 * @brief Check if a span handle is valid
 *
 * @param span The span to check (may be NULL)
 * @return 1 if valid, 0 otherwise
 */
DTL_API int dtl_span_is_valid(dtl_span_t span);

DTL_C_END

/* Mark header as available for master include */
#define DTL_SPAN_H_AVAILABLE

#endif /* DTL_SPAN_H */
