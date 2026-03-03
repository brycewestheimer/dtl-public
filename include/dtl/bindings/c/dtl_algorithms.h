// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_algorithms.h
 * @brief DTL C bindings - Algorithm operations on containers
 * @since 0.1.0
 *
 * This header provides C bindings for algorithms operating on distributed
 * containers (vectors and arrays). These algorithms work on the local
 * partition of each container.
 *
 * For collective algorithms that involve communication (e.g., distributed
 * reduce), see dtl_communicator.h.
 *
 * @code
 * // Apply a function to each element
 * void print_element(const void* elem, dtl_size_t idx, void* user_data) {
 *     double* val = (double*)elem;
 *     printf("[%zu] = %f\n", idx, *val);
 * }
 *
 * dtl_for_each_vector(vec, print_element, NULL);
 * @endcode
 */

#ifndef DTL_ALGORITHMS_H
#define DTL_ALGORITHMS_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_vector.h"
#include "dtl_array.h"

DTL_C_BEGIN

/* ==========================================================================
 * Callback Function Types
 * ========================================================================== */

/**
 * @brief Unary function callback for for_each
 *
 * @param element Pointer to the element (type matches container dtype)
 * @param index Local index of the element
 * @param user_data User-provided context pointer
 */
typedef void (*dtl_unary_func)(void* element, dtl_size_t index, void* user_data);

/**
 * @brief Const unary function callback for read-only operations
 *
 * @param element Pointer to the element (type matches container dtype)
 * @param index Local index of the element
 * @param user_data User-provided context pointer
 */
typedef void (*dtl_const_unary_func)(const void* element, dtl_size_t index, void* user_data);

/**
 * @brief Transform callback function
 *
 * Transforms an input element to produce an output element.
 *
 * @param input Pointer to input element
 * @param output Pointer to output element (must be filled by callback)
 * @param index Local index of the element
 * @param user_data User-provided context pointer
 */
typedef void (*dtl_transform_func)(const void* input, void* output, dtl_size_t index, void* user_data);

/**
 * @brief Predicate callback function
 *
 * Returns non-zero if the element matches, zero otherwise.
 *
 * @param element Pointer to the element
 * @param user_data User-provided context pointer
 * @return Non-zero if predicate is true, zero otherwise
 */
typedef int (*dtl_predicate)(const void* element, void* user_data);

/**
 * @brief Comparator callback function for sorting
 *
 * Compares two elements.
 *
 * @param a Pointer to first element
 * @param b Pointer to second element
 * @param user_data User-provided context pointer
 * @return Negative if a < b, zero if a == b, positive if a > b
 */
typedef int (*dtl_comparator)(const void* a, const void* b, void* user_data);

/**
 * @brief Binary reduction callback function
 *
 * Combines two elements into a result.
 *
 * @param a Pointer to first element
 * @param b Pointer to second element
 * @param result Pointer to result (must be filled by callback)
 * @param user_data User-provided context pointer
 */
typedef void (*dtl_binary_func)(const void* a, const void* b, void* result, void* user_data);

/* ==========================================================================
 * For-Each Operations
 * ========================================================================== */

/**
 * @brief Apply a function to each local element of a vector
 *
 * Iterates over all local elements and calls the function for each one.
 * The function may modify elements.
 *
 * @param vec The vector
 * @param func Function to apply (receives mutable pointer)
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note This is a local operation - no communication occurs.
 */
DTL_API dtl_status dtl_for_each_vector(dtl_vector_t vec, dtl_unary_func func,
                                        void* user_data);

/**
 * @brief Apply a read-only function to each local element of a vector
 *
 * Like dtl_for_each_vector, but the function receives const pointers.
 *
 * @param vec The vector
 * @param func Function to apply (receives const pointer)
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_for_each_vector_const(dtl_vector_t vec, dtl_const_unary_func func,
                                              void* user_data);

/**
 * @brief Apply a function to each local element of an array
 *
 * @param arr The array
 * @param func Function to apply (receives mutable pointer)
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_for_each_array(dtl_array_t arr, dtl_unary_func func,
                                       void* user_data);

/**
 * @brief Apply a read-only function to each local element of an array
 *
 * @param arr The array
 * @param func Function to apply (receives const pointer)
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_for_each_array_const(dtl_array_t arr, dtl_const_unary_func func,
                                             void* user_data);

/* ==========================================================================
 * Transform Operations
 * ========================================================================== */

/**
 * @brief Transform elements from source vector to destination vector
 *
 * Applies the transform function to each local element of src and stores
 * the result in the corresponding element of dst.
 *
 * @param src Source vector
 * @param dst Destination vector (must have same local size)
 * @param func Transform function
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre src and dst must have the same local size
 * @note src and dst may be the same vector (in-place transform)
 */
DTL_API dtl_status dtl_transform_vector(dtl_vector_t src, dtl_vector_t dst,
                                         dtl_transform_func func, void* user_data);

/**
 * @brief Transform elements from source array to destination array
 *
 * @param src Source array
 * @param dst Destination array (must have same local size)
 * @param func Transform function
 * @param user_data User-provided context pointer (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_transform_array(dtl_array_t src, dtl_array_t dst,
                                        dtl_transform_func func, void* user_data);

/* ==========================================================================
 * Copy/Fill Operations
 * ========================================================================== */

/**
 * @brief Copy local data from source vector to destination vector
 *
 * Copies all local elements from src to dst. Both must have the same dtype.
 *
 * @param src Source vector
 * @param dst Destination vector
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre src and dst must have the same dtype
 * @pre src and dst must have the same local size
 */
DTL_API dtl_status dtl_copy_vector(dtl_vector_t src, dtl_vector_t dst);

/**
 * @brief Copy local data from source array to destination array
 *
 * @param src Source array
 * @param dst Destination array
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_copy_array(dtl_array_t src, dtl_array_t dst);

/**
 * @brief Fill all local elements of a vector with a value
 *
 * @param vec The vector
 * @param value Pointer to the fill value (must match dtype)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note Equivalent to dtl_vector_fill_local
 */
DTL_API dtl_status dtl_fill_vector(dtl_vector_t vec, const void* value);

/**
 * @brief Fill all local elements of an array with a value
 *
 * @param arr The array
 * @param value Pointer to the fill value (must match dtype)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_fill_array(dtl_array_t arr, const void* value);

/* ==========================================================================
 * Find Operations
 * ========================================================================== */

/**
 * @brief Find the first occurrence of a value in the local partition
 *
 * Searches the local elements for a match.
 *
 * @param vec The vector
 * @param value Pointer to the value to find
 * @return Local index of first match, or -1 if not found
 *
 * @note This searches only the local partition.
 */
DTL_API dtl_index_t dtl_find_vector(dtl_vector_t vec, const void* value);

/**
 * @brief Find the first element satisfying a predicate in the local partition
 *
 * @param vec The vector
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return Local index of first match, or -1 if not found
 */
DTL_API dtl_index_t dtl_find_if_vector(dtl_vector_t vec, dtl_predicate pred,
                                        void* user_data);

/**
 * @brief Find the first occurrence of a value in the local partition
 *
 * @param arr The array
 * @param value Pointer to the value to find
 * @return Local index of first match, or -1 if not found
 */
DTL_API dtl_index_t dtl_find_array(dtl_array_t arr, const void* value);

/**
 * @brief Find the first element satisfying a predicate in the local partition
 *
 * @param arr The array
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return Local index of first match, or -1 if not found
 */
DTL_API dtl_index_t dtl_find_if_array(dtl_array_t arr, dtl_predicate pred,
                                       void* user_data);

/* ==========================================================================
 * Count Operations
 * ========================================================================== */

/**
 * @brief Count occurrences of a value in the local partition
 *
 * @param vec The vector
 * @param value Pointer to the value to count
 * @return Number of matching elements in local partition
 */
DTL_API dtl_size_t dtl_count_vector(dtl_vector_t vec, const void* value);

/**
 * @brief Count elements satisfying a predicate in the local partition
 *
 * @param vec The vector
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return Number of elements satisfying the predicate
 */
DTL_API dtl_size_t dtl_count_if_vector(dtl_vector_t vec, dtl_predicate pred,
                                        void* user_data);

/**
 * @brief Count occurrences of a value in the local partition
 *
 * @param arr The array
 * @param value Pointer to the value to count
 * @return Number of matching elements in local partition
 */
DTL_API dtl_size_t dtl_count_array(dtl_array_t arr, const void* value);

/**
 * @brief Count elements satisfying a predicate in the local partition
 *
 * @param arr The array
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return Number of elements satisfying the predicate
 */
DTL_API dtl_size_t dtl_count_if_array(dtl_array_t arr, dtl_predicate pred,
                                       void* user_data);

/* ==========================================================================
 * Local Reduction Operations
 * ========================================================================== */

/**
 * @brief Reduce local elements of a vector using a built-in operation
 *
 * Reduces all local elements using the specified operation.
 * For distributed reduction, see dtl_reduce/dtl_allreduce in dtl_communicator.h.
 *
 * @param vec The vector
 * @param op Reduction operation (SUM, PROD, MIN, MAX, etc.)
 * @param[out] result Pointer to store the result (must match dtype)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre Vector must have at least one local element
 * @note For distributed reduction, combine with dtl_allreduce.
 */
DTL_API dtl_status dtl_reduce_local_vector(dtl_vector_t vec, dtl_reduce_op op,
                                            void* result);

/**
 * @brief Reduce local elements of a vector using a custom function
 *
 * @param vec The vector
 * @param func Binary reduction function
 * @param identity Initial value for reduction
 * @param[out] result Pointer to store the result
 * @param user_data User-provided context pointer
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_reduce_local_vector_func(dtl_vector_t vec, dtl_binary_func func,
                                                 const void* identity, void* result,
                                                 void* user_data);

/**
 * @brief Reduce local elements of an array using a built-in operation
 *
 * @param arr The array
 * @param op Reduction operation
 * @param[out] result Pointer to store the result
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_reduce_local_array(dtl_array_t arr, dtl_reduce_op op,
                                           void* result);

/**
 * @brief Reduce local elements of an array using a custom function
 *
 * @param arr The array
 * @param func Binary reduction function
 * @param identity Initial value for reduction
 * @param[out] result Pointer to store the result
 * @param user_data User-provided context pointer
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_reduce_local_array_func(dtl_array_t arr, dtl_binary_func func,
                                                const void* identity, void* result,
                                                void* user_data);

/* ==========================================================================
 * Sorting Operations
 * ========================================================================== */

/**
 * @brief Sort local elements of a vector in ascending order
 *
 * Sorts the local partition using the natural ordering for the dtype.
 *
 * @param vec The vector
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note This is a local operation - only the local partition is sorted.
 */
DTL_API dtl_status dtl_sort_vector(dtl_vector_t vec);

/**
 * @brief Sort local elements of a vector in descending order
 *
 * @param vec The vector
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sort_vector_descending(dtl_vector_t vec);

/**
 * @brief Sort local elements of a vector using a custom comparator
 *
 * @param vec The vector
 * @param cmp Comparator function
 * @param user_data User-provided context pointer
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sort_vector_func(dtl_vector_t vec, dtl_comparator cmp,
                                         void* user_data);

/**
 * @brief Sort local elements of an array in ascending order
 *
 * @param arr The array
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sort_array(dtl_array_t arr);

/**
 * @brief Sort local elements of an array in descending order
 *
 * @param arr The array
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sort_array_descending(dtl_array_t arr);

/**
 * @brief Sort local elements of an array using a custom comparator
 *
 * @param arr The array
 * @param cmp Comparator function
 * @param user_data User-provided context pointer
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sort_array_func(dtl_array_t arr, dtl_comparator cmp,
                                        void* user_data);

/* ==========================================================================
 * Min/Max Operations
 * ========================================================================== */

/**
 * @brief Find minimum and maximum values in local vector
 *
 * @param vec The vector
 * @param[out] min_val Pointer to store minimum (may be NULL)
 * @param[out] max_val Pointer to store maximum (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre Vector must have at least one local element
 */
DTL_API dtl_status dtl_minmax_vector(dtl_vector_t vec, void* min_val, void* max_val);

/**
 * @brief Find minimum and maximum values in local array
 *
 * @param arr The array
 * @param[out] min_val Pointer to store minimum (may be NULL)
 * @param[out] max_val Pointer to store maximum (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_minmax_array(dtl_array_t arr, void* min_val, void* max_val);

/* ==========================================================================
 * Predicate Query Operations (Phase 16)
 * ========================================================================== */

/**
 * @brief Check if all local elements satisfy a predicate (vector)
 *
 * @param vec The vector
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if all elements satisfy predicate (or vector is empty), 0 otherwise, -1 on error
 */
DTL_API int dtl_all_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data);

/**
 * @brief Check if any local element satisfies a predicate (vector)
 *
 * @param vec The vector
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if any element satisfies predicate, 0 if none (or empty), -1 on error
 */
DTL_API int dtl_any_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data);

/**
 * @brief Check if no local elements satisfy a predicate (vector)
 *
 * @param vec The vector
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if no elements satisfy predicate (or vector is empty), 0 otherwise, -1 on error
 */
DTL_API int dtl_none_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data);

/**
 * @brief Check if all local elements satisfy a predicate (array)
 *
 * @param arr The array
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if all elements satisfy predicate (or empty), 0 otherwise, -1 on error
 */
DTL_API int dtl_all_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data);

/**
 * @brief Check if any local element satisfies a predicate (array)
 *
 * @param arr The array
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if any element satisfies, 0 if none (or empty), -1 on error
 */
DTL_API int dtl_any_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data);

/**
 * @brief Check if no local elements satisfy a predicate (array)
 *
 * @param arr The array
 * @param pred Predicate function
 * @param user_data User-provided context pointer
 * @return 1 if none satisfy (or empty), 0 otherwise, -1 on error
 */
DTL_API int dtl_none_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data);

/* ==========================================================================
 * Extrema Element Operations (Phase 16)
 * ========================================================================== */

/**
 * @brief Find the index of the minimum element in local vector
 *
 * @param vec The vector
 * @return Local index of minimum element, or -1 if empty/error
 */
DTL_API dtl_index_t dtl_min_element_vector(dtl_vector_t vec);

/**
 * @brief Find the index of the maximum element in local vector
 *
 * @param vec The vector
 * @return Local index of maximum element, or -1 if empty/error
 */
DTL_API dtl_index_t dtl_max_element_vector(dtl_vector_t vec);

/**
 * @brief Find the index of the minimum element in local array
 *
 * @param arr The array
 * @return Local index of minimum element, or -1 if empty/error
 */
DTL_API dtl_index_t dtl_min_element_array(dtl_array_t arr);

/**
 * @brief Find the index of the maximum element in local array
 *
 * @param arr The array
 * @return Local index of maximum element, or -1 if empty/error
 */
DTL_API dtl_index_t dtl_max_element_array(dtl_array_t arr);

/* ==========================================================================
 * Scan / Prefix Operations (Phase 12.5)
 * ========================================================================== */

/**
 * @brief Inclusive scan (prefix sum) on local vector elements
 *
 * Computes in-place inclusive prefix reduction using the specified operation.
 * Element i of the result is the reduction of elements 0..i.
 *
 * @param vec The vector (modified in-place)
 * @param op Reduction operation to apply
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_inclusive_scan_vector(dtl_vector_t vec, dtl_reduce_op op);

/**
 * @brief Exclusive scan (prefix sum) on local vector elements
 *
 * Computes in-place exclusive prefix reduction. Element i of the result
 * is the reduction of elements 0..i-1. The first element is set to the
 * identity for the operation (0 for sum, 1 for product).
 *
 * @param vec The vector (modified in-place)
 * @param op Reduction operation to apply
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_exclusive_scan_vector(dtl_vector_t vec, dtl_reduce_op op);

/**
 * @brief Inclusive scan on local array elements
 *
 * @param arr The array (modified in-place)
 * @param op Reduction operation to apply
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_inclusive_scan_array(dtl_array_t arr, dtl_reduce_op op);

/**
 * @brief Exclusive scan on local array elements
 *
 * @param arr The array (modified in-place)
 * @param op Reduction operation to apply
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_exclusive_scan_array(dtl_array_t arr, dtl_reduce_op op);

/* ==========================================================================
 * Async Algorithm Operations (Phase 12.5 — Experimental)
 * ========================================================================== */

/*
 * WARNING: Async algorithm bindings depend on the DTL futures subsystem
 * which has known stability issues (see KNOWN_ISSUES.md). These functions
 * are experimental and may hang in multi-rank scenarios.
 */

/**
 * @brief Asynchronous for_each on vector (experimental)
 *
 * @param vec The vector
 * @param func Callback function applied to each element
 * @param user_data User context passed to callback
 * @param[out] req Request handle for completion tracking
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_for_each_vector(dtl_vector_t vec,
                                              dtl_unary_func func,
                                              void* user_data,
                                              dtl_request_t* req);

/**
 * @brief Asynchronous transform on vector (experimental)
 *
 * @param src Source vector
 * @param dst Destination vector (must be same size and dtype)
 * @param func Transform callback
 * @param user_data User context
 * @param[out] req Request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_transform_vector(dtl_vector_t src,
                                               dtl_vector_t dst,
                                               dtl_transform_func func,
                                               void* user_data,
                                               dtl_request_t* req);

/**
 * @brief Asynchronous reduce on vector (experimental)
 *
 * @param vec The vector
 * @param op Reduction operation
 * @param[out] result Pointer to store result (written on completion)
 * @param[out] req Request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_reduce_vector(dtl_vector_t vec,
                                            dtl_reduce_op op,
                                            void* result,
                                            dtl_request_t* req);

/**
 * @brief Asynchronous sort on vector (experimental)
 *
 * @param vec The vector (sorted in-place on completion)
 * @param[out] req Request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_sort_vector(dtl_vector_t vec,
                                          dtl_request_t* req);

/**
 * @brief Asynchronous for_each on array (experimental)
 *
 * @param arr The array
 * @param func Callback function
 * @param user_data User context
 * @param[out] req Request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_for_each_array(dtl_array_t arr,
                                             dtl_unary_func func,
                                             void* user_data,
                                             dtl_request_t* req);

/**
 * @brief Asynchronous reduce on array (experimental)
 *
 * @param arr The array
 * @param op Reduction operation
 * @param[out] result Pointer to store result
 * @param[out] req Request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @warning Experimental — may hang due to progress engine issues
 */
DTL_API dtl_status dtl_async_reduce_array(dtl_array_t arr,
                                           dtl_reduce_op op,
                                           void* result,
                                           dtl_request_t* req);

DTL_C_END

/* Mark header as available for master include */
#define DTL_ALGORITHMS_H_AVAILABLE

#endif /* DTL_ALGORITHMS_H */
