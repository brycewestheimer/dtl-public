// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file algorithm_bridge.hpp
 * @brief Bridge between C algorithm API and C++ implementations
 * @since 0.1.0
 *
 * This header provides dispatch helpers for routing C algorithm calls
 * through the vtable to C++ implementations. It ensures that algorithms
 * respect the container's placement policy.
 */

#ifndef DTL_C_DETAIL_ALGORITHM_BRIDGE_HPP
#define DTL_C_DETAIL_ALGORITHM_BRIDGE_HPP

#include "container_vtable.hpp"
#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>

#include <algorithm>
#include <cstring>
#include <functional>

namespace dtl::c::detail {

// ============================================================================
// Algorithm Dispatch Helpers
// ============================================================================

/**
 * @brief Check if an algorithm can run on the given placement
 *
 * Callback-based algorithms cannot run on device-only memory.
 * Built-in operations (fill, reduce) can run on device if implemented.
 */
inline bool can_run_callback_algorithm(dtl_placement_policy placement) noexcept {
    // Callback-based algorithms require host-accessible memory
    return placement == DTL_PLACEMENT_HOST ||
           placement == DTL_PLACEMENT_UNIFIED;
    // Device and device_preferred require explicit GPU kernels
}

/**
 * @brief Check if built-in operations are supported on placement
 */
inline bool can_run_builtin_algorithm(dtl_placement_policy placement) noexcept {
    // Built-in ops work on all placements (via vtable dispatch)
    return placement == DTL_PLACEMENT_HOST ||
           placement == DTL_PLACEMENT_UNIFIED ||
           placement == DTL_PLACEMENT_DEVICE ||
           placement == DTL_PLACEMENT_DEVICE_PREFERRED;
}

// ============================================================================
// Vector Algorithm Dispatch
// ============================================================================

/**
 * @brief Apply a unary function to each element (host-only)
 */
template <typename UnaryFunc>
dtl_status vector_for_each(
    vector_handle* h,
    UnaryFunc func,
    void* user_data) {

    if (!is_valid_vector_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    void* data = h->vtable->local_data_mut(h->impl);
    if (!data) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    std::size_t local_size = h->vtable->local_size(h->impl);
    std::size_t elem_size = dtl_dtype_size(h->base.dtype);

    for (std::size_t i = 0; i < local_size; ++i) {
        char* elem = static_cast<char*>(data) + i * elem_size;
        func(elem, i, user_data);
    }

    return DTL_SUCCESS;
}

/**
 * @brief Apply a transform function (host-only)
 */
template <typename TransformFunc>
dtl_status vector_transform(
    const vector_handle* src_h,
    vector_handle* dst_h,
    TransformFunc func,
    void* user_data) {

    if (!is_valid_vector_handle(src_h) || !is_valid_vector_handle(dst_h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    if (!can_run_callback_algorithm(src_h->base.options.placement) ||
        !can_run_callback_algorithm(dst_h->base.options.placement)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    const void* src_data = src_h->vtable->local_data(src_h->impl);
    void* dst_data = dst_h->vtable->local_data_mut(dst_h->impl);

    if (!src_data || !dst_data) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    std::size_t src_size = src_h->vtable->local_size(src_h->impl);
    std::size_t dst_size = dst_h->vtable->local_size(dst_h->impl);
    std::size_t count = std::min(src_size, dst_size);

    std::size_t src_elem_size = dtl_dtype_size(src_h->base.dtype);
    std::size_t dst_elem_size = dtl_dtype_size(dst_h->base.dtype);

    for (std::size_t i = 0; i < count; ++i) {
        const char* src = static_cast<const char*>(src_data) + i * src_elem_size;
        char* dst = static_cast<char*>(dst_data) + i * dst_elem_size;
        func(src, dst, i, user_data);
    }

    return DTL_SUCCESS;
}

/**
 * @brief Reduce sum (works on all placements via vtable)
 */
inline dtl_status vector_reduce_sum(const vector_handle* h, void* result) {
    if (!is_valid_vector_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_sum(h->impl, result);
}

/**
 * @brief Reduce min (works on all placements via vtable)
 */
inline dtl_status vector_reduce_min(const vector_handle* h, void* result) {
    if (!is_valid_vector_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_min(h->impl, result);
}

/**
 * @brief Reduce max (works on all placements via vtable)
 */
inline dtl_status vector_reduce_max(const vector_handle* h, void* result) {
    if (!is_valid_vector_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_max(h->impl, result);
}

/**
 * @brief Sort ascending (works on all placements via vtable)
 */
inline dtl_status vector_sort_ascending(vector_handle* h) {
    if (!is_valid_vector_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->sort_ascending(h->impl);
}

/**
 * @brief Sort descending (works on all placements via vtable)
 */
inline dtl_status vector_sort_descending(vector_handle* h) {
    if (!is_valid_vector_handle(h)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->sort_descending(h->impl);
}

/**
 * @brief Fill (works on all placements via vtable)
 */
inline dtl_status vector_fill(vector_handle* h, const void* value) {
    if (!is_valid_vector_handle(h) || !value) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->fill(h->impl, value);
}

// ============================================================================
// Array Algorithm Dispatch
// ============================================================================

// Array algorithms follow the same pattern as vectors
inline dtl_status array_reduce_sum(const array_handle* h, void* result) {
    if (!is_valid_array_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_sum(h->impl, result);
}

inline dtl_status array_reduce_min(const array_handle* h, void* result) {
    if (!is_valid_array_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_min(h->impl, result);
}

inline dtl_status array_reduce_max(const array_handle* h, void* result) {
    if (!is_valid_array_handle(h) || !result) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->reduce_max(h->impl, result);
}

inline dtl_status array_fill(array_handle* h, const void* value) {
    if (!is_valid_array_handle(h) || !value) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    return h->vtable->fill(h->impl, value);
}

}  // namespace dtl::c::detail

#endif /* DTL_C_DETAIL_ALGORITHM_BRIDGE_HPP */
