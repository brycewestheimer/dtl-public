// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_algorithms.cpp
 * @brief DTL C bindings - Algorithm operations implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_algorithms.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/futures/progress.hpp>

#include "dtl_internal.hpp"
#include "detail/algorithm_bridge.hpp"
#include "detail/container_vtable.hpp"
#include "detail/error_policy.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <limits>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// Handle Helpers (vtable-based C ABI)
// ============================================================================

using namespace dtl::c::detail;

static vector_handle* get_vector_handle(dtl_vector_t vec) {
    auto* h = reinterpret_cast<vector_handle*>(vec);
    return is_valid_vector_handle(h) ? h : nullptr;
}

static const vector_handle* get_vector_handle_const(dtl_vector_t vec) {
    auto* h = reinterpret_cast<const vector_handle*>(vec);
    return is_valid_vector_handle(h) ? h : nullptr;
}

static array_handle* get_array_handle(dtl_array_t arr) {
    auto* h = reinterpret_cast<array_handle*>(arr);
    return is_valid_array_handle(h) ? h : nullptr;
}

static const array_handle* get_array_handle_const(dtl_array_t arr) {
    auto* h = reinterpret_cast<const array_handle*>(arr);
    return is_valid_array_handle(h) ? h : nullptr;
}

// ============================================================================
// Type-safe algorithm implementations
// ============================================================================

// For-each template
template <typename T>
static void for_each_impl(T* data, dtl_size_t size, dtl_unary_func func, void* user_data) {
    for (dtl_size_t i = 0; i < size; ++i) {
        func(&data[i], i, user_data);
    }
}

template <typename T>
static void for_each_const_impl(const T* data, dtl_size_t size, dtl_const_unary_func func, void* user_data) {
    for (dtl_size_t i = 0; i < size; ++i) {
        func(&data[i], i, user_data);
    }
}

// Transform template
template <typename SrcT, typename DstT>
static void transform_impl(const SrcT* src, DstT* dst, dtl_size_t size,
                           dtl_transform_func func, void* user_data) {
    for (dtl_size_t i = 0; i < size; ++i) {
        func(&src[i], &dst[i], i, user_data);
    }
}

// Find template
template <typename T>
static dtl_index_t find_impl(const T* data, dtl_size_t size, const T& value) {
    for (dtl_size_t i = 0; i < size; ++i) {
        if (data[i] == value) {
            return static_cast<dtl_index_t>(i);
        }
    }
    return -1;
}

template <typename T>
static dtl_index_t find_if_impl(const T* data, dtl_size_t size,
                                dtl_predicate pred, void* user_data) {
    for (dtl_size_t i = 0; i < size; ++i) {
        if (pred(&data[i], user_data)) {
            return static_cast<dtl_index_t>(i);
        }
    }
    return -1;
}

// Count template
template <typename T>
static dtl_size_t count_impl(const T* data, dtl_size_t size, const T& value) {
    dtl_size_t count = 0;
    for (dtl_size_t i = 0; i < size; ++i) {
        if (data[i] == value) {
            ++count;
        }
    }
    return count;
}

template <typename T>
static dtl_size_t count_if_impl(const T* data, dtl_size_t size,
                                 dtl_predicate pred, void* user_data) {
    dtl_size_t count = 0;
    for (dtl_size_t i = 0; i < size; ++i) {
        if (pred(&data[i], user_data)) {
            ++count;
        }
    }
    return count;
}

// Reduce template with built-in ops
template <typename T>
static T reduce_sum(const T* data, dtl_size_t size) {
    T result = T{};
    for (dtl_size_t i = 0; i < size; ++i) {
        result += data[i];
    }
    return result;
}

template <typename T>
static T reduce_prod(const T* data, dtl_size_t size) {
    T result = T{1};
    for (dtl_size_t i = 0; i < size; ++i) {
        result *= data[i];
    }
    return result;
}

template <typename T>
static T reduce_min(const T* data, dtl_size_t size) {
    T result = data[0];
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i] < result) result = data[i];
    }
    return result;
}

template <typename T>
static T reduce_max(const T* data, dtl_size_t size) {
    T result = data[0];
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// Sort templates
template <typename T>
static void sort_ascending_impl(T* data, dtl_size_t size) {
    std::sort(data, data + size);
}

template <typename T>
static void sort_descending_impl(T* data, dtl_size_t size) {
    std::sort(data, data + size, std::greater<T>());
}

template <typename T>
static void sort_func_impl(T* data, dtl_size_t size, dtl_comparator cmp, void* user_data) {
    std::sort(data, data + size, [cmp, user_data](const T& a, const T& b) {
        return cmp(&a, &b, user_data) < 0;
    });
}

// MinMax template
template <typename T>
static void minmax_impl(const T* data, dtl_size_t size, T* min_val, T* max_val) {
    if (size == 0) return;

    T min_v = data[0];
    T max_v = data[0];
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }
    if (min_val) *min_val = min_v;
    if (max_val) *max_val = max_v;
}

// ============================================================================
// For-Each Operations
// ============================================================================

extern "C" {

dtl_status dtl_for_each_vector(dtl_vector_t vec, dtl_unary_func func, void* user_data) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) {
        return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* bytes = static_cast<char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(bytes + i * elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_for_each_vector_const(dtl_vector_t vec, dtl_const_unary_func func, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) {
        return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(bytes + i * elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_for_each_array(dtl_array_t arr, dtl_unary_func func, void* user_data) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) {
        return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* bytes = static_cast<char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(bytes + i * elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_for_each_array_const(dtl_array_t arr, dtl_const_unary_func func, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) {
        return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(bytes + i * elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Transform Operations
// ============================================================================

dtl_status dtl_transform_vector(dtl_vector_t src, dtl_vector_t dst,
                                 dtl_transform_func func, void* user_data) {
    const auto* src_h = get_vector_handle_const(src);
    auto* dst_h = get_vector_handle(dst);
    if (!src_h || !dst_h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(dst_h, DTL_ERROR_NULL_POINTER);

    std::size_t src_size = src_h->vtable->local_size(src_h->impl);
    std::size_t dst_size = dst_h->vtable->local_size(dst_h->impl);
    if (src_size != dst_size) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    if (!can_run_callback_algorithm(src_h->base.options.placement) ||
        !can_run_callback_algorithm(dst_h->base.options.placement)) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* src_data = src_h->vtable->local_data(src_h->impl);
    void* dst_data = dst_h->vtable->local_data_mut(dst_h->impl);
    if ((!src_data || !dst_data) && src_size != 0) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t src_elem_size = dtl_dtype_size(src_h->base.dtype);
    dtl_size_t dst_elem_size = dtl_dtype_size(dst_h->base.dtype);
    if (src_elem_size == 0 || dst_elem_size == 0) {
        return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* src_bytes = static_cast<const char*>(src_data);
    auto* dst_bytes = static_cast<char*>(dst_data);
    for (std::size_t i = 0; i < src_size; ++i) {
        func(src_bytes + i * src_elem_size, dst_bytes + i * dst_elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_transform_array(dtl_array_t src, dtl_array_t dst,
                                dtl_transform_func func, void* user_data) {
    const auto* src_h = get_array_handle_const(src);
    auto* dst_h = get_array_handle(dst);
    if (!src_h || !dst_h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(dst_h, DTL_ERROR_NULL_POINTER);

    std::size_t src_size = src_h->vtable->local_size(src_h->impl);
    std::size_t dst_size = dst_h->vtable->local_size(dst_h->impl);
    if (src_size != dst_size) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    if (!can_run_callback_algorithm(src_h->base.options.placement) ||
        !can_run_callback_algorithm(dst_h->base.options.placement)) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* src_data = src_h->vtable->local_data(src_h->impl);
    void* dst_data = dst_h->vtable->local_data_mut(dst_h->impl);
    if ((!src_data || !dst_data) && src_size != 0) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    dtl_size_t src_elem_size = dtl_dtype_size(src_h->base.dtype);
    dtl_size_t dst_elem_size = dtl_dtype_size(dst_h->base.dtype);
    if (src_elem_size == 0 || dst_elem_size == 0) {
        return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);
    }

    auto* src_bytes = static_cast<const char*>(src_data);
    auto* dst_bytes = static_cast<char*>(dst_data);
    for (std::size_t i = 0; i < src_size; ++i) {
        func(src_bytes + i * src_elem_size, dst_bytes + i * dst_elem_size, static_cast<dtl_size_t>(i), user_data);
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Copy/Fill Operations
// ============================================================================

dtl_status dtl_copy_vector(dtl_vector_t src, dtl_vector_t dst) {
    const auto* src_h = get_vector_handle_const(src);
    auto* dst_h = get_vector_handle(dst);
    if (!src_h || !dst_h) return DTL_ERROR_INVALID_ARGUMENT;
    if (src_h->base.dtype != dst_h->base.dtype) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    std::size_t src_size = src_h->vtable->local_size(src_h->impl);
    std::size_t dst_size = dst_h->vtable->local_size(dst_h->impl);
    if (src_size != dst_size) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    if (!can_run_callback_algorithm(src_h->base.options.placement) ||
        !can_run_callback_algorithm(dst_h->base.options.placement)) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* src_data = src_h->vtable->local_data(src_h->impl);
    void* dst_data = dst_h->vtable->local_data_mut(dst_h->impl);
    if ((!src_data || !dst_data) && src_size != 0) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const dtl_size_t elem_size = dtl_dtype_size(src_h->base.dtype);
    if (elem_size == 0) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    const dtl_size_t bytes = static_cast<dtl_size_t>(src_size) * elem_size;
    std::memcpy(dst_data, src_data, bytes);

    return DTL_SUCCESS;
}

dtl_status dtl_copy_array(dtl_array_t src, dtl_array_t dst) {
    const auto* src_h = get_array_handle_const(src);
    auto* dst_h = get_array_handle(dst);
    if (!src_h || !dst_h) return DTL_ERROR_INVALID_ARGUMENT;
    if (src_h->base.dtype != dst_h->base.dtype) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    std::size_t src_size = src_h->vtable->local_size(src_h->impl);
    std::size_t dst_size = dst_h->vtable->local_size(dst_h->impl);
    if (src_size != dst_size) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    if (!can_run_callback_algorithm(src_h->base.options.placement) ||
        !can_run_callback_algorithm(dst_h->base.options.placement)) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const void* src_data = src_h->vtable->local_data(src_h->impl);
    void* dst_data = dst_h->vtable->local_data_mut(dst_h->impl);
    if ((!src_data || !dst_data) && src_size != 0) {
        return apply_error_policy(dst_h, DTL_ERROR_NOT_SUPPORTED);
    }

    const dtl_size_t elem_size = dtl_dtype_size(src_h->base.dtype);
    if (elem_size == 0) return apply_error_policy(dst_h, DTL_ERROR_INVALID_ARGUMENT);

    const dtl_size_t bytes = static_cast<dtl_size_t>(src_size) * elem_size;
    std::memcpy(dst_data, src_data, bytes);

    return DTL_SUCCESS;
}

dtl_status dtl_fill_vector(dtl_vector_t vec, const void* value) {
    return dtl_vector_fill_local(vec, value);
}

dtl_status dtl_fill_array(dtl_array_t arr, const void* value) {
    return dtl_array_fill_local(arr, value);
}

// ============================================================================
// Find Operations
// ============================================================================

dtl_index_t dtl_find_vector(dtl_vector_t vec, const void* value) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !value) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return -1;

    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            return find_impl(static_cast<const int8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int8_t*>(value));
        case DTL_DTYPE_INT16:
            return find_impl(static_cast<const int16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int16_t*>(value));
        case DTL_DTYPE_INT32:
            return find_impl(static_cast<const int32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int32_t*>(value));
        case DTL_DTYPE_INT64:
            return find_impl(static_cast<const int64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int64_t*>(value));
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            return find_impl(static_cast<const uint8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint8_t*>(value));
        case DTL_DTYPE_UINT16:
            return find_impl(static_cast<const uint16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint16_t*>(value));
        case DTL_DTYPE_UINT32:
            return find_impl(static_cast<const uint32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint32_t*>(value));
        case DTL_DTYPE_UINT64:
            return find_impl(static_cast<const uint64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint64_t*>(value));
        case DTL_DTYPE_FLOAT32:
            return find_impl(static_cast<const float*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const float*>(value));
        case DTL_DTYPE_FLOAT64:
            return find_impl(static_cast<const double*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const double*>(value));
        default:
            return -1;
    }
}

dtl_index_t dtl_find_if_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) {
            return static_cast<dtl_index_t>(i);
        }
    }
    return -1;
}

dtl_index_t dtl_find_array(dtl_array_t arr, const void* value) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !value) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return -1;

    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            return find_impl(static_cast<const int8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int8_t*>(value));
        case DTL_DTYPE_INT16:
            return find_impl(static_cast<const int16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int16_t*>(value));
        case DTL_DTYPE_INT32:
            return find_impl(static_cast<const int32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int32_t*>(value));
        case DTL_DTYPE_INT64:
            return find_impl(static_cast<const int64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int64_t*>(value));
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            return find_impl(static_cast<const uint8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint8_t*>(value));
        case DTL_DTYPE_UINT16:
            return find_impl(static_cast<const uint16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint16_t*>(value));
        case DTL_DTYPE_UINT32:
            return find_impl(static_cast<const uint32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint32_t*>(value));
        case DTL_DTYPE_UINT64:
            return find_impl(static_cast<const uint64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint64_t*>(value));
        case DTL_DTYPE_FLOAT32:
            return find_impl(static_cast<const float*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const float*>(value));
        case DTL_DTYPE_FLOAT64:
            return find_impl(static_cast<const double*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const double*>(value));
        default:
            return -1;
    }
}

dtl_index_t dtl_find_if_array(dtl_array_t arr, dtl_predicate pred, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) {
            return static_cast<dtl_index_t>(i);
        }
    }
    return -1;
}

// ============================================================================
// Count Operations
// ============================================================================

dtl_size_t dtl_count_vector(dtl_vector_t vec, const void* value) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !value) return 0;
    if (!can_run_callback_algorithm(h->base.options.placement)) return 0;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return 0;

    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            return count_impl(static_cast<const int8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int8_t*>(value));
        case DTL_DTYPE_INT16:
            return count_impl(static_cast<const int16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int16_t*>(value));
        case DTL_DTYPE_INT32:
            return count_impl(static_cast<const int32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int32_t*>(value));
        case DTL_DTYPE_INT64:
            return count_impl(static_cast<const int64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int64_t*>(value));
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            return count_impl(static_cast<const uint8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint8_t*>(value));
        case DTL_DTYPE_UINT16:
            return count_impl(static_cast<const uint16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint16_t*>(value));
        case DTL_DTYPE_UINT32:
            return count_impl(static_cast<const uint32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint32_t*>(value));
        case DTL_DTYPE_UINT64:
            return count_impl(static_cast<const uint64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint64_t*>(value));
        case DTL_DTYPE_FLOAT32:
            return count_impl(static_cast<const float*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const float*>(value));
        case DTL_DTYPE_FLOAT64:
            return count_impl(static_cast<const double*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const double*>(value));
        default:
            return 0;
    }
}

dtl_size_t dtl_count_if_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !pred) return 0;
    if (!can_run_callback_algorithm(h->base.options.placement)) return 0;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return 0;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return 0;

    dtl_size_t count = 0;
    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) {
            ++count;
        }
    }
    return count;
}

dtl_size_t dtl_count_array(dtl_array_t arr, const void* value) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !value) return 0;
    if (!can_run_callback_algorithm(h->base.options.placement)) return 0;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return 0;

    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            return count_impl(static_cast<const int8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int8_t*>(value));
        case DTL_DTYPE_INT16:
            return count_impl(static_cast<const int16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int16_t*>(value));
        case DTL_DTYPE_INT32:
            return count_impl(static_cast<const int32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int32_t*>(value));
        case DTL_DTYPE_INT64:
            return count_impl(static_cast<const int64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const int64_t*>(value));
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            return count_impl(static_cast<const uint8_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint8_t*>(value));
        case DTL_DTYPE_UINT16:
            return count_impl(static_cast<const uint16_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint16_t*>(value));
        case DTL_DTYPE_UINT32:
            return count_impl(static_cast<const uint32_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint32_t*>(value));
        case DTL_DTYPE_UINT64:
            return count_impl(static_cast<const uint64_t*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const uint64_t*>(value));
        case DTL_DTYPE_FLOAT32:
            return count_impl(static_cast<const float*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const float*>(value));
        case DTL_DTYPE_FLOAT64:
            return count_impl(static_cast<const double*>(data), static_cast<dtl_size_t>(local_size), *static_cast<const double*>(value));
        default:
            return 0;
    }
}

dtl_size_t dtl_count_if_array(dtl_array_t arr, dtl_predicate pred, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !pred) return 0;
    if (!can_run_callback_algorithm(h->base.options.placement)) return 0;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return 0;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return 0;

    dtl_size_t count = 0;
    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) {
            ++count;
        }
    }
    return count;
}

// ============================================================================
// Local Reduction Operations
// ============================================================================

dtl_status dtl_reduce_local_vector(dtl_vector_t vec, dtl_reduce_op op, void* result) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    switch (op) {
        case DTL_OP_SUM:
            return apply_error_policy(h, h->vtable->reduce_sum(h->impl, result));
        case DTL_OP_MIN:
            return apply_error_policy(h, h->vtable->reduce_min(h->impl, result));
        case DTL_OP_MAX:
            return apply_error_policy(h, h->vtable->reduce_max(h->impl, result));
        case DTL_OP_PROD: {
            const void* data = h->vtable->local_data(h->impl);
            if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

            const auto size = static_cast<dtl_size_t>(local_size);
            switch (h->base.dtype) {
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(result) = reduce_prod(static_cast<const int8_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT16:
                    *static_cast<int16_t*>(result) = reduce_prod(static_cast<const int16_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(result) = reduce_prod(static_cast<const int32_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(result) = reduce_prod(static_cast<const int64_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT8:
                case DTL_DTYPE_BYTE:
                case DTL_DTYPE_BOOL:
                    *static_cast<uint8_t*>(result) = reduce_prod(static_cast<const uint8_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT16:
                    *static_cast<uint16_t*>(result) = reduce_prod(static_cast<const uint16_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(result) = reduce_prod(static_cast<const uint32_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(result) = reduce_prod(static_cast<const uint64_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(result) = reduce_prod(static_cast<const float*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(result) = reduce_prod(static_cast<const double*>(data), size);
                    return DTL_SUCCESS;
                default:
                    return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
            }
        }
        default:
            return apply_error_policy(h, h->vtable->reduce_sum(h->impl, result));
    }
}

dtl_status dtl_reduce_local_vector_func(dtl_vector_t vec, dtl_binary_func func,
                                         const void* identity, void* result,
                                         void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func || !identity || !result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    std::size_t local_size = h->vtable->local_size(h->impl);
    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    const void* data = h->vtable->local_data(h->impl);
    if (!data && local_size != 0) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    // Copy identity to result
    std::memcpy(result, identity, elem_size);

    // Fold left
    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(result, bytes + i * elem_size, result, user_data);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_reduce_local_array(dtl_array_t arr, dtl_reduce_op op, void* result) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    switch (op) {
        case DTL_OP_SUM:
            return apply_error_policy(h, h->vtable->reduce_sum(h->impl, result));
        case DTL_OP_MIN:
            return apply_error_policy(h, h->vtable->reduce_min(h->impl, result));
        case DTL_OP_MAX:
            return apply_error_policy(h, h->vtable->reduce_max(h->impl, result));
        case DTL_OP_PROD: {
            const void* data = h->vtable->local_data(h->impl);
            if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

            const auto size = static_cast<dtl_size_t>(local_size);
            switch (h->base.dtype) {
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(result) = reduce_prod(static_cast<const int8_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT16:
                    *static_cast<int16_t*>(result) = reduce_prod(static_cast<const int16_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(result) = reduce_prod(static_cast<const int32_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(result) = reduce_prod(static_cast<const int64_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT8:
                case DTL_DTYPE_BYTE:
                case DTL_DTYPE_BOOL:
                    *static_cast<uint8_t*>(result) = reduce_prod(static_cast<const uint8_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT16:
                    *static_cast<uint16_t*>(result) = reduce_prod(static_cast<const uint16_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(result) = reduce_prod(static_cast<const uint32_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(result) = reduce_prod(static_cast<const uint64_t*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(result) = reduce_prod(static_cast<const float*>(data), size);
                    return DTL_SUCCESS;
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(result) = reduce_prod(static_cast<const double*>(data), size);
                    return DTL_SUCCESS;
                default:
                    return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
            }
        }
        default:
            return apply_error_policy(h, h->vtable->reduce_sum(h->impl, result));
    }
}

dtl_status dtl_reduce_local_array_func(dtl_array_t arr, dtl_binary_func func,
                                        const void* identity, void* result,
                                        void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func || !identity || !result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    std::size_t local_size = h->vtable->local_size(h->impl);
    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    const void* data = h->vtable->local_data(h->impl);
    if (!data && local_size != 0) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    std::memcpy(result, identity, elem_size);

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        func(result, bytes + i * elem_size, result, user_data);
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Sorting Operations
// ============================================================================

dtl_status dtl_sort_vector(dtl_vector_t vec) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, h->vtable->sort_ascending(h->impl));
}

dtl_status dtl_sort_vector_descending(dtl_vector_t vec) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    return apply_error_policy(h, h->vtable->sort_descending(h->impl));
}

dtl_status dtl_sort_vector_func(dtl_vector_t vec, dtl_comparator cmp, void* user_data) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!cmp) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    const auto size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            sort_func_impl(static_cast<int8_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT16:
            sort_func_impl(static_cast<int16_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT32:
            sort_func_impl(static_cast<int32_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT64:
            sort_func_impl(static_cast<int64_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            sort_func_impl(static_cast<uint8_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT16:
            sort_func_impl(static_cast<uint16_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT32:
            sort_func_impl(static_cast<uint32_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT64:
            sort_func_impl(static_cast<uint64_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_FLOAT32:
            sort_func_impl(static_cast<float*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_FLOAT64:
            sort_func_impl(static_cast<double*>(data), size, cmp, user_data);
            break;
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_sort_array(dtl_array_t arr) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const auto size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            sort_ascending_impl(static_cast<int8_t*>(data), size);
            break;
        case DTL_DTYPE_INT16:
            sort_ascending_impl(static_cast<int16_t*>(data), size);
            break;
        case DTL_DTYPE_INT32:
            sort_ascending_impl(static_cast<int32_t*>(data), size);
            break;
        case DTL_DTYPE_INT64:
            sort_ascending_impl(static_cast<int64_t*>(data), size);
            break;
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            sort_ascending_impl(static_cast<uint8_t*>(data), size);
            break;
        case DTL_DTYPE_UINT16:
            sort_ascending_impl(static_cast<uint16_t*>(data), size);
            break;
        case DTL_DTYPE_UINT32:
            sort_ascending_impl(static_cast<uint32_t*>(data), size);
            break;
        case DTL_DTYPE_UINT64:
            sort_ascending_impl(static_cast<uint64_t*>(data), size);
            break;
        case DTL_DTYPE_FLOAT32:
            sort_ascending_impl(static_cast<float*>(data), size);
            break;
        case DTL_DTYPE_FLOAT64:
            sort_ascending_impl(static_cast<double*>(data), size);
            break;
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_sort_array_descending(dtl_array_t arr) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const auto size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            sort_descending_impl(static_cast<int8_t*>(data), size);
            break;
        case DTL_DTYPE_INT16:
            sort_descending_impl(static_cast<int16_t*>(data), size);
            break;
        case DTL_DTYPE_INT32:
            sort_descending_impl(static_cast<int32_t*>(data), size);
            break;
        case DTL_DTYPE_INT64:
            sort_descending_impl(static_cast<int64_t*>(data), size);
            break;
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            sort_descending_impl(static_cast<uint8_t*>(data), size);
            break;
        case DTL_DTYPE_UINT16:
            sort_descending_impl(static_cast<uint16_t*>(data), size);
            break;
        case DTL_DTYPE_UINT32:
            sort_descending_impl(static_cast<uint32_t*>(data), size);
            break;
        case DTL_DTYPE_UINT64:
            sort_descending_impl(static_cast<uint64_t*>(data), size);
            break;
        case DTL_DTYPE_FLOAT32:
            sort_descending_impl(static_cast<float*>(data), size);
            break;
        case DTL_DTYPE_FLOAT64:
            sort_descending_impl(static_cast<double*>(data), size);
            break;
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_sort_array_func(dtl_array_t arr, dtl_comparator cmp, void* user_data) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!cmp) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);

    if (!can_run_callback_algorithm(h->base.options.placement)) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    void* data = h->vtable->local_data_mut(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data && local_size != 0) {
        return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);
    }

    const auto size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8:
            sort_func_impl(static_cast<int8_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT16:
            sort_func_impl(static_cast<int16_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT32:
            sort_func_impl(static_cast<int32_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_INT64:
            sort_func_impl(static_cast<int64_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:
            sort_func_impl(static_cast<uint8_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT16:
            sort_func_impl(static_cast<uint16_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT32:
            sort_func_impl(static_cast<uint32_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_UINT64:
            sort_func_impl(static_cast<uint64_t*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_FLOAT32:
            sort_func_impl(static_cast<float*>(data), size, cmp, user_data);
            break;
        case DTL_DTYPE_FLOAT64:
            sort_func_impl(static_cast<double*>(data), size, cmp, user_data);
            break;
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Min/Max Operations
// ============================================================================

dtl_status dtl_minmax_vector(dtl_vector_t vec, void* min_val, void* max_val) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    if (!min_val && !max_val) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    if (min_val) {
        dtl_status st = h->vtable->reduce_min(h->impl, min_val);
        if (st != DTL_SUCCESS) return apply_error_policy(h, st);
    }
    if (max_val) {
        dtl_status st = h->vtable->reduce_max(h->impl, max_val);
        if (st != DTL_SUCCESS) return apply_error_policy(h, st);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_minmax_array(dtl_array_t arr, void* min_val, void* max_val) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    if (!min_val && !max_val) return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);

    if (min_val) {
        dtl_status st = h->vtable->reduce_min(h->impl, min_val);
        if (st != DTL_SUCCESS) return apply_error_policy(h, st);
    }
    if (max_val) {
        dtl_status st = h->vtable->reduce_max(h->impl, max_val);
        if (st != DTL_SUCCESS) return apply_error_policy(h, st);
    }

    return DTL_SUCCESS;
}

} // extern "C" (pause for C++ template helpers)

// ============================================================================
// Predicate Query Operations (Phase 16)
// ============================================================================

extern "C" {

int dtl_all_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 1;  // vacuously true
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (!pred(bytes + i * elem_size, user_data)) return 0;
    }
    return 1;
}

int dtl_any_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 0;
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) return 1;
    }
    return 0;
}

int dtl_none_of_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data) {
    const auto* h = get_vector_handle_const(vec);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 1;  // vacuously true
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) return 0;
    }
    return 1;
}

int dtl_all_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 1;
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (!pred(bytes + i * elem_size, user_data)) return 0;
    }
    return 1;
}

int dtl_any_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 0;
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) return 1;
    }
    return 0;
}

int dtl_none_of_array(dtl_array_t arr, dtl_predicate pred, void* user_data) {
    const auto* h = get_array_handle_const(arr);
    if (!h || !pred) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return 1;
    if (!data) return -1;

    dtl_size_t elem_size = dtl_dtype_size(h->base.dtype);
    if (elem_size == 0) return -1;

    auto* bytes = static_cast<const char*>(data);
    for (std::size_t i = 0; i < local_size; ++i) {
        if (pred(bytes + i * elem_size, user_data)) return 0;
    }
    return 1;
}

} // extern "C"

// ============================================================================
// Extrema Element Operations (Phase 16)
// ============================================================================

template <typename T>
static dtl_index_t min_element_impl(const T* data, dtl_size_t size) {
    if (size == 0) return -1;
    dtl_index_t idx = 0;
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i] < data[idx]) idx = static_cast<dtl_index_t>(i);
    }
    return idx;
}

template <typename T>
static dtl_index_t max_element_impl(const T* data, dtl_size_t size) {
    if (size == 0) return -1;
    dtl_index_t idx = 0;
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i] > data[idx]) idx = static_cast<dtl_index_t>(i);
    }
    return idx;
}

// Macro for dispatching min/max element across dtypes
#define DISPATCH_EXTREMA(FUNC, DATA, SIZE, DTYPE) \
    switch (DTYPE) { \
        case DTL_DTYPE_INT8:    return FUNC(static_cast<const int8_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_INT16:   return FUNC(static_cast<const int16_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_INT32:   return FUNC(static_cast<const int32_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_INT64:   return FUNC(static_cast<const int64_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_UINT8: \
        case DTL_DTYPE_BYTE: \
        case DTL_DTYPE_BOOL:    return FUNC(static_cast<const uint8_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_UINT16:  return FUNC(static_cast<const uint16_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_UINT32:  return FUNC(static_cast<const uint32_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_UINT64:  return FUNC(static_cast<const uint64_t*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_FLOAT32: return FUNC(static_cast<const float*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        case DTL_DTYPE_FLOAT64: return FUNC(static_cast<const double*>(DATA), static_cast<dtl_size_t>(SIZE)); \
        default: return -1; \
    }

extern "C" {

dtl_index_t dtl_min_element_vector(dtl_vector_t vec) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data || local_size == 0) return -1;

    DISPATCH_EXTREMA(min_element_impl, data, local_size, h->base.dtype)
}

dtl_index_t dtl_max_element_vector(dtl_vector_t vec) {
    const auto* h = get_vector_handle_const(vec);
    if (!h) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data || local_size == 0) return -1;

    DISPATCH_EXTREMA(max_element_impl, data, local_size, h->base.dtype)
}

dtl_index_t dtl_min_element_array(dtl_array_t arr) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data || local_size == 0) return -1;

    DISPATCH_EXTREMA(min_element_impl, data, local_size, h->base.dtype)
}

dtl_index_t dtl_max_element_array(dtl_array_t arr) {
    const auto* h = get_array_handle_const(arr);
    if (!h) return -1;
    if (!can_run_callback_algorithm(h->base.options.placement)) return -1;

    const void* data = h->vtable->local_data(h->impl);
    std::size_t local_size = h->vtable->local_size(h->impl);
    if (!data || local_size == 0) return -1;

    DISPATCH_EXTREMA(max_element_impl, data, local_size, h->base.dtype)
}

#undef DISPATCH_EXTREMA

} // extern "C" (pause for C++ template helpers)

// ============================================================================
// Scan / Prefix Operations
// ============================================================================

// Inclusive scan template: element i = reduction of elements 0..i
template <typename T>
static void inclusive_scan_sum_impl(T* data, dtl_size_t size) {
    for (dtl_size_t i = 1; i < size; ++i) {
        data[i] += data[i - 1];
    }
}

template <typename T>
static void inclusive_scan_prod_impl(T* data, dtl_size_t size) {
    for (dtl_size_t i = 1; i < size; ++i) {
        data[i] *= data[i - 1];
    }
}

template <typename T>
static void inclusive_scan_min_impl(T* data, dtl_size_t size) {
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i - 1] < data[i]) data[i] = data[i - 1];
    }
}

template <typename T>
static void inclusive_scan_max_impl(T* data, dtl_size_t size) {
    for (dtl_size_t i = 1; i < size; ++i) {
        if (data[i - 1] > data[i]) data[i] = data[i - 1];
    }
}

// Exclusive scan template: element i = reduction of elements 0..i-1,
// first element set to identity (0 for sum, 1 for product)
template <typename T>
static void exclusive_scan_sum_impl(T* data, dtl_size_t size) {
    T acc = T{0};
    for (dtl_size_t i = 0; i < size; ++i) {
        T cur = data[i];
        data[i] = acc;
        acc += cur;
    }
}

template <typename T>
static void exclusive_scan_prod_impl(T* data, dtl_size_t size) {
    T acc = T{1};
    for (dtl_size_t i = 0; i < size; ++i) {
        T cur = data[i];
        data[i] = acc;
        acc *= cur;
    }
}

template <typename T>
static void exclusive_scan_min_impl(T* data, dtl_size_t size) {
    T acc = std::numeric_limits<T>::max();
    for (dtl_size_t i = 0; i < size; ++i) {
        T cur = data[i];
        data[i] = acc;
        if (cur < acc) acc = cur;
    }
}

template <typename T>
static void exclusive_scan_max_impl(T* data, dtl_size_t size) {
    T acc = std::numeric_limits<T>::lowest();
    for (dtl_size_t i = 0; i < size; ++i) {
        T cur = data[i];
        data[i] = acc;
        if (cur > acc) acc = cur;
    }
}

extern "C" {

dtl_status dtl_inclusive_scan_vector(dtl_vector_t vec, dtl_reduce_op op) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return DTL_SUCCESS;

    void* data = h->vtable->local_data_mut(h->impl);
    if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const dtl_size_t size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8: {
            auto* p = static_cast<int8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT16: {
            auto* p = static_cast<int16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT32: {
            auto* p = static_cast<int32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT64: {
            auto* p = static_cast<int64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL: {
            auto* p = static_cast<uint8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT16: {
            auto* p = static_cast<uint16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT32: {
            auto* p = static_cast<uint32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT64: {
            auto* p = static_cast<uint64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT32: {
            auto* p = static_cast<float*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT64: {
            auto* p = static_cast<double*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_exclusive_scan_vector(dtl_vector_t vec, dtl_reduce_op op) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return DTL_SUCCESS;

    void* data = h->vtable->local_data_mut(h->impl);
    if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const dtl_size_t size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8: {
            auto* p = static_cast<int8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT16: {
            auto* p = static_cast<int16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT32: {
            auto* p = static_cast<int32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT64: {
            auto* p = static_cast<int64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL: {
            auto* p = static_cast<uint8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT16: {
            auto* p = static_cast<uint16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT32: {
            auto* p = static_cast<uint32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT64: {
            auto* p = static_cast<uint64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT32: {
            auto* p = static_cast<float*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT64: {
            auto* p = static_cast<double*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_inclusive_scan_array(dtl_array_t arr, dtl_reduce_op op) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return DTL_SUCCESS;

    void* data = h->vtable->local_data_mut(h->impl);
    if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const dtl_size_t size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8: {
            auto* p = static_cast<int8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT16: {
            auto* p = static_cast<int16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT32: {
            auto* p = static_cast<int32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT64: {
            auto* p = static_cast<int64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL: {
            auto* p = static_cast<uint8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT16: {
            auto* p = static_cast<uint16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT32: {
            auto* p = static_cast<uint32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT64: {
            auto* p = static_cast<uint64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT32: {
            auto* p = static_cast<float*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT64: {
            auto* p = static_cast<double*>(data);
            switch (op) {
                case DTL_OP_SUM:  inclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: inclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  inclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  inclusive_scan_max_impl(p, size);  break;
                default:          inclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

dtl_status dtl_exclusive_scan_array(dtl_array_t arr, dtl_reduce_op op) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!can_run_callback_algorithm(h->base.options.placement)) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    std::size_t local_size = h->vtable->local_size(h->impl);
    if (local_size == 0) return DTL_SUCCESS;

    void* data = h->vtable->local_data_mut(h->impl);
    if (!data) return apply_error_policy(h, DTL_ERROR_NOT_SUPPORTED);

    const dtl_size_t size = static_cast<dtl_size_t>(local_size);
    switch (h->base.dtype) {
        case DTL_DTYPE_INT8: {
            auto* p = static_cast<int8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT16: {
            auto* p = static_cast<int16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT32: {
            auto* p = static_cast<int32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_INT64: {
            auto* p = static_cast<int64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL: {
            auto* p = static_cast<uint8_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT16: {
            auto* p = static_cast<uint16_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT32: {
            auto* p = static_cast<uint32_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_UINT64: {
            auto* p = static_cast<uint64_t*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT32: {
            auto* p = static_cast<float*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        case DTL_DTYPE_FLOAT64: {
            auto* p = static_cast<double*>(data);
            switch (op) {
                case DTL_OP_SUM:  exclusive_scan_sum_impl(p, size);  break;
                case DTL_OP_PROD: exclusive_scan_prod_impl(p, size); break;
                case DTL_OP_MIN:  exclusive_scan_min_impl(p, size);  break;
                case DTL_OP_MAX:  exclusive_scan_max_impl(p, size);  break;
                default:          exclusive_scan_sum_impl(p, size);  break;
            }
            break;
        }
        default:
            return apply_error_policy(h, DTL_ERROR_INVALID_ARGUMENT);
    }

    return DTL_SUCCESS;
}

// ============================================================================
// Async Algorithm Implementations (Phase 16 — True Async)
// ============================================================================

// Helper: create a pending async request driven by the progress engine.
static dtl_status create_async_request(dtl_request_t* req,
                                        std::function<void()> work) {
    if (!req) return DTL_ERROR_NULL_POINTER;

    dtl_request_s* r = nullptr;
    try {
        r = new dtl_request_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }
    r->magic = dtl_request_s::VALID_MAGIC;
#ifdef DTL_HAS_MPI
    r->is_mpi_request = false;
    r->mpi_request = MPI_REQUEST_NULL;
#endif
    r->state = std::make_shared<dtl_request_s::async_state>();
    auto started = std::make_shared<std::atomic<bool>>(false);

    try {
        r->progress_callback_id = dtl::futures::progress_engine::instance().register_callback(
            [state = r->state, started, work = std::move(work)]() mutable -> bool {
                bool expected = false;
                if (!started->compare_exchange_strong(expected, true,
                                                      std::memory_order_acq_rel)) {
                    return !state->completed.load(std::memory_order_acquire);
                }

                if (state->cancelled.load(std::memory_order_acquire)) {
                    state->completed.store(true, std::memory_order_release);
                    return false;
                }

                try {
                    work();
                } catch (...) {
                    // C async request API has no error channel on request completion.
                    // Mark complete and let operation-level status remain best-effort.
                }

                state->completed.store(true, std::memory_order_release);
                return false;
            });
    } catch (...) {
        delete r;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    *req = r;
    return DTL_SUCCESS;
}

dtl_status dtl_async_for_each_vector(dtl_vector_t vec,
                                      dtl_unary_func func,
                                      void* user_data,
                                      dtl_request_t* req) {
    // Validate eagerly before launching thread
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [vec, func, user_data]() {
        dtl_for_each_vector(vec, func, user_data);
    });
}

dtl_status dtl_async_transform_vector(dtl_vector_t src,
                                       dtl_vector_t dst,
                                       dtl_transform_func func,
                                       void* user_data,
                                       dtl_request_t* req) {
    auto* src_h = get_vector_handle(src);
    auto* dst_h = get_vector_handle(dst);
    if (!src_h || !dst_h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(dst_h, DTL_ERROR_NULL_POINTER);
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [src, dst, func, user_data]() {
        dtl_transform_vector(src, dst, func, user_data);
    });
}

dtl_status dtl_async_reduce_vector(dtl_vector_t vec,
                                    dtl_reduce_op op,
                                    void* result,
                                    dtl_request_t* req) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [vec, op, result]() {
        dtl_reduce_local_vector(vec, op, result);
    });
}

dtl_status dtl_async_sort_vector(dtl_vector_t vec,
                                  dtl_request_t* req) {
    auto* h = get_vector_handle(vec);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [vec]() {
        dtl_sort_vector(vec);
    });
}

dtl_status dtl_async_for_each_array(dtl_array_t arr,
                                     dtl_unary_func func,
                                     void* user_data,
                                     dtl_request_t* req) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!func) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [arr, func, user_data]() {
        dtl_for_each_array(arr, func, user_data);
    });
}

dtl_status dtl_async_reduce_array(dtl_array_t arr,
                                   dtl_reduce_op op,
                                   void* result,
                                   dtl_request_t* req) {
    auto* h = get_array_handle(arr);
    if (!h) return DTL_ERROR_INVALID_ARGUMENT;
    if (!result) return apply_error_policy(h, DTL_ERROR_NULL_POINTER);
    if (!req) return DTL_ERROR_NULL_POINTER;

    return create_async_request(req, [arr, op, result]() {
        dtl_reduce_local_array(arr, op, result);
    });
}

} // extern "C"
