// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_span.cpp
 * @brief DTL C bindings - Distributed span implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_span.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_tensor.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

struct dtl_span_s {
    dtl_dtype dtype;
    unsigned char* local_data;
    dtl_size_t local_size;
    dtl_size_t global_size;
    dtl_rank_t rank;
    dtl_rank_t num_ranks;
    int mutable_data;
    std::uint32_t magic;

    static constexpr std::uint32_t VALID_MAGIC = 0x5FAA77E;
};

namespace {

bool is_valid_dtype(dtl_dtype dtype) noexcept {
    return dtype >= 0 && dtype < DTL_DTYPE_COUNT;
}

bool is_valid_span(const dtl_span_t span) noexcept {
    if (!span || span->magic != dtl_span_s::VALID_MAGIC) {
        return false;
    }
    if (!is_valid_dtype(span->dtype)) {
        return false;
    }
    if (span->num_ranks <= 0) {
        return false;
    }
    if (span->rank < 0 || span->rank >= span->num_ranks) {
        return false;
    }
    if (span->global_size < span->local_size) {
        return false;
    }
    if (span->local_size > 0 && span->local_data == nullptr) {
        return false;
    }
    return true;
}

dtl_status create_span_handle(
    dtl_dtype dtype,
    void* local_data,
    dtl_size_t local_size,
    dtl_size_t global_size,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    int mutable_data,
    dtl_span_t* out_span) {

    if (!out_span) {
        return DTL_ERROR_NULL_POINTER;
    }
    *out_span = nullptr;

    if (!is_valid_dtype(dtype)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (global_size < local_size) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (num_ranks <= 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (rank < 0 || rank >= num_ranks) {
        return DTL_ERROR_INVALID_RANK;
    }
    if (local_size > 0 && local_data == nullptr) {
        return DTL_ERROR_NULL_POINTER;
    }

    auto span = std::unique_ptr<dtl_span_s>(new (std::nothrow) dtl_span_s{});
    if (!span) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    span->dtype = dtype;
    span->local_data = static_cast<unsigned char*>(local_data);
    span->local_size = local_size;
    span->global_size = global_size;
    span->rank = rank;
    span->num_ranks = num_ranks;
    span->mutable_data = mutable_data ? 1 : 0;
    span->magic = dtl_span_s::VALID_MAGIC;

    *out_span = span.release();
    return DTL_SUCCESS;
}

dtl_status create_subspan(
    dtl_span_t span,
    dtl_size_t offset,
    dtl_size_t count,
    dtl_span_t* out_span) {

    if (!out_span) {
        return DTL_ERROR_NULL_POINTER;
    }
    *out_span = nullptr;

    if (!is_valid_span(span)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (offset > span->local_size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }
    if (count == DTL_SPAN_NPOS) {
        count = span->local_size - offset;
    }
    if (count > span->local_size - offset) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    const dtl_size_t elem_size = dtl_dtype_size(span->dtype);
    if (elem_size == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    void* sub_data = nullptr;
    if (count > 0) {
        sub_data = span->local_data + static_cast<std::size_t>(offset * elem_size);
    }

    return create_span_handle(
        span->dtype,
        sub_data,
        count,
        count,
        span->rank,
        span->num_ranks,
        span->mutable_data,
        out_span);
}

}  // namespace

extern "C" {

dtl_status dtl_span_create(
    dtl_dtype dtype,
    void* local_data,
    dtl_size_t local_size,
    dtl_size_t global_size,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    dtl_span_t* span) {
    return create_span_handle(
        dtype, local_data, local_size, global_size, rank, num_ranks, 1, span);
}

dtl_status dtl_span_from_vector(dtl_vector_t vec, dtl_span_t* span) {
    if (!span) {
        return DTL_ERROR_NULL_POINTER;
    }
    *span = nullptr;
    if (!dtl_vector_is_valid(vec)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    void* data = dtl_vector_local_data_mut(vec);
    const dtl_size_t local_size = dtl_vector_local_size(vec);
    if (local_size > 0 && data == nullptr) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    return create_span_handle(
        dtl_vector_dtype(vec),
        data,
        local_size,
        dtl_vector_global_size(vec),
        dtl_vector_rank(vec),
        dtl_vector_num_ranks(vec),
        1,
        span);
}

dtl_status dtl_span_from_array(dtl_array_t arr, dtl_span_t* span) {
    if (!span) {
        return DTL_ERROR_NULL_POINTER;
    }
    *span = nullptr;
    if (!dtl_array_is_valid(arr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    void* data = dtl_array_local_data_mut(arr);
    const dtl_size_t local_size = dtl_array_local_size(arr);
    if (local_size > 0 && data == nullptr) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    return create_span_handle(
        dtl_array_dtype(arr),
        data,
        local_size,
        dtl_array_global_size(arr),
        dtl_array_rank(arr),
        dtl_array_num_ranks(arr),
        1,
        span);
}

dtl_status dtl_span_from_tensor(dtl_tensor_t tensor, dtl_span_t* span) {
    if (!span) {
        return DTL_ERROR_NULL_POINTER;
    }
    *span = nullptr;
    if (!dtl_tensor_is_valid(tensor)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    void* data = dtl_tensor_local_data_mut(tensor);
    const dtl_size_t local_size = dtl_tensor_local_size(tensor);
    if (local_size > 0 && data == nullptr) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    return create_span_handle(
        dtl_tensor_dtype(tensor),
        data,
        local_size,
        dtl_tensor_global_size(tensor),
        dtl_tensor_rank(tensor),
        dtl_tensor_num_ranks(tensor),
        1,
        span);
}

void dtl_span_destroy(dtl_span_t span) {
    if (!span) {
        return;
    }
    span->magic = 0;
    delete span;
}

dtl_size_t dtl_span_size(dtl_span_t span) {
    return is_valid_span(span) ? span->global_size : 0;
}

dtl_size_t dtl_span_local_size(dtl_span_t span) {
    return is_valid_span(span) ? span->local_size : 0;
}

dtl_size_t dtl_span_size_bytes(dtl_span_t span) {
    if (!is_valid_span(span)) {
        return 0;
    }
    const dtl_size_t elem_size = dtl_dtype_size(span->dtype);
    return elem_size == 0 ? 0 : span->local_size * elem_size;
}

int dtl_span_empty(dtl_span_t span) {
    return is_valid_span(span) ? (span->global_size == 0 ? 1 : 0) : 1;
}

dtl_dtype dtl_span_dtype(dtl_span_t span) {
    return is_valid_span(span) ? span->dtype : static_cast<dtl_dtype>(-1);
}

const void* dtl_span_data(dtl_span_t span) {
    return is_valid_span(span) ? span->local_data : nullptr;
}

void* dtl_span_data_mut(dtl_span_t span) {
    if (!is_valid_span(span) || !span->mutable_data) {
        return nullptr;
    }
    return span->local_data;
}

dtl_rank_t dtl_span_rank(dtl_span_t span) {
    return is_valid_span(span) ? span->rank : DTL_NO_RANK;
}

dtl_rank_t dtl_span_num_ranks(dtl_span_t span) {
    return is_valid_span(span) ? span->num_ranks : 0;
}

dtl_status dtl_span_get_local(dtl_span_t span, dtl_size_t local_idx, void* value) {
    if (!is_valid_span(span)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!value) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (local_idx >= span->local_size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    const dtl_size_t elem_size = dtl_dtype_size(span->dtype);
    if (elem_size == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    std::memcpy(
        value,
        span->local_data + static_cast<std::size_t>(local_idx * elem_size),
        static_cast<std::size_t>(elem_size));
    return DTL_SUCCESS;
}

dtl_status dtl_span_set_local(dtl_span_t span, dtl_size_t local_idx, const void* value) {
    if (!is_valid_span(span)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!span->mutable_data) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!value) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (local_idx >= span->local_size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }

    const dtl_size_t elem_size = dtl_dtype_size(span->dtype);
    if (elem_size == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    std::memcpy(
        span->local_data + static_cast<std::size_t>(local_idx * elem_size),
        value,
        static_cast<std::size_t>(elem_size));
    return DTL_SUCCESS;
}

dtl_status dtl_span_first(dtl_span_t span, dtl_size_t count, dtl_span_t* out_span) {
    return create_subspan(span, 0, count, out_span);
}

dtl_status dtl_span_last(dtl_span_t span, dtl_size_t count, dtl_span_t* out_span) {
    if (!is_valid_span(span)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (count > span->local_size) {
        return DTL_ERROR_OUT_OF_BOUNDS;
    }
    return create_subspan(span, span->local_size - count, count, out_span);
}

dtl_status dtl_span_subspan(
    dtl_span_t span,
    dtl_size_t offset,
    dtl_size_t count,
    dtl_span_t* out_span) {
    return create_subspan(span, offset, count, out_span);
}

int dtl_span_is_valid(dtl_span_t span) {
    return is_valid_span(span) ? 1 : 0;
}

}  // extern "C"
