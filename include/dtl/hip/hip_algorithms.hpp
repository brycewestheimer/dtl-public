// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hip_algorithms.hpp
/// @brief GPU-accelerated algorithm implementations using rocThrust/Thrust for HIP
/// @details Provides Thrust-based implementations of DTL algorithms for
///          AMD GPU execution via HIP. These wrap Thrust algorithms with
///          DTL-compatible interfaces for use with hip_exec execution policy.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/logical.h>
#endif

#include <functional>
#include <type_traits>

namespace dtl {
namespace hip {

// ============================================================================
// HIP Algorithm Execution Context
// ============================================================================

/// @brief Configuration for HIP algorithm execution
struct hip_algorithm_config {
    /// @brief HIP stream for asynchronous execution
    hipStream_t stream = 0;

    /// @brief Whether to synchronize after algorithm completion
    bool synchronize_after = false;
};

// ============================================================================
// Thrust-Based Algorithm Implementations
// ============================================================================

#if DTL_ENABLE_HIP

/// @brief GPU for_each using Thrust (HIP backend)
template <typename T, typename UnaryFunc>
void for_each_device(T* data, size_type n, UnaryFunc f, hipStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::for_each(thrust::hip::par.on(stream), begin, end, f);
}

/// @brief GPU transform using Thrust (HIP backend)
template <typename InputT, typename OutputT, typename UnaryOp>
void transform_device(const InputT* in, OutputT* out, size_type n, UnaryOp op,
                      hipStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const InputT> in_begin(in);
    thrust::device_ptr<const InputT> in_end = in_begin + n;
    thrust::device_ptr<OutputT> out_begin(out);

    thrust::transform(thrust::hip::par.on(stream), in_begin, in_end, out_begin, op);
}

/// @brief GPU binary transform using Thrust (HIP backend)
template <typename InputT1, typename InputT2, typename OutputT, typename BinaryOp>
void transform_device(const InputT1* in1, const InputT2* in2, OutputT* out,
                      size_type n, BinaryOp op, hipStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const InputT1> in1_begin(in1);
    thrust::device_ptr<const InputT1> in1_end = in1_begin + n;
    thrust::device_ptr<const InputT2> in2_begin(in2);
    thrust::device_ptr<OutputT> out_begin(out);

    thrust::transform(thrust::hip::par.on(stream), in1_begin, in1_end,
                      in2_begin, out_begin, op);
}

/// @brief GPU reduce using Thrust (HIP backend)
template <typename T, typename BinaryOp>
T reduce_device(const T* data, size_type n, T init, BinaryOp op,
                hipStream_t stream = 0) {
    if (n == 0) return init;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::reduce(thrust::hip::par.on(stream), begin, end, init, op);
}

/// @brief GPU sum reduction using Thrust (HIP backend)
template <typename T>
T reduce_sum_device(const T* data, size_type n, hipStream_t stream = 0) {
    return reduce_device(data, n, T{0}, thrust::plus<T>{}, stream);
}

/// @brief GPU min reduction using Thrust (HIP backend)
template <typename T>
T reduce_min_device(const T* data, size_type n, hipStream_t stream = 0) {
    if (n == 0) return T{};

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::min_element(thrust::hip::par.on(stream), begin, end);
    return *result;
}

/// @brief GPU max reduction using Thrust (HIP backend)
template <typename T>
T reduce_max_device(const T* data, size_type n, hipStream_t stream = 0) {
    if (n == 0) return T{};

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::max_element(thrust::hip::par.on(stream), begin, end);
    return *result;
}

/// @brief GPU sort using Thrust (HIP backend)
template <typename T>
void sort_device(T* data, size_type n, hipStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::sort(thrust::hip::par.on(stream), begin, end);
}

/// @brief GPU sort with comparator using Thrust (HIP backend)
template <typename T, typename Compare>
void sort_device(T* data, size_type n, Compare comp, hipStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::sort(thrust::hip::par.on(stream), begin, end, comp);
}

/// @brief GPU stable sort using Thrust (HIP backend)
template <typename T>
void stable_sort_device(T* data, size_type n, hipStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::stable_sort(thrust::hip::par.on(stream), begin, end);
}

/// @brief GPU fill using Thrust (HIP backend)
template <typename T>
void fill_device(T* data, size_type n, const T& value, hipStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::fill(thrust::hip::par.on(stream), begin, end, value);
}

/// @brief GPU copy using Thrust (HIP backend)
template <typename T>
void copy_device(const T* src, T* dst, size_type n, hipStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const T> src_begin(src);
    thrust::device_ptr<const T> src_end = src_begin + n;
    thrust::device_ptr<T> dst_begin(dst);

    thrust::copy(thrust::hip::par.on(stream), src_begin, src_end, dst_begin);
}

/// @brief GPU count using Thrust (HIP backend)
template <typename T>
size_type count_device(const T* data, size_type n, const T& value,
                       hipStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return static_cast<size_type>(
        thrust::count(thrust::hip::par.on(stream), begin, end, value));
}

/// @brief GPU count_if using Thrust (HIP backend)
template <typename T, typename Pred>
size_type count_if_device(const T* data, size_type n, Pred pred,
                          hipStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return static_cast<size_type>(
        thrust::count_if(thrust::hip::par.on(stream), begin, end, pred));
}

/// @brief GPU find using Thrust (HIP backend)
template <typename T>
size_type find_device(const T* data, size_type n, const T& value,
                      hipStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::find(thrust::hip::par.on(stream), begin, end, value);
    return static_cast<size_type>(result - begin);
}

/// @brief GPU find_if using Thrust (HIP backend)
template <typename T, typename Pred>
size_type find_if_device(const T* data, size_type n, Pred pred,
                         hipStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::find_if(thrust::hip::par.on(stream), begin, end, pred);
    return static_cast<size_type>(result - begin);
}

/// @brief GPU all_of using Thrust (HIP backend)
template <typename T, typename Pred>
bool all_of_device(const T* data, size_type n, Pred pred,
                   hipStream_t stream = 0) {
    if (n == 0) return true;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::all_of(thrust::hip::par.on(stream), begin, end, pred);
}

/// @brief GPU any_of using Thrust (HIP backend)
template <typename T, typename Pred>
bool any_of_device(const T* data, size_type n, Pred pred,
                   hipStream_t stream = 0) {
    if (n == 0) return false;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::any_of(thrust::hip::par.on(stream), begin, end, pred);
}

/// @brief GPU none_of using Thrust (HIP backend)
template <typename T, typename Pred>
bool none_of_device(const T* data, size_type n, Pred pred,
                    hipStream_t stream = 0) {
    if (n == 0) return true;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::none_of(thrust::hip::par.on(stream), begin, end, pred);
}

// ============================================================================
// Synchronization Helpers
// ============================================================================

/// @brief Synchronize the specified HIP stream
inline result<void> synchronize_stream(hipStream_t stream = 0) {
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        return make_error<void>(status_code::backend_error,
                                "hipStreamSynchronize failed");
    }
    return {};
}

/// @brief Synchronize all HIP devices
inline result<void> synchronize_device() {
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        return make_error<void>(status_code::backend_error,
                                "hipDeviceSynchronize failed");
    }
    return {};
}

#else // !DTL_ENABLE_HIP

// Stub implementations when HIP is not enabled

template <typename T, typename UnaryFunc>
void for_each_device(T*, size_type, UnaryFunc, void* = nullptr) {}

template <typename InputT, typename OutputT, typename UnaryOp>
void transform_device(const InputT*, OutputT*, size_type, UnaryOp, void* = nullptr) {}

template <typename T, typename BinaryOp>
T reduce_device(const T*, size_type, T init, BinaryOp, void* = nullptr) {
    return init;
}

template <typename T>
T reduce_sum_device(const T*, size_type, void* = nullptr) {
    return T{0};
}

template <typename T>
T reduce_min_device(const T*, size_type, void* = nullptr) {
    return T{};
}

template <typename T>
T reduce_max_device(const T*, size_type, void* = nullptr) {
    return T{};
}

template <typename T>
void sort_device(T*, size_type, void* = nullptr) {}

template <typename T>
void stable_sort_device(T*, size_type, void* = nullptr) {}

template <typename T>
void fill_device(T*, size_type, const T&, void* = nullptr) {}

template <typename T>
void copy_device(const T*, T*, size_type, void* = nullptr) {}

template <typename T>
size_type count_device(const T*, size_type, const T&, void* = nullptr) {
    return 0;
}

template <typename T, typename Pred>
size_type count_if_device(const T*, size_type, Pred, void* = nullptr) {
    return 0;
}

template <typename T>
size_type find_device(const T*, size_type n, const T&, void* = nullptr) {
    return n;
}

template <typename T, typename Pred>
size_type find_if_device(const T*, size_type n, Pred, void* = nullptr) {
    return n;
}

template <typename T, typename Pred>
bool all_of_device(const T*, size_type, Pred, void* = nullptr) {
    return true;
}

template <typename T, typename Pred>
bool any_of_device(const T*, size_type, Pred, void* = nullptr) {
    return false;
}

template <typename T, typename Pred>
bool none_of_device(const T*, size_type, Pred, void* = nullptr) {
    return true;
}

inline result<void> synchronize_stream(void* = nullptr) {
    return {};
}

inline result<void> synchronize_device() {
    return {};
}

#endif // DTL_ENABLE_HIP

}  // namespace hip
}  // namespace dtl
