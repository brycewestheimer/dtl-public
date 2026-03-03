// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_algorithms.hpp
/// @brief GPU-accelerated algorithm implementations using Thrust
/// @details Provides Thrust-based implementations of DTL algorithms for
///          GPU execution. These wrap Thrust algorithms with DTL-compatible
///          interfaces for use with cuda_exec execution policy.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
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
namespace cuda {

// ============================================================================
// CUDA Algorithm Execution Context
// ============================================================================

/// @brief Configuration for CUDA algorithm execution
struct cuda_algorithm_config {
    /// @brief CUDA stream for asynchronous execution
    cudaStream_t stream = 0;

    /// @brief Whether to synchronize after algorithm completion
    bool synchronize_after = false;
};

// ============================================================================
// Thrust-Based Algorithm Implementations
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief GPU for_each using Thrust
/// @tparam T Element type
/// @tparam UnaryFunc Functor type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param f Unary function to apply
/// @param stream CUDA stream for execution
/// @note UnaryFunc must be callable from device code (__device__ or __host__ __device__)
template <typename T, typename UnaryFunc>
void for_each_device(T* data, size_type n, UnaryFunc f, cudaStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::for_each(thrust::cuda::par.on(stream), begin, end, f);
}

/// @brief GPU transform using Thrust
/// @tparam InputT Input element type
/// @tparam OutputT Output element type
/// @tparam UnaryOp Transform operation type (must be __device__ callable)
/// @param in Device pointer to input data
/// @param out Device pointer to output data
/// @param n Number of elements
/// @param op Unary transform operation
/// @param stream CUDA stream for execution
template <typename InputT, typename OutputT, typename UnaryOp>
void transform_device(const InputT* in, OutputT* out, size_type n, UnaryOp op,
                      cudaStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const InputT> in_begin(in);
    thrust::device_ptr<const InputT> in_end = in_begin + n;
    thrust::device_ptr<OutputT> out_begin(out);

    thrust::transform(thrust::cuda::par.on(stream), in_begin, in_end, out_begin, op);
}

/// @brief GPU binary transform using Thrust
/// @tparam InputT1 First input element type
/// @tparam InputT2 Second input element type
/// @tparam OutputT Output element type
/// @tparam BinaryOp Transform operation type (must be __device__ callable)
/// @param in1 Device pointer to first input data
/// @param in2 Device pointer to second input data
/// @param out Device pointer to output data
/// @param n Number of elements
/// @param op Binary transform operation
/// @param stream CUDA stream for execution
template <typename InputT1, typename InputT2, typename OutputT, typename BinaryOp>
void transform_device(const InputT1* in1, const InputT2* in2, OutputT* out,
                      size_type n, BinaryOp op, cudaStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const InputT1> in1_begin(in1);
    thrust::device_ptr<const InputT1> in1_end = in1_begin + n;
    thrust::device_ptr<const InputT2> in2_begin(in2);
    thrust::device_ptr<OutputT> out_begin(out);

    thrust::transform(thrust::cuda::par.on(stream), in1_begin, in1_end,
                      in2_begin, out_begin, op);
}

/// @brief GPU reduce using Thrust
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param init Initial value
/// @param op Binary reduction operation
/// @param stream CUDA stream for execution
/// @return Reduction result
template <typename T, typename BinaryOp>
T reduce_device(const T* data, size_type n, T init, BinaryOp op,
                cudaStream_t stream = 0) {
    if (n == 0) return init;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::reduce(thrust::cuda::par.on(stream), begin, end, init, op);
}

/// @brief GPU sum reduction using Thrust
/// @tparam T Value type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param stream CUDA stream for execution
/// @return Sum of all elements
template <typename T>
T reduce_sum_device(const T* data, size_type n, cudaStream_t stream = 0) {
    return reduce_device(data, n, T{0}, thrust::plus<T>{}, stream);
}

/// @brief GPU min reduction using Thrust
/// @tparam T Value type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param stream CUDA stream for execution
/// @return Minimum element
template <typename T>
T reduce_min_device(const T* data, size_type n, cudaStream_t stream = 0) {
    if (n == 0) return T{};

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::min_element(thrust::cuda::par.on(stream), begin, end);
    return *result;
}

/// @brief GPU max reduction using Thrust
/// @tparam T Value type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param stream CUDA stream for execution
/// @return Maximum element
template <typename T>
T reduce_max_device(const T* data, size_type n, cudaStream_t stream = 0) {
    if (n == 0) return T{};

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::max_element(thrust::cuda::par.on(stream), begin, end);
    return *result;
}

/// @brief GPU sort using Thrust
/// @tparam T Element type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param stream CUDA stream for execution
template <typename T>
void sort_device(T* data, size_type n, cudaStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::sort(thrust::cuda::par.on(stream), begin, end);
}

/// @brief GPU sort with comparator using Thrust
/// @tparam T Element type
/// @tparam Compare Comparator type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param comp Comparison functor
/// @param stream CUDA stream for execution
template <typename T, typename Compare>
void sort_device(T* data, size_type n, Compare comp, cudaStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::sort(thrust::cuda::par.on(stream), begin, end, comp);
}

/// @brief GPU stable sort using Thrust
/// @tparam T Element type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param stream CUDA stream for execution
template <typename T>
void stable_sort_device(T* data, size_type n, cudaStream_t stream = 0) {
    if (n <= 1) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::stable_sort(thrust::cuda::par.on(stream), begin, end);
}

/// @brief GPU fill using Thrust
/// @tparam T Element type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param value Value to fill with
/// @param stream CUDA stream for execution
template <typename T>
void fill_device(T* data, size_type n, const T& value, cudaStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<T> begin(data);
    thrust::device_ptr<T> end = begin + n;

    thrust::fill(thrust::cuda::par.on(stream), begin, end, value);
}

/// @brief GPU copy using Thrust
/// @tparam T Element type
/// @param src Device pointer to source data
/// @param dst Device pointer to destination data
/// @param n Number of elements
/// @param stream CUDA stream for execution
template <typename T>
void copy_device(const T* src, T* dst, size_type n, cudaStream_t stream = 0) {
    if (n == 0) return;

    thrust::device_ptr<const T> src_begin(src);
    thrust::device_ptr<const T> src_end = src_begin + n;
    thrust::device_ptr<T> dst_begin(dst);

    thrust::copy(thrust::cuda::par.on(stream), src_begin, src_end, dst_begin);
}

/// @brief GPU count using Thrust
/// @tparam T Element type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param value Value to count
/// @param stream CUDA stream for execution
/// @return Number of elements equal to value
template <typename T>
size_type count_device(const T* data, size_type n, const T& value,
                       cudaStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return static_cast<size_type>(
        thrust::count(thrust::cuda::par.on(stream), begin, end, value));
}

/// @brief GPU count_if using Thrust
/// @tparam T Element type
/// @tparam Pred Predicate type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param pred Predicate function
/// @param stream CUDA stream for execution
/// @return Number of elements satisfying the predicate
template <typename T, typename Pred>
size_type count_if_device(const T* data, size_type n, Pred pred,
                          cudaStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return static_cast<size_type>(
        thrust::count_if(thrust::cuda::par.on(stream), begin, end, pred));
}

/// @brief GPU find using Thrust
/// @tparam T Element type
/// @param data Device pointer to data
/// @param n Number of elements
/// @param value Value to find
/// @param stream CUDA stream for execution
/// @return Index of first matching element, or n if not found
template <typename T>
size_type find_device(const T* data, size_type n, const T& value,
                      cudaStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::find(thrust::cuda::par.on(stream), begin, end, value);
    return static_cast<size_type>(result - begin);
}

/// @brief GPU find_if using Thrust
/// @tparam T Element type
/// @tparam Pred Predicate type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param pred Predicate function
/// @param stream CUDA stream for execution
/// @return Index of first element satisfying predicate, or n if not found
template <typename T, typename Pred>
size_type find_if_device(const T* data, size_type n, Pred pred,
                         cudaStream_t stream = 0) {
    if (n == 0) return 0;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    auto result = thrust::find_if(thrust::cuda::par.on(stream), begin, end, pred);
    return static_cast<size_type>(result - begin);
}

/// @brief GPU all_of using Thrust
/// @tparam T Element type
/// @tparam Pred Predicate type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param pred Predicate function
/// @param stream CUDA stream for execution
/// @return true if all elements satisfy predicate
template <typename T, typename Pred>
bool all_of_device(const T* data, size_type n, Pred pred,
                   cudaStream_t stream = 0) {
    if (n == 0) return true;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::all_of(thrust::cuda::par.on(stream), begin, end, pred);
}

/// @brief GPU any_of using Thrust
/// @tparam T Element type
/// @tparam Pred Predicate type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param pred Predicate function
/// @param stream CUDA stream for execution
/// @return true if any element satisfies predicate
template <typename T, typename Pred>
bool any_of_device(const T* data, size_type n, Pred pred,
                   cudaStream_t stream = 0) {
    if (n == 0) return false;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::any_of(thrust::cuda::par.on(stream), begin, end, pred);
}

/// @brief GPU none_of using Thrust
/// @tparam T Element type
/// @tparam Pred Predicate type (must be __device__ callable)
/// @param data Device pointer to data
/// @param n Number of elements
/// @param pred Predicate function
/// @param stream CUDA stream for execution
/// @return true if no element satisfies predicate
template <typename T, typename Pred>
bool none_of_device(const T* data, size_type n, Pred pred,
                    cudaStream_t stream = 0) {
    if (n == 0) return true;

    thrust::device_ptr<const T> begin(data);
    thrust::device_ptr<const T> end = begin + n;

    return thrust::none_of(thrust::cuda::par.on(stream), begin, end, pred);
}

// ============================================================================
// Synchronization Helpers
// ============================================================================

/// @brief Synchronize the specified CUDA stream
/// @param stream CUDA stream to synchronize
/// @return Success or error result
inline result<void> synchronize_stream(cudaStream_t stream = 0) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return make_error<void>(status_code::backend_error,
                                "cudaStreamSynchronize failed");
    }
    return {};
}

/// @brief Synchronize all CUDA devices
/// @return Success or error result
inline result<void> synchronize_device() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return make_error<void>(status_code::backend_error,
                                "cudaDeviceSynchronize failed");
    }
    return {};
}

#else // !DTL_ENABLE_CUDA

// Stub implementations when CUDA is not enabled

template <typename T, typename UnaryFunc>
void for_each_device(T*, size_type, UnaryFunc, void* = nullptr) {
    // No-op when CUDA is disabled
}

template <typename InputT, typename OutputT, typename UnaryOp>
void transform_device(const InputT*, OutputT*, size_type, UnaryOp, void* = nullptr) {
    // No-op when CUDA is disabled
}

template <typename T, typename BinaryOp>
T reduce_device(const T*, size_type, T init, BinaryOp, void* = nullptr) {
    return init;
}

template <typename T>
T reduce_sum_device(const T*, size_type, void* = nullptr) {
    return T{0};
}

template <typename T>
void sort_device(T*, size_type, void* = nullptr) {
    // No-op when CUDA is disabled
}

template <typename T>
void fill_device(T*, size_type, const T&, void* = nullptr) {
    // No-op when CUDA is disabled
}

template <typename T>
void copy_device(const T*, T*, size_type, void* = nullptr) {
    // No-op when CUDA is disabled
}

inline result<void> synchronize_stream(void* = nullptr) {
    return {};
}

inline result<void> synchronize_device() {
    return {};
}

#endif // DTL_ENABLE_CUDA

}  // namespace cuda
}  // namespace dtl
