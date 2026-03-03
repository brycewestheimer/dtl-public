// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file dispatch.hpp
/// @brief Execution policy dispatch mechanism for DTL algorithms
/// @details Provides compile-time dispatch based on execution policy.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/policies/execution/cuda_exec.hpp>

#include <algorithm>
#include <execution>
#include <functional>
#include <future>
#include <numeric>
#include <type_traits>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/cuda_algorithms.hpp>
#endif

namespace dtl {

// =============================================================================
// Execution Policy Type Detection
// =============================================================================

/// @brief Check if type is a sequential execution policy
template <typename T>
struct is_seq_policy : std::false_type {};

template <>
struct is_seq_policy<seq> : std::true_type {};

template <typename T>
inline constexpr bool is_seq_policy_v = is_seq_policy<std::decay_t<T>>::value;

/// @brief Check if type is a parallel execution policy
template <typename T>
struct is_par_policy : std::false_type {};

template <>
struct is_par_policy<par> : std::true_type {};

template <unsigned int N>
struct is_par_policy<par_n<N>> : std::true_type {};

template <typename T>
inline constexpr bool is_par_policy_v = is_par_policy<std::decay_t<T>>::value;

/// @brief Check if type is an async execution policy
template <typename T>
struct is_async_policy : std::false_type {};

template <>
struct is_async_policy<async> : std::true_type {};

template <typename T>
inline constexpr bool is_async_policy_v = is_async_policy<std::decay_t<T>>::value;

/// @brief Check if type is a CUDA execution policy
template <typename T>
struct is_cuda_policy : std::false_type {};

template <>
struct is_cuda_policy<cuda_exec> : std::true_type {};

template <typename T>
inline constexpr bool is_cuda_policy_v = is_cuda_policy<std::decay_t<T>>::value;

// =============================================================================
// Execution Dispatcher
// =============================================================================

/// @brief Dispatcher for execution policy-aware algorithm invocation
/// @tparam Policy The execution policy type
/// @details Specializations handle different execution policies to invoke
///          the appropriate underlying implementation (STL sequential,
///          parallel execution, or async dispatch).
template <typename Policy>
struct execution_dispatcher {
    /// @brief Invoke function with sequential semantics (default)
    template <typename Func, typename... Args>
    static auto invoke(Func&& f, Args&&... args) {
        return std::invoke(std::forward<Func>(f), std::forward<Args>(args)...);
    }
};

/// @brief Specialization for sequential execution policy
template <>
struct execution_dispatcher<seq> {
    /// @brief Invoke with sequential execution
    template <typename Func, typename... Args>
    static auto invoke(Func&& f, Args&&... args) {
        return std::invoke(std::forward<Func>(f), std::forward<Args>(args)...);
    }

    /// @brief for_each with sequential execution
    template <typename InputIt, typename UnaryFunc>
    static UnaryFunc for_each(InputIt first, InputIt last, UnaryFunc f) {
        return std::for_each(first, last, std::move(f));
    }

    /// @brief transform with sequential execution
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    static OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
        return std::transform(first, last, d_first, std::move(op));
    }

    /// @brief reduce with sequential execution
    template <typename InputIt, typename T, typename BinaryOp>
    static T reduce(InputIt first, InputIt last, T init, BinaryOp op) {
        return std::accumulate(first, last, std::move(init), std::move(op));
    }

    /// @brief sort with sequential execution
    template <typename RandomIt>
    static void sort(RandomIt first, RandomIt last) {
        std::sort(first, last);
    }

    /// @brief sort with comparator and sequential execution
    template <typename RandomIt, typename Compare>
    static void sort(RandomIt first, RandomIt last, Compare comp) {
        std::sort(first, last, std::move(comp));
    }

    /// @brief fill with sequential execution
    template <typename ForwardIt, typename T>
    static void fill(ForwardIt first, ForwardIt last, const T& value) {
        std::fill(first, last, value);
    }

    /// @brief copy with sequential execution
    template <typename InputIt, typename OutputIt>
    static OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
        return std::copy(first, last, d_first);
    }

    /// @brief count with sequential execution
    template <typename InputIt, typename T>
    static typename std::iterator_traits<InputIt>::difference_type
    count(InputIt first, InputIt last, const T& value) {
        return std::count(first, last, value);
    }

    /// @brief count_if with sequential execution
    template <typename InputIt, typename Pred>
    static typename std::iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Pred pred) {
        return std::count_if(first, last, std::move(pred));
    }

    /// @brief find with sequential execution
    template <typename InputIt, typename T>
    static InputIt find(InputIt first, InputIt last, const T& value) {
        return std::find(first, last, value);
    }

    /// @brief find_if with sequential execution
    template <typename InputIt, typename Pred>
    static InputIt find_if(InputIt first, InputIt last, Pred pred) {
        return std::find_if(first, last, std::move(pred));
    }

    /// @brief all_of with sequential execution
    template <typename InputIt, typename Pred>
    static bool all_of(InputIt first, InputIt last, Pred pred) {
        return std::all_of(first, last, std::move(pred));
    }

    /// @brief any_of with sequential execution
    template <typename InputIt, typename Pred>
    static bool any_of(InputIt first, InputIt last, Pred pred) {
        return std::any_of(first, last, std::move(pred));
    }

    /// @brief none_of with sequential execution
    template <typename InputIt, typename Pred>
    static bool none_of(InputIt first, InputIt last, Pred pred) {
        return std::none_of(first, last, std::move(pred));
    }
};

/// @brief Specialization for parallel execution policy
template <>
struct execution_dispatcher<par> {
    /// @brief Invoke with parallel execution wrapper
    template <typename Func, typename... Args>
    static auto invoke(Func&& f, Args&&... args) {
        return std::invoke(std::forward<Func>(f), std::forward<Args>(args)...);
    }

    /// @brief for_each with parallel execution
    template <typename InputIt, typename UnaryFunc>
    static void for_each(InputIt first, InputIt last, UnaryFunc f) {
        std::for_each(std::execution::par, first, last, std::move(f));
    }

    /// @brief transform with parallel execution
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    static OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
        return std::transform(std::execution::par, first, last, d_first, std::move(op));
    }

    /// @brief reduce with parallel execution
    template <typename InputIt, typename T, typename BinaryOp>
    static T reduce(InputIt first, InputIt last, T init, BinaryOp op) {
        return std::reduce(std::execution::par, first, last, std::move(init), std::move(op));
    }

    /// @brief sort with parallel execution
    template <typename RandomIt>
    static void sort(RandomIt first, RandomIt last) {
        std::sort(std::execution::par, first, last);
    }

    /// @brief sort with comparator and parallel execution
    template <typename RandomIt, typename Compare>
    static void sort(RandomIt first, RandomIt last, Compare comp) {
        std::sort(std::execution::par, first, last, std::move(comp));
    }

    /// @brief fill with parallel execution
    template <typename ForwardIt, typename T>
    static void fill(ForwardIt first, ForwardIt last, const T& value) {
        std::fill(std::execution::par, first, last, value);
    }

    /// @brief copy with parallel execution
    template <typename InputIt, typename OutputIt>
    static OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
        return std::copy(std::execution::par, first, last, d_first);
    }

    /// @brief count with parallel execution
    template <typename InputIt, typename T>
    static typename std::iterator_traits<InputIt>::difference_type
    count(InputIt first, InputIt last, const T& value) {
        return std::count(std::execution::par, first, last, value);
    }

    /// @brief count_if with parallel execution
    template <typename InputIt, typename Pred>
    static typename std::iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Pred pred) {
        return std::count_if(std::execution::par, first, last, std::move(pred));
    }

    /// @brief find with parallel execution
    template <typename InputIt, typename T>
    static InputIt find(InputIt first, InputIt last, const T& value) {
        return std::find(std::execution::par, first, last, value);
    }

    /// @brief find_if with parallel execution
    template <typename InputIt, typename Pred>
    static InputIt find_if(InputIt first, InputIt last, Pred pred) {
        return std::find_if(std::execution::par, first, last, std::move(pred));
    }

    /// @brief all_of with parallel execution
    template <typename InputIt, typename Pred>
    static bool all_of(InputIt first, InputIt last, Pred pred) {
        return std::all_of(std::execution::par, first, last, std::move(pred));
    }

    /// @brief any_of with parallel execution
    template <typename InputIt, typename Pred>
    static bool any_of(InputIt first, InputIt last, Pred pred) {
        return std::any_of(std::execution::par, first, last, std::move(pred));
    }

    /// @brief none_of with parallel execution
    template <typename InputIt, typename Pred>
    static bool none_of(InputIt first, InputIt last, Pred pred) {
        return std::none_of(std::execution::par, first, last, std::move(pred));
    }
};

/// @brief Specialization for async execution policy
/// @details For low-level dispatch, async executes sequentially; the actual
///          asynchrony is handled at the algorithm level via distributed_future.
template <>
struct execution_dispatcher<async> {
    /// @brief Invoke asynchronously, returning a future
    template <typename Func, typename... Args>
    static auto invoke(Func&& f, Args&&... args) {
        return std::async(std::launch::async,
                          std::forward<Func>(f),
                          std::forward<Args>(args)...);
    }

    /// @brief for_each with async dispatch (executes sequentially at dispatch level)
    template <typename InputIt, typename UnaryFunc>
    static UnaryFunc for_each(InputIt first, InputIt last, UnaryFunc f) {
        return std::for_each(first, last, std::move(f));
    }

    /// @brief transform with async dispatch
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    static OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
        return std::transform(first, last, d_first, std::move(op));
    }

    /// @brief reduce with async dispatch
    template <typename InputIt, typename T, typename BinaryOp>
    static T reduce(InputIt first, InputIt last, T init, BinaryOp op) {
        return std::accumulate(first, last, std::move(init), std::move(op));
    }

    /// @brief sort with async dispatch
    template <typename RandomIt>
    static void sort(RandomIt first, RandomIt last) {
        std::sort(first, last);
    }

    /// @brief sort with comparator and async dispatch
    template <typename RandomIt, typename Compare>
    static void sort(RandomIt first, RandomIt last, Compare comp) {
        std::sort(first, last, std::move(comp));
    }

    /// @brief fill with async dispatch
    template <typename ForwardIt, typename T>
    static void fill(ForwardIt first, ForwardIt last, const T& value) {
        std::fill(first, last, value);
    }

    /// @brief copy with async dispatch
    template <typename InputIt, typename OutputIt>
    static OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
        return std::copy(first, last, d_first);
    }

    /// @brief count with async dispatch
    template <typename InputIt, typename T>
    static typename std::iterator_traits<InputIt>::difference_type
    count(InputIt first, InputIt last, const T& value) {
        return std::count(first, last, value);
    }

    /// @brief count_if with async dispatch
    template <typename InputIt, typename Pred>
    static typename std::iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Pred pred) {
        return std::count_if(first, last, std::move(pred));
    }

    /// @brief find with async dispatch
    template <typename InputIt, typename T>
    static InputIt find(InputIt first, InputIt last, const T& value) {
        return std::find(first, last, value);
    }

    /// @brief find_if with async dispatch
    template <typename InputIt, typename Pred>
    static InputIt find_if(InputIt first, InputIt last, Pred pred) {
        return std::find_if(first, last, std::move(pred));
    }

    /// @brief all_of with async dispatch
    template <typename InputIt, typename Pred>
    static bool all_of(InputIt first, InputIt last, Pred pred) {
        return std::all_of(first, last, std::move(pred));
    }

    /// @brief any_of with async dispatch
    template <typename InputIt, typename Pred>
    static bool any_of(InputIt first, InputIt last, Pred pred) {
        return std::any_of(first, last, std::move(pred));
    }

    /// @brief none_of with async dispatch
    template <typename InputIt, typename Pred>
    static bool none_of(InputIt first, InputIt last, Pred pred) {
        return std::none_of(first, last, std::move(pred));
    }
};

/// @brief Specialization for cuda_exec execution policy (GPU execution)
/// @details Delegates to Thrust-based implementations for GPU algorithms.
///          Requires data to be accessible from device (device_only or unified_memory).
template <>
struct execution_dispatcher<cuda_exec> {
    /// @brief Invoke with GPU execution context
    template <typename Func, typename... Args>
    static auto invoke(Func&& f, Args&&... args) {
        // For general invocation, fallback to sequential on host
        // GPU-specific algorithms should use the specialized methods below
        return std::invoke(std::forward<Func>(f), std::forward<Args>(args)...);
    }

#if DTL_ENABLE_CUDA
    /// @brief for_each with GPU execution
    /// @note Functor must be __device__ callable
    template <typename InputIt, typename UnaryFunc>
    static void for_each(InputIt first, InputIt last, UnaryFunc f,
                         cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return;

        // Get raw pointer from iterator
        using value_type = typename std::iterator_traits<InputIt>::value_type;
        value_type* ptr = &(*first);

        cuda::for_each_device(ptr, static_cast<size_type>(n), f, stream);
    }

    /// @brief transform with GPU execution
    /// @note UnaryOp must be __device__ callable
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    static OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op,
                              cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return d_first;

        using input_type = typename std::iterator_traits<InputIt>::value_type;
        using output_type = typename std::iterator_traits<OutputIt>::value_type;
        const input_type* in_ptr = &(*first);
        output_type* out_ptr = &(*d_first);

        cuda::transform_device(in_ptr, out_ptr, static_cast<size_type>(n), op, stream);
        return d_first + n;
    }

    /// @brief reduce with GPU execution
    /// @note BinaryOp must be __device__ callable
    template <typename InputIt, typename T, typename BinaryOp>
    static T reduce(InputIt first, InputIt last, T init, BinaryOp op,
                    cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return init;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return cuda::reduce_device(ptr, static_cast<size_type>(n), init, op, stream);
    }

    /// @brief sort with GPU execution
    template <typename RandomIt>
    static void sort(RandomIt first, RandomIt last, cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 1) return;

        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        value_type* ptr = &(*first);

        cuda::sort_device(ptr, static_cast<size_type>(n), stream);
    }

    /// @brief sort with comparator and GPU execution
    /// @note Compare must be __device__ callable
    template <typename RandomIt, typename Compare>
    static void sort(RandomIt first, RandomIt last, Compare comp,
                     cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 1) return;

        using value_type = typename std::iterator_traits<RandomIt>::value_type;
        value_type* ptr = &(*first);

        cuda::sort_device(ptr, static_cast<size_type>(n), comp, stream);
    }

    /// @brief fill with GPU execution
    template <typename ForwardIt, typename T>
    static void fill(ForwardIt first, ForwardIt last, const T& value,
                     cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return;

        using value_type = typename std::iterator_traits<ForwardIt>::value_type;
        value_type* ptr = &(*first);

        cuda::fill_device(ptr, static_cast<size_type>(n), value, stream);
    }

    /// @brief copy with GPU execution
    template <typename InputIt, typename OutputIt>
    static OutputIt copy(InputIt first, InputIt last, OutputIt d_first,
                         cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return d_first;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* src_ptr = &(*first);
        value_type* dst_ptr = &(*d_first);

        cuda::copy_device(src_ptr, dst_ptr, static_cast<size_type>(n), stream);
        return d_first + n;
    }

    /// @brief count with GPU execution
    template <typename InputIt, typename T>
    static typename std::iterator_traits<InputIt>::difference_type
    count(InputIt first, InputIt last, const T& value, cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return 0;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return static_cast<typename std::iterator_traits<InputIt>::difference_type>(
            cuda::count_device(ptr, static_cast<size_type>(n), value, stream));
    }

    /// @brief count_if with GPU execution
    /// @note Pred must be __device__ callable
    template <typename InputIt, typename Pred>
    static typename std::iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Pred pred, cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return 0;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return static_cast<typename std::iterator_traits<InputIt>::difference_type>(
            cuda::count_if_device(ptr, static_cast<size_type>(n), pred, stream));
    }

    /// @brief all_of with GPU execution
    /// @note Pred must be __device__ callable
    template <typename InputIt, typename Pred>
    static bool all_of(InputIt first, InputIt last, Pred pred,
                       cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return true;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return cuda::all_of_device(ptr, static_cast<size_type>(n), pred, stream);
    }

    /// @brief any_of with GPU execution
    /// @note Pred must be __device__ callable
    template <typename InputIt, typename Pred>
    static bool any_of(InputIt first, InputIt last, Pred pred,
                       cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return false;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return cuda::any_of_device(ptr, static_cast<size_type>(n), pred, stream);
    }

    /// @brief none_of with GPU execution
    /// @note Pred must be __device__ callable
    template <typename InputIt, typename Pred>
    static bool none_of(InputIt first, InputIt last, Pred pred,
                        cudaStream_t stream = 0) {
        auto n = std::distance(first, last);
        if (n <= 0) return true;

        using value_type = typename std::iterator_traits<InputIt>::value_type;
        const value_type* ptr = &(*first);

        return cuda::none_of_device(ptr, static_cast<size_type>(n), pred, stream);
    }

#else
    // Fallback implementations when CUDA is disabled
    template <typename InputIt, typename UnaryFunc>
    static void for_each(InputIt first, InputIt last, UnaryFunc f,
                         void* = nullptr) {
        std::for_each(first, last, std::move(f));
    }

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    static OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op,
                              void* = nullptr) {
        return std::transform(first, last, d_first, std::move(op));
    }

    template <typename InputIt, typename T, typename BinaryOp>
    static T reduce(InputIt first, InputIt last, T init, BinaryOp op,
                    void* = nullptr) {
        return std::accumulate(first, last, std::move(init), std::move(op));
    }

    template <typename RandomIt>
    static void sort(RandomIt first, RandomIt last, void* = nullptr) {
        std::sort(first, last);
    }

    template <typename RandomIt, typename Compare>
    static void sort(RandomIt first, RandomIt last, Compare comp,
                     void* = nullptr) {
        std::sort(first, last, std::move(comp));
    }

    template <typename ForwardIt, typename T>
    static void fill(ForwardIt first, ForwardIt last, const T& value,
                     void* = nullptr) {
        std::fill(first, last, value);
    }

    template <typename InputIt, typename OutputIt>
    static OutputIt copy(InputIt first, InputIt last, OutputIt d_first,
                         void* = nullptr) {
        return std::copy(first, last, d_first);
    }

    template <typename InputIt, typename T>
    static typename std::iterator_traits<InputIt>::difference_type
    count(InputIt first, InputIt last, const T& value, void* = nullptr) {
        return std::count(first, last, value);
    }

    template <typename InputIt, typename Pred>
    static typename std::iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Pred pred, void* = nullptr) {
        return std::count_if(first, last, std::move(pred));
    }

    template <typename InputIt, typename Pred>
    static bool all_of(InputIt first, InputIt last, Pred pred,
                       void* = nullptr) {
        return std::all_of(first, last, std::move(pred));
    }

    template <typename InputIt, typename Pred>
    static bool any_of(InputIt first, InputIt last, Pred pred,
                       void* = nullptr) {
        return std::any_of(first, last, std::move(pred));
    }

    template <typename InputIt, typename Pred>
    static bool none_of(InputIt first, InputIt last, Pred pred,
                        void* = nullptr) {
        return std::none_of(first, last, std::move(pred));
    }
#endif
};

// =============================================================================
// Dispatch Helper Functions
// =============================================================================

/// @brief Dispatch for_each based on execution policy
/// @tparam ExecutionPolicy The execution policy type
/// @tparam InputIt Iterator type
/// @tparam UnaryFunc Function type
template <typename ExecutionPolicy, typename InputIt, typename UnaryFunc>
auto dispatch_for_each(ExecutionPolicy&&, InputIt first, InputIt last, UnaryFunc f) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::for_each(first, last, std::move(f));
}

/// @brief Dispatch transform based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename OutputIt, typename UnaryOp>
auto dispatch_transform(ExecutionPolicy&&, InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::transform(first, last, d_first, std::move(op));
}

/// @brief Dispatch reduce based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename T, typename BinaryOp>
auto dispatch_reduce(ExecutionPolicy&&, InputIt first, InputIt last, T init, BinaryOp op) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::reduce(first, last, std::move(init), std::move(op));
}

/// @brief Dispatch sort based on execution policy
template <typename ExecutionPolicy, typename RandomIt>
void dispatch_sort(ExecutionPolicy&&, RandomIt first, RandomIt last) {
    using Policy = std::decay_t<ExecutionPolicy>;
    execution_dispatcher<Policy>::sort(first, last);
}

/// @brief Dispatch sort with comparator based on execution policy
template <typename ExecutionPolicy, typename RandomIt, typename Compare>
void dispatch_sort(ExecutionPolicy&&, RandomIt first, RandomIt last, Compare comp) {
    using Policy = std::decay_t<ExecutionPolicy>;
    execution_dispatcher<Policy>::sort(first, last, std::move(comp));
}

/// @brief Dispatch fill based on execution policy
template <typename ExecutionPolicy, typename ForwardIt, typename T>
void dispatch_fill(ExecutionPolicy&&, ForwardIt first, ForwardIt last, const T& value) {
    using Policy = std::decay_t<ExecutionPolicy>;
    execution_dispatcher<Policy>::fill(first, last, value);
}

/// @brief Dispatch copy based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename OutputIt>
auto dispatch_copy(ExecutionPolicy&&, InputIt first, InputIt last, OutputIt d_first) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::copy(first, last, d_first);
}

/// @brief Dispatch count based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename T>
auto dispatch_count(ExecutionPolicy&&, InputIt first, InputIt last, const T& value) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::count(first, last, value);
}

/// @brief Dispatch count_if based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename Pred>
auto dispatch_count_if(ExecutionPolicy&&, InputIt first, InputIt last, Pred pred) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::count_if(first, last, std::move(pred));
}

/// @brief Dispatch find based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename T>
auto dispatch_find(ExecutionPolicy&&, InputIt first, InputIt last, const T& value) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::find(first, last, value);
}

/// @brief Dispatch find_if based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename Pred>
auto dispatch_find_if(ExecutionPolicy&&, InputIt first, InputIt last, Pred pred) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::find_if(first, last, std::move(pred));
}

/// @brief Dispatch all_of based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename Pred>
auto dispatch_all_of(ExecutionPolicy&&, InputIt first, InputIt last, Pred pred) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::all_of(first, last, std::move(pred));
}

/// @brief Dispatch any_of based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename Pred>
auto dispatch_any_of(ExecutionPolicy&&, InputIt first, InputIt last, Pred pred) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::any_of(first, last, std::move(pred));
}

/// @brief Dispatch none_of based on execution policy
template <typename ExecutionPolicy, typename InputIt, typename Pred>
auto dispatch_none_of(ExecutionPolicy&&, InputIt first, InputIt last, Pred pred) {
    using Policy = std::decay_t<ExecutionPolicy>;
    return execution_dispatcher<Policy>::none_of(first, last, std::move(pred));
}

}  // namespace dtl
