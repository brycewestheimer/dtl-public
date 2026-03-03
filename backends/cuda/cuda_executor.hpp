// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_executor.hpp
/// @brief CUDA executor with stream management
/// @details Provides execution abstraction for CUDA kernel launches
///          with stream-based asynchronous execution. Integrates with
///          the progress engine for event-based async completion tracking.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/executor.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace dtl {
namespace cuda {

// Forward declarations
class cuda_event;

// ============================================================================
// CUDA Future Factory
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Create a promise/future pair for CUDA async operations
/// @tparam T Value type for the future
/// @return Pair of (future, promise) for async completion
template <typename T>
[[nodiscard]] std::pair<futures::distributed_future<T>, futures::distributed_promise<T>>
make_cuda_future() {
    futures::distributed_promise<T> promise;
    auto future = promise.get_future();
    return {std::move(future), std::move(promise)};
}

/// @brief Specialization for void type
template <>
[[nodiscard]] inline std::pair<futures::distributed_future<void>, futures::distributed_promise<void>>
make_cuda_future<void>() {
    futures::distributed_promise<void> promise;
    auto future = promise.get_future();
    return {std::move(future), std::move(promise)};
}

/// @brief Dispatch a GPU operation asynchronously with event-based completion
/// @details Executes the kernel on the specified stream, records a CUDA event,
///          and returns a future that resolves when the event completes.
///          The progress engine polls the event via cudaEventQuery().
/// @tparam F Kernel launcher callable type (takes cudaStream_t parameter)
/// @param stream CUDA stream for execution ordering
/// @param kernel Function that launches GPU work on the given stream
/// @return Future that resolves when the GPU operation completes
template <typename F>
[[nodiscard]] futures::distributed_future<void> dispatch_gpu_async(cudaStream_t stream, F&& kernel) {
    // Launch the kernel on the stream
    kernel(stream);

    // Create and record a CUDA event
    cudaEvent_t event;
    cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        // Event creation failed - return a failed future
        return futures::make_failed_distributed_future<void>(
            status(status_code::backend_error, no_rank, "cudaEventCreate failed"));
    }

    err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(event);
        return futures::make_failed_distributed_future<void>(
            status(status_code::backend_error, no_rank, "cudaEventRecord failed"));
    }

    // Create the future/promise pair
    auto [future, promise] = make_cuda_future<void>();

    // Register the event with the progress engine
    // When the event completes, the callback sets the promise value and destroys the event
    futures::progress_engine::instance().register_cuda_event(
        event,
        [p = std::move(promise), event]() mutable {
            p.set_value();
            cudaEventDestroy(event);
        });

    return future;
}

/// @brief Dispatch a GPU operation asynchronously and retrieve a result
/// @details Executes the kernel on the specified stream, records a CUDA event,
///          and returns a future containing the computed result.
///          The result is copied from device to host after the kernel completes.
/// @tparam T Result type
/// @tparam F Kernel launcher callable type
/// @param stream CUDA stream for execution ordering
/// @param kernel Function that launches GPU work and returns a pointer to the result
/// @param result_ptr Pointer to device memory containing the result
/// @return Future containing the result value
template <typename T, typename F>
[[nodiscard]] futures::distributed_future<T> dispatch_gpu_async_result(
    cudaStream_t stream, F&& kernel, T* result_ptr) {
    // Launch the kernel on the stream
    kernel(stream);

    // Create and record a CUDA event
    cudaEvent_t event;
    cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return futures::make_failed_distributed_future<T>(
            status(status_code::backend_error, no_rank, "cudaEventCreate failed"));
    }

    err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(event);
        return futures::make_failed_distributed_future<T>(
            status(status_code::backend_error, no_rank, "cudaEventRecord failed"));
    }

    // Create the future/promise pair
    futures::distributed_promise<T> promise;
    auto future = promise.get_future();

    // Register the event with the progress engine
    // When the event completes, copy the result and set the promise
    futures::progress_engine::instance().register_cuda_event(
        event,
        [p = std::move(promise), event, result_ptr, stream]() mutable {
            // Copy result from device to host (synchronous after event completion)
            T result;
            cudaError_t copy_err = cudaMemcpyAsync(
                &result, result_ptr, sizeof(T), cudaMemcpyDeviceToHost, stream);
            if (copy_err != cudaSuccess) {
                p.set_error(status(status_code::backend_error, no_rank,
                                   "cudaMemcpyAsync failed for result"));
            } else {
                // Need to sync for the memcpy since we used async
                cudaStreamSynchronize(stream);
                p.set_value(std::move(result));
            }
            cudaEventDestroy(event);
        });

    return future;
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// CUDA Stream
// ============================================================================

/// @brief Stream creation flags
enum class stream_flags : unsigned int {
    /// @brief Default stream behavior
    default_stream = 0,

    /// @brief Non-blocking stream
    non_blocking = 1
};

/// @brief RAII wrapper for CUDA streams
class cuda_stream {
public:
    /// @brief Create default stream wrapper (uses stream 0)
    cuda_stream() = default;

    /// @brief Create a new stream
    /// @param flags Stream creation flags
    explicit cuda_stream(stream_flags flags) {
#if DTL_ENABLE_CUDA
        unsigned int cuda_flags = 0;
        if (flags == stream_flags::non_blocking) {
            cuda_flags = cudaStreamNonBlocking;
        }

        cudaError_t err = cudaStreamCreateWithFlags(&stream_, cuda_flags);
        if (err != cudaSuccess) {
            stream_ = nullptr;
        }
        owns_stream_ = true;
#else
        (void)flags;
#endif
    }

#if DTL_ENABLE_CUDA
    /// @brief Wrap an existing stream
    /// @param stream CUDA stream handle
    /// @param owns Whether to take ownership
    explicit cuda_stream(cudaStream_t stream, bool owns = false)
        : stream_(stream)
        , owns_stream_(owns) {}
#endif

    /// @brief Destructor
    ~cuda_stream() {
#if DTL_ENABLE_CUDA
        if (owns_stream_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
#endif
    }

    // Non-copyable
    cuda_stream(const cuda_stream&) = delete;
    cuda_stream& operator=(const cuda_stream&) = delete;

    // Movable
    cuda_stream(cuda_stream&& other) noexcept
#if DTL_ENABLE_CUDA
        : stream_(other.stream_)
        , owns_stream_(other.owns_stream_)
#endif
    {
#if DTL_ENABLE_CUDA
        other.stream_ = nullptr;
        other.owns_stream_ = false;
#endif
    }

    cuda_stream& operator=(cuda_stream&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_CUDA
            if (owns_stream_ && stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            owns_stream_ = other.owns_stream_;
            other.stream_ = nullptr;
            other.owns_stream_ = false;
#endif
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Stream Operations
    // ------------------------------------------------------------------------

    /// @brief Check if stream is valid
    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_CUDA
        return stream_ != nullptr || !owns_stream_;  // nullptr is valid default stream
#else
        return false;
#endif
    }

    /// @brief Synchronize the stream (wait for all operations)
    void synchronize() {
#if DTL_ENABLE_CUDA
        cudaStreamSynchronize(stream_);
#endif
    }

    /// @brief Query if stream is empty (all operations complete)
    [[nodiscard]] bool query() const noexcept {
#if DTL_ENABLE_CUDA
        cudaError_t err = cudaStreamQuery(stream_);
        return (err == cudaSuccess);
#else
        return true;
#endif
    }

    /// @brief Make stream wait for an event
    /// @param event Event to wait for
    /// @return Success or error
    result<void> wait_event(const cuda_event& event);

#if DTL_ENABLE_CUDA
    /// @brief Get the native CUDA stream handle
    [[nodiscard]] cudaStream_t native_handle() const noexcept { return stream_; }
#endif

private:
#if DTL_ENABLE_CUDA
    cudaStream_t stream_ = nullptr;
#endif
    bool owns_stream_ = false;
};

// ============================================================================
// CUDA Executor
// ============================================================================

/// @brief CUDA executor for GPU kernel execution
/// @details Provides stream-based asynchronous execution of GPU operations.
///          Satisfies the Executor concept.
class cuda_executor {
public:
    /// @brief Default constructor (uses default stream)
    cuda_executor() = default;

    /// @brief Construct with a specific stream
    /// @param stream CUDA stream to use
    explicit cuda_executor(cuda_stream stream)
        : stream_(std::move(stream)) {}

    /// @brief Construct with stream flags
    /// @param flags Stream creation flags
    explicit cuda_executor(stream_flags flags)
        : stream_(flags) {}

    /// @brief Destructor
    ~cuda_executor() = default;

    // Non-copyable
    cuda_executor(const cuda_executor&) = delete;
    cuda_executor& operator=(const cuda_executor&) = delete;

    // Movable
    cuda_executor(cuda_executor&&) = default;
    cuda_executor& operator=(cuda_executor&&) = default;

    // ------------------------------------------------------------------------
    // Executor Concept Interface
    // ------------------------------------------------------------------------

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cuda";
    }

    /// @brief Execute a host function on the stream (concept-compliant)
    /// @tparam F Callable type
    /// @param func Function to execute
    /// @note Function runs on host but is ordered with GPU operations
    template <typename F>
    void execute(F&& func) {
#if DTL_ENABLE_CUDA
        // Use CUDA host function callback
        auto* task = new std::function<void()>(std::forward<F>(func));

        cudaError_t err = cudaLaunchHostFunc(
            stream_.native_handle(),
            [](void* data) {
                auto* fn = static_cast<std::function<void()>*>(data);
                (*fn)();
                delete fn;
            },
            task);

        if (err != cudaSuccess) {
            delete task;
            std::fprintf(stderr, "[DTL WARNING] cudaLaunchHostFunc failed (%d: %s), "
                         "falling back to synchronous execution\n",
                         static_cast<int>(err), cudaGetErrorString(err));
            std::forward<F>(func)();
        }
#else
        std::forward<F>(func)();
#endif
    }

    /// @brief Try to execute a host function on the stream, returning an error on failure
    /// @details Unlike execute(), this method does NOT silently fall back to synchronous
    ///          execution on CUDA error. Instead, it returns a result<void> indicating
    ///          success or failure, allowing callers to handle the error explicitly.
    /// @tparam F Callable type
    /// @param func Function to execute
    /// @return result<void> indicating success, or an error if cudaLaunchHostFunc failed
    template <typename F>
    result<void> try_execute(F&& func) {
#if DTL_ENABLE_CUDA
        auto* task = new std::function<void()>(std::forward<F>(func));

        cudaError_t err = cudaLaunchHostFunc(
            stream_.native_handle(),
            [](void* data) {
                auto* fn = static_cast<std::function<void()>*>(data);
                (*fn)();
                delete fn;
            },
            task);

        if (err != cudaSuccess) {
            delete task;
            return make_error<void>(status_code::backend_error,
                                   std::string("cudaLaunchHostFunc failed: ") + cudaGetErrorString(err));
        }
        return {};
#else
        std::forward<F>(func)();
        return {};
#endif
    }

    /// @brief Execute synchronously (wait for completion)
    /// @tparam F Callable type
    /// @param func Function to execute
    template <typename F>
    void sync_execute(F&& func) {
        execute(std::forward<F>(func));
        synchronize();
    }

    /// @brief Wait for all submitted work to complete
    void synchronize() {
        stream_.synchronize();
    }

    /// @brief Query if all work is complete
    [[nodiscard]] bool is_idle() const noexcept {
        return stream_.query();
    }

    // ------------------------------------------------------------------------
    // Kernel Launch (CUDA-Specific)
    // ------------------------------------------------------------------------

#if DTL_ENABLE_CUDA
    /// @brief Launch a CUDA kernel
    /// @tparam Kernel Kernel function type
    /// @tparam Args Kernel argument types
    /// @param grid Grid dimensions
    /// @param block Block dimensions
    /// @param shared_mem Shared memory size in bytes
    /// @param kernel Kernel function pointer
    /// @param args Kernel arguments
    template <typename Kernel, typename... Args>
    result<void> launch(dim3 grid, dim3 block, size_type shared_mem,
                        Kernel kernel, Args... args) {
        void* kernel_args[] = {&args...};

        cudaError_t err = cudaLaunchKernel(
            reinterpret_cast<void*>(kernel),
            grid, block,
            kernel_args,
            shared_mem,
            stream_.native_handle());

        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaLaunchKernel failed");
        }
        return {};
    }

    /// @brief Launch a kernel with simple 1D grid
    /// @param num_elements Total number of elements
    /// @param block_size Threads per block
    /// @param kernel Kernel function
    /// @param args Kernel arguments
    template <typename Kernel, typename... Args>
    result<void> launch_1d(size_type num_elements, size_type block_size,
                           Kernel kernel, Args... args) {
        dim3 grid((num_elements + block_size - 1) / block_size);
        dim3 block(block_size);
        return launch(grid, block, 0, kernel, args...);
    }

    // ------------------------------------------------------------------------
    // Async Dispatch with Event-Based Futures
    // ------------------------------------------------------------------------

    /// @brief Execute GPU work asynchronously and return a future
    /// @details Records a CUDA event after the work is submitted and registers
    ///          it with the progress engine. The returned future resolves when
    ///          the GPU work completes (event signals ready).
    /// @tparam F Callable type that takes cudaStream_t parameter
    /// @param kernel Function that launches GPU work on the stream
    /// @return Future that resolves when GPU work completes
    template <typename F>
    [[nodiscard]] futures::distributed_future<void> execute_async(F&& kernel) {
        return dispatch_gpu_async(stream_.native_handle(), std::forward<F>(kernel));
    }

    /// @brief Execute GPU work asynchronously and return a future with result
    /// @details Records a CUDA event after the work is submitted. When the event
    ///          completes, copies the result from device memory and fulfills the future.
    /// @tparam T Result type
    /// @tparam F Callable type that takes cudaStream_t parameter
    /// @param kernel Function that launches GPU work
    /// @param result_ptr Pointer to device memory containing the result
    /// @return Future containing the result value when GPU work completes
    template <typename T, typename F>
    [[nodiscard]] futures::distributed_future<T> execute_async_result(F&& kernel, T* result_ptr) {
        return dispatch_gpu_async_result<T>(
            stream_.native_handle(), std::forward<F>(kernel), result_ptr);
    }

    /// @brief Launch a kernel asynchronously and return a future
    /// @tparam Kernel Kernel function type
    /// @tparam Args Kernel argument types
    /// @param grid Grid dimensions
    /// @param block Block dimensions
    /// @param shared_mem Shared memory size in bytes
    /// @param kernel Kernel function pointer
    /// @param args Kernel arguments
    /// @return Future that resolves when kernel completes
    template <typename Kernel, typename... Args>
    [[nodiscard]] futures::distributed_future<void> launch_async(
        dim3 grid, dim3 block, size_type shared_mem, Kernel kernel, Args... args) {
        return execute_async([=, this](cudaStream_t stream) {
            void* kernel_args[] = {const_cast<std::remove_const_t<Args>*>(&args)...};
            cudaLaunchKernel(
                reinterpret_cast<void*>(kernel),
                grid, block,
                kernel_args,
                shared_mem,
                stream);
        });
    }

    /// @brief Launch a 1D kernel asynchronously and return a future
    /// @param num_elements Total number of elements
    /// @param block_size Threads per block
    /// @param kernel Kernel function
    /// @param args Kernel arguments
    /// @return Future that resolves when kernel completes
    template <typename Kernel, typename... Args>
    [[nodiscard]] futures::distributed_future<void> launch_1d_async(
        size_type num_elements, size_type block_size, Kernel kernel, Args... args) {
        dim3 grid_dim((num_elements + block_size - 1) / block_size);
        dim3 block_dim(block_size);
        return launch_async(grid_dim, block_dim, 0, kernel, args...);
    }
#endif

    // ------------------------------------------------------------------------
    // Stream Access
    // ------------------------------------------------------------------------

    /// @brief Get the underlying stream
    [[nodiscard]] cuda_stream& stream() noexcept { return stream_; }

    /// @brief Get the underlying stream (const)
    [[nodiscard]] const cuda_stream& stream() const noexcept { return stream_; }

#if DTL_ENABLE_CUDA
    /// @brief Get native stream handle
    [[nodiscard]] cudaStream_t native_handle() const noexcept {
        return stream_.native_handle();
    }
#endif

    /// @brief Check if executor is valid
    [[nodiscard]] bool valid() const noexcept { return stream_.valid(); }

private:
    cuda_stream stream_;
};

// ============================================================================
// Multi-Stream Executor
// ============================================================================

/// @brief Executor that manages multiple CUDA streams
class multi_stream_executor {
public:
    /// @brief Construct with number of streams
    /// @param num_streams Number of streams to create
    explicit multi_stream_executor(size_type num_streams = 4) {
        executors_.reserve(num_streams);
        for (size_type i = 0; i < num_streams; ++i) {
            executors_.emplace_back(stream_flags::non_blocking);
        }
    }

    /// @brief Get number of streams
    [[nodiscard]] size_type num_streams() const noexcept {
        return executors_.size();
    }

    /// @brief Get executor for a specific stream
    /// @param idx Stream index
    [[nodiscard]] cuda_executor& operator[](size_type idx) {
        return executors_[idx % executors_.size()];
    }

    /// @brief Synchronize all streams
    void synchronize_all() {
        for (auto& exec : executors_) {
            exec.synchronize();
        }
    }

    /// @brief Query if all streams are idle
    [[nodiscard]] bool all_idle() const noexcept {
        for (const auto& exec : executors_) {
            if (!exec.is_idle()) return false;
        }
        return true;
    }

private:
    std::vector<cuda_executor> executors_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the default CUDA executor (uses default stream)
/// @return Reference to default executor
[[nodiscard]] inline cuda_executor& default_cuda_executor() {
    static cuda_executor executor;
    return executor;
}

/// @brief Create a new CUDA executor with its own stream
/// @return New executor with non-blocking stream
[[nodiscard]] inline std::unique_ptr<cuda_executor> make_cuda_executor() {
    return std::make_unique<cuda_executor>(stream_flags::non_blocking);
}

/// @brief Create a multi-stream executor
/// @param num_streams Number of streams
/// @return Multi-stream executor
[[nodiscard]] inline std::unique_ptr<multi_stream_executor>
make_multi_stream_executor(size_type num_streams = 4) {
    return std::make_unique<multi_stream_executor>(num_streams);
}

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Executor<cuda_executor>, "cuda_executor must satisfy Executor concept");

}  // namespace cuda
}  // namespace dtl
