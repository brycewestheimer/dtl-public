// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hip_executor.hpp
/// @brief HIP executor with stream management for AMD GPUs
/// @details Provides execution abstraction for HIP kernel launches
///          with stream-based asynchronous execution.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/executor.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <functional>
#include <memory>
#include <vector>

namespace dtl {
namespace hip {

// ============================================================================
// HIP Stream
// ============================================================================

/// @brief Stream creation flags
enum class stream_flags : unsigned int {
    /// @brief Default stream behavior
    default_stream = 0,

    /// @brief Non-blocking stream
    non_blocking = 1
};

/// @brief RAII wrapper for HIP streams
class hip_stream {
public:
    /// @brief Create default stream wrapper
    hip_stream() = default;

    /// @brief Create a new stream
    /// @param flags Stream creation flags
    explicit hip_stream(stream_flags flags) {
#if DTL_ENABLE_HIP
        unsigned int hip_flags = 0;
        if (flags == stream_flags::non_blocking) {
            hip_flags = hipStreamNonBlocking;
        }

        hipError_t err = hipStreamCreateWithFlags(&stream_, hip_flags);
        if (err != hipSuccess) {
            stream_ = nullptr;
        }
        owns_stream_ = true;
#else
        (void)flags;
#endif
    }

#if DTL_ENABLE_HIP
    /// @brief Wrap an existing stream
    explicit hip_stream(hipStream_t stream, bool owns = false)
        : stream_(stream)
        , owns_stream_(owns) {}
#endif

    /// @brief Destructor
    ~hip_stream() {
#if DTL_ENABLE_HIP
        if (owns_stream_ && stream_ != nullptr) {
            hipStreamDestroy(stream_);
        }
#endif
    }

    // Non-copyable
    hip_stream(const hip_stream&) = delete;
    hip_stream& operator=(const hip_stream&) = delete;

    // Movable
    hip_stream(hip_stream&& other) noexcept
#if DTL_ENABLE_HIP
        : stream_(other.stream_)
        , owns_stream_(other.owns_stream_)
#endif
    {
#if DTL_ENABLE_HIP
        other.stream_ = nullptr;
        other.owns_stream_ = false;
#endif
    }

    hip_stream& operator=(hip_stream&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_HIP
            if (owns_stream_ && stream_ != nullptr) {
                hipStreamDestroy(stream_);
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
#if DTL_ENABLE_HIP
        return stream_ != nullptr || !owns_stream_;
#else
        return false;
#endif
    }

    /// @brief Synchronize the stream
    result<void> synchronize() {
#if DTL_ENABLE_HIP
        hipError_t err = hipStreamSynchronize(stream_);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipStreamSynchronize failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Query if stream is empty
    [[nodiscard]] bool query() const noexcept {
#if DTL_ENABLE_HIP
        hipError_t err = hipStreamQuery(stream_);
        return (err == hipSuccess);
#else
        return true;
#endif
    }

#if DTL_ENABLE_HIP
    /// @brief Get the native HIP stream handle
    [[nodiscard]] hipStream_t native_handle() const noexcept { return stream_; }
#endif

private:
#if DTL_ENABLE_HIP
    hipStream_t stream_ = nullptr;
#endif
    bool owns_stream_ = false;
};

// ============================================================================
// HIP Executor
// ============================================================================

/// @brief HIP executor for AMD GPU kernel execution
/// @details Provides stream-based asynchronous execution of GPU operations.
class hip_executor {
public:
    /// @brief Default constructor (uses default stream)
    hip_executor() = default;

    /// @brief Construct with a specific stream
    explicit hip_executor(hip_stream stream)
        : stream_(std::move(stream)) {}

    /// @brief Construct with stream flags
    explicit hip_executor(stream_flags flags)
        : stream_(flags) {}

    /// @brief Destructor
    ~hip_executor() = default;

    // Non-copyable
    hip_executor(const hip_executor&) = delete;
    hip_executor& operator=(const hip_executor&) = delete;

    // Movable
    hip_executor(hip_executor&&) = default;
    hip_executor& operator=(hip_executor&&) = default;

    // ------------------------------------------------------------------------
    // Executor Concept Interface
    // ------------------------------------------------------------------------

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "hip";
    }

    /// @brief Execute a host function on the stream (concept-compliant)
    /// @tparam F Callable type
    /// @param func Function to execute
    /// @note Function runs on host but is ordered with GPU operations
    template <typename F>
    void execute(F&& func) {
#if DTL_ENABLE_HIP
        auto* task = new std::function<void()>(std::forward<F>(func));

        hipError_t err = hipLaunchHostFunc(
            stream_.native_handle(),
            [](void* data) {
                auto* fn = static_cast<std::function<void()>*>(data);
                (*fn)();
                delete fn;
            },
            task);

        if (err != hipSuccess) {
            delete task;
            // Fallback to synchronous execution
            std::forward<F>(func)();
        }
#else
        std::forward<F>(func)();
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
#if DTL_ENABLE_HIP
        hipStreamSynchronize(stream_.native_handle());
#endif
    }

    /// @brief Query if all work is complete
    [[nodiscard]] bool is_idle() const noexcept {
        return stream_.query();
    }

    // ------------------------------------------------------------------------
    // Kernel Launch (HIP-Specific)
    // ------------------------------------------------------------------------

#if DTL_ENABLE_HIP
    /// @brief Launch a HIP kernel
    template <typename Kernel, typename... Args>
    result<void> launch(dim3 grid, dim3 block, size_type shared_mem,
                        Kernel kernel, Args... args) {
        hipLaunchKernelGGL(kernel, grid, block, shared_mem,
                           stream_.native_handle(), args...);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "HIP kernel launch failed");
        }
        return {};
    }

    /// @brief Launch with simple 1D grid
    template <typename Kernel, typename... Args>
    result<void> launch_1d(size_type num_elements, size_type block_size,
                           Kernel kernel, Args... args) {
        dim3 grid((num_elements + block_size - 1) / block_size);
        dim3 block(block_size);
        return launch(grid, block, 0, kernel, args...);
    }
#endif

    // ------------------------------------------------------------------------
    // Stream Access
    // ------------------------------------------------------------------------

    [[nodiscard]] hip_stream& stream() noexcept { return stream_; }
    [[nodiscard]] const hip_stream& stream() const noexcept { return stream_; }

#if DTL_ENABLE_HIP
    [[nodiscard]] hipStream_t native_handle() const noexcept {
        return stream_.native_handle();
    }
#endif

    [[nodiscard]] bool valid() const noexcept { return stream_.valid(); }

private:
    hip_stream stream_;
};

// ============================================================================
// Multi-Stream Executor
// ============================================================================

/// @brief Executor managing multiple HIP streams
class multi_stream_executor {
public:
    explicit multi_stream_executor(size_type num_streams = 4) {
        executors_.reserve(num_streams);
        for (size_type i = 0; i < num_streams; ++i) {
            executors_.emplace_back(stream_flags::non_blocking);
        }
    }

    [[nodiscard]] size_type num_streams() const noexcept {
        return executors_.size();
    }

    [[nodiscard]] hip_executor& operator[](size_type idx) {
        return executors_[idx % executors_.size()];
    }

    result<void> synchronize_all() {
        for (auto& exec : executors_) {
            auto result = exec.synchronize();
            if (!result) return result;
        }
        return {};
    }

    [[nodiscard]] bool all_idle() const noexcept {
        for (const auto& exec : executors_) {
            if (!exec.is_idle()) return false;
        }
        return true;
    }

private:
    std::vector<hip_executor> executors_;
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] inline hip_executor& default_hip_executor() {
    static hip_executor executor;
    return executor;
}

[[nodiscard]] inline std::unique_ptr<hip_executor> make_hip_executor() {
    return std::make_unique<hip_executor>(stream_flags::non_blocking);
}

[[nodiscard]] inline std::unique_ptr<multi_stream_executor>
make_multi_stream_executor(size_type num_streams = 4) {
    return std::make_unique<multi_stream_executor>(num_streams);
}

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Executor<hip_executor>, "hip_executor must satisfy Executor concept");

}  // namespace hip
}  // namespace dtl
