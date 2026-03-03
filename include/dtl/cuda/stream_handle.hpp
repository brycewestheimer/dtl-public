// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stream_handle.hpp
/// @brief Native CUDA stream handle wrapper
/// @details Provides a type-safe wrapper around cudaStream_t for use with
///          DTL execution policies and GPU algorithms.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {
namespace cuda {

#if DTL_ENABLE_CUDA

/// @brief RAII wrapper for CUDA stream
/// @details Wraps a cudaStream_t with type-safe accessors and optional
///          ownership semantics. Can either own a stream (destroying it
///          on destruction) or wrap an existing stream.
///
/// @par Usage
/// @code
/// // Create owned stream
/// dtl::cuda::stream_handle stream(true);  // Creates and owns stream
///
/// // Wrap existing stream
/// cudaStream_t raw_stream;
/// cudaStreamCreate(&raw_stream);
/// dtl::cuda::stream_handle wrapper(raw_stream, false);  // Does not own
/// @endcode
class stream_handle {
public:
    /// @brief Default constructor (default stream, no ownership)
    stream_handle() noexcept
        : stream_(0)  // Default stream
        , owns_(false) {}

    /// @brief Construct from raw stream
    /// @param stream The CUDA stream to wrap
    /// @param owns Whether this wrapper owns the stream (default: false)
    explicit stream_handle(cudaStream_t stream, bool owns = false) noexcept
        : stream_(stream)
        , owns_(owns) {}

    /// @brief Construct and optionally create a new stream
    /// @param create_stream If true, creates a new stream and owns it
    explicit stream_handle(bool create_stream)
        : stream_(0)
        , owns_(false) {
        if (create_stream) {
            cudaError_t err = cudaStreamCreate(&stream_);
            owns_ = (err == cudaSuccess);
        }
    }

    /// @brief Destructor - destroys stream if owned
    ~stream_handle() noexcept {
        if (owns_ && stream_ != 0) {
            cudaStreamDestroy(stream_);
        }
    }

    // Move-only semantics
    stream_handle(const stream_handle&) = delete;
    stream_handle& operator=(const stream_handle&) = delete;

    /// @brief Move constructor
    stream_handle(stream_handle&& other) noexcept
        : stream_(other.stream_)
        , owns_(other.owns_) {
        other.stream_ = 0;
        other.owns_ = false;
    }

    /// @brief Move assignment
    stream_handle& operator=(stream_handle&& other) noexcept {
        if (this != &other) {
            if (owns_ && stream_ != 0) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            owns_ = other.owns_;
            other.stream_ = 0;
            other.owns_ = false;
        }
        return *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// @brief Get the native CUDA stream handle
    [[nodiscard]] cudaStream_t native_handle() const noexcept {
        return stream_;
    }

    /// @brief Get the native handle (alias for Thrust compatibility)
    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

    /// @brief Implicit conversion to cudaStream_t
    [[nodiscard]] operator cudaStream_t() const noexcept {
        return stream_;
    }

    /// @brief Check if this is the default stream
    [[nodiscard]] bool is_default() const noexcept {
        return stream_ == 0;
    }

    /// @brief Check if this wrapper owns the stream
    [[nodiscard]] bool owns() const noexcept {
        return owns_;
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// @brief Synchronize the stream (wait for all work to complete)
    /// @return true on success, false on failure
    [[nodiscard]] bool synchronize() const {
        return cudaStreamSynchronize(stream_) == cudaSuccess;
    }

    /// @brief Query if stream has completed all work
    /// @return true if complete, false if work pending or error
    [[nodiscard]] bool query() const {
        cudaError_t err = cudaStreamQuery(stream_);
        return err == cudaSuccess;
    }

    /// @brief Wait on an event
    /// @param event Event to wait on
    /// @return true on success
    bool wait_event(cudaEvent_t event) {
        return cudaStreamWaitEvent(stream_, event, 0) == cudaSuccess;
    }

    // ========================================================================
    // Factory Functions
    // ========================================================================

    /// @brief Create a new stream with flags
    /// @param flags cudaStreamFlags (e.g., cudaStreamNonBlocking)
    /// @return stream_handle owning the new stream
    [[nodiscard]] static stream_handle create(unsigned int flags = cudaStreamDefault) {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err == cudaSuccess) {
            return stream_handle(stream, true);
        }
        return stream_handle();  // Default stream on failure
    }

    /// @brief Create a non-blocking stream
    [[nodiscard]] static stream_handle create_non_blocking() {
        return create(cudaStreamNonBlocking);
    }

    /// @brief Get a handle wrapping the default stream
    [[nodiscard]] static stream_handle default_stream() noexcept {
        return stream_handle(0, false);
    }

private:
    cudaStream_t stream_;
    bool owns_;
};

#else  // !DTL_ENABLE_CUDA

/// @brief Stub stream handle when CUDA is disabled
class stream_handle {
public:
    stream_handle() noexcept = default;
    explicit stream_handle(bool /*create_stream*/) noexcept {}

    [[nodiscard]] void* native_handle() const noexcept { return nullptr; }
    [[nodiscard]] void* get() const noexcept { return nullptr; }
    [[nodiscard]] bool is_default() const noexcept { return true; }
    [[nodiscard]] bool synchronize() const { return true; }
    [[nodiscard]] bool query() const { return true; }

    [[nodiscard]] static stream_handle create(unsigned int /*flags*/ = 0) {
        return stream_handle{};
    }
    [[nodiscard]] static stream_handle default_stream() noexcept {
        return stream_handle{};
    }
};

#endif  // DTL_ENABLE_CUDA

}  // namespace cuda
}  // namespace dtl
