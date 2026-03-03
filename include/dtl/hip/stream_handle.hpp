// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stream_handle.hpp
/// @brief Native HIP stream handle wrapper
/// @details Provides a type-safe wrapper around hipStream_t for use with
///          DTL execution policies and GPU algorithms.
/// @since 0.1.0
/// @see Phase 03: GPU-Safe Containers + Algorithm Dispatch

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace dtl {
namespace hip {

#if DTL_ENABLE_HIP

/// @brief RAII wrapper for HIP stream
/// @details Wraps a hipStream_t with type-safe accessors and optional
///          ownership semantics. Can either own a stream (destroying it
///          on destruction) or wrap an existing stream.
class stream_handle {
public:
    /// @brief Default constructor (default stream, no ownership)
    stream_handle() noexcept
        : stream_(0)
        , owns_(false) {}

    /// @brief Construct from raw stream
    /// @param stream The HIP stream to wrap
    /// @param owns Whether this wrapper owns the stream
    explicit stream_handle(hipStream_t stream, bool owns = false) noexcept
        : stream_(stream)
        , owns_(owns) {}

    /// @brief Construct and optionally create a new stream
    /// @param create_stream If true, creates a new stream and owns it
    explicit stream_handle(bool create_stream)
        : stream_(0)
        , owns_(false) {
        if (create_stream) {
            hipError_t err = hipStreamCreate(&stream_);
            owns_ = (err == hipSuccess);
        }
    }

    /// @brief Destructor
    ~stream_handle() noexcept {
        if (owns_ && stream_ != 0) {
            hipStreamDestroy(stream_);
        }
    }

    stream_handle(const stream_handle&) = delete;
    stream_handle& operator=(const stream_handle&) = delete;

    stream_handle(stream_handle&& other) noexcept
        : stream_(other.stream_)
        , owns_(other.owns_) {
        other.stream_ = 0;
        other.owns_ = false;
    }

    stream_handle& operator=(stream_handle&& other) noexcept {
        if (this != &other) {
            if (owns_ && stream_ != 0) {
                hipStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            owns_ = other.owns_;
            other.stream_ = 0;
            other.owns_ = false;
        }
        return *this;
    }

    [[nodiscard]] hipStream_t native_handle() const noexcept { return stream_; }
    [[nodiscard]] hipStream_t get() const noexcept { return stream_; }
    [[nodiscard]] operator hipStream_t() const noexcept { return stream_; }
    [[nodiscard]] bool is_default() const noexcept { return stream_ == 0; }
    [[nodiscard]] bool owns() const noexcept { return owns_; }

    [[nodiscard]] bool synchronize() const {
        return hipStreamSynchronize(stream_) == hipSuccess;
    }

    [[nodiscard]] bool query() const {
        return hipStreamQuery(stream_) == hipSuccess;
    }

    [[nodiscard]] static stream_handle create(unsigned int flags = hipStreamDefault) {
        hipStream_t stream;
        hipError_t err = hipStreamCreateWithFlags(&stream, flags);
        if (err == hipSuccess) {
            return stream_handle(stream, true);
        }
        return stream_handle();
    }

    [[nodiscard]] static stream_handle create_non_blocking() {
        return create(hipStreamNonBlocking);
    }

    [[nodiscard]] static stream_handle default_stream() noexcept {
        return stream_handle(0, false);
    }

private:
    hipStream_t stream_;
    bool owns_;
};

#else  // !DTL_ENABLE_HIP

/// @brief Stub stream handle when HIP is disabled
class stream_handle {
public:
    stream_handle() noexcept = default;
    explicit stream_handle(bool) noexcept {}

    [[nodiscard]] void* native_handle() const noexcept { return nullptr; }
    [[nodiscard]] void* get() const noexcept { return nullptr; }
    [[nodiscard]] bool is_default() const noexcept { return true; }
    [[nodiscard]] bool synchronize() const { return true; }
    [[nodiscard]] bool query() const { return true; }

    [[nodiscard]] static stream_handle create(unsigned int = 0) {
        return stream_handle{};
    }
    [[nodiscard]] static stream_handle default_stream() noexcept {
        return stream_handle{};
    }
};

#endif  // DTL_ENABLE_HIP

}  // namespace hip
}  // namespace dtl
