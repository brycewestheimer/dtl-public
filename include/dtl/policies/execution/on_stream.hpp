// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file on_stream.hpp
/// @brief GPU stream execution policy
/// @details Execute operations asynchronously on a specific GPU stream.
/// @since 0.1.0
/// @note Updated in 1.0.2: Added native stream handle support for CUDA/HIP.

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/cuda/stream_handle.hpp>
#endif

#if DTL_ENABLE_HIP
#include <dtl/hip/stream_handle.hpp>
#endif

namespace dtl {

// ============================================================================
// Default Stream Handle Type Selection
// ============================================================================

#if DTL_ENABLE_CUDA
/// @brief Default stream handle type when CUDA is enabled
using default_stream_handle = cuda::stream_handle;
#elif DTL_ENABLE_HIP
/// @brief Default stream handle type when HIP is enabled
using default_stream_handle = hip::stream_handle;
#else
/// @brief Placeholder stream type (no GPU backend enabled)
/// @note Will be replaced with actual CUDA/HIP stream type when backend enabled
struct default_stream_handle {
    void* native_handle = nullptr;

    /// @brief Check if this is the default stream
    [[nodiscard]] bool is_default() const noexcept {
        return native_handle == nullptr;
    }

    /// @brief Get native handle (for compatibility)
    [[nodiscard]] void* get() const noexcept {
        return native_handle;
    }

    /// @brief Synchronize (no-op without GPU)
    [[nodiscard]] bool synchronize() const noexcept {
        return true;
    }
};
#endif

/// @brief Legacy alias for backward compatibility
using stream_handle = default_stream_handle;

/// @brief GPU stream execution policy
/// @tparam Stream Stream type (e.g., cudaStream_t wrapper)
/// @details Execute operations asynchronously on a specific GPU stream,
///          enabling fine-grained control over GPU execution ordering
///          and overlap.
///
/// @par Characteristics:
/// - Non-blocking initiation
/// - Operations queued to specific stream
/// - Ordering guaranteed within stream
/// - No ordering between different streams
///
/// @par Synchronization:
/// - Use stream.synchronize() to wait for stream completion
/// - Use events for inter-stream synchronization
/// - Host-device synchronization via cudaDeviceSynchronize equivalent
///
/// @par Use Cases:
/// - Overlapping kernel execution with data transfers
/// - Multiple independent GPU operations
/// - Fine-grained GPU scheduling
template <typename Stream = stream_handle>
struct on_stream {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

    /// @brief The stream type
    using stream_type = Stream;

    /// @brief The stream to execute on
    Stream stream;

    /// @brief Construct with a stream
    /// @param s The stream to use
    explicit on_stream(Stream s) : stream{s} {}

    /// @brief Default construct (uses default stream)
    on_stream() : stream{} {}

    /// @brief Get the execution mode
    [[nodiscard]] static constexpr execution_mode mode() noexcept {
        return execution_mode::asynchronous;
    }

    /// @brief Check if execution is blocking
    [[nodiscard]] static constexpr bool is_blocking() noexcept {
        return false;  // Stream operations are asynchronous
    }

    /// @brief Check if execution is parallel
    [[nodiscard]] static constexpr bool is_parallel() noexcept {
        return true;  // GPU execution is massively parallel
    }

    /// @brief Get the parallelism level
    [[nodiscard]] static constexpr parallelism_level parallelism() noexcept {
        return parallelism_level::heterogeneous;
    }

    /// @brief Check if this targets a GPU
    [[nodiscard]] static constexpr bool is_device_execution() noexcept {
        return true;
    }

    /// @brief Get the stream
    [[nodiscard]] const Stream& get_stream() const noexcept {
        return stream;
    }

    /// @brief Synchronize the stream (wait for completion)
    void synchronize() const {
        if constexpr (requires { stream.synchronize(); }) {
            stream.synchronize();
        }
#if DTL_ENABLE_CUDA
        else if constexpr (std::is_same_v<Stream, cudaStream_t>) {
            cudaStreamSynchronize(stream);
        }
#endif
#if DTL_ENABLE_HIP
        else if constexpr (std::is_same_v<Stream, hipStream_t>) {
            hipStreamSynchronize(stream);
        }
#endif
    }

    /// @brief Get the native stream handle for backend APIs
    /// @return Native stream handle (cudaStream_t, hipStream_t, or void*)
    [[nodiscard]] auto native_handle() const noexcept {
        if constexpr (requires { stream.native_handle(); }) {
            return stream.native_handle();
        } else if constexpr (requires { stream.get(); }) {
            return stream.get();
        } else {
            return stream;
        }
    }
};

/// @brief Factory function to create on_stream policy
/// @tparam Stream Stream type
/// @param s The stream to use
/// @return on_stream<Stream> policy
template <typename Stream>
[[nodiscard]] auto make_on_stream(Stream&& s) {
    return on_stream<std::decay_t<Stream>>{std::forward<Stream>(s)};
}

/// @brief Specialization of execution_traits for on_stream
template <typename Stream>
struct execution_traits<on_stream<Stream>> {
    static constexpr bool is_blocking = false;
    static constexpr bool is_parallel = true;
    static constexpr execution_mode mode = execution_mode::asynchronous;
    static constexpr parallelism_level parallelism = parallelism_level::heterogeneous;
};

}  // namespace dtl
