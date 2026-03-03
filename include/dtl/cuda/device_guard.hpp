// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_guard.hpp
/// @brief RAII device guard for CUDA device selection
/// @details Provides scoped device selection that restores the previous device on destruction.
///          Thread-safe and exception-safe.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl {
namespace cuda {

/// @brief Invalid device ID sentinel
inline constexpr int invalid_device_id = -1;

/// @brief RAII guard for CUDA device selection
/// @details Sets the CUDA device on construction and restores the previous device
///          on destruction. Thread-safe: each thread maintains its own device context.
///
/// @par Example
/// @code
/// int prev = dtl::cuda::current_device_id();
/// {
///     dtl::cuda::device_guard guard(1);
///     // Operations here target device 1
///     cudaMalloc(...);  // Allocates on device 1
/// }
/// // Previous device restored
/// assert(dtl::cuda::current_device_id() == prev);
/// @endcode
///
/// @note When CUDA is disabled at compile time, this is a no-op.
class device_guard {
public:
    /// @brief Construct guard for specific device
    /// @param target_device Device ID to switch to
    /// @note If target_device is invalid_device_id, no device switch occurs
    explicit device_guard(int target_device) noexcept
        : target_device_(target_device)
        , previous_device_(invalid_device_id)
        , switched_(false) {
#if DTL_ENABLE_CUDA
        if (target_device_ != invalid_device_id) {
            cudaError_t err = cudaGetDevice(&previous_device_);
            if (err == cudaSuccess && previous_device_ != target_device_) {
                err = cudaSetDevice(target_device_);
                switched_ = (err == cudaSuccess);
            } else if (err == cudaSuccess) {
                // Already on target device, nothing to restore
                previous_device_ = invalid_device_id;
            }
        }
#else
        (void)target_device_;
#endif
    }

    /// @brief Destructor restores previous device
    /// @note Best-effort restoration, errors are swallowed (noexcept)
    ~device_guard() noexcept {
#if DTL_ENABLE_CUDA
        if (switched_ && previous_device_ != invalid_device_id) {
            // Best-effort restore, ignore errors
            cudaSetDevice(previous_device_);
        }
#endif
    }

    // Non-copyable, non-movable (RAII scope guard)
    device_guard(const device_guard&) = delete;
    device_guard& operator=(const device_guard&) = delete;
    device_guard(device_guard&&) = delete;
    device_guard& operator=(device_guard&&) = delete;

    /// @brief Get the previous device ID (before guard was constructed)
    /// @return Previous device ID or invalid_device_id if not switched
    [[nodiscard]] int previous_device() const noexcept {
        return previous_device_;
    }

    /// @brief Get the target device ID
    /// @return Target device ID
    [[nodiscard]] int target_device() const noexcept {
        return target_device_;
    }

    /// @brief Check if device switch occurred
    /// @return true if device was changed
    [[nodiscard]] bool switched() const noexcept {
        return switched_;
    }

    /// @brief Check if CUDA device guards are available at runtime
    /// @return true if CUDA is enabled and at least one device is available
    [[nodiscard]] static bool available() noexcept {
#if DTL_ENABLE_CUDA
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
        return false;
#endif
    }

private:
    int target_device_;
    int previous_device_;
    bool switched_;
};

/// @brief Get current CUDA device ID
/// @return Current device ID or invalid_device_id if CUDA not enabled/no devices
[[nodiscard]] inline int current_device_id() noexcept {
#if DTL_ENABLE_CUDA
    int device;
    if (cudaGetDevice(&device) == cudaSuccess) {
        return device;
    }
#endif
    return invalid_device_id;
}

/// @brief Get number of CUDA devices
/// @return Number of devices or 0 if CUDA not enabled
[[nodiscard]] inline int device_count() noexcept {
#if DTL_ENABLE_CUDA
    int count;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        return count;
    }
#endif
    return 0;
}

}  // namespace cuda
}  // namespace dtl
