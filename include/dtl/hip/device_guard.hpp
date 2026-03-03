// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file device_guard.hpp
/// @brief RAII device guard for HIP device selection
/// @details Provides scoped device selection that restores the previous device on destruction.
///          Thread-safe and exception-safe.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace dtl {
namespace hip {

/// @brief Invalid device ID sentinel
inline constexpr int invalid_device_id = -1;

/// @brief RAII guard for HIP device selection
/// @details Sets the HIP device on construction and restores the previous device
///          on destruction. Thread-safe: each thread maintains its own device context.
///
/// @par Example
/// @code
/// int prev = dtl::hip::current_device_id();
/// {
///     dtl::hip::device_guard guard(1);
///     // Operations here target device 1
///     hipMalloc(...);  // Allocates on device 1
/// }
/// // Previous device restored
/// assert(dtl::hip::current_device_id() == prev);
/// @endcode
///
/// @note When HIP is disabled at compile time, this is a no-op.
class device_guard {
public:
    /// @brief Construct guard for specific device
    /// @param target_device Device ID to switch to
    /// @note If target_device is invalid_device_id, no device switch occurs
    explicit device_guard(int target_device) noexcept
        : target_device_(target_device)
        , previous_device_(invalid_device_id)
        , switched_(false) {
#if DTL_ENABLE_HIP
        if (target_device_ != invalid_device_id) {
            hipError_t err = hipGetDevice(&previous_device_);
            if (err == hipSuccess && previous_device_ != target_device_) {
                err = hipSetDevice(target_device_);
                switched_ = (err == hipSuccess);
            } else if (err == hipSuccess) {
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
#if DTL_ENABLE_HIP
        if (switched_ && previous_device_ != invalid_device_id) {
            // Best-effort restore, ignore errors
            hipSetDevice(previous_device_);
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

    /// @brief Check if HIP device guards are available at runtime
    /// @return true if HIP is enabled and at least one device is available
    [[nodiscard]] static bool available() noexcept {
#if DTL_ENABLE_HIP
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#else
        return false;
#endif
    }

private:
    int target_device_;
    int previous_device_;
    bool switched_;
};

/// @brief Get current HIP device ID
/// @return Current device ID or invalid_device_id if HIP not enabled/no devices
[[nodiscard]] inline int current_device_id() noexcept {
#if DTL_ENABLE_HIP
    int device;
    if (hipGetDevice(&device) == hipSuccess) {
        return device;
    }
#endif
    return invalid_device_id;
}

/// @brief Get number of HIP devices
/// @return Number of devices or 0 if HIP not enabled
[[nodiscard]] inline int device_count() noexcept {
#if DTL_ENABLE_HIP
    int count;
    if (hipGetDeviceCount(&count) == hipSuccess) {
        return count;
    }
#endif
    return 0;
}

}  // namespace hip
}  // namespace dtl
