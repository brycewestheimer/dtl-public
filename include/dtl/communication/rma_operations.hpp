// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_operations.hpp
/// @brief Core RMA (Remote Memory Access) operations
/// @details Provides put/get operations for one-sided communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>
#include <dtl/communication/memory_window.hpp>

#include <cstring>
#include <span>
#include <type_traits>

namespace dtl::rma {

// ============================================================================
// RMA Operation Implementation (Null Backend)
// ============================================================================

/// @brief RMA operations implementation interface
class rma_operations_impl {
public:
    virtual ~rma_operations_impl() = default;

    /// @brief Put data to a target window
    [[nodiscard]] virtual result<void> put(
        rank_t target,
        size_type target_offset,
        const void* data,
        size_type size,
        memory_window& window) = 0;

    /// @brief Get data from a target window
    [[nodiscard]] virtual result<void> get(
        rank_t target,
        size_type target_offset,
        void* buffer,
        size_type size,
        memory_window& window) = 0;

    /// @brief Non-blocking put
    [[nodiscard]] virtual result<request_handle> put_async(
        rank_t target,
        size_type target_offset,
        const void* data,
        size_type size,
        memory_window& window) = 0;

    /// @brief Non-blocking get
    [[nodiscard]] virtual result<request_handle> get_async(
        rank_t target,
        size_type target_offset,
        void* buffer,
        size_type size,
        memory_window& window) = 0;
};

/// @brief Null RMA implementation for testing and single-process use
class null_rma_impl : public rma_operations_impl {
public:
    [[nodiscard]] result<void> put(
        rank_t target,
        size_type target_offset,
        const void* data,
        size_type size,
        memory_window& window) override {

        // Validate parameters
        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if (data == nullptr && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }

        // In null implementation, only target 0 (self) is valid
        if (target != 0) {
            return status_code::invalid_rank;
        }

        // Perform local copy
        if (size > 0) {
            auto* base = static_cast<std::byte*>(window.base());
            if (base == nullptr) {
                return status_code::invalid_state;
            }
            auto* dest = base + target_offset;
            std::memcpy(dest, data, size);
        }

        return result<void>{};
    }

    [[nodiscard]] result<void> get(
        rank_t target,
        size_type target_offset,
        void* buffer,
        size_type size,
        memory_window& window) override {

        // Validate parameters
        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if (buffer == nullptr && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }

        // In null implementation, only target 0 (self) is valid
        if (target != 0) {
            return status_code::invalid_rank;
        }

        // Perform local copy
        if (size > 0) {
            const auto* base = static_cast<const std::byte*>(window.base());
            if (base == nullptr) {
                return status_code::invalid_state;
            }
            const auto* src = base + target_offset;
            std::memcpy(buffer, src, size);
        }

        return result<void>{};
    }

    [[nodiscard]] result<request_handle> put_async(
        rank_t target,
        size_type target_offset,
        const void* data,
        size_type size,
        memory_window& window) override {

        // Synchronous put in null implementation
        auto res = put(target, target_offset, data, size, window);
        if (res.has_error()) {
            return res.error();
        }
        // Return a completed request handle
        return request_handle{reinterpret_cast<void*>(1)};
    }

    [[nodiscard]] result<request_handle> get_async(
        rank_t target,
        size_type target_offset,
        void* buffer,
        size_type size,
        memory_window& window) override {

        // Synchronous get in null implementation
        auto res = get(target, target_offset, buffer, size, window);
        if (res.has_error()) {
            return res.error();
        }
        // Return a completed request handle
        return request_handle{reinterpret_cast<void*>(1)};
    }
};

// ============================================================================
// Global RMA Operations Instance
// ============================================================================

/// @brief Get the default RMA operations implementation
/// @return Reference to the RMA operations implementation
/// @details Returns the null implementation by default. Backends can
///          replace this with their own implementation.
inline rma_operations_impl& get_rma_impl() {
    static null_rma_impl impl;
    return impl;
}

// ============================================================================
// Put Operations
// ============================================================================

/// @brief Put data to a remote window
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param data Data to send
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> put(rank_t target,
                                size_type target_offset,
                                std::span<const T> data,
                                memory_window& window) {
    return get_rma_impl().put(
        target,
        target_offset,
        data.data(),
        data.size_bytes(),
        window
    );
}

/// @brief Put a single value to a remote window
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param value Value to send
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> put(rank_t target,
                                size_type target_offset,
                                const T& value,
                                memory_window& window) {
    return get_rma_impl().put(
        target,
        target_offset,
        &value,
        sizeof(T),
        window
    );
}

/// @brief Put raw bytes to a remote window
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param data Pointer to data
/// @param size Size in bytes
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> put(rank_t target,
                                       size_type target_offset,
                                       const void* data,
                                       size_type size,
                                       memory_window& window) {
    return get_rma_impl().put(target, target_offset, data, size, window);
}

// ============================================================================
// Get Operations
// ============================================================================

/// @brief Get data from a remote window
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param buffer Buffer to receive data
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> get(rank_t target,
                                size_type target_offset,
                                std::span<T> buffer,
                                memory_window& window) {
    return get_rma_impl().get(
        target,
        target_offset,
        buffer.data(),
        buffer.size_bytes(),
        window
    );
}

/// @brief Get a single value from a remote window
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param value Reference to receive value
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> get(rank_t target,
                                size_type target_offset,
                                T& value,
                                memory_window& window) {
    return get_rma_impl().get(
        target,
        target_offset,
        &value,
        sizeof(T),
        window
    );
}

/// @brief Get raw bytes from a remote window
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param buffer Buffer to receive data
/// @param size Size in bytes
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> get(rank_t target,
                                       size_type target_offset,
                                       void* buffer,
                                       size_type size,
                                       memory_window& window) {
    return get_rma_impl().get(target, target_offset, buffer, size, window);
}

// ============================================================================
// Asynchronous Put Operations
// ============================================================================

/// @brief Non-blocking put operation
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param data Data to send
/// @param window Memory window
/// @return Result containing request handle or error
template <typename T>
[[nodiscard]] result<request_handle> put_async(rank_t target,
                                                size_type target_offset,
                                                std::span<const T> data,
                                                memory_window& window) {
    return get_rma_impl().put_async(
        target,
        target_offset,
        data.data(),
        data.size_bytes(),
        window
    );
}

/// @brief Non-blocking put of raw bytes
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param data Pointer to data
/// @param size Size in bytes
/// @param window Memory window
/// @return Result containing request handle or error
[[nodiscard]] inline result<request_handle> put_async(rank_t target,
                                                       size_type target_offset,
                                                       const void* data,
                                                       size_type size,
                                                       memory_window& window) {
    return get_rma_impl().put_async(target, target_offset, data, size, window);
}

// ============================================================================
// Asynchronous Get Operations
// ============================================================================

/// @brief Non-blocking get operation
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param buffer Buffer to receive data
/// @param window Memory window
/// @return Result containing request handle or error
template <typename T>
[[nodiscard]] result<request_handle> get_async(rank_t target,
                                                size_type target_offset,
                                                std::span<T> buffer,
                                                memory_window& window) {
    return get_rma_impl().get_async(
        target,
        target_offset,
        buffer.data(),
        buffer.size_bytes(),
        window
    );
}

/// @brief Non-blocking get of raw bytes
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param buffer Buffer to receive data
/// @param size Size in bytes
/// @param window Memory window
/// @return Result containing request handle or error
[[nodiscard]] inline result<request_handle> get_async(rank_t target,
                                                       size_type target_offset,
                                                       void* buffer,
                                                       size_type size,
                                                       memory_window& window) {
    return get_rma_impl().get_async(target, target_offset, buffer, size, window);
}

// ============================================================================
// Flush Operations
// ============================================================================

/// @brief Flush operations to a specific target
/// @param target Target rank
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> flush(rank_t target, memory_window& window) {
    return window.flush(target);
}

/// @brief Flush operations to all targets
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> flush_all(memory_window& window) {
    return window.flush_all();
}

/// @brief Flush local completion for a target
/// @param target Target rank
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> flush_local(rank_t target, memory_window& window) {
    return window.flush_local(target);
}

/// @brief Flush local completion for all targets
/// @param window Memory window
/// @return Result indicating success or failure
[[nodiscard]] inline result<void> flush_local_all(memory_window& window) {
    return window.flush_local_all();
}

}  // namespace dtl::rma
