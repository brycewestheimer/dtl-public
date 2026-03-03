// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_atomic.hpp
/// @brief Atomic RMA operations (accumulate, fetch_and_op, compare_and_swap)
/// @details Provides atomic operations for one-sided communication.
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
#include <functional>
#include <algorithm>

namespace dtl::rma {

// ============================================================================
// Atomic RMA Operations Implementation Interface
// ============================================================================

/// @brief Atomic RMA operations implementation interface
class atomic_rma_impl {
public:
    virtual ~atomic_rma_impl() = default;

    /// @brief Accumulate to remote window
    [[nodiscard]] virtual result<void> accumulate(
        rank_t target,
        size_type target_offset,
        const void* origin,
        size_type size,
        rma_reduce_op op,
        memory_window& window) = 0;

    /// @brief Fetch and operation
    [[nodiscard]] virtual result<void> fetch_and_op(
        rank_t target,
        size_type target_offset,
        const void* origin,
        void* result_buf,
        size_type size,
        rma_reduce_op op,
        memory_window& window) = 0;

    /// @brief Compare and swap
    [[nodiscard]] virtual result<void> compare_and_swap(
        rank_t target,
        size_type target_offset,
        const void* compare,
        const void* swap_val,
        void* result_buf,
        size_type size,
        memory_window& window) = 0;

    /// @brief Get-accumulate (combines get with accumulate)
    [[nodiscard]] virtual result<void> get_accumulate(
        rank_t target,
        size_type target_offset,
        const void* origin,
        void* result_buf,
        size_type size,
        rma_reduce_op op,
        memory_window& window) = 0;
};

// ============================================================================
// Null Atomic RMA Implementation
// ============================================================================

/// @brief Null atomic RMA implementation for testing
class null_atomic_rma_impl : public atomic_rma_impl {
public:
    [[nodiscard]] result<void> accumulate(
        rank_t target,
        size_type target_offset,
        const void* origin,
        size_type size,
        rma_reduce_op op,
        memory_window& window) override {

        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if (origin == nullptr && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        // Perform local accumulate operation
        auto* dest = static_cast<std::byte*>(window.base()) + target_offset;
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

    [[nodiscard]] result<void> fetch_and_op(
        rank_t target,
        size_type target_offset,
        const void* origin,
        void* result_buf,
        size_type size,
        rma_reduce_op op,
        memory_window& window) override {

        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if ((origin == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(window.base()) + target_offset;

        // Fetch old value
        std::memcpy(result_buf, dest, size);

        // Apply operation
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

    [[nodiscard]] result<void> compare_and_swap(
        rank_t target,
        size_type target_offset,
        const void* compare,
        const void* swap_val,
        void* result_buf,
        size_type size,
        memory_window& window) override {

        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if ((compare == nullptr || swap_val == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(window.base()) + target_offset;

        // Fetch current value
        std::memcpy(result_buf, dest, size);

        // Compare and swap if equal
        if (std::memcmp(dest, compare, size) == 0) {
            std::memcpy(dest, swap_val, size);
        }

        return result<void>{};
    }

    [[nodiscard]] result<void> get_accumulate(
        rank_t target,
        size_type target_offset,
        const void* origin,
        void* result_buf,
        size_type size,
        rma_reduce_op op,
        memory_window& window) override {

        if (!window.valid()) {
            return status_code::invalid_state;
        }
        if ((origin == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (!window_range_valid(window, target_offset, size)) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(window.base()) + target_offset;

        // Fetch old value
        std::memcpy(result_buf, dest, size);

        // Apply operation
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

private:
    /// @brief Apply a reduction operation element-by-element
    void apply_reduce_op(void* dest, const void* origin, size_type size, rma_reduce_op op) {
        // For simplicity, treat as array of bytes and apply operation byte-by-byte
        // In real implementation, this would be type-aware
        auto* d = static_cast<std::byte*>(dest);
        const auto* o = static_cast<const std::byte*>(origin);

        switch (op) {
            case rma_reduce_op::replace:
                std::memcpy(dest, origin, size);
                break;
            case rma_reduce_op::no_op:
                // Do nothing
                break;
            case rma_reduce_op::sum:
                // For demonstration, treat as uint8_t array
                for (size_type i = 0; i < size; ++i) {
                    d[i] = static_cast<std::byte>(
                        static_cast<uint8_t>(d[i]) + static_cast<uint8_t>(o[i]));
                }
                break;
            case rma_reduce_op::prod:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = static_cast<std::byte>(
                        static_cast<uint8_t>(d[i]) * static_cast<uint8_t>(o[i]));
                }
                break;
            case rma_reduce_op::min:
                for (size_type i = 0; i < size; ++i) {
                    if (static_cast<uint8_t>(o[i]) < static_cast<uint8_t>(d[i])) {
                        d[i] = o[i];
                    }
                }
                break;
            case rma_reduce_op::max:
                for (size_type i = 0; i < size; ++i) {
                    if (static_cast<uint8_t>(o[i]) > static_cast<uint8_t>(d[i])) {
                        d[i] = o[i];
                    }
                }
                break;
            case rma_reduce_op::band:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] & o[i];
                }
                break;
            case rma_reduce_op::bor:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] | o[i];
                }
                break;
            case rma_reduce_op::bxor:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] ^ o[i];
                }
                break;
        }
    }
};

// ============================================================================
// Global Atomic RMA Instance
// ============================================================================

/// @brief Get the default atomic RMA operations implementation
inline atomic_rma_impl& get_atomic_rma_impl() {
    static null_atomic_rma_impl impl;
    return impl;
}

// ============================================================================
// Accumulate Operations
// ============================================================================

/// @brief Accumulate to remote window with reduction operation
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param data Data to accumulate
/// @param op Reduction operation
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> accumulate(rank_t target,
                                       size_type target_offset,
                                       std::span<const T> data,
                                       rma_reduce_op op,
                                       memory_window& window) {
    return window.accumulate(data.data(), data.size_bytes(), target, target_offset, op);
}

/// @brief Accumulate single value to remote window
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param value Value to accumulate
/// @param op Reduction operation
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> accumulate(rank_t target,
                                       size_type target_offset,
                                       const T& value,
                                       rma_reduce_op op,
                                       memory_window& window) {
    return window.accumulate(&value, sizeof(T), target, target_offset, op);
}

// ============================================================================
// Fetch and Op Operations
// ============================================================================

/// @brief Fetch old value and apply operation atomically
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param origin_value Value to use in operation
/// @param result_value Reference to store fetched value
/// @param op Reduction operation
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> fetch_and_op(rank_t target,
                                         size_type target_offset,
                                         const T& origin_value,
                                         T& result_value,
                                         rma_reduce_op op,
                                         memory_window& window) {
    return window.fetch_and_op(&origin_value, &result_value, sizeof(T), target, target_offset, op);
}

// ============================================================================
// Compare and Swap Operations
// ============================================================================

/// @brief Atomic compare and swap
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param compare_value Value to compare against
/// @param swap_value Value to swap in if comparison succeeds
/// @param result_value Reference to store original value
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> compare_and_swap(rank_t target,
                                             size_type target_offset,
                                             const T& compare_value,
                                             const T& swap_value,
                                             T& result_value,
                                             memory_window& window) {
    return window.compare_and_swap(&swap_value, &compare_value, &result_value, sizeof(T), target, target_offset);
}

// ============================================================================
// Get-Accumulate Operations
// ============================================================================

/// @brief Get old value and accumulate new value atomically
/// @tparam T Element type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param origin Data to accumulate
/// @param result_buf Buffer to store fetched values
/// @param op Reduction operation
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> get_accumulate(rank_t target,
                                           size_type target_offset,
                                           std::span<const T> origin,
                                           std::span<T> result_buf,
                                           rma_reduce_op op,
                                           memory_window& window) {
    if (origin.size() != result_buf.size()) {
        return status_code::extent_mismatch;
    }
    return window.get_accumulate(origin.data(), result_buf.data(), origin.size_bytes(), target, target_offset, op);
}

/// @brief Get old value and accumulate single value atomically
/// @tparam T Value type
/// @param target Target rank
/// @param target_offset Offset in target window (in bytes)
/// @param origin_value Value to accumulate
/// @param result_value Reference to store fetched value
/// @param op Reduction operation
/// @param window Memory window
/// @return Result indicating success or failure
template <typename T>
[[nodiscard]] result<void> get_accumulate(rank_t target,
                                           size_type target_offset,
                                           const T& origin_value,
                                           T& result_value,
                                           rma_reduce_op op,
                                           memory_window& window) {
    return window.get_accumulate(&origin_value, &result_value, sizeof(T), target, target_offset, op);
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/// @brief Alias for sum accumulate
template <typename T>
[[nodiscard]] result<void> accumulate_sum(rank_t target,
                                           size_type target_offset,
                                           std::span<const T> data,
                                           memory_window& window) {
    return accumulate(target, target_offset, data, rma_reduce_op::sum, window);
}

/// @brief Alias for max accumulate
template <typename T>
[[nodiscard]] result<void> accumulate_max(rank_t target,
                                           size_type target_offset,
                                           std::span<const T> data,
                                           memory_window& window) {
    return accumulate(target, target_offset, data, rma_reduce_op::max, window);
}

/// @brief Alias for min accumulate
template <typename T>
[[nodiscard]] result<void> accumulate_min(rank_t target,
                                           size_type target_offset,
                                           std::span<const T> data,
                                           memory_window& window) {
    return accumulate(target, target_offset, data, rma_reduce_op::min, window);
}

/// @brief Atomic fetch-and-add
template <typename T>
[[nodiscard]] result<void> fetch_and_add(rank_t target,
                                          size_type target_offset,
                                          const T& addend,
                                          T& result_value,
                                          memory_window& window) {
    return fetch_and_op(target, target_offset, addend, result_value,
                        rma_reduce_op::sum, window);
}

}  // namespace dtl::rma
