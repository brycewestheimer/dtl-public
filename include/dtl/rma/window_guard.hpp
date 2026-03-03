// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file window_guard.hpp
/// @brief RAII guards for RMA window synchronization
/// @details Provides fence_guard, lock_guard, and rma_batch for safe
///          RMA epoch management.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>

#include <utility>
#include <vector>

namespace dtl::rma {

// ============================================================================
// Fence Guard
// ============================================================================

/// @brief RAII guard for fence synchronization epochs
/// @details Automatically calls fence() on destruction, ensuring proper
///          synchronization of RMA operations.
///
/// @par Example Usage
/// @code
/// memory_window win = ...;
/// {
///     fence_guard guard(win);
///     // Perform RMA operations within this epoch
///     rma::put(target, offset, data, win);
///     rma::get(target, offset, buffer, win);
/// } // fence() called automatically here
/// @endcode
class fence_guard {
public:
    /// @brief Construct a fence guard for the window
    /// @param window The memory window to guard
    /// @param assert_flags Initial fence assertion flags
    explicit fence_guard(memory_window& window, int assert_flags = 0)
        : window_(&window), assert_flags_(assert_flags) {
        // Optionally call initial fence to begin epoch
        if (window_->valid()) {
            (void)window_->fence(assert_flags_);
        }
    }

    /// @brief Destructor - calls fence to complete epoch
    ~fence_guard() {
        if (window_ && window_->valid()) {
            (void)window_->fence(assert_flags_);
        }
    }

    /// @brief Manual fence within the epoch
    /// @param assert_flags Assertion flags for this fence
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> fence(int assert_flags = 0) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return window_->fence(assert_flags);
    }

    /// @brief Check if the guard is valid
    [[nodiscard]] bool valid() const noexcept {
        return window_ && window_->valid();
    }

    // Non-copyable
    fence_guard(const fence_guard&) = delete;
    fence_guard& operator=(const fence_guard&) = delete;

    // Non-movable (to ensure deterministic fence calls)
    fence_guard(fence_guard&&) = delete;
    fence_guard& operator=(fence_guard&&) = delete;

private:
    memory_window* window_;
    int assert_flags_;
};

// ============================================================================
// Lock Guard
// ============================================================================

/// @brief RAII guard for lock/unlock synchronization epochs
/// @details Automatically calls unlock() on destruction for passive-target
///          RMA operations.
///
/// @par Example Usage
/// @code
/// memory_window win = ...;
/// {
///     lock_guard guard(target_rank, win);  // lock acquired
///     // Perform RMA operations to target
///     rma::put(target_rank, offset, data, win);
///     guard.flush();  // ensure operations complete
/// } // unlock() called automatically here
/// @endcode
class lock_guard {
public:
    /// @brief Construct a lock guard for a specific target
    /// @param target Target rank to lock
    /// @param window Memory window
    /// @param mode Lock mode (exclusive or shared)
    lock_guard(rank_t target, memory_window& window,
               rma_lock_mode mode = rma_lock_mode::exclusive)
        : window_(&window), target_(target), is_all_(false), locked_(false) {
        if (window_->valid()) {
            auto res = window_->lock(target, mode);
            locked_ = res.has_value();
        }
    }

    /// @brief Construct a lock_all guard for all targets
    /// @param window Memory window
    explicit lock_guard(memory_window& window)
        : window_(&window), target_(no_rank), is_all_(true), locked_(false) {
        if (window_->valid()) {
            auto res = window_->lock_all();
            locked_ = res.has_value();
        }
    }

    /// @brief Destructor - unlocks the window
    ~lock_guard() {
        if (locked_ && window_ && window_->valid()) {
            if (is_all_) {
                (void)window_->unlock_all();
            } else {
                (void)window_->unlock(target_);
            }
        }
    }

    /// @brief Check if lock was successfully acquired
    [[nodiscard]] bool locked() const noexcept {
        return locked_;
    }

    /// @brief Flush operations to the locked target
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush() {
        if (!locked_ || !window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        if (is_all_) {
            return window_->flush_all();
        } else {
            return window_->flush(target_);
        }
    }

    /// @brief Flush local completion
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_local() {
        if (!locked_ || !window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        if (is_all_) {
            return window_->flush_local_all();
        } else {
            return window_->flush_local(target_);
        }
    }

    /// @brief Get the locked target rank
    [[nodiscard]] rank_t target() const noexcept {
        return target_;
    }

    /// @brief Check if this is a lock_all guard
    [[nodiscard]] bool is_all() const noexcept {
        return is_all_;
    }

    // Non-copyable
    lock_guard(const lock_guard&) = delete;
    lock_guard& operator=(const lock_guard&) = delete;

    // Non-movable (to ensure deterministic unlock)
    lock_guard(lock_guard&&) = delete;
    lock_guard& operator=(lock_guard&&) = delete;

private:
    memory_window* window_;
    rank_t target_;
    bool is_all_;
    bool locked_;
};

// ============================================================================
// RMA Batch
// ============================================================================

/// @brief Scoped RMA operation batch with automatic flush
/// @details Collects RMA operations and flushes them on destruction.
///          Useful for grouping related operations.
///
/// @par Example Usage
/// @code
/// memory_window win = ...;
/// {
///     rma_batch batch(win);
///     batch.put(target1, offset1, data1);
///     batch.put(target2, offset2, data2);
///     batch.get(target3, offset3, buffer);
/// } // flush_all() called automatically
/// @endcode
class rma_batch {
public:
    /// @brief Construct an RMA batch for a window
    /// @param window Memory window
    explicit rma_batch(memory_window& window)
        : window_(&window) {}

    /// @brief Destructor - flushes all pending operations
    ~rma_batch() {
        if (window_ && window_->valid()) {
            (void)window_->flush_all();
        }
    }

    /// @brief Put data to target window
    /// @tparam T Element type
    /// @param target Target rank
    /// @param offset Target offset in bytes
    /// @param data Data to put
    /// @return Result indicating success or failure
    template <typename T>
    [[nodiscard]] result<void> put(rank_t target, size_type offset, std::span<const T> data) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return rma::put(target, offset, data, *window_);
    }

    /// @brief Put single value to target window
    /// @tparam T Value type
    /// @param target Target rank
    /// @param offset Target offset in bytes
    /// @param value Value to put
    /// @return Result indicating success or failure
    template <typename T>
    [[nodiscard]] result<void> put(rank_t target, size_type offset, const T& value) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return rma::put(target, offset, value, *window_);
    }

    /// @brief Get data from target window
    /// @tparam T Element type
    /// @param target Target rank
    /// @param offset Target offset in bytes
    /// @param buffer Buffer to receive data
    /// @return Result indicating success or failure
    template <typename T>
    [[nodiscard]] result<void> get(rank_t target, size_type offset, std::span<T> buffer) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return rma::get(target, offset, buffer, *window_);
    }

    /// @brief Get single value from target window
    /// @tparam T Value type
    /// @param target Target rank
    /// @param offset Target offset in bytes
    /// @param value Reference to receive value
    /// @return Result indicating success or failure
    template <typename T>
    [[nodiscard]] result<void> get(rank_t target, size_type offset, T& value) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return rma::get(target, offset, value, *window_);
    }

    /// @brief Manually flush all operations
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_all() {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return window_->flush_all();
    }

    /// @brief Flush operations to a specific target
    /// @param target Target rank
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush(rank_t target) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }
        return window_->flush(target);
    }

    /// @brief Check if the batch is valid
    [[nodiscard]] bool valid() const noexcept {
        return window_ && window_->valid();
    }

    /// @brief Get the number of operations in the batch
    /// @note This is an estimate - actual count depends on implementation
    [[nodiscard]] size_type operation_count() const noexcept {
        return operation_count_;
    }

    // Non-copyable
    rma_batch(const rma_batch&) = delete;
    rma_batch& operator=(const rma_batch&) = delete;

    // Non-movable
    rma_batch(rma_batch&&) = delete;
    rma_batch& operator=(rma_batch&&) = delete;

private:
    memory_window* window_;
    size_type operation_count_ = 0;
};

// ============================================================================
// Scoped Epoch Helper
// ============================================================================

/// @brief Helper to create scoped fence epochs
/// @param window Memory window
/// @param f Function to execute within the epoch
/// @return Result from the function
template <typename F>
    requires std::invocable<F>
[[nodiscard]] auto with_fence_epoch(memory_window& window, F&& f) -> result<void> {
    fence_guard guard(window);
    if (!guard.valid()) {
        return status_code::invalid_state;
    }

    if constexpr (std::is_same_v<std::invoke_result_t<F>, void>) {
        std::forward<F>(f)();
        return result<void>{};
    } else {
        return std::forward<F>(f)();
    }
}

/// @brief Helper to create scoped lock epochs
/// @param target Target rank
/// @param window Memory window
/// @param mode Lock mode
/// @param f Function to execute within the epoch
/// @return Result from the function
template <typename F>
    requires std::invocable<F>
[[nodiscard]] auto with_lock_epoch(rank_t target, memory_window& window,
                                    rma_lock_mode mode, F&& f) -> result<void> {
    lock_guard guard(target, window, mode);
    if (!guard.locked()) {
        return status_code::invalid_state;
    }

    if constexpr (std::is_same_v<std::invoke_result_t<F>, void>) {
        std::forward<F>(f)();
        return guard.flush();
    } else {
        auto res = std::forward<F>(f)();
        if (res.has_error()) {
            return res;
        }
        return guard.flush();
    }
}

}  // namespace dtl::rma
