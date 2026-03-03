// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file completion.hpp
/// @brief Completion tracking primitives for coordinating async operations
/// @details Provides completion_set and completion_token for tracking when
///          multiple async operations complete, used by when_all/when_any.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/progress.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace dtl::futures {

// ============================================================================
// Completion Token
// ============================================================================

/// @brief Token representing a single completable operation
/// @details Used to signal when an individual operation completes.
///          Multiple tokens can be grouped into a completion_set.
class completion_token {
public:
    /// @brief Completion callback type
    using callback_type = std::function<void(size_type index, bool success)>;

    /// @brief Construct a completion token
    /// @param index Index of this operation in its completion set
    /// @param on_complete Callback when operation completes
    explicit completion_token(size_type index, callback_type on_complete = nullptr)
        : index_(index)
        , completed_(false)
        , success_(false)
        , on_complete_(std::move(on_complete)) {}

    /// @brief Mark this token as complete with success
    void complete() {
        complete_impl(true);
    }

    /// @brief Mark this token as complete with failure
    void fail() {
        complete_impl(false);
    }

    /// @brief Check if this token has completed
    [[nodiscard]] bool is_complete() const noexcept {
        return completed_.load(std::memory_order_acquire);
    }

    /// @brief Check if this token completed successfully
    [[nodiscard]] bool is_success() const noexcept {
        return success_.load(std::memory_order_acquire);
    }

    /// @brief Get the index of this token
    [[nodiscard]] size_type index() const noexcept {
        return index_;
    }

private:
    void complete_impl(bool success) {
        bool expected = false;
        if (completed_.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
            success_.store(success, std::memory_order_release);
            if (on_complete_) {
                on_complete_(index_, success);
            }
        }
    }

    size_type index_;
    std::atomic<bool> completed_;
    std::atomic<bool> success_;
    callback_type on_complete_;
};

// ============================================================================
// Completion Set
// ============================================================================

/// @brief Tracks completion of multiple operations
/// @details Provides coordination primitives for when_all and when_any patterns.
class completion_set {
public:
    /// @brief Completion modes
    enum class mode {
        all,    ///< Wait for all operations to complete
        any     ///< Wait for any single operation to complete
    };

    /// @brief Completion callback type (called when set criteria met)
    using callback_type = std::function<void()>;

    /// @brief Construct a completion set
    /// @param count Number of operations to track
    /// @param m Completion mode (all or any)
    explicit completion_set(size_type count, mode m = mode::all)
        : count_(count)
        , mode_(m)
        , completed_count_(0)
        , first_completed_index_(count)  // Invalid index initially
        , set_complete_(count == 0) {}  // Empty set is immediately complete

    /// @brief Create a token for tracking operation at given index
    /// @param index Index of the operation
    /// @return A completion_token for signaling completion
    [[nodiscard]] std::shared_ptr<completion_token> create_token(size_type index) {
        return std::make_shared<completion_token>(
            index,
            [this](size_type idx, bool success) {
                on_token_complete(idx, success);
            });
    }

    /// @brief Check if the set's completion criteria is met
    /// @return true if complete according to mode
    [[nodiscard]] bool is_complete() const noexcept {
        return set_complete_.load(std::memory_order_acquire);
    }

    /// @brief Get index of first completed operation (for when_any)
    /// @return Index of first operation to complete, or count_ if none
    [[nodiscard]] size_type first_completed() const noexcept {
        return first_completed_index_.load(std::memory_order_acquire);
    }

    /// @brief Get count of completed operations
    [[nodiscard]] size_type completed_count() const noexcept {
        return completed_count_.load(std::memory_order_acquire);
    }

    /// @brief Get total operation count
    [[nodiscard]] size_type total_count() const noexcept {
        return count_;
    }

    /// @brief Set callback for when completion criteria is met
    /// @param callback Function to call on completion
    void on_complete(callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        on_complete_ = std::move(callback);
        // If already complete, fire immediately
        if (set_complete_.load(std::memory_order_acquire) && on_complete_) {
            on_complete_();
        }
    }

    /// @brief Wait for completion using progress-based polling
    /// @return true when complete
    bool wait_with_progress() {
        while (!is_complete()) {
            make_progress();
            if (is_complete()) break;
            std::this_thread::yield();
        }
        return true;
    }

private:
    void on_token_complete(size_type index, bool /*success*/) {
        size_type new_count = ++completed_count_;

        // For when_any mode, record first completed
        if (mode_ == mode::any) {
            size_type expected = count_;  // Invalid initially
            first_completed_index_.compare_exchange_strong(
                expected, index, std::memory_order_acq_rel);
        }

        // Check if set criteria is met
        bool should_complete = false;
        if (mode_ == mode::all && new_count >= count_) {
            should_complete = true;
        } else if (mode_ == mode::any && new_count >= 1) {
            should_complete = true;
        }

        if (should_complete) {
            bool expected = false;
            if (set_complete_.compare_exchange_strong(expected, true,
                                                       std::memory_order_acq_rel)) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (on_complete_) {
                    on_complete_();
                }
            }
        }
    }

    size_type count_;
    mode mode_;
    std::atomic<size_type> completed_count_;
    std::atomic<size_type> first_completed_index_;
    std::atomic<bool> set_complete_;
    std::mutex mutex_;
    callback_type on_complete_;
};

// ============================================================================
// Completion Waiter
// ============================================================================

/// @brief Helper for progress-based waiting on completion sets
/// @details Integrates with the progress engine for non-blocking coordination.
class completion_waiter {
public:
    /// @brief Create a waiter for a completion set
    /// @param set The completion set to wait on
    explicit completion_waiter(std::shared_ptr<completion_set> set)
        : set_(std::move(set)) {}

    /// @brief Wait for completion using progress-based polling
    /// @return true when set criteria is met
    bool wait() {
        return set_->wait_with_progress();
    }

    /// @brief Poll once without blocking
    /// @return true if complete
    bool poll() {
        make_progress();
        return set_->is_complete();
    }

    /// @brief Check if complete without polling
    [[nodiscard]] bool is_complete() const noexcept {
        return set_->is_complete();
    }

private:
    std::shared_ptr<completion_set> set_;
};

}  // namespace dtl::futures
