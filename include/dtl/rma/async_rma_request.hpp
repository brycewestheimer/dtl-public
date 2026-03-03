// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file async_rma_request.hpp
/// @brief Async RMA request with progress engine integration
/// @details Provides request types for async RMA operations that complete
///          via the progress engine instead of blocking.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/communication/memory_window.hpp>

#include <atomic>
#include <memory>
#include <optional>

namespace dtl::rma {

// ============================================================================
// Async RMA Request State
// ============================================================================

/// @brief State of an async RMA request
enum class rma_request_state {
    pending,    ///< Operation in progress
    ready,      ///< Operation completed successfully
    error       ///< Operation completed with error
};

// ============================================================================
// Async RMA Request Base
// ============================================================================

/// @brief Base class for async RMA requests
/// @details Provides common functionality for async get/put requests,
///          including progress engine integration and state management.
class async_rma_request_base {
public:
    /// @brief Check if the operation is complete
    [[nodiscard]] bool ready() const noexcept {
        return state_.load(std::memory_order_acquire) != rma_request_state::pending;
    }

    /// @brief Wait for the operation to complete
    void wait() {
        while (!ready()) {
            futures::progress_engine::instance().poll();
        }
    }

    /// @brief Get the error status (if any)
    [[nodiscard]] status get_error() const noexcept {
        return error_;
    }

    /// @brief Check if the operation completed with an error
    [[nodiscard]] bool has_error() const noexcept {
        return state_.load(std::memory_order_acquire) == rma_request_state::error;
    }

protected:
    async_rma_request_base() = default;
    ~async_rma_request_base() = default;

    // Non-copyable
    async_rma_request_base(const async_rma_request_base&) = delete;
    async_rma_request_base& operator=(const async_rma_request_base&) = delete;

    // Movable
    async_rma_request_base(async_rma_request_base&&) = default;
    async_rma_request_base& operator=(async_rma_request_base&&) = default;

    std::atomic<rma_request_state> state_{rma_request_state::pending};
    status error_;
    memory_window_impl::rma_request_handle request_handle_;
    memory_window_impl* window_ = nullptr;
    size_type callback_id_ = static_cast<size_type>(-1);
};

// ============================================================================
// Async Get Request
// ============================================================================

/// @brief Async get request for single-element RMA reads
/// @tparam T The element type
template <typename T>
class async_get_request : public async_rma_request_base {
public:
    /// @brief Construct an async get request
    /// @param target Target rank
    /// @param offset Offset in target window (bytes)
    /// @param window Memory window implementation
    async_get_request(rank_t target, size_type offset, memory_window_impl* window)
        : target_(target)
        , offset_(offset) {
        window_ = window;

        if (!window_ || !window_->valid()) {
            error_ = status{status_code::invalid_state};
            state_.store(rma_request_state::error, std::memory_order_release);
            return;
        }

        // Initiate the async get operation
        auto res = window_->async_get(&value_, sizeof(T), target_, offset_, request_handle_);
        if (res.has_error()) {
            error_ = res.error();
            state_.store(rma_request_state::error, std::memory_order_release);
            return;
        }

        // Check if already complete (e.g., synchronous fallback or local)
        if (request_handle_.completed) {
            state_.store(rma_request_state::ready, std::memory_order_release);
            return;
        }

        // Register with progress engine for polling
        callback_id_ = futures::progress_engine::instance().register_callback(
            [this]() { return this->poll_completion(); }
        );
    }

    /// @brief Destructor - unregisters callback
    ~async_get_request() {
        if (callback_id_ != static_cast<size_type>(-1)) {
            if (!ready()) {
                wait();  // Ensure completion before destruction
            }
            futures::progress_engine::instance().unregister_callback(callback_id_);
        }
    }

    // Non-copyable
    async_get_request(const async_get_request&) = delete;
    async_get_request& operator=(const async_get_request&) = delete;

    // Movable
    async_get_request(async_get_request&& other) noexcept
        : async_rma_request_base(std::move(other))
        , target_(other.target_)
        , offset_(other.offset_)
        , value_(std::move(other.value_)) {
        // Update callback registration with new this pointer
        if (callback_id_ != static_cast<size_type>(-1)) {
            futures::progress_engine::instance().unregister_callback(callback_id_);
            if (!ready()) {
                callback_id_ = futures::progress_engine::instance().register_callback(
                    [this]() { return this->poll_completion(); }
                );
            } else {
                callback_id_ = static_cast<size_type>(-1);
            }
        }
        other.callback_id_ = static_cast<size_type>(-1);
    }

    /// @brief Get the result value
    /// @return The fetched value or error
    [[nodiscard]] result<T> get() {
        wait();
        if (has_error()) {
            return error_;
        }
        return value_;
    }

private:
    /// @brief Poll for completion (called by progress engine)
    /// @return true if still pending, false if complete
    bool poll_completion() {
        if (ready()) {
            return false;  // Already complete
        }

        auto test_result = window_->test_async(request_handle_);
        if (test_result.has_error()) {
            error_ = test_result.error();
            state_.store(rma_request_state::error, std::memory_order_release);
            return false;
        }

        if (test_result.value()) {
            state_.store(rma_request_state::ready, std::memory_order_release);
            return false;
        }

        return true;  // Still pending
    }

    rank_t target_;
    size_type offset_;
    T value_{};
};

// ============================================================================
// Async Put Request
// ============================================================================

/// @brief Async put request for single-element RMA writes
/// @tparam T The element type
template <typename T>
class async_put_request : public async_rma_request_base {
public:
    /// @brief Construct an async put request
    /// @param target Target rank
    /// @param offset Offset in target window (bytes)
    /// @param value The value to write
    /// @param window Memory window implementation
    async_put_request(rank_t target, size_type offset, const T& value, memory_window_impl* window)
        : target_(target)
        , offset_(offset)
        , value_(value) {
        window_ = window;

        if (!window_ || !window_->valid()) {
            error_ = status{status_code::invalid_state};
            state_.store(rma_request_state::error, std::memory_order_release);
            return;
        }

        // Initiate the async put operation
        auto res = window_->async_put(&value_, sizeof(T), target_, offset_, request_handle_);
        if (res.has_error()) {
            error_ = res.error();
            state_.store(rma_request_state::error, std::memory_order_release);
            return;
        }

        // Check if already complete (e.g., synchronous fallback or local)
        if (request_handle_.completed) {
            state_.store(rma_request_state::ready, std::memory_order_release);
            return;
        }

        // Register with progress engine for polling
        callback_id_ = futures::progress_engine::instance().register_callback(
            [this]() { return this->poll_completion(); }
        );
    }

    /// @brief Destructor - unregisters callback
    ~async_put_request() {
        if (callback_id_ != static_cast<size_type>(-1)) {
            if (!ready()) {
                wait();  // Ensure completion before destruction
            }
            futures::progress_engine::instance().unregister_callback(callback_id_);
        }
    }

    // Non-copyable
    async_put_request(const async_put_request&) = delete;
    async_put_request& operator=(const async_put_request&) = delete;

    // Movable
    async_put_request(async_put_request&& other) noexcept
        : async_rma_request_base(std::move(other))
        , target_(other.target_)
        , offset_(other.offset_)
        , value_(std::move(other.value_)) {
        // Update callback registration with new this pointer
        if (callback_id_ != static_cast<size_type>(-1)) {
            futures::progress_engine::instance().unregister_callback(callback_id_);
            if (!ready()) {
                callback_id_ = futures::progress_engine::instance().register_callback(
                    [this]() { return this->poll_completion(); }
                );
            } else {
                callback_id_ = static_cast<size_type>(-1);
            }
        }
        other.callback_id_ = static_cast<size_type>(-1);
    }

    /// @brief Get the result (wait for completion)
    /// @return Success or error status
    [[nodiscard]] result<void> get() {
        wait();
        if (has_error()) {
            return error_;
        }
        return {};
    }

private:
    /// @brief Poll for completion (called by progress engine)
    /// @return true if still pending, false if complete
    bool poll_completion() {
        if (ready()) {
            return false;  // Already complete
        }

        auto test_result = window_->test_async(request_handle_);
        if (test_result.has_error()) {
            error_ = test_result.error();
            state_.store(rma_request_state::error, std::memory_order_release);
            return false;
        }

        if (test_result.value()) {
            state_.store(rma_request_state::ready, std::memory_order_release);
            return false;
        }

        return true;  // Still pending
    }

    rank_t target_;
    size_type offset_;
    T value_;
};

}  // namespace dtl::rma
