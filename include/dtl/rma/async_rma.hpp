// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file async_rma.hpp
/// @brief Asynchronous RMA operations with progress engine integration
/// @details Provides async_put and async_get classes that integrate with
///          dtl::futures::progress_engine for non-blocking RMA operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <span>

namespace dtl::rma {

// ============================================================================
// Async RMA State
// ============================================================================

/// @brief State of an async RMA operation
enum class async_rma_state {
    pending,    ///< Operation not yet started
    initiated,  ///< RMA operation issued, awaiting flush completion
    ready,      ///< Operation completed successfully
    error       ///< Operation completed with error
};

// ============================================================================
// Async Put
// ============================================================================

/// @brief Invalid callback ID sentinel value
inline constexpr size_type invalid_callback_id = static_cast<size_type>(-1);

/// @brief Asynchronous put operation with progress engine integration
/// @tparam T Element type
template <typename T>
class async_put {
public:
    /// @brief Construct an async put operation
    /// @param target Target rank
    /// @param offset Offset in target window (bytes)
    /// @param data Data to put
    /// @param window Memory window
    async_put(rank_t target, size_type offset, std::span<const T> data, memory_window& window)
        : target_(target)
        , offset_(offset)
        , data_(data)
        , window_(&window)
        , state_(async_rma_state::pending)
        , callback_id_(invalid_callback_id) {

        // Register with progress engine
        callback_id_ = futures::progress_engine::instance().register_callback(
            [this]() { return this->make_progress(); }
        );
    }

    /// @brief Destructor - ensures operation completes
    ~async_put() {
        if (callback_id_ != invalid_callback_id) {
            if (state_.load() == async_rma_state::pending ||
                state_.load() == async_rma_state::initiated) {
                wait();
            }
            futures::progress_engine::instance().unregister_callback(callback_id_);
        }
    }

    // Non-copyable
    async_put(const async_put&) = delete;
    async_put& operator=(const async_put&) = delete;

    // Movable
    async_put(async_put&& other) noexcept
        : target_(other.target_)
        , offset_(other.offset_)
        , data_(other.data_)
        , window_(other.window_)
        , state_(other.state_.load())
        , callback_id_(invalid_callback_id)
        , result_(std::move(other.result_)) {
        // Unregister callback from moved-from object
        if (other.callback_id_ != invalid_callback_id) {
            futures::progress_engine::instance().unregister_callback(other.callback_id_);
        }
        other.window_ = nullptr;
        other.callback_id_ = invalid_callback_id;
        other.state_.store(async_rma_state::ready);

        // Re-register callback with new this pointer if still pending
        if ((state_.load() == async_rma_state::pending ||
             state_.load() == async_rma_state::initiated) && window_ != nullptr) {
            callback_id_ = futures::progress_engine::instance().register_callback(
                [this]() { return this->make_progress(); }
            );
        }
    }

    /// @brief Check if operation is complete
    [[nodiscard]] bool ready() const noexcept {
        return state_.load() != async_rma_state::pending &&
               state_.load() != async_rma_state::initiated;
    }

    /// @brief Wait for operation to complete
    void wait() {
        while (!ready()) {
            futures::progress_engine::instance().poll();
        }
    }

    /// @brief Get the result of the operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> get_result() {
        wait();
        return result_;
    }

private:
    /// @brief Make progress on the operation (non-blocking, request-based)
    /// @return true if operation still pending, false if complete
    bool make_progress() {
        auto current_state = state_.load();

        if (current_state == async_rma_state::ready ||
            current_state == async_rma_state::error) {
            return false;
        }

        if (!window_ || !window_->valid()) {
            result_ = status_code::invalid_state;
            state_.store(async_rma_state::error);
            return false;
        }

        if (current_state == async_rma_state::pending) {
            // Phase 1: Initiate the put operation
            result_ = rma::put(target_, offset_, data_, *window_);

            if (result_.has_value()) {
                state_.store(async_rma_state::initiated);
                return true;  // Still in progress, need flush
            } else {
                state_.store(async_rma_state::error);
                return false;
            }
        }

        if (current_state == async_rma_state::initiated) {
            // Phase 2: Flush to ensure completion
            auto flush_result = window_->flush(target_);
            if (!flush_result.has_value()) {
                result_ = flush_result.error();
                state_.store(async_rma_state::error);
            } else {
                state_.store(async_rma_state::ready);
            }
            return false;  // Operation complete (success or error)
        }

        return false;
    }

    rank_t target_;
    size_type offset_;
    std::span<const T> data_;
    memory_window* window_;
    std::atomic<async_rma_state> state_;
    size_type callback_id_;
    result<void> result_;
};

// ============================================================================
// Async Get
// ============================================================================

/// @brief Asynchronous get operation with progress engine integration
/// @tparam T Element type
template <typename T>
class async_get {
public:
    /// @brief Construct an async get operation
    /// @param target Target rank
    /// @param offset Offset in target window (bytes)
    /// @param buffer Buffer to receive data
    /// @param window Memory window
    async_get(rank_t target, size_type offset, std::span<T> buffer, memory_window& window)
        : target_(target)
        , offset_(offset)
        , buffer_(buffer)
        , window_(&window)
        , state_(async_rma_state::pending)
        , callback_id_(invalid_callback_id) {

        // Register with progress engine
        callback_id_ = futures::progress_engine::instance().register_callback(
            [this]() { return this->make_progress(); }
        );
    }

    /// @brief Destructor - ensures operation completes
    ~async_get() {
        if (callback_id_ != invalid_callback_id) {
            if (state_.load() == async_rma_state::pending ||
                state_.load() == async_rma_state::initiated) {
                wait();
            }
            futures::progress_engine::instance().unregister_callback(callback_id_);
        }
    }

    // Non-copyable
    async_get(const async_get&) = delete;
    async_get& operator=(const async_get&) = delete;

    // Movable
    async_get(async_get&& other) noexcept
        : target_(other.target_)
        , offset_(other.offset_)
        , buffer_(other.buffer_)
        , window_(other.window_)
        , state_(other.state_.load())
        , callback_id_(invalid_callback_id)
        , result_(std::move(other.result_)) {
        // Unregister callback from moved-from object
        if (other.callback_id_ != invalid_callback_id) {
            futures::progress_engine::instance().unregister_callback(other.callback_id_);
        }
        other.window_ = nullptr;
        other.callback_id_ = invalid_callback_id;
        other.state_.store(async_rma_state::ready);

        // Re-register callback with new this pointer if still pending
        if ((state_.load() == async_rma_state::pending ||
             state_.load() == async_rma_state::initiated) && window_ != nullptr) {
            callback_id_ = futures::progress_engine::instance().register_callback(
                [this]() { return this->make_progress(); }
            );
        }
    }

    /// @brief Check if operation is complete
    [[nodiscard]] bool ready() const noexcept {
        return state_.load() != async_rma_state::pending &&
               state_.load() != async_rma_state::initiated;
    }

    /// @brief Wait for operation to complete
    void wait() {
        while (!ready()) {
            futures::progress_engine::instance().poll();
        }
    }

    /// @brief Get the result of the operation
    /// @return Result containing the buffer span on success
    [[nodiscard]] result<std::span<T>> get_result() {
        wait();
        if (result_.has_error()) {
            return result_.error();
        }
        return buffer_;
    }

private:
    /// @brief Make progress on the operation (non-blocking, request-based)
    /// @return true if operation still pending, false if complete
    bool make_progress() {
        auto current_state = state_.load();

        if (current_state == async_rma_state::ready ||
            current_state == async_rma_state::error) {
            return false;
        }

        if (!window_ || !window_->valid()) {
            result_ = status_code::invalid_state;
            state_.store(async_rma_state::error);
            return false;
        }

        if (current_state == async_rma_state::pending) {
            // Phase 1: Initiate the get operation
            result_ = rma::get(target_, offset_, buffer_, *window_);

            if (result_.has_value()) {
                state_.store(async_rma_state::initiated);
                return true;  // Still in progress, need flush
            } else {
                state_.store(async_rma_state::error);
                return false;
            }
        }

        if (current_state == async_rma_state::initiated) {
            // Phase 2: Flush to ensure completion
            auto flush_result = window_->flush_local(target_);
            if (!flush_result.has_value()) {
                result_ = flush_result.error();
                state_.store(async_rma_state::error);
            } else {
                state_.store(async_rma_state::ready);
            }
            return false;  // Operation complete (success or error)
        }

        return false;
    }

    rank_t target_;
    size_type offset_;
    std::span<T> buffer_;
    memory_window* window_;
    std::atomic<async_rma_state> state_;
    size_type callback_id_;
    result<void> result_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create an async put operation
/// @tparam T Element type
/// @param target Target rank
/// @param offset Offset in target window (bytes)
/// @param data Data to put
/// @param window Memory window
/// @return Async put operation
template <typename T>
[[nodiscard]] async_put<T> async_put_to(rank_t target, size_type offset,
                                        std::span<const T> data, memory_window& window) {
    return async_put<T>(target, offset, data, window);
}

/// @brief Create an async get operation
/// @tparam T Element type
/// @param target Target rank
/// @param offset Offset in target window (bytes)
/// @param buffer Buffer to receive data
/// @param window Memory window
/// @return Async get operation
template <typename T>
[[nodiscard]] async_get<T> async_get_from(rank_t target, size_type offset,
                                          std::span<T> buffer, memory_window& window) {
    return async_get<T>(target, offset, buffer, window);
}

// ============================================================================
// Completion Callback Support
// ============================================================================

/// @brief Async put with completion callback
/// @tparam T Element type
/// @tparam Callback Callback type (invoked with result<void>)
template <typename T, typename Callback>
class async_put_with_callback {
public:
    /// @brief Construct with completion callback
    async_put_with_callback(rank_t target, size_type offset, std::span<const T> data,
                            memory_window& window, Callback callback)
        : put_(target, offset, data, window)
        , callback_(std::move(callback))
        , callback_invoked_(false) {}

    /// @brief Check if complete and invoke callback if so
    void check_completion() {
        if (put_.ready() && !callback_invoked_) {
            callback_invoked_ = true;
            callback_(put_.get_result());
        }
    }

    /// @brief Check if operation is ready
    [[nodiscard]] bool ready() const noexcept {
        return put_.ready();
    }

    /// @brief Wait for completion
    void wait() {
        put_.wait();
        check_completion();
    }

private:
    async_put<T> put_;
    Callback callback_;
    bool callback_invoked_;
};

/// @brief Create async put with completion callback
template <typename T, typename Callback>
[[nodiscard]] auto async_put_then(rank_t target, size_type offset,
                                  std::span<const T> data, memory_window& window,
                                  Callback&& callback) {
    return async_put_with_callback<T, std::decay_t<Callback>>(
        target, offset, data, window, std::forward<Callback>(callback));
}

}  // namespace dtl::rma
