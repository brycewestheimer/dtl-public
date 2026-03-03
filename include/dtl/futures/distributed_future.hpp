// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_future.hpp
/// @brief Distributed future implementation
/// @details Provides futures for asynchronous distributed computations.
/// @since 0.1.0
/// @note Updated in 1.3.0: Added configurable timeouts and diagnostic integration.

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/distributed_future.hpp>
#include <dtl/backend/concepts/event.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/futures/diagnostics.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <stdexcept>
#include <variant>

namespace dtl::futures {

// ============================================================================
// Shared State
// ============================================================================

/// @brief Shared state for distributed futures
/// @tparam T Value type
template <typename T>
class shared_state {
public:
    using value_type = T;

    /// @brief Default constructor
    shared_state() : ready_(false) {}

    /// @brief Set the diagnostic tracking ID
    /// @param id ID returned from diagnostic_collector::register_future()
    void set_diagnostic_id(size_type id) noexcept { diagnostic_id_ = id; }

    /// @brief Get the diagnostic tracking ID (0 = not tracked)
    [[nodiscard]] size_type diagnostic_id() const noexcept { return diagnostic_id_; }

    /// @brief Set the value (makes future ready)
    /// @param value The value to set
    void set_value(T value) {
        if (diagnostic_id_ != 0) {
            diagnostic_collector::instance().unregister_future(diagnostic_id_);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            value_.emplace(std::move(value));
            ready_ = true;
        }
        cv_.notify_all();
    }

    /// @brief Set an error
    /// @param error The error status
    void set_error(status error) {
        if (diagnostic_id_ != 0) {
            diagnostic_collector::instance().unregister_future(diagnostic_id_);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_.emplace(std::move(error));
            ready_ = true;
        }
        cv_.notify_all();
    }

    /// @brief Wait for the result to be ready
    /// @details Integrates with the progress engine to avoid deadlock.
    ///          Polls progress callbacks between short timed waits on the
    ///          condition variable, ensuring continuations and combinators
    ///          (when_all, when_any) can complete even when called from the
    ///          same thread.
    ///
    ///          Uses configurable timeout from global_timeout_config().
    ///          Set DTL_CI_MODE environment variable for CI-specific timeouts.
    /// @throws timeout_exception if wait exceeds configured timeout
    void wait() {
        // Fast path: already ready
        if (ready_.load(std::memory_order_acquire)) return;

        auto timeout = effective_wait_timeout();
        // timeout of 0 means no timeout
        auto deadline = timeout.count() > 0
            ? std::chrono::steady_clock::now() + timeout
            : std::chrono::steady_clock::time_point::max();

        const auto& config = global_timeout_config();

        while (!ready_.load(std::memory_order_acquire)) {
            // Drive the progress engine to advance pending operations
            futures::progress_engine::instance().poll();

            // Short wait on condition variable to avoid busy-spin
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (cv_.wait_for(lock, config.poll_interval,
                                 [this] { return ready_.load(std::memory_order_acquire); })) {
                    return;
                }
            }

            if (std::chrono::steady_clock::now() > deadline) {
                if (config.enable_timeout_diagnostics) {
                    auto diag = diagnostic_collector::instance().get_diagnostics();
                    if (config.on_timeout_callback) {
                        config.on_timeout_callback(diag.to_string());
                    }
                    throw timeout_exception(
                        "Future wait exceeded timeout (" +
                        std::to_string(timeout.count()) + "ms)",
                        std::move(diag));
                }
                throw std::runtime_error(
                    "Future wait exceeded maximum timeout (" +
                    std::to_string(timeout.count()) + "ms)");
            }
        }
    }

    /// @brief Wait with timeout
    /// @param timeout Maximum time to wait
    /// @return future_status indicating result
    template <typename Rep, typename Period>
    future_status wait_for(std::chrono::duration<Rep, Period> timeout) {
        // Fast path
        if (ready_.load(std::memory_order_acquire)) {
            return error_.has_value() ? future_status::error : future_status::ready;
        }

        const auto& config = global_timeout_config();
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!ready_.load(std::memory_order_acquire)) {
            futures::progress_engine::instance().poll();

            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (cv_.wait_for(lock, config.poll_interval,
                                 [this] { return ready_.load(std::memory_order_acquire); })) {
                    return error_.has_value() ? future_status::error : future_status::ready;
                }
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                if (diagnostic_id_ != 0) {
                    diagnostic_collector::instance().record_timeout(diagnostic_id_);
                }
                return future_status::timeout;
            }
        }
        return error_.has_value() ? future_status::error : future_status::ready;
    }

    /// @brief Check if ready
    [[nodiscard]] bool is_ready() const noexcept {
        return ready_.load();
    }

    /// @brief Get the value (throws if error)
    [[nodiscard]] T get() {
        wait();
        if (error_.has_value()) {
            throw std::runtime_error(error_->message());
        }
        return std::move(*value_);
    }

    /// @brief Get result (value or error)
    [[nodiscard]] result<T> get_result() {
        wait();
        if (error_.has_value()) {
            return make_error(error_->code(), error_->message());
        }
        return std::move(*value_);
    }

    /// @brief Check if has error
    [[nodiscard]] bool has_error() const noexcept {
        return error_.has_value();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> ready_;
    std::optional<T> value_;
    std::optional<status> error_;
    size_type diagnostic_id_ = 0;
};

/// @brief Shared state specialization for void
template <>
class shared_state<void> {
public:
    using value_type = void;

    shared_state() : ready_(false) {}

    /// @brief Set the diagnostic tracking ID
    void set_diagnostic_id(size_type id) noexcept { diagnostic_id_ = id; }

    /// @brief Get the diagnostic tracking ID (0 = not tracked)
    [[nodiscard]] size_type diagnostic_id() const noexcept { return diagnostic_id_; }

    void set_value() {
        if (diagnostic_id_ != 0) {
            diagnostic_collector::instance().unregister_future(diagnostic_id_);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = true;
        }
        cv_.notify_all();
    }

    void set_error(status error) {
        if (diagnostic_id_ != 0) {
            diagnostic_collector::instance().unregister_future(diagnostic_id_);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error_.emplace(std::move(error));
            ready_ = true;
        }
        cv_.notify_all();
    }

    void wait() {
        // Fast path: already ready
        if (ready_.load(std::memory_order_acquire)) return;

        auto timeout = effective_wait_timeout();
        auto deadline = timeout.count() > 0
            ? std::chrono::steady_clock::now() + timeout
            : std::chrono::steady_clock::time_point::max();

        const auto& config = global_timeout_config();

        while (!ready_.load(std::memory_order_acquire)) {
            futures::progress_engine::instance().poll();

            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (cv_.wait_for(lock, config.poll_interval,
                                 [this] { return ready_.load(std::memory_order_acquire); })) {
                    return;
                }
            }

            if (std::chrono::steady_clock::now() > deadline) {
                if (config.enable_timeout_diagnostics) {
                    auto diag = diagnostic_collector::instance().get_diagnostics();
                    if (config.on_timeout_callback) {
                        config.on_timeout_callback(diag.to_string());
                    }
                    throw timeout_exception(
                        "Future wait exceeded timeout (" +
                        std::to_string(timeout.count()) + "ms)",
                        std::move(diag));
                }
                throw std::runtime_error(
                    "Future wait exceeded maximum timeout (" +
                    std::to_string(timeout.count()) + "ms)");
            }
        }
    }

    template <typename Rep, typename Period>
    future_status wait_for(std::chrono::duration<Rep, Period> timeout) {
        // Fast path
        if (ready_.load(std::memory_order_acquire)) {
            return error_.has_value() ? future_status::error : future_status::ready;
        }

        const auto& config = global_timeout_config();
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!ready_.load(std::memory_order_acquire)) {
            futures::progress_engine::instance().poll();

            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (cv_.wait_for(lock, config.poll_interval,
                                 [this] { return ready_.load(std::memory_order_acquire); })) {
                    return error_.has_value() ? future_status::error : future_status::ready;
                }
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                if (diagnostic_id_ != 0) {
                    diagnostic_collector::instance().record_timeout(diagnostic_id_);
                }
                return future_status::timeout;
            }
        }
        return error_.has_value() ? future_status::error : future_status::ready;
    }

    [[nodiscard]] bool is_ready() const noexcept {
        return ready_.load();
    }

    void get() {
        wait();
        if (error_.has_value()) {
            throw std::runtime_error(error_->message());
        }
    }

    [[nodiscard]] result<void> get_result() {
        wait();
        if (error_.has_value()) {
            return make_error(error_->code(), error_->message());
        }
        return {};
    }

    [[nodiscard]] bool has_error() const noexcept {
        return error_.has_value();
    }

private:

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> ready_;
    std::optional<status> error_;
    size_type diagnostic_id_ = 0;
};

// ============================================================================
// Distributed Future
// ============================================================================

/// @brief Future representing an asynchronous distributed computation
/// @tparam T Value type
template <typename T>
class distributed_future {
public:
    using value_type = T;

    /// @brief Default constructor (invalid future)
    distributed_future() = default;

    /// @brief Construct with shared state
    explicit distributed_future(std::shared_ptr<shared_state<T>> state)
        : state_(std::move(state)) {}

    /// @brief Check if future is valid
    [[nodiscard]] bool valid() const noexcept {
        return state_ != nullptr;
    }

    /// @brief Wait for result and return it
    /// @return The computed value
    /// @throws std::runtime_error if error occurred
    [[nodiscard]] T get() {
        if (!valid()) {
            throw std::runtime_error("get() called on invalid future");
        }
        auto s = std::move(state_);
        return s->get();
    }

    /// @brief Wait for result (non-throwing)
    /// @return Result containing value or error
    [[nodiscard]] result<T> get_result() {
        if (!valid()) {
            return make_error(status_code::invalid_state, "Invalid future");
        }
        auto s = std::move(state_);
        return s->get_result();
    }

    /// @brief Wait for result to become available
    void wait() {
        if (valid()) {
            state_->wait();
        }
    }

    /// @brief Wait with timeout
    /// @param timeout Maximum time to wait
    /// @return future_status indicating result
    template <typename Rep, typename Period>
    [[nodiscard]] future_status wait_for(std::chrono::duration<Rep, Period> timeout) {
        if (!valid()) {
            return future_status::error;
        }
        return state_->wait_for(timeout);
    }

    /// @brief Check if result is ready
    [[nodiscard]] bool is_ready() const noexcept {
        return valid() && state_->is_ready();
    }

    /// @brief Add continuation
    /// @tparam F Callable type
    /// @param func Function to call when ready
    /// @return Future for continuation result
    template <typename F>
    auto then(F&& func);  // Defined in continuation.hpp

    /// @brief Access shared state (for internal use by combinators)
    /// @return Shared pointer to state
    [[nodiscard]] std::shared_ptr<shared_state<T>> get_state() const noexcept {
        return state_;
    }

private:
    std::shared_ptr<shared_state<T>> state_;

    // Friend declarations for combinators that need state access
    template <typename U, typename Handler>
        requires std::invocable<Handler, const status&>
    friend auto on_error(distributed_future<U> future, Handler&& handler);

    template <typename U>
    friend distributed_future<U> flatten(distributed_future<distributed_future<U>> future);
};

/// @brief Specialization for void
template <>
class distributed_future<void> {
public:
    using value_type = void;

    distributed_future() = default;

    explicit distributed_future(std::shared_ptr<shared_state<void>> state)
        : state_(std::move(state)) {}

    [[nodiscard]] bool valid() const noexcept {
        return state_ != nullptr;
    }

    void get() {
        if (!valid()) {
            throw std::runtime_error("get() called on invalid future");
        }
        auto s = std::move(state_);
        s->get();
    }

    [[nodiscard]] result<void> get_result() {
        if (!valid()) {
            return make_error(status_code::invalid_state, "Invalid future");
        }
        auto s = std::move(state_);
        return s->get_result();
    }

    void wait() {
        if (valid()) {
            state_->wait();
        }
    }

    template <typename Rep, typename Period>
    [[nodiscard]] future_status wait_for(std::chrono::duration<Rep, Period> timeout) {
        if (!valid()) {
            return future_status::error;
        }
        return state_->wait_for(timeout);
    }

    [[nodiscard]] bool is_ready() const noexcept {
        return valid() && state_->is_ready();
    }

    template <typename F>
    auto then(F&& func);  // Defined in continuation.hpp

    /// @brief Access shared state (for internal use by combinators)
    [[nodiscard]] std::shared_ptr<shared_state<void>> get_state() const noexcept {
        return state_;
    }

private:
    std::shared_ptr<shared_state<void>> state_;

    // Friend declarations for combinators
    template <typename U, typename Handler>
        requires std::invocable<Handler, const status&>
    friend auto on_error(distributed_future<U> future, Handler&& handler);
};

// ============================================================================
// Promise
// ============================================================================

/// @brief Promise for setting the value of a distributed future
/// @tparam T Value type
template <typename T>
class distributed_promise {
public:
    /// @brief Default constructor
    /// @details Registers with diagnostic_collector for lifecycle tracking
    distributed_promise()
        : state_(std::make_shared<shared_state<T>>()) {
        auto id = diagnostic_collector::instance().register_future("distributed_promise");
        state_->set_diagnostic_id(id);
    }

    /// @brief Get the associated future
    [[nodiscard]] distributed_future<T> get_future() {
        return distributed_future<T>(state_);
    }

    /// @brief Set the value
    /// @param value The value to set
    void set_value(T value) {
        state_->set_value(std::move(value));
    }

    /// @brief Set an error
    /// @param error The error status
    void set_error(status error) {
        state_->set_error(std::move(error));
    }

private:
    std::shared_ptr<shared_state<T>> state_;
};

/// @brief Promise specialization for void
template <>
class distributed_promise<void> {
public:
    /// @details Registers with diagnostic_collector for lifecycle tracking
    distributed_promise()
        : state_(std::make_shared<shared_state<void>>()) {
        auto id = diagnostic_collector::instance().register_future("distributed_promise<void>");
        state_->set_diagnostic_id(id);
    }

    [[nodiscard]] distributed_future<void> get_future() {
        return distributed_future<void>(state_);
    }

    void set_value() {
        state_->set_value();
    }

    void set_error(status error) {
        state_->set_error(std::move(error));
    }

private:
    std::shared_ptr<shared_state<void>> state_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a ready future with a value
/// @tparam T Value type
/// @param value The value
/// @return Ready future
template <typename T>
[[nodiscard]] distributed_future<std::decay_t<T>> make_ready_distributed_future(T&& value) {
    using V = std::decay_t<T>;
    auto state = std::make_shared<shared_state<V>>();
    state->set_value(std::forward<T>(value));
    return distributed_future<V>(std::move(state));
}

/// @brief Create a ready void future
/// @return Ready void future
[[nodiscard]] inline distributed_future<void> make_ready_distributed_future() {
    auto state = std::make_shared<shared_state<void>>();
    state->set_value();
    return distributed_future<void>(std::move(state));
}

/// @brief Create a failed future
/// @tparam T Expected value type
/// @param error The error
/// @return Failed future
template <typename T>
[[nodiscard]] distributed_future<T> make_failed_distributed_future(status error) {
    auto state = std::make_shared<shared_state<T>>();
    state->set_error(std::move(error));
    return distributed_future<T>(std::move(state));
}

}  // namespace dtl::futures
