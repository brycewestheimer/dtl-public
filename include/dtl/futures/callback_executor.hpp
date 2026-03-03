// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file callback_executor.hpp
/// @brief Isolated callback executor for futures continuations
/// @details Provides a separate execution context for user callbacks to prevent
///          long-running callbacks from blocking progress on unrelated futures.
///
///          **Integration with `.then()` continuations:**
///          The `.then()` method (continuation.hpp) executes callbacks inline on
///          the progress engine thread for simplicity and low latency. If a
///          callback is long-running and may block progress, users should manually
///          dispatch heavy work through `global_callback_executor().enqueue(...)`.
///
///          Example:
///          @code
///          future.then([](int value) {
///              // Fast path: quick transform — runs inline, no overhead
///              return value * 2;
///          });
///
///          future.then([](int value) {
///              // Heavy path: offload to callback executor
///              global_callback_executor().enqueue([value] {
///                  heavy_computation(value);
///              });
///          });
///          @endcode
///
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dtl::futures {

// ============================================================================
// Callback Executor Configuration
// ============================================================================

/// @brief Execution mode for callback executor
enum class executor_mode {
    /// @brief Inline execution (callbacks run on progress thread)
    /// @details Simple but long callbacks block progress
    inline_mode,

    /// @brief Single-threaded queue (callbacks run on dedicated thread)
    /// @details Prevents blocking but limited throughput
    single_thread,

    /// @brief Thread pool (callbacks run on pool threads)
    /// @details Best throughput but more resource usage
    thread_pool
};

/// @brief Configuration for callback executor
struct executor_config {
    /// @brief Execution mode
    executor_mode mode = executor_mode::single_thread;

    /// @brief Number of threads for thread_pool mode (ignored for other modes)
    /// @details 0 = use hardware_concurrency()
    size_type thread_count = 0;

    /// @brief Maximum queue depth before blocking (0 = unlimited)
    size_type max_queue_depth = 0;

    /// @brief Timeout for draining queued callbacks on shutdown
    std::chrono::milliseconds shutdown_timeout{5000};

    /// @brief Create default configuration (single-threaded queue)
    [[nodiscard]] static executor_config defaults() {
        return executor_config{};
    }

    /// @brief Create inline configuration (no isolation)
    [[nodiscard]] static executor_config inline_execution() {
        executor_config cfg;
        cfg.mode = executor_mode::inline_mode;
        return cfg;
    }

    /// @brief Create thread pool configuration
    [[nodiscard]] static executor_config thread_pool_execution(size_type threads = 0) {
        executor_config cfg;
        cfg.mode = executor_mode::thread_pool;
        cfg.thread_count = threads;
        return cfg;
    }
};

// ============================================================================
// Callback Executor
// ============================================================================

/// @brief Isolated executor for callback/continuation execution
/// @details Runs user callbacks on a separate thread or thread pool to prevent
///          them from blocking progress engine polling. This addresses the
///          KNOWN_ISSUES limitation where long-running callbacks delay other
///          completions.
///
/// Thread safety:
/// - enqueue() is thread-safe and can be called from any thread
/// - drain() blocks until all queued callbacks complete
/// - shutdown() stops accepting new callbacks and drains remaining work
class callback_executor {
public:
    using callback_type = std::function<void()>;

    /// @brief Construct executor with configuration
    /// @param config Execution configuration
    explicit callback_executor(executor_config config = executor_config::defaults())
        : config_(config)
        , running_(true)
        , pending_count_(0)
        , total_executed_(0)
        , total_enqueued_(0) {
        start_workers();
    }

    /// @brief Destructor - shuts down and drains callbacks
    ~callback_executor() {
        shutdown();
    }

    // Non-copyable, non-movable
    callback_executor(const callback_executor&) = delete;
    callback_executor& operator=(const callback_executor&) = delete;
    callback_executor(callback_executor&&) = delete;
    callback_executor& operator=(callback_executor&&) = delete;

    /// @brief Enqueue a callback for execution
    /// @param callback The callback to execute
    /// @return true if enqueued, false if executor is shut down
    bool enqueue(callback_type callback) {
        if (!callback) return true;  // Null callback is a no-op

        // Fast path: inline mode executes immediately
        if (config_.mode == executor_mode::inline_mode) {
            ++total_enqueued_;
            try {
                callback();
            } catch (...) {
                // Swallow exceptions in callbacks
            }
            ++total_executed_;
            return true;
        }

        // Queue-based execution
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            if (!running_) {
                return false;  // Executor is shut down
            }

            // Wait if queue is at capacity
            if (config_.max_queue_depth > 0) {
                queue_not_full_.wait(lock, [this] {
                    return queue_.size() < config_.max_queue_depth || !running_;
                });
                if (!running_) return false;
            }

            queue_.push_back(std::move(callback));
            ++pending_count_;
            ++total_enqueued_;
        }

        queue_not_empty_.notify_one();
        return true;
    }

    /// @brief Drain all queued callbacks synchronously
    /// @details Blocks until all currently queued callbacks complete.
    ///          New callbacks enqueued during drain will also be processed.
    void drain() {
        if (config_.mode == executor_mode::inline_mode) {
            return;  // Nothing to drain
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        drained_.wait(lock, [this] {
            return pending_count_.load(std::memory_order_acquire) == 0;
        });
    }

    /// @brief Shutdown the executor
    /// @details Stops accepting new callbacks and drains remaining work.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!running_) return;  // Already shut down
            running_ = false;
        }

        queue_not_empty_.notify_all();
        queue_not_full_.notify_all();

        // Join worker threads
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    /// @brief Check if executor is running
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// @brief Get number of pending callbacks
    [[nodiscard]] size_type pending_count() const noexcept {
        return pending_count_.load(std::memory_order_acquire);
    }

    /// @brief Get total callbacks executed
    [[nodiscard]] size_type total_executed() const noexcept {
        return total_executed_.load(std::memory_order_acquire);
    }

    /// @brief Get total callbacks enqueued
    [[nodiscard]] size_type total_enqueued() const noexcept {
        return total_enqueued_.load(std::memory_order_acquire);
    }

    /// @brief Get execution mode
    [[nodiscard]] executor_mode mode() const noexcept {
        return config_.mode;
    }

private:
    void start_workers() {
        if (config_.mode == executor_mode::inline_mode) {
            return;  // No workers needed
        }

        size_type num_workers = 1;
        if (config_.mode == executor_mode::thread_pool) {
            num_workers = config_.thread_count > 0
                ? config_.thread_count
                : std::max(1u, std::thread::hardware_concurrency());
        }

        workers_.reserve(num_workers);
        for (size_type i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void worker_loop() {
        while (true) {
            callback_type callback;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_not_empty_.wait(lock, [this] {
                    return !queue_.empty() || !running_;
                });

                if (!running_ && queue_.empty()) {
                    return;  // Shutdown and queue empty
                }

                if (!queue_.empty()) {
                    callback = std::move(queue_.front());
                    queue_.pop_front();
                    queue_not_full_.notify_one();
                }
            }

            if (callback) {
                try {
                    callback();
                } catch (...) {
                    // Swallow exceptions in callbacks
                }

                ++total_executed_;
                if (--pending_count_ == 0) {
                    drained_.notify_all();
                }
            }
        }
    }

    executor_config config_;
    std::atomic<bool> running_;
    std::atomic<size_type> pending_count_;
    std::atomic<size_type> total_executed_;
    std::atomic<size_type> total_enqueued_;

    std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;
    std::condition_variable drained_;
    std::deque<callback_type> queue_;

    std::vector<std::thread> workers_;
};

// ============================================================================
// Global Callback Executor Access
// ============================================================================

/// @brief Get the global callback executor instance
/// @return Reference to the global callback executor
/// @details The executor is created on first access with default configuration.
///          Use set_global_callback_executor() before first access to customize.
[[nodiscard]] inline callback_executor& global_callback_executor() {
    static callback_executor executor{executor_config::defaults()};
    return executor;
}

}  // namespace dtl::futures
