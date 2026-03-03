// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file background_progress.hpp
/// @brief Optional background progress mode for automatic progress advancement
/// @details Provides a background thread that automatically polls the progress
///          engine, eliminating the need for explicit poll() calls.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/futures/diagnostics.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

namespace dtl::futures {

// ============================================================================
// Background Progress Configuration
// ============================================================================

/// @brief Progress mode for the progress engine
enum class progress_mode {
    /// @brief Explicit polling only (default)
    /// @details User must call poll() or wait functions to advance progress
    explicit_mode,

    /// @brief Background thread automatically polls
    /// @details A dedicated thread polls the progress engine periodically
    background
};

/// @brief Configuration for background progress mode
struct background_progress_config {
    /// @brief Progress mode
    progress_mode mode = progress_mode::explicit_mode;

    /// @brief Poll interval for background mode
    std::chrono::microseconds poll_interval{100};

    /// @brief Use adaptive polling (reduce frequency when idle)
    bool adaptive_polling = true;

    /// @brief Maximum poll interval when adaptive polling is enabled
    std::chrono::milliseconds max_poll_interval{10};

    /// @brief Backoff multiplier for adaptive polling
    double backoff_multiplier = 1.5;

    /// @brief Shutdown timeout
    std::chrono::milliseconds shutdown_timeout{5000};

    /// @brief Create default configuration (explicit polling)
    [[nodiscard]] static background_progress_config defaults() {
        return background_progress_config{};
    }

    /// @brief Create background mode configuration
    [[nodiscard]] static background_progress_config background_mode() {
        background_progress_config cfg;
        cfg.mode = progress_mode::background;
        return cfg;
    }

    /// @brief Create aggressive background polling configuration
    [[nodiscard]] static background_progress_config aggressive_background() {
        background_progress_config cfg;
        cfg.mode = progress_mode::background;
        cfg.poll_interval = std::chrono::microseconds{10};
        cfg.adaptive_polling = false;
        return cfg;
    }
};

// ============================================================================
// Background Progress Controller
// ============================================================================

/// @brief Controls background progress thread
/// @details Manages a background thread that automatically polls the progress
///          engine. This addresses the KNOWN_ISSUES limitation where progress
///          may not advance without explicit polling.
///
/// Thread safety:
/// - All methods are thread-safe
/// - start() and stop() are idempotent
/// - Multiple calls to start()/stop() are safe
class background_progress_controller {
public:
    /// @brief Get the singleton instance
    [[nodiscard]] static background_progress_controller& instance() noexcept {
        static background_progress_controller controller;
        return controller;
    }

    /// @brief Start background progress (if not already running)
    /// @param config Configuration for background progress
    void start(background_progress_config config = background_progress_config::background_mode()) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (running_) {
            return;  // Already running
        }

        config_ = config;
        running_ = true;
        should_stop_ = false;

        diagnostic_collector::instance().set_background_progress(true, true);

        std::promise<void> done_promise;
        worker_done_ = done_promise.get_future();

        worker_ = std::thread([this, p = std::move(done_promise)]() mutable {
            background_loop();
            p.set_value();
        });
    }

    /// @brief Stop background progress
    /// @details Respects shutdown_timeout configuration. If the worker thread
    ///          does not stop within the timeout, logs a warning and detaches.
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                return;  // Not running
            }
            should_stop_ = true;
        }

        wake_cv_.notify_all();

        if (worker_.joinable()) {
            if (worker_done_.valid()) {
                auto wait_status = worker_done_.wait_for(config_.shutdown_timeout);
                if (wait_status == std::future_status::timeout) {
                    // Worker thread did not stop within timeout — detach
                    worker_.detach();
                } else {
                    worker_.join();
                }
            } else {
                worker_.join();
            }
        }

        running_ = false;
        diagnostic_collector::instance().set_background_progress(false, false);
    }

    /// @brief Check if background progress is running
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// @brief Get current configuration
    [[nodiscard]] background_progress_config config() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

    /// @brief Get number of background polls performed
    [[nodiscard]] size_type poll_count() const noexcept {
        return poll_count_.load(std::memory_order_relaxed);
    }

    /// @brief Wake background thread for immediate poll
    void wake() {
        wake_cv_.notify_one();
    }

    /// @brief Destructor - stops background thread
    ~background_progress_controller() {
        stop();
    }

private:
    background_progress_controller() = default;

    // Non-copyable, non-movable
    background_progress_controller(const background_progress_controller&) = delete;
    background_progress_controller& operator=(const background_progress_controller&) = delete;

    void background_loop() {
        auto current_interval = config_.poll_interval;
        size_type idle_count = 0;

        while (!should_stop_.load(std::memory_order_acquire)) {
            // Poll the progress engine
            size_type progress = progress_engine::instance().poll();
            ++poll_count_;
            diagnostic_collector::instance().record_poll();

            // Adaptive polling: increase interval when idle
            if (config_.adaptive_polling) {
                if (progress == 0 && !progress_engine::instance().has_pending()) {
                    ++idle_count;
                    if (idle_count > 10) {
                        // Increase interval up to max
                        auto new_interval = std::chrono::duration_cast<std::chrono::microseconds>(
                            current_interval * config_.backoff_multiplier);
                        current_interval = std::min(
                            new_interval,
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                config_.max_poll_interval));
                    }
                } else {
                    // Reset to base interval when work is found
                    idle_count = 0;
                    current_interval = config_.poll_interval;
                }
            }

            // Wait for next poll interval or wake signal
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_cv_.wait_for(lock, current_interval, [this] {
                    return should_stop_.load(std::memory_order_acquire);
                });
            }
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable wake_cv_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<size_type> poll_count_{0};
    background_progress_config config_;
    std::thread worker_;
    std::future<void> worker_done_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Start background progress mode
/// @param config Configuration for background progress
inline void start_background_progress(
    background_progress_config config = background_progress_config::background_mode()) {
    background_progress_controller::instance().start(std::move(config));
}

/// @brief Stop background progress mode
inline void stop_background_progress() {
    background_progress_controller::instance().stop();
}

/// @brief Check if background progress is active
[[nodiscard]] inline bool is_background_progress_enabled() {
    return background_progress_controller::instance().is_running();
}

/// @brief RAII guard for background progress mode
/// @details Starts background progress on construction, stops on destruction
class scoped_background_progress {
public:
    /// @brief Start background progress
    explicit scoped_background_progress(
        background_progress_config config = background_progress_config::background_mode()) {
        background_progress_controller::instance().start(std::move(config));
    }

    /// @brief Stop background progress
    ~scoped_background_progress() {
        background_progress_controller::instance().stop();
    }

    // Non-copyable, non-movable
    scoped_background_progress(const scoped_background_progress&) = delete;
    scoped_background_progress& operator=(const scoped_background_progress&) = delete;
    scoped_background_progress(scoped_background_progress&&) = delete;
    scoped_background_progress& operator=(scoped_background_progress&&) = delete;
};

}  // namespace dtl::futures
