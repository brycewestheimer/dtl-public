// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file diagnostics.hpp
/// @brief Futures diagnostics and timeout configuration
/// @details Provides diagnostic information for debugging deadlocks, hangs,
///          and configurable timeout settings for futures operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace dtl::futures {

// ============================================================================
// Timeout Configuration
// ============================================================================

/// @brief Configuration for futures timeout behavior
struct timeout_config {
    /// @brief Default wait timeout for future.wait()
    /// @details 0 = no timeout (wait indefinitely)
    std::chrono::milliseconds default_wait_timeout{30000};

    /// @brief Timeout for CI/test environments
    /// @details Applied when DTL_CI_MODE environment variable is set
    std::chrono::milliseconds ci_wait_timeout{30000};

    /// @brief Interval between progress polls during wait
    std::chrono::milliseconds poll_interval{1};

    /// @brief Enable diagnostic logging on timeout
    bool enable_timeout_diagnostics = true;

    /// @brief Diagnostic callback invoked on timeout
    /// @details Called with diagnostic message when a wait times out
    std::function<void(const std::string&)> on_timeout_callback = nullptr;

    /// @brief Create default timeout configuration
    [[nodiscard]] static timeout_config defaults() {
        return timeout_config{};
    }

    /// @brief Create strict timeout configuration for CI
    [[nodiscard]] static timeout_config ci_mode() {
        timeout_config cfg;
        cfg.default_wait_timeout = std::chrono::milliseconds{30000};
        cfg.ci_wait_timeout = std::chrono::milliseconds{30000};
        cfg.enable_timeout_diagnostics = true;
        return cfg;
    }

    /// @brief Create lenient timeout configuration
    [[nodiscard]] static timeout_config lenient() {
        timeout_config cfg;
        cfg.default_wait_timeout = std::chrono::milliseconds{300000};  // 5 minutes
        cfg.ci_wait_timeout = std::chrono::milliseconds{60000};
        return cfg;
    }

    /// @brief Create configuration with no timeout
    [[nodiscard]] static timeout_config no_timeout() {
        timeout_config cfg;
        cfg.default_wait_timeout = std::chrono::milliseconds{0};
        cfg.ci_wait_timeout = std::chrono::milliseconds{0};
        return cfg;
    }
};

// ============================================================================
// Diagnostic State
// ============================================================================

/// @brief Diagnostic information about a pending future
struct pending_future_info {
    /// @brief Unique identifier for the future
    size_type id = 0;

    /// @brief Time when the future was created
    std::chrono::steady_clock::time_point created_at;

    /// @brief Description of the operation (if available)
    std::string description;

    /// @brief Whether waiting is in progress
    bool is_waiting = false;
};

/// @brief Diagnostic snapshot of the progress engine state
struct progress_diagnostics {
    /// @brief Timestamp of the diagnostic snapshot
    std::chrono::steady_clock::time_point snapshot_time;

    /// @brief Number of pending callbacks
    size_type pending_callback_count = 0;

    /// @brief Number of pending CUDA events (if CUDA enabled)
    size_type pending_cuda_event_count = 0;

    /// @brief Number of pending futures
    size_type pending_future_count = 0;

    /// @brief Total polls since engine creation
    size_type total_polls = 0;

    /// @brief Timestamp of last poll
    std::chrono::steady_clock::time_point last_poll_time;

    /// @brief Whether background progress is enabled
    bool background_progress_enabled = false;

    /// @brief Whether background thread is running
    bool background_thread_running = false;

    /// @brief Information about pending futures
    std::vector<pending_future_info> pending_futures;

    /// @brief Format diagnostics as human-readable string
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "=== DTL Futures Diagnostics ===\n";
        oss << "Snapshot time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
            snapshot_time.time_since_epoch()).count() << "ms since epoch\n";
        oss << "Pending callbacks: " << pending_callback_count << "\n";
        oss << "Pending CUDA events: " << pending_cuda_event_count << "\n";
        oss << "Pending futures: " << pending_future_count << "\n";
        oss << "Total polls: " << total_polls << "\n";

        auto since_last_poll = snapshot_time - last_poll_time;
        oss << "Time since last poll: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(since_last_poll).count()
            << "ms\n";

        oss << "Background progress: "
            << (background_progress_enabled ? "enabled" : "disabled") << "\n";
        if (background_progress_enabled) {
            oss << "Background thread: "
                << (background_thread_running ? "running" : "stopped") << "\n";
        }

        if (!pending_futures.empty()) {
            oss << "\nPending futures:\n";
            for (const auto& future : pending_futures) {
                auto age = snapshot_time - future.created_at;
                oss << "  - ID " << future.id << ": "
                    << (future.description.empty() ? "(no description)" : future.description)
                    << " (age: " << std::chrono::duration_cast<std::chrono::milliseconds>(age).count()
                    << "ms, waiting: " << (future.is_waiting ? "yes" : "no") << ")\n";
            }
        }

        oss << "================================\n";
        return oss.str();
    }
};

// ============================================================================
// Diagnostic Collector
// ============================================================================

/// @brief Singleton collector for futures diagnostics
/// @details Tracks registered futures and provides diagnostic snapshots.
///          This is opt-in and only active when diagnostics are enabled.
class diagnostic_collector {
public:
    /// @brief Get the singleton instance
    [[nodiscard]] static diagnostic_collector& instance() noexcept {
        static diagnostic_collector collector;
        return collector;
    }

    /// @brief Register a new future for tracking
    /// @param description Optional description of the operation
    /// @return Unique ID for the future
    size_type register_future(std::string description = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        size_type id = next_id_++;
        pending_future_info info;
        info.id = id;
        info.created_at = std::chrono::steady_clock::now();
        info.description = std::move(description);
        info.is_waiting = false;
        futures_.push_back(std::move(info));
        return id;
    }

    /// @brief Unregister a future (when it completes)
    /// @param id The future ID returned from register_future
    void unregister_future(size_type id) {
        std::lock_guard<std::mutex> lock(mutex_);
        futures_.erase(
            std::remove_if(futures_.begin(), futures_.end(),
                          [id](const pending_future_info& f) { return f.id == id; }),
            futures_.end());
    }

    /// @brief Mark a future as waiting
    /// @param id The future ID
    void mark_waiting(size_type id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& future : futures_) {
            if (future.id == id) {
                future.is_waiting = true;
                break;
            }
        }
    }

    /// @brief Record a poll event
    void record_poll() {
        ++total_polls_;
        last_poll_time_ = std::chrono::steady_clock::now();
    }

    /// @brief Record a timeout event for a future
    /// @param id The future ID
    void record_timeout(size_type id) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++timeout_count_;
        for (auto& future : futures_) {
            if (future.id == id) {
                future.is_waiting = true;
                break;
            }
        }
    }

    /// @brief Get timeout count
    [[nodiscard]] size_type timeout_count() const noexcept {
        return timeout_count_.load(std::memory_order_relaxed);
    }

    /// @brief Set background progress state
    void set_background_progress(bool enabled, bool running) {
        background_enabled_ = enabled;
        background_running_ = running;
    }

    /// @brief Get current diagnostic snapshot
    [[nodiscard]] progress_diagnostics get_diagnostics() const {
        progress_diagnostics diag;
        diag.snapshot_time = std::chrono::steady_clock::now();
        diag.total_polls = total_polls_.load(std::memory_order_relaxed);
        diag.last_poll_time = last_poll_time_.load(std::memory_order_relaxed);
        diag.background_progress_enabled = background_enabled_.load(std::memory_order_relaxed);
        diag.background_thread_running = background_running_.load(std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            diag.pending_future_count = futures_.size();
            diag.pending_futures = futures_;
        }

        return diag;
    }

    /// @brief Get total poll count
    [[nodiscard]] size_type total_polls() const noexcept {
        return total_polls_.load(std::memory_order_relaxed);
    }

private:
    diagnostic_collector() = default;

    mutable std::mutex mutex_;
    std::vector<pending_future_info> futures_;
    size_type next_id_ = 1;

    std::atomic<size_type> total_polls_{0};
    std::atomic<size_type> timeout_count_{0};
    std::atomic<std::chrono::steady_clock::time_point> last_poll_time_{
        std::chrono::steady_clock::now()};
    std::atomic<bool> background_enabled_{false};
    std::atomic<bool> background_running_{false};
};

// ============================================================================
// Timeout Exception
// ============================================================================

/// @brief Exception thrown when a future wait times out
class timeout_exception : public std::runtime_error {
public:
    /// @brief Construct with message and diagnostics
    explicit timeout_exception(const std::string& message,
                               progress_diagnostics diagnostics = {})
        : std::runtime_error(message)
        , diagnostics_(std::move(diagnostics)) {}

    /// @brief Get the diagnostic snapshot at timeout
    [[nodiscard]] const progress_diagnostics& diagnostics() const noexcept {
        return diagnostics_;
    }

private:
    progress_diagnostics diagnostics_;
};

// ============================================================================
// Global Timeout Configuration
// ============================================================================

/// @brief Get the global timeout configuration
/// @return Reference to the global timeout configuration
[[nodiscard]] inline timeout_config& global_timeout_config() {
    static timeout_config config = timeout_config::defaults();
    return config;
}

/// @brief Set the global timeout configuration
/// @param config New timeout configuration
inline void set_global_timeout_config(timeout_config config) {
    global_timeout_config() = std::move(config);
}

/// @brief Check if DTL_CI_MODE environment variable is set
/// @details Caches the result at static initialization (Meyers' singleton).
///          Thread-safe in C++11+. Called once; subsequent calls are zero-cost.
[[nodiscard]] inline bool is_ci_mode() noexcept {
    static const bool ci_mode = (std::getenv("DTL_CI_MODE") != nullptr);
    return ci_mode;
}

/// @brief Get the effective wait timeout
/// @details Returns CI timeout if DTL_CI_MODE is set, otherwise default timeout
[[nodiscard]] inline std::chrono::milliseconds effective_wait_timeout() {
    const auto& config = global_timeout_config();

    // Check for CI mode via cached environment variable
    if (is_ci_mode()) {
        return config.ci_wait_timeout;
    }

    return config.default_wait_timeout;
}

/// @brief Generate timeout diagnostic message
/// @param operation Description of the operation that timed out
/// @return Formatted diagnostic message
[[nodiscard]] inline std::string format_timeout_diagnostics(const std::string& operation) {
    auto diag = diagnostic_collector::instance().get_diagnostics();
    std::ostringstream oss;
    oss << "Timeout waiting for: " << operation << "\n";
    oss << diag.to_string();
    return oss.str();
}

}  // namespace dtl::futures
