// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file progress.hpp
/// @brief Progress engine and progress-based completion for async operations
/// @details Provides a poll-based progress model that drives pending asynchronous
///          operations without using detached threads. This replaces the detached
///          thread pattern with explicit progress calls.
///
///          As of v1.3.0, the progress engine supports:
///          - Explicit polling via poll(), poll_one(), poll_for()
///          - Optional background progress mode (see background_progress.hpp)
///          - Callback isolation via callback_executor (see callback_executor.hpp)
///          - Configurable timeouts and diagnostics (see diagnostics.hpp)
///
/// @since 0.1.0
/// @note Updated in 1.3.0: Added public polling API, background mode, callback isolation.

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dtl::futures {

// ============================================================================
// Progress State
// ============================================================================

/// @brief State of a progress-trackable operation
enum class progress_state {
    pending,    ///< Operation not yet complete
    ready,      ///< Operation completed successfully
    error       ///< Operation completed with error
};

// ============================================================================
// Progress Callback
// ============================================================================

/// @brief Type for progress callback functions
/// @details Progress callbacks are invoked during make_progress() calls.
///          They return true if more progress is possible, false if complete.
using progress_callback = std::function<bool()>;

// ============================================================================
// Progress Engine
// ============================================================================

/// @brief Singleton progress engine that drives all pending async operations
/// @details The progress engine maintains a list of pending operations and
///          drives them forward when make_progress() is called. This replaces
///          the detached thread pattern with explicit, controlled progress.
class progress_engine {
public:
    /// @brief Get the singleton progress engine instance
    /// @return Reference to the global progress engine
    [[nodiscard]] static progress_engine& instance() noexcept {
        static progress_engine engine;
        return engine;
    }

    /// @brief Register a progress callback
    /// @param callback Function that drives progress and returns true if not yet complete
    /// @return Handle that can be used to unregister the callback
    size_type register_callback(progress_callback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_type id = next_id_++;
        callbacks_.emplace_back(id, std::move(callback));
        return id;
    }

    /// @brief Unregister a progress callback
    /// @param id Handle returned from register_callback
    void unregister_callback(size_type id) {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.erase(
            std::remove_if(callbacks_.begin(), callbacks_.end(),
                          [id](const auto& p) { return p.first == id; }),
            callbacks_.end());
    }

    /// @brief Drive all pending operations (unbounded)
    /// @details Polls all registered callbacks once. Callbacks that return
    ///          false are automatically removed. When CUDA is enabled, also
    ///          polls registered CUDA events via cudaEventQuery().
    ///
    ///          Thread-safety: Only one thread may execute poll() at a time.
    ///          If another thread is already polling, concurrent callers return
    ///          immediately with 0. This prevents double-invocation of callbacks
    ///          that may not be idempotent and avoids data races on shared
    ///          captured state within callbacks.
    /// @return Number of callbacks that made progress (0 if another thread is polling)
    size_type poll() {
        // Acquire exclusive polling rights. If another thread is already
        // polling, skip this call to prevent double callback invocation.
        bool expected = false;
        if (!polling_.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel)) {
            return 0;
        }

        // RAII guard to ensure polling_ is reset even if a callback throws
        struct polling_guard {
            std::atomic<bool>& flag;
            ~polling_guard() { flag.store(false, std::memory_order_release); }
        } guard{polling_};

#if DTL_ENABLE_CUDA
        poll_cuda_events();
#endif

        std::vector<std::pair<size_type, progress_callback>> to_process;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            to_process = callbacks_;
        }

        std::vector<size_type> completed;
        size_type progress_count = 0;

        for (auto& [id, cb] : to_process) {
            try {
                if (!cb()) {
                    // Callback complete - mark for removal
                    completed.push_back(id);
                } else {
                    ++progress_count;
                }
            } catch (...) {
                // On exception, remove the callback
                completed.push_back(id);
            }
        }

        // Remove completed callbacks
        if (!completed.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto id : completed) {
                callbacks_.erase(
                    std::remove_if(callbacks_.begin(), callbacks_.end(),
                                  [id](const auto& p) { return p.first == id; }),
                    callbacks_.end());
            }
        }

        return progress_count;
    }

    /// @brief Check if there are pending operations
    /// @return true if any callbacks or CUDA events are registered
    [[nodiscard]] bool has_pending() const noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!callbacks_.empty()) return true;
        }
#if DTL_ENABLE_CUDA
        {
            std::lock_guard<std::mutex> cuda_lock(cuda_events_mutex_);
            if (!pending_cuda_events_.empty()) return true;
        }
#endif
        return false;
    }

    /// @brief Get number of pending operations
    /// @return Count of registered callbacks
    [[nodiscard]] size_type pending_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return callbacks_.size();
    }

#if DTL_ENABLE_CUDA
    /// @brief Register a CUDA event with a completion callback
    /// @details The progress engine will poll the event via cudaEventQuery()
    ///          during each poll() call. When the event completes, the callback
    ///          is invoked and the event is removed from the pending list.
    /// @param event The CUDA event to poll for completion
    /// @param callback Function to invoke when the event completes
    void register_cuda_event(cudaEvent_t event, std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(cuda_events_mutex_);
        pending_cuda_events_.emplace_back(event, std::move(callback));
    }

    /// @brief Check if there are pending CUDA events
    /// @return true if any CUDA events are registered
    [[nodiscard]] bool has_pending_cuda() const noexcept {
        std::lock_guard<std::mutex> lock(cuda_events_mutex_);
        return !pending_cuda_events_.empty();
    }

    /// @brief Get number of pending CUDA events
    /// @return Count of registered CUDA events
    [[nodiscard]] size_type pending_cuda_count() const noexcept {
        std::lock_guard<std::mutex> lock(cuda_events_mutex_);
        return pending_cuda_events_.size();
    }
#endif

private:
    progress_engine() = default;
    ~progress_engine() = default;

    progress_engine(const progress_engine&) = delete;
    progress_engine& operator=(const progress_engine&) = delete;

    mutable std::mutex mutex_;
    std::vector<std::pair<size_type, progress_callback>> callbacks_;
    size_type next_id_ = 0;
    std::atomic<bool> polling_{false};

#if DTL_ENABLE_CUDA
    /// @brief Poll all registered CUDA events and invoke callbacks for completed ones
    /// @details Collects completed events under lock, then invokes callbacks outside
    ///          the lock to prevent deadlocks from re-entrant registration.
    /// @return Number of CUDA events that completed
    size_type poll_cuda_events() {
        std::vector<std::function<void()>> ready_callbacks;
        {
            std::lock_guard<std::mutex> lock(cuda_events_mutex_);
            auto it = pending_cuda_events_.begin();
            while (it != pending_cuda_events_.end()) {
                if (cudaEventQuery(it->first) == cudaSuccess) {
                    ready_callbacks.push_back(std::move(it->second));
                    it = pending_cuda_events_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& cb : ready_callbacks) {
            cb();
        }
        return ready_callbacks.size();
    }

    mutable std::mutex cuda_events_mutex_;
    std::vector<std::pair<cudaEvent_t, std::function<void()>>> pending_cuda_events_;
#endif
};

// ============================================================================
// Free Functions (Public Progress API)
// ============================================================================

/// @brief Drive pending operations forward (single poll)
/// @details Calls poll() on the global progress engine once.
///          This drives all registered async operations.
///
///          This is the primary explicit polling API. Call this function
///          periodically in application loops to ensure progress is made
///          on pending async operations.
///
/// @return Number of operations that made progress
/// @see poll_one(), poll_for(), poll_until()
inline size_type poll() {
    return progress_engine::instance().poll();
}

/// @brief Drive pending operations forward (alias for poll())
/// @deprecated Use poll() instead for clarity
/// @return Number of operations that made progress
inline size_type make_progress() {
    return poll();
}

/// @brief Attempt to complete exactly one pending operation
/// @details Polls until at least one operation completes or no pending work.
///          Useful for fine-grained progress control.
/// @return true if one operation completed, false if no pending operations
inline bool poll_one() {
    auto& engine = progress_engine::instance();
    if (!engine.has_pending()) {
        return false;
    }

    // Poll until we complete at least one operation
    size_type initial_count = engine.pending_count();
    while (engine.has_pending()) {
        engine.poll();
        if (engine.pending_count() < initial_count) {
            return true;  // Completed at least one
        }
        std::this_thread::yield();
    }
    return true;  // All completed
}

/// @brief Poll for a specified duration
/// @details Continuously polls the progress engine for the specified duration.
///          Useful for applications that want to dedicate a time slice to progress.
/// @param duration Maximum time to spend polling
/// @return Total number of operations that made progress
template <typename Rep, typename Period>
size_type poll_for(std::chrono::duration<Rep, Period> duration) {
    auto deadline = std::chrono::steady_clock::now() + duration;
    size_type total = 0;

    while (std::chrono::steady_clock::now() < deadline) {
        size_type count = progress_engine::instance().poll();
        total += count;

        if (!progress_engine::instance().has_pending()) {
            break;  // No more work
        }

        if (count == 0) {
            std::this_thread::yield();
        }
    }

    return total;
}

/// @brief Poll until a predicate becomes true
/// @details Continuously polls the progress engine until the predicate returns true.
/// @param predicate Function that returns true when polling should stop
/// @param timeout Maximum time to poll (0 = no timeout)
/// @return true if predicate became true, false if timeout
template <typename Predicate>
bool poll_until(Predicate predicate,
                std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    auto deadline = timeout.count() > 0
        ? std::chrono::steady_clock::now() + timeout
        : std::chrono::steady_clock::time_point::max();

    while (!predicate()) {
        progress_engine::instance().poll();

        if (std::chrono::steady_clock::now() >= deadline) {
            return false;  // Timeout
        }

        std::this_thread::yield();
    }

    return true;
}

/// @brief Drive pending operations with bounded polling
/// @param max_polls Maximum number of poll iterations
/// @return Total number of operations that made progress across all polls
inline size_type make_progress(size_type max_polls) {
    size_type total = 0;
    for (size_type i = 0; i < max_polls; ++i) {
        size_type count = progress_engine::instance().poll();
        total += count;
        if (count == 0 && !progress_engine::instance().has_pending()) {
            break;  // No more work to do
        }
    }
    return total;
}

/// @brief Block until all pending operations complete
/// @param max_iterations Maximum number of poll iterations (0 = unlimited)
/// @return true if all operations completed, false if max_iterations reached
inline bool drain_progress(size_type max_iterations = 0) {
    size_type iterations = 0;
    while (progress_engine::instance().has_pending()) {
        progress_engine::instance().poll();
        ++iterations;
        if (max_iterations > 0 && iterations >= max_iterations) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Progress Guard (RAII)
// ============================================================================

/// @brief RAII guard that calls make_progress() on destruction
/// @details Use this in scopes where you want to ensure progress is made
///          even if the scope exits early (e.g., via exception).
class progress_guard {
public:
    /// @brief Constructor (optionally specifies max polls on destruction)
    /// @param max_polls Maximum polls on destruction (0 = single poll)
    explicit progress_guard(size_type max_polls = 1) noexcept
        : max_polls_(max_polls) {}

    /// @brief Destructor - drives progress
    ~progress_guard() {
        if (max_polls_ == 0) {
            make_progress();
        } else {
            make_progress(max_polls_);
        }
    }

    /// @brief Manually trigger progress (can be called multiple times)
    void poll() {
        make_progress();
    }

    // Non-copyable, non-movable
    progress_guard(const progress_guard&) = delete;
    progress_guard& operator=(const progress_guard&) = delete;
    progress_guard(progress_guard&&) = delete;
    progress_guard& operator=(progress_guard&&) = delete;

private:
    size_type max_polls_;
};

// ============================================================================
// Scoped Progress Registration
// ============================================================================

/// @brief RAII wrapper for progress callback registration
/// @details Automatically unregisters the callback on destruction.
class scoped_progress_callback {
public:
    /// @brief Register a callback with automatic cleanup
    /// @param callback Progress callback function
    explicit scoped_progress_callback(progress_callback callback)
        : id_(progress_engine::instance().register_callback(std::move(callback))) {}

    /// @brief Destructor - unregisters the callback
    ~scoped_progress_callback() {
        progress_engine::instance().unregister_callback(id_);
    }

    /// @brief Get the registration ID
    [[nodiscard]] size_type id() const noexcept { return id_; }

    // Non-copyable, non-movable
    scoped_progress_callback(const scoped_progress_callback&) = delete;
    scoped_progress_callback& operator=(const scoped_progress_callback&) = delete;
    scoped_progress_callback(scoped_progress_callback&&) = delete;
    scoped_progress_callback& operator=(scoped_progress_callback&&) = delete;

private:
    size_type id_;
};

}  // namespace dtl::futures
