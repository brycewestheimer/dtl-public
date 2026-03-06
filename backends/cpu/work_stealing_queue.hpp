// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file work_stealing_queue.hpp
/// @brief Per-worker queue for the work-stealing thread pool
/// @details Simple mutex-protected deque. Contention is distributed across
///          N per-worker queues rather than concentrated on a single shared
///          queue, which is the primary scalability win.
/// @since 0.1.0

#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>

namespace dtl {
namespace cpu {

/// @brief Per-worker queue supporting push, pop, and steal
/// @details Push and pop operate on the back (LIFO for the owner).
///          Steal operates on the front (FIFO). All operations are
///          mutex-protected but contention is low since each worker
///          primarily accesses its own queue.
class work_stealing_queue {
public:
    using task_type = std::function<void()>;

    work_stealing_queue() = default;

    // Non-copyable, non-movable
    work_stealing_queue(const work_stealing_queue&) = delete;
    work_stealing_queue& operator=(const work_stealing_queue&) = delete;
    work_stealing_queue(work_stealing_queue&&) = delete;
    work_stealing_queue& operator=(work_stealing_queue&&) = delete;

    /// @brief Push a task to the back (thread-safe)
    /// @param task The task to push
    /// @return Always true (unbounded)
    bool push(task_type task) {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
        return true;
    }

    /// @brief Pop a task from the back — LIFO (thread-safe)
    /// @return The task, or std::nullopt if empty
    std::optional<task_type> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.empty()) return std::nullopt;
        auto task = std::move(tasks_.back());
        tasks_.pop_back();
        return task;
    }

    /// @brief Steal a task from the front — FIFO (thread-safe)
    /// @return The task, or std::nullopt if empty
    std::optional<task_type> steal() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.empty()) return std::nullopt;
        auto task = std::move(tasks_.front());
        tasks_.pop_front();
        return task;
    }

    /// @brief Check if the queue is empty (approximate)
    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.empty();
    }

    /// @brief Get number of pending tasks
    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<task_type> tasks_;
};

}  // namespace cpu
}  // namespace dtl
