// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file work_stealing_pool.hpp
/// @brief Work-stealing thread pool — same interface as thread_pool
/// @details N workers each with their own work_stealing_queue. Workers
///          pop locally first, then steal from a random other thread.
///          Reduces contention compared to a single shared std::mutex queue.
/// @since 0.1.0

#pragma once

#include "work_stealing_queue.hpp"

#include <dtl/core/types.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace dtl {
namespace cpu {

/// @brief Work-stealing thread pool with per-thread queues
/// @details Drop-in replacement for thread_pool. Workers prioritize their
///          local queue (LIFO) and steal from others (FIFO) when idle.
class work_stealing_pool {
public:
    /// @brief Construct work-stealing pool
    /// @param num_threads Number of worker threads (0 = hardware concurrency)
    explicit work_stealing_pool(size_type num_threads = 0)
        : stop_(false)
        , next_queue_(0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }

        queues_.reserve(num_threads);
        for (size_type i = 0; i < num_threads; ++i) {
            queues_.push_back(std::make_unique<work_stealing_queue>());
        }

        idle_flags_ = std::make_unique<std::atomic<bool>[]>(num_threads);
        num_workers_ = num_threads;
        for (size_type i = 0; i < num_threads; ++i) {
            idle_flags_[i].store(false, std::memory_order_relaxed);
        }

        workers_.reserve(num_threads);
        for (size_type i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    /// @brief Destructor (waits for all tasks to complete)
    ~work_stealing_pool() {
        stop_.store(true, std::memory_order_release);

        // Wake all workers
        {
            std::lock_guard<std::mutex> lock(wake_mutex_);
        }
        wake_cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Non-copyable, non-movable
    work_stealing_pool(const work_stealing_pool&) = delete;
    work_stealing_pool& operator=(const work_stealing_pool&) = delete;
    work_stealing_pool(work_stealing_pool&&) = delete;
    work_stealing_pool& operator=(work_stealing_pool&&) = delete;

    /// @brief Submit a task for execution
    /// @tparam F Callable type
    /// @tparam Args Argument types
    /// @param func Function to execute
    /// @param args Arguments to pass
    /// @return Future for the result
    template <typename F, typename... Args>
    auto submit(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();

        // Round-robin queue selection for submitted tasks
        auto idx = next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
        if (!queues_[idx]->push([task]() { (*task)(); })) {
            // If the selected queue is full, try others
            bool pushed = false;
            for (size_type i = 0; i < queues_.size(); ++i) {
                if (queues_[(idx + i + 1) % queues_.size()]->push([task]() { (*task)(); })) {
                    pushed = true;
                    break;
                }
            }
            if (!pushed) {
                // Fallback: execute inline if all queues are full
                (*task)();
                return result;
            }
        }

        // Wake one idle worker
        wake_cv_.notify_one();

        return result;
    }

    /// @brief Get number of worker threads
    [[nodiscard]] size_type size() const noexcept {
        return workers_.size();
    }

    /// @brief Get approximate number of pending tasks
    [[nodiscard]] size_type pending() const {
        size_type total = 0;
        for (const auto& q : queues_) {
            total += q->size();
        }
        return total;
    }

    /// @brief Wait for all pending tasks to complete
    void wait() {
        // Spin until all queues are drained, then confirm with barriers
        while (pending() > 0) {
            std::this_thread::yield();
        }
        // Submit a barrier to every queue and wait — ensures in-flight
        // tasks (being executed by workers) have completed
        std::vector<std::future<void>> barriers;
        barriers.reserve(queues_.size());
        for (size_type i = 0; i < queues_.size(); ++i) {
            auto task = std::make_shared<std::packaged_task<void()>>([](){});
            barriers.push_back(task->get_future());
            queues_[i]->push([task]() { (*task)(); });
            wake_cv_.notify_one();
        }
        for (auto& f : barriers) {
            f.wait();
        }
    }

private:
    void worker_loop(size_type my_index) {
        // Thread-local RNG for random victim selection
        std::mt19937 rng(static_cast<unsigned>(my_index * 7919 + 104729));

        while (true) {
            // 1. Try local pop
            auto task = queues_[my_index]->pop();

            // 2. If no local work, try stealing
            if (!task) {
                task = try_steal(my_index, rng);
            }

            if (task) {
                (*task)();
                continue;
            }

            // 3. No work found — go idle
            if (stop_.load(std::memory_order_acquire)) {
                // Drain any remaining tasks before exiting
                drain_and_exit(my_index, rng);
                return;
            }

            idle_flags_[my_index].store(true, std::memory_order_relaxed);
            {
                std::unique_lock<std::mutex> lock(wake_mutex_);
                wake_cv_.wait_for(lock, std::chrono::microseconds(100), [this, my_index] {
                    return stop_.load(std::memory_order_relaxed) ||
                           !queues_[my_index]->empty();
                });
            }
            idle_flags_[my_index].store(false, std::memory_order_relaxed);
        }
    }

    std::optional<work_stealing_queue::task_type> try_steal(
            size_type my_index, std::mt19937& rng) {
        const auto n = queues_.size();
        if (n <= 1) return std::nullopt;

        // Try random victim first, then linear scan
        auto victim = std::uniform_int_distribution<size_type>(0, n - 2)(rng);
        if (victim >= my_index) ++victim;

        auto task = queues_[victim]->steal();
        if (task) return task;

        // Linear scan of remaining queues
        for (size_type i = 0; i < n; ++i) {
            if (i == my_index) continue;
            task = queues_[i]->steal();
            if (task) return task;
        }
        return std::nullopt;
    }

    void drain_and_exit(size_type my_index, std::mt19937& rng) {
        // Process remaining tasks
        while (true) {
            auto task = queues_[my_index]->pop();
            if (!task) task = try_steal(my_index, rng);
            if (!task) break;
            (*task)();
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<work_stealing_queue>> queues_;
    std::unique_ptr<std::atomic<bool>[]> idle_flags_;
    size_type num_workers_{0};
    std::atomic<bool> stop_;
    std::atomic<size_type> next_queue_;
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
};

}  // namespace cpu
}  // namespace dtl
