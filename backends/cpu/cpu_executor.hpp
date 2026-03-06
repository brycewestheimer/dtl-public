// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cpu_executor.hpp
/// @brief Thread pool executor for CPU-based parallel execution
/// @details Provides parallel execution on CPU using a thread pool.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/executor.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Work-stealing pool (opt-in via DTL_USE_WORK_STEALING)
#if DTL_USE_WORK_STEALING
#include "work_stealing_pool.hpp"
#endif

namespace dtl {
namespace cpu {

// ============================================================================
// Thread Pool
// ============================================================================

/// @brief Simple thread pool for parallel task execution
/// @details Manages a pool of worker threads that execute submitted tasks.
class thread_pool {
public:
    /// @brief Construct thread pool
    /// @param num_threads Number of worker threads (0 = hardware concurrency)
    explicit thread_pool(size_type num_threads = 0)
        : stop_(false) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }

        workers_.reserve(num_threads);
        for (size_type i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    /// @brief Destructor (waits for all tasks to complete)
    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Non-copyable, non-movable
    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

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

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("Cannot submit to stopped thread pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();

        return result;
    }

    /// @brief Get number of worker threads
    [[nodiscard]] size_type size() const noexcept {
        return workers_.size();
    }

    /// @brief Get number of pending tasks
    [[nodiscard]] size_type pending() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    /// @brief Wait for all pending tasks to complete
    void wait() {
        // Submit a barrier task and wait for it
        submit([](){}).wait();
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

// ============================================================================
// CPU Executor
// ============================================================================

/// @brief CPU-based parallel executor using thread pool
/// @details Satisfies the Executor and ParallelExecutor concepts.
///          When DTL_USE_WORK_STEALING is defined, uses the work-stealing pool
///          for reduced contention on high-core-count systems.
class cpu_executor {
public:
#if DTL_USE_WORK_STEALING
    using pool_type = work_stealing_pool;
#else
    using pool_type = thread_pool;
#endif

    /// @brief Construct with default thread count
    cpu_executor() : pool_(std::make_unique<pool_type>()) {}

    /// @brief Construct with specified thread count
    /// @param num_threads Number of worker threads
    explicit cpu_executor(size_type num_threads)
        : pool_(std::make_unique<pool_type>(num_threads)) {}

    /// @brief Destructor
    ~cpu_executor() = default;

    // Non-copyable
    cpu_executor(const cpu_executor&) = delete;
    cpu_executor& operator=(const cpu_executor&) = delete;

    // Movable
    cpu_executor(cpu_executor&&) = default;
    cpu_executor& operator=(cpu_executor&&) = default;

    // ------------------------------------------------------------------------
    // Executor Interface
    // ------------------------------------------------------------------------

    /// @brief Execute a function
    /// @tparam F Callable type
    /// @param func Function to execute
    template <typename F>
    void execute(F&& func) {
        pool_->submit(std::forward<F>(func));
    }

    /// @brief Execute a function and get a future
    /// @tparam F Callable type
    /// @param func Function to execute
    /// @return Future for the result
    template <typename F>
    auto async_execute(F&& func) {
        return pool_->submit(std::forward<F>(func));
    }

    /// @brief Execute a function synchronously
    /// @tparam F Callable type
    /// @param func Function to execute
    template <typename F>
    void sync_execute(F&& func) {
        pool_->submit(std::forward<F>(func)).wait();
    }

    /// @brief Wait for all submitted tasks to complete
    void synchronize() {
        pool_->wait();
    }

    // ------------------------------------------------------------------------
    // Parallel Execution
    // ------------------------------------------------------------------------

    /// @brief Execute in parallel over a range
    /// @tparam F Callable type (takes index)
    /// @param begin Start index
    /// @param end End index
    /// @param func Function to execute for each index
    template <typename F>
    void parallel_for(index_t begin, index_t end, F&& func) {
        if (begin >= end) return;

        const size_type num_threads = pool_->size();
        const index_t num_threads_i = static_cast<index_t>(num_threads);
        const index_t total = end - begin;
        const index_t chunk_size = (total + num_threads_i - 1) / num_threads_i;

        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (index_t t = 0; t < num_threads_i; ++t) {
            index_t chunk_begin = begin + t * chunk_size;
            index_t chunk_end = std::min(chunk_begin + chunk_size, end);

            if (chunk_begin >= end) break;

            futures.push_back(pool_->submit([=, &func]() {
                for (index_t i = chunk_begin; i < chunk_end; ++i) {
                    func(i);
                }
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

    /// @brief Execute in parallel with custom chunk size
    /// @tparam F Callable type
    /// @param begin Start index
    /// @param end End index
    /// @param chunk_size Size of each chunk
    /// @param func Function to execute for each index
    template <typename F>
    void parallel_for_chunked(index_t begin, index_t end,
                              index_t chunk_size, F&& func) {
        if (begin >= end) return;

        std::vector<std::future<void>> futures;

        for (index_t chunk_begin = begin; chunk_begin < end;
             chunk_begin += chunk_size) {
            index_t chunk_end = std::min(chunk_begin + chunk_size, end);

            futures.push_back(pool_->submit([=, &func]() {
                for (index_t i = chunk_begin; i < chunk_end; ++i) {
                    func(i);
                }
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

    /// @brief Parallel reduce
    /// @tparam T Value type
    /// @tparam F Function type (takes begin, end indices, returns T)
    /// @tparam R Reduction operation type
    /// @param begin Start index
    /// @param end End index
    /// @param identity Identity element
    /// @param map_func Function to compute partial result
    /// @param reduce_func Function to combine results
    /// @return Reduced result
    template <typename T, typename F, typename R>
    T parallel_reduce(index_t begin, index_t end, T identity,
                      F&& map_func, R&& reduce_func) {
        if (begin >= end) return identity;

        const size_type num_threads = pool_->size();
        const index_t num_threads_i = static_cast<index_t>(num_threads);
        const index_t total = end - begin;
        const index_t chunk_size = (total + num_threads_i - 1) / num_threads_i;

        std::vector<std::future<T>> futures;
        futures.reserve(num_threads);

        for (index_t t = 0; t < num_threads_i; ++t) {
            index_t chunk_begin = begin + t * chunk_size;
            index_t chunk_end = std::min(chunk_begin + chunk_size, end);

            if (chunk_begin >= end) break;

            futures.push_back(pool_->submit([=, &map_func]() {
                return map_func(chunk_begin, chunk_end);
            }));
        }

        T result = identity;
        for (auto& f : futures) {
            result = reduce_func(result, f.get());
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Concept-Required Interface (Executor/ParallelExecutor)
    // ------------------------------------------------------------------------

    /// @brief Get executor name
    /// @return Executor name string
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "cpu";
    }

    /// @brief Parallel for over count elements (concept-compliant overload)
    /// @tparam F Callable type (takes index)
    /// @param count Number of iterations
    /// @param func Function to execute for each index
    template <typename F>
    void parallel_for(size_type count, F&& func) {
        parallel_for(static_cast<index_t>(0), static_cast<index_t>(count),
                     [&](index_t i) { func(static_cast<size_type>(i)); });
    }

    /// @brief Get maximum parallelism (number of threads)
    /// @return Maximum concurrent work items
    [[nodiscard]] size_type max_parallelism() const noexcept {
        return pool_ ? pool_->size() : 1;
    }

    /// @brief Get suggested parallelism
    /// @return Suggested number of concurrent work items
    [[nodiscard]] size_type suggested_parallelism() const noexcept {
        return max_parallelism();
    }

    // ------------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------------

    /// @brief Get number of worker threads
    [[nodiscard]] size_type num_threads() const noexcept {
        return pool_ ? pool_->size() : 0;
    }

    /// @brief Check if executor is valid
    [[nodiscard]] bool valid() const noexcept {
        return pool_ != nullptr;
    }

private:
    std::unique_ptr<pool_type> pool_;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Executor<cpu_executor>, "cpu_executor must satisfy Executor concept");
static_assert(ParallelExecutor<cpu_executor>, "cpu_executor must satisfy ParallelExecutor concept");

// ============================================================================
// Global Executor
// ============================================================================

/// @brief Get the default CPU executor (lazily initialized)
/// @return Reference to default executor
[[nodiscard]] inline cpu_executor& default_executor() {
    static cpu_executor executor;
    return executor;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Execute a parallel for loop on the default executor
/// @tparam F Callable type
/// @param begin Start index
/// @param end End index
/// @param func Function to execute for each index
template <typename F>
void parallel_for(index_t begin, index_t end, F&& func) {
    default_executor().parallel_for(begin, end, std::forward<F>(func));
}

/// @brief Execute a parallel reduce on the default executor
/// @tparam T Value type
/// @tparam F Map function type
/// @tparam R Reduce function type
/// @param begin Start index
/// @param end End index
/// @param identity Identity element
/// @param map_func Function to compute partial result
/// @param reduce_func Function to combine results
/// @return Reduced result
template <typename T, typename F, typename R>
T parallel_reduce(index_t begin, index_t end, T identity,
                  F&& map_func, R&& reduce_func) {
    return default_executor().parallel_reduce(
        begin, end, identity,
        std::forward<F>(map_func),
        std::forward<R>(reduce_func));
}

}  // namespace cpu
}  // namespace dtl
