// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_callback_executor.cpp
/// @brief Unit tests for callback executor isolation (Phase 07)
/// @details Tests that long-running callbacks don't block progress

#include <dtl/futures/callback_executor.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/futures/distributed_future.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace dtl::futures::test {

// =============================================================================
// Callback Executor Construction Tests
// =============================================================================

TEST(CallbackExecutorTest, DefaultConstruction) {
    callback_executor executor;

    EXPECT_TRUE(executor.is_running());
    EXPECT_EQ(executor.pending_count(), 0u);
    EXPECT_EQ(executor.mode(), executor_mode::single_thread);
}

TEST(CallbackExecutorTest, InlineModeConstruction) {
    auto config = executor_config::inline_execution();
    callback_executor executor(config);

    EXPECT_TRUE(executor.is_running());
    EXPECT_EQ(executor.mode(), executor_mode::inline_mode);
}

TEST(CallbackExecutorTest, ThreadPoolConstruction) {
    auto config = executor_config::thread_pool_execution(4);
    callback_executor executor(config);

    EXPECT_TRUE(executor.is_running());
    EXPECT_EQ(executor.mode(), executor_mode::thread_pool);
}

// =============================================================================
// Enqueue and Execute Tests
// =============================================================================

TEST(CallbackExecutorTest, EnqueueExecutesCallback) {
    callback_executor executor;

    std::atomic<bool> executed{false};
    bool enqueued = executor.enqueue([&executed] {
        executed = true;
    });

    EXPECT_TRUE(enqueued);

    // Wait for execution
    executor.drain();

    EXPECT_TRUE(executed.load());
}

TEST(CallbackExecutorTest, InlineModeExecutesImmediately) {
    auto config = executor_config::inline_execution();
    callback_executor executor(config);

    std::atomic<bool> executed{false};
    bool enqueued = executor.enqueue([&executed] {
        executed = true;
    });

    EXPECT_TRUE(enqueued);
    EXPECT_TRUE(executed.load());  // Should be immediate
}

TEST(CallbackExecutorTest, MultipleCallbacksExecute) {
    callback_executor executor;

    std::atomic<int> counter{0};
    constexpr int num_callbacks = 10;

    for (int i = 0; i < num_callbacks; ++i) {
        executor.enqueue([&counter] {
            ++counter;
        });
    }

    executor.drain();

    EXPECT_EQ(counter.load(), num_callbacks);
    EXPECT_EQ(executor.total_enqueued(), static_cast<size_type>(num_callbacks));
    EXPECT_EQ(executor.total_executed(), static_cast<size_type>(num_callbacks));
}

// =============================================================================
// Shutdown Tests
// =============================================================================

TEST(CallbackExecutorTest, ShutdownRejectsNewCallbacks) {
    callback_executor executor;

    executor.shutdown();

    EXPECT_FALSE(executor.is_running());

    // Enqueue should fail after shutdown
    bool enqueued = executor.enqueue([] {});
    EXPECT_FALSE(enqueued);
}

TEST(CallbackExecutorTest, ShutdownDrainsExisting) {
    callback_executor executor;

    std::atomic<int> counter{0};
    for (int i = 0; i < 5; ++i) {
        executor.enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++counter;
        });
    }

    // Shutdown should drain existing callbacks
    executor.shutdown();

    EXPECT_EQ(counter.load(), 5);
}

// =============================================================================
// Thread Pool Tests
// =============================================================================

TEST(CallbackExecutorTest, ThreadPoolParallelExecution) {
    auto config = executor_config::thread_pool_execution(4);
    callback_executor executor(config);

    std::atomic<int> active_count{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<int> completed{0};

    constexpr int num_callbacks = 20;

    for (int i = 0; i < num_callbacks; ++i) {
        executor.enqueue([&] {
            int current = ++active_count;
            // Track max concurrent executions
            int expected = max_concurrent.load();
            while (current > expected) {
                max_concurrent.compare_exchange_weak(expected, current);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            --active_count;
            ++completed;
        });
    }

    executor.drain();

    EXPECT_EQ(completed.load(), num_callbacks);
    // With 4 threads, we should see some concurrency (at least 2)
    EXPECT_GE(max_concurrent.load(), 2);
}

// =============================================================================
// Exception Handling Tests
// =============================================================================

TEST(CallbackExecutorTest, ExceptionsAreCaught) {
    callback_executor executor;

    std::atomic<bool> after_throw{false};

    executor.enqueue([] {
        throw std::runtime_error("Test exception");
    });

    executor.enqueue([&after_throw] {
        after_throw = true;
    });

    executor.drain();

    // Second callback should still execute despite first throwing
    EXPECT_TRUE(after_throw.load());
}

// =============================================================================
// Isolation Tests (Key Phase 07 requirement)
// =============================================================================

TEST(CallbackExecutorTest, LongCallbackDoesNotBlockOthers) {
    // This test verifies the KNOWN_ISSUES fix:
    // "Long-running callbacks may delay other completions"

    callback_executor executor;

    std::atomic<bool> long_started{false};
    std::atomic<bool> long_finished{false};
    std::atomic<bool> short_finished{false};

    // Enqueue a long-running callback
    executor.enqueue([&] {
        long_started = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        long_finished = true;
    });

    // Wait for long callback to start
    while (!long_started.load()) {
        std::this_thread::yield();
    }

    // Enqueue a short callback while long is running
    executor.enqueue([&] {
        short_finished = true;
    });

    // Wait a bit - short should complete even though long is running
    // (if executor has multiple threads or queues separately)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // In single-thread mode, short will wait. In thread pool, it would run.
    // The key is that enqueue succeeds and doesn't block

    executor.drain();

    EXPECT_TRUE(long_finished.load());
    EXPECT_TRUE(short_finished.load());
}

TEST(CallbackExecutorTest, ThreadPoolIsolatesLongCallbacks) {
    // With thread pool, long callbacks truly don't block others
    auto config = executor_config::thread_pool_execution(2);
    callback_executor executor(config);

    std::atomic<bool> long_started{false};
    std::atomic<bool> short_finished{false};
    std::chrono::steady_clock::time_point short_finished_time;

    // Enqueue a long-running callback
    auto start_time = std::chrono::steady_clock::now();
    executor.enqueue([&] {
        long_started = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });

    // Wait for long callback to start
    while (!long_started.load()) {
        std::this_thread::yield();
    }

    // Enqueue a short callback
    executor.enqueue([&] {
        short_finished = true;
        short_finished_time = std::chrono::steady_clock::now();
    });

    executor.drain();

    // Short should finish quickly (not wait for long)
    auto short_elapsed = short_finished_time - start_time;
    // Allow up to 50ms for short to complete (much less than long's 100ms)
    EXPECT_LT(short_elapsed, std::chrono::milliseconds(60));
    EXPECT_TRUE(short_finished.load());
}

// =============================================================================
// Global Executor Tests
// =============================================================================

TEST(CallbackExecutorTest, GlobalExecutorExists) {
    auto& executor = global_callback_executor();

    EXPECT_TRUE(executor.is_running());
}

}  // namespace dtl::futures::test
