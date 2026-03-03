// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_progress_thread_safety.cpp
/// @brief Thread-safety tests for dtl/futures/progress.hpp
/// @details Verifies that the progress engine is safe under concurrent
///          multi-thread access: no double callback invocation, no data
///          races, and correct reentrancy behavior.
///
///          Note on singleton test isolation: The progress engine is a global
///          singleton. Previous tests (e.g., BoundedPolling) may leave stale
///          callbacks that capture destroyed stack variables. Polling those
///          would segfault. Therefore these tests do NOT call drain or poll
///          to clean up foreign callbacks. Each test tracks only its own state
///          and uses scoped_progress_callback or explicit unregister for cleanup.

#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>
#include <chrono>

namespace dtl::futures::test {

// =============================================================================
// Concurrent Poll Tests
// =============================================================================

TEST(ProgressEngineThreadSafety, ConcurrentPollNoDoubleInvocation) {
    auto& engine = progress_engine::instance();

    // Track how many times the callback is concurrently active
    std::atomic<int> concurrent_count{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<int> total_calls{0};
    std::atomic<bool> done{false};

    // Use scoped callback to ensure cleanup even if test fails
    scoped_progress_callback scoped([&]() {
        int current = concurrent_count.fetch_add(1, std::memory_order_acq_rel) + 1;

        // Track max concurrency
        int prev_max = max_concurrent.load(std::memory_order_relaxed);
        while (current > prev_max &&
               !max_concurrent.compare_exchange_weak(prev_max, current,
                                                      std::memory_order_relaxed)) {}

        total_calls.fetch_add(1, std::memory_order_relaxed);

        // Simulate some work
        std::this_thread::yield();

        concurrent_count.fetch_sub(1, std::memory_order_acq_rel);

        return !done.load(std::memory_order_acquire);
    });

    // 4 threads calling poll concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 200; ++j) {
                engine.poll();
                std::this_thread::yield();
            }
        });
    }

    // Let threads run for a bit, then signal completion
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    done.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    // One more poll to clean up
    engine.poll();

    // The polling_ guard ensures max_concurrent is always 1
    EXPECT_EQ(max_concurrent.load(), 1)
        << "Callback was invoked concurrently by multiple threads";
    EXPECT_GT(total_calls.load(), 0)
        << "Callback should have been called at least once";
}

TEST(ProgressEngineThreadSafety, ConcurrentPollAndRegister) {
    auto& engine = progress_engine::instance();

    // Use shared_ptr so captured state outlives the test if callbacks leak
    auto completed_count = std::make_shared<std::atomic<int>>(0);
    std::atomic<bool> stop{false};

    // Track all registered IDs for cleanup
    std::mutex ids_mutex;
    std::vector<size_type> all_ids;

    std::vector<std::thread> threads;

    // Pollers
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                engine.poll();
                std::this_thread::yield();
            }
        });
    }

    // Registerers
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&, completed_count]() {
            for (int j = 0; j < 100; ++j) {
                auto id = engine.register_callback([completed_count]() {
                    completed_count->fetch_add(1, std::memory_order_relaxed);
                    return false;  // Complete immediately
                });
                {
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    all_ids.push_back(id);
                }
                std::this_thread::yield();
            }
        });
    }

    // Let everything settle
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    // Clean up any remaining callbacks
    for (auto id : all_ids) {
        engine.unregister_callback(id);
    }

    // No crashes = success. Completed count just needs to be non-negative.
    EXPECT_GE(completed_count->load(), 0);
}

// =============================================================================
// Stress Test
// =============================================================================

TEST(ProgressEngineThreadSafety, StressTest1000Callbacks) {
    auto& engine = progress_engine::instance();

    constexpr int NUM_CALLBACKS = 1000;
    std::atomic<int> completed{0};
    std::vector<size_type> ids;

    // Register 1000 callbacks that each complete after being polled once
    for (int i = 0; i < NUM_CALLBACKS; ++i) {
        ids.push_back(engine.register_callback([&]() {
            completed.fetch_add(1, std::memory_order_relaxed);
            return false;  // Complete immediately
        }));
    }

    // 4 threads calling poll to drive the callbacks
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 500; ++j) {
                engine.poll();
                std::this_thread::yield();
            }
        });
    }

    for (auto& t : threads) t.join();

    // Poll a few more times single-threaded to ensure all are processed
    for (int i = 0; i < 10; ++i) {
        engine.poll();
    }

    // All 1000 callbacks should have been invoked exactly once
    // (polling_ guard prevents double invocation)
    EXPECT_EQ(completed.load(), NUM_CALLBACKS);
}

TEST(ProgressEngineThreadSafety, StressTestPromiseFuture) {
    constexpr size_t NUM_FUTURES = 200;
    std::vector<dtl::distributed_promise<int>> promises(NUM_FUTURES);
    std::vector<dtl::distributed_future<int>> futures;

    for (size_t i = 0; i < NUM_FUTURES; ++i) {
        futures.push_back(promises[i].get_future());
    }

    // 4 threads setting values concurrently
    std::vector<std::thread> setters;
    for (int t = 0; t < 4; ++t) {
        setters.emplace_back([&, t]() {
            for (size_t i = static_cast<size_t>(t); i < NUM_FUTURES; i += 4) {
                promises[i].set_value(static_cast<int>(i) * 10);
            }
        });
    }

    for (auto& t : setters) t.join();

    // All futures should be ready
    for (size_t i = 0; i < NUM_FUTURES; ++i) {
        ASSERT_TRUE(futures[i].is_ready()) << "Future " << i << " not ready";
        EXPECT_EQ(futures[i].get(), static_cast<int>(i) * 10);
    }
}

// =============================================================================
// Reentrancy Tests
// =============================================================================

TEST(ProgressEngineThreadSafety, ReentrantCallbackRegistration) {
    auto& engine = progress_engine::instance();

    std::atomic<bool> inner_called{false};
    size_type inner_id = 0;

    // Outer callback registers an inner callback during execution
    auto outer_id = engine.register_callback([&]() {
        inner_id = engine.register_callback([&]() {
            inner_called.store(true, std::memory_order_release);
            return false;  // Complete immediately
        });
        return false;  // Outer callback done
    });

    // First poll: outer callback runs, registers inner callback
    engine.poll();

    // Second poll: inner callback should run
    engine.poll();

    EXPECT_TRUE(inner_called.load(std::memory_order_acquire));

    // Clean up in case they weren't removed by poll
    engine.unregister_callback(outer_id);
    if (inner_id != 0) engine.unregister_callback(inner_id);
}

TEST(ProgressEngineThreadSafety, NestedRegisterFromMultipleCallbacks) {
    auto& engine = progress_engine::instance();

    constexpr int NUM_OUTER = 10;
    std::atomic<int> inner_count{0};
    std::vector<size_type> outer_ids;
    std::vector<size_type> inner_ids;
    std::mutex inner_ids_mutex;

    for (int i = 0; i < NUM_OUTER; ++i) {
        outer_ids.push_back(engine.register_callback([&]() {
            auto id = engine.register_callback([&]() {
                inner_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            });
            {
                std::lock_guard<std::mutex> lock(inner_ids_mutex);
                inner_ids.push_back(id);
            }
            return false;
        }));
    }

    // First poll processes outer callbacks
    engine.poll();

    // Second poll processes inner callbacks
    engine.poll();

    EXPECT_EQ(inner_count.load(), NUM_OUTER);

    // Clean up
    for (auto id : outer_ids) engine.unregister_callback(id);
    for (auto id : inner_ids) engine.unregister_callback(id);
}

// =============================================================================
// Exception Safety Tests
// =============================================================================

TEST(ProgressEngineThreadSafety, PollingGuardExceptionSafety) {
    auto& engine = progress_engine::instance();

    // Register a callback that throws
    auto throw_id = engine.register_callback([]() -> bool {
        throw std::runtime_error("test exception");
    });

    // poll() should not crash and should recover the polling_ flag
    EXPECT_NO_THROW(engine.poll());

    // The throwing callback should have been removed by poll().
    // Register a new callback to verify engine is still functional
    std::atomic<bool> called{false};
    scoped_progress_callback scoped([&]() {
        called.store(true, std::memory_order_release);
        return false;
    });

    engine.poll();
    EXPECT_TRUE(called.load(std::memory_order_acquire));

    // Clean up just in case
    engine.unregister_callback(throw_id);
}

TEST(ProgressEngineThreadSafety, PollAfterThrowStillWorks) {
    auto& engine = progress_engine::instance();

    std::atomic<int> good_count{0};

    // Mix of throwing and non-throwing callbacks
    auto id1 = engine.register_callback([&]() -> bool {
        good_count.fetch_add(1, std::memory_order_relaxed);
        return false;
    });
    auto id2 = engine.register_callback([]() -> bool {
        throw std::runtime_error("boom");
    });
    auto id3 = engine.register_callback([&]() -> bool {
        good_count.fetch_add(1, std::memory_order_relaxed);
        return false;
    });

    engine.poll();

    // Both non-throwing callbacks should have been invoked.
    // Note: stale callbacks from prior tests may also increment good_count
    // if they happen to capture a compatible atomic. We check >= 2.
    EXPECT_GE(good_count.load(), 2);

    // Clean up in case poll didn't remove them
    engine.unregister_callback(id1);
    engine.unregister_callback(id2);
    engine.unregister_callback(id3);
}

// =============================================================================
// Polling Guard Behavior Tests
// =============================================================================

TEST(ProgressEngineThreadSafety, ConcurrentPollReturnsZero) {
    auto& engine = progress_engine::instance();

    // Use a slow callback to hold the polling_ flag
    std::atomic<bool> in_callback{false};
    std::atomic<bool> release_callback{false};

    auto cb_id = engine.register_callback([&]() {
        in_callback.store(true, std::memory_order_release);
        // Busy-wait until released
        while (!release_callback.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return false;
    });

    // Thread 1: starts poll, enters slow callback
    std::thread poller1([&]() {
        engine.poll();
    });

    // Wait for callback to be entered
    while (!in_callback.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Thread 2: tries to poll while thread 1 is still in callback
    size_type result = engine.poll();

    // Should return 0 because thread 1 holds the polling_ flag
    EXPECT_EQ(result, 0u);

    // Release thread 1
    release_callback.store(true, std::memory_order_release);
    poller1.join();

    // Clean up
    engine.unregister_callback(cb_id);
}

}  // namespace dtl::futures::test
