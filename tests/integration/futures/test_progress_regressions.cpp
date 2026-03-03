// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_progress_regressions.cpp
/// @brief Regression tests for progress engine fairness (Phase 07)
/// @details Tests that reproduce known issues and prevent regressions

#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

namespace dtl::futures::test {

// =============================================================================
// Regression: Progress Without Explicit Polling
// =============================================================================
// KNOWN_ISSUES: "Progress engine may not advance in all scenarios without
//                explicit polling"

TEST(ProgressRegressionsTest, ProgressAdvancesWithExplicitPoll) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Set value in background
    std::thread setter([&promise] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        promise.set_value(42);
    });

    // Explicitly poll until ready (should not hang)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!future.is_ready()) {
        poll();
        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "Timed out waiting for future with explicit polling";
        }
        std::this_thread::yield();
    }

    EXPECT_EQ(future.get(), 42);
    setter.join();
}

// =============================================================================
// Regression: Long Callbacks Blocking Progress
// =============================================================================
// KNOWN_ISSUES: "Long-running callbacks may delay other completions"

TEST(ProgressRegressionsTest, LongCallbacksDoNotStarveOthers) {
    // Register a callback that simulates long work
    std::atomic<bool> long_running{true};
    std::atomic<int> long_iterations{0};

    auto long_id = progress_engine::instance().register_callback([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++long_iterations;
        return long_running.load();
    });

    // Create futures that should complete despite long callback
    std::vector<distributed_promise<int>> promises(5);
    std::vector<distributed_future<int>> futures;

    for (size_t i = 0; i < promises.size(); ++i) {
        futures.push_back(promises[i].get_future());
    }

    // Set values immediately
    for (size_t i = 0; i < promises.size(); ++i) {
        promises[i].set_value(static_cast<int>(i));
    }

    // Poll and verify futures complete
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        poll();

        bool all_ready = true;
        for (const auto& f : futures) {
            if (!f.is_ready()) {
                all_ready = false;
                break;
            }
        }

        if (all_ready) break;
        std::this_thread::yield();
    }

    // All futures should be ready
    for (size_t i = 0; i < futures.size(); ++i) {
        EXPECT_TRUE(futures[i].is_ready());
        EXPECT_EQ(futures[i].get(), static_cast<int>(i));
    }

    // Stop long callback
    long_running = false;
    progress_engine::instance().unregister_callback(long_id);
}

// =============================================================================
// Regression: Timeout Protection
// =============================================================================
// KNOWN_ISSUES: "30-second timeout protects against indefinite hangs in CI"

TEST(ProgressRegressionsTest, TimeoutProtectionWorks) {
    // Save original config
    auto original = global_timeout_config();

    // Set very short timeout for testing
    auto config = timeout_config::defaults();
    config.default_wait_timeout = std::chrono::milliseconds(100);
    config.ci_wait_timeout = std::chrono::milliseconds(100);
    config.enable_timeout_diagnostics = true;
    set_global_timeout_config(config);

    // Create a future that will never complete
    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Wait should throw timeout exception
    bool timed_out = false;
    try {
        future.wait();
    } catch (const timeout_exception&) {
        timed_out = true;
    } catch (const std::runtime_error& e) {
        // Also acceptable
        timed_out = std::string(e.what()).find("timeout") != std::string::npos;
    }

    EXPECT_TRUE(timed_out);

    // Restore config
    set_global_timeout_config(original);

    // Clean up
    promise.set_value(0);
}

// =============================================================================
// Fairness: Many Futures Completing Concurrently
// =============================================================================

TEST(ProgressRegressionsTest, ManyFuturesConcurrent) {
    constexpr std::size_t num_futures = 50;

    std::vector<distributed_promise<int>> promises(num_futures);
    std::vector<distributed_future<int>> futures;

    for (std::size_t i = 0; i < num_futures; ++i) {
        futures.push_back(promises[i].get_future());
    }

    // Set values from multiple threads
    std::vector<std::thread> setters;
    for (std::size_t i = 0; i < num_futures; ++i) {
        setters.emplace_back([&promises, i] {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(i % 10)));
            promises[i].set_value(static_cast<int>(i) * 100);
        });
    }

    // Poll until all ready
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::size_t ready_count = 0;
    while (ready_count < num_futures) {
        poll();
        ready_count = 0;
        for (const auto& f : futures) {
            if (f.is_ready()) ++ready_count;
        }

        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "Timed out waiting for " << num_futures << " futures. "
                   << ready_count << " completed.";
        }
        std::this_thread::yield();
    }

    // Verify all values
    for (std::size_t i = 0; i < num_futures; ++i) {
        EXPECT_EQ(futures[i].get(), static_cast<int>(i) * 100);
    }

    for (auto& t : setters) {
        t.join();
    }
}

// =============================================================================
// Fairness: Continuation Chains
// =============================================================================

TEST(ProgressRegressionsTest, ContinuationChainsComplete) {
    distributed_promise<int> initial_promise;
    auto initial_future = initial_promise.get_future();

    // Create a chain of continuations
    auto chain1 = initial_future.then([](int x) { return x * 2; });
    auto chain2 = chain1.then([](int x) { return x + 10; });
    auto chain3 = chain2.then([](int x) { return x * 3; });

    // Set initial value
    initial_promise.set_value(5);

    // Poll until final chain is ready
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!chain3.is_ready()) {
        poll();
        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "Continuation chain did not complete";
        }
        std::this_thread::yield();
    }

    // Expected: ((5 * 2) + 10) * 3 = 60
    EXPECT_EQ(chain3.get(), 60);
}

// =============================================================================
// Fairness: Mixed Wait/Test Patterns
// =============================================================================

TEST(ProgressRegressionsTest, MixedWaitAndTestPatterns) {
    constexpr std::size_t num_futures = 10;

    std::vector<distributed_promise<int>> promises(num_futures);
    std::vector<distributed_future<int>> futures;

    for (std::size_t i = 0; i < num_futures; ++i) {
        futures.push_back(promises[i].get_future());
    }

    // Setter thread
    std::thread setter([&promises, num_futures] {
        for (std::size_t i = 0; i < num_futures; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            promises[i].set_value(static_cast<int>(i));
        }
    });

    // Mix of wait_for and is_ready patterns
    for (std::size_t i = 0; i < num_futures; ++i) {
        if (i % 2 == 0) {
            // Use wait_for
            auto status = futures[i].wait_for(std::chrono::seconds(5));
            EXPECT_EQ(status, future_status::ready);
        } else {
            // Use polling with is_ready
            while (!futures[i].is_ready()) {
                poll();
                std::this_thread::yield();
            }
        }
        EXPECT_EQ(futures[i].get(), static_cast<int>(i));
    }

    setter.join();
}

// =============================================================================
// Fairness: Progress Under Limited Polling
// =============================================================================

TEST(ProgressRegressionsTest, ProgressUnderLimitedPolling) {
    constexpr std::size_t num_futures = 20;
    constexpr int max_polls = 100;

    std::vector<distributed_promise<int>> promises(num_futures);
    std::vector<distributed_future<int>> futures;

    for (std::size_t i = 0; i < num_futures; ++i) {
        futures.push_back(promises[i].get_future());
        promises[i].set_value(static_cast<int>(i));  // Set immediately
    }

    // Limited polling
    for (int poll_count = 0; poll_count < max_polls; ++poll_count) {
        poll();

        // Check if all ready
        bool all_ready = true;
        for (const auto& f : futures) {
            if (!f.is_ready()) {
                all_ready = false;
                break;
            }
        }
        if (all_ready) break;
    }

    // All should complete within limited polls
    for (std::size_t i = 0; i < num_futures; ++i) {
        EXPECT_TRUE(futures[i].is_ready());
    }
}

// =============================================================================
// Stress: Rapid Create/Complete Cycles
// =============================================================================

TEST(ProgressRegressionsTest, RapidCreateCompleteCycles) {
    constexpr int num_cycles = 100;

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        distributed_promise<int> promise;
        auto future = promise.get_future();

        promise.set_value(cycle);
        poll();

        EXPECT_TRUE(future.is_ready());
        EXPECT_EQ(future.get(), cycle);
    }
}

// =============================================================================
// Stress: Interleaved Promises and Polls
// =============================================================================

TEST(ProgressRegressionsTest, InterleavedPromisesAndPolls) {
    std::vector<distributed_future<int>> pending_futures;
    std::vector<distributed_promise<int>> pending_promises;

    for (int i = 0; i < 50; ++i) {
        // Create new promise/future
        pending_promises.emplace_back();
        pending_futures.push_back(pending_promises.back().get_future());

        // Set value for random earlier promise
        if (!pending_promises.empty() && i > 0) {
            size_t idx = static_cast<size_t>(i) % pending_promises.size();
            if (!pending_futures[idx].is_ready()) {
                pending_promises[idx].set_value(static_cast<int>(idx * 10));
            }
        }

        // Poll
        poll();
    }

    // Complete remaining
    for (size_t i = 0; i < pending_promises.size(); ++i) {
        if (!pending_futures[i].is_ready()) {
            pending_promises[i].set_value(static_cast<int>(i * 10));
        }
    }

    // Drain
    drain_progress();

    // All should be ready
    for (const auto& f : pending_futures) {
        EXPECT_TRUE(f.is_ready());
    }
}

}  // namespace dtl::futures::test
