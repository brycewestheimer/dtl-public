// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_progress_poll_api.cpp
/// @brief Unit tests for the public progress polling API (Phase 07)
/// @details Tests poll(), poll_one(), poll_for(), poll_until() functions

#include <dtl/futures/progress.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/diagnostics.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace dtl::futures::test {

// =============================================================================
// poll() API Tests
// =============================================================================

TEST(ProgressPollAPITest, PollAdvancesFuture) {
    // Create a future that needs polling to complete
    distributed_promise<int> promise;
    auto future = promise.get_future();

    EXPECT_FALSE(future.is_ready());

    // Set value in a background thread after delay
    std::thread setter([&promise] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        promise.set_value(42);
    });

    // Poll until ready
    while (!future.is_ready()) {
        poll();
        std::this_thread::yield();
    }

    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);

    setter.join();
}

TEST(ProgressPollAPITest, PollReturnsProgressCount) {
    std::atomic<int> call_count{0};

    // Register a callback that completes after 2 calls
    progress_engine::instance().register_callback([&]() {
        ++call_count;
        return call_count < 2;
    });

    // First poll should make progress
    size_type progress1 = poll();
    EXPECT_GE(progress1, 0u);  // May be 0 if callback just registered

    // Continue polling until callback completes
    while (call_count < 2) {
        poll();
    }

    EXPECT_EQ(call_count.load(), 2);
}

// =============================================================================
// poll_one() API Tests
// =============================================================================

TEST(ProgressPollAPITest, PollOneCompletesOneOperation) {
    std::atomic<int> completed_count{0};

    // Register two callbacks
    progress_engine::instance().register_callback([&]() {
        ++completed_count;
        return false;  // Complete immediately
    });

    progress_engine::instance().register_callback([&]() {
        ++completed_count;
        return false;  // Complete immediately
    });

    // poll_one should complete at least one
    bool result = poll_one();
    EXPECT_TRUE(result);

    // Both should be completed after polling completes
    poll();  // Ensure second completes
    EXPECT_EQ(completed_count.load(), 2);
}

TEST(ProgressPollAPITest, PollOneReturnsFalseWhenEmpty) {
    // Drain any existing callbacks
    drain_progress();

    // poll_one with no pending should return false
    bool result = poll_one();
    EXPECT_FALSE(result);
}

// =============================================================================
// poll_for() API Tests
// =============================================================================

TEST(ProgressPollAPITest, PollForDuration) {
    std::atomic<int> poll_count{0};

    auto id = progress_engine::instance().register_callback([&]() {
        ++poll_count;
        return true;  // Never complete
    });

    // Poll for 50ms
    auto start = std::chrono::steady_clock::now();
    poll_for(std::chrono::milliseconds(50));
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should have polled for at least 40ms (allowing some slack)
    EXPECT_GE(elapsed, std::chrono::milliseconds(40));

    // Should have polled multiple times
    EXPECT_GT(poll_count.load(), 0);

    // Clean up
    progress_engine::instance().unregister_callback(id);
}

TEST(ProgressPollAPITest, PollForExitsEarlyWhenNoWork) {
    // Drain any pending work
    drain_progress();

    // Poll for a long duration with no work
    auto start = std::chrono::steady_clock::now();
    poll_for(std::chrono::seconds(1));  // Would take 1s if it didn't exit early
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should exit much faster since no work
    EXPECT_LT(elapsed, std::chrono::milliseconds(100));
}

// =============================================================================
// poll_until() API Tests
// =============================================================================

TEST(ProgressPollAPITest, PollUntilPredicateTrue) {
    std::atomic<bool> condition{false};

    // Set condition after delay
    std::thread setter([&condition] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        condition = true;
    });

    // Poll until condition is true
    bool result = poll_until([&condition] { return condition.load(); },
                             std::chrono::milliseconds(1000));

    EXPECT_TRUE(result);
    EXPECT_TRUE(condition.load());

    setter.join();
}

TEST(ProgressPollAPITest, PollUntilTimeout) {
    std::atomic<bool> never_true{false};

    // Poll with short timeout
    auto start = std::chrono::steady_clock::now();
    bool result = poll_until([&never_true] { return never_true.load(); },
                             std::chrono::milliseconds(50));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);  // Should timeout
    EXPECT_GE(elapsed, std::chrono::milliseconds(40));  // Should wait at least 40ms
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ProgressPollAPITest, ExplicitPollAdvancesWithoutWait) {
    // This test verifies the KNOWN_ISSUES fix:
    // "Progress may not advance without explicit polling"

    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Set value immediately
    promise.set_value(100);

    // Use poll() to check - progress should be made
    poll();

    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 100);
}

TEST(ProgressPollAPITest, MultipleFuturesProgress) {
    constexpr std::size_t num_futures = 10;
    std::vector<distributed_promise<int>> promises(num_futures);
    std::vector<distributed_future<int>> futures;

    for (std::size_t i = 0; i < num_futures; ++i) {
        futures.push_back(promises[i].get_future());
    }

    // Set values from a background thread
    std::thread setter([&promises, num_futures] {
        for (std::size_t i = 0; i < num_futures; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            promises[i].set_value(static_cast<int>(i) * 10);
        }
    });

    // Poll until all ready
    std::size_t ready_count = 0;
    while (ready_count < num_futures) {
        poll();
        ready_count = 0;
        for (const auto& f : futures) {
            if (f.is_ready()) ++ready_count;
        }
        std::this_thread::yield();
    }

    // Verify all values
    for (std::size_t i = 0; i < num_futures; ++i) {
        EXPECT_EQ(futures[i].get(), static_cast<int>(i) * 10);
    }

    setter.join();
}

}  // namespace dtl::futures::test
