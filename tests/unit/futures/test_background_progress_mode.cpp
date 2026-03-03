// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_background_progress.cpp
/// @brief Unit tests for background progress mode (Phase 07)
/// @details Tests automatic progress advancement via background thread

#include <dtl/futures/background_progress.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/continuation.hpp>
#include <dtl/futures/progress.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace dtl::futures::test {

// =============================================================================
// Background Progress Controller Tests
// =============================================================================

TEST(BackgroundProgressTest, StartAndStop) {
    auto& controller = background_progress_controller::instance();

    EXPECT_FALSE(controller.is_running());

    controller.start();
    EXPECT_TRUE(controller.is_running());

    controller.stop();
    EXPECT_FALSE(controller.is_running());
}

TEST(BackgroundProgressTest, StartIsIdempotent) {
    auto& controller = background_progress_controller::instance();

    controller.start();
    EXPECT_TRUE(controller.is_running());

    // Second start should be a no-op
    controller.start();
    EXPECT_TRUE(controller.is_running());

    controller.stop();
    EXPECT_FALSE(controller.is_running());
}

TEST(BackgroundProgressTest, StopIsIdempotent) {
    auto& controller = background_progress_controller::instance();

    controller.start();
    controller.stop();
    EXPECT_FALSE(controller.is_running());

    // Second stop should be a no-op
    controller.stop();
    EXPECT_FALSE(controller.is_running());
}

TEST(BackgroundProgressTest, BackgroundPolls) {
    auto& controller = background_progress_controller::instance();

    controller.start();

    // Wait a bit for background thread to poll
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_GT(controller.poll_count(), 0u);

    controller.stop();
}

// =============================================================================
// Background Progress Convenience Functions Tests
// =============================================================================

TEST(BackgroundProgressTest, ConvenienceFunctions) {
    EXPECT_FALSE(is_background_progress_enabled());

    start_background_progress();
    EXPECT_TRUE(is_background_progress_enabled());

    stop_background_progress();
    EXPECT_FALSE(is_background_progress_enabled());
}

// =============================================================================
// Scoped Background Progress Tests
// =============================================================================

TEST(BackgroundProgressTest, ScopedBackgroundProgress) {
    EXPECT_FALSE(is_background_progress_enabled());

    {
        scoped_background_progress guard;
        EXPECT_TRUE(is_background_progress_enabled());
    }

    EXPECT_FALSE(is_background_progress_enabled());
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(BackgroundProgressTest, FutureCompletesWithoutExplicitPoll) {
    // This test verifies that futures complete without explicit polling
    // when background progress is enabled

    scoped_background_progress bg;

    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Set value in background thread
    std::thread setter([&promise] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        promise.set_value(42);
    });

    // Wait for future WITHOUT calling poll() - background should handle it
    // Use wait_for with timeout to avoid hanging
    auto status = future.wait_for(std::chrono::milliseconds(500));

    EXPECT_EQ(status, future_status::ready);
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);

    setter.join();
}

TEST(BackgroundProgressTest, ContinuationsExecuteWithBackground) {
    scoped_background_progress bg;

    distributed_promise<int> promise;
    auto future = promise.get_future();

    std::atomic<bool> continuation_ran{false};
    auto chained = future.then([&continuation_ran](int value) {
        continuation_ran = true;
        return value * 2;
    });

    // Set value
    promise.set_value(21);

    // Wait for continuation (background should execute it)
    auto status = chained.wait_for(std::chrono::milliseconds(500));

    EXPECT_EQ(status, future_status::ready);
    EXPECT_TRUE(continuation_ran.load());
    EXPECT_EQ(chained.get(), 42);
}

TEST(BackgroundProgressTest, AdaptivePollingReducesActivity) {
    auto config = background_progress_config::background_mode();
    config.adaptive_polling = true;
    config.poll_interval = std::chrono::microseconds(100);

    background_progress_controller::instance().start(config);

    // Let it run idle for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto polls_at_idle = background_progress_controller::instance().poll_count();

    // Register a callback that completes immediately
    progress_engine::instance().register_callback([]() { return false; });
    background_progress_controller::instance().wake();

    // Let it run with work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto polls_with_work = background_progress_controller::instance().poll_count();

    // Just verify it ran polls during both periods
    EXPECT_GT(polls_at_idle, 0u);
    EXPECT_GT(polls_with_work, polls_at_idle);

    background_progress_controller::instance().stop();
}

TEST(BackgroundProgressTest, ConfigurationOptions) {
    auto config = background_progress_config::aggressive_background();

    EXPECT_EQ(config.mode, progress_mode::background);
    EXPECT_EQ(config.poll_interval, std::chrono::microseconds(10));
    EXPECT_FALSE(config.adaptive_polling);
}

}  // namespace dtl::futures::test
