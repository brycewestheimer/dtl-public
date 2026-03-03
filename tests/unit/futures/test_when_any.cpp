// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_when_any.cpp
/// @brief Unit tests for when_any combinator
/// @details Tests when_any with progress-based completion.

#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace dtl::test {

// =============================================================================
// Basic when_any Tests
// =============================================================================

TEST(WhenAnyTest, SingleFuture) {
    std::vector<distributed_future<int>> futures;
    futures.push_back(make_ready_distributed_future(42));

    auto result_future = when_any(futures);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 0u);
    EXPECT_EQ(result.value, 42);
}

TEST(WhenAnyTest, FirstReady) {
    std::vector<distributed_future<int>> futures;
    futures.push_back(make_ready_distributed_future(10));
    futures.push_back(make_ready_distributed_future(20));
    futures.push_back(make_ready_distributed_future(30));

    auto result_future = when_any(futures);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    // Should complete with the first one
    EXPECT_EQ(result.index, 0u);
    EXPECT_EQ(result.value, 10);
}

TEST(WhenAnyTest, ReturnsFirstCompleted) {
    distributed_promise<int> p1, p2, p3;

    std::vector<distributed_future<int>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());
    futures.push_back(p3.get_future());

    auto result_future = when_any(futures);

    // Complete the second one first
    p2.set_value(200);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 1u);
    EXPECT_EQ(result.value, 200);

    // Clean up other promises to avoid dangling
    p1.set_value(100);
    p3.set_value(300);
}

TEST(WhenAnyTest, EmptyVectorError) {
    std::vector<distributed_future<int>> futures;
    auto result_future = when_any(futures);

    // Should be ready immediately with an error
    EXPECT_TRUE(result_future.is_ready());
    EXPECT_THROW(
        {
            auto ignored = result_future.get();
            (void)ignored;
        },
        std::runtime_error);
}

TEST(WhenAnyTest, IndexIsCorrect) {
    distributed_promise<int> p1, p2, p3;

    std::vector<distributed_future<int>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());
    futures.push_back(p3.get_future());

    auto result_future = when_any(futures);

    // Complete the last one first
    p3.set_value(300);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 2u);
    EXPECT_EQ(result.value, 300);

    p1.set_value(100);
    p2.set_value(200);
}

// =============================================================================
// Void Futures Tests
// =============================================================================

TEST(WhenAnyVoidTest, SingleVoidFuture) {
    std::vector<distributed_future<void>> futures;
    futures.push_back(make_ready_distributed_future());

    auto result_future = when_any(futures);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 0u);
}

TEST(WhenAnyVoidTest, MultipleVoidFutures) {
    distributed_promise<void> p1, p2;

    std::vector<distributed_future<void>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());

    auto result_future = when_any(futures);

    p2.set_value();

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 1u);

    p1.set_value();
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(WhenAnyErrorTest, PropagatesFirstError) {
    distributed_promise<int> p1, p2;

    std::vector<distributed_future<int>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());

    auto result_future = when_any(futures);

    // Set error on first to complete
    p1.set_error(status(status_code::operation_failed, no_rank, "Test error"));

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_THROW(
        {
            auto ignored = result_future.get();
            (void)ignored;
        },
        std::runtime_error);

    p2.set_value(200);
}

// =============================================================================
// Race Condition Tests
// =============================================================================

TEST(WhenAnyRaceTest, ConcurrentCompletion) {
    // Test that when_any handles concurrent completion correctly
    distributed_promise<int> p1, p2, p3;

    std::vector<distributed_future<int>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());
    futures.push_back(p3.get_future());

    auto result_future = when_any(futures);

    // Complete all at once (simulating race)
    p1.set_value(100);
    p2.set_value(200);
    p3.set_value(300);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    // Should get one of them (first to be detected)
    EXPECT_TRUE(result.index < 3);
    EXPECT_TRUE(result.value == 100 || result.value == 200 || result.value == 300);
}

// =============================================================================
// Use-After-Free Safety Tests (Phase 01 / CR-P01-T05)
// =============================================================================

TEST(WhenAnySafetyTest, VectorDestroyedBeforeCallbackFires) {
    // This test verifies that when_any safely owns the futures vector
    // so that destroying the caller's vector does not cause use-after-free.
    distributed_promise<int> p1;

    distributed_future<when_any_result<int>> result_future;
    {
        std::vector<distributed_future<int>> futures;
        futures.push_back(p1.get_future());
        result_future = when_any(std::move(futures));
        // futures vector goes out of scope here
    }

    // Complete the promise after the original vector is destroyed
    p1.set_value(42);

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    // Should not crash and value should be correct
    auto result = result_future.get();
    EXPECT_EQ(result.index, 0u);
    EXPECT_EQ(result.value, 42);
}

TEST(WhenAnySafetyTest, VoidVectorDestroyedBeforeCallbackFires) {
    distributed_promise<void> p1;

    distributed_future<when_any_result<void>> result_future;
    {
        std::vector<distributed_future<void>> futures;
        futures.push_back(p1.get_future());
        result_future = when_any(std::move(futures));
    }

    p1.set_value();

    while (!result_future.is_ready()) {
        futures::make_progress();
    }

    auto result = result_future.get();
    EXPECT_EQ(result.index, 0u);
}

}  // namespace dtl::test
