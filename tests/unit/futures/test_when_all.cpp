// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_when_all.cpp
/// @brief Unit tests for when_all combinator
/// @details Tests variadic and vector forms of when_all with progress-based completion.

#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace dtl::test {

// =============================================================================
// Variadic when_all Tests
// =============================================================================

TEST(WhenAllVariadicTest, SingleFuture) {
    auto f = make_ready_distributed_future(42);
    auto combined = when_all(std::move(f));

    // Drive progress
    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    EXPECT_EQ(std::get<0>(result), 42);
}

TEST(WhenAllVariadicTest, TwoFutures) {
    auto f1 = make_ready_distributed_future(10);
    auto f2 = make_ready_distributed_future(20);

    auto combined = when_all(std::move(f1), std::move(f2));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    EXPECT_EQ(std::get<0>(result), 10);
    EXPECT_EQ(std::get<1>(result), 20);
}

TEST(WhenAllVariadicTest, ThreeFutures) {
    auto f1 = make_ready_distributed_future(1);
    auto f2 = make_ready_distributed_future(2);
    auto f3 = make_ready_distributed_future(3);

    auto combined = when_all(std::move(f1), std::move(f2), std::move(f3));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    EXPECT_EQ(std::get<0>(result), 1);
    EXPECT_EQ(std::get<1>(result), 2);
    EXPECT_EQ(std::get<2>(result), 3);
}

TEST(WhenAllVariadicTest, HeterogeneousTypes) {
    auto f1 = make_ready_distributed_future(42);
    auto f2 = make_ready_distributed_future(std::string("hello"));
    auto f3 = make_ready_distributed_future(3.14);

    auto combined = when_all(std::move(f1), std::move(f2), std::move(f3));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    EXPECT_EQ(std::get<0>(result), 42);
    EXPECT_EQ(std::get<1>(result), "hello");
    EXPECT_DOUBLE_EQ(std::get<2>(result), 3.14);
}

TEST(WhenAllVariadicTest, DeferredCompletion) {
    distributed_promise<int> p1;
    distributed_promise<int> p2;

    auto f1 = p1.get_future();
    auto f2 = p2.get_future();

    auto combined = when_all(std::move(f1), std::move(f2));

    // Not ready yet
    EXPECT_FALSE(combined.is_ready());

    // Complete first promise
    p1.set_value(10);
    futures::make_progress();
    EXPECT_FALSE(combined.is_ready());

    // Complete second promise
    p2.set_value(20);

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    EXPECT_EQ(std::get<0>(result), 10);
    EXPECT_EQ(std::get<1>(result), 20);
}

// =============================================================================
// Vector when_all Tests
// =============================================================================

TEST(WhenAllVectorTest, EmptyVector) {
    std::vector<distributed_future<int>> futures;
    auto combined = when_all(std::move(futures));

    EXPECT_TRUE(combined.is_ready());

    auto result = combined.get();
    EXPECT_TRUE(result.empty());
}

TEST(WhenAllVectorTest, SingleElement) {
    std::vector<distributed_future<int>> futures;
    futures.push_back(make_ready_distributed_future(42));

    auto combined = when_all(std::move(futures));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 42);
}

TEST(WhenAllVectorTest, MultipleElements) {
    std::vector<distributed_future<int>> futures;
    futures.push_back(make_ready_distributed_future(1));
    futures.push_back(make_ready_distributed_future(2));
    futures.push_back(make_ready_distributed_future(3));
    futures.push_back(make_ready_distributed_future(4));

    auto combined = when_all(std::move(futures));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
    EXPECT_EQ(result[3], 4);
}

TEST(WhenAllVectorTest, DeferredCompletion) {
    distributed_promise<int> p1, p2, p3;

    std::vector<distributed_future<int>> futures;
    futures.push_back(p1.get_future());
    futures.push_back(p2.get_future());
    futures.push_back(p3.get_future());

    auto combined = when_all(std::move(futures));

    EXPECT_FALSE(combined.is_ready());

    p1.set_value(10);
    futures::make_progress();
    EXPECT_FALSE(combined.is_ready());

    p2.set_value(20);
    futures::make_progress();
    EXPECT_FALSE(combined.is_ready());

    p3.set_value(30);

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    auto result = combined.get();
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 20);
    EXPECT_EQ(result[2], 30);
}

// =============================================================================
// Void Futures Tests
// =============================================================================

TEST(WhenAllVoidTest, VectorOfVoid) {
    std::vector<distributed_future<void>> futures;
    futures.push_back(make_ready_distributed_future());
    futures.push_back(make_ready_distributed_future());

    auto combined = when_all(std::move(futures));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    // Should not throw
    combined.get();
}

TEST(WhenAllVoidTest, EmptyVoidVector) {
    std::vector<distributed_future<void>> futures;
    auto combined = when_all(std::move(futures));

    EXPECT_TRUE(combined.is_ready());
    combined.get();  // Should not throw
}

// =============================================================================
// Error Propagation Tests
// =============================================================================

TEST(WhenAllErrorTest, PropagatesError) {
    auto f1 = make_ready_distributed_future(10);
    auto f2 = make_failed_distributed_future<int>(
        status(status_code::operation_failed, no_rank, "Test error"));

    auto combined = when_all(std::move(f1), std::move(f2));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    EXPECT_THROW(combined.get(), std::runtime_error);
}

TEST(WhenAllErrorTest, VectorPropagatesError) {
    std::vector<distributed_future<int>> futures;
    futures.push_back(make_ready_distributed_future(1));
    futures.push_back(make_failed_distributed_future<int>(
        status(status_code::operation_failed, no_rank, "Error")));

    auto combined = when_all(std::move(futures));

    while (!combined.is_ready()) {
        futures::make_progress();
    }

    EXPECT_THROW(combined.get(), std::runtime_error);
}

}  // namespace dtl::test
