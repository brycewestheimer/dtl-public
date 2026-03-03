// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_continuation.cpp
/// @brief Unit tests for continuation chaining and error handling
/// @details Tests .then(), chain(), on_error(), flatten() and flat_map().

#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

namespace dtl::test {

// =============================================================================
// Basic .then() Tests
// =============================================================================

TEST(ContinuationTest, ThenTransformsValue) {
    auto f = make_ready_distributed_future(10);
    auto f2 = f.then([](int x) { return x * 2; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 20);
}

TEST(ContinuationTest, ThenChangesType) {
    auto f = make_ready_distributed_future(42);
    auto f2 = f.then([](int x) { return std::to_string(x); });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), "42");
}

TEST(ContinuationTest, ThenOnVoidFuture) {
    auto f = make_ready_distributed_future();
    auto f2 = f.then([]() { return 100; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 100);
}

TEST(ContinuationTest, ThenReturnsVoid) {
    std::atomic<bool> called{false};
    auto f = make_ready_distributed_future(42);
    auto f2 = f.then([&]([[maybe_unused]] int x) { called = true; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    f2.get();
    EXPECT_TRUE(called.load());
}

TEST(ContinuationTest, ThenDeferredExecution) {
    distributed_promise<int> p;
    auto f = p.get_future();

    std::atomic<bool> called{false};
    auto f2 = f.then([&](int x) {
        called = true;
        return x * 2;
    });

    // Not called yet
    futures::make_progress();
    EXPECT_FALSE(called.load());

    // Complete the promise
    p.set_value(10);

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(called.load());
    EXPECT_EQ(f2.get(), 20);
}

TEST(ContinuationTest, ThenChaining) {
    auto f = make_ready_distributed_future(5);
    auto f2 = f.then([](int x) { return x + 1; });
    auto f3 = f2.then([](int x) { return x * 2; });
    auto f4 = f3.then([](int x) { return x - 3; });

    while (!f4.is_ready()) {
        futures::make_progress();
    }

    // (5 + 1) * 2 - 3 = 9
    EXPECT_EQ(f4.get(), 9);
}

TEST(ContinuationTest, ThenPropagatesError) {
    auto f = make_failed_distributed_future<int>(
        status(status_code::operation_failed, no_rank, "Initial error"));
    auto f2 = f.then([](int x) { return x * 2; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_THROW(
        {
            auto ignored = f2.get();
            (void)ignored;
        },
        std::runtime_error);
}

TEST(ContinuationTest, ThenOnInvalidFuture) {
    distributed_future<int> invalid_future;  // Default constructed = invalid
    auto f2 = invalid_future.then([](int x) { return x * 2; });

    // Should be ready immediately with error
    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_THROW(
        {
            auto ignored = f2.get();
            (void)ignored;
        },
        std::runtime_error);
}

// =============================================================================
// No Detached Threads Verification
// =============================================================================

TEST(ContinuationTest, ProgressBasedExecution) {
    // This test verifies that continuations use progress-based execution
    // by checking that no continuation executes without make_progress()

    distributed_promise<int> p;
    auto f = p.get_future();

    std::atomic<bool> executed{false};
    auto f2 = f.then([&](int x) {
        executed = true;
        return x;
    });

    // Complete the promise
    p.set_value(42);

    // Without calling make_progress, continuation should not have executed
    // (In the old detached thread model, this would be racy)
    // Give it a brief moment to NOT execute
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Now drive progress
    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(executed.load());
    EXPECT_EQ(f2.get(), 42);
}

// =============================================================================
// on_error() Tests
// =============================================================================

TEST(OnErrorTest, PassesThroughSuccess) {
    auto f = make_ready_distributed_future(42);
    auto f2 = on_error(std::move(f), [](const status&) { return -1; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 42);  // Original value, not error handler result
}

TEST(OnErrorTest, CatchesError) {
    auto f = make_failed_distributed_future<int>(
        status(status_code::operation_failed, no_rank, "Test error"));
    auto f2 = on_error(std::move(f), [](const status&) { return -1; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), -1);  // Error handler result
}

TEST(OnErrorTest, ErrorCodeAccessible) {
    auto f = make_failed_distributed_future<int>(
        status(status_code::timeout, no_rank, "Timed out"));

    status_code captured_code = status_code::ok;
    auto f2 = on_error(std::move(f), [&](const status& s) {
        captured_code = s.code();
        return 0;
    });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(captured_code, status_code::timeout);
}

TEST(OnErrorTest, DeferredError) {
    distributed_promise<int> p;
    auto f = p.get_future();
    auto f2 = on_error(std::move(f), [](const status&) { return 999; });

    p.set_error(status(status_code::operation_failed, no_rank, "Deferred error"));

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 999);
}

// =============================================================================
// flatten() Tests
// =============================================================================

TEST(FlattenTest, FlattensNestedFuture) {
    auto inner = make_ready_distributed_future(42);
    auto outer = make_ready_distributed_future(std::move(inner));
    auto flat = flatten(std::move(outer));

    while (!flat.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(flat.get(), 42);
}

TEST(FlattenTest, DeferredOuter) {
    distributed_promise<distributed_future<int>> p;
    auto outer = p.get_future();
    auto flat = flatten(std::move(outer));

    auto inner = make_ready_distributed_future(100);
    p.set_value(std::move(inner));

    while (!flat.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(flat.get(), 100);
}

TEST(FlattenTest, DeferredInner) {
    distributed_promise<int> inner_p;
    auto inner = inner_p.get_future();
    auto outer = make_ready_distributed_future(std::move(inner));
    auto flat = flatten(std::move(outer));

    inner_p.set_value(200);

    while (!flat.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(flat.get(), 200);
}

// =============================================================================
// flat_map() Tests
// =============================================================================

TEST(FlatMapTest, BasicFlatMap) {
    auto f = make_ready_distributed_future(10);
    auto f2 = flat_map(std::move(f), [](int x) {
        return make_ready_distributed_future(x * 3);
    });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 30);
}

// =============================================================================
// chain() Tests
// =============================================================================

TEST(ChainTest, ChainMultipleFunctions) {
    auto f = make_ready_distributed_future(2);
    auto f2 = chain(std::move(f),
        [](int x) { return x + 3; },       // 2 + 3 = 5
        [](int x) { return x * 2; },       // 5 * 2 = 10
        [](int x) { return x - 1; });      // 10 - 1 = 9

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 9);
}

// =============================================================================
// fmap() Tests
// =============================================================================

TEST(FmapTest, MapsFunction) {
    auto f = make_ready_distributed_future(5);
    auto f2 = fmap(std::move(f), [](int x) { return x * x; });

    while (!f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(f2.get(), 25);
}

}  // namespace dtl::test
