// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_future.cpp
/// @brief Unit tests for distributed_future
/// @details Tests for Phase 11.5: async future operations

#include <dtl/futures/futures.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/status.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace dtl::test {

// =============================================================================
// shared_state Tests
// =============================================================================

TEST(DistributedFutureTest, SharedStateDefaultConstruction) {
    shared_state<int> state;

    EXPECT_FALSE(state.is_ready());
}

TEST(DistributedFutureTest, SharedStateSetValue) {
    shared_state<int> state;

    state.set_value(42);

    EXPECT_TRUE(state.is_ready());
    EXPECT_EQ(state.get(), 42);
}

TEST(DistributedFutureTest, SharedStateSetError) {
    shared_state<int> state;

    state.set_error(status{status_code::invalid_argument, no_rank, "bad"});

    EXPECT_TRUE(state.is_ready());
    EXPECT_TRUE(state.has_error());
}

TEST(DistributedFutureTest, SharedStateVoid) {
    shared_state<void> state;

    EXPECT_FALSE(state.is_ready());

    state.set_value();

    EXPECT_TRUE(state.is_ready());
}

// =============================================================================
// distributed_future Construction Tests
// =============================================================================

TEST(DistributedFutureTest, DefaultConstruction) {
    distributed_future<int> future;

    EXPECT_FALSE(future.valid());
}

TEST(DistributedFutureTest, ValidFuture) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    EXPECT_TRUE(future.valid());
}

TEST(DistributedFutureTest, VoidFuture) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    EXPECT_TRUE(future.valid());
}

// =============================================================================
// distributed_promise Tests
// =============================================================================

TEST(DistributedFutureTest, PromiseGetFuture) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future.is_ready());
}

TEST(DistributedFutureTest, PromiseSetValue) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);
}

TEST(DistributedFutureTest, PromiseSetError) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_error(status{status_code::internal_error, no_rank, "failed"});

    EXPECT_TRUE(future.is_ready());
    EXPECT_THROW((void)future.get(), std::runtime_error);
}

TEST(DistributedFutureTest, VoidPromiseSetValue) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    promise.set_value();

    EXPECT_TRUE(future.is_ready());
    EXPECT_NO_THROW(future.get());
}

// =============================================================================
// Future get() Tests
// =============================================================================

TEST(DistributedFutureTest, GetValue) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(99);

    int result = future.get();
    EXPECT_EQ(result, 99);
}

TEST(DistributedFutureTest, GetInvalidFuture) {
    distributed_future<int> future;

    EXPECT_THROW((void)future.get(), std::runtime_error);
}

TEST(DistributedFutureTest, GetMovesValue) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    int result = future.get();
    EXPECT_EQ(result, 42);

    // Future should be invalid after get()
    EXPECT_FALSE(future.valid());
}

// =============================================================================
// Future get_result() Tests
// =============================================================================

TEST(DistributedFutureTest, GetResultSuccess) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    auto result = future.get_result();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(DistributedFutureTest, GetResultError) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_error(status{status_code::invalid_argument, no_rank, "bad"});

    auto result = future.get_result();
    EXPECT_FALSE(result.has_value());
}

TEST(DistributedFutureTest, GetResultInvalidFuture) {
    distributed_future<int> future;

    auto result = future.get_result();
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Future wait() Tests
// =============================================================================

TEST(DistributedFutureTest, Wait) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Set value in background
    std::thread t([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        promise.set_value(42);
    });

    future.wait();

    EXPECT_TRUE(future.is_ready());

    t.join();
}

TEST(DistributedFutureTest, WaitAlreadyReady) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    // Should return immediately
    future.wait();

    EXPECT_TRUE(future.is_ready());
}

// =============================================================================
// Future wait_for() Tests
// =============================================================================

TEST(DistributedFutureTest, WaitForReady) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    auto status = future.wait_for(std::chrono::milliseconds(100));
    EXPECT_EQ(status, future_status::ready);
}

TEST(DistributedFutureTest, WaitForTimeout) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Don't set value - should timeout
    auto status = future.wait_for(std::chrono::milliseconds(10));
    EXPECT_EQ(status, future_status::timeout);
}

TEST(DistributedFutureTest, WaitForError) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_error(status{status_code::internal_error});

    auto status = future.wait_for(std::chrono::milliseconds(100));
    EXPECT_EQ(status, future_status::error);
}

TEST(DistributedFutureTest, WaitForInvalidFuture) {
    distributed_future<int> future;

    auto status = future.wait_for(std::chrono::milliseconds(10));
    EXPECT_EQ(status, future_status::error);
}

// =============================================================================
// is_ready() Tests
// =============================================================================

TEST(DistributedFutureTest, IsReadyFalseInitially) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    EXPECT_FALSE(future.is_ready());
}

TEST(DistributedFutureTest, IsReadyAfterSetValue) {
    distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_value(42);

    EXPECT_TRUE(future.is_ready());
}

TEST(DistributedFutureTest, IsReadyInvalidFuture) {
    distributed_future<int> future;

    EXPECT_FALSE(future.is_ready());
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(DistributedFutureTest, MakeReadyFuture) {
    auto future = make_ready_distributed_future(42);

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);
}

TEST(DistributedFutureTest, MakeReadyVoidFuture) {
    auto future = make_ready_distributed_future();

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());
    EXPECT_NO_THROW(future.get());
}

TEST(DistributedFutureTest, MakeFailedFuture) {
    auto future = make_failed_distributed_future<int>(
        status{status_code::invalid_argument, no_rank, "test error"}
    );

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());
    EXPECT_THROW((void)future.get(), std::runtime_error);
}

TEST(DistributedFutureTest, MakeReadyFutureDouble) {
    auto future = make_ready_distributed_future(3.14);

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());
    EXPECT_DOUBLE_EQ(future.get(), 3.14);
}

TEST(DistributedFutureTest, MakeReadyFutureString) {
    auto future = make_ready_distributed_future(std::string("hello"));

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), "hello");
}

// =============================================================================
// Void Future Specialization Tests
// =============================================================================

TEST(DistributedFutureTest, VoidFutureValid) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    EXPECT_TRUE(future.valid());
}

TEST(DistributedFutureTest, VoidFutureGet) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    promise.set_value();

    EXPECT_NO_THROW(future.get());
}

TEST(DistributedFutureTest, VoidFutureGetResult) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    promise.set_value();

    auto result = future.get_result();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedFutureTest, VoidFutureError) {
    distributed_promise<void> promise;
    auto future = promise.get_future();

    promise.set_error(status{status_code::internal_error});

    auto result = future.get_result();
    EXPECT_FALSE(result.has_value());
}

}  // namespace dtl::test
