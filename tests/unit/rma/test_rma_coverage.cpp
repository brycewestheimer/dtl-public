// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rma_coverage.cpp
/// @brief Unit tests for RMA module: async requests, rma_batch, window_guard
/// @details Phase 14 T04: rma_request_state enum, fence_guard, lock_guard,
///          rma_batch, scoped epoch helpers, window_guard RAII lifecycle.

#include <dtl/rma/window_guard.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <dtl/rma/async_rma_request.hpp>
#pragma GCC diagnostic pop

#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

namespace dtl::test {

// =============================================================================
// Helper: create a valid local memory window for single-process testing
// =============================================================================

static memory_window make_test_window(size_type sz = 1024) {
    auto res = memory_window::allocate(sz);
    EXPECT_TRUE(res.has_value());
    auto win = std::move(res.value());
    // Zero-initialise the memory
    if (win.base()) {
        std::memset(win.base(), 0, sz);
    }
    return win;
}

// =============================================================================
// rma_request_state Enum Tests
// =============================================================================

TEST(RmaRequestStateTest, EnumValuesDistinct) {
    EXPECT_NE(static_cast<int>(rma::rma_request_state::pending),
              static_cast<int>(rma::rma_request_state::ready));
    EXPECT_NE(static_cast<int>(rma::rma_request_state::ready),
              static_cast<int>(rma::rma_request_state::error));
    EXPECT_NE(static_cast<int>(rma::rma_request_state::pending),
              static_cast<int>(rma::rma_request_state::error));
}

// =============================================================================
// Fence Guard Tests
// =============================================================================

TEST(FenceGuardTest, ConstructValid) {
    auto win = make_test_window();
    {
        rma::fence_guard guard(win);
        EXPECT_TRUE(guard.valid());
    }
    // Destructor calls fence – should not crash
}

TEST(FenceGuardTest, ManualFenceSucceeds) {
    auto win = make_test_window();
    rma::fence_guard guard(win);
    auto res = guard.fence();
    EXPECT_TRUE(res.has_value());
}

TEST(FenceGuardTest, InvalidWindowGuard) {
    memory_window invalid_win;
    rma::fence_guard guard(invalid_win);
    EXPECT_FALSE(guard.valid());
}

TEST(FenceGuardTest, ManualFenceOnInvalidReturnError) {
    memory_window invalid_win;
    rma::fence_guard guard(invalid_win);
    auto res = guard.fence();
    EXPECT_TRUE(res.has_error());
}

TEST(FenceGuardTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::fence_guard>);
    static_assert(!std::is_copy_assignable_v<rma::fence_guard>);
}

TEST(FenceGuardTest, NonMovable) {
    static_assert(!std::is_move_constructible_v<rma::fence_guard>);
    static_assert(!std::is_move_assignable_v<rma::fence_guard>);
}

// =============================================================================
// Lock Guard Tests
// =============================================================================

TEST(LockGuardTest, ConstructWithTarget) {
    auto win = make_test_window();
    {
        rma::lock_guard guard(0, win);
        EXPECT_TRUE(guard.locked());
        EXPECT_EQ(guard.target(), 0);
        EXPECT_FALSE(guard.is_all());
    }
}

TEST(LockGuardTest, ConstructLockAll) {
    auto win = make_test_window();
    {
        rma::lock_guard guard(win);
        EXPECT_TRUE(guard.locked());
        EXPECT_TRUE(guard.is_all());
    }
}

TEST(LockGuardTest, FlushSucceeds) {
    auto win = make_test_window();
    rma::lock_guard guard(0, win);
    ASSERT_TRUE(guard.locked());
    auto res = guard.flush();
    EXPECT_TRUE(res.has_value());
}

TEST(LockGuardTest, FlushLocalSucceeds) {
    auto win = make_test_window();
    rma::lock_guard guard(0, win);
    ASSERT_TRUE(guard.locked());
    auto res = guard.flush_local();
    EXPECT_TRUE(res.has_value());
}

TEST(LockGuardTest, FlushOnLockAllSucceeds) {
    auto win = make_test_window();
    rma::lock_guard guard(win);
    ASSERT_TRUE(guard.locked());
    auto res = guard.flush();
    EXPECT_TRUE(res.has_value());
}

TEST(LockGuardTest, FlushLocalOnLockAllSucceeds) {
    auto win = make_test_window();
    rma::lock_guard guard(win);
    ASSERT_TRUE(guard.locked());
    auto res = guard.flush_local();
    EXPECT_TRUE(res.has_value());
}

TEST(LockGuardTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::lock_guard>);
    static_assert(!std::is_copy_assignable_v<rma::lock_guard>);
}

TEST(LockGuardTest, NonMovable) {
    static_assert(!std::is_move_constructible_v<rma::lock_guard>);
    static_assert(!std::is_move_assignable_v<rma::lock_guard>);
}

// =============================================================================
// RMA Batch Tests
// =============================================================================

TEST(RmaBatchTest, ConstructValid) {
    auto win = make_test_window();
    {
        rma::rma_batch batch(win);
        EXPECT_TRUE(batch.valid());
    }
}

TEST(RmaBatchTest, OperationCountInitiallyZero) {
    auto win = make_test_window();
    rma::rma_batch batch(win);
    EXPECT_EQ(batch.operation_count(), 0u);
}

TEST(RmaBatchTest, ManualFlushAll) {
    auto win = make_test_window();
    rma::rma_batch batch(win);
    auto res = batch.flush_all();
    EXPECT_TRUE(res.has_value());
}

TEST(RmaBatchTest, ManualFlushTarget) {
    auto win = make_test_window();
    rma::rma_batch batch(win);
    auto res = batch.flush(0);
    EXPECT_TRUE(res.has_value());
}

TEST(RmaBatchTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::rma_batch>);
    static_assert(!std::is_copy_assignable_v<rma::rma_batch>);
}

TEST(RmaBatchTest, NonMovable) {
    static_assert(!std::is_move_constructible_v<rma::rma_batch>);
    static_assert(!std::is_move_assignable_v<rma::rma_batch>);
}

TEST(RmaBatchTest, InvalidWindowBatch) {
    memory_window invalid_win;
    rma::rma_batch batch(invalid_win);
    EXPECT_FALSE(batch.valid());
    auto res = batch.flush_all();
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// With Fence Epoch Helper Tests
// =============================================================================

TEST(WithFenceEpochTest, VoidLambdaSuccess) {
    auto win = make_test_window();
    int counter = 0;
    auto res = rma::with_fence_epoch(win, [&] { counter = 42; });
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(counter, 42);
}

TEST(WithFenceEpochTest, ResultLambdaSuccess) {
    auto win = make_test_window();
    auto res = rma::with_fence_epoch(win, [&]() -> result<void> {
        return result<void>{};
    });
    EXPECT_TRUE(res.has_value());
}

TEST(WithFenceEpochTest, ResultLambdaError) {
    auto win = make_test_window();
    auto res = rma::with_fence_epoch(win, [&]() -> result<void> {
        return status_code::internal_error;
    });
    EXPECT_TRUE(res.has_error());
}

TEST(WithFenceEpochTest, InvalidWindowReturnsError) {
    memory_window invalid_win;
    auto res = rma::with_fence_epoch(invalid_win, [&] { /* should not execute */ });
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// With Lock Epoch Helper Tests
// =============================================================================

TEST(WithLockEpochTest, VoidLambdaSuccess) {
    auto win = make_test_window();
    int counter = 0;
    auto res = rma::with_lock_epoch(0, win, rma_lock_mode::exclusive, [&] {
        counter = 99;
    });
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(counter, 99);
}

TEST(WithLockEpochTest, ResultLambdaError) {
    auto win = make_test_window();
    auto res = rma::with_lock_epoch(0, win, rma_lock_mode::exclusive, [&]() -> result<void> {
        return status_code::internal_error;
    });
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// RMA Request State Enum Tests (from async_rma_request.hpp)
// =============================================================================

TEST(RmaRequestStateTest, PendingIsZero) {
    EXPECT_EQ(static_cast<int>(rma::rma_request_state::pending), 0);
}

TEST(RmaRequestStateTest, ReadyIsOne) {
    EXPECT_EQ(static_cast<int>(rma::rma_request_state::ready), 1);
}

TEST(RmaRequestStateTest, ErrorIsTwo) {
    EXPECT_EQ(static_cast<int>(rma::rma_request_state::error), 2);
}

// =============================================================================
// Memory Window Factory Tests (supporting the RMA module)
// =============================================================================

TEST(MemoryWindowTest, AllocateSuccess) {
    auto res = memory_window::allocate(256);
    ASSERT_TRUE(res.has_value());
    auto win = std::move(res.value());
    EXPECT_TRUE(win.valid());
    EXPECT_GE(win.size(), 256u);
    EXPECT_NE(win.base(), nullptr);
}

TEST(MemoryWindowTest, AllocateZeroSizeValid) {
    auto res = memory_window::allocate(0);
    ASSERT_TRUE(res.has_value());
    // Zero-size window is valid but has zero size
}

TEST(MemoryWindowTest, CreateFromData) {
    std::vector<int> data(10, 42);
    auto res = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(res.has_value());
    auto win = std::move(res.value());
    EXPECT_TRUE(win.valid());
    EXPECT_EQ(win.base(), data.data());
}

TEST(MemoryWindowTest, DefaultConstructorInvalid) {
    memory_window win;
    EXPECT_FALSE(win.valid());
    EXPECT_EQ(win.base(), nullptr);
    EXPECT_EQ(win.size(), 0u);
}

TEST(MemoryWindowTest, MoveSemantics) {
    auto res = memory_window::allocate(128);
    ASSERT_TRUE(res.has_value());
    auto win1 = std::move(res.value());
    void* base = win1.base();

    memory_window win2 = std::move(win1);
    EXPECT_FALSE(win1.valid());
    EXPECT_TRUE(win2.valid());
    EXPECT_EQ(win2.base(), base);
}

TEST(MemoryWindowCoverageTest, FenceOnInvalidWindowFails) {
    memory_window win;
    auto res = win.fence();
    EXPECT_TRUE(res.has_error());
}

TEST(MemoryWindowCoverageTest, FlushOnInvalidWindowFails) {
    memory_window win;
    EXPECT_TRUE(win.flush(0).has_error());
    EXPECT_TRUE(win.flush_all().has_error());
    EXPECT_TRUE(win.flush_local(0).has_error());
}

}  // namespace dtl::test
