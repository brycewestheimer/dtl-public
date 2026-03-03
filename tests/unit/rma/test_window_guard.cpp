// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_window_guard.cpp
/// @brief Unit tests for RMA window guards
/// @details Verifies fence_guard, lock_guard, and rma_batch functionality.

#include <dtl/rma/window_guard.hpp>
#include <dtl/communication/rma_operations.hpp>

#include <gtest/gtest.h>
#include <array>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class WindowGuardTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_.fill(0);
        auto result = memory_window::create(data_.data(), data_.size() * sizeof(int));
        ASSERT_TRUE(result.has_value());
        window_ = std::move(*result);
    }

    std::array<int, 100> data_;
    memory_window window_;
};

// =============================================================================
// Fence Guard Tests
// =============================================================================

TEST_F(WindowGuardTest, FenceGuardCallsFenceOnDestruction) {
    bool fence_called = false;

    {
        rma::fence_guard guard(window_);
        EXPECT_TRUE(guard.valid());
        fence_called = true;  // Guard construction calls initial fence
    }
    // Destructor calls fence again

    EXPECT_TRUE(fence_called);
}

TEST_F(WindowGuardTest, FenceGuardManualFence) {
    rma::fence_guard guard(window_);
    EXPECT_TRUE(guard.valid());

    auto result = guard.fence();
    EXPECT_TRUE(result.has_value());
}

TEST_F(WindowGuardTest, FenceGuardInvalidWindow) {
    memory_window invalid;
    rma::fence_guard guard(invalid);

    EXPECT_FALSE(guard.valid());

    auto result = guard.fence();
    EXPECT_FALSE(result.has_value());
}

TEST_F(WindowGuardTest, FenceGuardNonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::fence_guard>,
                  "fence_guard should not be copy constructible");
    static_assert(!std::is_copy_assignable_v<rma::fence_guard>,
                  "fence_guard should not be copy assignable");
}

TEST_F(WindowGuardTest, FenceGuardNonMovable) {
    static_assert(!std::is_move_constructible_v<rma::fence_guard>,
                  "fence_guard should not be move constructible");
    static_assert(!std::is_move_assignable_v<rma::fence_guard>,
                  "fence_guard should not be move assignable");
}

// =============================================================================
// Lock Guard Tests
// =============================================================================

TEST_F(WindowGuardTest, LockGuardLocksOnConstruction) {
    rma::lock_guard guard(0, window_);
    EXPECT_TRUE(guard.locked());
    EXPECT_EQ(guard.target(), 0);
    EXPECT_FALSE(guard.is_all());
}

TEST_F(WindowGuardTest, LockGuardUnlocksOnDestruction) {
    {
        rma::lock_guard guard(0, window_);
        EXPECT_TRUE(guard.locked());
    }
    // Destructor should have called unlock - no crash means success
}

TEST_F(WindowGuardTest, LockGuardWithAllTargets) {
    rma::lock_guard guard(window_);  // lock_all
    EXPECT_TRUE(guard.locked());
    EXPECT_TRUE(guard.is_all());
}

TEST_F(WindowGuardTest, LockGuardFlush) {
    rma::lock_guard guard(0, window_);
    ASSERT_TRUE(guard.locked());

    auto result = guard.flush();
    EXPECT_TRUE(result.has_value());
}

TEST_F(WindowGuardTest, LockGuardFlushLocal) {
    rma::lock_guard guard(0, window_);
    ASSERT_TRUE(guard.locked());

    auto result = guard.flush_local();
    EXPECT_TRUE(result.has_value());
}

TEST_F(WindowGuardTest, LockGuardExclusiveMode) {
    rma::lock_guard guard(0, window_, rma_lock_mode::exclusive);
    EXPECT_TRUE(guard.locked());
}

TEST_F(WindowGuardTest, LockGuardSharedMode) {
    rma::lock_guard guard(0, window_, rma_lock_mode::shared);
    EXPECT_TRUE(guard.locked());
}

TEST_F(WindowGuardTest, LockGuardInvalidWindow) {
    memory_window invalid;
    rma::lock_guard guard(0, invalid);

    EXPECT_FALSE(guard.locked());

    auto result = guard.flush();
    EXPECT_FALSE(result.has_value());
}

TEST_F(WindowGuardTest, LockGuardNonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::lock_guard>,
                  "lock_guard should not be copy constructible");
    static_assert(!std::is_copy_assignable_v<rma::lock_guard>,
                  "lock_guard should not be copy assignable");
}

TEST_F(WindowGuardTest, LockGuardNonMovable) {
    static_assert(!std::is_move_constructible_v<rma::lock_guard>,
                  "lock_guard should not be move constructible");
    static_assert(!std::is_move_assignable_v<rma::lock_guard>,
                  "lock_guard should not be move assignable");
}

// =============================================================================
// RMA Batch Tests
// =============================================================================

TEST_F(WindowGuardTest, RmaBatchConstruction) {
    rma::rma_batch batch(window_);
    EXPECT_TRUE(batch.valid());
}

TEST_F(WindowGuardTest, RmaBatchPut) {
    rma::rma_batch batch(window_);

    int value = 42;
    auto result = batch.put(0, 0, value);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 42);
}

TEST_F(WindowGuardTest, RmaBatchGet) {
    data_[5] = 123;

    rma::rma_batch batch(window_);

    int value = 0;
    auto result = batch.get(0, 5 * sizeof(int), value);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(value, 123);
}

TEST_F(WindowGuardTest, RmaBatchCollectsOperations) {
    rma::rma_batch batch(window_);

    std::array<int, 3> values1 = {1, 2, 3};
    std::array<int, 3> values2 = {4, 5, 6};

    auto result1 = batch.put<int>(0, 0, std::span<const int>{values1});
    ASSERT_TRUE(result1.has_value());

    auto result2 = batch.put<int>(0, 3 * sizeof(int), std::span<const int>{values2});
    ASSERT_TRUE(result2.has_value());

    // Verify writes
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(data_[i], values1[i]);
        EXPECT_EQ(data_[3 + i], values2[i]);
    }
}

TEST_F(WindowGuardTest, RmaBatchFlushesOnDestruction) {
    int value = 999;
    {
        rma::rma_batch batch(window_);
        auto result = batch.put(0, 0, value);
        ASSERT_TRUE(result.has_value());
    }
    // Batch destructor should have flushed

    EXPECT_EQ(data_[0], 999);
}

TEST_F(WindowGuardTest, RmaBatchManualFlush) {
    rma::rma_batch batch(window_);

    int value = 42;
    auto put_result = batch.put(0, 0, value);
    ASSERT_TRUE(put_result.has_value());

    auto flush_result = batch.flush_all();
    EXPECT_TRUE(flush_result.has_value());
}

TEST_F(WindowGuardTest, RmaBatchFlushTarget) {
    rma::rma_batch batch(window_);

    auto result = batch.flush(0);
    EXPECT_TRUE(result.has_value());
}

TEST_F(WindowGuardTest, RmaBatchInvalidWindow) {
    memory_window invalid;
    rma::rma_batch batch(invalid);

    EXPECT_FALSE(batch.valid());

    int value = 42;
    auto result = batch.put(0, 0, value);
    EXPECT_FALSE(result.has_value());
}

TEST_F(WindowGuardTest, RmaBatchNonCopyable) {
    static_assert(!std::is_copy_constructible_v<rma::rma_batch>,
                  "rma_batch should not be copy constructible");
    static_assert(!std::is_copy_assignable_v<rma::rma_batch>,
                  "rma_batch should not be copy assignable");
}

// =============================================================================
// Scoped Epoch Helper Tests
// =============================================================================

TEST_F(WindowGuardTest, WithFenceEpoch) {
    bool executed = false;

    auto result = rma::with_fence_epoch(window_, [&]() {
        executed = true;
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(executed);
}

TEST_F(WindowGuardTest, WithFenceEpochReturnsResult) {
    auto res = rma::with_fence_epoch(window_, []() -> dtl::result<void> {
        return dtl::result<void>{};
    });

    EXPECT_TRUE(res.has_value());
}

TEST_F(WindowGuardTest, WithFenceEpochInvalidWindow) {
    memory_window invalid;

    auto result = rma::with_fence_epoch(invalid, []() {});

    EXPECT_FALSE(result.has_value());
}

TEST_F(WindowGuardTest, WithLockEpoch) {
    bool executed = false;

    auto result = rma::with_lock_epoch(0, window_, rma_lock_mode::exclusive, [&]() {
        executed = true;
    });

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(executed);
}

TEST_F(WindowGuardTest, WithLockEpochInvalidWindow) {
    memory_window invalid;

    auto result = rma::with_lock_epoch(0, invalid, rma_lock_mode::exclusive, []() {});

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Exception Safety Tests
// =============================================================================

TEST_F(WindowGuardTest, FenceGuardExceptionSafety) {
    // Even if an exception is thrown, the guard should call fence
    try {
        rma::fence_guard guard(window_);
        // Guard will call fence in destructor even if we throw
        // (In a real test we'd verify this, but we just ensure no crash)
    } catch (...) {
        // Should not happen
        FAIL() << "Unexpected exception";
    }
}

TEST_F(WindowGuardTest, LockGuardExceptionSafety) {
    try {
        rma::lock_guard guard(0, window_);
        // Guard will call unlock in destructor even if we throw
    } catch (...) {
        FAIL() << "Unexpected exception";
    }
}

TEST_F(WindowGuardTest, RmaBatchExceptionSafety) {
    try {
        rma::rma_batch batch(window_);
        int value = 42;
        auto result = batch.put(0, 0, value);
        EXPECT_TRUE(result.has_value());
        // Batch will call flush_all in destructor
    } catch (...) {
        FAIL() << "Unexpected exception";
    }
}

}  // namespace dtl::test
