// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_memory_window.cpp
/// @brief Unit tests for memory_window abstraction
/// @details Verifies RAII memory window functionality.

#include <dtl/communication/memory_window.hpp>

#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace dtl::test {

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(MemoryWindowTest, DefaultConstruction) {
    memory_window win;
    EXPECT_FALSE(win.valid());
    EXPECT_EQ(win.base(), nullptr);
    EXPECT_EQ(win.size(), 0);
    EXPECT_EQ(win.native_handle(), nullptr);
}

TEST(MemoryWindowTest, CreateFromRawPointer) {
    std::array<int, 100> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_TRUE(win.valid());
    EXPECT_EQ(win.base(), data.data());
    EXPECT_EQ(win.size(), data.size() * sizeof(int));
}

TEST(MemoryWindowTest, CreateFromNullptrWithZeroSize) {
    auto result = memory_window::create(nullptr, 0);

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_FALSE(win.valid());  // nullptr is not valid
    EXPECT_EQ(win.base(), nullptr);
    EXPECT_EQ(win.size(), 0);
}

TEST(MemoryWindowTest, CreateFromNullptrWithNonZeroSizeFails) {
    auto result = memory_window::create(nullptr, 100);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
}

TEST(MemoryWindowTest, CreateWithAllocation) {
    auto result = memory_window::allocate(1024);

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_TRUE(win.valid());
    EXPECT_NE(win.base(), nullptr);
    EXPECT_EQ(win.size(), 1024);
}

TEST(MemoryWindowTest, AllocateZeroSize) {
    auto result = memory_window::allocate(0);

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_FALSE(win.valid());  // Zero-size window is not valid
    EXPECT_EQ(win.size(), 0);
}

TEST(MemoryWindowTest, CreateFromSpan) {
    std::vector<double> data(100, 3.14);
    std::span<double> span{data};

    auto result = memory_window::from_span(span);

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_TRUE(win.valid());
    EXPECT_EQ(win.base(), data.data());
    EXPECT_EQ(win.size(), data.size() * sizeof(double));
}

TEST(MemoryWindowTest, CreateFromConstSpan) {
    const std::vector<int> data(50, 42);
    std::span<const int> span{data};

    auto result = memory_window::from_span(span);

    ASSERT_TRUE(result.has_value());
    auto& win = *result;

    EXPECT_TRUE(win.valid());
    EXPECT_EQ(win.size(), data.size() * sizeof(int));
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(MemoryWindowTest, MoveConstruction) {
    std::array<int, 50> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    memory_window win1 = std::move(*result);
    EXPECT_TRUE(win1.valid());
    EXPECT_EQ(win1.base(), data.data());

    memory_window win2 = std::move(win1);
    EXPECT_TRUE(win2.valid());
    EXPECT_EQ(win2.base(), data.data());

    // win1 should be invalid after move
    EXPECT_FALSE(win1.valid());  // NOLINT: testing post-move state
}

TEST(MemoryWindowTest, MoveAssignment) {
    std::array<int, 50> data1{};
    std::array<int, 100> data2{};

    auto result1 = memory_window::create(data1.data(), data1.size() * sizeof(int));
    auto result2 = memory_window::create(data2.data(), data2.size() * sizeof(int));
    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    memory_window win1 = std::move(*result1);
    memory_window win2 = std::move(*result2);

    win1 = std::move(win2);

    EXPECT_TRUE(win1.valid());
    EXPECT_EQ(win1.base(), data2.data());
    EXPECT_FALSE(win2.valid());  // NOLINT: testing post-move state
}

// =============================================================================
// Non-Copyable Verification
// =============================================================================

TEST(MemoryWindowTest, NonCopyableStaticAssert) {
    static_assert(!std::is_copy_constructible_v<memory_window>,
                  "memory_window should not be copy constructible");
    static_assert(!std::is_copy_assignable_v<memory_window>,
                  "memory_window should not be copy assignable");
    static_assert(std::is_move_constructible_v<memory_window>,
                  "memory_window should be move constructible");
    static_assert(std::is_move_assignable_v<memory_window>,
                  "memory_window should be move assignable");
}

// =============================================================================
// Query Methods Tests
// =============================================================================

TEST(MemoryWindowTest, BaseReturnsCorrectPointer) {
    std::array<char, 256> data{};
    auto result = memory_window::create(data.data(), data.size());
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->base(), data.data());
}

TEST(MemoryWindowTest, SizeReturnsCorrectSize) {
    std::array<double, 64> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(double));
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->size(), 64 * sizeof(double));
}

TEST(MemoryWindowTest, InvalidWindowDetection) {
    memory_window invalid_win;
    EXPECT_FALSE(invalid_win.valid());
    EXPECT_FALSE(static_cast<bool>(invalid_win));
}

TEST(MemoryWindowTest, ValidWindowDetection) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    EXPECT_TRUE(result->valid());
    EXPECT_TRUE(static_cast<bool>(*result));
}

TEST(MemoryWindowTest, WindowInfo) {
    std::array<int, 32> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    auto info = result->info();
    EXPECT_EQ(info.base, data.data());
    EXPECT_EQ(info.size, data.size() * sizeof(int));
}

// =============================================================================
// Synchronization Tests (with null implementation)
// =============================================================================

TEST(MemoryWindowTest, FenceOnValidWindow) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    auto fence_result = result->fence();
    EXPECT_TRUE(fence_result.has_value());
}

TEST(MemoryWindowTest, FenceOnInvalidWindowFails) {
    memory_window invalid_win;
    auto fence_result = invalid_win.fence();
    EXPECT_FALSE(fence_result.has_value());
    EXPECT_TRUE(fence_result.has_error());
}

TEST(MemoryWindowTest, LockUnlockLifecycle) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    auto lock_result = result->lock(0, rma_lock_mode::exclusive);
    EXPECT_TRUE(lock_result.has_value());

    auto unlock_result = result->unlock(0);
    EXPECT_TRUE(unlock_result.has_value());
}

TEST(MemoryWindowTest, LockModes) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    // Test exclusive lock
    auto exclusive_lock = result->lock(0, rma_lock_mode::exclusive);
    EXPECT_TRUE(exclusive_lock.has_value());
    auto unlock1 = result->unlock(0);
    EXPECT_TRUE(unlock1.has_value());

    // Test shared lock
    auto shared_lock = result->lock(0, rma_lock_mode::shared);
    EXPECT_TRUE(shared_lock.has_value());
    auto unlock2 = result->unlock(0);
    EXPECT_TRUE(unlock2.has_value());
}

TEST(MemoryWindowTest, LockAllUnlockAll) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    auto lock_all_result = result->lock_all();
    EXPECT_TRUE(lock_all_result.has_value());

    auto unlock_all_result = result->unlock_all();
    EXPECT_TRUE(unlock_all_result.has_value());
}

TEST(MemoryWindowTest, FlushOperations) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    auto flush_result = result->flush(0);
    EXPECT_TRUE(flush_result.has_value());

    auto flush_all_result = result->flush_all();
    EXPECT_TRUE(flush_all_result.has_value());

    auto flush_local_result = result->flush_local(0);
    EXPECT_TRUE(flush_local_result.has_value());

    auto flush_local_all_result = result->flush_local_all();
    EXPECT_TRUE(flush_local_all_result.has_value());
}

// =============================================================================
// Operations on Invalid Window
// =============================================================================

TEST(MemoryWindowTest, OperationsOnInvalidWindowFail) {
    memory_window invalid_win;

    EXPECT_FALSE(invalid_win.lock(0).has_value());
    EXPECT_FALSE(invalid_win.unlock(0).has_value());
    EXPECT_FALSE(invalid_win.lock_all().has_value());
    EXPECT_FALSE(invalid_win.unlock_all().has_value());
    EXPECT_FALSE(invalid_win.flush(0).has_value());
    EXPECT_FALSE(invalid_win.flush_all().has_value());
    EXPECT_FALSE(invalid_win.flush_local(0).has_value());
    EXPECT_FALSE(invalid_win.flush_local_all().has_value());
}

// =============================================================================
// Multiple Windows Tests
// =============================================================================

TEST(MemoryWindowTest, MultipleWindowsCanCoexist) {
    std::array<int, 100> data1{};
    std::array<double, 200> data2{};
    std::array<char, 50> data3{};

    auto win1 = memory_window::create(data1.data(), data1.size() * sizeof(int));
    auto win2 = memory_window::create(data2.data(), data2.size() * sizeof(double));
    auto win3 = memory_window::create(data3.data(), data3.size() * sizeof(char));

    ASSERT_TRUE(win1.has_value());
    ASSERT_TRUE(win2.has_value());
    ASSERT_TRUE(win3.has_value());

    EXPECT_TRUE(win1->valid());
    EXPECT_TRUE(win2->valid());
    EXPECT_TRUE(win3->valid());

    EXPECT_EQ(win1->base(), data1.data());
    EXPECT_EQ(win2->base(), data2.data());
    EXPECT_EQ(win3->base(), data3.data());
}

TEST(MemoryWindowTest, WindowDestructionOrderSafety) {
    std::array<int, 100> data{};

    {
        auto outer = memory_window::create(data.data(), data.size() * sizeof(int) / 2);
        ASSERT_TRUE(outer.has_value());

        {
            auto inner = memory_window::create(
                data.data() + 50,
                data.size() * sizeof(int) / 2
            );
            ASSERT_TRUE(inner.has_value());
            EXPECT_TRUE(inner->valid());
        }
        // inner destroyed here

        EXPECT_TRUE(outer->valid());
    }
    // outer destroyed here
}

// =============================================================================
// Native Handle Tests
// =============================================================================

TEST(MemoryWindowTest, NativeHandleAccessNull) {
    std::array<int, 10> data{};
    auto result = memory_window::create(data.data(), data.size() * sizeof(int));
    ASSERT_TRUE(result.has_value());

    // Null implementation returns nullptr
    EXPECT_EQ(result->native_handle(), nullptr);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST(MemoryWindowUtilTest, WindowsOverlapDetection) {
    std::array<char, 100> data{};

    // Create two overlapping windows
    auto win1 = memory_window::create(data.data(), 60);
    auto win2 = memory_window::create(data.data() + 40, 60);
    ASSERT_TRUE(win1.has_value() && win2.has_value());

    EXPECT_TRUE(windows_overlap(*win1, *win2));
}

TEST(MemoryWindowUtilTest, WindowsNoOverlap) {
    std::array<char, 100> data{};

    // Create two non-overlapping windows
    auto win1 = memory_window::create(data.data(), 40);
    auto win2 = memory_window::create(data.data() + 60, 40);
    ASSERT_TRUE(win1.has_value() && win2.has_value());

    EXPECT_FALSE(windows_overlap(*win1, *win2));
}

TEST(MemoryWindowUtilTest, WindowsOverlapWithInvalid) {
    std::array<char, 100> data{};
    memory_window invalid;

    auto win = memory_window::create(data.data(), 50);
    ASSERT_TRUE(win.has_value());

    EXPECT_FALSE(windows_overlap(*win, invalid));
    EXPECT_FALSE(windows_overlap(invalid, *win));
    EXPECT_FALSE(windows_overlap(invalid, invalid));
}

TEST(MemoryWindowUtilTest, WindowRangeValid) {
    std::array<char, 100> data{};
    auto win = memory_window::create(data.data(), 100);
    ASSERT_TRUE(win.has_value());

    // Valid ranges
    EXPECT_TRUE(window_range_valid(*win, 0, 100));
    EXPECT_TRUE(window_range_valid(*win, 0, 50));
    EXPECT_TRUE(window_range_valid(*win, 50, 50));
    EXPECT_TRUE(window_range_valid(*win, 99, 1));
    EXPECT_TRUE(window_range_valid(*win, 100, 0));

    // Invalid ranges
    EXPECT_FALSE(window_range_valid(*win, 0, 101));
    EXPECT_FALSE(window_range_valid(*win, 50, 51));
    EXPECT_FALSE(window_range_valid(*win, 101, 0));
}

TEST(MemoryWindowUtilTest, WindowRangeValidWithInvalid) {
    memory_window invalid;
    EXPECT_FALSE(window_range_valid(invalid, 0, 10));
}

}  // namespace dtl::test
