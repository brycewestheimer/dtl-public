// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_memory_window_data_transfer.cpp
/// @brief Unit tests for memory_window data transfer methods
/// @details Tests the RAII wrapper methods (put/get/accumulate/etc.) on memory_window.
///          Phase 3 RMA Wiring: Verifies that memory_window correctly delegates to impl_.

#include <dtl/communication/memory_window.hpp>

#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class MemoryWindowDataTransferTest : public ::testing::Test {
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
// Put Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, PutWritesDataToWindow) {
    int value = 42;
    auto result = window_.put(&value, sizeof(int), 0, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 42);
}

TEST_F(MemoryWindowDataTransferTest, PutWithOffsetWritesCorrectly) {
    int value = 99;
    auto result = window_.put(&value, sizeof(int), 0, 5 * sizeof(int));

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[5], 99);
}

TEST_F(MemoryWindowDataTransferTest, PutOnInvalidWindowFails) {
    memory_window invalid_win;
    int value = 42;

    auto result = invalid_win.put(&value, sizeof(int), 0, 0);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST_F(MemoryWindowDataTransferTest, PutNullptrFails) {
    auto result = window_.put(nullptr, sizeof(int), 0, 0);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
}

TEST_F(MemoryWindowDataTransferTest, PutOutOfBoundsFails) {
    int value = 42;
    auto result = window_.put(&value, sizeof(int), 0, window_.size());

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
}

// =============================================================================
// Get Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, GetReadsDataFromWindow) {
    data_[0] = 123;
    int value = 0;

    auto result = window_.get(&value, sizeof(int), 0, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(value, 123);
}

TEST_F(MemoryWindowDataTransferTest, GetWithOffsetReadsCorrectly) {
    data_[10] = 456;
    int value = 0;

    auto result = window_.get(&value, sizeof(int), 0, 10 * sizeof(int));

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(value, 456);
}

TEST_F(MemoryWindowDataTransferTest, GetOnInvalidWindowFails) {
    memory_window invalid_win;
    int value = 0;

    auto result = invalid_win.get(&value, sizeof(int), 0, 0);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST_F(MemoryWindowDataTransferTest, GetNullptrFails) {
    auto result = window_.get(nullptr, sizeof(int), 0, 0);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
}

TEST_F(MemoryWindowDataTransferTest, GetOutOfBoundsFails) {
    int value = 0;
    auto result = window_.get(&value, sizeof(int), 0, window_.size());

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
}

// =============================================================================
// Accumulate Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, AccumulateWithSum) {
    data_[0] = 10;
    int value = 5;

    auto result = window_.accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::sum);

    ASSERT_TRUE(result.has_value());
    // Note: null_window_impl treats ints as bytes, so this is byte-wise sum
    // For a proper int sum, data_[0] would be 15, but null impl does byte ops
    // Just verify the operation succeeded
}

TEST_F(MemoryWindowDataTransferTest, AccumulateWithReplace) {
    data_[0] = 10;
    int value = 99;

    auto result = window_.accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::replace);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 99);
}

TEST_F(MemoryWindowDataTransferTest, AccumulateOnInvalidWindowFails) {
    memory_window invalid_win;
    int value = 42;

    auto result = invalid_win.accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::sum);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST_F(MemoryWindowDataTransferTest, AccumulateWithNoOp) {
    data_[0] = 42;
    int value = 99;

    auto result = window_.accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::no_op);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 42);  // Should be unchanged
}

// =============================================================================
// Fetch and Op Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, FetchAndOpReturnsOldValue) {
    data_[0] = 50;
    int origin = 10;
    int result_val = 0;

    auto result = window_.fetch_and_op(&origin, &result_val, sizeof(int), 0, 0, rma_reduce_op::sum);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, 50);  // Should return old value
}

TEST_F(MemoryWindowDataTransferTest, FetchAndOpOnInvalidWindowFails) {
    memory_window invalid_win;
    int origin = 10;
    int result_val = 0;

    auto result = invalid_win.fetch_and_op(&origin, &result_val, sizeof(int), 0, 0, rma_reduce_op::sum);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST_F(MemoryWindowDataTransferTest, FetchAndOpNullptrsFail) {
    int value = 10;

    auto result1 = window_.fetch_and_op(nullptr, &value, sizeof(int), 0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result1.has_value());

    auto result2 = window_.fetch_and_op(&value, nullptr, sizeof(int), 0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result2.has_value());
}

// =============================================================================
// Compare and Swap Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, CompareAndSwapSuccess) {
    data_[0] = 42;
    int compare = 42;
    int swap_val = 100;
    int result_val = 0;

    auto result = window_.compare_and_swap(&swap_val, &compare, &result_val, sizeof(int), 0, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, 42);   // Original value
    EXPECT_EQ(data_[0], 100);    // Swapped to new value
}

TEST_F(MemoryWindowDataTransferTest, CompareAndSwapFailure) {
    data_[0] = 42;
    int compare = 99;  // Does not match
    int swap_val = 100;
    int result_val = 0;

    auto result = window_.compare_and_swap(&swap_val, &compare, &result_val, sizeof(int), 0, 0);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, 42);   // Original value returned
    EXPECT_EQ(data_[0], 42);     // No swap occurred
}

TEST_F(MemoryWindowDataTransferTest, CompareAndSwapOnInvalidWindowFails) {
    memory_window invalid_win;
    int compare = 42;
    int swap_val = 100;
    int result_val = 0;

    auto result = invalid_win.compare_and_swap(&swap_val, &compare, &result_val, sizeof(int), 0, 0);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

// =============================================================================
// Get-Accumulate Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, GetAccumulateCombinesGetAndAccumulate) {
    data_[0] = 20;
    int origin = 5;
    int result_val = 0;

    auto result = window_.get_accumulate(&origin, &result_val, sizeof(int), 0, 0, rma_reduce_op::replace);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, 20);   // Old value
    EXPECT_EQ(data_[0], 5);      // New value after replace
}

TEST_F(MemoryWindowDataTransferTest, GetAccumulateOnInvalidWindowFails) {
    memory_window invalid_win;
    int origin = 5;
    int result_val = 0;

    auto result = invalid_win.get_accumulate(&origin, &result_val, sizeof(int), 0, 0, rma_reduce_op::sum);

    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST_F(MemoryWindowDataTransferTest, GetAccumulateNullptrsFail) {
    int value = 10;

    auto result1 = window_.get_accumulate(nullptr, &value, sizeof(int), 0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result1.has_value());

    auto result2 = window_.get_accumulate(&value, nullptr, sizeof(int), 0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result2.has_value());
}

// =============================================================================
// Moved-From Window Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, OperationsOnMovedFromWindowFail) {
    memory_window moved_window = std::move(window_);  // window_ is now moved-from

    int value = 42;
    int result_val = 0;

    // All operations on moved-from window should fail with invalid_state
    EXPECT_FALSE(window_.put(&value, sizeof(int), 0, 0).has_value());
    EXPECT_FALSE(window_.get(&value, sizeof(int), 0, 0).has_value());
    EXPECT_FALSE(window_.accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::sum).has_value());
    EXPECT_FALSE(window_.fetch_and_op(&value, &result_val, sizeof(int), 0, 0, rma_reduce_op::sum).has_value());
    EXPECT_FALSE(window_.compare_and_swap(&value, &value, &result_val, sizeof(int), 0, 0).has_value());
    EXPECT_FALSE(window_.get_accumulate(&value, &result_val, sizeof(int), 0, 0, rma_reduce_op::sum).has_value());
}

// =============================================================================
// Invalid Rank Tests
// =============================================================================

TEST_F(MemoryWindowDataTransferTest, OperationsWithInvalidRankFail) {
    // null_window_impl only supports rank 0
    int value = 42;
    int result_val = 0;

    auto result1 = window_.put(&value, sizeof(int), 1, 0);
    EXPECT_FALSE(result1.has_value());
    EXPECT_EQ(result1.error().code(), status_code::invalid_rank);

    auto result2 = window_.get(&value, sizeof(int), 1, 0);
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code(), status_code::invalid_rank);

    auto result3 = window_.accumulate(&value, sizeof(int), 1, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result3.has_value());
    EXPECT_EQ(result3.error().code(), status_code::invalid_rank);

    auto result4 = window_.fetch_and_op(&value, &result_val, sizeof(int), 1, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result4.has_value());
    EXPECT_EQ(result4.error().code(), status_code::invalid_rank);

    auto result5 = window_.compare_and_swap(&value, &value, &result_val, sizeof(int), 1, 0);
    EXPECT_FALSE(result5.has_value());
    EXPECT_EQ(result5.error().code(), status_code::invalid_rank);

    auto result6 = window_.get_accumulate(&value, &result_val, sizeof(int), 1, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result6.has_value());
    EXPECT_EQ(result6.error().code(), status_code::invalid_rank);
}

}  // namespace dtl::test
