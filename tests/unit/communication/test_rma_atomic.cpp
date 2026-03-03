// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rma_atomic.cpp
/// @brief Unit tests for atomic RMA operations
/// @details Verifies accumulate, fetch_and_op, compare_and_swap operations.

#include <dtl/communication/rma_atomic.hpp>

#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class RmaAtomicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a window with test data (initialized to zero)
        data_.fill(static_cast<std::byte>(0));
        auto result = memory_window::create(data_.data(), data_.size());
        ASSERT_TRUE(result.has_value());
        window_ = std::move(*result);
    }

    std::array<std::byte, 256> data_;
    memory_window window_;
};

// =============================================================================
// Accumulate Tests
// =============================================================================

TEST_F(RmaAtomicTest, AccumulateWithReplace) {
    const std::array<std::byte, 4> value = {
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}
    };

    std::span<const std::byte> value_span{value};
    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::replace, window_);

    ASSERT_TRUE(result.has_value());

    // Verify data was replaced
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(data_[i], value[i]);
    }
}

TEST_F(RmaAtomicTest, AccumulateWithSum) {
    // Set initial value
    data_[0] = std::byte{10};
    data_[1] = std::byte{20};

    const std::array<std::byte, 2> value = {std::byte{5}, std::byte{10}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::sum, window_);

    ASSERT_TRUE(result.has_value());

    // Verify sum: 10+5=15, 20+10=30
    EXPECT_EQ(data_[0], std::byte{15});
    EXPECT_EQ(data_[1], std::byte{30});
}

TEST_F(RmaAtomicTest, AccumulateWithMax) {
    // Set initial values
    data_[0] = std::byte{10};
    data_[1] = std::byte{50};

    const std::array<std::byte, 2> value = {std::byte{20}, std::byte{30}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::max, window_);

    ASSERT_TRUE(result.has_value());

    // Verify max: max(10,20)=20, max(50,30)=50
    EXPECT_EQ(data_[0], std::byte{20});
    EXPECT_EQ(data_[1], std::byte{50});
}

TEST_F(RmaAtomicTest, AccumulateWithMin) {
    // Set initial values
    data_[0] = std::byte{10};
    data_[1] = std::byte{50};

    const std::array<std::byte, 2> value = {std::byte{20}, std::byte{30}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::min, window_);

    ASSERT_TRUE(result.has_value());

    // Verify min: min(10,20)=10, min(50,30)=30
    EXPECT_EQ(data_[0], std::byte{10});
    EXPECT_EQ(data_[1], std::byte{30});
}

TEST_F(RmaAtomicTest, AccumulateWithBitwiseAnd) {
    // Set initial value (binary: 0b11110000 = 240)
    data_[0] = std::byte{0xF0};

    // AND with 0b10101010 = 170
    const std::array<std::byte, 1> value = {std::byte{0xAA}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::band, window_);

    ASSERT_TRUE(result.has_value());

    // Result: 0xF0 & 0xAA = 0xA0 = 160
    EXPECT_EQ(data_[0], std::byte{0xA0});
}

TEST_F(RmaAtomicTest, AccumulateWithBitwiseOr) {
    // Set initial value
    data_[0] = std::byte{0x0F};

    // OR with 0xF0
    const std::array<std::byte, 1> value = {std::byte{0xF0}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::bor, window_);

    ASSERT_TRUE(result.has_value());

    // Result: 0x0F | 0xF0 = 0xFF
    EXPECT_EQ(data_[0], std::byte{0xFF});
}

TEST_F(RmaAtomicTest, AccumulateWithBitwiseXor) {
    // Set initial value
    data_[0] = std::byte{0xFF};

    // XOR with 0x0F
    const std::array<std::byte, 1> value = {std::byte{0x0F}};
    std::span<const std::byte> value_span{value};

    auto result = rma::accumulate<std::byte>(0, 0, value_span, rma_reduce_op::bxor, window_);

    ASSERT_TRUE(result.has_value());

    // Result: 0xFF ^ 0x0F = 0xF0
    EXPECT_EQ(data_[0], std::byte{0xF0});
}

TEST_F(RmaAtomicTest, AccumulateSingleValue) {
    data_[0] = std::byte{10};
    std::byte value = std::byte{5};

    auto result = rma::accumulate(0, 0, value, rma_reduce_op::sum, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], std::byte{15});
}

// =============================================================================
// Fetch and Op Tests
// =============================================================================

TEST_F(RmaAtomicTest, FetchAndOpReturnsOldValue) {
    data_[0] = std::byte{42};
    std::byte origin = std::byte{10};
    std::byte result_val{};

    auto result = rma::fetch_and_op(0, 0, origin, result_val, rma_reduce_op::sum, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{42});  // Old value
    EXPECT_EQ(data_[0], std::byte{52});     // New value: 42 + 10
}

TEST_F(RmaAtomicTest, FetchAndOpAppliesOperation) {
    data_[0] = std::byte{100};
    std::byte origin = std::byte{5};
    std::byte result_val{};

    auto result = rma::fetch_and_op(0, 0, origin, result_val, rma_reduce_op::max, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{100});  // Old value
    EXPECT_EQ(data_[0], std::byte{100});    // max(100, 5) = 100
}

TEST_F(RmaAtomicTest, FetchAndAdd) {
    data_[0] = std::byte{50};
    std::byte addend = std::byte{25};
    std::byte result_val{};

    auto result = rma::fetch_and_add(0, 0, addend, result_val, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{50});   // Old value
    EXPECT_EQ(data_[0], std::byte{75});     // 50 + 25
}

// =============================================================================
// Compare and Swap Tests
// =============================================================================

TEST_F(RmaAtomicTest, CompareAndSwapSuccess) {
    data_[0] = std::byte{42};
    std::byte compare = std::byte{42};
    std::byte swap_val = std::byte{100};
    std::byte result_val{};

    auto result = rma::compare_and_swap(0, 0, compare, swap_val, result_val, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{42});   // Original value
    EXPECT_EQ(data_[0], std::byte{100});    // Swapped to new value
}

TEST_F(RmaAtomicTest, CompareAndSwapFailure) {
    data_[0] = std::byte{42};
    std::byte compare = std::byte{99};      // Does not match
    std::byte swap_val = std::byte{100};
    std::byte result_val{};

    auto result = rma::compare_and_swap(0, 0, compare, swap_val, result_val, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{42});   // Original value returned
    EXPECT_EQ(data_[0], std::byte{42});     // No swap occurred
}

// =============================================================================
// Get-Accumulate Tests
// =============================================================================

TEST_F(RmaAtomicTest, GetAccumulateCombinesGetAndAccumulate) {
    data_[0] = std::byte{10};
    data_[1] = std::byte{20};

    const std::array<std::byte, 2> origin = {std::byte{5}, std::byte{10}};
    std::array<std::byte, 2> result_buf{};

    std::span<const std::byte> origin_span{origin};
    std::span<std::byte> result_span{result_buf};

    auto result = rma::get_accumulate<std::byte>(0, 0, origin_span, result_span,
                                       rma_reduce_op::sum, window_);

    ASSERT_TRUE(result.has_value());

    // Check old values returned
    EXPECT_EQ(result_buf[0], std::byte{10});
    EXPECT_EQ(result_buf[1], std::byte{20});

    // Check new values in window
    EXPECT_EQ(data_[0], std::byte{15});  // 10 + 5
    EXPECT_EQ(data_[1], std::byte{30});  // 20 + 10
}

TEST_F(RmaAtomicTest, GetAccumulateSingleValue) {
    data_[0] = std::byte{100};
    std::byte origin = std::byte{50};
    std::byte result_val{};

    auto result = rma::get_accumulate(0, 0, origin, result_val, rma_reduce_op::max, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result_val, std::byte{100});  // Old value
    EXPECT_EQ(data_[0], std::byte{100});    // max(100, 50) = 100
}

TEST_F(RmaAtomicTest, GetAccumulateSizeMismatchFails) {
    std::array<std::byte, 4> origin{};
    std::array<std::byte, 2> result_buf{};

    // Use std::span with dynamic extent to allow different sizes
    std::span<const std::byte> origin_span{origin};
    std::span<std::byte> result_span{result_buf};

    auto result = rma::get_accumulate<std::byte>(0, 0, origin_span, result_span,
                                       rma_reduce_op::sum, window_);

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(RmaAtomicTest, InvalidReduceOpHandled) {
    std::array<std::byte, 4> origin = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};

    // no_op should not modify data
    auto result = rma::accumulate(0, 0, std::span{origin}, rma_reduce_op::no_op, window_);

    ASSERT_TRUE(result.has_value());

    // Data should be unchanged
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(data_[i], std::byte{0});
    }
}

TEST_F(RmaAtomicTest, InvalidWindowFails) {
    memory_window invalid_win;
    std::byte value{};
    std::byte result_val{};

    auto acc_result = rma::accumulate(0, 0, value, rma_reduce_op::sum, invalid_win);
    EXPECT_FALSE(acc_result.has_value());

    auto fao_result = rma::fetch_and_op(0, 0, value, result_val, rma_reduce_op::sum, invalid_win);
    EXPECT_FALSE(fao_result.has_value());

    auto cas_result = rma::compare_and_swap(0, 0, value, value, result_val, invalid_win);
    EXPECT_FALSE(cas_result.has_value());
}

TEST_F(RmaAtomicTest, OutOfBoundsFails) {
    std::byte value{};

    auto result = rma::accumulate(0, 1000, value, rma_reduce_op::sum, window_);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

TEST_F(RmaAtomicTest, AccumulateSumConvenience) {
    data_[0] = std::byte{10};
    const std::array<std::byte, 1> value = {std::byte{5}};

    auto result = rma::accumulate_sum<std::byte>(0, 0, std::span{value}, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], std::byte{15});
}

TEST_F(RmaAtomicTest, AccumulateMaxConvenience) {
    data_[0] = std::byte{10};
    const std::array<std::byte, 1> value = {std::byte{50}};

    auto result = rma::accumulate_max<std::byte>(0, 0, std::span{value}, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], std::byte{50});
}

TEST_F(RmaAtomicTest, AccumulateMinConvenience) {
    data_[0] = std::byte{50};
    const std::array<std::byte, 1> value = {std::byte{10}};

    auto result = rma::accumulate_min<std::byte>(0, 0, std::span{value}, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], std::byte{10});
}

}  // namespace dtl::test
