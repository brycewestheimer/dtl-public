// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rma_operations.cpp
/// @brief Unit tests for RMA operations
/// @details Verifies put/get operations using the null implementation.

#include <dtl/communication/rma_operations.hpp>

#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class RmaOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a window with test data
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

TEST_F(RmaOperationsTest, PutToLocalWindow) {
    std::array<int, 10> source = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result = rma::put<int>(0, 0, std::span{source}, window_);

    ASSERT_TRUE(result.has_value());

    // Verify data was copied
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(data_[i], source[i]);
    }
}

TEST_F(RmaOperationsTest, GetFromLocalWindow) {
    // Set up source data
    for (size_t i = 0; i < 10; ++i) {
        data_[i] = static_cast<int>(i * 10);
    }

    std::array<int, 10> dest{};

    auto result = rma::get<int>(0, 0, std::span{dest}, window_);

    ASSERT_TRUE(result.has_value());

    // Verify data was copied
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(dest[i], static_cast<int>(i * 10));
    }
}

TEST_F(RmaOperationsTest, PutWithOffset) {
    std::array<int, 5> source = {100, 200, 300, 400, 500};

    // Put at offset 20 (5 ints = 20 bytes)
    auto result = rma::put<int>(0, 5 * sizeof(int), std::span{source}, window_);

    ASSERT_TRUE(result.has_value());

    // Verify data was copied at correct offset
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(data_[5 + i], source[i]);
    }

    // Verify other data unchanged
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(data_[i], 0);
    }
}

TEST_F(RmaOperationsTest, GetWithOffset) {
    // Set up source data at offset
    for (size_t i = 0; i < 5; ++i) {
        data_[10 + i] = static_cast<int>((i + 1) * 111);
    }

    std::array<int, 5> dest{};

    // Get from offset 40 (10 ints = 40 bytes)
    auto result = rma::get<int>(0, 10 * sizeof(int), std::span{dest}, window_);

    ASSERT_TRUE(result.has_value());

    // Verify data was copied from correct offset
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(dest[i], static_cast<int>((i + 1) * 111));
    }
}

TEST_F(RmaOperationsTest, EmptySpanPut) {
    std::span<const int> empty_span;

    auto result = rma::put<int>(0, 0, empty_span, window_);

    EXPECT_TRUE(result.has_value());
}

TEST_F(RmaOperationsTest, EmptySpanGet) {
    std::span<int> empty_span;

    auto result = rma::get<int>(0, 0, empty_span, window_);

    EXPECT_TRUE(result.has_value());
}

TEST_F(RmaOperationsTest, PutSingleValue) {
    int value = 42;

    auto result = rma::put<int>(0, 0, value, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 42);
}

TEST_F(RmaOperationsTest, GetSingleValue) {
    data_[5] = 123;
    int value = 0;

    auto result = rma::get<int>(0, 5 * sizeof(int), value, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(value, 123);
}

// =============================================================================
// Async Tests
// =============================================================================

TEST_F(RmaOperationsTest, PutAsyncReturnsValidRequest) {
    std::array<int, 5> source = {1, 2, 3, 4, 5};

    auto result = rma::put_async<int>(0, 0, std::span{source}, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->valid());
}

TEST_F(RmaOperationsTest, GetAsyncReturnsValidRequest) {
    std::array<int, 5> dest{};

    auto result = rma::get_async<int>(0, 0, std::span{dest}, window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->valid());
}

// =============================================================================
// Flush Tests
// =============================================================================

TEST_F(RmaOperationsTest, FlushOnValidWindow) {
    auto result = rma::flush(0, window_);
    EXPECT_TRUE(result.has_value());
}

TEST_F(RmaOperationsTest, FlushAllOnValidWindow) {
    auto result = rma::flush_all(window_);
    EXPECT_TRUE(result.has_value());
}

TEST_F(RmaOperationsTest, FlushLocalCompletesPut) {
    std::array<int, 5> source = {1, 2, 3, 4, 5};

    auto put_result = rma::put<int>(0, 0, std::span{source}, window_);
    ASSERT_TRUE(put_result.has_value());

    auto flush_result = rma::flush_local(0, window_);
    EXPECT_TRUE(flush_result.has_value());
}

TEST_F(RmaOperationsTest, FlushLocalAll) {
    auto result = rma::flush_local_all(window_);
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(RmaOperationsTest, InvalidWindowOperationsFail) {
    memory_window invalid_win;
    std::array<int, 5> buf{};

    auto put_result = rma::put<int>(0, 0, std::span{buf}, invalid_win);
    EXPECT_FALSE(put_result.has_value());

    auto get_result = rma::get<int>(0, 0, std::span{buf}, invalid_win);
    EXPECT_FALSE(get_result.has_value());
}

TEST_F(RmaOperationsTest, OutOfBoundsOffsetDetection) {
    std::array<int, 5> buf{};

    // Offset beyond window size
    auto result = rma::put<int>(0, 1000, std::span{buf}, window_);
    EXPECT_FALSE(result.has_value());
}

TEST_F(RmaOperationsTest, TypeSizeCalculation) {
    // Put double values and verify correct size
    std::array<double, 5> source = {1.1, 2.2, 3.3, 4.4, 5.5};

    // Create a double-sized window
    std::array<double, 100> double_data{};
    auto win_result = memory_window::create(double_data.data(),
                                            double_data.size() * sizeof(double));
    ASSERT_TRUE(win_result.has_value());
    auto& double_win = *win_result;

    auto result = rma::put<double>(0, 0, std::span{source}, double_win);
    ASSERT_TRUE(result.has_value());

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(double_data[i], source[i]);
    }
}

TEST_F(RmaOperationsTest, MultipleOperationsOnSameWindow) {
    // Perform multiple operations on the same window
    std::array<int, 5> data1 = {1, 2, 3, 4, 5};
    std::array<int, 5> data2 = {10, 20, 30, 40, 50};

    auto result1 = rma::put<int>(0, 0, std::span{data1}, window_);
    ASSERT_TRUE(result1.has_value());

    auto result2 = rma::put<int>(0, 5 * sizeof(int), std::span{data2}, window_);
    ASSERT_TRUE(result2.has_value());

    // Verify both writes
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(data_[i], data1[i]);
        EXPECT_EQ(data_[5 + i], data2[i]);
    }

    // Now read back
    std::array<int, 10> read_buf{};
    auto result3 = rma::get<int>(0, 0, std::span{read_buf}, window_);
    ASSERT_TRUE(result3.has_value());

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(read_buf[i], data1[i]);
        EXPECT_EQ(read_buf[5 + i], data2[i]);
    }
}

TEST_F(RmaOperationsTest, InvalidRankFails) {
    // In null implementation, only rank 0 is valid
    std::array<int, 5> buf{};

    auto result = rma::put<int>(1, 0, std::span{buf}, window_);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Raw Pointer Interface Tests
// =============================================================================

TEST_F(RmaOperationsTest, RawPointerPut) {
    int value = 999;

    auto result = rma::put(0, 0, &value, sizeof(int), window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 999);
}

TEST_F(RmaOperationsTest, RawPointerGet) {
    data_[0] = 888;
    int value = 0;

    auto result = rma::get(0, 0, &value, sizeof(int), window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(value, 888);
}

TEST_F(RmaOperationsTest, RawPointerPutAsync) {
    int value = 777;

    auto result = rma::put_async(0, 0, &value, sizeof(int), window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->valid());
}

TEST_F(RmaOperationsTest, RawPointerGetAsync) {
    int value = 0;

    auto result = rma::get_async(0, 0, &value, sizeof(int), window_);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->valid());
}

}  // namespace dtl::test
