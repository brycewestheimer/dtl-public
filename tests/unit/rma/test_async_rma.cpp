// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_rma.cpp
/// @brief Unit tests for async RMA operations
/// @details Verifies async_put and async_get with progress engine integration.

#include <dtl/rma/async_rma.hpp>
#include <dtl/futures/progress.hpp>

#include <gtest/gtest.h>
#include <array>
#include <atomic>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class AsyncRmaTest : public ::testing::Test {
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
// Async Put Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncPutReadyReturnsFalseInitially) {
    std::array<int, 5> send_data = {1, 2, 3, 4, 5};

    rma::async_put<int> put(0, 0, std::span{send_data}, window_);

    // After construction, operation is registered but may not be complete yet
    // In our null implementation, it completes immediately on first poll
    EXPECT_FALSE(put.ready());  // Before any polling
}

TEST_F(AsyncRmaTest, AsyncPutWaitCompletesOperation) {
    std::array<int, 5> send_data = {10, 20, 30, 40, 50};

    rma::async_put<int> put(0, 0, std::span{send_data}, window_);
    put.wait();

    EXPECT_TRUE(put.ready());
    EXPECT_TRUE(put.get_result().has_value());

    // Verify data was written
    EXPECT_EQ(data_[0], 10);
    EXPECT_EQ(data_[4], 50);
}

TEST_F(AsyncRmaTest, AsyncPutGetResultReturnsSuccess) {
    std::array<int, 3> send_data = {100, 200, 300};

    rma::async_put<int> put(0, 0, std::span{send_data}, window_);
    auto result = put.get_result();

    EXPECT_TRUE(result.has_value());
}

TEST_F(AsyncRmaTest, AsyncPutToFactoryFunction) {
    std::array<int, 2> send_data = {42, 84};
    std::span<const int> send_span{send_data};

    auto put = rma::async_put_to<int>(0, 0, send_span, window_);
    put.wait();

    EXPECT_TRUE(put.ready());
    EXPECT_EQ(data_[0], 42);
    EXPECT_EQ(data_[1], 84);
}

// =============================================================================
// Async Get Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncGetReadyReturnsFalseInitially) {
    std::array<int, 5> recv_data{};

    rma::async_get<int> get(0, 0, std::span{recv_data}, window_);

    EXPECT_FALSE(get.ready());
}

TEST_F(AsyncRmaTest, AsyncGetWaitCompletesOperation) {
    // Set up source data
    data_[0] = 111;
    data_[1] = 222;
    data_[2] = 333;

    std::array<int, 3> recv_data{};

    rma::async_get<int> get(0, 0, std::span{recv_data}, window_);
    get.wait();

    EXPECT_TRUE(get.ready());

    auto result = get.get_result();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(recv_data[0], 111);
    EXPECT_EQ(recv_data[1], 222);
    EXPECT_EQ(recv_data[2], 333);
}

TEST_F(AsyncRmaTest, AsyncGetFromFactoryFunction) {
    data_[5] = 500;
    data_[6] = 600;

    std::array<int, 2> recv_data{};
    std::span<int> recv_span{recv_data};

    auto get = rma::async_get_from<int>(0, 5 * sizeof(int), recv_span, window_);
    auto result = get.get_result();

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(recv_data[0], 500);
    EXPECT_EQ(recv_data[1], 600);
}

// =============================================================================
// Progress Engine Integration Tests
// =============================================================================

TEST_F(AsyncRmaTest, MultipleAsyncOperationsProgress) {
    std::array<int, 2> send_data1 = {1, 2};
    std::array<int, 2> send_data2 = {3, 4};

    rma::async_put<int> put1(0, 0, std::span{send_data1}, window_);
    rma::async_put<int> put2(0, 2 * sizeof(int), std::span{send_data2}, window_);

    // Both operations registered with progress engine
    // Wait for both
    put1.wait();
    put2.wait();

    EXPECT_TRUE(put1.ready());
    EXPECT_TRUE(put2.ready());

    EXPECT_EQ(data_[0], 1);
    EXPECT_EQ(data_[1], 2);
    EXPECT_EQ(data_[2], 3);
    EXPECT_EQ(data_[3], 4);
}

TEST_F(AsyncRmaTest, ProgressPollingDrivesCompletion) {
    std::array<int, 3> send_data = {7, 8, 9};

    rma::async_put<int> put(0, 0, std::span{send_data}, window_);

    // Manual polling instead of wait()
    while (!put.ready()) {
        futures::progress_engine::instance().poll();
    }

    EXPECT_TRUE(put.ready());
    EXPECT_EQ(data_[0], 7);
}

// =============================================================================
// Callback Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncPutWithCallback) {
    std::array<int, 2> send_data = {99, 100};
    std::span<const int> send_span{send_data};
    std::atomic<bool> callback_called{false};
    std::atomic<bool> callback_success{false};

    auto put = rma::async_put_then<int>(0, 0, send_span, window_,
        [&callback_called, &callback_success](result<void> r) {
            callback_called.store(true);
            callback_success.store(r.has_value());
        });

    put.wait();

    EXPECT_TRUE(callback_called.load());
    EXPECT_TRUE(callback_success.load());
}

// =============================================================================
// Invalid Window Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncPutWithInvalidWindowFails) {
    memory_window invalid;
    std::array<int, 2> send_data = {1, 2};

    rma::async_put<int> put(0, 0, std::span{send_data}, invalid);
    auto result = put.get_result();

    EXPECT_FALSE(result.has_value());
}

TEST_F(AsyncRmaTest, AsyncGetWithInvalidWindowFails) {
    memory_window invalid;
    std::array<int, 2> recv_data{};

    rma::async_get<int> get(0, 0, std::span{recv_data}, invalid);
    auto result = get.get_result();

    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncPutMoveConstruction) {
    std::array<int, 2> send_data = {55, 66};

    rma::async_put<int> put1(0, 0, std::span{send_data}, window_);
    rma::async_put<int> put2(std::move(put1));

    // put2 should work
    put2.wait();
    EXPECT_TRUE(put2.ready());
}

TEST_F(AsyncRmaTest, AsyncGetMoveConstruction) {
    data_[0] = 77;
    data_[1] = 88;
    std::array<int, 2> recv_data{};

    rma::async_get<int> get1(0, 0, std::span{recv_data}, window_);
    rma::async_get<int> get2(std::move(get1));

    get2.wait();
    EXPECT_TRUE(get2.ready());
}

// =============================================================================
// State Tests
// =============================================================================

TEST_F(AsyncRmaTest, AsyncRmaStateValues) {
    EXPECT_NE(static_cast<int>(rma::async_rma_state::pending),
              static_cast<int>(rma::async_rma_state::ready));
    EXPECT_NE(static_cast<int>(rma::async_rma_state::ready),
              static_cast<int>(rma::async_rma_state::error));
    EXPECT_NE(static_cast<int>(rma::async_rma_state::pending),
              static_cast<int>(rma::async_rma_state::error));
}

}  // namespace dtl::test
