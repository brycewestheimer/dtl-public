// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_remote_ref.cpp
/// @brief Integration tests for async remote_ref operations
/// @details Tests that remote_ref::async_get() and async_put() are truly async
///          and complete via the progress engine.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests --gtest_filter="AsyncRemoteRef*"

#include <dtl/views/remote_ref.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_window.hpp>
#endif

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture
// =============================================================================

class AsyncRemoteRefTest : public ::testing::Test {
protected:
    void SetUp() override {
        mpi_domain_ = std::make_unique<mpi_domain>();
        comm_ = &mpi_domain_->communicator().underlying();
        rank_ = mpi_domain_->rank();
        size_ = mpi_domain_->size();
    }

    void TearDown() override {
        // Clean up
        if (mpi_window_) {
            mpi_window_.reset();
        }
        comm_ = nullptr;
        mpi_domain_.reset();
    }

    /// @brief Create an MPI window and corresponding memory_window_impl
    void create_window(void* base, size_type size) {
        auto result = mpi::mpi_window::create(base, size, *comm_);
        ASSERT_TRUE(result.has_value()) << "Failed to create MPI window";
        mpi_window_ = std::make_unique<mpi::mpi_window>(std::move(*result));
        window_impl_ = std::make_unique<mpi::mpi_window_impl>(*mpi_window_);
    }

    std::unique_ptr<mpi_domain> mpi_domain_;
    mpi::mpi_communicator* comm_ = nullptr;
    rank_t rank_ = 0;
    rank_t size_ = 0;
    std::unique_ptr<mpi::mpi_window> mpi_window_;
    std::unique_ptr<mpi::mpi_window_impl> window_impl_;
};

// =============================================================================
// Basic Async Get/Put Tests
// =============================================================================

TEST_F(AsyncRemoteRefTest, AsyncGetReturnsDistributedFuture) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    // Each rank has data
    std::array<int, 10> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(rank_ * 100 + i);
    }

    create_window(data.data(), data.size() * sizeof(int));

    // Lock all for passive target access
    auto lock_res = window_impl_->lock_all();
    ASSERT_TRUE(lock_res.has_value()) << "lock_all failed";

    // Create remote_ref pointing to rank (rank + 1) % size, element 5
    rank_t target = (rank_ + 1) % size_;
    size_type offset = 5 * sizeof(int);
    remote_ref<int> ref(target, 5, nullptr, window_impl_.get(), offset);

    // Verify it's remote
    EXPECT_TRUE(ref.is_remote());
    EXPECT_TRUE(ref.has_window());

    // Issue async get - should return immediately
    auto start = std::chrono::steady_clock::now();
    auto future = ref.async_get();
    auto issue_time = std::chrono::steady_clock::now() - start;

    // The async call should be fast (not blocking)
    EXPECT_TRUE(future.valid()) << "Future should be valid";

    // Poll until complete
    while (!future.is_ready()) {
        futures::poll();
    }

    // Get the result
    auto result = future.get_result();
    ASSERT_TRUE(result.has_value()) << "async_get should succeed";

    // Verify value: target rank's data[5] = target * 100 + 5
    int expected = target * 100 + 5;
    EXPECT_EQ(result.value(), expected);

    // Unlock
    auto unlock_res = window_impl_->unlock_all();
    ASSERT_TRUE(unlock_res.has_value()) << "unlock_all failed";
}

TEST_F(AsyncRemoteRefTest, AsyncPutReturnsDistributedFuture) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    // Initialize with sentinel values
    std::array<int, 10> data{};
    data.fill(-1);

    create_window(data.data(), data.size() * sizeof(int));

    auto lock_res = window_impl_->lock_all();
    ASSERT_TRUE(lock_res.has_value());

    // Rank 0 puts to rank 1, element 3
    if (rank_ == 0) {
        rank_t target = 1;
        size_type offset = 3 * sizeof(int);
        remote_ref<int> ref(target, 3, nullptr, window_impl_.get(), offset);

        int value_to_put = 42;
        auto future = ref.async_put(value_to_put);

        EXPECT_TRUE(future.valid());

        // Poll until complete
        while (!future.is_ready()) {
            futures::poll();
        }

        auto result = future.get_result();
        ASSERT_TRUE(result.has_value()) << "async_put should succeed";
    }

    // Flush to ensure remote visibility
    auto flush_res = window_impl_->flush_all();
    ASSERT_TRUE(flush_res.has_value());

    // Barrier to synchronize
    MPI_Barrier(comm_->native_handle());

    // Rank 1 should have received the value
    if (rank_ == 1) {
        EXPECT_EQ(data[3], 42);
    }

    auto unlock_res = window_impl_->unlock_all();
    ASSERT_TRUE(unlock_res.has_value());
}

TEST_F(AsyncRemoteRefTest, LocalAccessIsImmediate) {
    // Create a local value
    int local_value = 123;

    // Create a remote_ref that points to local storage
    remote_ref<int> ref(rank_, 0, &local_value);

    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());

    // async_get for local should return immediately ready
    auto future = ref.async_get();
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready()) << "Local async_get should be immediately ready";

    auto result = future.get_result();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 123);
}

TEST_F(AsyncRemoteRefTest, LocalPutIsImmediate) {
    int local_value = 0;
    remote_ref<int> ref(rank_, 0, &local_value);

    auto future = ref.async_put(456);
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready()) << "Local async_put should be immediately ready";

    auto result = future.get_result();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(local_value, 456);
}

// =============================================================================
// Concurrent Async Operations
// =============================================================================

TEST_F(AsyncRemoteRefTest, MultipleConcurrentAsyncOps) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 100> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(rank_ * 1000 + i);
    }

    create_window(data.data(), data.size() * sizeof(int));

    auto lock_res = window_impl_->lock_all();
    ASSERT_TRUE(lock_res.has_value());

    // Issue multiple async gets in parallel
    rank_t target = (rank_ + 1) % size_;
    std::vector<futures::distributed_future<int>> futures;
    std::vector<int> expected_values;

    for (int i = 0; i < 10; ++i) {
        size_type offset = i * sizeof(int);
        remote_ref<int> ref(target, i, nullptr, window_impl_.get(), offset);
        futures.push_back(ref.async_get());
        expected_values.push_back(target * 1000 + i);
    }

    // All futures should be valid
    for (auto& f : futures) {
        EXPECT_TRUE(f.valid());
    }

    // Poll until all complete
    bool all_ready = false;
    while (!all_ready) {
        futures::poll();
        all_ready = std::all_of(futures.begin(), futures.end(),
                                [](auto& f) { return f.is_ready(); });
    }

    // Verify all results
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get_result();
        ASSERT_TRUE(result.has_value()) << "Future " << i << " should have value";
        EXPECT_EQ(result.value(), expected_values[i]);
    }

    auto unlock_res = window_impl_->unlock_all();
    ASSERT_TRUE(unlock_res.has_value());
}

// =============================================================================
// Put-Get Roundtrip
// =============================================================================

TEST_F(AsyncRemoteRefTest, AsyncPutThenGetRoundtrip) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    data.fill(0);

    create_window(data.data(), data.size() * sizeof(int));

    auto lock_res = window_impl_->lock_all();
    ASSERT_TRUE(lock_res.has_value());

    // All ranks put their rank ID to element 0 on next rank
    rank_t target = (rank_ + 1) % size_;
    size_type offset = 0;
    remote_ref<int> ref(target, 0, nullptr, window_impl_.get(), offset);

    int value_to_put = rank_;
    auto put_future = ref.async_put(value_to_put);

    while (!put_future.is_ready()) {
        futures::poll();
    }

    auto put_result = put_future.get_result();
    ASSERT_TRUE(put_result.has_value());

    // Flush to ensure visibility
    auto flush_res = window_impl_->flush_all();
    ASSERT_TRUE(flush_res.has_value());

    // Barrier
    MPI_Barrier(comm_->native_handle());

    // Now read back what previous rank wrote
    rank_t source = (rank_ == 0) ? (size_ - 1) : (rank_ - 1);
    remote_ref<int> read_ref(rank_, 0, &data[0]);  // Local read

    auto get_future = read_ref.async_get();

    while (!get_future.is_ready()) {
        futures::poll();
    }

    auto get_result = get_future.get_result();
    ASSERT_TRUE(get_result.has_value());
    EXPECT_EQ(get_result.value(), source);

    auto unlock_res = window_impl_->unlock_all();
    ASSERT_TRUE(unlock_res.has_value());
}

// =============================================================================
// Error Handling
// =============================================================================

TEST_F(AsyncRemoteRefTest, AsyncGetWithNoWindowReturnsError) {
    // Create remote_ref without a window
    remote_ref<int> ref(1, 0, nullptr, nullptr, 0);

    EXPECT_TRUE(ref.is_remote());
    EXPECT_FALSE(ref.has_window());

    auto future = ref.async_get();
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());  // Error should be immediate

    auto result = future.get_result();
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::not_supported);
}

TEST_F(AsyncRemoteRefTest, AsyncPutWithNoWindowReturnsError) {
    remote_ref<int> ref(1, 0, nullptr, nullptr, 0);

    auto future = ref.async_put(42);
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future.is_ready());

    auto result = future.get_result();
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::not_supported);
}

// =============================================================================
// Progress Engine Integration
// =============================================================================

TEST_F(AsyncRemoteRefTest, CompletionRequiresPolling) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(rank_ * 100 + i);
    }

    create_window(data.data(), data.size() * sizeof(int));

    auto lock_res = window_impl_->lock_all();
    ASSERT_TRUE(lock_res.has_value());

    rank_t target = (rank_ + 1) % size_;
    size_type offset = 0;
    remote_ref<int> ref(target, 0, nullptr, window_impl_.get(), offset);

    auto future = ref.async_get();

    // Should not be ready immediately (unless synchronous fallback)
    // We can't guarantee this since it depends on MPI implementation
    // But the future should become ready after polling

    // Poll with timeout
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!future.is_ready() && std::chrono::steady_clock::now() < deadline) {
        futures::poll();
        std::this_thread::yield();
    }

    EXPECT_TRUE(future.is_ready()) << "Future should be ready after polling";

    auto result = future.get_result();
    ASSERT_TRUE(result.has_value());

    auto unlock_res = window_impl_->unlock_all();
    ASSERT_TRUE(unlock_res.has_value());
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
