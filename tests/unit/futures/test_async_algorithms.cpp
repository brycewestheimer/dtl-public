// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_algorithms.cpp
/// @brief Test async algorithm variants with progress engine
/// @details Verifies that async wrappers properly use the progress engine

#include <dtl/algorithms/algorithms.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/core/environment.hpp>

#include <gtest/gtest.h>
#include <functional>

using namespace dtl;

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

class AsyncAlgorithmsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Environment is already initialized by main
    }
};

// ============================================================================
// Async Reduce Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncReduceBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 1);
    }

    // Start async reduce
    auto future = async_reduce(vec, 0, std::plus<>{});

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    // Get result
    int sum = future.get();

    // Verify: local sum only (no communicator)
    int expected_local = 0;
    for (size_t i = 0; i < local_view.size(); ++i) {
        expected_local += local_view[i];
    }
    EXPECT_EQ(sum, expected_local);
}

// ============================================================================
// Async Sort Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncSortBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(local_view.size() - i);
    }

    // Start async sort
    auto future = async_sort(vec);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    // Wait for completion
    future.get();

    // Verify sorted
    for (size_t i = 1; i < local_view.size(); ++i) {
        EXPECT_LE(local_view[i - 1], local_view[i]);
    }
}

// ============================================================================
// Async Replace Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncReplaceBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = (i % 2 == 0) ? 0 : 1;
    }

    // Start async replace
    auto future = async_replace(vec, 0, 42);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();

    // Verify all zeros replaced with 42
    size_t count_42 = 0;
    for (size_t i = 0; i < local_view.size(); ++i) {
        if (local_view[i] == 42) ++count_42;
        EXPECT_NE(local_view[i], 0); // No zeros should remain
    }
    EXPECT_GT(count_42, 0u);
    EXPECT_EQ(result.count, count_42);
}

TEST_F(AsyncAlgorithmsTest, AsyncReplaceIfBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    // Replace negative values (none in this case)
    auto future = async_replace_if(vec, [](int x) { return x < 0; }, 0);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_EQ(result.count, 0u); // No replacements
}

// ============================================================================
// Async Scan Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncInclusiveScanBasic) {
    distributed_vector<int> input(50, test_context{0, 1});
    distributed_vector<int> output(50, test_context{0, 1});

    auto in_local = input.local_view();
    for (size_t i = 0; i < in_local.size(); ++i) {
        in_local[i] = 1;
    }

    // Start async inclusive scan
    auto future = async_inclusive_scan(input, output, 0, std::plus<>{});

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    future.get();

    // Verify prefix sum
    auto out_local = output.local_view();
    for (size_t i = 0; i < out_local.size(); ++i) {
        EXPECT_EQ(out_local[i], static_cast<int>(i + 1));
    }
}

TEST_F(AsyncAlgorithmsTest, AsyncExclusiveScanBasic) {
    distributed_vector<int> input(50, test_context{0, 1});
    distributed_vector<int> output(50, test_context{0, 1});

    auto in_local = input.local_view();
    for (size_t i = 0; i < in_local.size(); ++i) {
        in_local[i] = 1;
    }

    // Start async exclusive scan
    auto future = async_exclusive_scan(input, output, 0, std::plus<>{});

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    future.get();

    // Verify prefix sum (exclusive)
    auto out_local = output.local_view();
    for (size_t i = 0; i < out_local.size(); ++i) {
        EXPECT_EQ(out_local[i], static_cast<int>(i));
    }
}

// ============================================================================
// Async Accumulate Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncAccumulateBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 1);
    }

    // Start async accumulate
    auto future = async_accumulate(vec, 0, std::plus<>{});

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    int sum = future.get();

    // Verify: local sum only (no communicator)
    int expected_local = 0;
    for (size_t i = 0; i < local_view.size(); ++i) {
        expected_local += local_view[i];
    }
    EXPECT_EQ(sum, expected_local);
}

// ============================================================================
// Async MinMax Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncMinElementBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 10);
    }

    // Start async min_element
    auto future = async_min_element(vec);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_TRUE(result.valid);
    EXPECT_GE(result.value, 10);
}

TEST_F(AsyncAlgorithmsTest, AsyncMaxElementBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    // Start async max_element
    auto future = async_max_element(vec);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_TRUE(result.valid);
    EXPECT_LT(result.value, 100);
}

TEST_F(AsyncAlgorithmsTest, AsyncMinMaxElementBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    // Start async minmax_element
    auto future = async_minmax_element(vec);

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_TRUE(result.min.valid);
    EXPECT_TRUE(result.max.valid);
    EXPECT_LE(result.min.value, result.max.value);
}

// ============================================================================
// Multiple Async Operations
// ============================================================================

TEST_F(AsyncAlgorithmsTest, MultipleAsyncOperations) {
    distributed_vector<int> vec1(50, test_context{0, 1});
    distributed_vector<int> vec2(50, test_context{0, 1});

    auto local1 = vec1.local_view();
    auto local2 = vec2.local_view();
    for (size_t i = 0; i < local1.size(); ++i) {
        local1[i] = static_cast<int>(i);
        local2[i] = static_cast<int>(i * 2);
    }

    // Start multiple async operations
    auto future1 = async_reduce(vec1, 0, std::plus<>{});
    auto future2 = async_reduce(vec2, 0, std::plus<>{});
    auto future3 = async_min_element(vec1);

    // Poll progress until all complete
    while (!future1.is_ready() || !future2.is_ready() || !future3.is_ready()) {
        futures::make_progress();
    }

    // All should complete
    EXPECT_NO_THROW({
        auto sum1 = future1.get();
        auto sum2 = future2.get();
        auto min_res = future3.get();

        EXPECT_GT(sum1, 0);
        EXPECT_GT(sum2, sum1);
        EXPECT_TRUE(min_res.valid);
    });
}

// ============================================================================
// Async Transform-Reduce Tests
// ============================================================================

TEST_F(AsyncAlgorithmsTest, AsyncTransformReduceReturnsFuture) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 1);
    }

    auto future = async_transform_reduce(
        vec, 0, std::plus<>{}, [](int x) { return x * x; });

    // Verify it returns a distributed_future, not result<T>
    static_assert(std::is_same_v<decltype(future), futures::distributed_future<int>>,
                  "async_transform_reduce must return distributed_future<T>");

    while (!future.is_ready()) {
        futures::make_progress();
    }

    int sum_of_squares = future.get();

    int expected = 0;
    for (size_t i = 0; i < local_view.size(); ++i) {
        expected += local_view[i] * local_view[i];
    }
    EXPECT_EQ(sum_of_squares, expected);
}

TEST_F(AsyncAlgorithmsTest, AsyncTransformReduceFutureIsReady) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = 1;
    }

    auto future = async_transform_reduce(
        vec, 0, std::plus<>{}, [](int x) { return x; });

    // Synchronous execution under the hood means it should be immediately ready
    EXPECT_TRUE(future.is_ready());
    EXPECT_TRUE(future.valid());
    EXPECT_EQ(future.get(), 10);
}
