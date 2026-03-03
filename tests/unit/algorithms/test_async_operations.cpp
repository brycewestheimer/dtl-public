// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_operations.cpp
/// @brief Unit tests for Phase 09: Async algorithm variants returning distributed_future
/// @details Tests async_for_each, async_transform, async_count, async_count_if,
///          async_all_of, async_any_of, async_none_of, async_copy, async_fill,
///          and async dispatch routing.

#include <dtl/algorithms/algorithms.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/futures/distributed_future.hpp>
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

class AsyncOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Environment is already initialized by main
    }
};

// ============================================================================
// T01: async_for_each tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncForEachReturnsFuture) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    int sum = 0;
    auto future = async_for_each(vec, [&sum](int x) { sum += x; });

    // Should return a valid distributed_future<void>
    static_assert(std::is_same_v<decltype(future), futures::distributed_future<void>>,
                  "async_for_each must return distributed_future<void>");

    // Poll progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    future.get();

    // Verify all elements were processed
    int expected = 0;
    for (size_t i = 0; i < 100; ++i) {
        expected += static_cast<int>(i);
    }
    EXPECT_EQ(sum, expected);
}

TEST_F(AsyncOperationsTest, AsyncForEachModifiesElements) {
    distributed_vector<int> vec(50, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 1);
    }

    auto future = async_for_each(vec, [](int& x) { x *= 2; });

    while (!future.is_ready()) {
        futures::make_progress();
    }
    future.get();

    for (size_t i = 0; i < local_view.size(); ++i) {
        EXPECT_EQ(local_view[i], static_cast<int>((i + 1) * 2));
    }
}

TEST_F(AsyncOperationsTest, AsyncForEachEmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});

    int count = 0;
    auto future = async_for_each(vec, [&count](int) { ++count; });

    while (!future.is_ready()) {
        futures::make_progress();
    }
    future.get();

    EXPECT_EQ(count, 0);
}

// ============================================================================
// T02: async_transform tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncTransformTwoContainers) {
    distributed_vector<int> src(100, test_context{0, 1});
    distributed_vector<int> dst(100, test_context{0, 1});

    auto src_view = src.local_view();
    for (size_t i = 0; i < src_view.size(); ++i) {
        src_view[i] = static_cast<int>(i + 1);
    }

    auto future = async_transform(src, dst, [](int x) { return x * 3; });

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<void>>,
                  "async_transform must return distributed_future<void>");

    while (!future.is_ready()) {
        futures::make_progress();
    }
    future.get();

    auto dst_view = dst.local_view();
    for (size_t i = 0; i < dst_view.size(); ++i) {
        EXPECT_EQ(dst_view[i], static_cast<int>((i + 1) * 3));
    }
}

TEST_F(AsyncOperationsTest, AsyncTransformInPlace) {
    distributed_vector<int> vec(50, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_transform(vec, [](int x) { return x + 10; });

    while (!future.is_ready()) {
        futures::make_progress();
    }
    future.get();

    for (size_t i = 0; i < local_view.size(); ++i) {
        EXPECT_EQ(local_view[i], static_cast<int>(i + 10));
    }
}

// ============================================================================
// T03: async_count / async_count_if tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncCountBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = (i % 3 == 0) ? 42 : static_cast<int>(i);
    }

    auto future = async_count(vec, 42);

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<size_type>>,
                  "async_count must return distributed_future<size_type>");

    while (!future.is_ready()) {
        futures::make_progress();
    }

    size_type result = future.get();

    // Count how many multiples of 3 in [0, 100)
    size_type expected = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (i % 3 == 0) ++expected;
    }
    EXPECT_EQ(result, expected);
}

TEST_F(AsyncOperationsTest, AsyncCountIfBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_count_if(vec, [](int x) { return x >= 50; });

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<size_type>>,
                  "async_count_if must return distributed_future<size_type>");

    while (!future.is_ready()) {
        futures::make_progress();
    }

    size_type result = future.get();
    EXPECT_EQ(result, 50u);
}

TEST_F(AsyncOperationsTest, AsyncCountZeroMatches) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = 0;
    }

    auto future = async_count(vec, 999);
    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_EQ(future.get(), 0u);
}

// ============================================================================
// T04: async predicates tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncAllOfTrue) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i + 1);  // all positive
    }

    auto future = async_all_of(vec, [](int x) { return x > 0; });

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<bool>>,
                  "async_all_of must return distributed_future<bool>");

    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_TRUE(future.get());
}

TEST_F(AsyncOperationsTest, AsyncAllOfFalse) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);  // includes 0
    }

    auto future = async_all_of(vec, [](int x) { return x > 0; });
    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_FALSE(future.get());
}

TEST_F(AsyncOperationsTest, AsyncAnyOfTrue) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_any_of(vec, [](int x) { return x == 50; });

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<bool>>,
                  "async_any_of must return distributed_future<bool>");

    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_TRUE(future.get());
}

TEST_F(AsyncOperationsTest, AsyncAnyOfFalse) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_any_of(vec, [](int x) { return x > 1000; });
    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_FALSE(future.get());
}

TEST_F(AsyncOperationsTest, AsyncNoneOfTrue) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_none_of(vec, [](int x) { return x < 0; });

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<bool>>,
                  "async_none_of must return distributed_future<bool>");

    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_TRUE(future.get());
}

TEST_F(AsyncOperationsTest, AsyncNoneOfFalse) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_none_of(vec, [](int x) { return x == 0; });
    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_FALSE(future.get());
}

// ============================================================================
// T05: async_copy / async_fill tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncCopyBasic) {
    distributed_vector<int> src(100, test_context{0, 1});
    distributed_vector<int> dst(100, test_context{0, 1});

    auto src_view = src.local_view();
    for (size_t i = 0; i < src_view.size(); ++i) {
        src_view[i] = static_cast<int>(i * 7);
    }

    auto future = async_copy(src, dst);

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<copy_result>>,
                  "async_copy must return distributed_future<copy_result>");

    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_EQ(result.count, 100u);
    EXPECT_TRUE(result.success);

    auto dst_view = dst.local_view();
    for (size_t i = 0; i < dst_view.size(); ++i) {
        EXPECT_EQ(dst_view[i], static_cast<int>(i * 7));
    }
}

TEST_F(AsyncOperationsTest, AsyncCopyEmptyContainers) {
    distributed_vector<int> src(0, test_context{0, 1});
    distributed_vector<int> dst(0, test_context{0, 1});

    auto future = async_copy(src, dst);
    while (!future.is_ready()) {
        futures::make_progress();
    }

    auto result = future.get();
    EXPECT_EQ(result.count, 0u);
}

TEST_F(AsyncOperationsTest, AsyncFillBasic) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto future = async_fill(vec, 42);

    static_assert(std::is_same_v<decltype(future), futures::distributed_future<void>>,
                  "async_fill must return distributed_future<void>");

    while (!future.is_ready()) {
        futures::make_progress();
    }
    future.get();

    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        EXPECT_EQ(local_view[i], 42);
    }
}

TEST_F(AsyncOperationsTest, AsyncFillEmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});

    auto future = async_fill(vec, 99);
    while (!future.is_ready()) {
        futures::make_progress();
    }
    EXPECT_NO_THROW(future.get());
}

// ============================================================================
// T06: Dispatch with async policy tests
// ============================================================================

TEST_F(AsyncOperationsTest, AsyncDispatchForEach) {
    // Verify that dispatch_for_each with async{} works correctly
    std::vector<int> data = {1, 2, 3, 4, 5};
    int sum = 0;

    dispatch_for_each(async{}, data.begin(), data.end(), [&sum](int x) { sum += x; });
    EXPECT_EQ(sum, 15);
}

TEST_F(AsyncOperationsTest, AsyncDispatchTransform) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5, 0);

    dispatch_transform(async{}, src.begin(), src.end(), dst.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(dst, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST_F(AsyncOperationsTest, AsyncDispatchFill) {
    std::vector<int> data(5, 0);

    dispatch_fill(async{}, data.begin(), data.end(), 7);
    EXPECT_EQ(data, (std::vector<int>{7, 7, 7, 7, 7}));
}

TEST_F(AsyncOperationsTest, AsyncDispatchCopy) {
    std::vector<int> src = {10, 20, 30};
    std::vector<int> dst(3, 0);

    dispatch_copy(async{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, (std::vector<int>{10, 20, 30}));
}

TEST_F(AsyncOperationsTest, AsyncDispatchCount) {
    std::vector<int> data = {1, 2, 2, 3, 2};

    auto result = dispatch_count(async{}, data.begin(), data.end(), 2);
    EXPECT_EQ(result, 3);
}

TEST_F(AsyncOperationsTest, AsyncDispatchCountIf) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    auto result = dispatch_count_if(async{}, data.begin(), data.end(),
                                    [](int x) { return x > 3; });
    EXPECT_EQ(result, 2);
}

TEST_F(AsyncOperationsTest, AsyncDispatchAllOf) {
    std::vector<int> data = {2, 4, 6, 8};

    bool result = dispatch_all_of(async{}, data.begin(), data.end(),
                                  [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST_F(AsyncOperationsTest, AsyncDispatchAnyOf) {
    std::vector<int> data = {1, 3, 5, 6};

    bool result = dispatch_any_of(async{}, data.begin(), data.end(),
                                  [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST_F(AsyncOperationsTest, AsyncDispatchNoneOf) {
    std::vector<int> data = {1, 3, 5, 7};

    bool result = dispatch_none_of(async{}, data.begin(), data.end(),
                                   [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

// ============================================================================
// Multiple concurrent async operations
// ============================================================================

TEST_F(AsyncOperationsTest, MultipleConcurrentAsyncAlgorithms) {
    distributed_vector<int> vec1(50, test_context{0, 1});
    distributed_vector<int> vec2(50, test_context{0, 1});
    distributed_vector<int> vec3(50, test_context{0, 1});

    auto v1 = vec1.local_view();
    auto v2 = vec2.local_view();
    for (size_t i = 0; i < v1.size(); ++i) {
        v1[i] = static_cast<int>(i);
        v2[i] = 0;
    }

    // Launch multiple async operations simultaneously
    auto fill_future = async_fill(vec3, 5);
    auto count_future = async_count_if(vec1, [](int x) { return x % 2 == 0; });
    auto copy_future = async_copy(vec1, vec2);

    // Poll progress until all complete
    while (!fill_future.is_ready() || !count_future.is_ready() || !copy_future.is_ready()) {
        futures::make_progress();
    }

    // Verify all completed correctly
    fill_future.get();
    auto cnt = count_future.get();
    auto copy_res = copy_future.get();

    EXPECT_EQ(cnt, 25u);  // 0,2,4,...,48 = 25 even numbers
    EXPECT_EQ(copy_res.count, 50u);

    // Verify fill result
    auto v3 = vec3.local_view();
    for (size_t i = 0; i < v3.size(); ++i) {
        EXPECT_EQ(v3[i], 5);
    }
}

TEST_F(AsyncOperationsTest, AsyncFutureIsReadyBeforeGet) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local_view = vec.local_view();
    for (size_t i = 0; i < local_view.size(); ++i) {
        local_view[i] = static_cast<int>(i);
    }

    auto future = async_count(vec, 5);

    // Poll until ready
    while (!future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(future.is_ready());
    EXPECT_TRUE(future.valid());
    EXPECT_EQ(future.get(), 1u);
}
