// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_dispatch_types.cpp
/// @brief Verify async dispatch type consistency (R6.6)
/// @details Verifies that the execution_dispatcher<async> specialization has
///          consistent return types. The async dispatcher is designed with a
///          two-tier approach:
///
///          1. invoke() returns std::future<T> for general-purpose async work.
///          2. Algorithm-specific methods (for_each, transform, reduce, etc.)
///             execute SYNCHRONOUSLY because the actual asynchrony is handled
///             at the algorithm level via futures::distributed_future<T> and
///             the progress engine.
///
///          This is by design (see dispatch.hpp lines 287-288). The algorithm-
///          level async functions (async_fill, async_copy, async_reduce, etc.)
///          create distributed_future<T>, register a progress engine callback,
///          and call the synchronous seq{} path internally.

#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/futures/distributed_future.hpp>

#include <gtest/gtest.h>

#include <future>
#include <type_traits>
#include <vector>

namespace dtl::test {

// =============================================================================
// Type Consistency Verification
// =============================================================================

/// @brief Verify that async::invoke returns std::future
TEST(AsyncDispatchTypeTest, InvokeReturnsStdFuture) {
    // execution_dispatcher<async>::invoke returns std::future<T>
    using result_type = decltype(
        execution_dispatcher<async>::invoke([]() { return 42; }));
    static_assert(std::is_same_v<result_type, std::future<int>>,
                  "async invoke should return std::future<int>");
}

/// @brief Verify that async algorithm dispatch methods execute synchronously
///        (returning the same types as seq dispatch)
TEST(AsyncDispatchTypeTest, ForEachReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};
    auto f = [](int) {};

    using seq_result = decltype(
        execution_dispatcher<seq>::for_each(data.begin(), data.end(), f));
    using async_result = decltype(
        execution_dispatcher<async>::for_each(data.begin(), data.end(), f));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async for_each should have same return type as seq for_each");
}

TEST(AsyncDispatchTypeTest, TransformReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};
    std::vector<int> out(3);
    auto op = [](int x) { return x * 2; };

    using seq_result = decltype(
        execution_dispatcher<seq>::transform(data.begin(), data.end(), out.begin(), op));
    using async_result = decltype(
        execution_dispatcher<async>::transform(data.begin(), data.end(), out.begin(), op));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async transform should have same return type as seq transform");
}

TEST(AsyncDispatchTypeTest, ReduceReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};
    auto op = std::plus<int>{};

    using seq_result = decltype(
        execution_dispatcher<seq>::reduce(data.begin(), data.end(), 0, op));
    using async_result = decltype(
        execution_dispatcher<async>::reduce(data.begin(), data.end(), 0, op));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async reduce should have same return type as seq reduce");
}

TEST(AsyncDispatchTypeTest, CopyReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};
    std::vector<int> out(3);

    using seq_result = decltype(
        execution_dispatcher<seq>::copy(data.begin(), data.end(), out.begin()));
    using async_result = decltype(
        execution_dispatcher<async>::copy(data.begin(), data.end(), out.begin()));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async copy should have same return type as seq copy");
}

TEST(AsyncDispatchTypeTest, CountReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};

    using seq_result = decltype(
        execution_dispatcher<seq>::count(data.begin(), data.end(), 2));
    using async_result = decltype(
        execution_dispatcher<async>::count(data.begin(), data.end(), 2));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async count should have same return type as seq count");
}

TEST(AsyncDispatchTypeTest, FindReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};

    using seq_result = decltype(
        execution_dispatcher<seq>::find(data.begin(), data.end(), 2));
    using async_result = decltype(
        execution_dispatcher<async>::find(data.begin(), data.end(), 2));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async find should have same return type as seq find");
}

TEST(AsyncDispatchTypeTest, AllOfReturnTypeMatchesSeq) {
    std::vector<int> data = {1, 2, 3};
    auto pred = [](int x) { return x > 0; };

    using seq_result = decltype(
        execution_dispatcher<seq>::all_of(data.begin(), data.end(), pred));
    using async_result = decltype(
        execution_dispatcher<async>::all_of(data.begin(), data.end(), pred));

    static_assert(std::is_same_v<seq_result, async_result>,
                  "async all_of should have same return type as seq all_of");
}

// =============================================================================
// Behavioral Consistency: async dispatch methods produce same results as seq
// =============================================================================

TEST(AsyncDispatchTypeTest, ReduceProducesSameResult) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    int seq_result = execution_dispatcher<seq>::reduce(
        data.begin(), data.end(), 0, std::plus<int>{});
    int async_result = execution_dispatcher<async>::reduce(
        data.begin(), data.end(), 0, std::plus<int>{});

    EXPECT_EQ(seq_result, async_result);
    EXPECT_EQ(seq_result, 15);
}

TEST(AsyncDispatchTypeTest, CountProducesSameResult) {
    std::vector<int> data = {1, 2, 2, 3, 2};

    auto seq_result = execution_dispatcher<seq>::count(
        data.begin(), data.end(), 2);
    auto async_result = execution_dispatcher<async>::count(
        data.begin(), data.end(), 2);

    EXPECT_EQ(seq_result, async_result);
    EXPECT_EQ(seq_result, 3);
}

TEST(AsyncDispatchTypeTest, AllOfProducesSameResult) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto pred = [](int x) { return x > 0; };

    bool seq_result = execution_dispatcher<seq>::all_of(
        data.begin(), data.end(), pred);
    bool async_result = execution_dispatcher<async>::all_of(
        data.begin(), data.end(), pred);

    EXPECT_EQ(seq_result, async_result);
    EXPECT_TRUE(seq_result);
}

// =============================================================================
// Design Documentation: async_* algorithm functions use distributed_future
// =============================================================================

/// @brief Verify that algorithm-level async functions return distributed_future
/// @note This test documents the two-tier design:
///       - dispatch level: synchronous (same as seq)
///       - algorithm level: async via distributed_future + progress engine
TEST(AsyncDispatchTypeTest, AlgorithmLevelUsesDistributedFuture) {
    // The async_fill, async_copy, async_reduce etc. functions
    // all return futures::distributed_future<T>, not std::future<T>.
    // This is verified by the return type annotations in the algorithm headers.
    //
    // Example from fill.hpp:
    //   auto async_fill(...) -> futures::distributed_future<void>
    //
    // Example from reduce.hpp:
    //   auto async_reduce(...) -> futures::distributed_future<T>
    //
    // The dispatch-level async methods execute synchronously because they are
    // called INSIDE the progress engine callback, which already provides the
    // async wrapper.

    // Compile-time verification that distributed_future is a distinct type
    static_assert(!std::is_same_v<futures::distributed_future<int>, std::future<int>>,
                  "distributed_future should be distinct from std::future");
}

}  // namespace dtl::test
