// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_algorithm_result.cpp
/// @brief Unit tests for algorithm_result.hpp (Phase 10, T07)
/// @details Tests algorithm_result traits, unified_result, make_algorithm_result,
///          make_algorithm_error, and execute_algorithm.

// Suppress false-positive from GCC 13 in Release mode: variant destructor
// through unified_result triggers maybe-uninitialized on std::string capacity.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <dtl/futures/algorithm_result.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/error/result.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace dtl::futures::test {

// =============================================================================
// algorithm_result Trait Tests
// =============================================================================

TEST(AlgorithmResultTest, SyncPolicyResultTypeIsResult) {
    using result_type = algorithm_result_t<dtl::seq, int>;
    static_assert(std::is_same_v<result_type, result<int>>);
}

TEST(AlgorithmResultTest, ParPolicyResultTypeIsResult) {
    using result_type = algorithm_result_t<dtl::par, int>;
    static_assert(std::is_same_v<result_type, result<int>>);
}

TEST(AlgorithmResultTest, AsyncPolicyResultTypeIsFuture) {
    using result_type = algorithm_result_t<async_policy, int>;
    static_assert(std::is_same_v<result_type, distributed_future<int>>);
}

TEST(AlgorithmResultTest, SyncVoidResultType) {
    using result_type = algorithm_result_t<dtl::seq, void>;
    static_assert(std::is_same_v<result_type, result<void>>);
}

TEST(AlgorithmResultTest, AsyncVoidResultType) {
    using result_type = algorithm_result_t<async_policy, void>;
    static_assert(std::is_same_v<result_type, distributed_future<void>>);
}

// =============================================================================
// make_algorithm_result Tests
// =============================================================================

TEST(AlgorithmResultTest, MakeResultSyncInt) {
    auto r = make_algorithm_result<dtl::seq>(42);
    static_assert(std::is_same_v<decltype(r), result<int>>);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(AlgorithmResultTest, MakeResultSyncDouble) {
    auto r = make_algorithm_result<dtl::seq>(3.14);
    static_assert(std::is_same_v<decltype(r), result<double>>);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r.value(), 3.14);
}

TEST(AlgorithmResultTest, MakeResultSyncString) {
    auto r = make_algorithm_result<dtl::seq>(std::string("hello"));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), "hello");
}

TEST(AlgorithmResultTest, MakeResultAsyncInt) {
    auto f = make_algorithm_result<async_policy>(42);
    static_assert(std::is_same_v<decltype(f), distributed_future<int>>);
    ASSERT_TRUE(f.valid());
    ASSERT_TRUE(f.is_ready());
    EXPECT_EQ(f.get(), 42);
}

TEST(AlgorithmResultTest, MakeResultVoidSync) {
    auto r = make_algorithm_result_void<dtl::seq>();
    static_assert(std::is_same_v<decltype(r), result<void>>);
    EXPECT_TRUE(r.has_value());
}

TEST(AlgorithmResultTest, MakeResultVoidAsync) {
    auto f = make_algorithm_result_void<async_policy>();
    static_assert(std::is_same_v<decltype(f), distributed_future<void>>);
    ASSERT_TRUE(f.valid());
    ASSERT_TRUE(f.is_ready());
    // Should not throw
    EXPECT_NO_THROW(f.get());
}

// =============================================================================
// make_algorithm_error Tests
// =============================================================================

TEST(AlgorithmResultTest, MakeErrorSync) {
    auto r = make_algorithm_error<dtl::seq, int>(
        status(status_code::invalid_argument, no_rank, "bad arg"));
    EXPECT_FALSE(r.has_value());
    EXPECT_TRUE(r.has_error());
}

TEST(AlgorithmResultTest, MakeErrorAsync) {
    auto f = make_algorithm_error<async_policy, int>(
        status(status_code::invalid_argument, no_rank, "bad arg"));
    ASSERT_TRUE(f.valid());
    ASSERT_TRUE(f.is_ready());
    auto res = f.get_result();
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// unified_result Tests
// =============================================================================

TEST(AlgorithmResultTest, UnifiedResultSyncConstruction) {
    result<int> sync_result(42);
    unified_result<int> ur(std::move(sync_result));

    EXPECT_FALSE(ur.is_async());
    EXPECT_TRUE(ur.is_ready());
}

TEST(AlgorithmResultTest, UnifiedResultAsyncConstruction) {
    auto future = make_ready_distributed_future(42);
    unified_result<int> ur(std::move(future));

    EXPECT_TRUE(ur.is_async());
    EXPECT_TRUE(ur.is_ready());
}

TEST(AlgorithmResultTest, UnifiedResultSyncGet) {
    result<int> sync_result(99);
    unified_result<int> ur(std::move(sync_result));

    auto r = ur.get();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 99);
}

TEST(AlgorithmResultTest, UnifiedResultAsyncGet) {
    auto future = make_ready_distributed_future(77);
    unified_result<int> ur(std::move(future));

    auto r = ur.get();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 77);
}

TEST(AlgorithmResultTest, UnifiedResultSyncGetFuture) {
    result<int> sync_result(55);
    unified_result<int> ur(std::move(sync_result));

    auto f = ur.get_future();
    ASSERT_TRUE(f.valid());
    ASSERT_TRUE(f.is_ready());
    EXPECT_EQ(f.get(), 55);
}

TEST(AlgorithmResultTest, UnifiedResultAsyncGetFuture) {
    auto future = make_ready_distributed_future(33);
    unified_result<int> ur(std::move(future));

    auto f = ur.get_future();
    ASSERT_TRUE(f.valid());
    EXPECT_EQ(f.get(), 33);
}

TEST(AlgorithmResultTest, UnifiedResultSyncWait) {
    result<int> sync_result(1);
    unified_result<int> ur(std::move(sync_result));

    // Should not block since sync results are always ready
    EXPECT_NO_THROW(ur.wait());
}

TEST(AlgorithmResultTest, UnifiedResultSyncError) {
    auto err = make_error<int>(status_code::operation_failed, "oops");
    unified_result<int> ur(std::move(err));

    auto r = ur.get();
    EXPECT_TRUE(r.has_error());
}

TEST(AlgorithmResultTest, UnifiedResultAsyncError) {
    auto future = make_failed_distributed_future<int>(
        status(status_code::operation_failed, no_rank, "async oops"));
    unified_result<int> ur(std::move(future));

    auto r = ur.get();
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Policy Detection Tests
// =============================================================================

TEST(AlgorithmResultTest, IsSyncPolicySeq) {
    EXPECT_TRUE(is_sync_policy_v<dtl::seq>);
}

TEST(AlgorithmResultTest, IsSyncPolicyPar) {
    EXPECT_TRUE(is_sync_policy_v<dtl::par>);
}

TEST(AlgorithmResultTest, IsSyncPolicyNotAsync) {
    EXPECT_FALSE(is_sync_policy_v<async_policy>);
}

TEST(AlgorithmResultTest, IsAsyncPolicyAsync) {
    EXPECT_TRUE(is_async_policy_v<dtl::async>);
}

TEST(AlgorithmResultTest, IsAsyncPolicyNotSeq) {
    EXPECT_FALSE(is_async_policy_v<dtl::seq>);
}

// =============================================================================
// execute_algorithm Tests
// =============================================================================

TEST(AlgorithmResultTest, ExecuteAlgorithmSyncReturnsValue) {
    auto r = execute_algorithm<dtl::seq>([] { return 42; });
    static_assert(std::is_same_v<decltype(r), result<int>>);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(AlgorithmResultTest, ExecuteAlgorithmSyncVoid) {
    bool called = false;
    auto r = execute_algorithm<dtl::seq>([&called] { called = true; });
    static_assert(std::is_same_v<decltype(r), result<void>>);
    EXPECT_TRUE(r.has_value());
    EXPECT_TRUE(called);
}

TEST(AlgorithmResultTest, ExecuteAlgorithmSyncException) {
    auto r = execute_algorithm<dtl::seq>([]() -> int {
        throw std::runtime_error("test error");
    });
    EXPECT_TRUE(r.has_error());
}

TEST(AlgorithmResultTest, ExecuteAlgorithmAsyncReturnsValue) {
    auto f = execute_algorithm<dtl::async>([] { return 100; });
    static_assert(std::is_same_v<decltype(f), distributed_future<int>>);
    ASSERT_TRUE(f.valid());

    // Drive progress engine to execute the callback
    for (int i = 0; i < 100 && !f.is_ready(); ++i) {
        progress_engine::instance().poll();
    }

    ASSERT_TRUE(f.is_ready());
    EXPECT_EQ(f.get(), 100);
}

TEST(AlgorithmResultTest, ExecuteAlgorithmAsyncVoid) {
    std::atomic<bool> called{false};
    auto f = execute_algorithm<dtl::async>([&called] { called = true; });
    static_assert(std::is_same_v<decltype(f), distributed_future<void>>);
    ASSERT_TRUE(f.valid());

    // Drive progress engine
    for (int i = 0; i < 100 && !f.is_ready(); ++i) {
        progress_engine::instance().poll();
    }

    ASSERT_TRUE(f.is_ready());
    EXPECT_TRUE(called.load());
    EXPECT_NO_THROW(f.get());
}

TEST(AlgorithmResultTest, ExecuteAlgorithmAsyncException) {
    auto f = execute_algorithm<dtl::async>([]() -> int {
        throw std::runtime_error("async error");
    });
    ASSERT_TRUE(f.valid());

    // Drive progress engine
    for (int i = 0; i < 100 && !f.is_ready(); ++i) {
        progress_engine::instance().poll();
    }

    ASSERT_TRUE(f.is_ready());
    auto res = f.get_result();
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(AlgorithmResultTest, UnifiedResultMoveConstruction) {
    result<int> sync_result(42);
    unified_result<int> ur1(std::move(sync_result));
    unified_result<int> ur2(std::move(ur1));

    auto r = ur2.get();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(AlgorithmResultTest, UnifiedResultMoveAssignment) {
    result<int> sync_result1(10);
    result<int> sync_result2(20);
    unified_result<int> ur1(std::move(sync_result1));
    unified_result<int> ur2(std::move(sync_result2));

    ur2 = std::move(ur1);

    auto r = ur2.get();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 10);
}

}  // namespace dtl::futures::test
