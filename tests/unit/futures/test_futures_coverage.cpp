// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_futures_coverage.cpp
/// @brief Unit tests for additional futures module functionality
/// @details Phase 14 T10: unified_result, execute_algorithm, algorithm_result_t
///          type traits, policy detection, distributed_future/promise.

// Suppress false-positive maybe-uninitialized in optimized builds (GCC)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <dtl/futures/algorithm_result.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// algorithm_result_t Type Trait Tests
// =============================================================================

TEST(AlgorithmResultTraitsTest, SyncPolicyReturnsResult) {
    using res_t = futures::algorithm_result_t<dtl::seq, int>;
    static_assert(std::is_same_v<res_t, result<int>>);
    SUCCEED();
}

TEST(AlgorithmResultTraitsTest, SyncPolicyVoidReturnsResultVoid) {
    using res_t = futures::algorithm_result_t<dtl::seq, void>;
    static_assert(std::is_same_v<res_t, result<void>>);
    SUCCEED();
}

TEST(AlgorithmResultTraitsTest, AsyncPolicyReturnsFuture) {
    using res_t = futures::algorithm_result_t<futures::async_policy, int>;
    static_assert(std::is_same_v<res_t, futures::distributed_future<int>>);
    SUCCEED();
}

TEST(AlgorithmResultTraitsTest, AsyncPolicyVoidReturnsFutureVoid) {
    using res_t = futures::algorithm_result_t<futures::async_policy, void>;
    static_assert(std::is_same_v<res_t, futures::distributed_future<void>>);
    SUCCEED();
}

TEST(AlgorithmResultTraitsTest, ParPolicyReturnsResult) {
    using res_t = futures::algorithm_result_t<dtl::par, int>;
    static_assert(std::is_same_v<res_t, result<int>>);
    SUCCEED();
}

// =============================================================================
// Policy Detection Tests
// =============================================================================

TEST(PolicyDetectionTest, IsSyncPolicySeq) {
    EXPECT_TRUE(futures::is_sync_policy_v<dtl::seq>);
}

TEST(PolicyDetectionTest, IsSyncPolicyPar) {
    EXPECT_TRUE(futures::is_sync_policy_v<dtl::par>);
}

TEST(PolicyDetectionTest, AsyncPolicyIsNotSync) {
    EXPECT_FALSE(futures::is_sync_policy_v<futures::async_policy>);
}

TEST(PolicyDetectionTest, IsAsyncPolicyV) {
    EXPECT_TRUE(is_async_policy_v<dtl::async>);
    EXPECT_FALSE(is_async_policy_v<dtl::seq>);
    EXPECT_FALSE(is_async_policy_v<dtl::par>);
}

TEST(PolicyDetectionTest, IsSeqPolicyV) {
    EXPECT_TRUE(is_seq_policy_v<dtl::seq>);
    EXPECT_FALSE(is_seq_policy_v<dtl::par>);
    EXPECT_FALSE(is_seq_policy_v<dtl::async>);
}

TEST(PolicyDetectionTest, IsParPolicyV) {
    EXPECT_TRUE(is_par_policy_v<dtl::par>);
    EXPECT_FALSE(is_par_policy_v<dtl::seq>);
    EXPECT_FALSE(is_par_policy_v<dtl::async>);
}

// =============================================================================
// make_algorithm_result Tests
// =============================================================================

TEST(MakeAlgorithmResultTest, SyncResult) {
    auto r = futures::make_algorithm_result<dtl::seq>(42);
    static_assert(std::is_same_v<decltype(r), result<int>>);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(MakeAlgorithmResultTest, AsyncResult) {
    auto f = futures::make_algorithm_result<futures::async_policy>(42);
    static_assert(std::is_same_v<decltype(f), futures::distributed_future<int>>);
    ASSERT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
    EXPECT_EQ(f.get(), 42);
}

TEST(MakeAlgorithmResultTest, SyncVoidResult) {
    auto r = futures::make_algorithm_result_void<dtl::seq>();
    static_assert(std::is_same_v<decltype(r), result<void>>);
    EXPECT_TRUE(r.has_value());
}

TEST(MakeAlgorithmResultTest, AsyncVoidResult) {
    auto f = futures::make_algorithm_result_void<futures::async_policy>();
    static_assert(std::is_same_v<decltype(f), futures::distributed_future<void>>);
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
}

// =============================================================================
// make_algorithm_error Tests
// =============================================================================

TEST(MakeAlgorithmErrorTest, SyncError) {
    auto r = futures::make_algorithm_error<dtl::seq, int>(
        status(status_code::timeout, no_rank, "deadline"));
    EXPECT_TRUE(r.has_error());
}

TEST(MakeAlgorithmErrorTest, AsyncError) {
    auto f = futures::make_algorithm_error<futures::async_policy, int>(
        status(status_code::timeout, no_rank, "deadline"));
    ASSERT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
    auto r = f.get_result();
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// unified_result Tests
// =============================================================================

TEST(UnifiedResultTest, ConstructWithSyncResult) {
    result<int> r(42);
    futures::unified_result<int> ur(std::move(r));
    EXPECT_FALSE(ur.is_async());
    EXPECT_TRUE(ur.is_ready());
}

TEST(UnifiedResultTest, GetFromSyncResult) {
    futures::unified_result<int> ur(result<int>(42));
    auto val = ur.get();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(val.value(), 42);
}

TEST(UnifiedResultTest, ConstructWithAsyncResult) {
    auto future = futures::make_ready_distributed_future(99);
    futures::unified_result<int> ur(std::move(future));
    EXPECT_TRUE(ur.is_async());
    EXPECT_TRUE(ur.is_ready());
}

TEST(UnifiedResultTest, GetFromAsyncResult) {
    auto future = futures::make_ready_distributed_future(99);
    futures::unified_result<int> ur(std::move(future));
    auto val = ur.get();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(val.value(), 99);
}

TEST(UnifiedResultTest, WaitOnSyncIsNoop) {
    futures::unified_result<int> ur(result<int>(42));
    ur.wait();  // Should not hang
    EXPECT_TRUE(ur.is_ready());
}

TEST(UnifiedResultTest, WaitOnAsyncReady) {
    auto future = futures::make_ready_distributed_future(42);
    futures::unified_result<int> ur(std::move(future));
    ur.wait();  // Already ready
    EXPECT_TRUE(ur.is_ready());
}

TEST(UnifiedResultTest, GetFutureFromSync) {
    futures::unified_result<int> ur(result<int>(42));
    auto f = ur.get_future();
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
    EXPECT_EQ(f.get(), 42);
}

TEST(UnifiedResultTest, GetFutureFromAsync) {
    auto future = futures::make_ready_distributed_future(55);
    futures::unified_result<int> ur(std::move(future));
    auto f = ur.get_future();
    EXPECT_TRUE(f.valid());
    EXPECT_EQ(f.get(), 55);
}

TEST(UnifiedResultTest, GetFutureFromSyncError) {
    futures::unified_result<int> ur(result<int>(status_code::timeout));
    auto f = ur.get_future();
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
    auto r = f.get_result();
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Distributed Future Tests
// =============================================================================

TEST(DistributedFutureCovTest, DefaultInvalid) {
    futures::distributed_future<int> f;
    EXPECT_FALSE(f.valid());
}

TEST(DistributedFutureCovTest, ReadyFutureIsValid) {
    auto f = futures::make_ready_distributed_future(42);
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
}

TEST(DistributedFutureCovTest, GetValue) {
    auto f = futures::make_ready_distributed_future(42);
    EXPECT_EQ(f.get(), 42);
}

TEST(DistributedFutureCovTest, GetResult) {
    auto f = futures::make_ready_distributed_future(42);
    auto r = f.get_result();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(DistributedFutureCovTest, FailedFuture) {
    auto f = futures::make_failed_distributed_future<int>(
        status(status_code::timeout));
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
    auto r = f.get_result();
    EXPECT_TRUE(r.has_error());
}

TEST(DistributedFutureCovTest, WaitForReady) {
    auto f = futures::make_ready_distributed_future(42);
    auto s = f.wait_for(std::chrono::milliseconds(10));
    EXPECT_EQ(s, future_status::ready);
}

TEST(DistributedFutureCovTest, WaitForError) {
    auto f = futures::make_failed_distributed_future<int>(
        status(status_code::timeout));
    auto s = f.wait_for(std::chrono::milliseconds(10));
    EXPECT_EQ(s, future_status::error);
}

// =============================================================================
// Distributed Future Void Tests
// =============================================================================

TEST(DistributedFutureVoidTest, ReadyVoidFuture) {
    auto f = futures::make_ready_distributed_future();
    EXPECT_TRUE(f.valid());
    EXPECT_TRUE(f.is_ready());
}

TEST(DistributedFutureVoidTest, GetResultVoid) {
    auto f = futures::make_ready_distributed_future();
    auto r = f.get_result();
    EXPECT_TRUE(r.has_value());
}

TEST(DistributedFutureVoidTest, FailedVoidFuture) {
    auto f = futures::make_failed_distributed_future<void>(
        status(status_code::internal_error));
    EXPECT_TRUE(f.is_ready());
    auto r = f.get_result();
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Distributed Promise Tests
// =============================================================================

TEST(DistributedPromiseTest, SetValue) {
    futures::distributed_promise<int> promise;
    auto future = promise.get_future();
    EXPECT_FALSE(future.is_ready());

    promise.set_value(42);
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);
}

TEST(DistributedPromiseTest, SetError) {
    futures::distributed_promise<int> promise;
    auto future = promise.get_future();

    promise.set_error(status(status_code::timeout));
    EXPECT_TRUE(future.is_ready());
    auto r = future.get_result();
    EXPECT_TRUE(r.has_error());
}

TEST(DistributedPromiseTest, VoidPromise) {
    futures::distributed_promise<void> promise;
    auto future = promise.get_future();

    promise.set_value();
    EXPECT_TRUE(future.is_ready());
    auto r = future.get_result();
    EXPECT_TRUE(r.has_value());
}

// =============================================================================
// Progress Engine Basic Tests
// =============================================================================

TEST(ProgressEngineTest, SingletonInstance) {
    auto& e1 = futures::progress_engine::instance();
    auto& e2 = futures::progress_engine::instance();
    EXPECT_EQ(&e1, &e2);
}

TEST(ProgressEngineTest, RegisterAndUnregister) {
    auto& engine = futures::progress_engine::instance();
    auto id = engine.register_callback([] { return false; });
    engine.unregister_callback(id);
    // Should not crash
}

TEST(ProgressEngineTest, PollDrivesCallbacks) {
    auto& engine = futures::progress_engine::instance();
    int call_count = 0;
    auto id = engine.register_callback([&] {
        call_count++;
        return call_count < 3;  // Complete after 3 polls
    });

    while (call_count < 3) {
        engine.poll();
    }
    EXPECT_EQ(call_count, 3);
    // Callback should be auto-removed; poll again should not increment
    engine.poll();
    EXPECT_EQ(call_count, 3);

    // Just in case it wasn't auto-removed
    engine.unregister_callback(id);
}

// =============================================================================
// execute_algorithm Tests
// =============================================================================

TEST(ExecuteAlgorithmTest, SyncExecution) {
    auto r = futures::execute_algorithm<dtl::seq>([] { return 42; });
    static_assert(std::is_same_v<decltype(r), result<int>>);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ExecuteAlgorithmTest, SyncVoidExecution) {
    int counter = 0;
    auto r = futures::execute_algorithm<dtl::seq>([&] { counter = 1; });
    static_assert(std::is_same_v<decltype(r), result<void>>);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(counter, 1);
}

TEST(ExecuteAlgorithmTest, SyncExceptionHandled) {
    auto r = futures::execute_algorithm<dtl::seq>([]() -> int {
        throw std::runtime_error("boom");
    });
    EXPECT_TRUE(r.has_error());
}

}  // namespace dtl::test
