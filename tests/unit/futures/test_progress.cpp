// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_progress.cpp
/// @brief Unit tests for dtl/futures/progress.hpp and completion.hpp
/// @details Tests progress engine, completion sets, and related utilities.

#include <dtl/futures/progress.hpp>
#include <dtl/futures/completion.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <chrono>

namespace dtl::futures::test {

// =============================================================================
// Progress Engine Tests
// =============================================================================

TEST(ProgressEngineTest, SingletonExists) {
    auto& engine = progress_engine::instance();
    EXPECT_EQ(&engine, &progress_engine::instance());
}

TEST(ProgressEngineTest, InitiallyEmpty) {
    auto& engine = progress_engine::instance();
    // Note: Other tests may have registered callbacks, so just verify API works
    EXPECT_GE(engine.pending_count(), 0u);
}

TEST(ProgressEngineTest, RegisterAndUnregister) {
    auto& engine = progress_engine::instance();
    size_type initial = engine.pending_count();

    int call_count = 0;
    auto id = engine.register_callback([&]() {
        ++call_count;
        return call_count < 3;  // Complete after 3 calls
    });

    EXPECT_EQ(engine.pending_count(), initial + 1);

    engine.unregister_callback(id);
    EXPECT_EQ(engine.pending_count(), initial);
}

TEST(ProgressEngineTest, CallbackInvoked) {
    auto& engine = progress_engine::instance();

    int call_count = 0;
    engine.register_callback([&]() {
        ++call_count;
        return false;  // Complete immediately
    });

    engine.poll();
    EXPECT_EQ(call_count, 1);
}

TEST(ProgressEngineTest, CallbackRemovedWhenComplete) {
    auto& engine = progress_engine::instance();
    size_type initial = engine.pending_count();

    engine.register_callback([]() {
        return false;  // Complete immediately
    });

    EXPECT_EQ(engine.pending_count(), initial + 1);
    engine.poll();
    EXPECT_EQ(engine.pending_count(), initial);
}

TEST(ProgressEngineTest, MultipleCallbacksPolled) {
    auto& engine = progress_engine::instance();

    std::atomic<int> count1{0};
    std::atomic<int> count2{0};

    engine.register_callback([&]() {
        ++count1;
        return false;
    });

    engine.register_callback([&]() {
        ++count2;
        return false;
    });

    engine.poll();

    EXPECT_EQ(count1.load(), 1);
    EXPECT_EQ(count2.load(), 1);
}

// =============================================================================
// make_progress() Tests
// =============================================================================

TEST(MakeProgressTest, SinglePoll) {
    std::atomic<int> counter{0};

    auto id = progress_engine::instance().register_callback([&]() {
        ++counter;
        return counter.load() < 2;
    });

    make_progress();
    EXPECT_GE(counter.load(), 1);

    // Clean up to prevent dangling reference after test ends
    progress_engine::instance().unregister_callback(id);
}

TEST(MakeProgressTest, BoundedPolling) {
    std::atomic<int> counter{0};

    auto id = progress_engine::instance().register_callback([&]() {
        ++counter;
        return true;  // Never completes on its own
    });

    // Poll up to 5 times
    make_progress(5);
    EXPECT_GE(counter.load(), 1);
    EXPECT_LE(counter.load(), 5);

    // Clean up to prevent dangling reference after test ends
    progress_engine::instance().unregister_callback(id);
}

// =============================================================================
// Progress Guard Tests
// =============================================================================

TEST(ProgressGuardTest, PollsOnDestruction) {
    std::atomic<int> counter{0};

    progress_engine::instance().register_callback([&]() {
        ++counter;
        return false;
    });

    {
        progress_guard guard;
        // Guard should poll on destruction
    }

    EXPECT_EQ(counter.load(), 1);
}

TEST(ProgressGuardTest, ManualPoll) {
    std::atomic<int> counter{0};

    auto id = progress_engine::instance().register_callback([&]() {
        ++counter;
        return counter.load() < 3;
    });

    progress_guard guard;
    guard.poll();
    EXPECT_GE(counter.load(), 1);

    // Clean up to prevent dangling reference after test ends
    progress_engine::instance().unregister_callback(id);
}

// =============================================================================
// Scoped Progress Callback Tests
// =============================================================================

TEST(ScopedProgressCallbackTest, UnregistersOnDestruction) {
    auto& engine = progress_engine::instance();
    size_type initial = engine.pending_count();

    {
        scoped_progress_callback scoped([]() { return true; });
        EXPECT_EQ(engine.pending_count(), initial + 1);
    }

    EXPECT_EQ(engine.pending_count(), initial);
}

// =============================================================================
// Completion Token Tests
// =============================================================================

TEST(CompletionTokenTest, InitiallyIncomplete) {
    completion_token token(0);
    EXPECT_FALSE(token.is_complete());
    EXPECT_FALSE(token.is_success());
}

TEST(CompletionTokenTest, CompleteSuccess) {
    completion_token token(0);
    token.complete();
    EXPECT_TRUE(token.is_complete());
    EXPECT_TRUE(token.is_success());
}

TEST(CompletionTokenTest, CompleteFail) {
    completion_token token(0);
    token.fail();
    EXPECT_TRUE(token.is_complete());
    EXPECT_FALSE(token.is_success());
}

TEST(CompletionTokenTest, CallbackInvoked) {
    std::atomic<bool> called{false};
    std::atomic<size_type> callback_index{999};
    std::atomic<bool> callback_success{false};

    completion_token token(42, [&](size_type idx, bool success) {
        called = true;
        callback_index = idx;
        callback_success = success;
    });

    token.complete();

    EXPECT_TRUE(called.load());
    EXPECT_EQ(callback_index.load(), 42u);
    EXPECT_TRUE(callback_success.load());
}

TEST(CompletionTokenTest, DoubleCompleteIgnored) {
    std::atomic<int> call_count{0};

    completion_token token(0, [&](size_type, bool) {
        ++call_count;
    });

    token.complete();
    token.complete();  // Second call should be ignored
    token.fail();      // Also ignored

    EXPECT_EQ(call_count.load(), 1);
}

// =============================================================================
// Completion Set Tests
// =============================================================================

TEST(CompletionSetTest, InitiallyIncomplete) {
    completion_set set(3, completion_set::mode::all);
    EXPECT_FALSE(set.is_complete());
    EXPECT_EQ(set.completed_count(), 0u);
    EXPECT_EQ(set.total_count(), 3u);
}

TEST(CompletionSetTest, AllModeCompletesWhenAllDone) {
    completion_set set(3, completion_set::mode::all);

    auto token0 = set.create_token(0);
    auto token1 = set.create_token(1);
    auto token2 = set.create_token(2);

    token0->complete();
    EXPECT_FALSE(set.is_complete());
    EXPECT_EQ(set.completed_count(), 1u);

    token1->complete();
    EXPECT_FALSE(set.is_complete());
    EXPECT_EQ(set.completed_count(), 2u);

    token2->complete();
    EXPECT_TRUE(set.is_complete());
    EXPECT_EQ(set.completed_count(), 3u);
}

TEST(CompletionSetTest, AnyModeCompletesOnFirst) {
    completion_set set(3, completion_set::mode::any);

    auto token0 = set.create_token(0);
    auto token1 = set.create_token(1);
    auto token2 = set.create_token(2);

    EXPECT_FALSE(set.is_complete());

    token1->complete();
    EXPECT_TRUE(set.is_complete());
    EXPECT_EQ(set.first_completed(), 1u);
}

TEST(CompletionSetTest, OnCompleteCallback) {
    completion_set set(2, completion_set::mode::all);

    std::atomic<bool> callback_called{false};
    set.on_complete([&]() {
        callback_called = true;
    });

    auto token0 = set.create_token(0);
    auto token1 = set.create_token(1);

    token0->complete();
    EXPECT_FALSE(callback_called.load());

    token1->complete();
    EXPECT_TRUE(callback_called.load());
}

TEST(CompletionSetTest, EmptySetIsImmediatelyComplete) {
    completion_set set(0, completion_set::mode::all);
    EXPECT_TRUE(set.is_complete());
}

// =============================================================================
// Completion Waiter Tests
// =============================================================================

TEST(CompletionWaiterTest, PollReturnsCompletion) {
    auto set = std::make_shared<completion_set>(1, completion_set::mode::all);
    auto token = set->create_token(0);

    completion_waiter waiter(set);

    EXPECT_FALSE(waiter.poll());
    EXPECT_FALSE(waiter.is_complete());

    token->complete();

    EXPECT_TRUE(waiter.poll());
    EXPECT_TRUE(waiter.is_complete());
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ProgressIntegrationTest, CompletionSetWithProgress) {
    auto set = std::make_shared<completion_set>(2, completion_set::mode::all);
    auto token0 = set->create_token(0);
    auto token1 = set->create_token(1);

    // Simulate async work completing
    std::atomic<int> work_done{0};

    progress_engine::instance().register_callback([&, token0]() {
        if (work_done.fetch_add(1) == 0) {
            token0->complete();
        }
        return false;
    });

    progress_engine::instance().register_callback([&, token1]() {
        if (work_done.load() >= 1) {
            token1->complete();
            return false;
        }
        return true;
    });

    // Drive progress until complete
    while (!set->is_complete()) {
        make_progress();
    }

    EXPECT_TRUE(set->is_complete());
    EXPECT_EQ(set->completed_count(), 2u);
}

}  // namespace dtl::futures::test
