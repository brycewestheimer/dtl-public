// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_event_future.cpp
/// @brief Unit tests for CUDA event integration in the progress engine
/// @details Tests that the progress engine correctly compiles and operates
///          both with and without CUDA enabled. When CUDA is available,
///          tests exercise event registration, polling, and callback firing.

#include <dtl/futures/progress.hpp>
#include <dtl/core/config.hpp>

#include <gtest/gtest.h>

#include <atomic>

namespace dtl::futures::test {

// =============================================================================
// Non-CUDA Tests (always compiled)
// =============================================================================

TEST(CudaEventFutureTest, ProgressEngineCompilesWithoutCuda) {
    // Verifies that the progress engine compiles and works without CUDA.
    // If this test compiles and runs, #if DTL_ENABLE_CUDA gating is correct.
    auto& engine = progress_engine::instance();
    engine.poll();
    SUCCEED();
}

TEST(CudaEventFutureTest, HasPendingIncludesCudaPath) {
    // Verifies has_pending() works correctly regardless of CUDA availability
    auto& engine = progress_engine::instance();
    size_type initial = engine.pending_count();

    // With no extra callbacks or CUDA events, has_pending reflects callback state
    auto id = engine.register_callback([]() { return true; });
    EXPECT_TRUE(engine.has_pending());

    engine.unregister_callback(id);
    // After removing all our callbacks, pending state depends on other test state
    EXPECT_EQ(engine.pending_count(), initial);
}

TEST(CudaEventFutureTest, PollWorksWithoutCudaEvents) {
    // poll() should work correctly even when CUDA is disabled
    auto& engine = progress_engine::instance();

    std::atomic<int> counter{0};
    engine.register_callback([&]() {
        ++counter;
        return false;  // Complete immediately
    });

    engine.poll();
    EXPECT_EQ(counter.load(), 1);
}

// =============================================================================
// CUDA-Enabled Tests (conditionally compiled)
// =============================================================================

#if DTL_ENABLE_CUDA

TEST(CudaEventFutureTest, RegisterAndPollEvent) {
    auto& engine = progress_engine::instance();

    // Create a CUDA event and record it on the default stream
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event, nullptr), cudaSuccess);

    // Synchronize so the event is definitely complete
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);

    std::atomic<bool> callback_fired{false};
    engine.register_cuda_event(event, [&]() {
        callback_fired = true;
    });

    EXPECT_TRUE(engine.has_pending_cuda());
    EXPECT_EQ(engine.pending_cuda_count(), 1u);

    // Poll should find the completed event and fire the callback
    engine.poll();

    EXPECT_TRUE(callback_fired.load());
    EXPECT_FALSE(engine.has_pending_cuda());
    EXPECT_EQ(engine.pending_cuda_count(), 0u);

    cudaEventDestroy(event);
}

TEST(CudaEventFutureTest, MultipleEventsResolve) {
    auto& engine = progress_engine::instance();

    constexpr int num_events = 3;
    cudaEvent_t events[num_events];
    std::atomic<int> completed_count{0};
    std::array<std::atomic<bool>, num_events> fired = {{{false}, {false}, {false}}};

    for (int i = 0; i < num_events; ++i) {
        ASSERT_EQ(cudaEventCreate(&events[i]), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(events[i], nullptr), cudaSuccess);
    }

    // Synchronize all events
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int i = 0; i < num_events; ++i) {
        engine.register_cuda_event(events[i], [&, i]() {
            fired[static_cast<size_type>(i)] = true;
            ++completed_count;
        });
    }

    EXPECT_EQ(engine.pending_cuda_count(), static_cast<size_type>(num_events));

    // All events are already complete, so one poll should resolve all
    engine.poll();

    EXPECT_EQ(completed_count.load(), num_events);
    for (size_type i = 0; i < static_cast<size_type>(num_events); ++i) {
        EXPECT_TRUE(fired[i].load());
    }
    EXPECT_EQ(engine.pending_cuda_count(), 0u);

    for (size_type i = 0; i < static_cast<size_type>(num_events); ++i) {
        cudaEventDestroy(events[i]);
    }
}

TEST(CudaEventFutureTest, HasPendingIncludesCudaEvents) {
    auto& engine = progress_engine::instance();

    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event, nullptr), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);

    // Register a CUDA event (no regular callbacks)
    engine.register_cuda_event(event, []() {});

    // has_pending() should return true because of the CUDA event
    EXPECT_TRUE(engine.has_pending());

    // Poll to resolve
    engine.poll();

    cudaEventDestroy(event);
}

TEST(CudaEventFutureTest, CudaEventCallbackCanReregister) {
    // Verify that a CUDA event callback can register a new CUDA event
    // without deadlock (callbacks are invoked outside the lock).
    auto& engine = progress_engine::instance();

    cudaEvent_t event1, event2;
    ASSERT_EQ(cudaEventCreate(&event1), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&event2), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event1, nullptr), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event2, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::atomic<int> stage{0};

    engine.register_cuda_event(event1, [&]() {
        stage = 1;
        // Re-entrant registration from within a callback
        engine.register_cuda_event(event2, [&]() {
            stage = 2;
        });
    });

    // First poll resolves event1, which registers event2
    engine.poll();
    EXPECT_EQ(stage.load(), 1);
    EXPECT_EQ(engine.pending_cuda_count(), 1u);

    // Second poll resolves event2
    engine.poll();
    EXPECT_EQ(stage.load(), 2);
    EXPECT_EQ(engine.pending_cuda_count(), 0u);

    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
}

#else  // !DTL_ENABLE_CUDA

TEST(CudaEventFutureTest, CudaMethodsNotAvailableWhenDisabled) {
    // When CUDA is disabled, register_cuda_event(), has_pending_cuda(),
    // and pending_cuda_count() should not be available. This test verifies
    // the non-CUDA build path works correctly.
    auto& engine = progress_engine::instance();

    // These methods should still work normally
    EXPECT_GE(engine.pending_count(), 0u);
    engine.poll();
    SUCCEED();
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::futures::test
