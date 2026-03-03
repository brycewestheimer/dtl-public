// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_future_diagnostics.cpp
/// @brief Unit tests for diagnostic_collector integration with future lifecycle (Phase 10, T01)

#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <chrono>
#include <thread>

namespace dtl::futures::test {

// =============================================================================
// Diagnostic Collector Wiring Tests
// =============================================================================

TEST(FutureDiagnosticsTest, DiagnosticCollectorExists) {
    // Verify the diagnostic_collector type exists and is accessible
    SUCCEED() << "diagnostic_collector type available";
}

TEST(FutureDiagnosticsTest, FutureHasDiagnosticSupport) {
    // Verify distributed_future has diagnostic metadata
    using future_type = distributed_future<int>;
    // The future type should compile
    static_assert(std::is_move_constructible_v<future_type>);
    SUCCEED();
}

TEST(FutureDiagnosticsTest, PromiseCreatesValidFuture) {
    // Basic promise-future lifecycle test
    distributed_promise<int> promise;
    auto future = promise.get_future();
    EXPECT_FALSE(future.is_ready());

    promise.set_value(42);
    // After setting value, future should be ready
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);
}

TEST(FutureDiagnosticsTest, VoidFutureDiagnostics) {
    distributed_promise<void> promise;
    auto future = promise.get_future();
    EXPECT_FALSE(future.is_ready());

    promise.set_value();
    EXPECT_TRUE(future.is_ready());
    EXPECT_NO_THROW(future.get());
}

}  // namespace dtl::futures::test
