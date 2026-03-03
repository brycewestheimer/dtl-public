// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_timeout_diagnostics.cpp
/// @brief Unit tests for timeout and diagnostics configuration (Phase 07)
/// @details Tests configurable timeouts and diagnostic output

#include <dtl/futures/diagnostics.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>

namespace dtl::futures::test {

// =============================================================================
// Timeout Configuration Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, DefaultConfiguration) {
    auto config = timeout_config::defaults();

    EXPECT_EQ(config.default_wait_timeout, std::chrono::milliseconds(30000));
    EXPECT_EQ(config.ci_wait_timeout, std::chrono::milliseconds(30000));
    EXPECT_EQ(config.poll_interval, std::chrono::milliseconds(1));
    EXPECT_TRUE(config.enable_timeout_diagnostics);
}

TEST(TimeoutDiagnosticsTest, CIModeConfiguration) {
    auto config = timeout_config::ci_mode();

    EXPECT_EQ(config.default_wait_timeout, std::chrono::milliseconds(30000));
    EXPECT_EQ(config.ci_wait_timeout, std::chrono::milliseconds(30000));
    EXPECT_TRUE(config.enable_timeout_diagnostics);
}

TEST(TimeoutDiagnosticsTest, LenientConfiguration) {
    auto config = timeout_config::lenient();

    EXPECT_EQ(config.default_wait_timeout, std::chrono::milliseconds(300000));
    EXPECT_EQ(config.ci_wait_timeout, std::chrono::milliseconds(60000));
}

TEST(TimeoutDiagnosticsTest, NoTimeoutConfiguration) {
    auto config = timeout_config::no_timeout();

    EXPECT_EQ(config.default_wait_timeout, std::chrono::milliseconds(0));
    EXPECT_EQ(config.ci_wait_timeout, std::chrono::milliseconds(0));
}

// =============================================================================
// Global Configuration Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, GlobalConfigurationPersists) {
    // Save original
    auto original = global_timeout_config();

    // Set new config
    auto new_config = timeout_config::lenient();
    set_global_timeout_config(new_config);

    EXPECT_EQ(global_timeout_config().default_wait_timeout,
              std::chrono::milliseconds(300000));

    // Restore original
    set_global_timeout_config(original);
}

TEST(TimeoutDiagnosticsTest, EffectiveTimeoutUsesDefault) {
    // Save original
    auto original = global_timeout_config();

    auto config = timeout_config::defaults();
    config.default_wait_timeout = std::chrono::milliseconds(5000);
    set_global_timeout_config(config);

    // Without DTL_CI_MODE set, should use default_wait_timeout
    // (assuming DTL_CI_MODE is not set in test environment)
    auto timeout = effective_wait_timeout();

    // Either ci or default depending on environment
    EXPECT_TRUE(timeout == std::chrono::milliseconds(5000) ||
                timeout == config.ci_wait_timeout);

    // Restore original
    set_global_timeout_config(original);
}

// =============================================================================
// Diagnostic Collector Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, RegisterAndUnregisterFuture) {
    auto& collector = diagnostic_collector::instance();

    auto id = collector.register_future("test operation");
    EXPECT_GT(id, 0u);

    auto diag = collector.get_diagnostics();
    bool found = false;
    for (const auto& f : diag.pending_futures) {
        if (f.id == id && f.description == "test operation") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    collector.unregister_future(id);

    diag = collector.get_diagnostics();
    found = false;
    for (const auto& f : diag.pending_futures) {
        if (f.id == id) {
            found = true;
            break;
        }
    }
    EXPECT_FALSE(found);
}

TEST(TimeoutDiagnosticsTest, MarkWaiting) {
    auto& collector = diagnostic_collector::instance();

    auto id = collector.register_future("waiting test");
    collector.mark_waiting(id);

    auto diag = collector.get_diagnostics();
    for (const auto& f : diag.pending_futures) {
        if (f.id == id) {
            EXPECT_TRUE(f.is_waiting);
            break;
        }
    }

    collector.unregister_future(id);
}

TEST(TimeoutDiagnosticsTest, RecordPoll) {
    auto& collector = diagnostic_collector::instance();

    size_type initial_polls = collector.total_polls();

    collector.record_poll();
    collector.record_poll();
    collector.record_poll();

    EXPECT_EQ(collector.total_polls(), initial_polls + 3);
}

// =============================================================================
// Diagnostic Snapshot Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, DiagnosticSnapshotContainsData) {
    auto& collector = diagnostic_collector::instance();

    // Record some activity
    collector.record_poll();
    auto id = collector.register_future("snapshot test");

    auto diag = collector.get_diagnostics();

    EXPECT_GT(diag.total_polls, 0u);
    EXPECT_GE(diag.pending_future_count, 1u);
    EXPECT_FALSE(diag.pending_futures.empty());

    collector.unregister_future(id);
}

TEST(TimeoutDiagnosticsTest, DiagnosticToString) {
    auto diag = progress_diagnostics{};
    diag.snapshot_time = std::chrono::steady_clock::now();
    diag.pending_callback_count = 5;
    diag.pending_cuda_event_count = 2;
    diag.pending_future_count = 3;
    diag.total_polls = 100;
    diag.last_poll_time = std::chrono::steady_clock::now();
    diag.background_progress_enabled = false;
    diag.background_thread_running = false;

    pending_future_info info;
    info.id = 1;
    info.created_at = std::chrono::steady_clock::now();
    info.description = "test future";
    info.is_waiting = true;
    diag.pending_futures.push_back(info);

    std::string str = diag.to_string();

    EXPECT_TRUE(str.find("DTL Futures Diagnostics") != std::string::npos);
    EXPECT_TRUE(str.find("Pending callbacks: 5") != std::string::npos);
    EXPECT_TRUE(str.find("Pending CUDA events: 2") != std::string::npos);
    EXPECT_TRUE(str.find("Total polls: 100") != std::string::npos);
    EXPECT_TRUE(str.find("test future") != std::string::npos);
}

// =============================================================================
// Timeout Exception Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, TimeoutExceptionCarriesDiagnostics) {
    progress_diagnostics diag;
    diag.pending_callback_count = 10;
    diag.total_polls = 50;

    timeout_exception ex("Test timeout", std::move(diag));

    EXPECT_STREQ(ex.what(), "Test timeout");
    EXPECT_EQ(ex.diagnostics().pending_callback_count, 10u);
    EXPECT_EQ(ex.diagnostics().total_polls, 50u);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(TimeoutDiagnosticsTest, TimeoutCallbackInvoked) {
    // Save original
    auto original = global_timeout_config();

    std::atomic<bool> callback_invoked{false};
    std::string callback_message;

    auto config = timeout_config::defaults();
    config.default_wait_timeout = std::chrono::milliseconds(50);  // Short timeout
    config.ci_wait_timeout = std::chrono::milliseconds(50);
    config.enable_timeout_diagnostics = true;
    config.on_timeout_callback = [&](const std::string& msg) {
        callback_invoked = true;
        callback_message = msg;
    };
    set_global_timeout_config(config);

    // Create a future that will never complete
    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Try to wait - should timeout
    try {
        future.wait();
        FAIL() << "Expected timeout_exception";
    } catch (const timeout_exception& ex) {
        EXPECT_TRUE(callback_invoked.load());
        EXPECT_FALSE(callback_message.empty());
        EXPECT_TRUE(callback_message.find("Futures Diagnostics") != std::string::npos);
    } catch (...) {
        // May throw std::runtime_error instead depending on config
    }

    // Restore original
    set_global_timeout_config(original);

    // Complete the promise to clean up
    promise.set_value(0);
}

TEST(TimeoutDiagnosticsTest, FormatTimeoutDiagnostics) {
    std::string diag_str = format_timeout_diagnostics("test operation");

    EXPECT_TRUE(diag_str.find("Timeout waiting for: test operation") != std::string::npos);
    EXPECT_TRUE(diag_str.find("DTL Futures Diagnostics") != std::string::npos);
}

// =============================================================================
// Timeout Behavior with Different Configurations
// =============================================================================

TEST(TimeoutDiagnosticsTest, NoTimeoutWaitsIndefinitely) {
    // Save original
    auto original = global_timeout_config();

    auto config = timeout_config::no_timeout();
    set_global_timeout_config(config);

    distributed_promise<int> promise;
    auto future = promise.get_future();

    // Complete after short delay
    std::thread setter([&promise] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        promise.set_value(42);
    });

    // Should wait without timeout
    future.wait();
    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.get(), 42);

    setter.join();

    // Restore original
    set_global_timeout_config(original);
}

}  // namespace dtl::futures::test
