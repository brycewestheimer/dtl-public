// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_phase05_failure_handler.cpp
/// @brief Phase 05 tests: failure_handler_manager thread safety
/// @details Tests for T03 — concurrent handler registration and invocation.

#include <dtl/error/failure_handler.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace dtl::test {

class FailureHandlerThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        failure_handler_manager::instance().clear_handlers();
    }

    void TearDown() override {
        failure_handler_manager::instance().clear_handlers();
    }

    failure_context make_context(failure_category cat = failure_category::non_recoverable) {
        return failure_context{
            .failure_status = status{status_code::internal_error},
            .category = cat,
            .local_rank = 0,
            .failed_rank = no_rank,
            .operation = "test_op",
            .is_collective = false
        };
    }
};

TEST_F(FailureHandlerThreadSafetyTest, ConcurrentRegistration) {
    constexpr int num_threads = 8;
    constexpr int handlers_per_thread = 100;
    std::atomic<int> total_registered{0};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < handlers_per_thread; ++i) {
                failure_handler_manager::instance().register_handler(
                    [](const failure_context&) {
                        return failure_result{recovery_action::none, 0, ""};
                    }
                );
                total_registered.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(total_registered.load(), num_threads * handlers_per_thread);
}

TEST_F(FailureHandlerThreadSafetyTest, ConcurrentRegistrationAndInvocation) {
    std::atomic<int> invoke_count{0};
    std::atomic<bool> stop{false};

    // Register a baseline handler
    failure_handler_manager::instance().register_handler(
        [&](const failure_context&) {
            invoke_count.fetch_add(1);
            return failure_result{recovery_action::skip, 0, "test"};
        }
    );

    // Thread that repeatedly invokes handle_failure
    std::thread invoker([&]() {
        auto ctx = make_context();
        while (!stop.load()) {
            auto result = failure_handler_manager::instance().handle_failure(ctx);
            // Should get skip from our handler or abort from default
            EXPECT_TRUE(result.action == recovery_action::skip ||
                         result.action == recovery_action::abort);
        }
    });

    // Threads that register handlers concurrently
    std::vector<std::thread> registrars;
    for (int t = 0; t < 4; ++t) {
        registrars.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                failure_handler_manager::instance().register_handler(
                    [](const failure_context&) {
                        return failure_result{recovery_action::none, 0, ""};
                    }
                );
                std::this_thread::yield();
            }
        });
    }

    for (auto& t : registrars) {
        t.join();
    }

    stop.store(true);
    invoker.join();

    EXPECT_GT(invoke_count.load(), 0);
}

TEST_F(FailureHandlerThreadSafetyTest, HandlerInvokedOutsideLock) {
    // Test that a handler can register another handler without deadlock
    bool outer_called = false;
    bool inner_registered = false;

    failure_handler_manager::instance().register_handler(
        [&](const failure_context&) {
            outer_called = true;
            // This should NOT deadlock — handler invocation doesn't hold the lock
            failure_handler_manager::instance().register_handler(
                [](const failure_context&) {
                    return failure_result{recovery_action::none, 0, ""};
                }
            );
            inner_registered = true;
            return failure_result{recovery_action::skip, 0, "handled"};
        }
    );

    auto result = failure_handler_manager::instance().handle_failure(make_context());

    EXPECT_TRUE(outer_called);
    EXPECT_TRUE(inner_registered);
    EXPECT_EQ(result.action, recovery_action::skip);
}

TEST_F(FailureHandlerThreadSafetyTest, ConcurrentUnregistration) {
    std::vector<size_type> handles;
    for (int i = 0; i < 100; ++i) {
        handles.push_back(
            failure_handler_manager::instance().register_handler(
                [](const failure_context&) {
                    return failure_result{recovery_action::none, 0, ""};
                }
            )
        );
    }

    // Unregister from multiple threads
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = static_cast<size_t>(t); i < 100u; i += 4u) {
                failure_handler_manager::instance().unregister_handler(handles[i]);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should not crash; default handling should apply
    auto result = failure_handler_manager::instance().handle_failure(make_context());
    EXPECT_EQ(result.action, recovery_action::abort);
}

TEST_F(FailureHandlerThreadSafetyTest, DefaultTransientRetry) {
    auto ctx = make_context(failure_category::transient);
    auto result = failure_handler_manager::instance().handle_failure(ctx);
    EXPECT_EQ(result.action, recovery_action::retry);
    EXPECT_EQ(result.retry_count, 3);
}

TEST_F(FailureHandlerThreadSafetyTest, DefaultNonRecoverableAbort) {
    auto ctx = make_context(failure_category::non_recoverable);
    auto result = failure_handler_manager::instance().handle_failure(ctx);
    EXPECT_EQ(result.action, recovery_action::abort);
}

}  // namespace dtl::test
