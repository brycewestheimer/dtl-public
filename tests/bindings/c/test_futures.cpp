// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_futures.cpp
 * @brief Unit tests for DTL C bindings futures operations
 * @since 0.1.0
 *
 * @warning Futures API is experimental. The progress engine has known
 *          stability issues (see KNOWN_ISSUES.md). Tests involving
 *          when_all/when_any may hang.
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>

// ============================================================================
// Future Lifecycle Tests
// ============================================================================

TEST(CBindingsFutures, CreateSucceeds) {
    dtl_future_t fut = nullptr;
    dtl_status status = dtl_future_create(&fut);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(fut, nullptr);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, CreateWithNullFails) {
    dtl_status status = dtl_future_create(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsFutures, DestroyNullIsSafe) {
    dtl_future_destroy(nullptr);
}

// ============================================================================
// Future Test/Wait Tests
// ============================================================================

TEST(CBindingsFutures, TestIncompleteFuture) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    int completed = -1;
    dtl_status status = dtl_future_test(fut, &completed);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(completed, 0);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, TestWithNullFutureFails) {
    int completed = 0;
    dtl_status status = dtl_future_test(nullptr, &completed);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsFutures, TestWithNullOutputFails) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    dtl_status status = dtl_future_test(fut, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}

// ============================================================================
// Future Value Tests
// ============================================================================

TEST(CBindingsFutures, SetAndGetValue) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    double value = 42.0;
    dtl_status status = dtl_future_set(fut, &value, sizeof(value));
    EXPECT_EQ(status, DTL_SUCCESS);

    // Should now be complete
    int completed = 0;
    dtl_future_test(fut, &completed);
    EXPECT_EQ(completed, 1);

    double result = 0.0;
    status = dtl_future_get(fut, &result, sizeof(result));
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(result, 42.0);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, SetTwiceFails) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    double value = 1.0;
    dtl_future_set(fut, &value, sizeof(value));

    double value2 = 2.0;
    dtl_status status = dtl_future_set(fut, &value2, sizeof(value2));
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, GetBeforeSetFails) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    double result = 0.0;
    dtl_status status = dtl_future_get(fut, &result, sizeof(result));
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, WaitAfterSetSucceeds) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    int value = 123;
    dtl_future_set(fut, &value, sizeof(value));

    dtl_status status = dtl_future_wait(fut);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, SetZeroSizeSignalOnly) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    dtl_status status = dtl_future_set(fut, nullptr, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    int completed = 0;
    dtl_future_test(fut, &completed);
    EXPECT_EQ(completed, 1);

    dtl_future_destroy(fut);
}

// ============================================================================
// When_all / When_any Tests
// (Skipped by default due to known progress engine stability issues)
// ============================================================================

TEST(CBindingsFutures, WhenAllWithNullFuturesFails) {
    dtl_future_t result = nullptr;
    dtl_status status = dtl_when_all(nullptr, 1, &result);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsFutures, WhenAllWithZeroCountFails) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    dtl_future_t result = nullptr;
    dtl_status status = dtl_when_all(&fut, 0, &result);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}

TEST(CBindingsFutures, WhenAnyWithNullFuturesFails) {
    dtl_future_t result = nullptr;
    dtl_size_t index = 0;
    dtl_status status = dtl_when_any(nullptr, 1, &result, &index);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsFutures, WhenAnyWithZeroCountFails) {
    dtl_future_t fut = nullptr;
    dtl_future_create(&fut);

    dtl_future_t result = nullptr;
    dtl_size_t index = 0;
    dtl_status status = dtl_when_any(&fut, 0, &result, &index);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_future_destroy(fut);
}
