// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_error_policies.cpp
/// @brief Unit tests for error policies
/// @details Tests expected, throwing, terminating, and callback policies.

#include <dtl/policies/error/error_policy.hpp>
#include <dtl/policies/error/expected.hpp>
#include <dtl/policies/error/throwing.hpp>
#include <dtl/policies/error/terminating.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>
#include <dtl/core/concepts.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <stdexcept>

namespace dtl::test {

// =============================================================================
// Expected Policy Tests
// =============================================================================

TEST(ExpectedPolicyTest, ConceptSatisfaction) {
    static_assert(ErrorPolicyType<expected_policy>);
}

TEST(ExpectedPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename expected_policy::policy_category, error_policy_tag>);
}

TEST(ExpectedPolicyTest, Strategy) {
    EXPECT_EQ(expected_policy::strategy(), error_strategy::return_result);
}

TEST(ExpectedPolicyTest, Characteristics) {
    EXPECT_FALSE(expected_policy::uses_exceptions());
    EXPECT_FALSE(expected_policy::can_ignore_errors());
    EXPECT_FALSE(expected_policy::auto_propagates());
}

TEST(ExpectedPolicyTest, HandleSuccess) {
    auto res = expected_policy::handle_success<int>(42);
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 42);
}

TEST(ExpectedPolicyTest, HandleError) {
    status s = make_error(status_code::invalid_argument, "test error");
    auto res = expected_policy::handle_error<int>(s);
    EXPECT_FALSE(res.has_value());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
}

TEST(ExpectedPolicyTest, ErrorPolicyTraits) {
    static_assert(error_policy_traits<expected_policy>::strategy == error_strategy::return_result);
    static_assert(error_policy_traits<expected_policy>::uses_exceptions == false);
    static_assert(error_policy_traits<expected_policy>::can_ignore == false);
}

TEST(ExpectedPolicyTest, ReturnType) {
    using return_t = typename error_return_type<expected_policy, int, status>::type;
    static_assert(std::is_same_v<return_t, result<int>>);
}

// =============================================================================
// Throwing Policy Tests
// =============================================================================

TEST(ThrowingPolicyTest, ConceptSatisfaction) {
    static_assert(ErrorPolicyType<throwing_policy>);
}

TEST(ThrowingPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename throwing_policy::policy_category, error_policy_tag>);
}

TEST(ThrowingPolicyTest, Strategy) {
    EXPECT_EQ(throwing_policy::strategy(), error_strategy::throw_exception);
}

TEST(ThrowingPolicyTest, Characteristics) {
    EXPECT_TRUE(throwing_policy::uses_exceptions());
    EXPECT_FALSE(throwing_policy::can_ignore_errors());
    EXPECT_TRUE(throwing_policy::auto_propagates());
}

TEST(ThrowingPolicyTest, HandleSuccessNoThrow) {
    int result = 0;
    EXPECT_NO_THROW({
        result = throwing_policy::handle_success<int>(42);
    });
    EXPECT_EQ(result, 42);
}

TEST(ThrowingPolicyTest, HandleErrorThrows) {
    status s = make_error(status_code::invalid_argument, "test error");
    EXPECT_THROW({
        throwing_policy::handle_error<int>(s);
    }, std::runtime_error);
}

TEST(ThrowingPolicyTest, ErrorPolicyTraits) {
    static_assert(error_policy_traits<throwing_policy>::strategy == error_strategy::throw_exception);
    static_assert(error_policy_traits<throwing_policy>::uses_exceptions == true);
    static_assert(error_policy_traits<throwing_policy>::can_ignore == false);
}

TEST(ThrowingPolicyTest, ReturnType) {
    // Throwing returns T directly, not result<T>
    using return_t = typename error_return_type<throwing_policy, int, status>::type;
    static_assert(std::is_same_v<return_t, int>);
}

// =============================================================================
// Terminating Policy Tests
// =============================================================================

TEST(TerminatingPolicyTest, ConceptSatisfaction) {
    static_assert(ErrorPolicyType<terminating_policy>);
}

TEST(TerminatingPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename terminating_policy::policy_category, error_policy_tag>);
}

TEST(TerminatingPolicyTest, Strategy) {
    EXPECT_EQ(terminating_policy::strategy(), error_strategy::terminate);
}

TEST(TerminatingPolicyTest, Characteristics) {
    EXPECT_FALSE(terminating_policy::uses_exceptions());
    EXPECT_FALSE(terminating_policy::can_ignore_errors());
    EXPECT_FALSE(terminating_policy::auto_propagates());
}

TEST(TerminatingPolicyTest, HandleSuccessNoTerminate) {
    int result = 0;
    EXPECT_NO_FATAL_FAILURE({
        result = terminating_policy::handle_success<int>(42);
    });
    EXPECT_EQ(result, 42);
}

TEST(TerminatingPolicyTest, ErrorPolicyTraits) {
    static_assert(error_policy_traits<terminating_policy>::strategy == error_strategy::terminate);
    static_assert(error_policy_traits<terminating_policy>::uses_exceptions == false);
    static_assert(error_policy_traits<terminating_policy>::can_ignore == false);
}

// =============================================================================
// Error Strategy Enum Tests
// =============================================================================

TEST(ErrorStrategyTest, EnumValues) {
    EXPECT_NE(error_strategy::return_result, error_strategy::throw_exception);
    EXPECT_NE(error_strategy::return_result, error_strategy::terminate);
    EXPECT_NE(error_strategy::return_result, error_strategy::callback);
    EXPECT_NE(error_strategy::return_result, error_strategy::ignore);
    EXPECT_NE(error_strategy::throw_exception, error_strategy::terminate);
    EXPECT_NE(error_strategy::throw_exception, error_strategy::callback);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(ErrorPolicyConstexprTest, ExpectedConstexpr) {
    constexpr auto strat = expected_policy::strategy();
    constexpr bool throws = expected_policy::uses_exceptions();
    constexpr bool ignore = expected_policy::can_ignore_errors();

    static_assert(strat == error_strategy::return_result);
    static_assert(throws == false);
    static_assert(ignore == false);
}

TEST(ErrorPolicyConstexprTest, ThrowingConstexpr) {
    constexpr auto strat = throwing_policy::strategy();
    constexpr bool throws = throwing_policy::uses_exceptions();
    constexpr bool propagates = throwing_policy::auto_propagates();

    static_assert(strat == error_strategy::throw_exception);
    static_assert(throws == true);
    static_assert(propagates == true);
}

// =============================================================================
// Policy Comparison Tests
// =============================================================================

TEST(ErrorPolicyComparisonTest, ExceptionUsage) {
    EXPECT_TRUE(throwing_policy::uses_exceptions());
    EXPECT_FALSE(expected_policy::uses_exceptions());
    EXPECT_FALSE(terminating_policy::uses_exceptions());
}

TEST(ErrorPolicyComparisonTest, Propagation) {
    EXPECT_TRUE(throwing_policy::auto_propagates());
    EXPECT_FALSE(expected_policy::auto_propagates());
    EXPECT_FALSE(terminating_policy::auto_propagates());
}

TEST(ErrorPolicyComparisonTest, StrategiesDistinct) {
    EXPECT_EQ(expected_policy::strategy(), error_strategy::return_result);
    EXPECT_EQ(throwing_policy::strategy(), error_strategy::throw_exception);
    EXPECT_EQ(terminating_policy::strategy(), error_strategy::terminate);
}

}  // namespace dtl::test
