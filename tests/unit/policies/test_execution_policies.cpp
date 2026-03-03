// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_execution_policies.cpp
/// @brief Unit tests for execution policies
/// @details Tests seq, par, async, and related execution policies.

#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/core/concepts.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Sequential Policy Tests
// =============================================================================

TEST(SeqPolicyTest, ConceptSatisfaction) {
    static_assert(ExecutionPolicyType<seq>);
}

TEST(SeqPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename seq::policy_category, execution_policy_tag>);
}

TEST(SeqPolicyTest, ExecutionMode) {
    EXPECT_EQ(seq::mode(), execution_mode::synchronous);
}

TEST(SeqPolicyTest, BlockingBehavior) {
    EXPECT_TRUE(seq::is_blocking());
}

TEST(SeqPolicyTest, ParallelismProperties) {
    EXPECT_FALSE(seq::is_parallel());
    EXPECT_EQ(seq::parallelism(), parallelism_level::sequential);
}

TEST(SeqPolicyTest, VectorizationAndDeterminism) {
    EXPECT_TRUE(seq::allows_vectorization());
    EXPECT_TRUE(seq::is_deterministic());
}

TEST(SeqPolicyTest, ExecutionTraits) {
    static_assert(execution_traits<seq>::is_blocking == true);
    static_assert(execution_traits<seq>::is_parallel == false);
    static_assert(execution_traits<seq>::mode == execution_mode::synchronous);
    static_assert(execution_traits<seq>::parallelism == parallelism_level::sequential);
}

// =============================================================================
// Parallel Policy Tests
// =============================================================================

TEST(ParPolicyTest, ConceptSatisfaction) {
    static_assert(ExecutionPolicyType<par>);
}

TEST(ParPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename par::policy_category, execution_policy_tag>);
}

TEST(ParPolicyTest, ExecutionMode) {
    // Par is synchronous (blocking) but parallel
    EXPECT_EQ(par::mode(), execution_mode::synchronous);
}

TEST(ParPolicyTest, BlockingBehavior) {
    EXPECT_TRUE(par::is_blocking());
}

TEST(ParPolicyTest, ParallelismProperties) {
    EXPECT_TRUE(par::is_parallel());
    EXPECT_EQ(par::parallelism(), parallelism_level::parallel);
}

TEST(ParPolicyTest, VectorizationAndDeterminism) {
    EXPECT_TRUE(par::allows_vectorization());
    EXPECT_FALSE(par::is_deterministic());  // Order not guaranteed
}

TEST(ParPolicyTest, ThreadCount) {
    // Default par uses auto-detect (0)
    EXPECT_EQ(par::num_threads(), 0u);
}

TEST(ParPolicyTest, ExecutionTraits) {
    static_assert(execution_traits<par>::is_blocking == true);
    static_assert(execution_traits<par>::is_parallel == true);
    static_assert(execution_traits<par>::mode == execution_mode::synchronous);
    static_assert(execution_traits<par>::parallelism == parallelism_level::parallel);
}

// =============================================================================
// Parallel N Policy Tests
// =============================================================================

TEST(ParNPolicyTest, ConceptSatisfaction) {
    static_assert(ExecutionPolicyType<par_n<4>>);
    static_assert(ExecutionPolicyType<par_n<8>>);
}

TEST(ParNPolicyTest, SpecificThreadCount) {
    EXPECT_EQ(par_n<4>::num_threads(), 4u);
    EXPECT_EQ(par_n<8>::num_threads(), 8u);
    EXPECT_EQ(par_n<1>::num_threads(), 1u);
    EXPECT_EQ(par_n<0>::num_threads(), 0u);  // Auto
}

TEST(ParNPolicyTest, ThreadCountConstexpr) {
    static_assert(par_n<4>::thread_count == 4);
    static_assert(par_n<16>::thread_count == 16);
}

TEST(ParNPolicyTest, Properties) {
    EXPECT_TRUE(par_n<4>::is_blocking());
    EXPECT_TRUE(par_n<4>::is_parallel());
    EXPECT_EQ(par_n<4>::mode(), execution_mode::synchronous);
}

// =============================================================================
// Async Policy Tests
// =============================================================================

TEST(AsyncPolicyTest, ConceptSatisfaction) {
    static_assert(ExecutionPolicyType<async>);
}

TEST(AsyncPolicyTest, PolicyCategory) {
    static_assert(std::is_same_v<typename async::policy_category, execution_policy_tag>);
}

TEST(AsyncPolicyTest, ExecutionMode) {
    EXPECT_EQ(async::mode(), execution_mode::asynchronous);
}

TEST(AsyncPolicyTest, BlockingBehavior) {
    EXPECT_FALSE(async::is_blocking());
}

TEST(AsyncPolicyTest, ParallelismProperties) {
    EXPECT_TRUE(async::is_parallel());
    EXPECT_EQ(async::parallelism(), parallelism_level::parallel);
}

TEST(AsyncPolicyTest, AsyncSpecificProperties) {
    EXPECT_TRUE(async::requires_wait());
    EXPECT_TRUE(async::supports_continuations());
}

TEST(AsyncPolicyTest, ExecutionTraits) {
    static_assert(execution_traits<async>::is_blocking == false);
    static_assert(execution_traits<async>::is_parallel == true);
    static_assert(execution_traits<async>::mode == execution_mode::asynchronous);
    static_assert(execution_traits<async>::parallelism == parallelism_level::parallel);
}

// =============================================================================
// Execution Mode Enum Tests
// =============================================================================

TEST(ExecutionModeTest, EnumValues) {
    EXPECT_NE(execution_mode::synchronous, execution_mode::asynchronous);
    EXPECT_NE(execution_mode::synchronous, execution_mode::deferred);
    EXPECT_NE(execution_mode::asynchronous, execution_mode::deferred);
}

// =============================================================================
// Parallelism Level Enum Tests
// =============================================================================

TEST(ParallelismLevelTest, EnumValues) {
    EXPECT_NE(parallelism_level::sequential, parallelism_level::parallel);
    EXPECT_NE(parallelism_level::sequential, parallelism_level::distributed);
    EXPECT_NE(parallelism_level::parallel, parallelism_level::distributed);
    EXPECT_NE(parallelism_level::distributed, parallelism_level::heterogeneous);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(ExecutionConstexprTest, SeqConstexpr) {
    constexpr auto mode = seq::mode();
    constexpr bool blocking = seq::is_blocking();
    constexpr bool parallel = seq::is_parallel();

    static_assert(mode == execution_mode::synchronous);
    static_assert(blocking == true);
    static_assert(parallel == false);
}

TEST(ExecutionConstexprTest, ParConstexpr) {
    constexpr auto mode = par::mode();
    constexpr bool blocking = par::is_blocking();
    constexpr bool parallel = par::is_parallel();

    static_assert(mode == execution_mode::synchronous);
    static_assert(blocking == true);
    static_assert(parallel == true);
}

TEST(ExecutionConstexprTest, AsyncConstexpr) {
    constexpr auto mode = async::mode();
    constexpr bool blocking = async::is_blocking();
    constexpr bool parallel = async::is_parallel();

    static_assert(mode == execution_mode::asynchronous);
    static_assert(blocking == false);
    static_assert(parallel == true);
}

// =============================================================================
// Policy Distinctness Tests
// =============================================================================

TEST(PolicyDistinctnessTest, DifferentPolicies) {
    // Different policies should have different type characteristics
    static_assert(seq::is_parallel() != par::is_parallel());
    static_assert(seq::is_blocking() == par::is_blocking());
    static_assert(par::is_blocking() != async::is_blocking());
    static_assert(par::is_parallel() == async::is_parallel());
}

}  // namespace dtl::test
