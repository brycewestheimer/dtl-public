// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_consistency_policies.cpp
/// @brief Unit tests for consistency policies
/// @details Tests bulk_synchronous, sequential_consistent, release_acquire, and relaxed.

#include <dtl/policies/consistency/consistency_policy.hpp>
#include <dtl/policies/consistency/bulk_synchronous.hpp>
#include <dtl/policies/consistency/sequential_consistent.hpp>
#include <dtl/policies/consistency/release_acquire.hpp>
#include <dtl/policies/consistency/relaxed.hpp>
#include <dtl/core/concepts.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Bulk Synchronous Policy Tests
// =============================================================================

TEST(BulkSynchronousTest, ConceptSatisfaction) {
    static_assert(ConsistencyPolicyType<bulk_synchronous>);
}

TEST(BulkSynchronousTest, PolicyCategory) {
    static_assert(std::is_same_v<typename bulk_synchronous::policy_category, consistency_policy_tag>);
}

TEST(BulkSynchronousTest, MemoryOrdering) {
    EXPECT_EQ(bulk_synchronous::ordering(), memory_ordering::sequential);
}

TEST(BulkSynchronousTest, BarrierRequirements) {
    EXPECT_TRUE(bulk_synchronous::requires_barrier());
    EXPECT_TRUE(bulk_synchronous::needs_collective_sync());
}

TEST(BulkSynchronousTest, OverlapBehavior) {
    EXPECT_TRUE(bulk_synchronous::allows_overlap());
    EXPECT_TRUE(bulk_synchronous::buffers_writes());
}

TEST(BulkSynchronousTest, DefaultSync) {
    EXPECT_EQ(bulk_synchronous::default_sync(), sync_point::barrier);
}

TEST(BulkSynchronousTest, ConsistencyTraits) {
    static_assert(consistency_traits<bulk_synchronous>::requires_barrier == true);
    static_assert(consistency_traits<bulk_synchronous>::allows_overlap == true);
    static_assert(consistency_traits<bulk_synchronous>::default_ordering == memory_ordering::sequential);
}

// =============================================================================
// Sequential Consistent Policy Tests
// =============================================================================

TEST(SequentialConsistentTest, ConceptSatisfaction) {
    static_assert(ConsistencyPolicyType<sequential_consistent>);
}

TEST(SequentialConsistentTest, PolicyCategory) {
    static_assert(std::is_same_v<typename sequential_consistent::policy_category, consistency_policy_tag>);
}

TEST(SequentialConsistentTest, MemoryOrdering) {
    EXPECT_EQ(sequential_consistent::ordering(), memory_ordering::sequential);
}

TEST(SequentialConsistentTest, StrongestGuarantees) {
    // Sequential consistency typically requires barriers
    EXPECT_TRUE(sequential_consistent::requires_barrier());
    // All operations are ordered
    EXPECT_FALSE(sequential_consistent::allows_overlap());
}

// =============================================================================
// Release-Acquire Policy Tests
// =============================================================================

TEST(ReleaseAcquireTest, ConceptSatisfaction) {
    static_assert(ConsistencyPolicyType<release_acquire>);
}

TEST(ReleaseAcquireTest, PolicyCategory) {
    static_assert(std::is_same_v<typename release_acquire::policy_category, consistency_policy_tag>);
}

TEST(ReleaseAcquireTest, MemoryOrdering) {
    EXPECT_EQ(release_acquire::ordering(), memory_ordering::acquire_release);
}

TEST(ReleaseAcquireTest, Characteristics) {
    // Release-acquire typically uses fences, not full barriers
    EXPECT_FALSE(release_acquire::requires_barrier());
}

// =============================================================================
// Relaxed Policy Tests
// =============================================================================

TEST(RelaxedTest, ConceptSatisfaction) {
    static_assert(ConsistencyPolicyType<relaxed>);
}

TEST(RelaxedTest, PolicyCategory) {
    static_assert(std::is_same_v<typename relaxed::policy_category, consistency_policy_tag>);
}

TEST(RelaxedTest, MemoryOrdering) {
    EXPECT_EQ(relaxed::ordering(), memory_ordering::relaxed);
}

TEST(RelaxedTest, MinimalGuarantees) {
    // Relaxed provides minimal ordering
    EXPECT_FALSE(relaxed::requires_barrier());
    EXPECT_TRUE(relaxed::allows_overlap());
}

// =============================================================================
// Sync Point Enum Tests
// =============================================================================

TEST(SyncPointTest, EnumValues) {
    EXPECT_NE(sync_point::none, sync_point::barrier);
    EXPECT_NE(sync_point::none, sync_point::fence);
    EXPECT_NE(sync_point::none, sync_point::epoch);
    EXPECT_NE(sync_point::barrier, sync_point::fence);
    EXPECT_NE(sync_point::barrier, sync_point::epoch);
    EXPECT_NE(sync_point::fence, sync_point::epoch);
}

// =============================================================================
// Memory Ordering Enum Tests
// =============================================================================

TEST(MemoryOrderingTest, EnumValues) {
    EXPECT_NE(memory_ordering::relaxed, memory_ordering::acquire);
    EXPECT_NE(memory_ordering::relaxed, memory_ordering::release);
    EXPECT_NE(memory_ordering::relaxed, memory_ordering::acquire_release);
    EXPECT_NE(memory_ordering::relaxed, memory_ordering::sequential);
    EXPECT_NE(memory_ordering::acquire, memory_ordering::release);
    EXPECT_NE(memory_ordering::acquire, memory_ordering::sequential);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(ConsistencyConstexprTest, BulkSynchronousConstexpr) {
    constexpr auto ordering = bulk_synchronous::ordering();
    constexpr bool barrier = bulk_synchronous::requires_barrier();
    constexpr bool overlap = bulk_synchronous::allows_overlap();

    static_assert(ordering == memory_ordering::sequential);
    static_assert(barrier == true);
    static_assert(overlap == true);
}

TEST(ConsistencyConstexprTest, RelaxedConstexpr) {
    constexpr auto ordering = relaxed::ordering();
    constexpr bool barrier = relaxed::requires_barrier();
    constexpr bool overlap = relaxed::allows_overlap();

    static_assert(ordering == memory_ordering::relaxed);
    static_assert(barrier == false);
    static_assert(overlap == true);
}

// =============================================================================
// Policy Comparison Tests
// =============================================================================

TEST(ConsistencyComparisonTest, OrderingStrength) {
    // Sequential is strongest
    EXPECT_EQ(bulk_synchronous::ordering(), memory_ordering::sequential);
    EXPECT_EQ(sequential_consistent::ordering(), memory_ordering::sequential);

    // Relaxed is weakest
    EXPECT_EQ(relaxed::ordering(), memory_ordering::relaxed);

    // Release-acquire is in between
    EXPECT_EQ(release_acquire::ordering(), memory_ordering::acquire_release);
}

TEST(ConsistencyComparisonTest, BarrierRequirements) {
    // BSP and sequential require barriers
    EXPECT_TRUE(bulk_synchronous::requires_barrier());
    EXPECT_TRUE(sequential_consistent::requires_barrier());

    // Relaxed and release-acquire don't require barriers
    EXPECT_FALSE(relaxed::requires_barrier());
    EXPECT_FALSE(release_acquire::requires_barrier());
}

}  // namespace dtl::test
