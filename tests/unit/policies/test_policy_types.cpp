// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_policy_types.cpp
/// @brief Unit tests for all DTL policy types
/// @details Phase 13 T07: Tests basic construction, property queries,
///          and concept satisfaction for all policy categories.

#include <dtl/policies/policies.hpp>
#include <dtl/policies/partition/partition_policy.hpp>
#include <dtl/policies/placement/placement_policy.hpp>
#include <dtl/policies/consistency/consistency_policy.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/error/error_policy.hpp>

#include <dtl/core/concepts.hpp>
#include <dtl/core/traits.hpp>

#include <gtest/gtest.h>

#include <functional>

namespace dtl::test {

// =============================================================================
// Partition Policy Concept Conformance (compile-time)
// =============================================================================

// Static partition policies satisfy both concepts
static_assert(PartitionPolicyConcept<block_partition<0>>,
              "block_partition must satisfy PartitionPolicyConcept");
static_assert(PartitionPolicyConcept<cyclic_partition<0>>,
              "cyclic_partition must satisfy PartitionPolicyConcept");
static_assert(PartitionPolicyConcept<replicated>,
              "replicated must satisfy PartitionPolicyConcept");
static_assert(PartitionPolicyConcept<hash_partition<>>,
              "hash_partition must satisfy PartitionPolicyConcept");

// Runtime partition policies satisfy concept through instance-based syntax
static_assert(PartitionPolicyConcept<dynamic_block>,
              "dynamic_block must satisfy PartitionPolicyConcept");
static_assert(PartitionPolicyConcept<dynamic_custom_partition>,
              "dynamic_custom_partition must satisfy PartitionPolicyConcept");

// PartitionPolicy (from concepts.hpp) checks
static_assert(PartitionPolicy<block_partition<0>>);
static_assert(PartitionPolicy<cyclic_partition<0>>);
static_assert(PartitionPolicy<replicated>);
static_assert(PartitionPolicy<hash_partition<>>);
static_assert(PartitionPolicy<dynamic_block>);
static_assert(PartitionPolicy<dynamic_custom_partition>);

// Trait checks
static_assert(is_partition_policy_v<block_partition<0>>);
static_assert(is_partition_policy_v<cyclic_partition<0>>);
static_assert(is_partition_policy_v<replicated>);
static_assert(is_partition_policy_v<hash_partition<>>);
static_assert(is_partition_policy_v<custom_partition<std::function<rank_t(index_t, size_type, rank_t)>>>);
static_assert(is_partition_policy_v<dynamic_block>);

// =============================================================================
// Consistency Policy Concept Conformance (compile-time)
// =============================================================================

static_assert(ConsistencyPolicy<relaxed>);
static_assert(ConsistencyPolicy<release_acquire>);
static_assert(ConsistencyPolicy<sequential_consistent>);
static_assert(ConsistencyPolicy<bulk_synchronous>);

static_assert(is_consistency_policy_v<relaxed>);
static_assert(is_consistency_policy_v<release_acquire>);
static_assert(is_consistency_policy_v<sequential_consistent>);
static_assert(is_consistency_policy_v<bulk_synchronous>);

// =============================================================================
// Placement Policy Concept Conformance (compile-time)
// =============================================================================

static_assert(PlacementPolicy<host_only>);
static_assert(PlacementPolicy<device_only<0>>);
static_assert(PlacementPolicy<unified_memory>);
static_assert(PlacementPolicy<device_preferred>);

static_assert(is_placement_policy_v<host_only>);
static_assert(is_placement_policy_v<device_only<0>>);
static_assert(is_placement_policy_v<unified_memory>);
static_assert(is_placement_policy_v<device_preferred>);

// PlacementPolicyConcept checks
static_assert(PlacementPolicyConcept<host_only>);
static_assert(PlacementPolicyConcept<device_only<0>>);
static_assert(PlacementPolicyConcept<unified_memory>);
static_assert(PlacementPolicyConcept<device_preferred>);

// =============================================================================
// Execution Policy Concept Conformance (compile-time)
// =============================================================================

static_assert(ExecutionPolicy<seq>);
static_assert(ExecutionPolicy<par>);
static_assert(ExecutionPolicy<async>);

static_assert(is_execution_policy_v<seq>);
static_assert(is_execution_policy_v<par>);
static_assert(is_execution_policy_v<async>);

// =============================================================================
// Error Policy Concept Conformance (compile-time)
// =============================================================================

static_assert(ErrorPolicy<throwing_policy>);
static_assert(ErrorPolicy<terminating_policy>);
static_assert(ErrorPolicy<expected_policy>);

static_assert(is_error_policy_v<throwing_policy>);
static_assert(is_error_policy_v<terminating_policy>);
static_assert(is_error_policy_v<expected_policy>);

// =============================================================================
// Partition Policy Runtime Tests
// =============================================================================

TEST(BlockPartitionTest, OwnerDistribution) {
    // 10 elements across 3 ranks
    // Expected: rank 0 gets 4, rank 1 gets 3, rank 2 gets 3
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    EXPECT_EQ(block_partition<0>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(3, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(4, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(9, global_size, num_ranks), 2);
}

TEST(BlockPartitionTest, LocalSize) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    size_type total = 0;
    for (rank_t r = 0; r < num_ranks; ++r) {
        total += block_partition<0>::local_size(global_size, num_ranks, r);
    }
    EXPECT_EQ(total, global_size);
}

TEST(BlockPartitionTest, LocalGlobalRoundTrip) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    for (rank_t r = 0; r < num_ranks; ++r) {
        auto ls = block_partition<0>::local_size(global_size, num_ranks, r);
        for (size_type li = 0; li < ls; ++li) {
            index_t gi = block_partition<0>::to_global(static_cast<index_t>(li), global_size, num_ranks, r);
            index_t back = block_partition<0>::to_local(gi, global_size, num_ranks, r);
            EXPECT_EQ(back, static_cast<index_t>(li));
        }
    }
}

TEST(CyclicPartitionTest, OwnerDistribution) {
    constexpr size_type global_size = 9;
    constexpr rank_t num_ranks = 3;

    // Element i belongs to rank (i % 3)
    EXPECT_EQ(cyclic_partition<0>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<0>::owner(1, global_size, num_ranks), 1);
    EXPECT_EQ(cyclic_partition<0>::owner(2, global_size, num_ranks), 2);
    EXPECT_EQ(cyclic_partition<0>::owner(3, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<0>::owner(8, global_size, num_ranks), 2);
}

TEST(CyclicPartitionTest, LocalSize) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    size_type total = 0;
    for (rank_t r = 0; r < num_ranks; ++r) {
        total += cyclic_partition<0>::local_size(global_size, num_ranks, r);
    }
    EXPECT_EQ(total, global_size);
}

TEST(ReplicatedPartitionTest, AllRanksOwnAll) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 4;

    // Owner returns all_ranks
    EXPECT_EQ(replicated::owner(0, global_size, num_ranks), all_ranks);
    EXPECT_EQ(replicated::owner(50, global_size, num_ranks), all_ranks);
}

TEST(ReplicatedPartitionTest, LocalSizeEqualsGlobal) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 4;

    for (rank_t r = 0; r < num_ranks; ++r) {
        EXPECT_EQ(replicated::local_size(global_size, num_ranks, r), global_size);
    }
}

TEST(ReplicatedPartitionTest, IdentityMapping) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 2;

    for (index_t i = 0; i < static_cast<index_t>(global_size); ++i) {
        EXPECT_EQ(replicated::to_local(i, global_size, num_ranks, 0), i);
        EXPECT_EQ(replicated::to_global(i, global_size, num_ranks, 0), i);
    }
}

TEST(HashPartitionTest, OwnerInRange) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 4;

    for (index_t i = 0; i < static_cast<index_t>(global_size); ++i) {
        rank_t owner = hash_partition<>::owner(i, global_size, num_ranks);
        EXPECT_GE(owner, 0);
        EXPECT_LT(owner, num_ranks);
    }
}

TEST(HashPartitionTest, ConsistentMapping) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 4;

    // Same index should always map to same rank
    for (index_t i = 0; i < 10; ++i) {
        rank_t owner1 = hash_partition<>::owner(i, global_size, num_ranks);
        rank_t owner2 = hash_partition<>::owner(i, global_size, num_ranks);
        EXPECT_EQ(owner1, owner2);
    }
}

TEST(CustomPartitionTest, UserDefinedMapping) {
    // Custom function: even indices to rank 0, odd to rank 1
    auto partition = make_custom_partition(
        [](index_t idx, size_type /*gs*/, rank_t /*nr*/) -> rank_t {
            return static_cast<rank_t>(idx % 2);
        });

    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 2;

    EXPECT_EQ(partition.owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(partition.owner(1, global_size, num_ranks), 1);
    EXPECT_EQ(partition.owner(4, global_size, num_ranks), 0);
    EXPECT_EQ(partition.owner(7, global_size, num_ranks), 1);
}

TEST(CustomPartitionTest, ExactLocalSize) {
    auto partition = make_custom_partition(
        [](index_t idx, size_type /*gs*/, rank_t /*nr*/) -> rank_t {
            return static_cast<rank_t>(idx % 2);
        });

    // 10 elements, even/odd split → 5 each
    EXPECT_EQ(partition.exact_local_size(10, 2, 0), 5u);
    EXPECT_EQ(partition.exact_local_size(10, 2, 1), 5u);
}

TEST(CustomPartitionTest, ToLocalToGlobalRoundTrip) {
    auto partition = make_custom_partition(
        [](index_t idx, size_type /*gs*/, rank_t /*nr*/) -> rank_t {
            return static_cast<rank_t>(idx % 2);
        });

    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 2;

    // Rank 0 owns elements 0, 2, 4, 6, 8
    EXPECT_EQ(partition.to_local(0, global_size, num_ranks, 0), 0);
    EXPECT_EQ(partition.to_local(4, global_size, num_ranks, 0), 2);
    EXPECT_EQ(partition.to_global(0, global_size, num_ranks, 0), 0);
    EXPECT_EQ(partition.to_global(2, global_size, num_ranks, 0), 4);
}

TEST(DynamicBlockTest, UniformFallback) {
    dynamic_block db;

    // Without boundaries, falls back to block_partition
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    for (index_t i = 0; i < static_cast<index_t>(global_size); ++i) {
        EXPECT_EQ(db.owner(i, global_size, num_ranks),
                  block_partition<0>::owner(i, global_size, num_ranks));
    }
}

TEST(DynamicBlockTest, ExplicitBoundaries) {
    // 10 elements: rank 0 gets [0,3), rank 1 gets [3,7), rank 2 gets [7,10)
    dynamic_block db({0, 3, 7, 10});

    EXPECT_TRUE(db.has_explicit_boundaries());
    EXPECT_EQ(db.owner(0, 10, 3), 0);
    EXPECT_EQ(db.owner(2, 10, 3), 0);
    EXPECT_EQ(db.owner(3, 10, 3), 1);
    EXPECT_EQ(db.owner(6, 10, 3), 1);
    EXPECT_EQ(db.owner(7, 10, 3), 2);
    EXPECT_EQ(db.owner(9, 10, 3), 2);
}

TEST(DynamicBlockTest, FromSizes) {
    auto db = dynamic_block::from_sizes({3, 4, 3});

    EXPECT_EQ(db.local_size(10, 3, 0), 3u);
    EXPECT_EQ(db.local_size(10, 3, 1), 4u);
    EXPECT_EQ(db.local_size(10, 3, 2), 3u);
}

TEST(DynamicBlockTest, BinarySearchCorrectness) {
    // Ensure binary search produces same results as conceptual linear scan
    auto db = dynamic_block::from_sizes({5, 3, 2});

    for (index_t i = 0; i < 10; ++i) {
        rank_t owner = db.owner(i, 10, 3);
        EXPECT_GE(owner, 0);
        EXPECT_LT(owner, 3);
    }

    // Verify specific boundary cases
    EXPECT_EQ(db.owner(0, 10, 3), 0);
    EXPECT_EQ(db.owner(4, 10, 3), 0);
    EXPECT_EQ(db.owner(5, 10, 3), 1);
    EXPECT_EQ(db.owner(7, 10, 3), 1);
    EXPECT_EQ(db.owner(8, 10, 3), 2);
    EXPECT_EQ(db.owner(9, 10, 3), 2);
}

TEST(DynamicBlockTest, LocalGlobalRoundTrip) {
    auto db = dynamic_block::from_sizes({3, 4, 3});

    for (rank_t r = 0; r < 3; ++r) {
        auto ls = db.local_size(10, 3, r);
        for (size_type li = 0; li < ls; ++li) {
            index_t gi = db.to_global(static_cast<index_t>(li), 10, 3, r);
            index_t back = db.to_local(gi, 10, 3, r);
            EXPECT_EQ(back, static_cast<index_t>(li));
        }
    }
}

// =============================================================================
// Consistency Policy Runtime Tests
// =============================================================================

TEST(RelaxedTest, Properties) {
    EXPECT_EQ(relaxed::ordering(), memory_ordering::relaxed);
    EXPECT_FALSE(relaxed::requires_barrier());
    EXPECT_TRUE(relaxed::allows_overlap());
    EXPECT_FALSE(relaxed::needs_collective_sync());
    EXPECT_EQ(relaxed::default_sync(), sync_point::none);
    EXPECT_TRUE(relaxed::allows_stale_reads());
}

TEST(RelaxedTest, Traits) {
    EXPECT_FALSE(consistency_traits<relaxed>::requires_barrier);
    EXPECT_TRUE(consistency_traits<relaxed>::allows_overlap);
    EXPECT_EQ(consistency_traits<relaxed>::default_ordering, memory_ordering::relaxed);
}

TEST(ReleaseAcquireTest, Properties) {
    EXPECT_EQ(release_acquire::ordering(), memory_ordering::acquire_release);
    EXPECT_FALSE(release_acquire::requires_barrier());
    EXPECT_TRUE(release_acquire::allows_overlap());
    EXPECT_FALSE(release_acquire::needs_collective_sync());
    EXPECT_EQ(release_acquire::default_sync(), sync_point::fence);
}

TEST(ReleaseAcquireTest, Traits) {
    EXPECT_FALSE(consistency_traits<release_acquire>::requires_barrier);
    EXPECT_TRUE(consistency_traits<release_acquire>::allows_overlap);
    EXPECT_EQ(consistency_traits<release_acquire>::default_ordering, memory_ordering::acquire_release);
}

TEST(SequentialConsistentTest, Properties) {
    EXPECT_EQ(sequential_consistent::ordering(), memory_ordering::sequential);
    EXPECT_TRUE(sequential_consistent::requires_barrier());  // T02: must be true
    EXPECT_FALSE(sequential_consistent::allows_overlap());
    EXPECT_TRUE(sequential_consistent::needs_collective_sync());
    EXPECT_EQ(sequential_consistent::default_sync(), sync_point::barrier);
    EXPECT_TRUE(sequential_consistent::immediate_visibility());
    EXPECT_FALSE(sequential_consistent::allows_stale_reads());
}

TEST(SequentialConsistentTest, Traits) {
    EXPECT_TRUE(consistency_traits<sequential_consistent>::requires_barrier);
    EXPECT_FALSE(consistency_traits<sequential_consistent>::allows_overlap);
    EXPECT_EQ(consistency_traits<sequential_consistent>::default_ordering,
              memory_ordering::sequential);
}

TEST(BulkSynchronousTest, Properties) {
    EXPECT_EQ(bulk_synchronous::ordering(), memory_ordering::sequential);
    EXPECT_TRUE(bulk_synchronous::requires_barrier());
    EXPECT_TRUE(bulk_synchronous::allows_overlap());
    EXPECT_TRUE(bulk_synchronous::needs_collective_sync());
    EXPECT_EQ(bulk_synchronous::default_sync(), sync_point::barrier);
    EXPECT_TRUE(bulk_synchronous::buffers_writes());
}

TEST(BulkSynchronousTest, Traits) {
    EXPECT_TRUE(consistency_traits<bulk_synchronous>::requires_barrier);
    EXPECT_TRUE(consistency_traits<bulk_synchronous>::allows_overlap);
    EXPECT_EQ(consistency_traits<bulk_synchronous>::default_ordering,
              memory_ordering::sequential);
}

// =============================================================================
// Placement Policy Runtime Tests
// =============================================================================

TEST(HostOnlyTest, Properties) {
    EXPECT_EQ(host_only::preferred_location(), memory_location::host);
    EXPECT_TRUE(host_only::is_host_accessible());
    EXPECT_FALSE(host_only::is_device_accessible());
    EXPECT_TRUE(host_only::requires_device_copy());
    EXPECT_EQ(host_only::device_id(), -1);
}

TEST(DeviceOnlyTest, Properties) {
    EXPECT_EQ(device_only<0>::preferred_location(), memory_location::device);
    EXPECT_FALSE(device_only<0>::is_host_accessible());
    EXPECT_TRUE(device_only<0>::is_device_accessible());
    EXPECT_TRUE(device_only<0>::requires_host_copy());
    EXPECT_FALSE(device_only<0>::requires_device_copy());
    EXPECT_EQ(device_only<0>::device, 0);
    EXPECT_EQ(device_only<1>::device, 1);
}

TEST(DeviceOnlyTest, DifferentDeviceIds) {
    static_assert(!std::is_same_v<device_only<0>, device_only<1>>,
                  "Different device IDs should produce different types");
    EXPECT_EQ(device_only<0>::device, 0);
    EXPECT_EQ(device_only<2>::device, 2);
}

TEST(UnifiedMemoryTest, Properties) {
    EXPECT_EQ(unified_memory::preferred_location(), memory_location::unified);
    EXPECT_TRUE(unified_memory::is_host_accessible());
    EXPECT_TRUE(unified_memory::is_device_accessible());
    EXPECT_FALSE(unified_memory::requires_explicit_copy());
    EXPECT_TRUE(unified_memory::supports_prefetch());
}

TEST(DevicePreferredTest, Properties) {
    EXPECT_EQ(device_preferred::preferred_location(), memory_location::device);
    EXPECT_EQ(device_preferred::fallback_location(), memory_location::host);
    EXPECT_TRUE(device_preferred::allows_fallback());
}

TEST(ExplicitPlacementTest, UserMapping) {
    auto placement = make_explicit_placement(
        [](index_t idx, rank_t /*rank*/) -> memory_location {
            return (idx % 2 == 0) ? memory_location::host : memory_location::device;
        });

    EXPECT_EQ(placement.location_for(0, 0), memory_location::host);
    EXPECT_EQ(placement.location_for(1, 0), memory_location::device);
    EXPECT_TRUE(explicit_placement<decltype([](index_t, rank_t) { return memory_location::host; })>::is_heterogeneous());
}

// =============================================================================
// Execution Policy Runtime Tests
// =============================================================================

TEST(SeqTest, Properties) {
    EXPECT_EQ(seq::mode(), execution_mode::synchronous);
    EXPECT_TRUE(seq::is_blocking());
    EXPECT_FALSE(seq::is_parallel());
    EXPECT_EQ(seq::parallelism(), parallelism_level::sequential);
    EXPECT_TRUE(seq::allows_vectorization());
    EXPECT_TRUE(seq::is_deterministic());
}

TEST(SeqTest, Traits) {
    EXPECT_TRUE(execution_traits<seq>::is_blocking);
    EXPECT_FALSE(execution_traits<seq>::is_parallel);
    EXPECT_EQ(execution_traits<seq>::mode, execution_mode::synchronous);
    EXPECT_EQ(execution_traits<seq>::parallelism, parallelism_level::sequential);
}

TEST(ParTest, Properties) {
    EXPECT_EQ(par::mode(), execution_mode::synchronous);
    EXPECT_TRUE(par::is_blocking());
    EXPECT_TRUE(par::is_parallel());
    EXPECT_EQ(par::parallelism(), parallelism_level::parallel);
    EXPECT_FALSE(par::is_deterministic());
}

TEST(AsyncTest, Properties) {
    EXPECT_EQ(async::mode(), execution_mode::asynchronous);
    EXPECT_FALSE(async::is_blocking());
    EXPECT_TRUE(async::is_parallel());
    EXPECT_EQ(async::parallelism(), parallelism_level::parallel);
    EXPECT_TRUE(async::requires_wait());
    EXPECT_TRUE(async::supports_continuations());
}

// =============================================================================
// Error Policy Runtime Tests
// =============================================================================

TEST(ThrowingPolicyTest, Properties) {
    EXPECT_EQ(throwing_policy::strategy(), error_strategy::throw_exception);
    EXPECT_TRUE(throwing_policy::uses_exceptions());
}

TEST(ThrowingPolicyTest, Traits) {
    EXPECT_EQ(error_policy_traits<throwing_policy>::strategy, error_strategy::throw_exception);
    EXPECT_TRUE(error_policy_traits<throwing_policy>::uses_exceptions);
}

TEST(TerminatingPolicyTest, Properties) {
    EXPECT_EQ(terminating_policy::strategy(), error_strategy::terminate);
    EXPECT_FALSE(terminating_policy::uses_exceptions());
    EXPECT_FALSE(terminating_policy::can_ignore_errors());
}

TEST(ExpectedPolicyTest, Properties) {
    EXPECT_EQ(expected_policy::strategy(), error_strategy::return_result);
    EXPECT_FALSE(expected_policy::uses_exceptions());
    EXPECT_FALSE(expected_policy::can_ignore_errors());
    EXPECT_FALSE(expected_policy::auto_propagates());
}

TEST(ExpectedPolicyTest, HandleError) {
    auto res = expected_policy::handle_error<int>(
        status(status_code::out_of_bounds));
    EXPECT_FALSE(res.has_value());
    EXPECT_EQ(res.error().code(), status_code::out_of_bounds);
}

TEST(ExpectedPolicyTest, HandleSuccess) {
    auto res = expected_policy::handle_success<int>(42);
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 42);
}

TEST(CallbackPolicyTest, Properties) {
    auto policy = callback_policy{[](status /*s*/) {
        return error_action::continue_execution;
    }};
    EXPECT_EQ(policy.strategy(), error_strategy::callback);
    EXPECT_FALSE(policy.uses_exceptions());
    EXPECT_TRUE(policy.can_ignore_errors());
}

TEST(CallbackPolicyTest, ConceptSatisfaction) {
    using cb_type = callback_policy<std::function<error_action(status)>>;
    static_assert(is_error_policy_v<cb_type>);
    static_assert(ErrorPolicy<cb_type>);
    SUCCEED();
}

}  // namespace dtl::test
