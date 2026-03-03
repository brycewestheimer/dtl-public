// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_policy_set.cpp
/// @brief Unit tests for policy_set extraction and composition
/// @details Tests for Task 2.1.6: policy_set composition from variadic packs

#include <dtl/policies/policies.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Policy Extraction Tests
// =============================================================================

TEST(PolicySetTest, DefaultPoliciesApplied) {
    // When no policies specified, all defaults should be used
    using ps = make_policy_set<>;

    static_assert(std::is_same_v<typename ps::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename ps::placement_policy, default_placement>);
    static_assert(std::is_same_v<typename ps::consistency_policy, default_consistency>);
    static_assert(std::is_same_v<typename ps::execution_policy, default_execution>);
    static_assert(std::is_same_v<typename ps::error_policy, default_error>);
}

TEST(PolicySetTest, ExtractPartitionPolicy) {
    // Specify only partition policy
    using ps = make_policy_set<cyclic_partition<>>;

    static_assert(std::is_same_v<typename ps::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<typename ps::placement_policy, default_placement>);
    static_assert(std::is_same_v<typename ps::consistency_policy, default_consistency>);
}

TEST(PolicySetTest, ExtractPlacementPolicy) {
    // Specify only placement policy
    using ps = make_policy_set<device_only<0>>;

    static_assert(std::is_same_v<typename ps::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename ps::placement_policy, device_only<0>>);
}

TEST(PolicySetTest, ExtractConsistencyPolicy) {
    // Specify only consistency policy
    using ps = make_policy_set<sequential_consistent>;

    static_assert(std::is_same_v<typename ps::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename ps::consistency_policy, sequential_consistent>);
}

TEST(PolicySetTest, ExtractExecutionPolicy) {
    // Specify only execution policy
    using ps = make_policy_set<par>;

    static_assert(std::is_same_v<typename ps::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename ps::execution_policy, par>);
}

TEST(PolicySetTest, ExtractErrorPolicy) {
    // Specify only error policy
    using ps = make_policy_set<throwing_policy>;

    static_assert(std::is_same_v<typename ps::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename ps::error_policy, throwing_policy>);
}

TEST(PolicySetTest, MultiplePolicesInAnyOrder) {
    // Policies can be specified in any order
    using ps1 = make_policy_set<cyclic_partition<>, device_only<0>, par>;
    using ps2 = make_policy_set<device_only<0>, par, cyclic_partition<>>;
    using ps3 = make_policy_set<par, cyclic_partition<>, device_only<0>>;

    // All three should have the same policies
    static_assert(std::is_same_v<typename ps1::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<typename ps1::placement_policy, device_only<0>>);
    static_assert(std::is_same_v<typename ps1::execution_policy, par>);

    static_assert(std::is_same_v<typename ps2::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<typename ps2::placement_policy, device_only<0>>);
    static_assert(std::is_same_v<typename ps2::execution_policy, par>);

    static_assert(std::is_same_v<typename ps3::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<typename ps3::placement_policy, device_only<0>>);
    static_assert(std::is_same_v<typename ps3::execution_policy, par>);
}

TEST(PolicySetTest, AllPoliciesSpecified) {
    using ps = make_policy_set<
        cyclic_partition<>,
        device_only<0>,
        sequential_consistent,
        async,
        throwing_policy
    >;

    static_assert(std::is_same_v<typename ps::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<typename ps::placement_policy, device_only<0>>);
    static_assert(std::is_same_v<typename ps::consistency_policy, sequential_consistent>);
    static_assert(std::is_same_v<typename ps::execution_policy, async>);
    static_assert(std::is_same_v<typename ps::error_policy, throwing_policy>);
}

// =============================================================================
// Policy Count Tests
// =============================================================================

TEST(PolicyCountTest, CountSinglePolicies) {
    using count = policy_count<cyclic_partition<>>;

    static_assert(count::partition == 1);
    static_assert(count::placement == 0);
    static_assert(count::consistency == 0);
    static_assert(count::execution == 0);
    static_assert(count::error == 0);
    static_assert(count::no_duplicates);
}

TEST(PolicyCountTest, CountMultiplePolicies) {
    using count = policy_count<cyclic_partition<>, device_only<0>, par>;

    static_assert(count::partition == 1);
    static_assert(count::placement == 1);
    static_assert(count::consistency == 0);
    static_assert(count::execution == 1);
    static_assert(count::error == 0);
    static_assert(count::no_duplicates);
}

TEST(PolicyCountTest, NoDuplicatesAllowed) {
    // Test duplicate detection
    using count_dup = policy_count<cyclic_partition<>, block_partition<>>;

    static_assert(count_dup::partition == 2);
    static_assert(!count_dup::no_duplicates);
}

// =============================================================================
// Policy Trait Tests
// =============================================================================

TEST(PolicyTraitTest, IsPolicyV) {
    // Test is_policy_v for all policy types
    static_assert(is_policy_v<block_partition<>>);
    static_assert(is_policy_v<cyclic_partition<>>);
    static_assert(is_policy_v<replicated>);
    static_assert(is_policy_v<host_only>);
    static_assert(is_policy_v<device_only<0>>);
    static_assert(is_policy_v<bulk_synchronous>);
    static_assert(is_policy_v<seq>);
    static_assert(is_policy_v<par>);
    static_assert(is_policy_v<async>);
    static_assert(is_policy_v<expected_policy>);
    static_assert(is_policy_v<throwing_policy>);

    // Non-policies should return false
    static_assert(!is_policy_v<int>);
    static_assert(!is_policy_v<double>);
    static_assert(!is_policy_v<std::string>);
}

TEST(PolicyTraitTest, PolicyConcept) {
    // Test the Policy concept
    static_assert(Policy<block_partition<>>);
    static_assert(Policy<cyclic_partition<>>);
    static_assert(Policy<host_only>);
    static_assert(Policy<seq>);

    static_assert(!Policy<int>);
    static_assert(!Policy<double>);
}

TEST(PolicyTraitTest, AllPoliciesConcept) {
    // Test AllPolicies concept
    static_assert(AllPolicies<block_partition<>, host_only, seq>);
    static_assert(!AllPolicies<block_partition<>, int, seq>);
}

// =============================================================================
// Default Policy Tests
// =============================================================================

TEST(DefaultPolicyTest, DefaultPartitionIsBlock) {
    static_assert(std::is_same_v<default_partition, block_partition<>>);
}

TEST(DefaultPolicyTest, DefaultPlacementIsHostOnly) {
    static_assert(std::is_same_v<default_placement, host_only>);
}

TEST(DefaultPolicyTest, DefaultConsistencyIsBSP) {
    static_assert(std::is_same_v<default_consistency, bulk_synchronous>);
}

TEST(DefaultPolicyTest, DefaultExecutionIsSeq) {
    static_assert(std::is_same_v<default_execution, seq>);
}

TEST(DefaultPolicyTest, DefaultErrorIsExpected) {
    static_assert(std::is_same_v<default_error, expected_policy>);
}

}  // namespace dtl::test
