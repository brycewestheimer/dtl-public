// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_policy_mixin.cpp
/// @brief Unit tests for container_policy_types mixin
/// @details Phase 27 Task 27.1: Verify that the policy mixin correctly extracts
///          all 5 policy type aliases from variadic packs, matching the behavior
///          that containers get from inline make_policy_set usage.

#include <dtl/containers/detail/policy_mixin.hpp>
#include <dtl/policies/policies.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Default Policy Extraction
// =============================================================================

TEST(PolicyMixinTest, DefaultPoliciesMatchExpected) {
    using mixin = detail::container_policy_types<>;

    static_assert(std::is_same_v<mixin::partition_policy, default_partition>);
    static_assert(std::is_same_v<mixin::placement_policy, default_placement>);
    static_assert(std::is_same_v<mixin::consistency_policy, default_consistency>);
    static_assert(std::is_same_v<mixin::execution_policy, default_execution>);
    static_assert(std::is_same_v<mixin::error_policy, default_error>);

    // Verify the policies alias matches make_policy_set
    static_assert(std::is_same_v<mixin::policies, make_policy_set<>>);
}

// =============================================================================
// Custom Policy Extraction
// =============================================================================

TEST(PolicyMixinTest, CustomPoliciesExtracted) {
    using mixin = detail::container_policy_types<cyclic_partition<>, par>;

    static_assert(std::is_same_v<mixin::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<mixin::placement_policy, default_placement>,
                  "Unspecified placement should be default");
    static_assert(std::is_same_v<mixin::consistency_policy, default_consistency>,
                  "Unspecified consistency should be default");
    static_assert(std::is_same_v<mixin::execution_policy, par>);
    static_assert(std::is_same_v<mixin::error_policy, default_error>,
                  "Unspecified error should be default");
}

// =============================================================================
// Mixin Matches Container Inline Usage
// =============================================================================

TEST(PolicyMixinTest, MatchesInlineContainerPattern) {
    // This test verifies that the mixin produces identical types to what
    // containers compute inline with `using policies = make_policy_set<Policies...>;`
    using inline_policies = make_policy_set<cyclic_partition<>, throwing_policy>;
    using mixin = detail::container_policy_types<cyclic_partition<>, throwing_policy>;

    static_assert(std::is_same_v<mixin::partition_policy,
                                  typename inline_policies::partition_policy>);
    static_assert(std::is_same_v<mixin::placement_policy,
                                  typename inline_policies::placement_policy>);
    static_assert(std::is_same_v<mixin::consistency_policy,
                                  typename inline_policies::consistency_policy>);
    static_assert(std::is_same_v<mixin::execution_policy,
                                  typename inline_policies::execution_policy>);
    static_assert(std::is_same_v<mixin::error_policy,
                                  typename inline_policies::error_policy>);
}

// =============================================================================
// All Five Policies Specified
// =============================================================================

TEST(PolicyMixinTest, AllFivePoliciesSpecified) {
    using mixin = detail::container_policy_types<
        cyclic_partition<>,
        host_only,
        sequential_consistent,
        par,
        throwing_policy
    >;

    static_assert(std::is_same_v<mixin::partition_policy, cyclic_partition<>>);
    static_assert(std::is_same_v<mixin::placement_policy, host_only>);
    static_assert(std::is_same_v<mixin::consistency_policy, sequential_consistent>);
    static_assert(std::is_same_v<mixin::execution_policy, par>);
    static_assert(std::is_same_v<mixin::error_policy, throwing_policy>);
}

}  // namespace dtl::test
