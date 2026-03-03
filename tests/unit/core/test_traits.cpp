// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_traits.cpp
/// @brief Unit tests for dtl/core/traits.hpp
/// @details Tests type traits for serializability, transportability, and policies.

#include <dtl/core/traits.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Trivially Serializable Tests
// =============================================================================

TEST(TraitsTest, TriviallySerializablePrimitives) {
    static_assert(is_trivially_serializable_v<int>);
    static_assert(is_trivially_serializable_v<double>);
    static_assert(is_trivially_serializable_v<float>);
    static_assert(is_trivially_serializable_v<char>);
    static_assert(is_trivially_serializable_v<bool>);
    static_assert(is_trivially_serializable_v<std::uint64_t>);
    static_assert(is_trivially_serializable_v<std::int32_t>);
}

TEST(TraitsTest, TriviallySerializablePointers) {
    // Raw pointers are trivially copyable but semantically problematic
    // The trait correctly identifies them as trivially serializable
    static_assert(is_trivially_serializable_v<int*>);
    static_assert(is_trivially_serializable_v<void*>);
}

TEST(TraitsTest, TriviallySerializableStructs) {
    struct TrivialPOD {
        int x;
        double y;
        char z;
    };

    static_assert(is_trivially_serializable_v<TrivialPOD>);
}

TEST(TraitsTest, NotTriviallySerializableTypes) {
    // String has non-trivial copy constructor
    static_assert(!is_trivially_serializable_v<std::string>);

    // Vector has non-trivial copy constructor
    static_assert(!is_trivially_serializable_v<std::vector<int>>);

    // Types with virtual functions are not standard layout
    struct WithVirtual {
        virtual ~WithVirtual() = default;
    };
    static_assert(!is_trivially_serializable_v<WithVirtual>);
}

// =============================================================================
// Transportable Tests
// =============================================================================

TEST(TraitsTest, TransportablePrimitives) {
    // All trivially serializable types are transportable
    static_assert(is_transportable_v<int>);
    static_assert(is_transportable_v<double>);
    static_assert(is_transportable_v<float>);
}

TEST(TraitsTest, TransportableStructs) {
    struct Simple {
        int a;
        double b;
    };
    static_assert(is_transportable_v<Simple>);
}

// =============================================================================
// View Trait Tests
// =============================================================================

TEST(TraitsTest, ViewTraitDefaults) {
    // Default implementations return false
    static_assert(!is_distributed_container_v<int>);
    static_assert(!is_local_view_v<int>);
    static_assert(!is_global_view_v<int>);
    static_assert(!is_segmented_view_v<int>);
    static_assert(!is_remote_ref_v<int>);
}

// =============================================================================
// Policy Trait Tests
// =============================================================================

namespace {
// Test policies with correct tags
struct TestPartitionPolicy {
    using policy_category = partition_policy_tag;
};

struct TestPlacementPolicy {
    using policy_category = placement_policy_tag;
};

struct TestConsistencyPolicy {
    using policy_category = consistency_policy_tag;
};

struct TestExecutionPolicy {
    using policy_category = execution_policy_tag;
};

struct TestErrorPolicy {
    using policy_category = error_policy_tag;
};

// Non-policy type (no policy_category)
struct NotAPolicy {
    int x;
};
}  // namespace

TEST(TraitsTest, PartitionPolicyDetection) {
    static_assert(is_partition_policy_v<TestPartitionPolicy>);
    static_assert(!is_partition_policy_v<TestPlacementPolicy>);
    static_assert(!is_partition_policy_v<NotAPolicy>);
    static_assert(!is_partition_policy_v<int>);
}

TEST(TraitsTest, PlacementPolicyDetection) {
    static_assert(is_placement_policy_v<TestPlacementPolicy>);
    static_assert(!is_placement_policy_v<TestPartitionPolicy>);
    static_assert(!is_placement_policy_v<NotAPolicy>);
}

TEST(TraitsTest, ConsistencyPolicyDetection) {
    static_assert(is_consistency_policy_v<TestConsistencyPolicy>);
    static_assert(!is_consistency_policy_v<TestPartitionPolicy>);
    static_assert(!is_consistency_policy_v<int>);
}

TEST(TraitsTest, ExecutionPolicyDetection) {
    static_assert(is_execution_policy_v<TestExecutionPolicy>);
    static_assert(!is_execution_policy_v<TestPartitionPolicy>);
    static_assert(!is_execution_policy_v<int>);
}

TEST(TraitsTest, ErrorPolicyDetection) {
    static_assert(is_error_policy_v<TestErrorPolicy>);
    static_assert(!is_error_policy_v<TestPartitionPolicy>);
    static_assert(!is_error_policy_v<int>);
}

// =============================================================================
// Extent Trait Tests
// =============================================================================

TEST(TraitsTest, StaticExtentsDetection) {
    using all_static = extents<10, 20, 30>;
    using all_dynamic = extents<dynamic_extent, dynamic_extent>;
    using mixed = extents<10, dynamic_extent>;

    static_assert(is_static_extents_v<all_static>);
    static_assert(!is_static_extents_v<all_dynamic>);
    static_assert(!is_static_extents_v<mixed>);
}

TEST(TraitsTest, DynamicExtentsDetection) {
    using all_static = extents<10, 20>;
    using all_dynamic = extents<dynamic_extent, dynamic_extent>;
    using mixed = extents<10, dynamic_extent>;

    static_assert(!is_dynamic_extents_v<all_static>);
    static_assert(is_dynamic_extents_v<all_dynamic>);
    static_assert(!is_dynamic_extents_v<mixed>);
}

TEST(TraitsTest, ExtentsRank) {
    using ext_1d = extents<10>;
    using ext_2d = extents<dynamic_extent, dynamic_extent>;
    using ext_3d = extents<5, dynamic_extent, 10>;

    static_assert(extents_rank_v<ext_1d> == 1);
    static_assert(extents_rank_v<ext_2d> == 2);
    static_assert(extents_rank_v<ext_3d> == 3);
}

// =============================================================================
// Type Manipulation Tests
// =============================================================================

TEST(TraitsTest, RemoveCvref) {
    static_assert(std::is_same_v<remove_cvref_t<int>, int>);
    static_assert(std::is_same_v<remove_cvref_t<int&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<int&&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<const int>, int>);
    static_assert(std::is_same_v<remove_cvref_t<const int&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<volatile int>, int>);
    static_assert(std::is_same_v<remove_cvref_t<const volatile int&>, int>);
}

TEST(TraitsTest, SameCvref) {
    static_assert(is_same_cvref_v<int, int>);
    static_assert(is_same_cvref_v<int&, int>);
    static_assert(is_same_cvref_v<const int&, int&&>);
    static_assert(is_same_cvref_v<const int, volatile int>);
    static_assert(!is_same_cvref_v<int, double>);
    static_assert(!is_same_cvref_v<int*, int>);
}

}  // namespace dtl::test
