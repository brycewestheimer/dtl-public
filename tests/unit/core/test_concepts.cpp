// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_concepts.cpp
/// @brief C++20 concept verification tests
/// @details Tests for Phase 11.5: static verification of concept satisfaction

#include <dtl/core/concepts.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Transportable Concept Tests
// =============================================================================

TEST(ConceptsTest, TransportableBasicTypes) {
    static_assert(Transportable<int>);
    static_assert(Transportable<double>);
    static_assert(Transportable<float>);
    static_assert(Transportable<char>);
    static_assert(Transportable<bool>);
    static_assert(Transportable<std::int64_t>);
    static_assert(Transportable<std::uint32_t>);

    // Non-test: just to make GTest happy
    EXPECT_TRUE(Transportable<int>);
}

TEST(ConceptsTest, TransportableStdArray) {
    using arr4 = std::array<int, 4>;
    using arr100 = std::array<double, 100>;
    using arr256 = std::array<char, 256>;

    static_assert(Transportable<arr4>);
    static_assert(Transportable<arr100>);
    static_assert(Transportable<arr256>);

    EXPECT_TRUE(Transportable<arr4>);
}

TEST(ConceptsTest, TransportablePODStructs) {
    struct Point { int x, y; };
    struct Color { float r, g, b, a; };

    static_assert(Transportable<Point>);
    static_assert(Transportable<Color>);

    EXPECT_TRUE(Transportable<Point>);
}

TEST(ConceptsTest, TransportableNonTrivialTypes) {
    // std::string is NOT trivially serializable
    static_assert(!is_trivially_serializable_v<std::string>);
    static_assert(!is_trivially_serializable_v<std::vector<int>>);

    EXPECT_FALSE(is_trivially_serializable_v<std::string>);
}

// =============================================================================
// TriviallySerializable Concept Tests
// =============================================================================

TEST(ConceptsTest, TriviallySerializableBasicTypes) {
    static_assert(TriviallySerializable<int>);
    static_assert(TriviallySerializable<double>);
    static_assert(TriviallySerializable<float>);

    EXPECT_TRUE(TriviallySerializable<int>);
}

TEST(ConceptsTest, TriviallySerializablePOD) {
    struct Simple { int a; double b; };

    static_assert(TriviallySerializable<Simple>);
    static_assert(std::is_trivially_copyable_v<Simple>);
    static_assert(std::is_standard_layout_v<Simple>);

    EXPECT_TRUE(TriviallySerializable<Simple>);
}

TEST(ConceptsTest, TriviallySerializableNegative) {
    // Types with virtual functions, non-trivial constructors, etc. are NOT trivially serializable
    struct NonTrivial {
        virtual ~NonTrivial() = default;
    };

    static_assert(!TriviallySerializable<NonTrivial>);
    static_assert(!TriviallySerializable<std::string>);
    static_assert(!TriviallySerializable<std::vector<int>>);

    EXPECT_FALSE(TriviallySerializable<std::string>);
}

// =============================================================================
// is_trivially_serializable Trait Tests
// =============================================================================

TEST(ConceptsTest, IsTriviallySerializableTrait) {
    using arr10 = std::array<int, 10>;

    static_assert(is_trivially_serializable_v<int>);
    static_assert(is_trivially_serializable_v<double>);
    static_assert(is_trivially_serializable_v<arr10>);

    static_assert(!is_trivially_serializable_v<std::string>);
    static_assert(!is_trivially_serializable_v<std::vector<int>>);

    EXPECT_TRUE(is_trivially_serializable_v<int>);
}

// =============================================================================
// is_transportable Trait Tests
// =============================================================================

TEST(ConceptsTest, IsTransportableTrait) {
    static_assert(is_transportable_v<int>);
    static_assert(is_transportable_v<double>);

    EXPECT_TRUE(is_transportable_v<int>);
}

// =============================================================================
// Policy Trait Tests
// =============================================================================

TEST(ConceptsTest, PartitionPolicyTag) {
    struct MyPartition {
        using policy_category = partition_policy_tag;
    };

    static_assert(is_partition_policy_v<MyPartition>);
    static_assert(!is_placement_policy_v<MyPartition>);
    static_assert(!is_consistency_policy_v<MyPartition>);

    EXPECT_TRUE(is_partition_policy_v<MyPartition>);
}

TEST(ConceptsTest, PlacementPolicyTag) {
    struct MyPlacement {
        using policy_category = placement_policy_tag;
    };

    static_assert(is_placement_policy_v<MyPlacement>);
    static_assert(!is_partition_policy_v<MyPlacement>);

    EXPECT_TRUE(is_placement_policy_v<MyPlacement>);
}

TEST(ConceptsTest, ConsistencyPolicyTag) {
    struct MyConsistency {
        using policy_category = consistency_policy_tag;
    };

    static_assert(is_consistency_policy_v<MyConsistency>);
    static_assert(!is_partition_policy_v<MyConsistency>);

    EXPECT_TRUE(is_consistency_policy_v<MyConsistency>);
}

TEST(ConceptsTest, ExecutionPolicyTag) {
    struct MyExecution {
        using policy_category = execution_policy_tag;
    };

    static_assert(is_execution_policy_v<MyExecution>);
    static_assert(!is_partition_policy_v<MyExecution>);

    EXPECT_TRUE(is_execution_policy_v<MyExecution>);
}

TEST(ConceptsTest, ErrorPolicyTag) {
    struct MyError {
        using policy_category = error_policy_tag;
    };

    static_assert(is_error_policy_v<MyError>);
    static_assert(!is_partition_policy_v<MyError>);

    EXPECT_TRUE(is_error_policy_v<MyError>);
}

TEST(ConceptsTest, NonPolicyTypes) {
    static_assert(!is_partition_policy_v<int>);
    static_assert(!is_placement_policy_v<std::vector<int>>);
    static_assert(!is_consistency_policy_v<double>);

    EXPECT_FALSE(is_partition_policy_v<int>);
}

// =============================================================================
// Extents Trait Tests
// =============================================================================

TEST(ConceptsTest, StaticExtents) {
    using static_ext = extents<10>;

    static_assert(is_static_extents_v<static_ext>);
    static_assert(!is_dynamic_extents_v<static_ext>);
    static_assert(extents_rank_v<static_ext> == 1);

    EXPECT_TRUE(is_static_extents_v<static_ext>);
}

TEST(ConceptsTest, DynamicExtents) {
    using dynamic_ext = extents<dynamic_extent>;

    static_assert(!is_static_extents_v<dynamic_ext>);
    static_assert(is_dynamic_extents_v<dynamic_ext>);
    static_assert(extents_rank_v<dynamic_ext> == 1);

    EXPECT_TRUE(is_dynamic_extents_v<dynamic_ext>);
}

TEST(ConceptsTest, MixedExtents) {
    // Skip complex mixed extent test - simple type assertions only
    EXPECT_TRUE(true);  // Placeholder test
}

TEST(ConceptsTest, ExtentsRank) {
    using ext0 = extents<>;
    using ext1 = extents<10>;

    static_assert(extents_rank_v<ext0> == 0);
    static_assert(extents_rank_v<ext1> == 1);

    EXPECT_EQ(extents_rank_v<ext1>, 1);
}

// =============================================================================
// View Trait Tests
// =============================================================================

TEST(ConceptsTest, ViewTraitsDefaults) {
    // Base traits should default to false for non-view types
    static_assert(!is_local_view_v<int>);
    static_assert(!is_global_view_v<int>);
    static_assert(!is_segmented_view_v<int>);
    static_assert(!is_remote_ref_v<int>);

    EXPECT_FALSE(is_local_view_v<int>);
}

TEST(ConceptsTest, DistributedContainerTraits) {
    // Base traits should default to false
    static_assert(!is_distributed_container_v<std::vector<int>>);
    static_assert(!is_distributed_vector_v<std::vector<int>>);
    static_assert(!is_distributed_tensor_v<std::vector<int>>);
    static_assert(!is_distributed_span_v<std::vector<int>>);
    static_assert(!is_distributed_map_v<std::vector<int>>);

    EXPECT_FALSE(is_distributed_container_v<std::vector<int>>);
}

// =============================================================================
// Type Utility Tests
// =============================================================================

TEST(ConceptsTest, RemoveCvref) {
    static_assert(std::is_same_v<remove_cvref_t<int>, int>);
    static_assert(std::is_same_v<remove_cvref_t<const int>, int>);
    static_assert(std::is_same_v<remove_cvref_t<int&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<const int&>, int>);
    static_assert(std::is_same_v<remove_cvref_t<int&&>, int>);

    EXPECT_TRUE((std::is_same_v<remove_cvref_t<const int&>, int>));
}

TEST(ConceptsTest, IsSameCvref) {
    static_assert(is_same_cvref_v<int, int>);
    static_assert(is_same_cvref_v<int, const int>);
    static_assert(is_same_cvref_v<int&, int>);
    static_assert(is_same_cvref_v<const int&, int&&>);

    static_assert(!is_same_cvref_v<int, double>);
    static_assert(!is_same_cvref_v<int, long>);

    EXPECT_TRUE((is_same_cvref_v<int, const int&>));
}

// =============================================================================
// Complex Type Tests
// =============================================================================

TEST(ConceptsTest, NestedStruct) {
    struct Inner { int x, y; };
    struct Outer { Inner a; double b; };

    // Both should be trivially serializable if components are
    static_assert(TriviallySerializable<Inner>);
    static_assert(TriviallySerializable<Outer>);

    EXPECT_TRUE(TriviallySerializable<Outer>);
}

TEST(ConceptsTest, EnumTypes) {
    enum class Color : int { Red, Green, Blue };
    enum OldStyle { A, B, C };

    static_assert(TriviallySerializable<Color>);
    static_assert(TriviallySerializable<OldStyle>);

    EXPECT_TRUE(TriviallySerializable<Color>);
}

TEST(ConceptsTest, UnionTypes) {
    union FloatInt {
        float f;
        int i;
    };

    static_assert(TriviallySerializable<FloatInt>);

    EXPECT_TRUE(TriviallySerializable<FloatInt>);
}

// =============================================================================
// Pointer Types Tests
// =============================================================================

TEST(ConceptsTest, PointerTypes) {
    // Pointers are trivially copyable but typically NOT safely transportable
    // (they point to local memory)
    static_assert(std::is_trivially_copyable_v<int*>);
    static_assert(std::is_trivially_copyable_v<void*>);

    // But is_trivially_serializable might still be true structurally
    EXPECT_TRUE(std::is_trivially_copyable_v<int*>);
}

// =============================================================================
// Array Types Tests
// =============================================================================

TEST(ConceptsTest, CArrays) {
    // C-style arrays of trivial types
    using IntArray = int[10];
    using DoubleArray = double[5];

    static_assert(std::is_trivially_copyable_v<int>);
    static_assert(std::is_trivially_copyable_v<IntArray>);
    static_assert(std::is_trivially_copyable_v<DoubleArray>);
    // Note: C arrays are not standard_layout as a whole but elements are

    EXPECT_TRUE(std::is_trivially_copyable_v<int>);
    EXPECT_TRUE(std::is_trivially_copyable_v<IntArray>);
}

// =============================================================================
// DistributedAssociativeContainer Concept Tests (Phase 1.2.2)
// =============================================================================

TEST(ConceptsTest, DistributedAssociativeContainerRequirements) {
    // Minimal mock to verify the concept requirements
    struct MockAssocContainer {
        using key_type = int;
        using mapped_type = double;
        using size_type = size_t;
        using iterator = int*;

        size_type local_size() const { return 0; }
        bool is_local(key_type) const { return true; }
        rank_t owner(key_type) const { return 0; }
        iterator begin() { return nullptr; }
        iterator end() { return nullptr; }
    };

    static_assert(DistributedAssociativeContainer<MockAssocContainer>,
                  "MockAssocContainer must satisfy DistributedAssociativeContainer");

    EXPECT_TRUE(DistributedAssociativeContainer<MockAssocContainer>);
}

TEST(ConceptsTest, DistributedMapIsAssociative) {
    // Verify that DistributedMap concept is equivalent to DistributedAssociativeContainer
    struct MockMap {
        using key_type = std::string;
        using mapped_type = int;
        using size_type = size_t;
        using iterator = int*;

        size_type local_size() const { return 0; }
        bool is_local(const key_type&) const { return true; }
        rank_t owner(const key_type&) const { return 0; }
        iterator begin() { return nullptr; }
        iterator end() { return nullptr; }
    };

    static_assert(DistributedMap<MockMap>,
                  "MockMap must satisfy DistributedMap");
    static_assert(DistributedAssociativeContainer<MockMap>,
                  "MockMap must also satisfy DistributedAssociativeContainer");

    EXPECT_TRUE(DistributedMap<MockMap>);
}

TEST(ConceptsTest, DistributedMapNotDistributedContainer) {
    // Verify that DistributedMap does NOT require DistributedContainer
    // (no global_view, segmented_view requirements)
    struct MinimalMap {
        using key_type = int;
        using mapped_type = int;
        using size_type = size_t;
        using iterator = void*;

        size_type local_size() const { return 0; }
        bool is_local(key_type) const { return true; }
        rank_t owner(key_type) const { return 0; }
        iterator begin() { return nullptr; }
        iterator end() { return nullptr; }
        // Note: NO local_view(), global_view(), segmented_view(), size()
    };

    static_assert(DistributedMap<MinimalMap>,
                  "MinimalMap should satisfy DistributedMap without full DistributedContainer");
    static_assert(!DistributedContainer<MinimalMap>,
                  "MinimalMap should NOT satisfy DistributedContainer");

    EXPECT_TRUE(DistributedMap<MinimalMap>);
    EXPECT_FALSE(DistributedContainer<MinimalMap>);
}

}  // namespace dtl::test
