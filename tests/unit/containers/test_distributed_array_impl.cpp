// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_array_impl.cpp
/// @brief Unit tests for distributed_array implementation
/// @details Tests for Phase 10A: distributed_array container

#include <dtl/containers/distributed_array.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// Construction Tests
// =============================================================================

TEST(DistributedArrayTest, DefaultConstruction) {
    distributed_array<int, 100> arr;

    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.local_size(), 100);
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.num_ranks(), 1);
    EXPECT_EQ(arr.rank(), 0);
}

TEST(DistributedArrayTest, ConstructWithDistributionInfo) {
    // 100 elements, 4 ranks, I'm rank 1
    distributed_array<int, 100> arr(test_context{1, 4});

    EXPECT_EQ(arr.size(), 100);  // size() is always 100
    EXPECT_EQ(arr.global_size(), 100);
    EXPECT_EQ(arr.local_size(), 25);
    EXPECT_EQ(arr.num_ranks(), 4);
    EXPECT_EQ(arr.rank(), 1);
}

TEST(DistributedArrayTest, ConstructWithValue) {
    distributed_array<int, 100> arr(42, test_context{1, 4});

    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.local_size(), 25);

    // All local elements should be initialized to 42
    auto local = arr.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(DistributedArrayTest, ConstructSingleRank) {
    distributed_array<int, 100> arr(test_context{0, 1});

    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.local_size(), 100);  // All elements local
    EXPECT_EQ(arr.num_ranks(), 1);
}

TEST(DistributedArrayTest, FactoryCreate) {
    auto result = distributed_array<int, 100>::create(4, 1);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 100);
}

TEST(DistributedArrayTest, CompileTimeExtent) {
    // extent is a compile-time constant
    static_assert(distributed_array<int, 100>::extent == 100);
    static_assert(distributed_array<double, 500>::extent == 500);
    static_assert(distributed_array<char, 0>::extent == 0);
}

TEST(DistributedArrayTest, EmptyArray) {
    distributed_array<int, 0> arr;

    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    EXPECT_EQ(arr.local_size(), 0);
}

// =============================================================================
// Size Query Tests
// =============================================================================

TEST(DistributedArrayTest, SizeQueries) {
    distributed_array<int, 100> arr(test_context{2, 4});

    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.global_size(), 100);
    EXPECT_EQ(arr.local_size(), 25);
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.max_size(), 100);  // max_size == extent for arrays
}

TEST(DistributedArrayTest, LocalSizeForRank) {
    // 10 elements / 4 ranks with remainder
    distributed_array<int, 10> arr(test_context{0, 4});

    // First 2 ranks get 3, last 2 get 2
    EXPECT_EQ(arr.local_size_for_rank(0), 3);
    EXPECT_EQ(arr.local_size_for_rank(1), 3);
    EXPECT_EQ(arr.local_size_for_rank(2), 2);
    EXPECT_EQ(arr.local_size_for_rank(3), 2);
}

TEST(DistributedArrayTest, SizeIsConstexpr) {
    // These should be constexpr
    constexpr size_type s1 = distributed_array<int, 100>::size();
    constexpr size_type s2 = distributed_array<int, 100>::global_size();
    constexpr bool e = distributed_array<int, 100>::empty();
    constexpr size_type m = distributed_array<int, 100>::max_size();

    EXPECT_EQ(s1, 100);
    EXPECT_EQ(s2, 100);
    EXPECT_FALSE(e);
    EXPECT_EQ(m, 100);
}

// =============================================================================
// Local View Tests
// =============================================================================

TEST(DistributedArrayTest, LocalViewAccess) {
    distributed_array<int, 100> arr(test_context{1, 4});

    auto local = arr.local_view();
    EXPECT_EQ(local.size(), 25);
    EXPECT_EQ(local.data(), arr.local_data());

    // Write through view
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i * 10);
    }

    // Verify through direct access
    EXPECT_EQ(arr.local(0), 0);
    EXPECT_EQ(arr.local(1), 10);
    EXPECT_EQ(arr.local(24), 240);
}

TEST(DistributedArrayTest, LocalViewSTLSort) {
    distributed_array<int, 100> arr(test_context{0, 1});

    // Fill with descending values
    auto local = arr.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(100 - i);
    }

    // Sort using STL
    std::sort(local.begin(), local.end());

    // Verify sorted
    for (size_type i = 1; i < local.size(); ++i) {
        EXPECT_LE(local[i - 1], local[i]);
    }
}

TEST(DistributedArrayTest, LocalViewSTLAccumulate) {
    distributed_array<int, 100> arr(1, test_context{0, 1});  // All 1s

    auto local = arr.local_view();
    int sum = std::accumulate(local.begin(), local.end(), 0);
    EXPECT_EQ(sum, 100);
}

// =============================================================================
// Index Translation Tests
// =============================================================================

TEST(DistributedArrayTest, IsLocal) {
    // Rank 1 of 4 with 10 elements
    // Block partition: rank 1 owns [3, 6)
    distributed_array<int, 10> arr(test_context{1, 4});

    EXPECT_FALSE(arr.is_local(0));
    EXPECT_FALSE(arr.is_local(2));
    EXPECT_TRUE(arr.is_local(3));
    EXPECT_TRUE(arr.is_local(4));
    EXPECT_TRUE(arr.is_local(5));
    EXPECT_FALSE(arr.is_local(6));
    EXPECT_FALSE(arr.is_local(9));
}

TEST(DistributedArrayTest, Owner) {
    distributed_array<int, 10> arr(test_context{0, 4});

    EXPECT_EQ(arr.owner(0), 0);
    EXPECT_EQ(arr.owner(2), 0);
    EXPECT_EQ(arr.owner(3), 1);
    EXPECT_EQ(arr.owner(5), 1);
    EXPECT_EQ(arr.owner(6), 2);
    EXPECT_EQ(arr.owner(7), 2);
    EXPECT_EQ(arr.owner(8), 3);
    EXPECT_EQ(arr.owner(9), 3);
}

TEST(DistributedArrayTest, ToLocalToGlobal) {
    distributed_array<int, 100> arr(test_context{2, 4});  // Rank 2

    // For rank 2, local index 0 should be global index 50
    index_t local0_global = arr.to_global(0);
    EXPECT_TRUE(arr.is_local(local0_global));
    EXPECT_EQ(arr.to_local(local0_global), 0);

    // Roundtrip for all local indices
    for (index_t local = 0; local < static_cast<index_t>(arr.local_size()); ++local) {
        index_t global = arr.to_global(local);
        EXPECT_EQ(arr.to_local(global), local);
    }
}

TEST(DistributedArrayTest, GlobalOffset) {
    // Rank 2 of 4 with 100 elements
    // Block partition: rank 2 offset is 50
    distributed_array<int, 100> arr(test_context{2, 4});

    EXPECT_EQ(arr.global_offset(), 50);
}

// =============================================================================
// Global View Tests
// =============================================================================

TEST(DistributedArrayTest, GlobalViewReturnsRemoteRef) {
    distributed_array<int, 100> arr(test_context{1, 4});

    auto global = arr.global_view();
    EXPECT_EQ(global.size(), 100);

    // Access returns remote_ref
    auto ref = global[50];  // This is a global index

    // The ref type should be remote_ref<int>
    static_assert(std::is_same_v<decltype(ref), remote_ref<int>>);
}

TEST(DistributedArrayTest, GlobalViewLocalAccess) {
    distributed_array<int, 100> arr(42, test_context{1, 4});

    auto global = arr.global_view();

    // Access a local element (rank 1 owns [25, 50))
    auto ref = global[25];
    EXPECT_TRUE(ref.is_local());
    EXPECT_EQ(ref.owner_rank(), 1);

    // Get the value
    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(DistributedArrayTest, GlobalViewRemoteAccess) {
    distributed_array<int, 100> arr(test_context{1, 4});

    auto global = arr.global_view();

    // Access a remote element (rank 0 owns [0, 25))
    auto ref = global[0];
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
    EXPECT_EQ(ref.owner_rank(), 0);
}

// =============================================================================
// Segmented View Tests
// =============================================================================

TEST(DistributedArrayTest, SegmentedViewNumSegments) {
    distributed_array<int, 100> arr(test_context{1, 4});

    auto segmented = arr.segmented_view();
    EXPECT_EQ(segmented.num_segments(), 4);
}

TEST(DistributedArrayTest, SegmentedViewLocalSegment) {
    distributed_array<int, 100> arr(42, test_context{1, 4});

    auto segmented = arr.segmented_view();
    auto local_seg = segmented.local_segment();

    EXPECT_TRUE(local_seg.is_local());
    EXPECT_EQ(local_seg.rank, 1);
    EXPECT_EQ(local_seg.size(), 25);
    EXPECT_EQ(local_seg[0], 42);
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST(DistributedArrayTest, Barrier) {
    distributed_array<int, 100> arr(test_context{0, 1});

    auto result = arr.barrier();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedArrayTest, Fence) {
    distributed_array<int, 100> arr(test_context{0, 1});

    auto result = arr.fence();
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Fill Operation Tests
// =============================================================================

TEST(DistributedArrayTest, Fill) {
    distributed_array<int, 100> arr(test_context{1, 4});

    // Fill with value
    arr.fill(99);

    auto local = arr.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 99);
    }
}

// =============================================================================
// No Resize Tests (arrays are fixed size)
// =============================================================================

TEST(DistributedArrayTest, NoResizeMethod) {
    // This test verifies at compile time that resize doesn't exist
    // The code below would fail to compile if uncommented:
    // distributed_array<int, 100> arr;
    // arr.resize(200);  // Should not exist

    // We just verify the size is always the template parameter
    distributed_array<int, 100> arr(test_context{1, 4});
    EXPECT_EQ(arr.size(), 100);

    // Create a new array with different distribution
    distributed_array<int, 100> arr2(test_context{0, 2});
    EXPECT_EQ(arr2.size(), 100);  // Still 100
}

// =============================================================================
// Policy Tests
// =============================================================================

TEST(DistributedArrayTest, DefaultPolicies) {
    using arr_type = distributed_array<int, 100>;

    static_assert(std::is_same_v<typename arr_type::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename arr_type::placement_policy, default_placement>);
    static_assert(std::is_same_v<typename arr_type::consistency_policy, default_consistency>);
    static_assert(std::is_same_v<typename arr_type::execution_policy, default_execution>);
    static_assert(std::is_same_v<typename arr_type::error_policy, default_error>);
}

TEST(DistributedArrayTest, CustomPartitionPolicy) {
    using arr_type = distributed_array<int, 10, cyclic_partition<>>;

    static_assert(std::is_same_v<typename arr_type::partition_policy, cyclic_partition<>>);

    // With cyclic partition, ownership is round-robin
    arr_type arr(test_context{0, 4});

    // Rank 0 owns elements 0, 4, 8
    EXPECT_TRUE(arr.is_local(0));
    EXPECT_FALSE(arr.is_local(1));
    EXPECT_TRUE(arr.is_local(4));
    EXPECT_TRUE(arr.is_local(8));
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DistributedArrayTest, IsDistributedContainer) {
    static_assert(is_distributed_container_v<distributed_array<int, 100>>);
    static_assert(is_distributed_container_v<distributed_array<double, 50>>);
    static_assert(is_distributed_container_v<distributed_array<int, 100, cyclic_partition<>>>);

    static_assert(!is_distributed_container_v<std::array<int, 100>>);
    static_assert(!is_distributed_container_v<int>);
}

TEST(DistributedArrayTest, IsDistributedArray) {
    static_assert(is_distributed_array_v<distributed_array<int, 100>>);
    static_assert(is_distributed_array_v<distributed_array<double, 50>>);

    static_assert(!is_distributed_array_v<distributed_vector<int>>);
    static_assert(!is_distributed_array_v<std::array<int, 100>>);
    static_assert(!is_distributed_array_v<int>);
}

// =============================================================================
// Partition Map Access Tests
// =============================================================================

TEST(DistributedArrayTest, PartitionAccess) {
    distributed_array<int, 100> arr(test_context{2, 4});

    const auto& partition = arr.partition();
    EXPECT_EQ(partition.global_size(), 100);
    EXPECT_EQ(partition.local_size(), 25);
    EXPECT_EQ(partition.my_rank(), 2);
}

// =============================================================================
// Different Types Tests
// =============================================================================

TEST(DistributedArrayTest, DoubleType) {
    distributed_array<double, 50> arr(3.14, test_context{0, 2});

    auto local = arr.local_view();
    EXPECT_DOUBLE_EQ(local[0], 3.14);
}

TEST(DistributedArrayTest, StructType) {
    struct Point { int x, y; };
    distributed_array<Point, 20> arr(test_context{0, 2});

    arr.fill(Point{1, 2});

    auto local = arr.local_view();
    EXPECT_EQ(local[0].x, 1);
    EXPECT_EQ(local[0].y, 2);
}

}  // namespace dtl::test
