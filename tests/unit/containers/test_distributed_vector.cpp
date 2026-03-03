// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_vector.cpp
/// @brief Unit tests for distributed_vector
/// @details Tests for Phase 11.5: distributed_vector container implementation

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/core/types.hpp>
#include <dtl/policies/policies.hpp>

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

TEST(DistributedVectorExtTest, DefaultConstruction) {
    distributed_vector<int> vec;

    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.num_ranks(), 1);
    EXPECT_EQ(vec.rank(), 0);
}

TEST(DistributedVectorExtTest, SizeRankConstruction) {
    distributed_vector<int> vec(100, test_context{0, 4});  // size=100, ranks=4, rank=0

    EXPECT_EQ(vec.global_size(), 100);
    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.num_ranks(), 4);
    EXPECT_EQ(vec.rank(), 0);
    EXPECT_EQ(vec.local_size(), 25);  // 100 / 4 = 25
}

TEST(DistributedVectorExtTest, ConstructWithValue) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    EXPECT_EQ(vec.global_size(), 100);
    EXPECT_EQ(vec.local_size(), 25);

    // All local elements should be initialized to 42
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(DistributedVectorExtTest, ConstructSingleRank) {
    distributed_vector<int> vec(100, test_context{0, 1});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.local_size(), 100);  // All elements local
    EXPECT_EQ(vec.num_ranks(), 1);
}

TEST(DistributedVectorExtTest, FactoryCreate) {
    auto result = distributed_vector<int>::create(100, 4, 1);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 100);
    EXPECT_EQ(result.value().num_ranks(), 4);
    EXPECT_EQ(result.value().rank(), 1);
}

TEST(DistributedVectorExtTest, EmptyVector) {
    distributed_vector<int> vec(0, test_context{0, 4});

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.local_size(), 0);
}

// =============================================================================
// Size Query Tests
// =============================================================================

TEST(DistributedVectorExtTest, SizeQueries) {
    distributed_vector<int> vec(100, test_context{2, 4});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.global_size(), 100);
    EXPECT_EQ(vec.local_size(), 25);
    EXPECT_FALSE(vec.empty());
    EXPECT_GT(vec.max_size(), 0);
}

TEST(DistributedVectorExtTest, LocalSizeForRank) {
    // 10 elements / 4 ranks with remainder
    distributed_vector<int> vec(10, test_context{0, 4});

    // First 2 ranks get 3, last 2 get 2
    EXPECT_EQ(vec.local_size_for_rank(0), 3);
    EXPECT_EQ(vec.local_size_for_rank(1), 3);
    EXPECT_EQ(vec.local_size_for_rank(2), 2);
    EXPECT_EQ(vec.local_size_for_rank(3), 2);
}

TEST(DistributedVectorExtTest, LocalSizeWithRemainder) {
    // 103 elements across 4 ranks
    distributed_vector<int> vec(103, test_context{0, 4});

    // 103 / 4 = 25 remainder 3
    // Ranks 0,1,2 get 26, rank 3 gets 25
    EXPECT_EQ(vec.local_size_for_rank(0), 26);
    EXPECT_EQ(vec.local_size_for_rank(1), 26);
    EXPECT_EQ(vec.local_size_for_rank(2), 26);
    EXPECT_EQ(vec.local_size_for_rank(3), 25);
}

// =============================================================================
// Local View Tests
// =============================================================================

TEST(DistributedVectorExtTest, LocalViewAccess) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 25);
    EXPECT_EQ(local.data(), vec.local_data());

    // Write through view
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i * 10);
    }

    // Verify through direct access
    EXPECT_EQ(vec.local(0), 0);
    EXPECT_EQ(vec.local(1), 10);
    EXPECT_EQ(vec.local(24), 240);
}

TEST(DistributedVectorExtTest, LocalViewSTLSort) {
    distributed_vector<int> vec(100, test_context{0, 1});

    // Fill with descending values
    auto local = vec.local_view();
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

TEST(DistributedVectorExtTest, LocalViewSTLAccumulate) {
    distributed_vector<int> vec(100, 1, test_context{0, 1});  // All 1s

    auto local = vec.local_view();
    int sum = std::accumulate(local.begin(), local.end(), 0);
    EXPECT_EQ(sum, 100);
}

TEST(DistributedVectorExtTest, LocalViewConstAccess) {
    const distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 25);
    EXPECT_EQ(local[0], 42);
}

// =============================================================================
// Index Translation Tests
// =============================================================================

TEST(DistributedVectorExtTest, IsLocal) {
    // Rank 1 of 4 with 10 elements
    // Block partition: rank 1 owns [3, 6)
    distributed_vector<int> vec(10, test_context{1, 4});

    EXPECT_FALSE(vec.is_local(0));
    EXPECT_FALSE(vec.is_local(2));
    EXPECT_TRUE(vec.is_local(3));
    EXPECT_TRUE(vec.is_local(4));
    EXPECT_TRUE(vec.is_local(5));
    EXPECT_FALSE(vec.is_local(6));
    EXPECT_FALSE(vec.is_local(9));
}

TEST(DistributedVectorExtTest, Owner) {
    distributed_vector<int> vec(10, test_context{0, 4});

    EXPECT_EQ(vec.owner(0), 0);
    EXPECT_EQ(vec.owner(2), 0);
    EXPECT_EQ(vec.owner(3), 1);
    EXPECT_EQ(vec.owner(5), 1);
    EXPECT_EQ(vec.owner(6), 2);
    EXPECT_EQ(vec.owner(7), 2);
    EXPECT_EQ(vec.owner(8), 3);
    EXPECT_EQ(vec.owner(9), 3);
}

TEST(DistributedVectorExtTest, ToLocalToGlobal) {
    distributed_vector<int> vec(100, test_context{2, 4});  // Rank 2

    // For rank 2, local index 0 should be global index 50
    index_t local0_global = vec.to_global(0);
    EXPECT_TRUE(vec.is_local(local0_global));
    EXPECT_EQ(vec.to_local(local0_global), 0);

    // Roundtrip for all local indices
    for (index_t local = 0; local < static_cast<index_t>(vec.local_size()); ++local) {
        index_t global = vec.to_global(local);
        EXPECT_EQ(vec.to_local(global), local);
    }
}

TEST(DistributedVectorExtTest, GlobalOffset) {
    // Rank 2 of 4 with 100 elements
    // Block partition: rank 2 offset is 50
    distributed_vector<int> vec(100, test_context{2, 4});

    EXPECT_EQ(vec.global_offset(), 50);
}

// =============================================================================
// Global View Tests
// =============================================================================

TEST(DistributedVectorExtTest, GlobalViewReturnsRemoteRef) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto global = vec.global_view();
    EXPECT_EQ(global.size(), 100);

    // Access returns remote_ref
    auto ref = global[50];  // This is a global index

    // The ref type should be remote_ref<int>
    static_assert(std::is_same_v<decltype(ref), remote_ref<int>>);
}

TEST(DistributedVectorExtTest, GlobalViewLocalAccess) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto global = vec.global_view();

    // Access a local element (rank 1 owns [25, 50))
    auto ref = global[25];
    EXPECT_TRUE(ref.is_local());
    EXPECT_EQ(ref.owner_rank(), 1);

    // Get the value
    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(DistributedVectorExtTest, GlobalViewRemoteAccess) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto global = vec.global_view();

    // Access a remote element (rank 0 owns [0, 25))
    auto ref = global[0];
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
    EXPECT_EQ(ref.owner_rank(), 0);
}

// =============================================================================
// Segmented View Tests
// =============================================================================

TEST(DistributedVectorExtTest, SegmentedViewNumSegments) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto segmented = vec.segmented_view();
    EXPECT_EQ(segmented.num_segments(), 4);
}

TEST(DistributedVectorExtTest, SegmentedViewLocalSegment) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto segmented = vec.segmented_view();
    auto local_seg = segmented.local_segment();

    EXPECT_TRUE(local_seg.is_local());
    EXPECT_EQ(local_seg.rank, 1);
    EXPECT_EQ(local_seg.size(), 25);
    EXPECT_EQ(local_seg[0], 42);
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST(DistributedVectorExtTest, Barrier) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.barrier();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedVectorExtTest, Fence) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.fence();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedVectorExtTest, Sync) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.sync();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(vec.is_clean());
}

// =============================================================================
// Sync State Tests
// =============================================================================

TEST(DistributedVectorExtTest, SyncStateInitiallyClean) {
    distributed_vector<int> vec(100, test_context{1, 4});

    EXPECT_TRUE(vec.is_clean());
    EXPECT_FALSE(vec.is_dirty());
}

TEST(DistributedVectorExtTest, SyncStateMarking) {
    distributed_vector<int> vec(100, test_context{1, 4});

    vec.mark_local_modified();
    EXPECT_TRUE(vec.is_dirty());
    EXPECT_FALSE(vec.is_clean());

    vec.mark_clean();
    EXPECT_TRUE(vec.is_clean());
    EXPECT_FALSE(vec.is_dirty());
}

TEST(DistributedVectorExtTest, SyncStateReference) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto& state = vec.sync_state_ref();
    state.mark_local_modified();
    EXPECT_TRUE(vec.is_dirty());
}

// =============================================================================
// Resize Tests
// =============================================================================

TEST(DistributedVectorExtTest, Resize) {
    distributed_vector<int> vec(100, 42, test_context{0, 1});

    auto result = vec.resize(200);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.global_size(), 200);
    EXPECT_EQ(vec.local_size(), 200);
}

TEST(DistributedVectorExtTest, ResizeWithValue) {
    distributed_vector<int> vec(100, 42, test_context{0, 1});

    auto result = vec.resize(200, 99);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.global_size(), 200);

    // New elements should have value 99
    auto local = vec.local_view();
    EXPECT_EQ(local[local.size() - 1], 99);
}

TEST(DistributedVectorExtTest, ResizeShrink) {
    distributed_vector<int> vec(100, 42, test_context{0, 1});

    auto result = vec.resize(40);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.global_size(), 40);
    EXPECT_EQ(vec.local_size(), 40);
}

TEST(DistributedVectorExtTest, Clear) {
    distributed_vector<int> vec(100, 42, test_context{0, 1});

    auto result = vec.clear();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.local_size(), 0);
}

TEST(DistributedVectorExtTest, ResizeFailsWithoutCollectivePathInMultiRank) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto result = vec.resize(200);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST(DistributedVectorExtTest, ClearFailsWithoutCollectivePathInMultiRank) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto result = vec.clear();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

// =============================================================================
// Partition Map Access Tests
// =============================================================================

TEST(DistributedVectorExtTest, PartitionAccess) {
    distributed_vector<int> vec(100, test_context{2, 4});

    const auto& partition = vec.partition();
    EXPECT_EQ(partition.global_size(), 100);
    EXPECT_EQ(partition.local_size(), 25);
    EXPECT_EQ(partition.my_rank(), 2);
}

// =============================================================================
// Policy Tests
// =============================================================================

TEST(DistributedVectorExtTest, DefaultPolicies) {
    using vec_type = distributed_vector<int>;

    static_assert(std::is_same_v<typename vec_type::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename vec_type::placement_policy, default_placement>);
    static_assert(std::is_same_v<typename vec_type::consistency_policy, default_consistency>);
    static_assert(std::is_same_v<typename vec_type::execution_policy, default_execution>);
    static_assert(std::is_same_v<typename vec_type::error_policy, default_error>);
}

TEST(DistributedVectorExtTest, CustomPartitionPolicy) {
    using vec_type = distributed_vector<int, cyclic_partition<>>;

    static_assert(std::is_same_v<typename vec_type::partition_policy, cyclic_partition<>>);

    // With cyclic partition, ownership is round-robin
    vec_type vec(10, test_context{0, 4});

    // Rank 0 owns elements 0, 4, 8
    EXPECT_TRUE(vec.is_local(0));
    EXPECT_FALSE(vec.is_local(1));
    EXPECT_TRUE(vec.is_local(4));
    EXPECT_TRUE(vec.is_local(8));
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DistributedVectorExtTest, IsDistributedContainer) {
    static_assert(is_distributed_container_v<distributed_vector<int>>);
    static_assert(is_distributed_container_v<distributed_vector<double>>);
    static_assert(is_distributed_container_v<distributed_vector<int, cyclic_partition<>>>);

    static_assert(!is_distributed_container_v<std::vector<int>>);
    static_assert(!is_distributed_container_v<int>);
}

TEST(DistributedVectorExtTest, IsDistributedVector) {
    static_assert(is_distributed_vector_v<distributed_vector<int>>);
    static_assert(is_distributed_vector_v<distributed_vector<double>>);

    static_assert(!is_distributed_vector_v<std::vector<int>>);
    static_assert(!is_distributed_vector_v<int>);
}

// =============================================================================
// Placement Policy Tests
// =============================================================================

TEST(DistributedVectorExtTest, HostAccessible) {
    using vec_type = distributed_vector<int>;

    EXPECT_TRUE(vec_type::is_host_accessible());
}

// =============================================================================
// Different Types Tests
// =============================================================================

TEST(DistributedVectorExtTest, DoubleType) {
    distributed_vector<double> vec(50, 3.14, test_context{0, 2});

    auto local = vec.local_view();
    EXPECT_DOUBLE_EQ(local[0], 3.14);
}

TEST(DistributedVectorExtTest, StructType) {
    struct Point { int x, y; };
    distributed_vector<Point> vec(20, test_context{0, 2});

    auto local = vec.local_view();
    local[0] = Point{1, 2};

    EXPECT_EQ(vec.local(0).x, 1);
    EXPECT_EQ(vec.local(0).y, 2);
}

TEST(DistributedVectorExtTest, LargeVector) {
    // Test with a larger vector to verify partition logic
    distributed_vector<int> vec(10000, test_context{3, 8});

    // Each rank should have approximately 1250 elements
    EXPECT_GE(vec.local_size(), 1250);
    EXPECT_LE(vec.local_size(), 1250 + 1);

    // Verify ownership calculation
    size_type total = 0;
    for (rank_t r = 0; r < 8; ++r) {
        total += vec.local_size_for_rank(r);
    }
    EXPECT_EQ(total, 10000);
}

// =============================================================================
// Const View Access Tests (R1.3 regression)
// =============================================================================

TEST(DistributedVectorExtTest, ConstGlobalViewAccess) {
    const distributed_vector<int> vec(100, 42, test_context{1, 4});

    // const global_view should work without const_cast
    auto global = vec.global_view();
    EXPECT_EQ(global.size(), 100);

    // Access a local element through const global view
    auto ref = global[25];
    EXPECT_TRUE(ref.is_local());
    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(DistributedVectorExtTest, ConstSegmentedViewAccess) {
    const distributed_vector<int> vec(100, 42, test_context{1, 4});

    // const segmented_view should work without const_cast
    auto segmented = vec.segmented_view();
    EXPECT_EQ(segmented.num_segments(), 4);

    auto local_seg = segmented.local_segment();
    EXPECT_TRUE(local_seg.is_local());
    EXPECT_EQ(local_seg.size(), 25);
    EXPECT_EQ(local_seg[0], 42);
}

TEST(DistributedVectorExtTest, ConstViewTypeTraits) {
    using vec_t = distributed_vector<int>;
    using const_gv = typename vec_t::const_global_view_type;
    using const_sv = typename vec_t::const_segmented_view_type;

    // Verify const view types are parameterized on const container
    static_assert(std::is_same_v<const_gv, global_view<const vec_t>>);
    static_assert(std::is_same_v<const_sv, segmented_view<const vec_t>>);
}

}  // namespace dtl::test
