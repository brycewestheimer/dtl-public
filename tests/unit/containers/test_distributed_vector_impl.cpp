// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_vector_impl.cpp
/// @brief Unit tests for distributed_vector implementation
/// @details Tests for Task 2.5: distributed_vector

#include <dtl/containers/distributed_vector.hpp>

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

TEST(DistributedVectorTest, DefaultConstruction) {
    distributed_vector<int> vec;

    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.local_size(), 0);
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.num_ranks(), 1);
    EXPECT_EQ(vec.rank(), 0);
}

TEST(DistributedVectorTest, ConstructWithSize) {
    // 100 elements, 4 ranks, I'm rank 1
    distributed_vector<int> vec(100, test_context{1, 4});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.global_size(), 100);
    EXPECT_EQ(vec.local_size(), 25);
    EXPECT_EQ(vec.num_ranks(), 4);
    EXPECT_EQ(vec.rank(), 1);
}

TEST(DistributedVectorTest, ConstructWithSizeAndValue) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.local_size(), 25);

    // All local elements should be initialized to 42
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(DistributedVectorTest, ConstructSingleRank) {
    distributed_vector<int> vec(100, test_context{0, 1});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.local_size(), 100);  // All elements local
    EXPECT_EQ(vec.num_ranks(), 1);
}

TEST(DistributedVectorTest, FactoryCreate) {
    auto result = distributed_vector<int>::create(100, 4, 1);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 100);
}

// =============================================================================
// Size Query Tests
// =============================================================================

TEST(DistributedVectorTest, SizeQueries) {
    distributed_vector<int> vec(100, test_context{2, 4});

    EXPECT_EQ(vec.size(), 100);
    EXPECT_EQ(vec.global_size(), 100);
    EXPECT_EQ(vec.local_size(), 25);
    EXPECT_FALSE(vec.empty());
}

TEST(DistributedVectorTest, LocalSizeForRank) {
    // 10 elements / 4 ranks with remainder
    distributed_vector<int> vec(10, test_context{0, 4});

    // First 2 ranks get 3, last 2 get 2
    EXPECT_EQ(vec.local_size_for_rank(0), 3);
    EXPECT_EQ(vec.local_size_for_rank(1), 3);
    EXPECT_EQ(vec.local_size_for_rank(2), 2);
    EXPECT_EQ(vec.local_size_for_rank(3), 2);
}

// =============================================================================
// Local View Tests
// =============================================================================

TEST(DistributedVectorTest, LocalViewAccess) {
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

TEST(DistributedVectorTest, LocalViewSTLSort) {
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

TEST(DistributedVectorTest, LocalViewSTLAccumulate) {
    distributed_vector<int> vec(100, 1, test_context{0, 1});  // All 1s

    auto local = vec.local_view();
    int sum = std::accumulate(local.begin(), local.end(), 0);
    EXPECT_EQ(sum, 100);
}

// =============================================================================
// Index Translation Tests
// =============================================================================

TEST(DistributedVectorTest, IsLocal) {
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

TEST(DistributedVectorTest, Owner) {
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

TEST(DistributedVectorTest, ToLocalToGlobal) {
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

TEST(DistributedVectorTest, GlobalOffset) {
    // Rank 2 of 4 with 100 elements
    // Block partition: rank 2 offset is 50
    distributed_vector<int> vec(100, test_context{2, 4});

    EXPECT_EQ(vec.global_offset(), 50);
}

// =============================================================================
// Global View Tests
// =============================================================================

TEST(DistributedVectorTest, GlobalViewReturnsRemoteRef) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto global = vec.global_view();
    EXPECT_EQ(global.size(), 100);

    // Access returns remote_ref
    auto ref = global[50];  // This is a global index

    // The ref type should be remote_ref<int>
    static_assert(std::is_same_v<decltype(ref), remote_ref<int>>);
}

TEST(DistributedVectorTest, GlobalViewLocalAccess) {
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

TEST(DistributedVectorTest, GlobalViewRemoteAccess) {
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

TEST(DistributedVectorTest, SegmentedViewNumSegments) {
    distributed_vector<int> vec(100, test_context{1, 4});

    auto segmented = vec.segmented_view();
    EXPECT_EQ(segmented.num_segments(), 4);
}

TEST(DistributedVectorTest, SegmentedViewLocalSegment) {
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

TEST(DistributedVectorTest, Barrier) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.barrier();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedVectorTest, Fence) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.fence();
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Structural Operation Tests
// =============================================================================

TEST(DistributedVectorTest, Resize) {
    distributed_vector<int> vec(100, test_context{0, 1});
    EXPECT_EQ(vec.size(), 100);

    auto result = vec.resize(200);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.size(), 200);
    EXPECT_EQ(vec.local_size(), 200);
}

TEST(DistributedVectorTest, ResizeWithValue) {
    distributed_vector<int> vec(10, 0, test_context{0, 1});

    // Verify initial values
    EXPECT_EQ(vec.local(0), 0);

    // Resize with new value
    auto result = vec.resize(20, 99);
    EXPECT_TRUE(result.has_value());

    // New elements should be 99
    EXPECT_EQ(vec.local(10), 99);
    EXPECT_EQ(vec.local(19), 99);
}

TEST(DistributedVectorTest, Clear) {
    distributed_vector<int> vec(100, test_context{0, 1});

    auto result = vec.clear();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());
}

TEST(DistributedVectorTest, ResizeRequiresCollectivePathForMultiRank) {
    distributed_vector<int> vec(100, test_context{1, 4});
    auto result = vec.resize(200);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST(DistributedVectorTest, ClearRequiresCollectivePathForMultiRank) {
    distributed_vector<int> vec(100, test_context{1, 4});
    auto result = vec.clear();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

TEST(DistributedVectorTest, StructuralMetadataConsistentAfterConstruction) {
    distributed_vector<int> vec(64, test_context{1, 4});
    EXPECT_TRUE(vec.structural_metadata_consistent());
}

TEST(DistributedVectorTest, ReplaceLocalPartitionPreservesMetadata) {
    distributed_vector<int> vec(8, test_context{0, 1});
    distributed_vector<int>::storage_type replacement(vec.local_size(), 17);

    auto result = vec.replace_local_partition(std::move(replacement));
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(vec.structural_metadata_consistent());

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 17);
    }
}

TEST(DistributedVectorTest, ReplaceLocalPartitionRejectsSizeMismatch) {
    distributed_vector<int> vec(8, test_context{0, 1});
    distributed_vector<int>::storage_type replacement(vec.local_size() + 1, 3);

    auto result = vec.replace_local_partition(std::move(replacement));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_argument);
}

TEST(DistributedVectorTest, ReplaceLocalPartitionWithNewGlobalSizeUpdatesMetadata) {
    distributed_vector<int> vec(8, test_context{0, 1});
    distributed_vector<int>::storage_type replacement(5, 9);

    auto result = vec.replace_local_partition(std::move(replacement), 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(vec.size(), 5u);
    EXPECT_EQ(vec.local_size(), 5u);
    EXPECT_TRUE(vec.structural_metadata_consistent());
}

TEST(DistributedVectorTest, SwapLocalStorageRejectsSizeMismatch) {
    distributed_vector<int> vec(6, test_context{0, 1});
    distributed_vector<int>::storage_type replacement(vec.local_size() + 2, 1);

    auto result = vec.swap_local_storage(replacement);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_argument);
}

// =============================================================================
// Policy Tests
// =============================================================================

TEST(DistributedVectorTest, DefaultPolicies) {
    using vec_type = distributed_vector<int>;

    static_assert(std::is_same_v<typename vec_type::partition_policy, default_partition>);
    static_assert(std::is_same_v<typename vec_type::placement_policy, default_placement>);
    static_assert(std::is_same_v<typename vec_type::consistency_policy, default_consistency>);
    static_assert(std::is_same_v<typename vec_type::execution_policy, default_execution>);
    static_assert(std::is_same_v<typename vec_type::error_policy, default_error>);
}

TEST(DistributedVectorTest, CustomPartitionPolicy) {
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

TEST(DistributedVectorTest, IsDistributedContainer) {
    static_assert(is_distributed_container_v<distributed_vector<int>>);
    static_assert(is_distributed_container_v<distributed_vector<double>>);
    static_assert(is_distributed_container_v<distributed_vector<int, cyclic_partition<>>>);

    static_assert(!is_distributed_container_v<std::vector<int>>);
    static_assert(!is_distributed_container_v<int>);
}

// =============================================================================
// Partition Map Access Tests
// =============================================================================

TEST(DistributedVectorTest, PartitionAccess) {
    distributed_vector<int> vec(100, test_context{2, 4});

    const auto& partition = vec.partition();
    EXPECT_EQ(partition.global_size(), 100);
    EXPECT_EQ(partition.local_size(), 25);
    EXPECT_EQ(partition.my_rank(), 2);
}

}  // namespace dtl::test
