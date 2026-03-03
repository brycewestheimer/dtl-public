// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_partition_policies.cpp
/// @brief Unit tests for partition policies
/// @details Tests block_partition, cyclic_partition, and hash_partition.

#include <dtl/policies/partition/block_partition.hpp>
#include <dtl/policies/partition/cyclic_partition.hpp>
#include <dtl/policies/partition/partition_policy.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

namespace dtl::test {

// =============================================================================
// Block Partition Tests
// =============================================================================

TEST(BlockPartitionTest, ConceptSatisfaction) {
    // block_partition should satisfy PartitionPolicyConcept
    static_assert(PartitionPolicyConcept<block_partition<0>>);
    static_assert(PartitionPolicyConcept<block_partition<1>>);
}

TEST(BlockPartitionTest, PolicyCategory) {
    static_assert(std::is_same_v<typename block_partition<0>::policy_category, partition_policy_tag>);
}

TEST(BlockPartitionTest, OwnershipEvenDistribution) {
    // 12 elements across 4 ranks = 3 per rank
    constexpr size_type global_size = 12;
    constexpr rank_t num_ranks = 4;

    EXPECT_EQ(block_partition<0>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(1, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(2, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(3, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(4, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(5, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(6, global_size, num_ranks), 2);
    EXPECT_EQ(block_partition<0>::owner(9, global_size, num_ranks), 3);
    EXPECT_EQ(block_partition<0>::owner(11, global_size, num_ranks), 3);
}

TEST(BlockPartitionTest, OwnershipUnevenDistribution) {
    // 10 elements across 3 ranks = 4, 3, 3
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    // Rank 0: [0-3], Rank 1: [4-6], Rank 2: [7-9]
    EXPECT_EQ(block_partition<0>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(3, global_size, num_ranks), 0);
    EXPECT_EQ(block_partition<0>::owner(4, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(6, global_size, num_ranks), 1);
    EXPECT_EQ(block_partition<0>::owner(7, global_size, num_ranks), 2);
    EXPECT_EQ(block_partition<0>::owner(9, global_size, num_ranks), 2);
}

TEST(BlockPartitionTest, LocalSizeEven) {
    constexpr size_type global_size = 12;
    constexpr rank_t num_ranks = 4;

    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 0), 3);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 1), 3);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 2), 3);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 3), 3);
}

TEST(BlockPartitionTest, LocalSizeUneven) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    // 10 / 3 = 3 with remainder 1. First rank gets extra.
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 0), 4);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 1), 3);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 2), 3);
}

TEST(BlockPartitionTest, LocalStart) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    EXPECT_EQ(block_partition<0>::local_start(global_size, num_ranks, 0), 0);
    EXPECT_EQ(block_partition<0>::local_start(global_size, num_ranks, 1), 4);
    EXPECT_EQ(block_partition<0>::local_start(global_size, num_ranks, 2), 7);
}

TEST(BlockPartitionTest, IndexConversion) {
    constexpr size_type global_size = 12;
    constexpr rank_t num_ranks = 4;

    // Rank 2 has elements [6, 7, 8]
    EXPECT_EQ(block_partition<0>::to_local(6, global_size, num_ranks, 2), 0);
    EXPECT_EQ(block_partition<0>::to_local(7, global_size, num_ranks, 2), 1);
    EXPECT_EQ(block_partition<0>::to_local(8, global_size, num_ranks, 2), 2);

    EXPECT_EQ(block_partition<0>::to_global(0, global_size, num_ranks, 2), 6);
    EXPECT_EQ(block_partition<0>::to_global(1, global_size, num_ranks, 2), 7);
    EXPECT_EQ(block_partition<0>::to_global(2, global_size, num_ranks, 2), 8);
}

TEST(BlockPartitionTest, RoundTripConversion) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 7;

    for (rank_t rank = 0; rank < num_ranks; ++rank) {
        auto local_sz = block_partition<0>::local_size(global_size, num_ranks, rank);
        for (size_type local_idx = 0; local_idx < local_sz; ++local_idx) {
            auto global_idx = block_partition<0>::to_global(
                static_cast<index_t>(local_idx), global_size, num_ranks, rank);
            auto round_trip = block_partition<0>::to_local(global_idx, global_size, num_ranks, rank);
            EXPECT_EQ(round_trip, static_cast<index_t>(local_idx));
        }
    }
}

TEST(BlockPartitionTest, EmptyGlobalSize) {
    EXPECT_EQ(block_partition<0>::local_size(0, 4, 0), 0);
    EXPECT_EQ(block_partition<0>::local_size(0, 4, 3), 0);
    EXPECT_EQ(block_partition<0>::owner(0, 0, 4), 0);
}

TEST(BlockPartitionTest, MakePartitionInfo) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    auto info = block_partition<0>::make_info(global_size, num_ranks, 1);

    EXPECT_EQ(info.global_size, 10);
    EXPECT_EQ(info.num_ranks, 3);
    EXPECT_EQ(info.my_rank, 1);
    EXPECT_EQ(info.local_size, 3);
    EXPECT_EQ(info.local_start, 4);
    EXPECT_EQ(info.local_end, 7);
}

// =============================================================================
// Cyclic Partition Tests
// =============================================================================

TEST(CyclicPartitionTest, ConceptSatisfaction) {
    static_assert(PartitionPolicyConcept<cyclic_partition<0>>);
    static_assert(PartitionPolicyConcept<cyclic_partition<1>>);
    static_assert(PartitionPolicyConcept<cyclic_partition<4>>);
}

TEST(CyclicPartitionTest, PolicyCategory) {
    static_assert(std::is_same_v<typename cyclic_partition<0>::policy_category, partition_policy_tag>);
}

TEST(CyclicPartitionTest, OwnershipCycle1) {
    // Cycle size 1 = round-robin
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    // Element i is owned by rank i % 3
    EXPECT_EQ(cyclic_partition<1>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<1>::owner(1, global_size, num_ranks), 1);
    EXPECT_EQ(cyclic_partition<1>::owner(2, global_size, num_ranks), 2);
    EXPECT_EQ(cyclic_partition<1>::owner(3, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<1>::owner(4, global_size, num_ranks), 1);
    EXPECT_EQ(cyclic_partition<1>::owner(5, global_size, num_ranks), 2);
    EXPECT_EQ(cyclic_partition<1>::owner(6, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<1>::owner(9, global_size, num_ranks), 0);
}

TEST(CyclicPartitionTest, OwnershipCycle2) {
    // Cycle size 2 = 2 elements per rank before cycling
    constexpr size_type global_size = 12;
    constexpr rank_t num_ranks = 3;

    // Elements 0-1: rank 0, 2-3: rank 1, 4-5: rank 2, 6-7: rank 0, etc.
    EXPECT_EQ(cyclic_partition<2>::owner(0, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<2>::owner(1, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<2>::owner(2, global_size, num_ranks), 1);
    EXPECT_EQ(cyclic_partition<2>::owner(3, global_size, num_ranks), 1);
    EXPECT_EQ(cyclic_partition<2>::owner(4, global_size, num_ranks), 2);
    EXPECT_EQ(cyclic_partition<2>::owner(5, global_size, num_ranks), 2);
    EXPECT_EQ(cyclic_partition<2>::owner(6, global_size, num_ranks), 0);
    EXPECT_EQ(cyclic_partition<2>::owner(7, global_size, num_ranks), 0);
}

TEST(CyclicPartitionTest, LocalSizeCycle1) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    // 10 elements / 3 ranks: ranks 0 and 1 get 4, rank 2 gets 3
    // (round-robin: 0,1,2,0,1,2,0,1,2,0 -> rank 0: 4, rank 1: 3, rank 2: 3)
    EXPECT_EQ(cyclic_partition<1>::local_size(global_size, num_ranks, 0), 4);
    EXPECT_EQ(cyclic_partition<1>::local_size(global_size, num_ranks, 1), 3);
    EXPECT_EQ(cyclic_partition<1>::local_size(global_size, num_ranks, 2), 3);
}

TEST(CyclicPartitionTest, IndexConversionCycle1) {
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 3;

    // Rank 0 has elements 0, 3, 6, 9
    EXPECT_EQ(cyclic_partition<1>::to_local(0, global_size, num_ranks, 0), 0);
    EXPECT_EQ(cyclic_partition<1>::to_local(3, global_size, num_ranks, 0), 1);
    EXPECT_EQ(cyclic_partition<1>::to_local(6, global_size, num_ranks, 0), 2);
    EXPECT_EQ(cyclic_partition<1>::to_local(9, global_size, num_ranks, 0), 3);

    EXPECT_EQ(cyclic_partition<1>::to_global(0, global_size, num_ranks, 0), 0);
    EXPECT_EQ(cyclic_partition<1>::to_global(1, global_size, num_ranks, 0), 3);
    EXPECT_EQ(cyclic_partition<1>::to_global(2, global_size, num_ranks, 0), 6);
    EXPECT_EQ(cyclic_partition<1>::to_global(3, global_size, num_ranks, 0), 9);
}

TEST(CyclicPartitionTest, RoundTripConversionCycle1) {
    constexpr size_type global_size = 50;
    constexpr rank_t num_ranks = 7;

    for (rank_t rank = 0; rank < num_ranks; ++rank) {
        auto local_sz = cyclic_partition<1>::local_size(global_size, num_ranks, rank);
        for (size_type local_idx = 0; local_idx < local_sz; ++local_idx) {
            auto global_idx = cyclic_partition<1>::to_global(
                static_cast<index_t>(local_idx), global_size, num_ranks, rank);
            auto round_trip = cyclic_partition<1>::to_local(global_idx, global_size, num_ranks, rank);
            EXPECT_EQ(round_trip, static_cast<index_t>(local_idx));
        }
    }
}

// =============================================================================
// Partition Info Tests
// =============================================================================

TEST(PartitionInfoTest, IsLocalCheck) {
    partition_info info{};
    info.global_size = 12;
    info.num_ranks = 4;
    info.my_rank = 2;
    info.local_size = 3;
    info.local_start = 6;
    info.local_end = 9;

    EXPECT_FALSE(info.is_local(5));
    EXPECT_TRUE(info.is_local(6));
    EXPECT_TRUE(info.is_local(7));
    EXPECT_TRUE(info.is_local(8));
    EXPECT_FALSE(info.is_local(9));
    EXPECT_FALSE(info.is_local(10));
}

TEST(PartitionInfoTest, IndexConversion) {
    partition_info info{};
    info.local_start = 6;

    EXPECT_EQ(info.to_local_unchecked(6), 0);
    EXPECT_EQ(info.to_local_unchecked(7), 1);
    EXPECT_EQ(info.to_local_unchecked(8), 2);

    EXPECT_EQ(info.to_global(0), 6);
    EXPECT_EQ(info.to_global(1), 7);
    EXPECT_EQ(info.to_global(2), 8);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(PartitionEdgeCasesTest, SingleRank) {
    constexpr size_type global_size = 100;
    constexpr rank_t num_ranks = 1;

    // All elements belong to rank 0
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 0), 100);
    EXPECT_EQ(block_partition<0>::owner(50, global_size, num_ranks), 0);

    EXPECT_EQ(cyclic_partition<1>::local_size(global_size, num_ranks, 0), 100);
    EXPECT_EQ(cyclic_partition<1>::owner(50, global_size, num_ranks), 0);
}

TEST(PartitionEdgeCasesTest, MoreRanksThanElements) {
    constexpr size_type global_size = 3;
    constexpr rank_t num_ranks = 10;

    // Only first 3 ranks get 1 element each
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 0), 1);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 1), 1);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 2), 1);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 3), 0);
    EXPECT_EQ(block_partition<0>::local_size(global_size, num_ranks, 9), 0);
}

}  // namespace dtl::test
