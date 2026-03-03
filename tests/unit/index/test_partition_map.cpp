// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_partition_map.cpp
/// @brief Unit tests for partition_map
/// @details Tests for Task 2.2.1: partition_map index translation

#include <dtl/index/partition_map.hpp>
#include <dtl/policies/partition/block_partition.hpp>
#include <dtl/policies/partition/cyclic_partition.hpp>
#include <dtl/policies/partition/replicated.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Block Partition Map Tests
// =============================================================================

TEST(PartitionMapTest, BlockPartitionConstructor) {
    partition_map<block_partition<>> map(100, 4, 0);

    EXPECT_EQ(map.global_size(), 100);
    EXPECT_EQ(map.num_ranks(), 4);
    EXPECT_EQ(map.my_rank(), 0);
}

TEST(PartitionMapTest, BlockPartitionLocalSizeEvenDivision) {
    // 100 elements / 4 ranks = 25 each
    partition_map<block_partition<>> map(100, 4, 0);

    EXPECT_EQ(map.local_size(), 25);
    EXPECT_EQ(map.local_size(0), 25);
    EXPECT_EQ(map.local_size(1), 25);
    EXPECT_EQ(map.local_size(2), 25);
    EXPECT_EQ(map.local_size(3), 25);
}

TEST(PartitionMapTest, BlockPartitionLocalSizeUnevenDivision) {
    // 10 elements / 4 ranks = 2, 2, 3, 3 (remainder goes to early ranks)
    partition_map<block_partition<>> map(10, 4, 0);

    // First 2 ranks get 3, last 2 get 2 (remainder = 2)
    EXPECT_EQ(map.local_size(0), 3);
    EXPECT_EQ(map.local_size(1), 3);
    EXPECT_EQ(map.local_size(2), 2);
    EXPECT_EQ(map.local_size(3), 2);

    // Verify sum equals global size
    size_type sum = 0;
    for (rank_t r = 0; r < 4; ++r) {
        sum += map.local_size(r);
    }
    EXPECT_EQ(sum, 10);
}

TEST(PartitionMapTest, BlockPartitionOwnership) {
    partition_map<block_partition<>> map(10, 4, 0);

    // Rank 0: [0, 3)
    EXPECT_EQ(map.owner(0), 0);
    EXPECT_EQ(map.owner(1), 0);
    EXPECT_EQ(map.owner(2), 0);

    // Rank 1: [3, 6)
    EXPECT_EQ(map.owner(3), 1);
    EXPECT_EQ(map.owner(4), 1);
    EXPECT_EQ(map.owner(5), 1);

    // Rank 2: [6, 8)
    EXPECT_EQ(map.owner(6), 2);
    EXPECT_EQ(map.owner(7), 2);

    // Rank 3: [8, 10)
    EXPECT_EQ(map.owner(8), 3);
    EXPECT_EQ(map.owner(9), 3);
}

TEST(PartitionMapTest, BlockPartitionIsLocal) {
    // Test from rank 1's perspective
    partition_map<block_partition<>> map(10, 4, 1);

    // Rank 1 owns [3, 6)
    EXPECT_FALSE(map.is_local(0));
    EXPECT_FALSE(map.is_local(2));
    EXPECT_TRUE(map.is_local(3));
    EXPECT_TRUE(map.is_local(4));
    EXPECT_TRUE(map.is_local(5));
    EXPECT_FALSE(map.is_local(6));
    EXPECT_FALSE(map.is_local(9));
}

TEST(PartitionMapTest, BlockPartitionToLocal) {
    partition_map<block_partition<>> map(10, 4, 1);

    // Rank 1 owns [3, 6), so:
    // global 3 -> local 0
    // global 4 -> local 1
    // global 5 -> local 2
    EXPECT_EQ(map.to_local(3), 0);
    EXPECT_EQ(map.to_local(4), 1);
    EXPECT_EQ(map.to_local(5), 2);
}

TEST(PartitionMapTest, BlockPartitionToGlobal) {
    partition_map<block_partition<>> map(10, 4, 1);

    // Rank 1's local indices map to:
    // local 0 -> global 3
    // local 1 -> global 4
    // local 2 -> global 5
    EXPECT_EQ(map.to_global(0), 3);
    EXPECT_EQ(map.to_global(1), 4);
    EXPECT_EQ(map.to_global(2), 5);
}

TEST(PartitionMapTest, BlockPartitionRoundtrip) {
    partition_map<block_partition<>> map(100, 4, 2);

    // For all local indices, roundtrip should work
    for (index_t local = 0; local < static_cast<index_t>(map.local_size()); ++local) {
        index_t global = map.to_global(local);
        EXPECT_TRUE(map.is_local(global));
        EXPECT_EQ(map.to_local(global), local);
    }
}

TEST(PartitionMapTest, BlockPartitionLocalOffset) {
    // Check offsets for each rank
    constexpr size_type global_size = 10;
    constexpr rank_t num_ranks = 4;

    partition_map<block_partition<>> map0(global_size, num_ranks, 0);
    partition_map<block_partition<>> map1(global_size, num_ranks, 1);
    partition_map<block_partition<>> map2(global_size, num_ranks, 2);
    partition_map<block_partition<>> map3(global_size, num_ranks, 3);

    EXPECT_EQ(map0.local_offset(), 0);
    EXPECT_EQ(map1.local_offset(), 3);  // 3 elements on rank 0
    EXPECT_EQ(map2.local_offset(), 6);  // 3+3 elements on ranks 0,1
    EXPECT_EQ(map3.local_offset(), 8);  // 3+3+2 elements on ranks 0,1,2
}

// =============================================================================
// Cyclic Partition Map Tests
// =============================================================================

TEST(PartitionMapTest, CyclicPartitionOwnership) {
    partition_map<cyclic_partition<>> map(10, 4, 0);

    // Element i is owned by rank (i % 4)
    EXPECT_EQ(map.owner(0), 0);
    EXPECT_EQ(map.owner(1), 1);
    EXPECT_EQ(map.owner(2), 2);
    EXPECT_EQ(map.owner(3), 3);
    EXPECT_EQ(map.owner(4), 0);
    EXPECT_EQ(map.owner(5), 1);
    EXPECT_EQ(map.owner(6), 2);
    EXPECT_EQ(map.owner(7), 3);
    EXPECT_EQ(map.owner(8), 0);
    EXPECT_EQ(map.owner(9), 1);
}

TEST(PartitionMapTest, CyclicPartitionIsLocal) {
    partition_map<cyclic_partition<>> map(10, 4, 2);

    // Rank 2 owns elements 2, 6
    EXPECT_FALSE(map.is_local(0));
    EXPECT_FALSE(map.is_local(1));
    EXPECT_TRUE(map.is_local(2));
    EXPECT_FALSE(map.is_local(3));
    EXPECT_FALSE(map.is_local(4));
    EXPECT_FALSE(map.is_local(5));
    EXPECT_TRUE(map.is_local(6));
    EXPECT_FALSE(map.is_local(7));
    EXPECT_FALSE(map.is_local(8));
    EXPECT_FALSE(map.is_local(9));
}

TEST(PartitionMapTest, CyclicPartitionLocalSize) {
    // 10 elements / 4 ranks: ranks 0,1 get 3 each, ranks 2,3 get 2 each
    partition_map<cyclic_partition<>> map(10, 4, 0);

    EXPECT_EQ(map.local_size(0), 3);  // elements 0, 4, 8
    EXPECT_EQ(map.local_size(1), 3);  // elements 1, 5, 9
    EXPECT_EQ(map.local_size(2), 2);  // elements 2, 6
    EXPECT_EQ(map.local_size(3), 2);  // elements 3, 7
}

TEST(PartitionMapTest, CyclicPartitionToLocalToGlobal) {
    partition_map<cyclic_partition<>> map(10, 4, 0);

    // Rank 0 owns elements 0, 4, 8
    // local 0 -> global 0
    // local 1 -> global 4
    // local 2 -> global 8
    EXPECT_EQ(map.to_global(0), 0);
    EXPECT_EQ(map.to_global(1), 4);
    EXPECT_EQ(map.to_global(2), 8);

    EXPECT_EQ(map.to_local(0), 0);
    EXPECT_EQ(map.to_local(4), 1);
    EXPECT_EQ(map.to_local(8), 2);
}

// =============================================================================
// Replicated Partition Map Tests
// =============================================================================

TEST(PartitionMapTest, ReplicatedLocalSizeEqualsGlobal) {
    partition_map<replicated> map(100, 4, 2);

    // All ranks have all elements
    EXPECT_EQ(map.local_size(), 100);
    EXPECT_EQ(map.local_size(0), 100);
    EXPECT_EQ(map.local_size(1), 100);
    EXPECT_EQ(map.local_size(2), 100);
    EXPECT_EQ(map.local_size(3), 100);
}

TEST(PartitionMapTest, ReplicatedAllLocal) {
    partition_map<replicated> map(100, 4, 2);

    // All elements are local on all ranks
    for (index_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(map.is_local(i));
    }
}

TEST(PartitionMapTest, ReplicatedIdentityMapping) {
    partition_map<replicated> map(100, 4, 2);

    // Global == Local for replicated
    for (index_t i = 0; i < 100; ++i) {
        EXPECT_EQ(map.to_local(i), i);
        EXPECT_EQ(map.to_global(i), i);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(PartitionMapTest, SingleRank) {
    partition_map<block_partition<>> map(100, 1, 0);

    EXPECT_EQ(map.local_size(), 100);
    EXPECT_EQ(map.local_offset(), 0);

    for (index_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(map.is_local(i));
        EXPECT_EQ(map.owner(i), 0);
        EXPECT_EQ(map.to_local(i), i);
        EXPECT_EQ(map.to_global(i), i);
    }
}

TEST(PartitionMapTest, EmptyContainer) {
    partition_map<block_partition<>> map(0, 4, 0);

    EXPECT_EQ(map.global_size(), 0);
    EXPECT_EQ(map.local_size(), 0);
    EXPECT_TRUE(map.empty());
}

TEST(PartitionMapTest, MoreRanksThanElements) {
    partition_map<block_partition<>> map(3, 10, 0);

    // Only first 3 ranks get 1 element each
    EXPECT_EQ(map.local_size(0), 1);
    EXPECT_EQ(map.local_size(1), 1);
    EXPECT_EQ(map.local_size(2), 1);
    EXPECT_EQ(map.local_size(3), 0);
    EXPECT_EQ(map.local_size(9), 0);
}

TEST(PartitionMapTest, ConstexprConstruction) {
    constexpr partition_map<block_partition<>> map(100, 4, 0);

    static_assert(map.global_size() == 100);
    static_assert(map.num_ranks() == 4);
    static_assert(map.my_rank() == 0);
    static_assert(map.local_size() == 25);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(PartitionMapTest, MakeBlockPartitionMap) {
    auto map = make_block_partition_map(100, 4, 1);

    EXPECT_EQ(map.global_size(), 100);
    EXPECT_EQ(map.local_size(), 25);
}

TEST(PartitionMapTest, MakeCyclicPartitionMap) {
    auto map = make_cyclic_partition_map(10, 4, 0);

    EXPECT_EQ(map.global_size(), 10);
    EXPECT_EQ(map.local_size(), 3);  // elements 0, 4, 8
}

}  // namespace dtl::test
