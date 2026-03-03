// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_index_translation.cpp
/// @brief Unit tests for index translation utilities
/// @details Tests for Phase 11.5: partition-agnostic index translation

#include <dtl/index/index_translation.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// Using the namespace shorthand for convenience
namespace block = block_partition_translation;
namespace cyclic = cyclic_partition_translation;

// =============================================================================
// Block Partition Owner Tests
// =============================================================================

TEST(IndexTranslationTest, BlockOwnerUniform) {
    // 100 elements across 4 ranks = 25 each
    for (index_t i = 0; i < 25; ++i) {
        EXPECT_EQ(block::owner(i, index_t{100}, 4), 0);
    }
    for (index_t i = 25; i < 50; ++i) {
        EXPECT_EQ(block::owner(i, index_t{100}, 4), 1);
    }
    for (index_t i = 50; i < 75; ++i) {
        EXPECT_EQ(block::owner(i, index_t{100}, 4), 2);
    }
    for (index_t i = 75; i < 100; ++i) {
        EXPECT_EQ(block::owner(i, index_t{100}, 4), 3);
    }
}

TEST(IndexTranslationTest, BlockOwnerWithRemainder) {
    // 10 elements across 4 ranks: ranks 0,1 get 3, ranks 2,3 get 2
    // Indices [0,3) -> rank 0
    // Indices [3,6) -> rank 1
    // Indices [6,8) -> rank 2
    // Indices [8,10) -> rank 3
    EXPECT_EQ(block::owner(index_t{0}, index_t{10}, 4), 0);
    EXPECT_EQ(block::owner(index_t{2}, index_t{10}, 4), 0);
    EXPECT_EQ(block::owner(index_t{3}, index_t{10}, 4), 1);
    EXPECT_EQ(block::owner(index_t{5}, index_t{10}, 4), 1);
    EXPECT_EQ(block::owner(index_t{6}, index_t{10}, 4), 2);
    EXPECT_EQ(block::owner(index_t{7}, index_t{10}, 4), 2);
    EXPECT_EQ(block::owner(index_t{8}, index_t{10}, 4), 3);
    EXPECT_EQ(block::owner(index_t{9}, index_t{10}, 4), 3);
}

TEST(IndexTranslationTest, BlockOwnerSingleRank) {
    // Single rank owns all elements
    for (index_t i = 0; i < 100; ++i) {
        EXPECT_EQ(block::owner(i, index_t{100}, 1), 0);
    }
}

TEST(IndexTranslationTest, BlockOwnerInvalidIndex) {
    EXPECT_EQ(block::owner(index_t{-1}, index_t{100}, 4), no_rank);
    EXPECT_EQ(block::owner(index_t{100}, index_t{100}, 4), no_rank);
}

// =============================================================================
// Block Partition Local Size Tests
// =============================================================================

TEST(IndexTranslationTest, BlockLocalSizeUniform) {
    // 100 elements across 4 ranks = 25 each
    for (rank_t r = 0; r < 4; ++r) {
        EXPECT_EQ(block::local_size(index_t{100}, 4, r), 25);
    }
}

TEST(IndexTranslationTest, BlockLocalSizeWithRemainder) {
    // 10 elements across 4 ranks: 10 = 4*2 + 2, so ranks 0,1 get 3, ranks 2,3 get 2
    EXPECT_EQ(block::local_size(index_t{10}, 4, 0), 3);
    EXPECT_EQ(block::local_size(index_t{10}, 4, 1), 3);
    EXPECT_EQ(block::local_size(index_t{10}, 4, 2), 2);
    EXPECT_EQ(block::local_size(index_t{10}, 4, 3), 2);
}

TEST(IndexTranslationTest, BlockLocalSizeZeroElements) {
    EXPECT_EQ(block::local_size(index_t{0}, 4, 0), 0);
    EXPECT_EQ(block::local_size(index_t{0}, 4, 3), 0);
}

// =============================================================================
// Block Partition Offset Tests
// =============================================================================

TEST(IndexTranslationTest, BlockRankOffset) {
    // 100 elements across 4 ranks = offset of 0, 25, 50, 75
    EXPECT_EQ(block::rank_offset(index_t{100}, 4, 0), 0);
    EXPECT_EQ(block::rank_offset(index_t{100}, 4, 1), 25);
    EXPECT_EQ(block::rank_offset(index_t{100}, 4, 2), 50);
    EXPECT_EQ(block::rank_offset(index_t{100}, 4, 3), 75);
}

TEST(IndexTranslationTest, BlockRankOffsetWithRemainder) {
    // 10 elements across 4 ranks
    EXPECT_EQ(block::rank_offset(index_t{10}, 4, 0), 0);
    EXPECT_EQ(block::rank_offset(index_t{10}, 4, 1), 3);
    EXPECT_EQ(block::rank_offset(index_t{10}, 4, 2), 6);
    EXPECT_EQ(block::rank_offset(index_t{10}, 4, 3), 8);
}

// =============================================================================
// Block Partition to_local Tests
// =============================================================================

TEST(IndexTranslationTest, BlockToLocal) {
    // Rank 2 owns [50, 75)
    auto result = block::to_local(index_t{50}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 0);

    result = block::to_local(index_t{51}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 1);

    result = block::to_local(index_t{74}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 24);
}

TEST(IndexTranslationTest, BlockToLocalWrongOwner) {
    // Index 0 is owned by rank 0, not rank 1
    auto result = block::to_local(index_t{0}, index_t{100}, 4, 1);
    EXPECT_FALSE(result.valid());
}

// =============================================================================
// Block Partition to_global Tests
// =============================================================================

TEST(IndexTranslationTest, BlockToGlobal) {
    // Rank 2 owns [50, 75)
    auto result = block::to_global(index_t{0}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 50);

    result = block::to_global(index_t{1}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 51);

    result = block::to_global(index_t{24}, index_t{100}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 74);
}

TEST(IndexTranslationTest, BlockToGlobalOutOfRange) {
    // Rank 2 only has 25 elements
    auto result = block::to_global(index_t{25}, index_t{100}, 4, 2);
    EXPECT_FALSE(result.valid());
}

// =============================================================================
// Block Partition Roundtrip Tests
// =============================================================================

TEST(IndexTranslationTest, BlockRoundtrip) {
    // Test roundtrip for all ranks
    for (rank_t rank = 0; rank < 4; ++rank) {
        index_t local_sz = block::local_size(index_t{100}, 4, rank);
        for (index_t local = 0; local < local_sz; ++local) {
            auto global = block::to_global(local, index_t{100}, 4, rank);
            EXPECT_TRUE(global.valid());
            auto back = block::to_local(global.value(), index_t{100}, 4, rank);
            EXPECT_TRUE(back.valid());
            EXPECT_EQ(back.value(), local);
        }
    }
}

// =============================================================================
// Block Partition is_local Tests
// =============================================================================

TEST(IndexTranslationTest, BlockIsLocal) {
    // Rank 1 owns [25, 50)
    EXPECT_FALSE(block::is_local(index_t{24}, index_t{100}, 4, 1));
    EXPECT_TRUE(block::is_local(index_t{25}, index_t{100}, 4, 1));
    EXPECT_TRUE(block::is_local(index_t{49}, index_t{100}, 4, 1));
    EXPECT_FALSE(block::is_local(index_t{50}, index_t{100}, 4, 1));
}

// =============================================================================
// Block Partition Owned Range Tests
// =============================================================================

TEST(IndexTranslationTest, BlockOwnedRange) {
    auto range = block::owned_range(index_t{100}, 4, 2);
    EXPECT_EQ(*range.begin(), 50);
    EXPECT_EQ(*range.end(), 75);
}

// =============================================================================
// Cyclic Partition Owner Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicOwner) {
    // Cyclic: index % num_ranks
    for (index_t i = 0; i < 12; ++i) {
        rank_t expected = static_cast<rank_t>(i % 4);
        EXPECT_EQ(cyclic::owner(i, 4), expected);
    }
}

TEST(IndexTranslationTest, CyclicOwnerInvalid) {
    EXPECT_EQ(cyclic::owner(index_t{-1}, 4), no_rank);
}

// =============================================================================
// Cyclic Partition Local Size Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicLocalSize) {
    // 10 elements across 4 ranks: ranks 0,1 get 3, ranks 2,3 get 2
    EXPECT_EQ(cyclic::local_size(index_t{10}, 4, 0), 3);
    EXPECT_EQ(cyclic::local_size(index_t{10}, 4, 1), 3);
    EXPECT_EQ(cyclic::local_size(index_t{10}, 4, 2), 2);
    EXPECT_EQ(cyclic::local_size(index_t{10}, 4, 3), 2);
}

TEST(IndexTranslationTest, CyclicLocalSizeUniform) {
    // 100 elements across 4 ranks = 25 each
    for (rank_t r = 0; r < 4; ++r) {
        EXPECT_EQ(cyclic::local_size(index_t{100}, 4, r), 25);
    }
}

// =============================================================================
// Cyclic Partition to_local Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicToLocal) {
    // Rank 2 owns indices 2, 6, 10, 14, ...
    auto result = cyclic::to_local(index_t{2}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 0);

    result = cyclic::to_local(index_t{6}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 1);

    result = cyclic::to_local(index_t{10}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 2);
}

TEST(IndexTranslationTest, CyclicToLocalWrongOwner) {
    // Index 0 is owned by rank 0, not rank 1
    auto result = cyclic::to_local(index_t{0}, 4, 1);
    EXPECT_FALSE(result.valid());
}

// =============================================================================
// Cyclic Partition to_global Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicToGlobal) {
    // Rank 2 owns indices 2, 6, 10, ...
    auto result = cyclic::to_global(index_t{0}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 2);

    result = cyclic::to_global(index_t{1}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 6);

    result = cyclic::to_global(index_t{2}, 4, 2);
    EXPECT_TRUE(result.valid());
    EXPECT_EQ(result.value(), 10);
}

// =============================================================================
// Cyclic Partition Roundtrip Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicRoundtrip) {
    // Test roundtrip for rank 1
    for (index_t local = 0; local < 5; ++local) {
        auto global = cyclic::to_global(local, 4, 1);
        EXPECT_TRUE(global.valid());
        auto back = cyclic::to_local(global.value(), 4, 1);
        EXPECT_TRUE(back.valid());
        EXPECT_EQ(back.value(), local);
    }
}

// =============================================================================
// Cyclic Partition is_local Tests
// =============================================================================

TEST(IndexTranslationTest, CyclicIsLocal) {
    // Rank 1 owns indices 1, 5, 9, 13, ...
    EXPECT_FALSE(cyclic::is_local(index_t{0}, 4, 1));
    EXPECT_TRUE(cyclic::is_local(index_t{1}, 4, 1));
    EXPECT_FALSE(cyclic::is_local(index_t{2}, 4, 1));
    EXPECT_TRUE(cyclic::is_local(index_t{5}, 4, 1));
}

// =============================================================================
// Global Index Range Tests
// =============================================================================

TEST(IndexTranslationTest, GlobalIndexRangeIteration) {
    auto range = block::owned_range(index_t{100}, 4, 1);

    // Range should be [25, 50)
    size_type count = 0;
    for (auto it = range.begin(); it != range.end(); ++it) {
        EXPECT_GE(*it, 25);
        EXPECT_LT(*it, 50);
        ++count;
    }
    EXPECT_EQ(count, 25);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(IndexTranslationTest, SingleElementSingleRank) {
    EXPECT_EQ(block::owner(index_t{0}, index_t{1}, 1), 0);
    EXPECT_EQ(block::local_size(index_t{1}, 1, 0), 1);
}

TEST(IndexTranslationTest, MoreRanksThanElements) {
    // 2 elements across 4 ranks
    EXPECT_EQ(block::local_size(index_t{2}, 4, 0), 1);
    EXPECT_EQ(block::local_size(index_t{2}, 4, 1), 1);
    EXPECT_EQ(block::local_size(index_t{2}, 4, 2), 0);
    EXPECT_EQ(block::local_size(index_t{2}, 4, 3), 0);
}

TEST(IndexTranslationTest, ZeroRanks) {
    // Should handle gracefully
    EXPECT_EQ(block::local_size(index_t{100}, 0, 0), 0);
}

}  // namespace dtl::test
