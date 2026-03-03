// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_index_edge_cases.cpp
/// @brief Edge-case unit tests for the DTL index module
/// @details Phase 14 T06: partition_map boundary values, global/local index
///          translation roundtrips, md index construction/arithmetic.

#include <dtl/index/index.hpp>
#include <dtl/index/partition_map.hpp>
#include <dtl/containers/distributed_tensor.hpp>  // for row_major, column_major

#include <gtest/gtest.h>

#include <limits>

namespace dtl::test {

// =============================================================================
// partition_map Edge Cases
// =============================================================================

TEST(PartitionMapEdgeTest, SingleElementSingleRank) {
    auto map = make_block_partition_map(1, 1, 0);
    EXPECT_EQ(map.global_size(), 1u);
    EXPECT_EQ(map.local_size(), 1u);
    EXPECT_EQ(map.owner(0), 0);
    EXPECT_TRUE(map.is_local(0));
    EXPECT_EQ(map.to_local(0), 0);
    EXPECT_EQ(map.to_global(0), 0);
}

TEST(PartitionMapEdgeTest, SingleElementMultipleRanks) {
    // 1 element across 4 ranks: only rank 0 gets it
    auto map0 = make_block_partition_map(1, 4, 0);

    EXPECT_EQ(map0.local_size(), 1u);
    EXPECT_EQ(map0.local_size(1), 0u);
    EXPECT_EQ(map0.local_size(2), 0u);
    EXPECT_EQ(map0.local_size(3), 0u);
    EXPECT_TRUE(map0.is_local(0));

    auto map1 = make_block_partition_map(1, 4, 1);
    EXPECT_FALSE(map1.is_local(0));
}

TEST(PartitionMapEdgeTest, ZeroElements) {
    auto map = make_block_partition_map(0, 4, 0);
    EXPECT_EQ(map.global_size(), 0u);
    EXPECT_EQ(map.local_size(), 0u);
    EXPECT_TRUE(map.empty());
}

TEST(PartitionMapEdgeTest, UnevenDistribution) {
    // 10 elements across 3 ranks: 4, 3, 3
    auto map0 = make_block_partition_map(10, 3, 0);
    auto map1 = make_block_partition_map(10, 3, 1);
    auto map2 = make_block_partition_map(10, 3, 2);

    EXPECT_EQ(map0.local_size(), 4u);
    EXPECT_EQ(map1.local_size(), 3u);
    EXPECT_EQ(map2.local_size(), 3u);

    // Sum of local sizes equals global size
    EXPECT_EQ(map0.local_size() + map1.local_size() + map2.local_size(), 10u);
}

TEST(PartitionMapEdgeTest, BoundaryOwnership) {
    auto map = make_block_partition_map(100, 4, 0);
    // First element of each rank
    EXPECT_EQ(map.owner(0), 0);

    auto map1 = make_block_partition_map(100, 4, 1);
    // Check boundary between rank 0 and rank 1
    // With 100/4 = 25 elements per rank, rank 0 owns [0,25), rank 1 owns [25,50)
    EXPECT_EQ(map.owner(24), 0);
    EXPECT_EQ(map1.owner(25), 1);
    EXPECT_EQ(map.owner(99), 3);
}

TEST(PartitionMapEdgeTest, LocalStartEnd) {
    auto map = make_block_partition_map(100, 4, 1);
    EXPECT_EQ(map.local_start(), 25);
    EXPECT_EQ(map.local_end(), 50);
    EXPECT_EQ(map.local_size(), 25u);
}

TEST(PartitionMapEdgeTest, LocalOffset) {
    auto map = make_block_partition_map(100, 4, 2);
    EXPECT_EQ(map.local_offset(), 50);
}

TEST(PartitionMapEdgeTest, LocalOffsetForRank) {
    auto map = make_block_partition_map(100, 4, 0);
    EXPECT_EQ(map.local_offset(0), 0);
    EXPECT_EQ(map.local_offset(1), 25);
    EXPECT_EQ(map.local_offset(2), 50);
    EXPECT_EQ(map.local_offset(3), 75);
}

TEST(PartitionMapEdgeTest, InfoAccess) {
    auto map = make_block_partition_map(100, 4, 2);
    const auto& info = map.info();
    EXPECT_EQ(info.global_size, 100u);
    EXPECT_EQ(info.num_ranks, 4);
    EXPECT_EQ(info.my_rank, 2);
    EXPECT_EQ(info.local_size, 25u);
}

TEST(PartitionMapEdgeTest, NumRanksAndMyRank) {
    auto map = make_block_partition_map(100, 4, 3);
    EXPECT_EQ(map.num_ranks(), 4);
    EXPECT_EQ(map.my_rank(), 3);
}

// =============================================================================
// Index Translation Roundtrips
// =============================================================================

TEST(IndexTranslationTest, BlockToLocalToGlobalRoundtrip) {
    const index_t global_size = 100;
    const rank_t num_ranks = 4;

    for (rank_t r = 0; r < num_ranks; ++r) {
        auto map = make_block_partition_map(
            static_cast<size_type>(global_size), num_ranks, r);
        for (index_t i = 0; i < static_cast<index_t>(map.local_size()); ++i) {
            index_t global_idx = map.to_global(i);
            EXPECT_TRUE(map.is_local(global_idx));
            index_t local_back = map.to_local(global_idx);
            EXPECT_EQ(local_back, i) << "rank=" << r << " local=" << i;
        }
    }
}

TEST(IndexTranslationTest, BlockTranslationOwner) {
    using namespace block_partition_translation;
    // 10 elements, 3 ranks
    EXPECT_EQ(owner<index_t>(0, 10, 3), 0);
    EXPECT_EQ(owner<index_t>(3, 10, 3), 0);
    EXPECT_EQ(owner<index_t>(4, 10, 3), 1);
    EXPECT_EQ(owner<index_t>(7, 10, 3), 2);
    EXPECT_EQ(owner<index_t>(9, 10, 3), 2);
}

TEST(IndexTranslationTest, BlockTranslationInvalidIndex) {
    using namespace block_partition_translation;
    EXPECT_EQ(owner<index_t>(-1, 10, 3), no_rank);
    EXPECT_EQ(owner<index_t>(10, 10, 3), no_rank);
}

TEST(IndexTranslationTest, BlockTranslationLocalSize) {
    using namespace block_partition_translation;
    EXPECT_EQ(local_size<index_t>(10, 3, 0), 4);
    EXPECT_EQ(local_size<index_t>(10, 3, 1), 3);
    EXPECT_EQ(local_size<index_t>(10, 3, 2), 3);
}

TEST(IndexTranslationTest, BlockTranslationToLocal) {
    using namespace block_partition_translation;
    auto li = to_local<index_t>(4, 10, 3, 1);
    EXPECT_TRUE(li.valid());
    EXPECT_EQ(li.value(), 0);

    // Asking for wrong rank returns invalid
    auto bad = to_local<index_t>(0, 10, 3, 1);
    EXPECT_FALSE(bad.valid());
}

TEST(IndexTranslationTest, BlockTranslationToGlobal) {
    using namespace block_partition_translation;
    auto gi = to_global<index_t>(0, 10, 3, 1);
    EXPECT_TRUE(gi.valid());
    EXPECT_EQ(gi.value(), 4);

    // Out of range local index
    auto bad = to_global<index_t>(100, 10, 3, 1);
    EXPECT_FALSE(bad.valid());
}

// =============================================================================
// global_index Tests
// =============================================================================

TEST(GlobalIndexTest, DefaultIsInvalid) {
    global_index<> idx;
    EXPECT_FALSE(idx.valid());
    EXPECT_FALSE(static_cast<bool>(idx));
}

TEST(GlobalIndexTest, ConstructFromValue) {
    auto idx = make_global_index<index_t>(42);
    EXPECT_TRUE(idx.valid());
    EXPECT_EQ(idx.value(), 42);
}

TEST(GlobalIndexTest, Increment) {
    auto idx = make_global_index<index_t>(10);
    ++idx;
    EXPECT_EQ(idx.value(), 11);
    auto old = idx++;
    EXPECT_EQ(old.value(), 11);
    EXPECT_EQ(idx.value(), 12);
}

TEST(GlobalIndexTest, Decrement) {
    auto idx = make_global_index<index_t>(10);
    --idx;
    EXPECT_EQ(idx.value(), 9);
}

TEST(GlobalIndexTest, AddSubtract) {
    auto idx = make_global_index<index_t>(10);
    auto plus = idx + index_t{5};
    EXPECT_EQ(plus.value(), 15);
    auto minus = idx - index_t{3};
    EXPECT_EQ(minus.value(), 7);
}

TEST(GlobalIndexTest, Difference) {
    auto a = make_global_index<index_t>(20);
    auto b = make_global_index<index_t>(10);
    EXPECT_EQ(a - b, 10);
}

TEST(GlobalIndexTest, Comparison) {
    auto a = make_global_index<index_t>(5);
    auto b = make_global_index<index_t>(10);
    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_GT(b, a);
    EXPECT_GE(b, a);
    EXPECT_NE(a, b);
    EXPECT_EQ(a, a);
}

TEST(GlobalIndexTest, CompoundAssignment) {
    auto idx = make_global_index<index_t>(10);
    idx += index_t{5};
    EXPECT_EQ(idx.value(), 15);
    idx -= index_t{3};
    EXPECT_EQ(idx.value(), 12);
}

// =============================================================================
// local_index Tests
// =============================================================================

TEST(LocalIndexTest, DefaultIsInvalid) {
    local_index<> idx;
    EXPECT_FALSE(idx.valid());
}

TEST(LocalIndexTest, ConstructFromValue) {
    auto idx = make_local_index<index_t>(5);
    EXPECT_TRUE(idx.valid());
    EXPECT_EQ(idx.value(), 5);
}

TEST(LocalIndexTest, Arithmetic) {
    auto idx = make_local_index<index_t>(10);
    ++idx;
    EXPECT_EQ(idx.value(), 11);
    auto next = idx + index_t{5};
    EXPECT_EQ(next.value(), 16);
}

TEST(LocalIndexTest, Comparison) {
    auto a = make_local_index<index_t>(3);
    auto b = make_local_index<index_t>(7);
    EXPECT_LT(a, b);
    EXPECT_EQ(a, a);
    EXPECT_NE(a, b);
}

// =============================================================================
// md_global_index / nd_index Tests
// =============================================================================

TEST(MdGlobalIndexTest, DefaultConstruction) {
    md_global_index<3> idx;
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 0);
    EXPECT_EQ(idx[2], 0);
}

TEST(MdGlobalIndexTest, VariadicConstruction) {
    md_global_index<3> idx(10, 20, 30);
    EXPECT_EQ(idx[0], 10);
    EXPECT_EQ(idx[1], 20);
    EXPECT_EQ(idx[2], 30);
}

TEST(MdGlobalIndexTest, Rank) {
    EXPECT_EQ(md_global_index<2>::rank, 2u);
    EXPECT_EQ(md_global_index<4>::rank, 4u);
}

TEST(MdGlobalIndexTest, DataAccess) {
    md_global_index<2> idx(5, 10);
    const auto* data = idx.data();
    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 10);
}

TEST(NdIndexTest, DefaultConstruction) {
    nd_index<3> idx;
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 0);
    EXPECT_EQ(idx[2], 0);
}

TEST(NdIndexTest, VariadicConstruction) {
    nd_index<2> idx(3, 7);
    EXPECT_EQ(idx[0], 3);
    EXPECT_EQ(idx[1], 7);
}

TEST(NdIndexTest, Equality) {
    nd_index<2> a(1, 2);
    nd_index<2> b(1, 2);
    nd_index<2> c(1, 3);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(NdIndexTest, MutableAccess) {
    nd_index<2> idx;
    idx[0] = 5;
    idx[1] = 10;
    EXPECT_EQ(idx[0], 5);
    EXPECT_EQ(idx[1], 10);
}

TEST(NdIndexTest, RankQuery) {
    EXPECT_EQ(nd_index<1>::rank(), 1u);
    EXPECT_EQ(nd_index<4>::rank(), 4u);
}

// =============================================================================
// Layout Linearization Tests (from distributed_tensor.hpp)
// =============================================================================

TEST(RowMajorTest, Linearize2D) {
    nd_extent<2> extents = {3, 4};
    nd_index<2> idx(1, 2);
    auto linear = row_major::linearize(idx, extents);
    EXPECT_EQ(linear, 1 * 4 + 2);  // = 6
}

TEST(RowMajorTest, Delinearize2D) {
    nd_extent<2> extents = {3, 4};
    auto idx = row_major::delinearize<2>(6, extents);
    EXPECT_EQ(idx[0], 1);
    EXPECT_EQ(idx[1], 2);
}

TEST(RowMajorTest, LinearizeDelinearizeRoundtrip) {
    nd_extent<3> extents = {5, 6, 7};
    for (index_t i = 0; i < 5; ++i) {
        for (index_t j = 0; j < 6; ++j) {
            for (index_t k = 0; k < 7; ++k) {
                nd_index<3> idx(i, j, k);
                auto linear = row_major::linearize(idx, extents);
                auto back = row_major::delinearize<3>(linear, extents);
                EXPECT_EQ(back, idx);
            }
        }
    }
}

TEST(RowMajorTest, Size) {
    nd_extent<3> extents = {2, 3, 4};
    EXPECT_EQ(row_major::size(extents), 24u);
}

TEST(ColumnMajorTest, Linearize2D) {
    nd_extent<2> extents = {3, 4};
    nd_index<2> idx(1, 2);
    auto linear = column_major::linearize(idx, extents);
    EXPECT_EQ(linear, 2 * 3 + 1);  // = 7
}

TEST(ColumnMajorTest, DelinearizeRoundtrip) {
    nd_extent<2> extents = {3, 4};
    for (index_t i = 0; i < 3; ++i) {
        for (index_t j = 0; j < 4; ++j) {
            nd_index<2> idx(i, j);
            auto linear = column_major::linearize(idx, extents);
            auto back = column_major::delinearize<2>(linear, extents);
            EXPECT_EQ(back, idx);
        }
    }
}

}  // namespace dtl::test
