// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_block_nd_partition.cpp
/// @brief Unit tests for block_nd_partition
/// @details Phase R7: N-dimensional block decomposition

#include <dtl/policies/partition/block_nd_partition.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <numeric>

namespace dtl::test {

// =============================================================================
// 2D Decomposition Tests
// =============================================================================

TEST(BlockNdPartitionTest, Construction2D) {
    block_2d_partition part({{2, 3}});

    EXPECT_EQ(part.proc_grid_dim(0), 2);
    EXPECT_EQ(part.proc_grid_dim(1), 3);
    EXPECT_EQ(part.total_ranks(), 6);
}

TEST(BlockNdPartitionTest, DefaultConstruction) {
    block_nd_partition<2> part;

    EXPECT_EQ(part.proc_grid_dim(0), 1);
    EXPECT_EQ(part.proc_grid_dim(1), 1);
    EXPECT_EQ(part.total_ranks(), 1);
}

TEST(BlockNdPartitionTest, RankToGrid2D) {
    // 2x3 process grid
    block_2d_partition part({{2, 3}});

    // Row-major ordering: rank = g[0] * 3 + g[1]
    auto c0 = part.rank_to_grid(0);  // (0, 0)
    EXPECT_EQ(c0[0], 0);
    EXPECT_EQ(c0[1], 0);

    auto c1 = part.rank_to_grid(1);  // (0, 1)
    EXPECT_EQ(c1[0], 0);
    EXPECT_EQ(c1[1], 1);

    auto c2 = part.rank_to_grid(2);  // (0, 2)
    EXPECT_EQ(c2[0], 0);
    EXPECT_EQ(c2[1], 2);

    auto c3 = part.rank_to_grid(3);  // (1, 0)
    EXPECT_EQ(c3[0], 1);
    EXPECT_EQ(c3[1], 0);

    auto c4 = part.rank_to_grid(4);  // (1, 1)
    EXPECT_EQ(c4[0], 1);
    EXPECT_EQ(c4[1], 1);

    auto c5 = part.rank_to_grid(5);  // (1, 2)
    EXPECT_EQ(c5[0], 1);
    EXPECT_EQ(c5[1], 2);
}

TEST(BlockNdPartitionTest, GridToRank2D) {
    block_2d_partition part({{2, 3}});

    EXPECT_EQ(part.grid_to_rank({{0, 0}}), 0);
    EXPECT_EQ(part.grid_to_rank({{0, 1}}), 1);
    EXPECT_EQ(part.grid_to_rank({{0, 2}}), 2);
    EXPECT_EQ(part.grid_to_rank({{1, 0}}), 3);
    EXPECT_EQ(part.grid_to_rank({{1, 1}}), 4);
    EXPECT_EQ(part.grid_to_rank({{1, 2}}), 5);
}

TEST(BlockNdPartitionTest, RoundtripRankGridRank2D) {
    block_2d_partition part({{3, 4}});

    for (rank_t r = 0; r < part.total_ranks(); ++r) {
        auto coords = part.rank_to_grid(r);
        rank_t reconstructed = part.grid_to_rank(coords);
        EXPECT_EQ(reconstructed, r) << "Roundtrip failed for rank " << r;
    }
}

// =============================================================================
// Local Extent Computation Tests
// =============================================================================

TEST(BlockNdPartitionTest, LocalExtents2DEvenDivision) {
    // 12x8 domain on 3x2 process grid
    block_2d_partition part({{3, 2}});
    nd_extent<2> global_extents = {12, 8};

    // Rank 0 = grid(0,0): 4 rows, 4 cols
    auto ext0 = part.local_extents(global_extents, 0);
    EXPECT_EQ(ext0[0], 4);
    EXPECT_EQ(ext0[1], 4);

    // Rank 5 = grid(2,1): 4 rows, 4 cols
    auto ext5 = part.local_extents(global_extents, 5);
    EXPECT_EQ(ext5[0], 4);
    EXPECT_EQ(ext5[1], 4);
}

TEST(BlockNdPartitionTest, LocalExtents2DUnevenDivision) {
    // 10x7 domain on 3x2 process grid
    block_2d_partition part({{3, 2}});
    nd_extent<2> global_extents = {10, 7};

    // Rank 0 = grid(0,0): dim0 has 10/3=3 rem=1, so first rank gets 4 rows
    //                      dim1 has 7/2=3 rem=1, so first col gets 4 cols
    auto ext0 = part.local_extents(global_extents, 0);
    EXPECT_EQ(ext0[0], 4);  // first row chunk gets extra row
    EXPECT_EQ(ext0[1], 4);  // first col chunk gets extra col

    // Rank 1 = grid(0,1): 4 rows, 3 cols
    auto ext1 = part.local_extents(global_extents, 1);
    EXPECT_EQ(ext1[0], 4);
    EXPECT_EQ(ext1[1], 3);

    // Rank 2 = grid(1,0): 3 rows, 4 cols
    auto ext2 = part.local_extents(global_extents, 2);
    EXPECT_EQ(ext2[0], 3);
    EXPECT_EQ(ext2[1], 4);

    // Rank 5 = grid(2,1): 3 rows, 3 cols
    auto ext5 = part.local_extents(global_extents, 5);
    EXPECT_EQ(ext5[0], 3);
    EXPECT_EQ(ext5[1], 3);
}

TEST(BlockNdPartitionTest, LocalExtentsSumToGlobal) {
    // Verify that local extents sum to global along each dimension
    block_2d_partition part({{3, 4}});
    nd_extent<2> global_extents = {17, 13};

    // Sum local extents along dim 0 (rows)
    // For each grid column, the row extents should sum to 17
    for (rank_t col = 0; col < 4; ++col) {
        size_type row_sum = 0;
        for (rank_t row = 0; row < 3; ++row) {
            rank_t rank = part.grid_to_rank({{row, col}});
            auto ext = part.local_extents(global_extents, rank);
            row_sum += ext[0];
        }
        EXPECT_EQ(row_sum, 17) << "Row sum mismatch for column " << col;
    }

    // Sum local extents along dim 1 (cols)
    for (rank_t row = 0; row < 3; ++row) {
        size_type col_sum = 0;
        for (rank_t col = 0; col < 4; ++col) {
            rank_t rank = part.grid_to_rank({{row, col}});
            auto ext = part.local_extents(global_extents, rank);
            col_sum += ext[1];
        }
        EXPECT_EQ(col_sum, 13) << "Col sum mismatch for row " << row;
    }
}

TEST(BlockNdPartitionTest, LocalSizeSumToGlobal) {
    block_2d_partition part({{3, 4}});
    nd_extent<2> global_extents = {17, 13};

    size_type total = 0;
    for (rank_t r = 0; r < part.total_ranks(); ++r) {
        total += part.local_size(global_extents, r);
    }
    EXPECT_EQ(total, 17 * 13);
}

// =============================================================================
// Ownership Tests
// =============================================================================

TEST(BlockNdPartitionTest, OwnerEvenDomain) {
    // 8x6 on 2x3 grid
    block_2d_partition part({{2, 3}});
    nd_extent<2> global_extents = {8, 6};

    // grid(0,0) owns rows [0,4), cols [0,2)
    EXPECT_EQ(part.owner(nd_index<2>{0, 0}, global_extents), 0);
    EXPECT_EQ(part.owner(nd_index<2>{3, 1}, global_extents), 0);

    // grid(0,1) owns rows [0,4), cols [2,4)
    EXPECT_EQ(part.owner(nd_index<2>{0, 2}, global_extents), 1);
    EXPECT_EQ(part.owner(nd_index<2>{3, 3}, global_extents), 1);

    // grid(1,2) owns rows [4,8), cols [4,6)
    EXPECT_EQ(part.owner(nd_index<2>{4, 4}, global_extents), 5);
    EXPECT_EQ(part.owner(nd_index<2>{7, 5}, global_extents), 5);
}

TEST(BlockNdPartitionTest, OwnerCoversEntireDomain) {
    block_2d_partition part({{3, 2}});
    nd_extent<2> global_extents = {10, 7};

    // Every element must have an owner in [0, total_ranks)
    for (index_t i = 0; i < static_cast<index_t>(global_extents[0]); ++i) {
        for (index_t j = 0; j < static_cast<index_t>(global_extents[1]); ++j) {
            rank_t r = part.owner(nd_index<2>{i, j}, global_extents);
            EXPECT_GE(r, 0);
            EXPECT_LT(r, part.total_ranks());
        }
    }
}

TEST(BlockNdPartitionTest, IsLocalConsistentWithOwner) {
    block_2d_partition part({{2, 3}});
    nd_extent<2> global_extents = {8, 9};

    for (rank_t rank = 0; rank < part.total_ranks(); ++rank) {
        for (index_t i = 0; i < static_cast<index_t>(global_extents[0]); ++i) {
            for (index_t j = 0; j < static_cast<index_t>(global_extents[1]); ++j) {
                nd_index<2> idx{i, j};
                bool is_local = part.is_local(idx, global_extents, rank);
                rank_t owner = part.owner(idx, global_extents);
                EXPECT_EQ(is_local, owner == rank)
                    << "Mismatch at (" << i << "," << j << ") for rank " << rank;
            }
        }
    }
}

// =============================================================================
// Index Translation Tests
// =============================================================================

TEST(BlockNdPartitionTest, ToLocalToGlobalRoundtrip) {
    block_2d_partition part({{3, 2}});
    nd_extent<2> global_extents = {10, 7};

    for (rank_t rank = 0; rank < part.total_ranks(); ++rank) {
        auto local_ext = part.local_extents(global_extents, rank);
        for (index_t li = 0; li < static_cast<index_t>(local_ext[0]); ++li) {
            for (index_t lj = 0; lj < static_cast<index_t>(local_ext[1]); ++lj) {
                nd_index<2> local_idx{li, lj};
                auto global_idx = part.to_global(local_idx, global_extents, rank);
                auto back_local = part.to_local(global_idx, global_extents, rank);

                EXPECT_EQ(back_local[0], local_idx[0])
                    << "Roundtrip failed dim 0 for rank " << rank
                    << " local (" << li << "," << lj << ")";
                EXPECT_EQ(back_local[1], local_idx[1])
                    << "Roundtrip failed dim 1 for rank " << rank
                    << " local (" << li << "," << lj << ")";
            }
        }
    }
}

TEST(BlockNdPartitionTest, LocalStartsCorrect) {
    // 12x8 on 3x2 grid
    block_2d_partition part({{3, 2}});
    nd_extent<2> global_extents = {12, 8};

    // Rank 0 = grid(0,0) -> start (0, 0)
    auto s0 = part.local_starts(global_extents, 0);
    EXPECT_EQ(s0[0], 0);
    EXPECT_EQ(s0[1], 0);

    // Rank 1 = grid(0,1) -> start (0, 4)
    auto s1 = part.local_starts(global_extents, 1);
    EXPECT_EQ(s1[0], 0);
    EXPECT_EQ(s1[1], 4);

    // Rank 2 = grid(1,0) -> start (4, 0)
    auto s2 = part.local_starts(global_extents, 2);
    EXPECT_EQ(s2[0], 4);
    EXPECT_EQ(s2[1], 0);

    // Rank 5 = grid(2,1) -> start (8, 4)
    auto s5 = part.local_starts(global_extents, 5);
    EXPECT_EQ(s5[0], 8);
    EXPECT_EQ(s5[1], 4);
}

// =============================================================================
// 3D Decomposition Tests
// =============================================================================

TEST(BlockNdPartitionTest, Construction3D) {
    block_3d_partition part({{2, 3, 4}});

    EXPECT_EQ(part.proc_grid_dim(0), 2);
    EXPECT_EQ(part.proc_grid_dim(1), 3);
    EXPECT_EQ(part.proc_grid_dim(2), 4);
    EXPECT_EQ(part.total_ranks(), 24);
}

TEST(BlockNdPartitionTest, RankToGrid3D) {
    block_3d_partition part({{2, 3, 4}});

    // Rank 0 -> (0,0,0)
    auto c0 = part.rank_to_grid(0);
    EXPECT_EQ(c0[0], 0);
    EXPECT_EQ(c0[1], 0);
    EXPECT_EQ(c0[2], 0);

    // Rank 23 (last) -> (1,2,3)
    auto c23 = part.rank_to_grid(23);
    EXPECT_EQ(c23[0], 1);
    EXPECT_EQ(c23[1], 2);
    EXPECT_EQ(c23[2], 3);

    // Rank 13 = 1*12 + 0*4 + 1 -> (1,0,1)
    auto c13 = part.rank_to_grid(13);
    EXPECT_EQ(c13[0], 1);
    EXPECT_EQ(c13[1], 0);
    EXPECT_EQ(c13[2], 1);
}

TEST(BlockNdPartitionTest, LocalExtents3D) {
    block_3d_partition part({{2, 2, 2}});
    nd_extent<3> global_extents = {10, 8, 6};

    // Rank 0 = grid(0,0,0): 5x4x3
    auto ext0 = part.local_extents(global_extents, 0);
    EXPECT_EQ(ext0[0], 5);
    EXPECT_EQ(ext0[1], 4);
    EXPECT_EQ(ext0[2], 3);

    // Rank 7 = grid(1,1,1): 5x4x3
    auto ext7 = part.local_extents(global_extents, 7);
    EXPECT_EQ(ext7[0], 5);
    EXPECT_EQ(ext7[1], 4);
    EXPECT_EQ(ext7[2], 3);
}

TEST(BlockNdPartitionTest, LocalSizeSum3D) {
    block_3d_partition part({{2, 3, 2}});
    nd_extent<3> global_extents = {11, 13, 7};

    size_type total = 0;
    for (rank_t r = 0; r < part.total_ranks(); ++r) {
        total += part.local_size(global_extents, r);
    }
    EXPECT_EQ(total, 11 * 13 * 7);
}

// =============================================================================
// 1D Edge Case
// =============================================================================

TEST(BlockNdPartitionTest, OneDimensional) {
    block_nd_partition<1> part({{4}});

    EXPECT_EQ(part.total_ranks(), 4);

    nd_extent<1> global_extents = {100};

    auto ext0 = part.local_extents(global_extents, 0);
    EXPECT_EQ(ext0[0], 25);

    auto ext3 = part.local_extents(global_extents, 3);
    EXPECT_EQ(ext3[0], 25);

    size_type total = 0;
    for (rank_t r = 0; r < 4; ++r) {
        total += part.local_size(global_extents, r);
    }
    EXPECT_EQ(total, 100);
}

TEST(BlockNdPartitionTest, SingleRankGrid) {
    block_2d_partition part({{1, 1}});

    EXPECT_EQ(part.total_ranks(), 1);

    nd_extent<2> global_extents = {10, 20};

    auto ext = part.local_extents(global_extents, 0);
    EXPECT_EQ(ext[0], 10);
    EXPECT_EQ(ext[1], 20);
    EXPECT_EQ(part.local_size(global_extents, 0), 200);
}

}  // namespace dtl::test
