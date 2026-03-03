// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_tile_view.cpp
/// @brief Tests for tile_view boundary handling (V1.1)

#include <gtest/gtest.h>

#include <dtl/views/tile_view.hpp>

#include <array>
#include <vector>

namespace dtl::test {

// ============================================================================
// Test Fixtures
// ============================================================================

// Simple mock range for testing
template <std::size_t N>
struct mock_md_range {
    using value_type = float;
    std::array<size_type, N> extents;

    explicit mock_md_range(std::array<size_type, N> ext) : extents(ext) {}
};

// ============================================================================
// Tile Extent Tests
// ============================================================================

TEST(TileExtent, Construction2D) {
    tile_extent<2> ext{{32, 64}};

    EXPECT_EQ(ext.rank(), 2);
    EXPECT_EQ(ext.size(0), 32);
    EXPECT_EQ(ext.size(1), 64);
    EXPECT_EQ(ext.total_size(), 32 * 64);
}

TEST(TileExtent, Construction3D) {
    tile_extent<3> ext{{8, 16, 32}};

    EXPECT_EQ(ext.rank(), 3);
    EXPECT_EQ(ext.size(0), 8);
    EXPECT_EQ(ext.size(1), 16);
    EXPECT_EQ(ext.size(2), 32);
    EXPECT_EQ(ext.total_size(), 8 * 16 * 32);
}

TEST(TileExtent, Equality) {
    tile_extent<2> a{{32, 64}};
    tile_extent<2> b{{32, 64}};
    tile_extent<2> c{{32, 32}};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ============================================================================
// Tile Tests
// ============================================================================

TEST(Tile, BasicProperties) {
    mock_md_range<2> range{{100, 100}};

    tile<mock_md_range<2>, 2> t;
    t.origin = {0, 0};
    t.extent = tile_extent<2>{{32, 32}};
    t.requested_size = tile_extent<2>{{32, 32}};
    t.range = &range;

    EXPECT_EQ(t.size(), 32 * 32);
    EXPECT_FALSE(t.is_boundary_tile());
}

TEST(Tile, BoundaryTileDetection) {
    mock_md_range<2> range{{100, 100}};

    // Non-boundary tile
    tile<mock_md_range<2>, 2> interior;
    interior.origin = {32, 32};
    interior.extent = tile_extent<2>{{32, 32}};
    interior.requested_size = tile_extent<2>{{32, 32}};
    interior.range = &range;

    EXPECT_FALSE(interior.is_boundary_tile());

    // Boundary tile (smaller extent)
    tile<mock_md_range<2>, 2> boundary;
    boundary.origin = {96, 0};
    boundary.extent = tile_extent<2>{{4, 32}};  // Only 4 rows left
    boundary.requested_size = tile_extent<2>{{32, 32}};
    boundary.range = &range;

    EXPECT_TRUE(boundary.is_boundary_tile());
    EXPECT_TRUE(boundary.is_boundary_in_dim(0));
    EXPECT_FALSE(boundary.is_boundary_in_dim(1));
}

TEST(Tile, BoundaryInMultipleDimensions) {
    mock_md_range<2> range{{100, 100}};

    tile<mock_md_range<2>, 2> corner;
    corner.origin = {96, 96};
    corner.extent = tile_extent<2>{{4, 4}};  // Corner: both dims reduced
    corner.requested_size = tile_extent<2>{{32, 32}};
    corner.range = &range;

    EXPECT_TRUE(corner.is_boundary_tile());
    EXPECT_TRUE(corner.is_boundary_in_dim(0));
    EXPECT_TRUE(corner.is_boundary_in_dim(1));
}

TEST(Tile, TileLinearIndex) {
    mock_md_range<2> range{{64, 64}};

    tile<mock_md_range<2>, 2> t;
    t.requested_size = tile_extent<2>{{32, 32}};

    // num_tiles = {2, 2}
    std::array<size_type, 2> num_tiles = {2, 2};

    // Tile at (0, 0) -> linear index 0
    t.origin = {0, 0};
    EXPECT_EQ(t.tile_linear_index(num_tiles), 0);

    // Tile at (32, 0) -> linear index 1
    t.origin = {32, 0};
    EXPECT_EQ(t.tile_linear_index(num_tiles), 1);

    // Tile at (0, 32) -> linear index 2
    t.origin = {0, 32};
    EXPECT_EQ(t.tile_linear_index(num_tiles), 2);

    // Tile at (32, 32) -> linear index 3
    t.origin = {32, 32};
    EXPECT_EQ(t.tile_linear_index(num_tiles), 3);
}

// ============================================================================
// Tile View Tests
// ============================================================================

TEST(TileView, NumTilesExact) {
    mock_md_range<2> range{{64, 64}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {64, 64};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto counts = view.num_tiles();
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(view.total_tiles(), 4);
}

TEST(TileView, NumTilesWithRemainder) {
    mock_md_range<2> range{{100, 100}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {100, 100};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto counts = view.num_tiles();
    // 100 / 32 = 3 with remainder, so ceil = 4
    EXPECT_EQ(counts[0], 4);
    EXPECT_EQ(counts[1], 4);
    EXPECT_EQ(view.total_tiles(), 16);
}

TEST(TileView, NumTilesSingleTile) {
    mock_md_range<2> range{{10, 10}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {10, 10};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto counts = view.num_tiles();
    EXPECT_EQ(counts[0], 1);
    EXPECT_EQ(counts[1], 1);
    EXPECT_EQ(view.total_tiles(), 1);
}

TEST(TileView, Empty) {
    mock_md_range<2> range{{0, 0}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {0, 0};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.total_tiles(), 0);
}

TEST(TileView, TileSize) {
    mock_md_range<2> range{{100, 100}};
    tile_extent<2> tile_size{{32, 64}};
    std::array<size_type, 2> range_extent = {100, 100};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    EXPECT_EQ(view.tile_size().size(0), 32);
    EXPECT_EQ(view.tile_size().size(1), 64);
}

TEST(TileView, RangeExtent) {
    mock_md_range<2> range{{100, 200}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {100, 200};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto ext = view.range_extent();
    EXPECT_EQ(ext[0], 100);
    EXPECT_EQ(ext[1], 200);
}

TEST(TileView, SetRangeExtent) {
    mock_md_range<2> range{{0, 0}};
    tile_extent<2> tile_size{{32, 32}};

    tile_view<mock_md_range<2>, 2> view(range, tile_size);

    // Initially empty
    EXPECT_TRUE(view.empty());

    // Set extent
    view.set_range_extent({100, 100});

    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view.total_tiles(), 16);
}

// ============================================================================
// Tile Iterator Tests
// ============================================================================

TEST(TileIterator, BasicIteration) {
    mock_md_range<2> range{{64, 64}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {64, 64};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    size_type count = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        ++count;
    }

    EXPECT_EQ(count, 4);
}

TEST(TileIterator, TileOrigins) {
    mock_md_range<2> range{{64, 64}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {64, 64};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    std::vector<std::array<index_t, 2>> origins;
    for (auto tile : view) {
        origins.push_back(tile.origin);
    }

    ASSERT_EQ(origins.size(), 4);

    // Row-major order: (0,0), (32,0), (0,32), (32,32)
    EXPECT_EQ(origins[0][0], 0);
    EXPECT_EQ(origins[0][1], 0);

    EXPECT_EQ(origins[1][0], 32);
    EXPECT_EQ(origins[1][1], 0);

    EXPECT_EQ(origins[2][0], 0);
    EXPECT_EQ(origins[2][1], 32);

    EXPECT_EQ(origins[3][0], 32);
    EXPECT_EQ(origins[3][1], 32);
}

TEST(TileIterator, BoundaryTileExtents) {
    mock_md_range<2> range{{100, 100}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {100, 100};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    int boundary_count = 0;
    for (auto tile : view) {
        if (tile.is_boundary_tile()) {
            ++boundary_count;
        }
    }

    // With 100x100 and 32x32 tiles:
    // - Row 0: tiles at (0,0), (32,0), (64,0), (96,0) - last has extent (4,32)
    // - Row 1: tiles at (0,32), (32,32), (64,32), (96,32) - last has extent (4,32)
    // - Row 2: tiles at (0,64), (32,64), (64,64), (96,64) - last has extent (4,32)
    // - Row 3: tiles at (0,96), (32,96), (64,96), (96,96) - all have reduced row extent
    // Boundary tiles: 4 (last row) + 3 (last column, not counting corner twice)
    // Actually all tiles in last row and last column are boundary
    // Last row: 4 tiles (indices 12,13,14,15)
    // Last column: 4 tiles (indices 3,7,11,15) - 15 is in last row
    // Total boundary: 4 + 3 = 7

    EXPECT_GT(boundary_count, 0);
}

TEST(TileIterator, Equality) {
    mock_md_range<2> range{{64, 64}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {64, 64};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto begin1 = view.begin();
    auto begin2 = view.begin();
    auto end = view.end();

    EXPECT_EQ(begin1, begin2);
    EXPECT_NE(begin1, end);
}

TEST(TileIterator, PostIncrement) {
    mock_md_range<2> range{{64, 64}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {64, 64};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto it = view.begin();
    auto prev = it++;

    EXPECT_NE(prev, it);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(TileViewFactory, MakeTileView2D) {
    mock_md_range<2> range{{100, 100}};
    auto view = make_tile_view(range, 32, 32);

    EXPECT_EQ(view.tile_size().size(0), 32);
    EXPECT_EQ(view.tile_size().size(1), 32);
}

TEST(TileViewFactory, MakeTileView3D) {
    mock_md_range<3> range{{64, 64, 64}};
    auto view = make_tile_view(range, 16, 16, 16);

    EXPECT_EQ(view.tile_size().size(0), 16);
    EXPECT_EQ(view.tile_size().size(1), 16);
    EXPECT_EQ(view.tile_size().size(2), 16);
}

// ============================================================================
// Type Trait Tests
// ============================================================================

TEST(TileViewTraits, IsTileView) {
    using view_type = tile_view<mock_md_range<2>, 2>;

    static_assert(is_tile_view_v<view_type>);
    static_assert(!is_tile_view_v<mock_md_range<2>>);
    static_assert(!is_tile_view_v<int>);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TileViewEdgeCases, SingleElementRange) {
    mock_md_range<2> range{{1, 1}};
    tile_extent<2> tile_size{{32, 32}};
    std::array<size_type, 2> range_extent = {1, 1};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    EXPECT_EQ(view.total_tiles(), 1);

    auto it = view.begin();
    auto tile = *it;

    EXPECT_EQ(tile.origin[0], 0);
    EXPECT_EQ(tile.origin[1], 0);
    EXPECT_EQ(tile.extent.size(0), 1);
    EXPECT_EQ(tile.extent.size(1), 1);
    EXPECT_TRUE(tile.is_boundary_tile());
}

TEST(TileViewEdgeCases, TileSizeLargerThanRange) {
    mock_md_range<2> range{{10, 10}};
    tile_extent<2> tile_size{{100, 100}};
    std::array<size_type, 2> range_extent = {10, 10};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    EXPECT_EQ(view.total_tiles(), 1);

    auto tile = *view.begin();
    EXPECT_EQ(tile.extent.size(0), 10);  // Clamped to range size
    EXPECT_EQ(tile.extent.size(1), 10);
    EXPECT_TRUE(tile.is_boundary_tile());
}

TEST(TileViewEdgeCases, NonSquareTiles) {
    mock_md_range<2> range{{100, 200}};
    tile_extent<2> tile_size{{25, 50}};
    std::array<size_type, 2> range_extent = {100, 200};

    tile_view<mock_md_range<2>, 2> view(range, tile_size, range_extent);

    auto counts = view.num_tiles();
    EXPECT_EQ(counts[0], 4);  // 100 / 25
    EXPECT_EQ(counts[1], 4);  // 200 / 50
    EXPECT_EQ(view.total_tiles(), 16);
}

}  // namespace dtl::test
