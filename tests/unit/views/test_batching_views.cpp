// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_batching_views.cpp
/// @brief Tests for batching views: chunk_view, tile_view, window_view, chunk_by_view
/// @since 0.1.0

#include <dtl/views/chunk_view.hpp>
#include <dtl/views/tile_view.hpp>
#include <dtl/views/window_view.hpp>
#include <dtl/views/chunk_by_view.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <array>
#include <numeric>
#include <functional>

namespace dtl::test {

// ============================================================================
// Chunk View Tests
// ============================================================================

class ChunkViewTest : public ::testing::Test {
protected:
    std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
};

TEST_F(ChunkViewTest, BasicChunking) {
    chunk_view view(data, 3);

    size_type count = 0;
    for (auto chunk : view) {
        ++count;
        EXPECT_LE(chunk.size(), 3u);
        EXPECT_FALSE(chunk.empty());
    }

    // 10 elements with chunk size 3 = 4 chunks (3+3+3+1)
    EXPECT_EQ(count, 4u);
}

TEST_F(ChunkViewTest, ChunkSizes) {
    chunk_view view(data, 3);

    std::vector<size_type> sizes;
    for (auto chunk : view) {
        sizes.push_back(chunk.size());
    }

    ASSERT_EQ(sizes.size(), 4u);
    EXPECT_EQ(sizes[0], 3u);
    EXPECT_EQ(sizes[1], 3u);
    EXPECT_EQ(sizes[2], 3u);
    EXPECT_EQ(sizes[3], 1u);  // Last chunk is smaller
}

TEST_F(ChunkViewTest, ChunkContents) {
    chunk_view view(data, 4);

    auto it = view.begin();
    auto first_chunk = *it;

    std::vector<int> first_values(first_chunk.begin(), first_chunk.end());
    EXPECT_EQ(first_values, (std::vector<int>{1, 2, 3, 4}));

    ++it;
    auto second_chunk = *it;
    std::vector<int> second_values(second_chunk.begin(), second_chunk.end());
    EXPECT_EQ(second_values, (std::vector<int>{5, 6, 7, 8}));
}

TEST_F(ChunkViewTest, ChunkIndexAccess) {
    chunk_view view(data, 5);

    auto chunk = *view.begin();
    EXPECT_EQ(chunk[0], 1);
    EXPECT_EQ(chunk[1], 2);
    EXPECT_EQ(chunk[4], 5);
}

TEST_F(ChunkViewTest, NumChunks) {
    EXPECT_EQ(chunk_view(data, 2).num_chunks(), 5u);
    EXPECT_EQ(chunk_view(data, 3).num_chunks(), 4u);
    EXPECT_EQ(chunk_view(data, 5).num_chunks(), 2u);
    EXPECT_EQ(chunk_view(data, 10).num_chunks(), 1u);
    EXPECT_EQ(chunk_view(data, 11).num_chunks(), 1u);
}

TEST_F(ChunkViewTest, ChunkSizeMethod) {
    chunk_view view(data, 7);
    EXPECT_EQ(view.chunk_size(), 7u);
}

TEST_F(ChunkViewTest, TotalSize) {
    chunk_view view(data, 3);
    EXPECT_EQ(view.total_size(), 10u);
}

TEST_F(ChunkViewTest, SingleElementChunks) {
    chunk_view view(data, 1);

    size_type count = 0;
    for (auto chunk : view) {
        EXPECT_EQ(chunk.size(), 1u);
        ++count;
    }
    EXPECT_EQ(count, 10u);
}

TEST_F(ChunkViewTest, ExactDivision) {
    std::vector<int> even_data{1, 2, 3, 4, 5, 6};
    chunk_view view(even_data, 2);

    size_type count = 0;
    for (auto chunk : view) {
        EXPECT_EQ(chunk.size(), 2u);
        ++count;
    }
    EXPECT_EQ(count, 3u);
}

TEST_F(ChunkViewTest, EmptyRange) {
    std::vector<int> empty;
    chunk_view view(empty, 3);

    EXPECT_EQ(view.begin(), view.end());
    EXPECT_EQ(view.num_chunks(), 0u);
}

TEST_F(ChunkViewTest, IteratorEquality) {
    chunk_view view(data, 3);

    auto it1 = view.begin();
    auto it2 = view.begin();
    EXPECT_EQ(it1, it2);

    ++it1;
    EXPECT_NE(it1, it2);
}

TEST_F(ChunkViewTest, PostIncrement) {
    chunk_view view(data, 3);
    auto it = view.begin();

    auto chunk1 = (*it++).size();
    auto chunk2 = (*it).size();

    EXPECT_EQ(chunk1, 3u);
    EXPECT_EQ(chunk2, 3u);
}

TEST_F(ChunkViewTest, MakeChunkView) {
    auto view = make_chunk_view(data, 4);
    EXPECT_EQ(view.num_chunks(), 3u);
}

TEST_F(ChunkViewTest, TypeTraits) {
    EXPECT_TRUE(is_chunk_view_v<chunk_view<std::vector<int>>>);
    EXPECT_FALSE(is_chunk_view_v<std::vector<int>>);
}

// ============================================================================
// Window View Tests
// ============================================================================

class WindowViewTest : public ::testing::Test {
protected:
    std::vector<int> data{1, 2, 3, 4, 5};
};

TEST_F(WindowViewTest, BasicWindowing) {
    window_view view(data, 3);

    size_type count = 0;
    for (auto win : view) {
        EXPECT_EQ(win.size(), 3u);
        ++count;
    }

    // 5 elements, window size 3, stride 1 = 3 windows
    EXPECT_EQ(count, 3u);
}

TEST_F(WindowViewTest, WindowContents) {
    window_view view(data, 3);

    std::vector<std::vector<int>> windows;
    for (auto win : view) {
        windows.emplace_back(win.begin(), win.end());
    }

    ASSERT_EQ(windows.size(), 3u);
    EXPECT_EQ(windows[0], (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(windows[1], (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(windows[2], (std::vector<int>{3, 4, 5}));
}

TEST_F(WindowViewTest, WindowWithStride) {
    window_view view(data, 2, 2);  // window size 2, stride 2

    std::vector<std::vector<int>> windows;
    for (auto win : view) {
        windows.emplace_back(win.begin(), win.end());
    }

    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0], (std::vector<int>{1, 2}));
    EXPECT_EQ(windows[1], (std::vector<int>{3, 4}));
}

TEST_F(WindowViewTest, WindowElementAccess) {
    window_view view(data, 3);

    auto win = *view.begin();
    EXPECT_EQ(win[0], 1);
    EXPECT_EQ(win[1], 2);
    EXPECT_EQ(win[2], 3);
}

TEST_F(WindowViewTest, WindowFrontBack) {
    window_view view(data, 3);

    auto win = *view.begin();
    EXPECT_EQ(win.front(), 1);
    EXPECT_EQ(win.back(), 3);
}

TEST_F(WindowViewTest, WindowCenter) {
    window_view view(data, 3);

    auto win = *view.begin();
    EXPECT_EQ(win.center(), 2);  // Center of [1, 2, 3] is 2
}

TEST_F(WindowViewTest, NumWindows) {
    EXPECT_EQ(window_view(data, 2).num_windows(), 4u);
    EXPECT_EQ(window_view(data, 3).num_windows(), 3u);
    EXPECT_EQ(window_view(data, 5).num_windows(), 1u);
    EXPECT_EQ(window_view(data, 6).num_windows(), 0u);  // Window larger than range
}

TEST_F(WindowViewTest, NumWindowsWithStride) {
    EXPECT_EQ(window_view(data, 2, 1).num_windows(), 4u);
    EXPECT_EQ(window_view(data, 2, 2).num_windows(), 2u);
    EXPECT_EQ(window_view(data, 2, 3).num_windows(), 2u);
}

TEST_F(WindowViewTest, WindowSizeGetter) {
    window_view view(data, 4);
    EXPECT_EQ(view.window_size(), 4u);
}

TEST_F(WindowViewTest, StrideGetter) {
    window_view view(data, 2, 3);
    EXPECT_EQ(view.stride(), 3u);
}

TEST_F(WindowViewTest, DefaultStride) {
    window_view view(data, 2);
    EXPECT_EQ(view.stride(), 1u);
}

TEST_F(WindowViewTest, EmptyRange) {
    std::vector<int> empty;
    window_view view(empty, 3);

    EXPECT_EQ(view.begin(), view.end());
    EXPECT_EQ(view.num_windows(), 0u);
}

TEST_F(WindowViewTest, WindowLargerThanRange) {
    window_view view(data, 10);

    EXPECT_EQ(view.begin(), view.end());
    EXPECT_EQ(view.num_windows(), 0u);
}

TEST_F(WindowViewTest, SingleElementWindow) {
    window_view view(data, 1);

    size_type count = 0;
    for (auto win : view) {
        EXPECT_EQ(win.size(), 1u);
        ++count;
    }
    EXPECT_EQ(count, 5u);
}

TEST_F(WindowViewTest, StencilPattern) {
    // 3-point stencil: typical for numerical methods
    std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
    window_view view(values, 3);

    std::vector<double> results;
    for (auto win : view) {
        // Apply stencil: 0.25 * left + 0.5 * center + 0.25 * right
        double result = 0.25 * win[0] + 0.5 * win[1] + 0.25 * win[2];
        results.push_back(result);
    }

    ASSERT_EQ(results.size(), 3u);
    EXPECT_DOUBLE_EQ(results[0], 2.0);  // 0.25*1 + 0.5*2 + 0.25*3 = 2.0
    EXPECT_DOUBLE_EQ(results[1], 3.0);  // 0.25*2 + 0.5*3 + 0.25*4 = 3.0
    EXPECT_DOUBLE_EQ(results[2], 4.0);  // 0.25*3 + 0.5*4 + 0.25*5 = 4.0
}

TEST_F(WindowViewTest, MakeWindowView) {
    auto view1 = make_window_view(data, 3);
    EXPECT_EQ(view1.num_windows(), 3u);

    auto view2 = make_window_view(data, 2, 2);
    EXPECT_EQ(view2.num_windows(), 2u);
}

TEST_F(WindowViewTest, TypeTraits) {
    EXPECT_TRUE(is_window_view_v<window_view<std::vector<int>>>);
    EXPECT_FALSE(is_window_view_v<std::vector<int>>);
}

// ============================================================================
// Chunk By View Tests
// ============================================================================

class ChunkByViewTest : public ::testing::Test {
protected:
    std::vector<int> runs{1, 1, 2, 2, 2, 3, 3};
};

TEST_F(ChunkByViewTest, BasicChunkByEqual) {
    chunk_by_view view(runs, std::equal_to<>{});

    size_type count = 0;
    for (auto chunk : view) {
        ++count;
        EXPECT_FALSE(chunk.empty());
    }

    EXPECT_EQ(count, 3u);  // Three runs: [1,1], [2,2,2], [3,3]
}

TEST_F(ChunkByViewTest, ChunkSizes) {
    chunk_by_view view(runs, std::equal_to<>{});

    std::vector<size_type> sizes;
    for (auto chunk : view) {
        sizes.push_back(chunk.size());
    }

    ASSERT_EQ(sizes.size(), 3u);
    EXPECT_EQ(sizes[0], 2u);  // [1, 1]
    EXPECT_EQ(sizes[1], 3u);  // [2, 2, 2]
    EXPECT_EQ(sizes[2], 2u);  // [3, 3]
}

TEST_F(ChunkByViewTest, ChunkContents) {
    chunk_by_view view(runs, std::equal_to<>{});

    std::vector<std::vector<int>> chunks;
    for (auto chunk : view) {
        chunks.emplace_back(chunk.begin(), chunk.end());
    }

    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0], (std::vector<int>{1, 1}));
    EXPECT_EQ(chunks[1], (std::vector<int>{2, 2, 2}));
    EXPECT_EQ(chunks[2], (std::vector<int>{3, 3}));
}

TEST_F(ChunkByViewTest, ChunkFront) {
    chunk_by_view view(runs, std::equal_to<>{});

    std::vector<int> fronts;
    for (auto chunk : view) {
        fronts.push_back(chunk.front());
    }

    EXPECT_EQ(fronts, (std::vector<int>{1, 2, 3}));
}

TEST_F(ChunkByViewTest, AllEqual) {
    std::vector<int> same{5, 5, 5, 5};
    chunk_by_view view(same, std::equal_to<>{});

    size_type count = 0;
    for (auto chunk : view) {
        EXPECT_EQ(chunk.size(), 4u);
        ++count;
    }
    EXPECT_EQ(count, 1u);
}

TEST_F(ChunkByViewTest, AllDifferent) {
    std::vector<int> different{1, 2, 3, 4, 5};
    chunk_by_view view(different, std::equal_to<>{});

    size_type count = 0;
    for (auto chunk : view) {
        EXPECT_EQ(chunk.size(), 1u);
        ++count;
    }
    EXPECT_EQ(count, 5u);
}

TEST_F(ChunkByViewTest, CustomPredicate) {
    // Group by same parity (odd/even)
    std::vector<int> mixed{1, 3, 2, 4, 6, 5};
    auto same_parity = [](int a, int b) { return (a % 2) == (b % 2); };

    chunk_by_view view(mixed, same_parity);

    std::vector<std::vector<int>> chunks;
    for (auto chunk : view) {
        chunks.emplace_back(chunk.begin(), chunk.end());
    }

    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0], (std::vector<int>{1, 3}));      // odd
    EXPECT_EQ(chunks[1], (std::vector<int>{2, 4, 6}));   // even
    EXPECT_EQ(chunks[2], (std::vector<int>{5}));         // odd
}

TEST_F(ChunkByViewTest, EmptyRange) {
    std::vector<int> empty;
    chunk_by_view view(empty, std::equal_to<>{});

    EXPECT_EQ(view.begin(), view.end());
}

TEST_F(ChunkByViewTest, SingleElement) {
    std::vector<int> single{42};
    chunk_by_view view(single, std::equal_to<>{});

    size_type count = 0;
    for (auto chunk : view) {
        EXPECT_EQ(chunk.size(), 1u);
        EXPECT_EQ(chunk.front(), 42);
        ++count;
    }
    EXPECT_EQ(count, 1u);
}

TEST_F(ChunkByViewTest, MakeChunkByView) {
    auto view = make_chunk_by_view(runs, std::equal_to<>{});

    size_type count = 0;
    for ([[maybe_unused]] auto chunk : view) {
        ++count;
    }
    EXPECT_EQ(count, 3u);
}

TEST_F(ChunkByViewTest, MakeChunkByEqual) {
    auto view = make_chunk_by_equal(runs);

    size_type count = 0;
    for ([[maybe_unused]] auto chunk : view) {
        ++count;
    }
    EXPECT_EQ(count, 3u);
}

TEST_F(ChunkByViewTest, TypeTraits) {
    using ViewType = chunk_by_view<std::vector<int>, std::equal_to<int>>;
    EXPECT_TRUE(is_chunk_by_view_v<ViewType>);
    EXPECT_FALSE(is_chunk_by_view_v<std::vector<int>>);
}

// ============================================================================
// Tile Extent Tests
// ============================================================================

class TileExtentTest : public ::testing::Test {};

TEST_F(TileExtentTest, Construction) {
    tile_extent<2> extent{{32, 32}};

    EXPECT_EQ(extent.sizes[0], 32u);
    EXPECT_EQ(extent.sizes[1], 32u);
}

TEST_F(TileExtentTest, TotalSize) {
    tile_extent<2> extent2d{{4, 4}};
    EXPECT_EQ(extent2d.total_size(), 16u);

    tile_extent<3> extent3d{{2, 3, 4}};
    EXPECT_EQ(extent3d.total_size(), 24u);
}

TEST_F(TileExtentTest, SizeAccessor) {
    tile_extent<3> extent{{2, 4, 8}};

    EXPECT_EQ(extent.size(0), 2u);
    EXPECT_EQ(extent.size(1), 4u);
    EXPECT_EQ(extent.size(2), 8u);
}

TEST_F(TileExtentTest, Rank) {
    EXPECT_EQ((tile_extent<2>::rank()), 2u);
    EXPECT_EQ((tile_extent<3>::rank()), 3u);
    EXPECT_EQ((tile_extent<4>::rank()), 4u);
}

// ============================================================================
// Tile View Tests (Structure Tests)
// ============================================================================

class TileViewTest : public ::testing::Test {
protected:
    // Simple 2D structure for testing
    struct MockRange2D {
        using value_type = int;
    };
};

TEST_F(TileViewTest, TypeTraits) {
    EXPECT_TRUE((is_tile_view_v<tile_view<MockRange2D, 2>>));
    EXPECT_FALSE(is_tile_view_v<std::vector<int>>);
}

TEST_F(TileViewTest, TileViewConstruction) {
    MockRange2D range;
    tile_extent<2> extent{{32, 32}};

    tile_view view(range, extent);

    EXPECT_EQ(view.tile_size().sizes[0], 32u);
    EXPECT_EQ(view.tile_size().sizes[1], 32u);
}

// ============================================================================
// Integration Tests
// ============================================================================

class BatchingViewIntegrationTest : public ::testing::Test {
protected:
    std::vector<int> data;

    void SetUp() override {
        data.resize(100);
        std::iota(data.begin(), data.end(), 0);
    }
};

TEST_F(BatchingViewIntegrationTest, ChunkThenProcess) {
    // Use chunks to batch process data
    chunk_view view(data, 10);

    std::vector<int> chunk_sums;
    for (auto chunk : view) {
        int sum = 0;
        for (int val : chunk) {
            sum += val;
        }
        chunk_sums.push_back(sum);
    }

    EXPECT_EQ(chunk_sums.size(), 10u);
    // First chunk: 0+1+2+...+9 = 45
    EXPECT_EQ(chunk_sums[0], 45);
}

TEST_F(BatchingViewIntegrationTest, WindowMovingAverage) {
    std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
    window_view view(values, 3);

    std::vector<double> averages;
    for (auto win : view) {
        double sum = 0.0;
        for (double val : win) {
            sum += val;
        }
        averages.push_back(sum / static_cast<double>(win.size()));
    }

    ASSERT_EQ(averages.size(), 3u);
    EXPECT_DOUBLE_EQ(averages[0], 2.0);  // (1+2+3)/3
    EXPECT_DOUBLE_EQ(averages[1], 3.0);  // (2+3+4)/3
    EXPECT_DOUBLE_EQ(averages[2], 4.0);  // (3+4+5)/3
}

TEST_F(BatchingViewIntegrationTest, ChunkByForGroupOperations) {
    // Simulate group-by key scenario
    std::vector<int> sorted_keys{1, 1, 1, 2, 2, 3, 3, 3, 3};

    chunk_by_view view(sorted_keys, std::equal_to<>{});

    std::vector<std::pair<int, size_type>> groups;
    for (auto chunk : view) {
        groups.emplace_back(chunk.front(), chunk.size());
    }

    ASSERT_EQ(groups.size(), 3u);
    EXPECT_EQ(groups[0], std::make_pair(1, size_type{3}));
    EXPECT_EQ(groups[1], std::make_pair(2, size_type{2}));
    EXPECT_EQ(groups[2], std::make_pair(3, size_type{4}));
}

TEST_F(BatchingViewIntegrationTest, ChunkModification) {
    std::vector<int> mutable_data{1, 2, 3, 4, 5, 6};
    chunk_view view(mutable_data, 2);

    // Double each element in each chunk
    for (auto chunk : view) {
        for (auto& elem : chunk) {
            elem *= 2;
        }
    }

    EXPECT_EQ(mutable_data, (std::vector<int>{2, 4, 6, 8, 10, 12}));
}

}  // namespace dtl::test
