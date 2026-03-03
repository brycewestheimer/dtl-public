// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_sort.cpp
/// @brief Unit tests for sample sort helper functions (Phase 12B)
/// @details Tests the local helper utilities in dtl::detail that support
///          distributed sample sort: sampling, pivot selection, bucket
///          partitioning, and alltoallv parameter computation.
///          Actual MPI-based distributed sort requires multi-rank execution.

#include <dtl/algorithms/sorting/sample_sort_detail.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace dtl::test {

// =============================================================================
// Local Sampling Tests
// =============================================================================

TEST(DistributedSortTest, SampleLocalEmpty) {
    std::vector<int> data;
    auto samples = detail::sample_local(data.begin(), data.end(), 5,
                                         std::less<>{});
    EXPECT_TRUE(samples.empty());
}

TEST(DistributedSortTest, SampleLocalSingle) {
    std::vector<int> data = {42};
    auto samples = detail::sample_local(data.begin(), data.end(), 1,
                                         std::less<>{});
    ASSERT_EQ(samples.size(), 1u);
    EXPECT_EQ(samples[0], 42);
}

TEST(DistributedSortTest, SampleLocalMultiple) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    auto samples = detail::sample_local(data.begin(), data.end(), 3,
                                         std::less<>{});
    EXPECT_EQ(samples.size(), 3u);
}

TEST(DistributedSortTest, SampleLocalEvenSpaced) {
    // 10 elements, 5 samples: should pick indices 0, 2, 4, 6, 8
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto samples = detail::sample_local(data.begin(), data.end(), 5,
                                         std::less<>{});
    ASSERT_EQ(samples.size(), 5u);
    EXPECT_EQ(samples[0], 0);
    EXPECT_EQ(samples[1], 2);
    EXPECT_EQ(samples[2], 4);
    EXPECT_EQ(samples[3], 6);
    EXPECT_EQ(samples[4], 8);
}

TEST(DistributedSortTest, SampleLocalZeroCount) {
    std::vector<int> data = {1, 2, 3};
    auto samples = detail::sample_local(data.begin(), data.end(), 0,
                                         std::less<>{});
    EXPECT_TRUE(samples.empty());
}

// =============================================================================
// Pivot Selection Tests
// =============================================================================

TEST(DistributedSortTest, SelectPivotsOneRank) {
    std::vector<int> samples = {1, 5, 10, 15, 20};
    auto pivots = detail::select_pivots(samples, 1, std::less<>{});
    EXPECT_TRUE(pivots.empty());
}

TEST(DistributedSortTest, SelectPivotsTwoRanks) {
    // 2 ranks -> 1 pivot (the median of the samples)
    std::vector<int> samples = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    auto pivots = detail::select_pivots(samples, 2, std::less<>{});
    ASSERT_EQ(pivots.size(), 1u);
    // Pivot should be near the middle of sorted samples
    EXPECT_GT(pivots[0], samples.front());
    EXPECT_LT(pivots[0], samples.back());
}

TEST(DistributedSortTest, SelectPivotsFourRanks) {
    // 4 ranks -> 3 pivots
    std::vector<int> samples = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto pivots = detail::select_pivots(samples, 4, std::less<>{});
    ASSERT_EQ(pivots.size(), 3u);
}

TEST(DistributedSortTest, SelectPivotsOrdered) {
    // Pivots should be in sorted order
    std::vector<int> samples = {50, 10, 30, 90, 70, 20, 80, 40, 60, 100};
    auto pivots = detail::select_pivots(samples, 5, std::less<>{});
    ASSERT_EQ(pivots.size(), 4u);
    EXPECT_TRUE(std::is_sorted(pivots.begin(), pivots.end()));
}

TEST(DistributedSortTest, SelectPivotsEmptySamples) {
    std::vector<int> samples;
    auto pivots = detail::select_pivots(samples, 4, std::less<>{});
    EXPECT_TRUE(pivots.empty());
}

// =============================================================================
// Partition by Pivots Tests
// =============================================================================

TEST(DistributedSortTest, PartitionByPivotsEmpty) {
    std::vector<int> data;
    std::vector<int> pivots = {5, 10};
    auto buckets = detail::partition_by_pivots(
        data.begin(), data.end(), pivots, std::less<>{});
    // 2 pivots -> 3 buckets, all empty
    ASSERT_EQ(buckets.size(), 3u);
    for (const auto& b : buckets) {
        EXPECT_TRUE(b.empty());
    }
}

TEST(DistributedSortTest, PartitionByPivotsSingle) {
    // Single pivot splits data into 2 buckets
    std::vector<int> data = {1, 2, 3, 7, 8, 9};
    std::vector<int> pivots = {5};
    auto buckets = detail::partition_by_pivots(
        data.begin(), data.end(), pivots, std::less<>{});
    ASSERT_EQ(buckets.size(), 2u);
    // Elements < 5 go to bucket 0, elements >= 5 go to bucket 1
    EXPECT_EQ(buckets[0].size(), 3u);  // 1, 2, 3
    EXPECT_EQ(buckets[1].size(), 3u);  // 7, 8, 9
}

TEST(DistributedSortTest, PartitionByPivotsMultiple) {
    // Multiple pivots, verify all elements in correct buckets
    std::vector<int> data = {1, 5, 10, 15, 20, 25};
    std::vector<int> pivots = {8, 18};
    auto buckets = detail::partition_by_pivots(
        data.begin(), data.end(), pivots, std::less<>{});
    ASSERT_EQ(buckets.size(), 3u);

    // Bucket 0: elements < 8 -> {1, 5}
    EXPECT_EQ(buckets[0].size(), 2u);
    // Bucket 1: elements in [8, 18) -> {10, 15}
    EXPECT_EQ(buckets[1].size(), 2u);
    // Bucket 2: elements >= 18 -> {20, 25}
    EXPECT_EQ(buckets[2].size(), 2u);
}

TEST(DistributedSortTest, PartitionByPivotsAllSame) {
    // All same values: everything lands in one bucket
    std::vector<int> data = {5, 5, 5, 5, 5};
    std::vector<int> pivots = {3, 7};
    auto buckets = detail::partition_by_pivots(
        data.begin(), data.end(), pivots, std::less<>{});
    ASSERT_EQ(buckets.size(), 3u);

    // All 5s are >= 3 and < 7, so they go to bucket 1
    size_type total = 0;
    for (const auto& b : buckets) total += b.size();
    EXPECT_EQ(total, 5u);
    EXPECT_EQ(buckets[1].size(), 5u);
}

// =============================================================================
// Alltoallv Parameter Tests
// =============================================================================

TEST(DistributedSortTest, AlltoallvParams) {
    // send_counts should match bucket sizes
    std::vector<std::vector<int>> buckets = {{1, 2}, {3}, {4, 5, 6}};
    auto params = detail::compute_alltoallv_params(buckets);

    ASSERT_EQ(params.send_counts.size(), 3u);
    EXPECT_EQ(params.send_counts[0], 2);
    EXPECT_EQ(params.send_counts[1], 1);
    EXPECT_EQ(params.send_counts[2], 3);
}

TEST(DistributedSortTest, AlltoallvDisplacements) {
    // send_displs should be cumulative sums
    std::vector<std::vector<int>> buckets = {{1, 2}, {3}, {4, 5, 6}};
    auto params = detail::compute_alltoallv_params(buckets);

    ASSERT_EQ(params.send_displs.size(), 3u);
    EXPECT_EQ(params.send_displs[0], 0);
    EXPECT_EQ(params.send_displs[1], 2);
    EXPECT_EQ(params.send_displs[2], 3);
}

TEST(DistributedSortTest, AlltoallvRecvInitialized) {
    // recv_counts and recv_displs should be initialized to zero
    std::vector<std::vector<int>> buckets = {{1}, {2}, {3}};
    auto params = detail::compute_alltoallv_params(buckets);

    ASSERT_EQ(params.recv_counts.size(), 3u);
    ASSERT_EQ(params.recv_displs.size(), 3u);
    for (size_type i = 0; i < 3; ++i) {
        EXPECT_EQ(params.recv_counts[i], 0);
        EXPECT_EQ(params.recv_displs[i], 0);
    }
}

// =============================================================================
// Flatten Buckets Tests
// =============================================================================

TEST(DistributedSortTest, FlattenBuckets) {
    std::vector<std::vector<int>> buckets = {{1, 2}, {3}, {4, 5, 6}};
    auto flat = detail::flatten_buckets(buckets);

    ASSERT_EQ(flat.size(), 6u);
    EXPECT_EQ(flat[0], 1);
    EXPECT_EQ(flat[1], 2);
    EXPECT_EQ(flat[2], 3);
    EXPECT_EQ(flat[3], 4);
    EXPECT_EQ(flat[4], 5);
    EXPECT_EQ(flat[5], 6);
}

TEST(DistributedSortTest, FlattenBucketsEmpty) {
    std::vector<std::vector<int>> buckets = {{}, {}, {}};
    auto flat = detail::flatten_buckets(buckets);
    EXPECT_TRUE(flat.empty());
}

// =============================================================================
// Local Sort Integration (using std::sort as baseline)
// =============================================================================

TEST(DistributedSortTest, LocalSortWorks) {
    std::vector<int> data = {9, 3, 7, 1, 5};
    std::sort(data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data.front(), 1);
    EXPECT_EQ(data.back(), 9);
}

TEST(DistributedSortTest, LocalStableSortWorks) {
    // Stable sort preserves relative order of equal elements
    struct item {
        int key;
        int order;
        bool operator<(const item& other) const { return key < other.key; }
    };
    std::vector<item> data = {{3, 0}, {1, 1}, {3, 2}, {2, 3}, {1, 4}};
    std::stable_sort(data.begin(), data.end());
    // After stable sort: key order is 1,1,2,3,3
    // Within key=1 group: order should be 1,4 (preserved)
    EXPECT_EQ(data[0].key, 1);
    EXPECT_EQ(data[0].order, 1);
    EXPECT_EQ(data[1].key, 1);
    EXPECT_EQ(data[1].order, 4);
    // Within key=3 group: order should be 0,2 (preserved)
    EXPECT_EQ(data[3].key, 3);
    EXPECT_EQ(data[3].order, 0);
    EXPECT_EQ(data[4].key, 3);
    EXPECT_EQ(data[4].order, 2);
}

TEST(DistributedSortTest, IsSortedTrue) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(DistributedSortTest, IsSortedFalse) {
    std::vector<int> data = {1, 3, 2, 4, 5};
    EXPECT_FALSE(std::is_sorted(data.begin(), data.end()));
}

TEST(DistributedSortTest, SortWithComparator) {
    // Custom comparator: descending order
    std::vector<int> data = {1, 5, 3, 9, 7};
    std::sort(data.begin(), data.end(), std::greater<>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<>{}));
    EXPECT_EQ(data.front(), 9);
    EXPECT_EQ(data.back(), 1);
}

TEST(DistributedSortTest, SortDefaultComparator) {
    // Default comparator (less): ascending order
    std::vector<int> data = {5, 3, 1, 4, 2};
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

}  // namespace dtl::test
