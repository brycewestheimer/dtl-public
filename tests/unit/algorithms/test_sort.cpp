// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_sort.cpp
/// @brief Unit tests for sort algorithm
/// @details Tests for Task 3.5: Sorting Algorithms

#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/views/local_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Basic Sort Tests
// =============================================================================

TEST(SortTest, SortsInAscendingOrder) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[8], 9);
}

TEST(SortTest, SortsWithComparator) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    dispatch_sort(seq{}, data.begin(), data.end(), std::greater<>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<>{}));
    EXPECT_EQ(data[0], 9);
    EXPECT_EQ(data[8], 1);
}

TEST(SortTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(SortTest, ReverseSorted) {
    std::vector<int> data = {5, 4, 3, 2, 1};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(SortTest, EmptyVector) {
    std::vector<int> data;
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(data.empty());
}

TEST(SortTest, SingleElement) {
    std::vector<int> data = {42};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data[0], 42);
}

TEST(SortTest, TwoElements) {
    std::vector<int> data = {2, 1};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2}));
}

TEST(SortTest, DuplicateElements) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortTest, AllSameElements) {
    std::vector<int> data(100, 42);
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 42; }));
}

// =============================================================================
// Parallel Sort Tests
// =============================================================================

TEST(SortTest, ParallelSort) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    dispatch_sort(par{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortTest, ParallelSortWithComparator) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    dispatch_sort(par{}, data.begin(), data.end(), std::greater<>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<>{}));
}

TEST(SortTest, LargeDataSetParallel) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);

    // Shuffle the data with a fixed seed for deterministic test behavior
    // (determinism requirements for test reproducibility)
    std::mt19937 gen(42);
    std::shuffle(data.begin(), data.end(), gen);

    dispatch_sort(par{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// =============================================================================
// Sort with Local View Tests
// =============================================================================

TEST(SortTest, LocalViewSort) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    local_view<int> view(data.data(), data.size());

    dispatch_sort(seq{}, view.begin(), view.end());

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortTest, LocalViewParallelSort) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    local_view<int> view(data.data(), data.size());

    dispatch_sort(par{}, view.begin(), view.end());

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// =============================================================================
// Different Value Types
// =============================================================================

TEST(SortTest, DoubleSort) {
    std::vector<double> data = {3.14, 2.71, 1.41, 1.73, 2.23};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortTest, StringSort) {
    std::vector<std::string> data = {"banana", "apple", "cherry", "date"};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<std::string>{"apple", "banana", "cherry", "date"}));
}

TEST(SortTest, CharSort) {
    std::vector<char> data = {'z', 'a', 'm', 'b', 'y'};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<char>{'a', 'b', 'm', 'y', 'z'}));
}

// =============================================================================
// Custom Comparator Tests
// =============================================================================

TEST(SortTest, AbsoluteValueSort) {
    std::vector<int> data = {-5, 3, -1, 4, -2};
    auto abs_compare = [](int a, int b) { return std::abs(a) < std::abs(b); };
    dispatch_sort(seq{}, data.begin(), data.end(), abs_compare);

    // Check absolute values are in order
    for (size_t i = 1; i < data.size(); ++i) {
        EXPECT_LE(std::abs(data[i - 1]), std::abs(data[i]));
    }
}

TEST(SortTest, LengthSort) {
    std::vector<std::string> data = {"a", "ccc", "bb", "dddd"};
    auto len_compare = [](const std::string& a, const std::string& b) {
        return a.length() < b.length();
    };
    dispatch_sort(seq{}, data.begin(), data.end(), len_compare);

    EXPECT_EQ(data[0].length(), 1u);
    EXPECT_EQ(data[1].length(), 2u);
    EXPECT_EQ(data[2].length(), 3u);
    EXPECT_EQ(data[3].length(), 4u);
}

// =============================================================================
// is_sorted Tests
// =============================================================================

TEST(IsSortedTest, SortedRange) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(IsSortedTest, UnsortedRange) {
    std::vector<int> data = {1, 3, 2, 4, 5};
    EXPECT_FALSE(std::is_sorted(data.begin(), data.end()));
}

TEST(IsSortedTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(IsSortedTest, SingleElement) {
    std::vector<int> data = {42};
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(IsSortedTest, DescendingOrder) {
    std::vector<int> data = {5, 4, 3, 2, 1};
    EXPECT_FALSE(std::is_sorted(data.begin(), data.end()));
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<>{}));
}

// =============================================================================
// Stable Sort Tests
// =============================================================================

TEST(StableSortTest, PreservesEqualElementOrder) {
    struct Item {
        int key;
        int order;  // Original position
        bool operator<(const Item& other) const { return key < other.key; }
    };

    std::vector<Item> data = {{2, 0}, {1, 1}, {2, 2}, {1, 3}, {2, 4}};
    std::stable_sort(data.begin(), data.end());

    // Verify sorted by key
    for (size_t i = 1; i < data.size(); ++i) {
        EXPECT_LE(data[i - 1].key, data[i].key);
    }

    // Verify stability: items with same key preserve original order
    std::vector<int> twos_order;
    for (const auto& item : data) {
        if (item.key == 2) twos_order.push_back(item.order);
    }
    EXPECT_EQ(twos_order, (std::vector<int>{0, 2, 4}));
}

// =============================================================================
// Partial Sort Tests
// =============================================================================

TEST(PartialSortTest, SmallestKElements) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::partial_sort(data.begin(), data.begin() + 3, data.end());

    // First 3 elements are the smallest, in sorted order
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 3);
}

TEST(PartialSortTest, LargestKElements) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::partial_sort(data.begin(), data.begin() + 3, data.end(), std::greater<>{});

    // First 3 elements are the largest, in descending order
    EXPECT_EQ(data[0], 9);
    EXPECT_EQ(data[1], 8);
    EXPECT_EQ(data[2], 7);
}

// =============================================================================
// nth_element Tests
// =============================================================================

TEST(NthElementTest, FindsMedian) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    size_t n = data.size() / 2;
    std::nth_element(data.begin(),
                     data.begin() + static_cast<std::vector<int>::difference_type>(n),
                     data.end());

    // Element at position n is the median (5)
    EXPECT_EQ(data[n], 5);

    // All elements before are <= data[n]
    for (size_t i = 0; i < n; ++i) {
        EXPECT_LE(data[i], data[n]);
    }

    // All elements after are >= data[n]
    for (size_t i = n + 1; i < data.size(); ++i) {
        EXPECT_GE(data[i], data[n]);
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SortTest, NegativeNumbers) {
    std::vector<int> data = {-5, -2, -8, -1, -9};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{-9, -8, -5, -2, -1}));
}

TEST(SortTest, MixedSignNumbers) {
    std::vector<int> data = {3, -1, 4, -1, 5, -9, 2, -6};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortTest, LargeValues) {
    std::vector<long long> data = {
        1000000000000LL, 500000000000LL, 2000000000000LL, 100000000000LL
    };
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// =============================================================================
// Sequential vs Parallel Consistency
// =============================================================================

TEST(SortTest, SeqAndParProduceSameResult) {
    std::vector<int> data1 = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::vector<int> data2 = data1;

    dispatch_sort(seq{}, data1.begin(), data1.end());
    dispatch_sort(par{}, data2.begin(), data2.end());

    EXPECT_EQ(data1, data2);
}

TEST(SortTest, LargeSeqParConsistency) {
    std::vector<int> data1(1000);
    std::iota(data1.begin(), data1.end(), 0);

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::shuffle(data1.begin(), data1.end(), gen);

    std::vector<int> data2 = data1;

    dispatch_sort(seq{}, data1.begin(), data1.end());
    dispatch_sort(par{}, data2.begin(), data2.end());

    EXPECT_EQ(data1, data2);
}

// =============================================================================
// distributed_sort_config Tests
// =============================================================================

TEST(DistributedSortConfigTest, DefaultConfig) {
    distributed_sort_config config;
    EXPECT_EQ(config.oversampling_factor, 3u);
    EXPECT_TRUE(config.use_parallel_local_sort);
    EXPECT_TRUE(config.use_parallel_merge);
}

TEST(DistributedSortConfigTest, CustomConfig) {
    distributed_sort_config config{5, false, true};
    EXPECT_EQ(config.oversampling_factor, 5u);
    EXPECT_FALSE(config.use_parallel_local_sort);
    EXPECT_TRUE(config.use_parallel_merge);
}

// =============================================================================
// distributed_sort_result Tests
// =============================================================================

TEST(DistributedSortResultTest, DefaultResult) {
    distributed_sort_result result;
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_sent, 0u);
    EXPECT_EQ(result.elements_received, 0u);
}

}  // namespace dtl::test
