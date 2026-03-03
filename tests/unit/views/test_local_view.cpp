// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_local_view.cpp
/// @brief Unit tests for local_view
/// @details Tests local_view iterator, range-for, and data() access.

#include <dtl/views/local_view.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

// =============================================================================
// Construction Tests
// =============================================================================

TEST(LocalViewTest, DefaultConstruction) {
    local_view<int> view;

    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), 0);
    EXPECT_TRUE(view.empty());
}

TEST(LocalViewTest, PointerSizeConstruction) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view.data(), arr);
    EXPECT_EQ(view.size(), 5);
    EXPECT_FALSE(view.empty());
}

TEST(LocalViewTest, PointerSizeWithMetadata) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5, 2, 100);

    EXPECT_EQ(view.data(), arr);
    EXPECT_EQ(view.size(), 5);
    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.global_offset(), 100);
}

TEST(LocalViewTest, SpanConstruction) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(std::span<int>(vec));

    EXPECT_EQ(view.data(), vec.data());
    EXPECT_EQ(view.size(), 5);
}

TEST(LocalViewTest, RangeConstruction) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    EXPECT_EQ(view.data(), vec.data());
    EXPECT_EQ(view.size(), 5);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(LocalViewTest, IteratorBasics) {
    int arr[] = {10, 20, 30, 40, 50};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view.begin(), arr);
    EXPECT_EQ(view.end(), arr + 5);
    EXPECT_EQ(view.cbegin(), arr);
    EXPECT_EQ(view.cend(), arr + 5);
}

TEST(LocalViewTest, RangeForLoop) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    int sum = 0;
    for (int val : view) {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

TEST(LocalViewTest, RangeForLoopMutation) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    for (int& val : view) {
        val *= 2;
    }

    EXPECT_EQ(vec[0], 2);
    EXPECT_EQ(vec[1], 4);
    EXPECT_EQ(vec[4], 10);
}

TEST(LocalViewTest, ReverseIterators) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5);

    auto rit = view.rbegin();
    EXPECT_EQ(*rit, 5);
    ++rit;
    EXPECT_EQ(*rit, 4);

    std::vector<int> reversed(view.rbegin(), view.rend());
    std::vector<int> expected = {5, 4, 3, 2, 1};
    EXPECT_EQ(reversed, expected);
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(LocalViewTest, BracketOperator) {
    int arr[] = {10, 20, 30, 40, 50};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view[0], 10);
    EXPECT_EQ(view[2], 30);
    EXPECT_EQ(view[4], 50);

    // Mutation
    view[0] = 100;
    EXPECT_EQ(arr[0], 100);
}

TEST(LocalViewTest, AtMethod) {
    int arr[] = {10, 20, 30};
    local_view<int> view(arr, 3);

    EXPECT_EQ(view.at(0), 10);
    EXPECT_EQ(view.at(1), 20);
    EXPECT_EQ(view.at(2), 30);

    EXPECT_THROW(view.at(3), std::out_of_range);
    EXPECT_THROW(view.at(100), std::out_of_range);
}

TEST(LocalViewTest, FrontAndBack) {
    int arr[] = {10, 20, 30, 40, 50};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view.front(), 10);
    EXPECT_EQ(view.back(), 50);

    view.front() = 100;
    view.back() = 500;
    EXPECT_EQ(arr[0], 100);
    EXPECT_EQ(arr[4], 500);
}

TEST(LocalViewTest, DataPointer) {
    std::vector<int> vec = {1, 2, 3};
    local_view<int> view(vec);

    int* ptr = view.data();
    EXPECT_EQ(ptr, vec.data());

    ptr[0] = 100;
    EXPECT_EQ(vec[0], 100);
}

// =============================================================================
// Capacity Tests
// =============================================================================

TEST(LocalViewTest, SizeAndEmpty) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view.size(), 5);
    EXPECT_EQ(view.length(), 5);
    EXPECT_FALSE(view.empty());

    local_view<int> empty_view;
    EXPECT_EQ(empty_view.size(), 0);
    EXPECT_TRUE(empty_view.empty());
}

TEST(LocalViewTest, SizeBytes) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5);

    EXPECT_EQ(view.size_bytes(), 5 * sizeof(int));
}

// =============================================================================
// Distribution Metadata Tests
// =============================================================================

TEST(LocalViewTest, RankAndOffset) {
    int arr[] = {1, 2, 3};
    local_view<int> view(arr, 3, 5, 1000);

    EXPECT_EQ(view.rank(), 5);
    EXPECT_EQ(view.global_offset(), 1000);
}

TEST(LocalViewTest, ToGlobal) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5, 2, 100);

    EXPECT_EQ(view.to_global(0), 100);
    EXPECT_EQ(view.to_global(1), 101);
    EXPECT_EQ(view.to_global(4), 104);
}

// =============================================================================
// Subview Tests
// =============================================================================

TEST(LocalViewTest, FirstSubview) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5);

    auto sub = view.first(3);
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], 1);
    EXPECT_EQ(sub[2], 3);
}

TEST(LocalViewTest, LastSubview) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5, 0, 100);

    auto sub = view.last(2);
    EXPECT_EQ(sub.size(), 2);
    EXPECT_EQ(sub[0], 4);
    EXPECT_EQ(sub[1], 5);
    EXPECT_EQ(sub.global_offset(), 103);
}

TEST(LocalViewTest, SubviewRange) {
    int arr[] = {1, 2, 3, 4, 5};
    local_view<int> view(arr, 5, 0, 100);

    auto sub = view.subview(1, 3);
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], 2);
    EXPECT_EQ(sub[1], 3);
    EXPECT_EQ(sub[2], 4);
    EXPECT_EQ(sub.global_offset(), 101);
}

TEST(LocalViewTest, SubviewOutOfBounds) {
    int arr[] = {1, 2, 3};
    local_view<int> view(arr, 3);

    auto sub = view.first(10);  // Clamps to 3
    EXPECT_EQ(sub.size(), 3);

    auto empty = view.subview(100, 5);  // Start past end
    EXPECT_TRUE(empty.empty());
}

// =============================================================================
// Conversion Tests
// =============================================================================

TEST(LocalViewTest, AsSpan) {
    int arr[] = {1, 2, 3};
    local_view<int> view(arr, 3);

    std::span<int> sp = view.as_span();
    EXPECT_EQ(sp.data(), arr);
    EXPECT_EQ(sp.size(), 3);
}

TEST(LocalViewTest, ImplicitSpanConversion) {
    int arr[] = {1, 2, 3};
    local_view<int> view(arr, 3);

    std::span<int> sp = view;
    EXPECT_EQ(sp.data(), arr);
    EXPECT_EQ(sp.size(), 3);
}

// =============================================================================
// STL Algorithm Compatibility Tests
// =============================================================================

TEST(LocalViewTest, StdSort) {
    std::vector<int> vec = {5, 2, 8, 1, 9, 3};
    local_view<int> view(vec);

    std::sort(view.begin(), view.end());

    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[5], 9);
}

TEST(LocalViewTest, StdTransform) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    std::transform(view.begin(), view.end(), view.begin(),
                   [](int x) { return x * 2; });

    EXPECT_EQ(vec[0], 2);
    EXPECT_EQ(vec[2], 6);
    EXPECT_EQ(vec[4], 10);
}

TEST(LocalViewTest, StdAccumulate) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    int sum = std::accumulate(view.begin(), view.end(), 0);
    EXPECT_EQ(sum, 15);
}

TEST(LocalViewTest, StdFind) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    auto it = std::find(view.begin(), view.end(), 3);
    EXPECT_NE(it, view.end());
    EXPECT_EQ(*it, 3);

    auto not_found = std::find(view.begin(), view.end(), 100);
    EXPECT_EQ(not_found, view.end());
}

TEST(LocalViewTest, StdCopy) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);

    local_view<int> view(src);
    std::copy(view.begin(), view.end(), dst.begin());

    EXPECT_EQ(dst, src);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(LocalViewTest, MakeLocalViewPointerSize) {
    int arr[] = {1, 2, 3};
    auto view = make_local_view(arr, 3);

    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view[0], 1);
}

TEST(LocalViewTest, MakeLocalViewSpan) {
    std::vector<int> vec = {1, 2, 3};
    auto view = make_local_view(std::span<int>(vec));

    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view.data(), vec.data());
}

TEST(LocalViewTest, MakeLocalViewRange) {
    std::vector<int> vec = {1, 2, 3};
    auto view = make_local_view(vec);

    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view.data(), vec.data());
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(LocalViewTest, IsLocalViewTrait) {
    static_assert(is_local_view<local_view<int>>::value);
    static_assert(is_local_view<local_view<double>>::value);
    static_assert(!is_local_view<std::vector<int>>::value);
    static_assert(!is_local_view<int>::value);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(LocalViewTest, ConstexprOperations) {
    // These should compile as constexpr
    constexpr local_view<int> empty_view;
    static_assert(empty_view.size() == 0);
    static_assert(empty_view.empty());
    static_assert(empty_view.data() == nullptr);
}

// =============================================================================
// Const View Tests
// =============================================================================

TEST(LocalViewConstTest, ConstView) {
    const std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<const int> view(vec.data(), vec.size());

    EXPECT_EQ(view.size(), 5);
    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[4], 5);

    // Should be able to iterate but not modify
    int sum = 0;
    for (const int& val : view) {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

}  // namespace dtl::test
