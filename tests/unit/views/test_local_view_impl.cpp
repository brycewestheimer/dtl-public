// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_local_view_impl.cpp
/// @brief Unit tests for local_view implementation
/// @details Tests for Task 2.3.2: local_view

#include <dtl/views/local_view.hpp>

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
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.size(), 5);
    EXPECT_FALSE(view.empty());
}

TEST(LocalViewTest, PointerSizeWithMetadata) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5, 2, 100);  // rank 2, offset 100

    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.size(), 5);
    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.global_offset(), 100);
}

TEST(LocalViewTest, SpanConstruction) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::span<int> span(vec);
    local_view<int> view(span);

    EXPECT_EQ(view.data(), vec.data());
    EXPECT_EQ(view.size(), 5);
}

TEST(LocalViewTest, VectorConstruction) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    local_view<int> view(vec);

    EXPECT_EQ(view.data(), vec.data());
    EXPECT_EQ(view.size(), 5);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(LocalViewTest, BeginEnd) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    EXPECT_EQ(view.begin(), data);
    EXPECT_EQ(view.end(), data + 5);
    EXPECT_EQ(view.cbegin(), data);
    EXPECT_EQ(view.cend(), data + 5);
}

TEST(LocalViewTest, ReverseIterators) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    auto rit = view.rbegin();
    EXPECT_EQ(*rit, 5);
    ++rit;
    EXPECT_EQ(*rit, 4);
}

TEST(LocalViewTest, RangeBasedFor) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    int sum = 0;
    for (int x : view) {
        sum += x;
    }
    EXPECT_EQ(sum, 15);
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(LocalViewTest, OperatorSubscript) {
    int data[] = {10, 20, 30, 40, 50};
    local_view<int> view(data, 5);

    EXPECT_EQ(view[0], 10);
    EXPECT_EQ(view[2], 30);
    EXPECT_EQ(view[4], 50);

    // Modify through subscript
    view[2] = 300;
    EXPECT_EQ(data[2], 300);
}

TEST(LocalViewTest, AtMethod) {
    int data[] = {10, 20, 30};
    local_view<int> view(data, 3);

    EXPECT_EQ(view.at(0), 10);
    EXPECT_EQ(view.at(1), 20);
    EXPECT_EQ(view.at(2), 30);

    EXPECT_THROW((void)view.at(3), std::out_of_range);
}

TEST(LocalViewTest, FrontBack) {
    int data[] = {10, 20, 30};
    local_view<int> view(data, 3);

    EXPECT_EQ(view.front(), 10);
    EXPECT_EQ(view.back(), 30);
}

TEST(LocalViewTest, DataPointer) {
    int data[] = {1, 2, 3};
    local_view<int> view(data, 3);

    EXPECT_EQ(view.data(), data);

    // const version
    const local_view<int>& cview = view;
    EXPECT_EQ(cview.data(), data);
}

// =============================================================================
// Capacity Tests
// =============================================================================

TEST(LocalViewTest, Size) {
    int data[10];
    local_view<int> view(data, 10);

    EXPECT_EQ(view.size(), 10);
    EXPECT_EQ(view.length(), 10);  // alias
}

TEST(LocalViewTest, Empty) {
    local_view<int> empty_view;
    EXPECT_TRUE(empty_view.empty());

    int data[1];
    local_view<int> non_empty(data, 1);
    EXPECT_FALSE(non_empty.empty());
}

TEST(LocalViewTest, SizeBytes) {
    int data[10];
    local_view<int> view(data, 10);

    EXPECT_EQ(view.size_bytes(), 10 * sizeof(int));
}

// =============================================================================
// Subview Tests
// =============================================================================

TEST(LocalViewTest, First) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    auto first3 = view.first(3);
    EXPECT_EQ(first3.size(), 3);
    EXPECT_EQ(first3[0], 1);
    EXPECT_EQ(first3[2], 3);
}

TEST(LocalViewTest, Last) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    auto last3 = view.last(3);
    EXPECT_EQ(last3.size(), 3);
    EXPECT_EQ(last3[0], 3);
    EXPECT_EQ(last3[2], 5);
}

TEST(LocalViewTest, Subview) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5, 0, 100);

    auto sub = view.subview(1, 3);  // offset 1, count 3
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], 2);
    EXPECT_EQ(sub[2], 4);
    EXPECT_EQ(sub.global_offset(), 101);  // 100 + 1
}

// =============================================================================
// Conversion Tests
// =============================================================================

TEST(LocalViewTest, AsSpan) {
    int data[] = {1, 2, 3};
    local_view<int> view(data, 3);

    std::span<int> span = view.as_span();
    EXPECT_EQ(span.size(), 3);
    EXPECT_EQ(span[0], 1);
}

TEST(LocalViewTest, ImplicitSpanConversion) {
    int data[] = {1, 2, 3};
    local_view<int> view(data, 3);

    std::span<int> span = static_cast<std::span<int>>(view);
    EXPECT_EQ(span.size(), 3);
}

// =============================================================================
// STL Algorithm Compatibility Tests
// =============================================================================

TEST(LocalViewTest, StdSort) {
    int data[] = {5, 2, 4, 1, 3};
    local_view<int> view(data, 5);

    std::sort(view.begin(), view.end());

    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[1], 2);
    EXPECT_EQ(view[2], 3);
    EXPECT_EQ(view[3], 4);
    EXPECT_EQ(view[4], 5);
}

TEST(LocalViewTest, StdFind) {
    int data[] = {10, 20, 30, 40, 50};
    local_view<int> view(data, 5);

    auto it = std::find(view.begin(), view.end(), 30);
    EXPECT_NE(it, view.end());
    EXPECT_EQ(*it, 30);

    auto not_found = std::find(view.begin(), view.end(), 999);
    EXPECT_EQ(not_found, view.end());
}

TEST(LocalViewTest, StdAccumulate) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    int sum = std::accumulate(view.begin(), view.end(), 0);
    EXPECT_EQ(sum, 15);
}

TEST(LocalViewTest, StdTransform) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    std::transform(view.begin(), view.end(), view.begin(),
                   [](int x) { return x * 2; });

    EXPECT_EQ(view[0], 2);
    EXPECT_EQ(view[4], 10);
}

TEST(LocalViewTest, StdFill) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5);

    std::fill(view.begin(), view.end(), 42);

    for (int x : view) {
        EXPECT_EQ(x, 42);
    }
}

TEST(LocalViewTest, StdCopy) {
    int src[] = {1, 2, 3, 4, 5};
    int dst[5] = {};
    local_view<int> src_view(src, 5);
    local_view<int> dst_view(dst, 5);

    std::copy(src_view.begin(), src_view.end(), dst_view.begin());

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

// =============================================================================
// Concept Satisfaction Tests
// =============================================================================

TEST(LocalViewTest, RandomAccessIteratorConcept) {
    static_assert(std::random_access_iterator<local_view<int>::iterator>);
}

TEST(LocalViewTest, ContiguousIteratorConcept) {
    static_assert(std::contiguous_iterator<local_view<int>::iterator>);
}

TEST(LocalViewTest, ContiguousRangeConcept) {
    static_assert(std::ranges::contiguous_range<local_view<int>>);
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(LocalViewTest, IsLocalViewTrait) {
    static_assert(is_local_view_v<local_view<int>>);
    static_assert(is_local_view_v<local_view<const int>>);
    static_assert(is_local_view_v<local_view<double>>);

    static_assert(!is_local_view_v<int>);
    static_assert(!is_local_view_v<std::vector<int>>);
}

// =============================================================================
// Distribution Metadata Tests
// =============================================================================

TEST(LocalViewTest, ToGlobal) {
    int data[] = {1, 2, 3, 4, 5};
    local_view<int> view(data, 5, 2, 100);  // rank 2, offset 100

    EXPECT_EQ(view.to_global(0), 100);
    EXPECT_EQ(view.to_global(1), 101);
    EXPECT_EQ(view.to_global(4), 104);
}

// =============================================================================
// Const View Tests
// =============================================================================

TEST(LocalViewTest, ConstView) {
    const int data[] = {1, 2, 3, 4, 5};
    local_view<const int> view(data, 5);

    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view.front(), 1);
    EXPECT_EQ(view.back(), 5);

    // Should compile - read-only access
    int sum = 0;
    for (const int& x : view) {
        sum += x;
    }
    EXPECT_EQ(sum, 15);
}

}  // namespace dtl::test
