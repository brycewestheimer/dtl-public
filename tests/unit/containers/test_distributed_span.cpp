// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_span.cpp
/// @brief Unit tests for distributed_span
/// @details Tests for Phase 11.5: distributed_span non-owning view

#include <dtl/containers/distributed_span.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// Construction Tests
// =============================================================================

TEST(DistributedSpanTest, DefaultConstruction) {
    distributed_span<int> span;

    EXPECT_TRUE(span.empty());
    EXPECT_EQ(span.size(), 0);
    EXPECT_EQ(span.local_size(), 0);
    EXPECT_EQ(span.data(), nullptr);
}

TEST(DistributedSpanTest, PointerAndSizeConstruction) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 100);

    EXPECT_EQ(span.size(), 100);  // Global size
    EXPECT_EQ(span.local_size(), 5);
    EXPECT_EQ(span.data(), vec.data());
    EXPECT_FALSE(span.empty());
}

TEST(DistributedSpanTest, ZeroLocalSize) {
    distributed_span<int> span(nullptr, 0, 100);

    EXPECT_EQ(span.size(), 100);  // Global size still 100
    EXPECT_EQ(span.local_size(), 0);
    EXPECT_FALSE(span.empty());  // empty() checks global_size, not local_size
}

// =============================================================================
// Size Query Tests
// =============================================================================

TEST(DistributedSpanTest, SizeQueries) {
    std::vector<int> vec(10);
    distributed_span<int> span(vec.data(), 10, 40);

    EXPECT_EQ(span.size(), 40);
    EXPECT_EQ(span.local_size(), 10);
    EXPECT_FALSE(span.empty());
}

TEST(DistributedSpanTest, SizeBytes) {
    std::vector<double> vec(10);
    distributed_span<double> span(vec.data(), 10, 40);

    EXPECT_EQ(span.size_bytes(), 10 * sizeof(double));
}

TEST(DistributedSpanTest, StaticExtent) {
    // dynamic_extent case
    using dynamic_span = distributed_span<int, dynamic_extent>;
    EXPECT_EQ(dynamic_span::extent, dynamic_extent);

    // Fixed extent case
    using fixed_span = distributed_span<int, 100>;
    EXPECT_EQ(fixed_span::extent, 100);
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(DistributedSpanTest, BracketOperator) {
    std::vector<int> vec = {10, 20, 30, 40, 50};
    distributed_span<int> span(vec.data(), 5, 5);

    EXPECT_EQ(span[0], 10);
    EXPECT_EQ(span[2], 30);
    EXPECT_EQ(span[4], 50);

    // Mutation
    span[0] = 100;
    EXPECT_EQ(vec[0], 100);
}

TEST(DistributedSpanTest, FrontBack) {
    std::vector<int> vec = {10, 20, 30, 40, 50};
    distributed_span<int> span(vec.data(), 5, 5);

    EXPECT_EQ(span.front(), 10);
    EXPECT_EQ(span.back(), 50);
}

TEST(DistributedSpanTest, DataPointer) {
    std::vector<int> vec = {1, 2, 3};
    distributed_span<int> span(vec.data(), 3, 3);

    EXPECT_EQ(span.data(), vec.data());
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(DistributedSpanTest, BeginEnd) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    EXPECT_EQ(span.begin(), vec.data());
    EXPECT_EQ(span.end(), vec.data() + 5);
}

TEST(DistributedSpanTest, RangeFor) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    int sum = 0;
    for (int val : span) {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

TEST(DistributedSpanTest, RangeForMutation) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    for (int& val : span) {
        val *= 2;
    }

    EXPECT_EQ(vec[0], 2);
    EXPECT_EQ(vec[4], 10);
}

// =============================================================================
// Subspan Tests
// =============================================================================

TEST(DistributedSpanTest, First) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    auto sub = span.first(3);
    EXPECT_EQ(sub.local_size(), 3);
    EXPECT_EQ(sub[0], 1);
    EXPECT_EQ(sub[2], 3);
}

TEST(DistributedSpanTest, Last) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    auto sub = span.last(2);
    EXPECT_EQ(sub.local_size(), 2);
    EXPECT_EQ(sub[0], 4);
    EXPECT_EQ(sub[1], 5);
}

TEST(DistributedSpanTest, Subspan) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    auto sub = span.subspan(1, 3);
    EXPECT_EQ(sub.local_size(), 3);
    EXPECT_EQ(sub[0], 2);
    EXPECT_EQ(sub[1], 3);
    EXPECT_EQ(sub[2], 4);
}

TEST(DistributedSpanTest, SubspanToEnd) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<int> span(vec.data(), 5, 5);

    auto sub = span.subspan(2);  // From index 2 to end
    EXPECT_EQ(sub.local_size(), 3);
    EXPECT_EQ(sub[0], 3);
    EXPECT_EQ(sub[2], 5);
}

// =============================================================================
// Distribution Query Tests
// =============================================================================

TEST(DistributedSpanTest, NumRanks) {
    std::vector<int> vec(10);
    distributed_span<int> span(vec.data(), 10, 10);

    EXPECT_EQ(span.num_ranks(), 1);  // Stub value
    EXPECT_EQ(span.rank(), 0);       // Stub value
}

// =============================================================================
// Const Span Tests
// =============================================================================

TEST(DistributedSpanTest, ConstSpan) {
    const std::vector<int> vec = {1, 2, 3, 4, 5};
    distributed_span<const int> span(vec.data(), 5, 5);

    EXPECT_EQ(span.size(), 5);
    EXPECT_EQ(span[0], 1);
    EXPECT_EQ(span[4], 5);

    int sum = 0;
    for (int val : span) {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

// =============================================================================
// Different Types Tests
// =============================================================================

TEST(DistributedSpanTest, DoubleType) {
    std::vector<double> vec = {1.1, 2.2, 3.3};
    distributed_span<double> span(vec.data(), 3, 3);

    EXPECT_DOUBLE_EQ(span[0], 1.1);
    EXPECT_DOUBLE_EQ(span[2], 3.3);
}

TEST(DistributedSpanTest, StructType) {
    struct Point { int x, y; };
    std::vector<Point> vec = {{1, 2}, {3, 4}, {5, 6}};
    distributed_span<Point> span(vec.data(), 3, 3);

    EXPECT_EQ(span[0].x, 1);
    EXPECT_EQ(span[1].y, 4);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(DistributedSpanTest, MakeDistributedSpanFromVector) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto span = make_distributed_span(vec);

    EXPECT_EQ(span.size(), 100);
    EXPECT_EQ(span.local_size(), 25);
}

TEST(DistributedSpanTest, MakeDistributedSpanFromConstVector) {
    const distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto span = make_distributed_span(vec);

    // Should be const span
    static_assert(std::is_const_v<typename decltype(span)::element_type>);
    EXPECT_EQ(span.size(), 100);
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DistributedSpanTest, IsDistributedContainer) {
    static_assert(is_distributed_container_v<distributed_span<int>>);
    static_assert(is_distributed_container_v<distributed_span<double>>);

    static_assert(!is_distributed_container_v<std::span<int>>);
}

TEST(DistributedSpanTest, IsDistributedSpan) {
    static_assert(is_distributed_span_v<distributed_span<int>>);
    static_assert(is_distributed_span_v<distributed_span<const int>>);
    static_assert(is_distributed_span_v<distributed_span<int, 100>>);

    static_assert(!is_distributed_span_v<std::span<int>>);
    static_assert(!is_distributed_span_v<distributed_vector<int>>);
}

}  // namespace dtl::test
