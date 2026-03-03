// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_types.cpp
/// @brief Unit tests for dtl/core/types.hpp
/// @details Tests fundamental types, sentinel values, tag types, and extents.

#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Type Size and Alias Tests
// =============================================================================

TEST(TypesTest, IndexTypeProperties) {
    // index_t should be signed and pointer-sized
    static_assert(std::is_signed_v<index_t>);
    static_assert(sizeof(index_t) == sizeof(void*));
    static_assert(std::is_same_v<index_t, std::ptrdiff_t>);
}

TEST(TypesTest, RankTypeProperties) {
    // rank_t should be int for MPI compatibility
    static_assert(std::is_signed_v<rank_t>);
    static_assert(std::is_same_v<rank_t, int>);
}

TEST(TypesTest, SizeTypeProperties) {
    // size_type should be unsigned and match std::size_t
    static_assert(std::is_unsigned_v<size_type>);
    static_assert(std::is_same_v<size_type, std::size_t>);
}

TEST(TypesTest, DifferenceTypeProperties) {
    // difference_type should match std::ptrdiff_t
    static_assert(std::is_signed_v<difference_type>);
    static_assert(std::is_same_v<difference_type, std::ptrdiff_t>);
}

// =============================================================================
// Sentinel Value Tests
// =============================================================================

TEST(TypesTest, SentinelValues) {
    // Verify sentinel values are distinct and have expected values
    EXPECT_EQ(no_rank, -1);
    EXPECT_EQ(all_ranks, -2);
    EXPECT_EQ(root_rank, 0);

    // Sentinel values should be distinct
    EXPECT_NE(no_rank, all_ranks);
    EXPECT_NE(no_rank, root_rank);
    EXPECT_NE(all_ranks, root_rank);
}

TEST(TypesTest, DynamicExtentValue) {
    // dynamic_extent should be the maximum size_type value
    EXPECT_EQ(dynamic_extent, static_cast<size_type>(-1));
    EXPECT_EQ(dynamic_extent, std::numeric_limits<size_type>::max());
}

TEST(TypesTest, SentinelConstexpr) {
    // All sentinels should be usable in constexpr contexts
    constexpr rank_t nr = no_rank;
    constexpr rank_t ar = all_ranks;
    constexpr rank_t rr = root_rank;
    constexpr size_type de = dynamic_extent;

    static_assert(nr == -1);
    static_assert(ar == -2);
    static_assert(rr == 0);
    static_assert(de == static_cast<size_type>(-1));
}

// =============================================================================
// Tag Type Tests
// =============================================================================

TEST(TypesTest, TagTypeDistinctness) {
    // All tag types should be distinct (no implicit conversions)
    static_assert(!std::is_convertible_v<local_tag, global_tag>);
    static_assert(!std::is_convertible_v<sync_tag, async_tag>);
    static_assert(!std::is_convertible_v<blocking_tag, nonblocking_tag>);
    static_assert(!std::is_convertible_v<read_only_tag, write_only_tag>);
}

TEST(TypesTest, TagTypeSizes) {
    // Tag types should be empty (for EBO optimization)
    static_assert(std::is_empty_v<local_tag>);
    static_assert(std::is_empty_v<global_tag>);
    static_assert(std::is_empty_v<sync_tag>);
    static_assert(std::is_empty_v<async_tag>);
    static_assert(std::is_empty_v<blocking_tag>);
    static_assert(std::is_empty_v<nonblocking_tag>);
    static_assert(std::is_empty_v<in_place_tag>);
    static_assert(std::is_empty_v<read_only_tag>);
    static_assert(std::is_empty_v<write_only_tag>);
    static_assert(std::is_empty_v<read_write_tag>);
}

TEST(TypesTest, TagInstancesExist) {
    // Global tag instances should be accessible
    [[maybe_unused]] const auto& l = local;
    [[maybe_unused]] const auto& g = global;
    [[maybe_unused]] const auto& sy = sync;
    [[maybe_unused]] const auto& as = async_v;
    [[maybe_unused]] const auto& bl = blocking;
    [[maybe_unused]] const auto& nb = nonblocking;
    [[maybe_unused]] const auto& ip = in_place;
    [[maybe_unused]] const auto& ro = read_only;
    [[maybe_unused]] const auto& wo = write_only;
    [[maybe_unused]] const auto& rw = read_write;

    // If we got here without errors, test passes
    SUCCEED();
}

// =============================================================================
// Extents Tests - Static
// =============================================================================

TEST(ExtentsTest, StaticExtentsRank) {
    using ext_1d = extents<10>;
    using ext_2d = extents<10, 20>;
    using ext_3d = extents<10, 20, 30>;

    static_assert(ext_1d::rank() == 1);
    static_assert(ext_2d::rank() == 2);
    static_assert(ext_3d::rank() == 3);

    EXPECT_EQ(ext_1d::rank(), 1);
    EXPECT_EQ(ext_2d::rank(), 2);
    EXPECT_EQ(ext_3d::rank(), 3);
}

TEST(ExtentsTest, StaticExtentsRankDynamic) {
    using all_static = extents<10, 20, 30>;
    using all_dynamic = extents<dynamic_extent, dynamic_extent, dynamic_extent>;
    using mixed = extents<10, dynamic_extent, 30>;

    static_assert(all_static::rank_dynamic() == 0);
    static_assert(all_dynamic::rank_dynamic() == 3);
    static_assert(mixed::rank_dynamic() == 1);

    EXPECT_EQ(all_static::rank_dynamic(), 0);
    EXPECT_EQ(all_dynamic::rank_dynamic(), 3);
    EXPECT_EQ(mixed::rank_dynamic(), 1);
}

TEST(ExtentsTest, StaticExtentQuery) {
    using ext = extents<10, 20, 30>;

    static_assert(ext::static_extent(0) == 10);
    static_assert(ext::static_extent(1) == 20);
    static_assert(ext::static_extent(2) == 30);

    EXPECT_EQ(ext::static_extent(0), 10);
    EXPECT_EQ(ext::static_extent(1), 20);
    EXPECT_EQ(ext::static_extent(2), 30);
}

TEST(ExtentsTest, StaticExtentQueryOutOfBounds) {
    using ext = extents<10, 20>;

    // Out of bounds should return dynamic_extent
    static_assert(ext::static_extent(5) == dynamic_extent);
    EXPECT_EQ(ext::static_extent(5), dynamic_extent);
}

TEST(ExtentsTest, StaticExtentSize) {
    using ext_1d = extents<10>;
    using ext_2d = extents<10, 20>;
    using ext_3d = extents<2, 3, 4>;

    ext_1d e1;
    ext_2d e2;
    ext_3d e3;

    EXPECT_EQ(e1.size(), 10);
    EXPECT_EQ(e2.size(), 200);
    EXPECT_EQ(e3.size(), 24);
}

// =============================================================================
// Extents Tests - Dynamic
// =============================================================================

TEST(ExtentsTest, DynamicExtentsConstruction) {
    using ext = extents<dynamic_extent, dynamic_extent>;

    ext e(10, 20);

    EXPECT_EQ(e.extent(0), 10);
    EXPECT_EQ(e.extent(1), 20);
    EXPECT_EQ(e.size(), 200);
}

TEST(ExtentsTest, MixedExtentsConstruction) {
    // Static first, dynamic second
    using ext1 = extents<10, dynamic_extent>;
    ext1 e1(20);
    EXPECT_EQ(e1.extent(0), 10);
    EXPECT_EQ(e1.extent(1), 20);
    EXPECT_EQ(e1.size(), 200);

    // Dynamic first, static second
    using ext2 = extents<dynamic_extent, 30>;
    ext2 e2(10);
    EXPECT_EQ(e2.extent(0), 10);
    EXPECT_EQ(e2.extent(1), 30);
    EXPECT_EQ(e2.size(), 300);

    // Mixed: static, dynamic, static
    using ext3 = extents<5, dynamic_extent, 7>;
    ext3 e3(6);
    EXPECT_EQ(e3.extent(0), 5);
    EXPECT_EQ(e3.extent(1), 6);
    EXPECT_EQ(e3.extent(2), 7);
    EXPECT_EQ(e3.size(), 210);
}

// =============================================================================
// Extents Tests - Aliases
// =============================================================================

TEST(ExtentsTest, ExtentAliases) {
    static_assert(extents_1d::rank() == 1);
    static_assert(extents_1d::rank_dynamic() == 1);

    static_assert(extents_2d::rank() == 2);
    static_assert(extents_2d::rank_dynamic() == 2);

    static_assert(extents_3d::rank() == 3);
    static_assert(extents_3d::rank_dynamic() == 3);
}

// =============================================================================
// Span Tests
// =============================================================================

TEST(SpanTest, DefaultConstruction) {
    span<int> s;

    EXPECT_EQ(s.data(), nullptr);
    EXPECT_EQ(s.size(), 0);
    EXPECT_TRUE(s.empty());
}

TEST(SpanTest, PointerSizeConstruction) {
    int arr[] = {1, 2, 3, 4, 5};
    span<int> s(arr, 5);

    EXPECT_EQ(s.data(), arr);
    EXPECT_EQ(s.size(), 5);
    EXPECT_FALSE(s.empty());
}

TEST(SpanTest, TwoPointerConstruction) {
    int arr[] = {1, 2, 3, 4, 5};
    span<int> s(arr, arr + 5);

    EXPECT_EQ(s.data(), arr);
    EXPECT_EQ(s.size(), 5);
}

TEST(SpanTest, ElementAccess) {
    int arr[] = {10, 20, 30};
    span<int> s(arr, 3);

    EXPECT_EQ(s[0], 10);
    EXPECT_EQ(s[1], 20);
    EXPECT_EQ(s[2], 30);
}

TEST(SpanTest, Iteration) {
    int arr[] = {1, 2, 3};
    span<int> s(arr, 3);

    int sum = 0;
    for (auto& val : s) {
        sum += val;
    }
    EXPECT_EQ(sum, 6);
}

TEST(SpanTest, ConstSpan) {
    const int arr[] = {1, 2, 3};
    span<const int> s(arr, 3);

    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s.size(), 3);
}

}  // namespace dtl::test
