// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_container_semantics.cpp
/// @brief Unit tests for container semantics and operations
/// @details Phase 14 T03: distributed_vector, distributed_array,
///          distributed_tensor, distributed_map construction, access,
///          views, and type trait verification.
///
/// Note: Distributed containers contain sync_state with std::atomic members,
/// so move/copy constructors and assignment operators are implicitly deleted.
/// Tests verify these traits correctly and exercise the available API.

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/distributed_tensor.hpp>
#include <dtl/containers/distributed_map.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

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
// Type Trait Verification (containers are non-copyable, non-movable)
// =============================================================================

TEST(ContainerTraitsTest, VectorIsNotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<distributed_vector<int>>);
    EXPECT_FALSE(std::is_copy_assignable_v<distributed_vector<int>>);
}

TEST(ContainerTraitsTest, VectorIsNotMovable) {
    EXPECT_FALSE(std::is_move_constructible_v<distributed_vector<int>>);
    EXPECT_FALSE(std::is_move_assignable_v<distributed_vector<int>>);
}

TEST(ContainerTraitsTest, ArrayIsNotCopyable) {
    EXPECT_FALSE((std::is_copy_constructible_v<distributed_array<int, 10>>));
    EXPECT_FALSE((std::is_copy_assignable_v<distributed_array<int, 10>>));
}

TEST(ContainerTraitsTest, ArrayIsNotMovable) {
    EXPECT_FALSE((std::is_move_constructible_v<distributed_array<int, 10>>));
    EXPECT_FALSE((std::is_move_assignable_v<distributed_array<int, 10>>));
}

TEST(ContainerTraitsTest, TensorIsNotCopyable) {
    EXPECT_FALSE((std::is_copy_constructible_v<distributed_tensor<int, 2>>));
    EXPECT_FALSE((std::is_copy_assignable_v<distributed_tensor<int, 2>>));
}

TEST(ContainerTraitsTest, TensorIsNotMovable) {
    EXPECT_FALSE((std::is_move_constructible_v<distributed_tensor<int, 2>>));
    EXPECT_FALSE((std::is_move_assignable_v<distributed_tensor<int, 2>>));
}

TEST(ContainerTraitsTest, MapIsNotCopyable) {
    EXPECT_FALSE((std::is_copy_constructible_v<distributed_map<int, int>>));
    EXPECT_FALSE((std::is_copy_assignable_v<distributed_map<int, int>>));
}

TEST(ContainerTraitsTest, MapIsNotMovable) {
    EXPECT_FALSE((std::is_move_constructible_v<distributed_map<int, int>>));
    EXPECT_FALSE((std::is_move_assignable_v<distributed_map<int, int>>));
}

TEST(ContainerTraitsTest, VectorIsDefaultConstructible) {
    EXPECT_TRUE(std::is_default_constructible_v<distributed_vector<int>>);
}

TEST(ContainerTraitsTest, MapIsDefaultConstructible) {
    EXPECT_TRUE((std::is_default_constructible_v<distributed_map<int, int>>));
}

// =============================================================================
// distributed_vector Construction and Access
// =============================================================================

TEST(VectorSemanticsTest, DefaultConstruct) {
    distributed_vector<int> vec;
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.rank(), 0);
    EXPECT_EQ(vec.num_ranks(), 1);
}

TEST(VectorSemanticsTest, ConstructWithSize) {
    distributed_vector<int> vec(100, test_context{0, 1});
    EXPECT_EQ(vec.size(), 100u);
    EXPECT_EQ(vec.local_size(), 100u);
    EXPECT_FALSE(vec.empty());
}

TEST(VectorSemanticsTest, ConstructWithValue) {
    distributed_vector<int> vec(50, 42, test_context{0, 1});
    EXPECT_EQ(vec.size(), 50u);
    auto lv = vec.local_view();
    for (size_type i = 0; i < lv.size(); ++i) {
        EXPECT_EQ(lv[i], 42);
    }
}

TEST(VectorSemanticsTest, ConstructZeroSize) {
    distributed_vector<int> vec(0, test_context{0, 4});
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.local_size(), 0u);
}

TEST(VectorSemanticsTest, LocalViewWriteRead) {
    distributed_vector<int> vec(20, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < lv.size(); ++i) {
        lv[i] = static_cast<int>(i * 3);
    }
    EXPECT_EQ(vec.local(0), 0);
    EXPECT_EQ(vec.local(5), 15);
    EXPECT_EQ(vec.local(19), 57);
}

TEST(VectorSemanticsTest, LocalElementAccess) {
    distributed_vector<int> vec(10, 7, test_context{0, 1});
    EXPECT_EQ(vec.local(0), 7);
    EXPECT_EQ(vec.local(9), 7);
}

TEST(VectorSemanticsTest, PartitionInfo) {
    distributed_vector<int> vec(100, test_context{2, 4});
    EXPECT_EQ(vec.size(), 100u);
    EXPECT_EQ(vec.local_size(), 25u);
    EXPECT_EQ(vec.rank(), 2);
    EXPECT_EQ(vec.num_ranks(), 4);
}

TEST(VectorSemanticsTest, LocalSizeForRank) {
    distributed_vector<int> vec(10, test_context{0, 4});
    // 10 elements / 4 ranks
    size_type total = 0;
    for (rank_t r = 0; r < 4; ++r) {
        total += vec.local_size_for_rank(r);
    }
    EXPECT_EQ(total, 10u);
}

TEST(VectorSemanticsTest, MaxSize) {
    distributed_vector<int> vec;
    EXPECT_GT(vec.max_size(), 0u);
}

TEST(VectorSemanticsTest, FactoryCreate) {
    auto res = distributed_vector<int>::create(50, 2, 0);
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value().size(), 50u);
    EXPECT_EQ(res.value().num_ranks(), 2);
}

TEST(VectorSemanticsTest, DoubleType) {
    distributed_vector<double> vec(20, 3.14, test_context{0, 1});
    auto lv = vec.local_view();
    EXPECT_DOUBLE_EQ(lv[0], 3.14);
    EXPECT_DOUBLE_EQ(lv[19], 3.14);
}

// =============================================================================
// distributed_array Construction and Access
// =============================================================================

TEST(ArraySemanticsTest, ConstructWithContext) {
    distributed_array<int, 50> arr(test_context{0, 1});
    EXPECT_EQ(arr.size(), 50u);
    EXPECT_EQ(arr.global_size(), 50u);
}

TEST(ArraySemanticsTest, FillAndAccess) {
    distributed_array<int, 20> arr(test_context{0, 1});
    arr.fill(42);
    EXPECT_EQ(arr.local(0), 42);
    EXPECT_EQ(arr.local(19), 42);
}

TEST(ArraySemanticsTest, CompileTimeExtent) {
    distributed_array<int, 100> arr(test_context{0, 1});
    static_assert(decltype(arr)::extent == 100);
    EXPECT_EQ(arr.global_size(), 100u);
}

TEST(ArraySemanticsTest, LocalViewAccess) {
    distributed_array<int, 10> arr(test_context{0, 1});
    arr.fill(5);
    auto lv = arr.local_view();
    EXPECT_EQ(lv.size(), 10u);
    for (size_type i = 0; i < lv.size(); ++i) {
        EXPECT_EQ(lv[i], 5);
    }
}

TEST(ArraySemanticsTest, PartitionAcrossRanks) {
    distributed_array<int, 100> arr(test_context{1, 4});
    EXPECT_EQ(arr.global_size(), 100u);
    EXPECT_EQ(arr.local_size(), 25u);
    EXPECT_EQ(arr.rank(), 1);
    EXPECT_EQ(arr.num_ranks(), 4);
}

TEST(ArraySemanticsTest, DoubleType) {
    distributed_array<double, 10> arr(test_context{0, 1});
    arr.fill(2.718);
    EXPECT_DOUBLE_EQ(arr.local(0), 2.718);
}

TEST(ArraySemanticsTest, WriteViaLocalView) {
    distributed_array<int, 5> arr(test_context{0, 1});
    auto lv = arr.local_view();
    for (size_type i = 0; i < lv.size(); ++i) {
        lv[i] = static_cast<int>(i * 10);
    }
    EXPECT_EQ(arr.local(0), 0);
    EXPECT_EQ(arr.local(4), 40);
}

// =============================================================================
// distributed_tensor Construction and Access
// =============================================================================

TEST(TensorSemanticsTest, Construct2D) {
    nd_extent<2> extents = {4, 6};
    distributed_tensor<int, 2> t(extents, test_context{0, 1});
    EXPECT_EQ(t.size(), 24u);
    EXPECT_FALSE(t.empty());
}

TEST(TensorSemanticsTest, DefaultConstruct) {
    distributed_tensor<int, 2> t;
    EXPECT_EQ(t.size(), 0u);
    EXPECT_TRUE(t.empty());
}

TEST(TensorSemanticsTest, ElementAccess2D) {
    nd_extent<2> extents = {3, 4};
    distributed_tensor<int, 2> t(extents, test_context{0, 1});
    t(0, 0) = 10;
    t(2, 3) = 99;
    EXPECT_EQ(t(0, 0), 10);
    EXPECT_EQ(t(2, 3), 99);
}

TEST(TensorSemanticsTest, Construct3D) {
    nd_extent<3> extents = {2, 3, 4};
    distributed_tensor<int, 3> t(extents, test_context{0, 1});
    EXPECT_EQ(t.size(), 24u);
    t(1, 2, 3) = 123;
    EXPECT_EQ(t(1, 2, 3), 123);
}

TEST(TensorSemanticsTest, PartitionAcrossRanks) {
    nd_extent<2> extents = {10, 20};
    distributed_tensor<int, 2> t(extents, test_context{1, 4});
    EXPECT_EQ(t.size(), 200u);
    EXPECT_EQ(t.num_ranks(), 4);
    EXPECT_EQ(t.my_rank(), 1);
}

TEST(TensorSemanticsTest, DoubleType2D) {
    nd_extent<2> extents = {2, 2};
    distributed_tensor<double, 2> t(extents, test_context{0, 1});
    t(0, 0) = 1.5;
    t(1, 1) = 2.5;
    EXPECT_DOUBLE_EQ(t(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(t(1, 1), 2.5);
}

TEST(TensorSemanticsTest, FillAllElements) {
    nd_extent<2> extents = {5, 5};
    distributed_tensor<int, 2> t(extents, test_context{0, 1});
    // Fill via local view
    auto lv = t.local_view();
    for (size_type i = 0; i < lv.size(); ++i) {
        lv[i] = 7;
    }
    EXPECT_EQ(t(0, 0), 7);
    EXPECT_EQ(t(4, 4), 7);
}

// =============================================================================
// distributed_map Construction and Access
// =============================================================================

TEST(MapSemanticsTest, DefaultConstruct) {
    distributed_map<std::string, int> m;
    EXPECT_EQ(m.local_size(), 0u);
    EXPECT_TRUE(m.empty());
}

TEST(MapSemanticsTest, InsertAndContains) {
    distributed_map<std::string, int> m;
    auto res = m.insert("key1", 42);
    EXPECT_TRUE(res.has_value());
    EXPECT_TRUE(m.contains("key1"));
    EXPECT_FALSE(m.contains("key2"));
    EXPECT_EQ(m.local_size(), 1u);
}

TEST(MapSemanticsTest, InsertMultiple) {
    distributed_map<int, int> m;
    m.insert(1, 100);
    m.insert(2, 200);
    m.insert(3, 300);
    EXPECT_EQ(m.local_size(), 3u);
    EXPECT_TRUE(m.contains(1));
    EXPECT_TRUE(m.contains(2));
    EXPECT_TRUE(m.contains(3));
    EXPECT_FALSE(m.contains(4));
}

TEST(MapSemanticsTest, Count) {
    distributed_map<int, int> m;
    m.insert(5, 50);
    EXPECT_EQ(m.count(5), 1u);
    EXPECT_EQ(m.count(6), 0u);
}

TEST(MapSemanticsTest, ConstructWithContext) {
    distributed_map<int, int> m(test_context{0, 4});
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.rank(), 0);
    EXPECT_EQ(m.num_ranks(), 4);
}

TEST(MapSemanticsTest, IntKeyIntValue) {
    distributed_map<int, int> m;
    m.insert(10, 100);
    m.insert(20, 200);
    EXPECT_EQ(m.local_size(), 2u);
    EXPECT_TRUE(m.contains(10));
    EXPECT_TRUE(m.contains(20));
}

TEST(MapSemanticsTest, StringKeyStringValue) {
    distributed_map<std::string, std::string> m;
    m.insert("hello", "world");
    EXPECT_TRUE(m.contains("hello"));
    EXPECT_EQ(m.local_size(), 1u);
}

TEST(MapSemanticsTest, EmptyAfterDefault) {
    distributed_map<int, double> m;
    EXPECT_TRUE(m.empty());
    EXPECT_TRUE(m.local_empty());
    EXPECT_EQ(m.local_size(), 0u);
}

TEST(MapSemanticsTest, OwnershipCheck) {
    distributed_map<int, int> m(test_context{0, 4});
    // owner() returns which rank owns a key based on hash
    rank_t o = m.owner(42);
    EXPECT_GE(o, 0);
    EXPECT_LT(o, 4);
}

TEST(MapSemanticsTest, IsLocalCheck) {
    distributed_map<int, int> m(test_context{0, 1});
    // With single rank, all keys should be local
    EXPECT_TRUE(m.is_local(42));
    EXPECT_TRUE(m.is_local(0));
}

// =============================================================================
// Container Local View Properties
// =============================================================================

TEST(ContainerViewTest, VectorLocalViewProperties) {
    distributed_vector<int> vec(50, 0, test_context{0, 1});
    auto lv = vec.local_view();
    EXPECT_EQ(lv.size(), 50u);
    EXPECT_FALSE(lv.empty());
    EXPECT_NE(lv.data(), nullptr);
}

TEST(ContainerViewTest, VectorConstLocalView) {
    const distributed_vector<int> vec(30, 5, test_context{0, 1});
    auto lv = vec.local_view();
    EXPECT_EQ(lv.size(), 30u);
    EXPECT_EQ(lv[0], 5);
}

TEST(ContainerViewTest, ArrayLocalViewProperties) {
    distributed_array<int, 25> arr(test_context{0, 1});
    arr.fill(3);
    auto lv = arr.local_view();
    EXPECT_EQ(lv.size(), 25u);
    EXPECT_EQ(lv[0], 3);
}

TEST(ContainerViewTest, TensorLocalView) {
    nd_extent<2> extents = {4, 5};
    distributed_tensor<int, 2> t(extents, test_context{0, 1});
    auto lv = t.local_view();
    EXPECT_EQ(lv.size(), 20u);
}

}  // namespace dtl::test
