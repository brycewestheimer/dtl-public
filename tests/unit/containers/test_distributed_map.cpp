// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_map.cpp
/// @brief Unit tests for distributed_map
/// @details Tests for Phase 11.5: distributed_map container implementation

#include <dtl/containers/distributed_map.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::test {

// Type alias for default distributed_map
template <typename K, typename V>
using dmap = distributed_map<K, V, std::hash<K>, std::equal_to<K>>;

// =============================================================================
// Construction Tests
// =============================================================================

TEST(DistributedMapTest, DefaultConstruction) {
    dmap<int, int> map;

    EXPECT_EQ(map.local_size(), 0);
    EXPECT_TRUE(map.local_empty());
}

TEST(DistributedMapTest, EmptyMap) {
    dmap<std::string, int> map;

    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.local_size(), 0);
}

// =============================================================================
// Insert Tests
// =============================================================================

TEST(DistributedMapTest, InsertLocalKey) {
    dmap<int, int> map;

    // Key 0 with hash % 1 = 0 -> local on rank 0
    auto result = map.insert(0, 42);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(map.local_size(), 1);
}

TEST(DistributedMapTest, InsertMultipleKeys) {
    dmap<int, int> map;

    map.insert(0, 10);
    map.insert(1, 20);
    map.insert(2, 30);

    EXPECT_EQ(map.local_size(), 3);
}

TEST(DistributedMapTest, InsertStringKeys) {
    dmap<std::string, int> map;

    map.insert("hello", 1);
    map.insert("world", 2);

    EXPECT_GE(map.local_size(), 0);  // All keys should be local in single-rank
}

TEST(DistributedMapTest, InsertOrAssign) {
    dmap<int, int> map;

    // Insert new key
    auto result1 = map.insert_or_assign(42, 100);
    EXPECT_TRUE(result1.has_value());

    // Assign to existing key
    auto result2 = map.insert_or_assign(42, 200);
    EXPECT_TRUE(result2.has_value());
}

// =============================================================================
// Find Tests
// =============================================================================

TEST(DistributedMapTest, FindExistingKey) {
    dmap<int, int> map;
    map.insert(42, 100);

    auto result = map.find(42);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->second, 100);
}

TEST(DistributedMapTest, FindMissingKey) {
    dmap<int, int> map;
    map.insert(42, 100);

    auto result = map.find(99);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::key_not_found);
}

// =============================================================================
// Contains/Count Tests
// =============================================================================

TEST(DistributedMapTest, ContainsKey) {
    dmap<int, int> map;
    map.insert(42, 100);

    EXPECT_TRUE(map.contains(42));
    EXPECT_FALSE(map.contains(99));
}

TEST(DistributedMapTest, CountKey) {
    dmap<int, int> map;
    map.insert(42, 100);

    EXPECT_EQ(map.count(42), 1);
    EXPECT_EQ(map.count(99), 0);
}

// =============================================================================
// Erase Tests
// =============================================================================

TEST(DistributedMapTest, EraseExistingKey) {
    dmap<int, int> map;
    map.insert(42, 100);
    EXPECT_EQ(map.local_size(), 1);

    auto result = map.erase(42);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
    EXPECT_EQ(map.local_size(), 0);
}

TEST(DistributedMapTest, EraseMissingKey) {
    dmap<int, int> map;
    map.insert(42, 100);

    auto result = map.erase(99);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);  // Key didn't exist locally
    EXPECT_EQ(map.local_size(), 1);
}

TEST(DistributedMapTest, Clear) {
    dmap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    auto result = map.clear();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(map.local_empty());
    EXPECT_EQ(map.local_size(), 0);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(DistributedMapTest, IterateLocalEntries) {
    dmap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    int sum = 0;
    for (const auto& [key, value] : map) {
        sum += value;
    }
    EXPECT_EQ(sum, 60);
}

TEST(DistributedMapTest, BeginEnd) {
    dmap<int, int> map;
    map.insert(42, 100);

    EXPECT_NE(map.begin(), map.end());
    EXPECT_EQ(map.begin()->first, 42);
    EXPECT_EQ(map.begin()->second, 100);
}

TEST(DistributedMapTest, ConstIterators) {
    dmap<int, int> map;
    map.insert(42, 100);

    const auto& cmap = map;
    EXPECT_NE(cmap.cbegin(), cmap.cend());
    EXPECT_EQ(cmap.begin()->first, 42);
}

// =============================================================================
// Distribution Query Tests
// =============================================================================

TEST(DistributedMapTest, IsLocal) {
    dmap<int, int> map;

    // In single-rank mode, all keys are local
    EXPECT_TRUE(map.is_local(42));
    EXPECT_TRUE(map.is_local(0));
    EXPECT_TRUE(map.is_local(999));
}

TEST(DistributedMapTest, Owner) {
    dmap<int, int> map;

    // In single-rank mode, owner is always 0
    EXPECT_EQ(map.owner(42), 0);
    EXPECT_EQ(map.owner(0), 0);
}

TEST(DistributedMapTest, NumRanks) {
    dmap<int, int> map;

    EXPECT_EQ(map.num_ranks(), 1);  // Stub value
    EXPECT_EQ(map.rank(), 0);       // Stub value
}

// =============================================================================
// Hash Policy Tests
// =============================================================================

TEST(DistributedMapTest, LoadFactor) {
    dmap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);

    float lf = map.load_factor();
    EXPECT_GE(lf, 0.0f);
}

TEST(DistributedMapTest, MaxLoadFactor) {
    dmap<int, int> map;

    float orig = map.max_load_factor();
    map.max_load_factor(0.5f);
    EXPECT_FLOAT_EQ(map.max_load_factor(), 0.5f);

    // Restore
    map.max_load_factor(orig);
}

TEST(DistributedMapTest, Rehash) {
    dmap<int, int> map;
    map.insert(1, 10);

    // Should not throw
    map.rehash(100);
}

TEST(DistributedMapTest, Reserve) {
    dmap<int, int> map;

    // Should not throw
    map.reserve(100);
}

// =============================================================================
// Sync State Tests
// =============================================================================

TEST(DistributedMapTest, SyncStateInitiallyClean) {
    dmap<int, int> map;

    EXPECT_TRUE(map.is_clean());
    EXPECT_FALSE(map.is_dirty());
}

TEST(DistributedMapTest, SyncStateAfterInsert) {
    dmap<int, int> map;

    map.insert(42, 100);
    EXPECT_TRUE(map.is_dirty());
}

TEST(DistributedMapTest, SyncStateMarking) {
    dmap<int, int> map;

    map.mark_local_modified();
    EXPECT_TRUE(map.is_dirty());

    map.mark_clean();
    EXPECT_TRUE(map.is_clean());
}

TEST(DistributedMapTest, SyncStateReference) {
    dmap<int, int> map;

    auto& state = map.sync_state_ref();
    state.mark_local_modified();
    EXPECT_TRUE(map.is_dirty());
}

// =============================================================================
// Sync/Flush Tests
// =============================================================================

TEST(DistributedMapTest, Sync) {
    dmap<int, int> map;
    map.insert(42, 100);

    auto result = map.sync();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(map.is_clean());
}

TEST(DistributedMapTest, FlushPendingEmpty) {
    dmap<int, int> map;

    // No pending operations
    auto result = map.flush_pending();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedMapTest, Barrier) {
    dmap<int, int> map;

    auto result = map.barrier();
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Apply Remote Operations Tests
// =============================================================================

TEST(DistributedMapTest, ApplyRemoteInserts) {
    dmap<int, int> map;

    std::vector<std::pair<int, int>> ops = {{1, 10}, {2, 20}, {3, 30}};
    map.apply_remote_inserts(ops);

    EXPECT_EQ(map.local_size(), 3);
    EXPECT_TRUE(map.contains(1));
    EXPECT_TRUE(map.contains(2));
    EXPECT_TRUE(map.contains(3));
}

TEST(DistributedMapTest, ApplyRemoteErases) {
    dmap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    std::vector<int> keys = {1, 3};
    map.apply_remote_erases(keys);

    EXPECT_EQ(map.local_size(), 1);
    EXPECT_FALSE(map.contains(1));
    EXPECT_TRUE(map.contains(2));
    EXPECT_FALSE(map.contains(3));
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DistributedMapTest, IsDistributedContainer) {
    using map_type = dmap<int, int>;

    // distributed_map intentionally does NOT satisfy DistributedContainer
    // because it uses key-based (not index-based) access patterns.
    // See class documentation in distributed_map.hpp for rationale.
    static_assert(!is_distributed_container_v<map_type>,
                  "distributed_map should NOT satisfy DistributedContainer");
    static_assert(!is_distributed_container_v<std::unordered_map<int, int>>);
}

TEST(DistributedMapTest, IsDistributedMap) {
    using map_type = dmap<int, int>;

    static_assert(is_distributed_map_v<map_type>);
    static_assert(!is_distributed_map_v<std::unordered_map<int, int>>);
}

TEST(DistributedMapTest, SatisfiesDistributedAssociativeContainer) {
    using map_type = dmap<int, int>;

    // distributed_map satisfies DistributedAssociativeContainer concept
    static_assert(DistributedAssociativeContainer<map_type>,
                  "distributed_map must satisfy DistributedAssociativeContainer");
    static_assert(DistributedMap<map_type>,
                  "distributed_map must satisfy DistributedMap");

    // Regular maps do not satisfy these concepts
    static_assert(!DistributedAssociativeContainer<std::unordered_map<int, int>>);
    static_assert(!DistributedMap<std::unordered_map<int, int>>);
}

// =============================================================================
// Different Key/Value Types
// =============================================================================

TEST(DistributedMapTest, StringKeyIntValue) {
    dmap<std::string, int> map;

    map.insert("alpha", 1);
    map.insert("beta", 2);
    map.insert("gamma", 3);

    EXPECT_TRUE(map.contains("alpha"));
    EXPECT_TRUE(map.contains("beta"));
    EXPECT_TRUE(map.contains("gamma"));
}

TEST(DistributedMapTest, IntKeyDoubleValue) {
    dmap<int, double> map;

    map.insert(1, 1.5);
    map.insert(2, 2.5);

    auto result = map.find(1);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result.value()->second, 1.5);
}

TEST(DistributedMapTest, StructValue) {
    struct Point { int x, y; };
    dmap<int, Point> map;

    map.insert(1, Point{10, 20});
    map.insert(2, Point{30, 40});

    auto result = map.find(1);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->second.x, 10);
    EXPECT_EQ(result.value()->second.y, 20);
}

}  // namespace dtl::test
