// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_map_batch.cpp
/// @brief Unit tests for distributed_map batch operations
/// @details Phase R7: batch_insert and batch_find

#include <dtl/containers/distributed_map.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dtl::test {

// Type alias for default distributed_map
template <typename K, typename V>
using dmap = distributed_map<K, V, std::hash<K>, std::equal_to<K>>;

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// batch_insert Tests
// =============================================================================

TEST(DistributedMapBatchTest, BatchInsertMultipleKeys) {
    dmap<int, int> map;

    std::vector<std::pair<int, int>> entries = {
        {1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}
    };

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);
    EXPECT_EQ(map.local_size(), 5);
}

TEST(DistributedMapBatchTest, BatchInsertEmptyRange) {
    dmap<int, int> map;

    std::vector<std::pair<int, int>> entries;

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
    EXPECT_EQ(map.local_size(), 0);
}

TEST(DistributedMapBatchTest, BatchInsertStringKeys) {
    dmap<std::string, int> map;

    std::vector<std::pair<std::string, int>> entries = {
        {"alpha", 1}, {"beta", 2}, {"gamma", 3}
    };

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 3);
    EXPECT_EQ(map.local_size(), 3);
}

TEST(DistributedMapBatchTest, BatchInsertDuplicateKeysNotInserted) {
    dmap<int, int> map;

    // Insert some keys first
    map.insert(1, 10);
    map.insert(2, 20);

    // Batch insert with overlapping keys
    std::vector<std::pair<int, int>> entries = {
        {2, 200}, {3, 30}, {4, 40}
    };

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    // Key 2 already exists, so only 2 new insertions
    EXPECT_EQ(result.value(), 2);
    EXPECT_EQ(map.local_size(), 4);

    // Original value for key 2 should be preserved (emplace semantics)
    EXPECT_TRUE(map.contains(2));
}

TEST(DistributedMapBatchTest, BatchInsertSingleElement) {
    dmap<int, int> map;

    std::vector<std::pair<int, int>> entries = {{42, 100}};

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
    EXPECT_EQ(map.local_size(), 1);
    EXPECT_TRUE(map.contains(42));
}

TEST(DistributedMapBatchTest, BatchInsertWithContext) {
    test_context ctx{0, 1};
    dmap<int, int> map(ctx);

    std::vector<std::pair<int, int>> entries = {
        {10, 100}, {20, 200}, {30, 300}
    };

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 3);
}

TEST(DistributedMapBatchTest, BatchInsertLargeRange) {
    dmap<int, int> map;

    std::vector<std::pair<int, int>> entries;
    entries.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        entries.emplace_back(i, i * 10);
    }

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1000);
    EXPECT_EQ(map.local_size(), 1000);
}

// =============================================================================
// batch_find Tests
// =============================================================================

TEST(DistributedMapBatchTest, BatchFindExistingKeys) {
    dmap<int, int> map;

    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    std::vector<int> keys = {1, 2, 3};

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 3);
    ASSERT_TRUE(values[0].has_value());
    ASSERT_TRUE(values[1].has_value());
    ASSERT_TRUE(values[2].has_value());
    EXPECT_EQ(values[0].value(), 10);
    EXPECT_EQ(values[1].value(), 20);
    EXPECT_EQ(values[2].value(), 30);
}

TEST(DistributedMapBatchTest, BatchFindMissingKeys) {
    dmap<int, int> map;

    map.insert(1, 10);

    std::vector<int> keys = {1, 99, 100};

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 3);
    ASSERT_TRUE(values[0].has_value());
    EXPECT_EQ(values[0].value(), 10);
    EXPECT_FALSE(values[1].has_value());
    EXPECT_FALSE(values[2].has_value());
}

TEST(DistributedMapBatchTest, BatchFindEmptyRange) {
    dmap<int, int> map;

    map.insert(1, 10);

    std::vector<int> keys;

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().empty());
}

TEST(DistributedMapBatchTest, BatchFindOnEmptyMap) {
    dmap<int, int> map;

    std::vector<int> keys = {1, 2, 3};

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 3);
    EXPECT_FALSE(values[0].has_value());
    EXPECT_FALSE(values[1].has_value());
    EXPECT_FALSE(values[2].has_value());
}

TEST(DistributedMapBatchTest, BatchFindStringKeys) {
    dmap<std::string, int> map;

    map.insert("hello", 1);
    map.insert("world", 2);

    std::vector<std::string> keys = {"hello", "missing", "world"};

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 3);
    ASSERT_TRUE(values[0].has_value());
    EXPECT_EQ(values[0].value(), 1);
    EXPECT_FALSE(values[1].has_value());
    ASSERT_TRUE(values[2].has_value());
    EXPECT_EQ(values[2].value(), 2);
}

TEST(DistributedMapBatchTest, BatchFindAfterBatchInsert) {
    dmap<int, int> map;

    // Batch insert
    std::vector<std::pair<int, int>> entries = {
        {10, 100}, {20, 200}, {30, 300}
    };
    map.batch_insert(entries.begin(), entries.end());

    // Batch find all inserted keys plus some missing
    std::vector<int> keys = {10, 15, 20, 25, 30};

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 5);
    ASSERT_TRUE(values[0].has_value());
    EXPECT_EQ(values[0].value(), 100);
    EXPECT_FALSE(values[1].has_value());  // 15 not found
    ASSERT_TRUE(values[2].has_value());
    EXPECT_EQ(values[2].value(), 200);
    EXPECT_FALSE(values[3].has_value());  // 25 not found
    ASSERT_TRUE(values[4].has_value());
    EXPECT_EQ(values[4].value(), 300);
}

// =============================================================================
// Multi-Rank Simulation Tests (single-rank with remote key awareness)
// =============================================================================

TEST(DistributedMapBatchTest, BatchInsertMultiRankQueuesRemote) {
    test_context ctx{0, 4};  // Rank 0 of 4
    dmap<int, int> map(ctx);

    // With 4 ranks, keys are distributed by hash % 4
    // Some keys will be local (owner == 0), some remote
    std::vector<std::pair<int, int>> entries;
    for (int i = 0; i < 20; ++i) {
        entries.emplace_back(i, i * 10);
    }

    auto result = map.batch_insert(entries.begin(), entries.end());
    ASSERT_TRUE(result.has_value());

    // The result counts only locally inserted keys
    size_type local_count = result.value();
    EXPECT_GT(local_count, 0);
    EXPECT_LE(local_count, 20);
    EXPECT_EQ(map.local_size(), local_count);
}

TEST(DistributedMapBatchTest, BatchFindMultiRankReturnsNulloptForRemote) {
    test_context ctx{0, 4};  // Rank 0 of 4
    dmap<int, int> map(ctx);

    // Insert only local keys
    for (int i = 0; i < 100; ++i) {
        if (map.is_local(i)) {
            map.insert(i, i * 10);
        }
    }

    // Find across all keys (some will be remote)
    std::vector<int> keys;
    for (int i = 0; i < 100; ++i) {
        keys.push_back(i);
    }

    auto result = map.batch_find(keys.begin(), keys.end());
    ASSERT_TRUE(result.has_value());

    auto& values = result.value();
    ASSERT_EQ(values.size(), 100);

    // Local keys should have values, remote keys should be nullopt
    for (int i = 0; i < 100; ++i) {
        if (map.is_local(i)) {
            ASSERT_TRUE(values[static_cast<size_type>(i)].has_value())
                << "Key " << i << " is local but batch_find returned nullopt";
            EXPECT_EQ(values[static_cast<size_type>(i)].value(), i * 10);
        } else {
            EXPECT_FALSE(values[static_cast<size_type>(i)].has_value())
                << "Key " << i << " is remote but batch_find returned a value";
        }
    }
}

}  // namespace dtl::test
