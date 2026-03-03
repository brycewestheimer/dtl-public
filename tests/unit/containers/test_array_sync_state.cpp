// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_array_sync_state.cpp
/// @brief Unit tests for distributed_array sync state tracking
/// @details Phase 08, Task 01: Verify sync state parity with distributed_vector

#include <dtl/containers/distributed_array.hpp>

#include <gtest/gtest.h>

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
// Sync State Presence Tests
// =============================================================================

TEST(ArraySyncStateTest, StartsClean) {
    distributed_array<int, 100> arr;
    EXPECT_TRUE(arr.is_clean());
    EXPECT_FALSE(arr.is_dirty());
}

TEST(ArraySyncStateTest, StartsCleanWithContext) {
    distributed_array<int, 100> arr(test_context{0, 4});
    EXPECT_TRUE(arr.is_clean());
    EXPECT_FALSE(arr.is_dirty());
}

TEST(ArraySyncStateTest, FillMarksDirty) {
    distributed_array<int, 100> arr;
    arr.fill(42);
    EXPECT_TRUE(arr.is_dirty());
    EXPECT_FALSE(arr.is_clean());
}

TEST(ArraySyncStateTest, MarkLocalModified) {
    distributed_array<int, 100> arr;
    arr.mark_local_modified();
    EXPECT_TRUE(arr.is_dirty());
}

TEST(ArraySyncStateTest, MarkCleanResetsState) {
    distributed_array<int, 100> arr;
    arr.fill(42);
    EXPECT_TRUE(arr.is_dirty());

    arr.mark_clean();
    EXPECT_TRUE(arr.is_clean());
    EXPECT_FALSE(arr.is_dirty());
}

TEST(ArraySyncStateTest, SyncMarksClean) {
    distributed_array<int, 100> arr;
    arr.fill(42);
    EXPECT_TRUE(arr.is_dirty());

    auto result = arr.sync();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(arr.is_clean());
}

TEST(ArraySyncStateTest, SyncStateRefAccessible) {
    distributed_array<int, 100> arr;
    const auto& ref = arr.sync_state_ref();
    EXPECT_EQ(ref.domain(), sync_domain::clean);

    arr.fill(42);
    EXPECT_NE(ref.domain(), sync_domain::clean);
}

TEST(ArraySyncStateTest, MutableSyncStateRef) {
    distributed_array<int, 100> arr;
    auto& ref = arr.sync_state_ref();
    ref.mark_global_dirty();
    EXPECT_TRUE(arr.is_dirty());
}

TEST(ArraySyncStateTest, MultipleWritesStayDirty) {
    distributed_array<int, 100> arr;
    arr.fill(1);
    arr.fill(2);
    arr.fill(3);
    EXPECT_TRUE(arr.is_dirty());
}

TEST(ArraySyncStateTest, CleanAfterSyncDirtyAfterWrite) {
    distributed_array<int, 100> arr;
    arr.fill(42);
    arr.sync();
    EXPECT_TRUE(arr.is_clean());

    arr.fill(99);
    EXPECT_TRUE(arr.is_dirty());
}

}  // namespace dtl::test
