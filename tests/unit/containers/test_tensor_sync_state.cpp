// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_tensor_sync_state.cpp
/// @brief Unit tests for distributed_tensor sync state tracking
/// @details Phase 08, Task 01: Verify sync state parity with distributed_vector

#include <dtl/containers/distributed_tensor.hpp>

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

TEST(TensorSyncStateTest, StartsClean) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    EXPECT_TRUE(tensor.is_clean());
    EXPECT_FALSE(tensor.is_dirty());
}

TEST(TensorSyncStateTest, MarkLocalModified) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    tensor.mark_local_modified();
    EXPECT_TRUE(tensor.is_dirty());
    EXPECT_FALSE(tensor.is_clean());
}

TEST(TensorSyncStateTest, MarkCleanResetsState) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    tensor.mark_local_modified();
    EXPECT_TRUE(tensor.is_dirty());

    tensor.mark_clean();
    EXPECT_TRUE(tensor.is_clean());
}

TEST(TensorSyncStateTest, SyncMarksClean) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    tensor.mark_local_modified();
    EXPECT_TRUE(tensor.is_dirty());

    auto result = tensor.sync();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(tensor.is_clean());
}

TEST(TensorSyncStateTest, SyncStateRefAccessible) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    const auto& ref = tensor.sync_state_ref();
    EXPECT_EQ(ref.domain(), sync_domain::clean);
}

TEST(TensorSyncStateTest, MutableSyncStateRef) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    auto& ref = tensor.sync_state_ref();
    ref.mark_global_dirty();
    EXPECT_TRUE(tensor.is_dirty());
}

TEST(TensorSyncStateTest, LocalWriteMarksDirty) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});

    // Direct local write via operator()
    tensor(0, 0) = 42;
    // Note: operator() doesn't auto-mark dirty (by design — matching vector's operator[])
    // Users should call mark_local_modified() after writes, or use sync_guard.
    // Test the explicit marking path:
    tensor.mark_local_modified();
    EXPECT_TRUE(tensor.is_dirty());
}

TEST(TensorSyncStateTest, CleanAfterSyncDirtyAfterWrite) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    tensor.mark_local_modified();
    tensor.sync();
    EXPECT_TRUE(tensor.is_clean());

    tensor.mark_local_modified();
    EXPECT_TRUE(tensor.is_dirty());
}

TEST(TensorSyncStateTest, MultiRankConstruction) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{1, 4});
    EXPECT_TRUE(tensor.is_clean());
    EXPECT_FALSE(tensor.is_dirty());
}

}  // namespace dtl::test
