// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_sync_domain.cpp
/// @brief Tests for sync_domain and dirty state tracking

#include <gtest/gtest.h>

#include <dtl/core/sync_domain.hpp>

#include <thread>
#include <atomic>

namespace dtl::test {

// ============================================================================
// Sync Domain Enum Tests
// ============================================================================

TEST(SyncDomain, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(sync_domain::clean), 0);
    EXPECT_EQ(static_cast<uint8_t>(sync_domain::local_dirty), 1);
    EXPECT_EQ(static_cast<uint8_t>(sync_domain::halo), 2);
    EXPECT_EQ(static_cast<uint8_t>(sync_domain::global_dirty), 3);
}

TEST(SyncDomain, RequiresCommunication) {
    EXPECT_FALSE(requires_communication(sync_domain::clean));
    EXPECT_FALSE(requires_communication(sync_domain::local_dirty));
    EXPECT_TRUE(requires_communication(sync_domain::halo));
    EXPECT_TRUE(requires_communication(sync_domain::global_dirty));
}

TEST(SyncDomain, IsAtLeastAsDirty) {
    EXPECT_TRUE(is_at_least_as_dirty(sync_domain::global_dirty, sync_domain::clean));
    EXPECT_TRUE(is_at_least_as_dirty(sync_domain::global_dirty, sync_domain::local_dirty));
    EXPECT_TRUE(is_at_least_as_dirty(sync_domain::global_dirty, sync_domain::halo));
    EXPECT_TRUE(is_at_least_as_dirty(sync_domain::global_dirty, sync_domain::global_dirty));

    EXPECT_FALSE(is_at_least_as_dirty(sync_domain::clean, sync_domain::local_dirty));
    EXPECT_FALSE(is_at_least_as_dirty(sync_domain::local_dirty, sync_domain::halo));
}

TEST(SyncDomain, MaxDomain) {
    EXPECT_EQ(max_domain(sync_domain::clean, sync_domain::local_dirty), sync_domain::local_dirty);
    EXPECT_EQ(max_domain(sync_domain::local_dirty, sync_domain::halo), sync_domain::halo);
    EXPECT_EQ(max_domain(sync_domain::global_dirty, sync_domain::local_dirty), sync_domain::global_dirty);
    EXPECT_EQ(max_domain(sync_domain::clean, sync_domain::clean), sync_domain::clean);
}

// ============================================================================
// Dirty Flags Tests
// ============================================================================

TEST(DirtyFlags, DefaultConstruction) {
    dirty_flags flags;

    EXPECT_FALSE(flags.local_modified);
    EXPECT_FALSE(flags.halo_stale);
    EXPECT_FALSE(flags.remote_stale);
    EXPECT_FALSE(flags.structure_changed);
    EXPECT_FALSE(flags.any());
    EXPECT_TRUE(flags.is_clean());
}

TEST(DirtyFlags, SetFlags) {
    dirty_flags flags;

    flags.local_modified = true;
    EXPECT_TRUE(flags.any());
    EXPECT_FALSE(flags.is_clean());
    EXPECT_EQ(flags.to_domain(), sync_domain::local_dirty);

    flags.halo_stale = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::halo);

    flags.remote_stale = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::global_dirty);
}

TEST(DirtyFlags, Clear) {
    dirty_flags flags;
    flags.local_modified = true;
    flags.halo_stale = true;
    flags.remote_stale = true;
    flags.structure_changed = true;

    EXPECT_TRUE(flags.any());

    flags.clear();

    EXPECT_FALSE(flags.any());
    EXPECT_TRUE(flags.is_clean());
}

TEST(DirtyFlags, ToDomain) {
    dirty_flags flags;

    // Clean
    EXPECT_EQ(flags.to_domain(), sync_domain::clean);

    // Local only
    flags.local_modified = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::local_dirty);

    // Halo
    flags.clear();
    flags.halo_stale = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::halo);

    // Global (remote_stale)
    flags.clear();
    flags.remote_stale = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::global_dirty);

    // Global (structure_changed)
    flags.clear();
    flags.structure_changed = true;
    EXPECT_EQ(flags.to_domain(), sync_domain::global_dirty);
}

// ============================================================================
// Sync State Tests
// ============================================================================

TEST(SyncState, DefaultConstruction) {
    sync_state state;

    EXPECT_EQ(state.domain(), sync_domain::clean);
    EXPECT_TRUE(state.is_clean());
    EXPECT_FALSE(state.is_dirty());
    EXPECT_FALSE(state.needs_communication());
}

TEST(SyncState, MarkLocalModified) {
    sync_state state;

    state.mark_local_modified();

    EXPECT_EQ(state.domain(), sync_domain::local_dirty);
    EXPECT_TRUE(state.is_dirty());
    EXPECT_FALSE(state.is_clean());
    EXPECT_FALSE(state.needs_communication());
}

TEST(SyncState, MarkHaloStale) {
    sync_state state;

    state.mark_halo_stale();

    EXPECT_EQ(state.domain(), sync_domain::halo);
    EXPECT_TRUE(state.is_dirty());
    EXPECT_TRUE(state.needs_communication());
}

TEST(SyncState, MarkGlobalDirty) {
    sync_state state;

    state.mark_global_dirty();

    EXPECT_EQ(state.domain(), sync_domain::global_dirty);
    EXPECT_TRUE(state.is_dirty());
    EXPECT_TRUE(state.needs_communication());
}

TEST(SyncState, MarkClean) {
    sync_state state;
    state.mark_global_dirty();

    state.mark_clean();

    EXPECT_EQ(state.domain(), sync_domain::clean);
    EXPECT_TRUE(state.is_clean());
}

TEST(SyncState, MarkHaloSynced) {
    sync_state state;
    state.mark_halo_stale();

    state.mark_halo_synced();

    EXPECT_EQ(state.domain(), sync_domain::local_dirty);
}

TEST(SyncState, DirtyLevelProgression) {
    sync_state state;

    // Clean -> local
    state.mark_local_modified();
    EXPECT_EQ(state.domain(), sync_domain::local_dirty);

    // local -> halo
    state.mark_halo_stale();
    EXPECT_EQ(state.domain(), sync_domain::halo);

    // halo -> global
    state.mark_global_dirty();
    EXPECT_EQ(state.domain(), sync_domain::global_dirty);

    // global doesn't go back to halo
    state.mark_halo_stale();
    EXPECT_EQ(state.domain(), sync_domain::global_dirty);

    // Must explicitly clean
    state.mark_clean();
    EXPECT_EQ(state.domain(), sync_domain::clean);
}

TEST(SyncState, SetDomainDirectly) {
    sync_state state;

    state.set_domain(sync_domain::halo);
    EXPECT_EQ(state.domain(), sync_domain::halo);

    state.set_domain(sync_domain::clean);
    EXPECT_EQ(state.domain(), sync_domain::clean);
}

TEST(SyncState, ThreadSafety) {
    sync_state state;
    std::atomic<int> counter{0};

    auto writer = [&]() {
        for (int i = 0; i < 1000; ++i) {
            state.mark_local_modified();
            state.mark_halo_stale();
            state.mark_global_dirty();
            state.mark_clean();
            counter.fetch_add(1);
        }
    };

    auto reader = [&]() {
        for (int i = 0; i < 1000; ++i) {
            volatile auto domain = state.domain();
            volatile bool dirty = state.is_dirty();
            volatile bool needs_comm = state.needs_communication();
            (void)domain;
            (void)dirty;
            (void)needs_comm;
            counter.fetch_add(1);
        }
    };

    std::thread t1(writer);
    std::thread t2(reader);
    std::thread t3(writer);
    std::thread t4(reader);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    EXPECT_EQ(counter.load(), 4000);
}

// ============================================================================
// Stale Policy Tests
// ============================================================================

TEST(StalePolicy, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(stale_policy::allow), 0);
    EXPECT_EQ(static_cast<uint8_t>(stale_policy::warn), 1);
    EXPECT_EQ(static_cast<uint8_t>(stale_policy::error), 2);
    EXPECT_EQ(static_cast<uint8_t>(stale_policy::auto_sync), 3);
}

// ============================================================================
// Loudness Level Tests
// ============================================================================

TEST(LoudnessLevel, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(loudness_level::l0_stl_compatible), 0);
    EXPECT_EQ(static_cast<uint8_t>(loudness_level::l1_syntactically_loud), 1);
    EXPECT_EQ(static_cast<uint8_t>(loudness_level::l2_statically_detectable), 2);
    EXPECT_EQ(static_cast<uint8_t>(loudness_level::l3_documented), 3);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(RequireClean, CleanContainer) {
    // Mock a syncable container via sync_state directly
    sync_state state;

    // When clean, should succeed
    EXPECT_TRUE(state.is_clean());
}

TEST(RequireClean, DirtyContainer) {
    sync_state state;
    state.mark_local_modified();

    EXPECT_TRUE(state.is_dirty());
}

}  // namespace dtl::test
