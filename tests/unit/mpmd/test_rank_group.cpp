// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rank_group.cpp
/// @brief Unit tests for rank_group
/// @details Tests for Phase 11.5: MPMD rank group management

#include <dtl/mpmd/mpmd.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// =============================================================================
// Construction Tests
// =============================================================================

TEST(RankGroupTest, DefaultConstruction) {
    rank_group group;

    EXPECT_FALSE(group.valid());
    EXPECT_TRUE(group.empty());
    EXPECT_EQ(group.size(), 0);
    EXPECT_EQ(group.id(), rank_group::invalid_group_id);
}

TEST(RankGroupTest, ConstructWithIdAndRole) {
    rank_group group(42, node_role::worker);

    EXPECT_TRUE(group.valid());
    EXPECT_EQ(group.id(), 42);
    EXPECT_EQ(group.role(), node_role::worker);
    EXPECT_TRUE(group.empty());
}

TEST(RankGroupTest, ConstructWithIdRoleAndName) {
    rank_group group(1, node_role::coordinator, "workers");

    EXPECT_TRUE(group.valid());
    EXPECT_EQ(group.id(), 1);
    EXPECT_EQ(group.role(), node_role::coordinator);
    EXPECT_EQ(group.name(), "workers");
}

// =============================================================================
// Member Management Tests
// =============================================================================

TEST(RankGroupTest, AddMember) {
    rank_group group(0, node_role::worker);

    group.add_member(5);

    EXPECT_EQ(group.size(), 1);
    EXPECT_TRUE(group.contains(5));
}

TEST(RankGroupTest, AddMultipleMembers) {
    rank_group group(0, node_role::worker);

    group.add_member(0);
    group.add_member(1);
    group.add_member(2);

    EXPECT_EQ(group.size(), 3);
    EXPECT_TRUE(group.contains(0));
    EXPECT_TRUE(group.contains(1));
    EXPECT_TRUE(group.contains(2));
}

TEST(RankGroupTest, ClearMembers) {
    rank_group group(0, node_role::worker);

    group.add_member(0);
    group.add_member(1);
    group.add_member(2);

    EXPECT_EQ(group.size(), 3);

    group.clear_members();

    EXPECT_TRUE(group.empty());
    EXPECT_EQ(group.size(), 0);
}

// =============================================================================
// Membership Query Tests
// =============================================================================

TEST(RankGroupTest, Contains) {
    rank_group group(0, node_role::worker);

    group.add_member(0);
    group.add_member(5);
    group.add_member(10);

    EXPECT_TRUE(group.contains(0));
    EXPECT_TRUE(group.contains(5));
    EXPECT_TRUE(group.contains(10));
    EXPECT_FALSE(group.contains(1));
    EXPECT_FALSE(group.contains(99));
}

TEST(RankGroupTest, ContainsEmptyGroup) {
    rank_group group(0, node_role::worker);

    EXPECT_FALSE(group.contains(0));
    EXPECT_FALSE(group.contains(1));
}

TEST(RankGroupTest, LocalRankDefault) {
    rank_group group(0, node_role::worker);

    EXPECT_EQ(group.local_rank(), no_rank);
    EXPECT_FALSE(group.is_member());
}

TEST(RankGroupTest, SetLocalRank) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.set_local_rank(1);  // Local rank 1 in the group

    EXPECT_EQ(group.local_rank(), 1);
    EXPECT_TRUE(group.is_member());
}

// =============================================================================
// Rank Translation Tests
// =============================================================================

TEST(RankGroupTest, ToLocalRank) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);
    group.add_member(20);

    // World rank 5 is local rank 0
    EXPECT_EQ(group.to_local_rank(5), 0);
    // World rank 10 is local rank 1
    EXPECT_EQ(group.to_local_rank(10), 1);
    // World rank 15 is local rank 2
    EXPECT_EQ(group.to_local_rank(15), 2);
    // World rank 20 is local rank 3
    EXPECT_EQ(group.to_local_rank(20), 3);
}

TEST(RankGroupTest, ToLocalRankNonMember) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);

    // Non-member should return no_rank
    EXPECT_EQ(group.to_local_rank(99), no_rank);
}

TEST(RankGroupTest, ToWorldRank) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);
    group.add_member(20);

    // Local rank 0 is world rank 5
    EXPECT_EQ(group.to_world_rank(0), 5);
    // Local rank 1 is world rank 10
    EXPECT_EQ(group.to_world_rank(1), 10);
    // Local rank 2 is world rank 15
    EXPECT_EQ(group.to_world_rank(2), 15);
    // Local rank 3 is world rank 20
    EXPECT_EQ(group.to_world_rank(3), 20);
}

TEST(RankGroupTest, ToWorldRankInvalid) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);

    // Out of bounds local rank
    EXPECT_EQ(group.to_world_rank(99), no_rank);
}

TEST(RankGroupTest, RankTranslationRoundtrip) {
    rank_group group(0, node_role::worker);

    group.add_member(2);
    group.add_member(7);
    group.add_member(12);
    group.add_member(17);
    group.add_member(22);

    for (rank_t local = 0; local < static_cast<rank_t>(group.size()); ++local) {
        rank_t world = group.to_world_rank(local);
        rank_t back = group.to_local_rank(world);
        EXPECT_EQ(back, local);
    }
}

// =============================================================================
// Leader Tests
// =============================================================================

TEST(RankGroupTest, LeaderDefault) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);

    // First member is leader
    EXPECT_EQ(group.leader(), 5);
}

TEST(RankGroupTest, LeaderEmptyGroup) {
    rank_group group(0, node_role::worker);

    // No leader for empty group
    EXPECT_EQ(group.leader(), no_rank);
}

TEST(RankGroupTest, IsLeader) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.set_local_rank(0);  // We are local rank 0 = leader

    EXPECT_TRUE(group.is_leader());

    group.set_local_rank(1);  // We are local rank 1 = not leader

    EXPECT_FALSE(group.is_leader());
}

// =============================================================================
// Members Vector Access
// =============================================================================

TEST(RankGroupTest, MembersVector) {
    rank_group group(0, node_role::worker);

    group.add_member(5);
    group.add_member(10);
    group.add_member(15);

    const auto& members = group.members();

    EXPECT_EQ(members.size(), 3);
    EXPECT_EQ(members[0], 5);
    EXPECT_EQ(members[1], 10);
    EXPECT_EQ(members[2], 15);
}

// =============================================================================
// Role Tests
// =============================================================================

TEST(RankGroupTest, RoleAccess) {
    rank_group group1(1, node_role::worker);
    rank_group group2(2, node_role::coordinator);
    rank_group group3(3, node_role::io_handler);

    EXPECT_EQ(group1.role(), node_role::worker);
    EXPECT_EQ(group2.role(), node_role::coordinator);
    EXPECT_EQ(group3.role(), node_role::io_handler);
}

// =============================================================================
// Size Tests
// =============================================================================

TEST(RankGroupTest, Empty) {
    rank_group group(0, node_role::worker);

    EXPECT_TRUE(group.empty());
    EXPECT_EQ(group.size(), 0);

    group.add_member(5);

    EXPECT_FALSE(group.empty());
    EXPECT_EQ(group.size(), 1);
}

// =============================================================================
// Boolean Conversion Tests
// =============================================================================

TEST(RankGroupTest, BoolConversionValid) {
    rank_group group(0, node_role::worker);

    EXPECT_TRUE(static_cast<bool>(group));
}

TEST(RankGroupTest, BoolConversionInvalid) {
    rank_group group;

    EXPECT_FALSE(static_cast<bool>(group));
}

// =============================================================================
// Communicator Tests
// =============================================================================

TEST(RankGroupTest, NoCommunicatorInitially) {
    rank_group group(0, node_role::worker);

    EXPECT_FALSE(group.has_communicator());
    EXPECT_EQ(group.communicator(), nullptr);
}

// =============================================================================
// Comparison Tests
// =============================================================================

TEST(RankGroupTest, EqualityById) {
    rank_group group1(42, node_role::worker);
    rank_group group2(42, node_role::coordinator);  // Different role, same ID
    rank_group group3(99, node_role::worker);

    EXPECT_EQ(group1, group2);  // Equality is by ID
    EXPECT_NE(group1, group3);
}

TEST(RankGroupTest, LessThanComparison) {
    rank_group group1(10, node_role::worker);
    rank_group group2(20, node_role::worker);

    EXPECT_TRUE(group1 < group2);
    EXPECT_FALSE(group2 < group1);
}

// =============================================================================
// Group Membership Struct Tests
// =============================================================================

TEST(GroupMembershipTest, DefaultConstruction) {
    group_membership membership;

    EXPECT_EQ(membership.group, nullptr);
    EXPECT_EQ(membership.local_rank, no_rank);
    EXPECT_FALSE(membership.valid());
    EXPECT_FALSE(static_cast<bool>(membership));
}

TEST(GroupMembershipTest, ValidMembership) {
    rank_group group(1, node_role::worker);
    group.add_member(0);

    group_membership membership;
    membership.group = &group;
    membership.local_rank = 0;

    EXPECT_TRUE(membership.valid());
    EXPECT_TRUE(static_cast<bool>(membership));
}

// =============================================================================
// Rank Info Struct Tests
// =============================================================================

TEST(RankInfoTest, DefaultConstruction) {
    rank_info info;

    EXPECT_EQ(info.world_rank, no_rank);
    EXPECT_TRUE(info.memberships.empty());
    EXPECT_EQ(info.primary_role, node_role::undefined);
}

TEST(RankInfoTest, HasRole) {
    rank_group group(1, node_role::worker);
    group.add_member(0);

    rank_info info;
    info.world_rank = 0;
    group_membership membership;
    membership.group = &group;
    membership.local_rank = 0;
    info.memberships.push_back(membership);

    EXPECT_TRUE(info.has_role(node_role::worker));
    EXPECT_FALSE(info.has_role(node_role::coordinator));
}

TEST(RankInfoTest, GroupForRole) {
    rank_group group(1, node_role::worker);
    group.add_member(0);

    rank_info info;
    info.world_rank = 0;
    group_membership membership;
    membership.group = &group;
    membership.local_rank = 0;
    info.memberships.push_back(membership);

    EXPECT_EQ(info.group_for_role(node_role::worker), &group);
    EXPECT_EQ(info.group_for_role(node_role::coordinator), nullptr);
}

}  // namespace dtl::test
