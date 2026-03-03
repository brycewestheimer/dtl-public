// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_role_manager.cpp
/// @brief Unit tests for role_manager
/// @details Tests for Phase 11.5: MPMD role assignment and management

#include <dtl/mpmd/mpmd.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Construction Tests
// =============================================================================

TEST(RoleManagerTest, DefaultConstruction) {
    role_manager mgr;

    EXPECT_FALSE(mgr.initialized());
    EXPECT_TRUE(mgr.groups().empty());
    EXPECT_TRUE(mgr.my_groups().empty());
}

TEST(RoleManagerTest, ConstructWithConfig) {
    role_manager_config config;
    config.create_communicators = false;
    config.allow_multiple_roles = true;
    config.default_role = node_role::coordinator;

    role_manager mgr(config);

    EXPECT_FALSE(mgr.initialized());
    EXPECT_EQ(mgr.config().default_role, node_role::coordinator);
    EXPECT_FALSE(mgr.config().create_communicators);
}

// =============================================================================
// Role Registration Tests
// =============================================================================

TEST(RoleManagerTest, RegisterRoleDescriptor) {
    role_manager mgr;

    role_descriptor desc(node_role::worker, role_properties{.name = "worker"});
    auto result = mgr.register_role(std::move(desc));

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mgr.descriptors().size(), 1);
}

TEST(RoleManagerTest, RegisterRoleSimple) {
    role_manager mgr;

    auto result = mgr.register_role(
        node_role::coordinator,
        "coordinator",
        role_assignment::first_rank_only()
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mgr.descriptors().size(), 1);
}

TEST(RoleManagerTest, RegisterMultipleRoles) {
    role_manager mgr;

    mgr.register_role(node_role::worker, "worker", role_assignment::all_ranks());
    mgr.register_role(node_role::coordinator, "coordinator", role_assignment::first_rank_only());
    mgr.register_role(node_role::io_handler, "io", role_assignment::last_n_ranks(1));

    EXPECT_EQ(mgr.descriptors().size(), 3);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST(RoleManagerTest, DefaultConfig) {
    role_manager mgr;

    EXPECT_TRUE(mgr.config().create_communicators);
    EXPECT_TRUE(mgr.config().allow_multiple_roles);
    EXPECT_TRUE(mgr.config().validate_requirements);
    EXPECT_EQ(mgr.config().default_role, node_role::worker);
}

TEST(RoleManagerTest, SetConfig) {
    role_manager mgr;

    role_manager_config config;
    config.create_communicators = false;
    config.default_role = node_role::monitor;

    auto result = mgr.set_config(std::move(config));

    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(mgr.config().create_communicators);
    EXPECT_EQ(mgr.config().default_role, node_role::monitor);
}

// =============================================================================
// Manual Group Creation Tests
// =============================================================================

TEST(RoleManagerTest, CreateGroup) {
    role_manager mgr;

    std::vector<rank_t> members = {0, 1, 2, 3};
    auto result = mgr.create_group(node_role::worker, "workers", members);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mgr.groups().size(), 1);
}

TEST(RoleManagerTest, CreateGroupMembers) {
    role_manager mgr;

    std::vector<rank_t> members = {0, 2, 4};
    auto result = mgr.create_group(node_role::worker, "workers", members);

    EXPECT_TRUE(result.has_value());
    auto* group = result.value();
    EXPECT_EQ(group->size(), 3);
    EXPECT_TRUE(group->contains(0));
    EXPECT_TRUE(group->contains(2));
    EXPECT_TRUE(group->contains(4));
    EXPECT_FALSE(group->contains(1));
}

TEST(RoleManagerTest, CreateMultipleGroups) {
    role_manager mgr;

    mgr.create_group(node_role::coordinator, "coordinators", {0});
    mgr.create_group(node_role::worker, "workers", {1, 2, 3});
    mgr.create_group(node_role::io_handler, "io", {4, 5});

    EXPECT_EQ(mgr.groups().size(), 3);
}

TEST(RoleManagerTest, GetGroupByRole) {
    role_manager mgr;

    mgr.create_group(node_role::worker, "workers", {0, 1, 2, 3});

    auto* group = mgr.get_group(node_role::worker);

    EXPECT_NE(group, nullptr);
    EXPECT_EQ(group->size(), 4);
    EXPECT_EQ(group->role(), node_role::worker);
}

TEST(RoleManagerTest, GetGroupByRoleNonExistent) {
    role_manager mgr;

    auto* group = mgr.get_group(node_role::coordinator);

    EXPECT_EQ(group, nullptr);
}

TEST(RoleManagerTest, GetGroupById) {
    role_manager mgr;

    mgr.create_group(node_role::worker, "workers", {0, 1});
    mgr.create_group(node_role::coordinator, "coordinators", {2, 3});

    auto* group = mgr.get_group_by_id(1);  // Second group has ID 1

    EXPECT_NE(group, nullptr);
    EXPECT_EQ(group->role(), node_role::coordinator);
}

TEST(RoleManagerTest, GetGroupByIdInvalid) {
    role_manager mgr;

    auto* group = mgr.get_group_by_id(99);

    EXPECT_EQ(group, nullptr);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST(RoleManagerTest, Reset) {
    role_manager mgr;

    mgr.create_group(node_role::worker, "workers", {0, 1, 2});

    EXPECT_EQ(mgr.groups().size(), 1);

    mgr.reset();

    EXPECT_FALSE(mgr.initialized());
    EXPECT_TRUE(mgr.groups().empty());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(RoleManagerTest, MoveConstruction) {
    role_manager mgr1;
    mgr1.register_role(node_role::worker, "worker", role_assignment::all_ranks());

    role_manager mgr2(std::move(mgr1));

    EXPECT_EQ(mgr2.descriptors().size(), 1);
}

TEST(RoleManagerTest, MoveAssignment) {
    role_manager mgr1;
    mgr1.register_role(node_role::worker, "worker", role_assignment::all_ranks());

    role_manager mgr2;
    mgr2 = std::move(mgr1);

    EXPECT_EQ(mgr2.descriptors().size(), 1);
}

// =============================================================================
// Role Descriptor Tests
// =============================================================================

TEST(RoleDescriptorTest, Construction) {
    role_descriptor desc(node_role::worker, role_properties{.name = "worker"});

    EXPECT_EQ(desc.role(), node_role::worker);
    EXPECT_EQ(desc.properties().name, "worker");
}

TEST(RoleDescriptorTest, ConstructionWithAssigner) {
    role_descriptor desc(
        node_role::coordinator,
        role_properties{.name = "coordinator"},
        role_assignment::first_rank_only()
    );

    EXPECT_EQ(desc.role(), node_role::coordinator);
    EXPECT_TRUE(desc.should_assign(0, 4));   // Rank 0 is coordinator
    EXPECT_FALSE(desc.should_assign(1, 4));  // Rank 1 is not
}

TEST(RoleDescriptorTest, ShouldAssignAllRanks) {
    role_descriptor desc(
        node_role::worker,
        role_properties{.name = "worker"},
        role_assignment::all_ranks()
    );

    for (rank_t r = 0; r < 8; ++r) {
        EXPECT_TRUE(desc.should_assign(r, 8));
    }
}

TEST(RoleDescriptorTest, ShouldAssignNoRanks) {
    role_descriptor desc(
        node_role::monitor,
        role_properties{.name = "monitor"},
        role_assignment::no_ranks()
    );

    for (rank_t r = 0; r < 8; ++r) {
        EXPECT_FALSE(desc.should_assign(r, 8));
    }
}

// =============================================================================
// Assignment Strategy Tests
// =============================================================================

TEST(RoleAssignmentStrategyTest, FirstRankOnly) {
    auto assigner = role_assignment::first_rank_only();

    EXPECT_TRUE(assigner(0, 4));
    EXPECT_FALSE(assigner(1, 4));
    EXPECT_FALSE(assigner(3, 4));
}

TEST(RoleAssignmentStrategyTest, LastRankOnly) {
    auto assigner = role_assignment::last_rank_only();

    EXPECT_FALSE(assigner(0, 4));
    EXPECT_FALSE(assigner(2, 4));
    EXPECT_TRUE(assigner(3, 4));
}

TEST(RoleAssignmentStrategyTest, FirstNRanks) {
    auto assigner = role_assignment::first_n_ranks(2);

    EXPECT_TRUE(assigner(0, 8));
    EXPECT_TRUE(assigner(1, 8));
    EXPECT_FALSE(assigner(2, 8));
    EXPECT_FALSE(assigner(7, 8));
}

TEST(RoleAssignmentStrategyTest, LastNRanks) {
    auto assigner = role_assignment::last_n_ranks(2);

    EXPECT_FALSE(assigner(0, 8));
    EXPECT_FALSE(assigner(5, 8));
    EXPECT_TRUE(assigner(6, 8));
    EXPECT_TRUE(assigner(7, 8));
}

TEST(RoleAssignmentStrategyTest, RankRange) {
    auto assigner = role_assignment::rank_range(2, 5);

    EXPECT_FALSE(assigner(0, 8));
    EXPECT_FALSE(assigner(1, 8));
    EXPECT_TRUE(assigner(2, 8));
    EXPECT_TRUE(assigner(3, 8));
    EXPECT_TRUE(assigner(4, 8));
    EXPECT_FALSE(assigner(5, 8));
}

TEST(RoleAssignmentStrategyTest, EveryNthRank) {
    auto assigner = role_assignment::every_nth_rank(2, 0);

    EXPECT_TRUE(assigner(0, 8));
    EXPECT_FALSE(assigner(1, 8));
    EXPECT_TRUE(assigner(2, 8));
    EXPECT_FALSE(assigner(3, 8));
    EXPECT_TRUE(assigner(4, 8));
}

TEST(RoleAssignmentStrategyTest, EveryNthRankWithOffset) {
    auto assigner = role_assignment::every_nth_rank(3, 1);

    EXPECT_FALSE(assigner(0, 8));
    EXPECT_TRUE(assigner(1, 8));
    EXPECT_FALSE(assigner(2, 8));
    EXPECT_FALSE(assigner(3, 8));
    EXPECT_TRUE(assigner(4, 8));
}

TEST(RoleAssignmentStrategyTest, SpecificRanks) {
    auto assigner = role_assignment::specific_ranks({0, 3, 7});

    EXPECT_TRUE(assigner(0, 8));
    EXPECT_FALSE(assigner(1, 8));
    EXPECT_FALSE(assigner(2, 8));
    EXPECT_TRUE(assigner(3, 8));
    EXPECT_TRUE(assigner(7, 8));
}

// =============================================================================
// Role Properties Tests
// =============================================================================

TEST(RolePropertiesTest, DefaultValues) {
    role_properties props;

    EXPECT_TRUE(props.name.empty());
    EXPECT_TRUE(props.description.empty());
    EXPECT_EQ(props.min_ranks, 0);
    EXPECT_EQ(props.max_ranks, 0);
    EXPECT_FALSE(props.required);
    EXPECT_FALSE(props.prefer_separate_nodes);
    EXPECT_FALSE(props.requires_gpu);
    EXPECT_EQ(props.memory_hint, 0);
}

TEST(RolePropertiesTest, NamedProperties) {
    role_properties props{
        .name = "worker",
        .description = "Worker node",
        .min_ranks = 1,
        .max_ranks = 100,
        .required = true
    };

    EXPECT_EQ(props.name, "worker");
    EXPECT_EQ(props.description, "Worker node");
    EXPECT_EQ(props.min_ranks, 1);
    EXPECT_EQ(props.max_ranks, 100);
    EXPECT_TRUE(props.required);
}

// =============================================================================
// Node Role Trait Tests
// =============================================================================

TEST(NodeRoleTest, RoleNamePredefined) {
    EXPECT_EQ(role_name(node_role::worker), "worker");
    EXPECT_EQ(role_name(node_role::coordinator), "coordinator");
    EXPECT_EQ(role_name(node_role::io_handler), "io_handler");
    EXPECT_EQ(role_name(node_role::aggregator), "aggregator");
    EXPECT_EQ(role_name(node_role::monitor), "monitor");
}

TEST(NodeRoleTest, RoleNameCustom) {
    node_role custom = make_custom_role(0);
    EXPECT_EQ(role_name(custom), "custom");
}

TEST(NodeRoleTest, IsPredefinedRole) {
    EXPECT_TRUE(is_predefined_role(node_role::worker));
    EXPECT_TRUE(is_predefined_role(node_role::coordinator));
    EXPECT_TRUE(is_predefined_role(node_role::io_handler));
    EXPECT_FALSE(is_predefined_role(make_custom_role(1)));
}

TEST(NodeRoleTest, IsCustomRole) {
    EXPECT_FALSE(is_custom_role(node_role::worker));
    EXPECT_TRUE(is_custom_role(make_custom_role(1)));
    EXPECT_TRUE(is_custom_role(make_custom_role(999)));
}

TEST(NodeRoleTest, MakeCustomRole) {
    node_role custom = make_custom_role(42);
    EXPECT_EQ(custom_role_id(custom), 42);

    node_role custom2 = make_custom_role(100);
    EXPECT_EQ(custom_role_id(custom2), 100);
}

TEST(NodeRoleTest, CustomRoleRoundtrip) {
    for (role_id id = 0; id < 10; ++id) {
        node_role custom = make_custom_role(id);
        EXPECT_TRUE(is_custom_role(custom));
        EXPECT_EQ(custom_role_id(custom), id);
    }
}

}  // namespace dtl::test
