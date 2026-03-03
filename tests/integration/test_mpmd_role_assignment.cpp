// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpmd_role_assignment.cpp
/// @brief Integration tests for role_manager.initialize() with mock communicator
/// @details Tests for Phase 12C: MPMD role assignment without real MPI

#include <dtl/mpmd/mpmd.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// =============================================================================
// Mock Communicator
// =============================================================================

/// @brief Mock communicator extending communicator_base for testing without MPI
/// @details Simulates a communicator with configurable rank and size.
class MockCommunicator : public dtl::communicator_base {
public:
    MockCommunicator(dtl::rank_t rank, dtl::rank_t size)
        : rank_(rank), size_(size) {}

    [[nodiscard]] dtl::rank_t rank() const noexcept override { return rank_; }
    [[nodiscard]] dtl::rank_t size() const noexcept override { return size_; }

    [[nodiscard]] dtl::communicator_properties properties() const noexcept override {
        return dtl::communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "MockCommunicator"
        };
    }

private:
    dtl::rank_t rank_;
    dtl::rank_t size_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class RoleAssignmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default: simulate rank 0 of 4
        comm_ = std::make_unique<MockCommunicator>(0, 4);
    }

    std::unique_ptr<MockCommunicator> comm_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(RoleAssignmentTest, DefaultConstruction) {
    role_manager mgr;

    EXPECT_FALSE(mgr.initialized());
    EXPECT_TRUE(mgr.groups().empty());
    EXPECT_TRUE(mgr.my_groups().empty());
    EXPECT_EQ(mgr.primary_role(), node_role::undefined);
}

// =============================================================================
// Registration Tests
// =============================================================================

TEST_F(RoleAssignmentTest, RegisterRole) {
    role_manager mgr;

    auto result = mgr.register_role(
        node_role::worker, "worker", role_assignment::all_ranks());

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mgr.descriptors().size(), 1);
}

TEST_F(RoleAssignmentTest, RegisterMultipleRoles) {
    role_manager mgr;

    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 1; });
    mgr.register_role(node_role::io_handler, "io",
                      role_assignment::last_rank_only());

    EXPECT_EQ(mgr.descriptors().size(), 3);
}

// =============================================================================
// Initialization State Tests
// =============================================================================

TEST_F(RoleAssignmentTest, InitializeNotCalled) {
    role_manager mgr;

    EXPECT_FALSE(mgr.initialized());
}

TEST_F(RoleAssignmentTest, InitializeWithMockComm) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    auto result = mgr.initialize(*comm_);

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(mgr.initialized());
}

// =============================================================================
// Worker/Coordinator Setup Tests
// =============================================================================

TEST_F(RoleAssignmentTest, WorkerCoordinatorSetup) {
    role_manager mgr;
    auto setup_result = setup_worker_coordinator(mgr, 1);
    EXPECT_TRUE(setup_result.has_value());

    auto init_result = mgr.initialize(*comm_);
    EXPECT_TRUE(init_result.has_value());

    // Rank 0 should be coordinator
    EXPECT_TRUE(mgr.has_role(node_role::coordinator));
    // Rank 0 should NOT be a worker (coordinator_count=1, workers are rank >= 1)
    EXPECT_FALSE(mgr.has_role(node_role::worker));
}

// =============================================================================
// Role Query Tests
// =============================================================================

TEST_F(RoleAssignmentTest, HasRoleAfterInit) {
    role_manager mgr;
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 1; });

    mgr.initialize(*comm_);

    // Rank 0 is coordinator
    EXPECT_TRUE(mgr.has_role(node_role::coordinator));
    // Rank 0 is NOT worker (workers are rank >= 1)
    EXPECT_FALSE(mgr.has_role(node_role::worker));
    // Rank 0 is NOT io_handler
    EXPECT_FALSE(mgr.has_role(node_role::io_handler));
}

TEST_F(RoleAssignmentTest, PrimaryRole) {
    role_manager mgr;
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 1; });

    mgr.initialize(*comm_);

    // Rank 0's primary role should be coordinator (non-worker preferred)
    EXPECT_EQ(mgr.primary_role(), node_role::coordinator);
}

// =============================================================================
// Group Access Tests
// =============================================================================

TEST_F(RoleAssignmentTest, GetGroup) {
    role_manager mgr;
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 1; });

    mgr.initialize(*comm_);

    auto* coord_group = mgr.get_group(node_role::coordinator);
    ASSERT_NE(coord_group, nullptr);
    EXPECT_EQ(coord_group->role(), node_role::coordinator);
    EXPECT_EQ(coord_group->size(), 1);  // Only rank 0

    auto* worker_group = mgr.get_group(node_role::worker);
    ASSERT_NE(worker_group, nullptr);
    EXPECT_EQ(worker_group->role(), node_role::worker);
    EXPECT_EQ(worker_group->size(), 3);  // Ranks 1, 2, 3
}

TEST_F(RoleAssignmentTest, GetGroupNotFound) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    mgr.initialize(*comm_);

    auto* group = mgr.get_group(node_role::io_handler);
    EXPECT_EQ(group, nullptr);
}

// =============================================================================
// my_groups Tests
// =============================================================================

TEST_F(RoleAssignmentTest, MyGroupsPopulated) {
    role_manager mgr;
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 1; });

    mgr.initialize(*comm_);

    // Rank 0 belongs to coordinator group only
    const auto& my_groups = mgr.my_groups();
    EXPECT_EQ(my_groups.size(), 1);
    EXPECT_EQ(my_groups[0]->role(), node_role::coordinator);
}

// =============================================================================
// Group Membership Tests
// =============================================================================

TEST_F(RoleAssignmentTest, GroupMembers) {
    role_manager mgr;
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_n_ranks(2));
    mgr.register_role(node_role::worker, "worker",
                      [](rank_t r, rank_t) { return r >= 2; });

    mgr.initialize(*comm_);

    auto* coord_group = mgr.get_group(node_role::coordinator);
    ASSERT_NE(coord_group, nullptr);

    const auto& members = coord_group->members();
    EXPECT_EQ(members.size(), 2);
    EXPECT_EQ(members[0], 0);
    EXPECT_EQ(members[1], 1);
}

TEST_F(RoleAssignmentTest, GroupSize) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    mgr.initialize(*comm_);

    auto* worker_group = mgr.get_group(node_role::worker);
    ASSERT_NE(worker_group, nullptr);
    EXPECT_EQ(worker_group->size(), 4);  // All 4 ranks
}

// =============================================================================
// Double Init / Reset Tests
// =============================================================================

TEST_F(RoleAssignmentTest, DoubleInitRejects) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    auto first = mgr.initialize(*comm_);
    EXPECT_TRUE(first.has_value());

    auto second = mgr.initialize(*comm_);
    EXPECT_TRUE(second.has_error());
}

TEST_F(RoleAssignmentTest, ResetWorks) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    mgr.initialize(*comm_);
    EXPECT_TRUE(mgr.initialized());

    mgr.reset();
    EXPECT_FALSE(mgr.initialized());
    EXPECT_TRUE(mgr.groups().empty());
    EXPECT_TRUE(mgr.my_groups().empty());
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(RoleAssignmentTest, ConfigAccess) {
    role_manager_config config;
    config.create_communicators = false;
    config.validate_requirements = false;
    config.default_role = node_role::monitor;

    role_manager mgr(config);

    EXPECT_FALSE(mgr.config().create_communicators);
    EXPECT_FALSE(mgr.config().validate_requirements);
    EXPECT_EQ(mgr.config().default_role, node_role::monitor);
}

TEST_F(RoleAssignmentTest, SetConfigBeforeInit) {
    role_manager mgr;

    role_manager_config config;
    config.default_role = node_role::aggregator;

    auto result = mgr.set_config(std::move(config));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(mgr.config().default_role, node_role::aggregator);
}

TEST_F(RoleAssignmentTest, SetConfigAfterInit) {
    role_manager mgr;
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());
    mgr.initialize(*comm_);

    role_manager_config config;
    config.default_role = node_role::aggregator;

    auto result = mgr.set_config(std::move(config));
    EXPECT_TRUE(result.has_error());
}

// =============================================================================
// Multiple Roles Per Rank Tests
// =============================================================================

TEST_F(RoleAssignmentTest, MultipleRolesPerRank) {
    role_manager mgr;

    // Rank 0 is both coordinator AND worker (all_ranks)
    mgr.register_role(node_role::coordinator, "coordinator",
                      role_assignment::first_rank_only());
    mgr.register_role(node_role::worker, "worker",
                      role_assignment::all_ranks());

    mgr.initialize(*comm_);

    // Rank 0 should have both roles
    EXPECT_TRUE(mgr.has_role(node_role::coordinator));
    EXPECT_TRUE(mgr.has_role(node_role::worker));

    // Should be in 2 groups
    EXPECT_EQ(mgr.my_groups().size(), 2);

    // Primary role should be coordinator (non-worker preferred)
    EXPECT_EQ(mgr.primary_role(), node_role::coordinator);
}

// =============================================================================
// Validation Requirements Tests
// =============================================================================

TEST_F(RoleAssignmentTest, ValidationRequirementsMinRanks) {
    role_manager_config config;
    config.validate_requirements = true;

    role_manager mgr(config);

    // Register a role requiring at least 10 ranks, but world has only 4
    role_properties props;
    props.name = "big_role";
    props.min_ranks = 10;
    role_descriptor desc(node_role::worker, std::move(props),
                         role_assignment::all_ranks());
    mgr.register_role(std::move(desc));

    auto result = mgr.initialize(*comm_);
    EXPECT_TRUE(result.has_error());
}

TEST_F(RoleAssignmentTest, ValidationRequirementsMaxRanks) {
    role_manager_config config;
    config.validate_requirements = true;

    role_manager mgr(config);

    // Register a role allowing at most 2 ranks, but all_ranks assigns 4
    role_properties props;
    props.name = "small_role";
    props.max_ranks = 2;
    role_descriptor desc(node_role::worker, std::move(props),
                         role_assignment::all_ranks());
    mgr.register_role(std::move(desc));

    auto result = mgr.initialize(*comm_);
    EXPECT_TRUE(result.has_error());
}

TEST_F(RoleAssignmentTest, ValidationRequirementsDisabled) {
    role_manager_config config;
    config.validate_requirements = false;

    role_manager mgr(config);

    // Same constraint that would fail, but validation is disabled
    role_properties props;
    props.name = "big_role";
    props.min_ranks = 10;
    role_descriptor desc(node_role::worker, std::move(props),
                         role_assignment::all_ranks());
    mgr.register_role(std::move(desc));

    auto result = mgr.initialize(*comm_);
    EXPECT_TRUE(result.has_value());
}

// =============================================================================
// Different Rank Perspective Tests
// =============================================================================

TEST(RoleAssignmentRankTest, WorkerRankPerspective) {
    // Simulate rank 2 of 4
    MockCommunicator comm(2, 4);
    role_manager mgr;
    setup_worker_coordinator(mgr, 1);

    mgr.initialize(comm);

    // Rank 2 should be worker, not coordinator
    EXPECT_TRUE(mgr.has_role(node_role::worker));
    EXPECT_FALSE(mgr.has_role(node_role::coordinator));
    EXPECT_EQ(mgr.primary_role(), node_role::worker);
}

TEST(RoleAssignmentRankTest, LastRankPerspective) {
    // Simulate rank 3 of 4
    MockCommunicator comm(3, 4);
    role_manager mgr;
    setup_with_io_handlers(mgr, 1);

    mgr.initialize(comm);

    // Rank 3 (last) should be io_handler
    EXPECT_TRUE(mgr.has_role(node_role::io_handler));
    EXPECT_FALSE(mgr.has_role(node_role::worker));
    EXPECT_EQ(mgr.primary_role(), node_role::io_handler);
}

// =============================================================================
// Inter-Communicator Stub Test
// =============================================================================

TEST_F(RoleAssignmentTest, CreateInterCommunicatorNotSupported) {
    role_manager mgr;
    rank_group g1(0, node_role::worker);
    rank_group g2(1, node_role::coordinator);

    auto result = mgr.create_inter_communicator(g1, g2);
    EXPECT_TRUE(result.has_error());
}

}  // namespace dtl::test

// =============================================================================
// Main (standalone test binary)
// =============================================================================
// Note: Uses gtest_main from the unit test target (no custom main needed)
