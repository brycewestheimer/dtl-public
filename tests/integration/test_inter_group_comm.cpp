// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_inter_group_comm.cpp
/// @brief Integration tests for inter-group communication utilities
/// @details Tests for Phase 12C: Inter-group rank translation and validation

#include <dtl/mpmd/mpmd.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// =============================================================================
// Mock Communicator (same as in test_mpmd_role_assignment.cpp)
// =============================================================================

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

class InterGroupCommTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a standard worker/coordinator topology with 8 ranks.
        // Coordinators: ranks 0, 1
        // Workers: ranks 2, 3, 4, 5, 6, 7
        coord_group_ = std::make_unique<rank_group>(0, node_role::coordinator, "coordinators");
        coord_group_->add_member(0);
        coord_group_->add_member(1);

        worker_group_ = std::make_unique<rank_group>(1, node_role::worker, "workers");
        worker_group_->add_member(2);
        worker_group_->add_member(3);
        worker_group_->add_member(4);
        worker_group_->add_member(5);
        worker_group_->add_member(6);
        worker_group_->add_member(7);

        empty_group_ = std::make_unique<rank_group>(2, node_role::monitor, "empty");

        single_group_ = std::make_unique<rank_group>(3, node_role::aggregator, "single");
        single_group_->add_member(5);
    }

    std::unique_ptr<rank_group> coord_group_;
    std::unique_ptr<rank_group> worker_group_;
    std::unique_ptr<rank_group> empty_group_;
    std::unique_ptr<rank_group> single_group_;
};

// =============================================================================
// Inter-Group Send Rank Translation Tests
// =============================================================================

TEST_F(InterGroupCommTest, InterGroupSendValidRank) {
    // Local rank 0 in coordinator group -> world rank 0
    auto result = inter_group_dest_rank(*coord_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);

    // Local rank 1 in coordinator group -> world rank 1
    result = inter_group_dest_rank(*coord_group_, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

TEST_F(InterGroupCommTest, InterGroupSendInvalidRank) {
    // Local rank 5 does not exist in coordinator group (only 2 members)
    auto result = inter_group_dest_rank(*coord_group_, 5);
    EXPECT_TRUE(result.has_error());
}

// =============================================================================
// Inter-Group Recv Rank Translation Tests
// =============================================================================

TEST_F(InterGroupCommTest, InterGroupRecvValidRank) {
    // Local rank 0 in worker group -> world rank 2
    auto result = inter_group_src_rank(*worker_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 2);

    // Local rank 3 in worker group -> world rank 5
    result = inter_group_src_rank(*worker_group_, 3);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);
}

// =============================================================================
// Inter-Group Broadcast Rank Translation Tests
// =============================================================================

TEST_F(InterGroupCommTest, InterGroupBroadcastValidRank) {
    // Broadcast root: local rank 0 in coordinator group -> world rank 0
    auto result = inter_group_broadcast_root(*coord_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
}

// =============================================================================
// Worker Group Translation Tests
// =============================================================================

TEST_F(InterGroupCommTest, RankTranslationWorkers) {
    // Worker group: members are [2, 3, 4, 5, 6, 7]
    // Local rank 0 -> world rank 2
    auto result = translate_to_world_rank(*worker_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 2);

    // Local rank 5 -> world rank 7
    result = translate_to_world_rank(*worker_group_, 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 7);
}

// =============================================================================
// Coordinator Group Translation Tests
// =============================================================================

TEST_F(InterGroupCommTest, RankTranslationCoordinators) {
    // Coordinator group: members are [0, 1]
    // Local rank 0 -> world rank 0
    auto result = translate_to_world_rank(*coord_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);

    // Local rank 1 -> world rank 1
    result = translate_to_world_rank(*coord_group_, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

// =============================================================================
// Cross-Group Channel Tests
// =============================================================================

TEST_F(InterGroupCommTest, CrossGroupSendRecvPair) {
    // Create a channel from coordinator local rank 0 to worker local rank 0
    auto channel = make_channel(*coord_group_, 0, *worker_group_, 0, 42);

    EXPECT_TRUE(channel.valid());

    auto src_world = channel.src_world_rank();
    ASSERT_TRUE(src_world.has_value());
    EXPECT_EQ(src_world.value(), 0);  // Coord local 0 -> world 0

    auto dst_world = channel.dest_world_rank();
    ASSERT_TRUE(dst_world.has_value());
    EXPECT_EQ(dst_world.value(), 2);  // Worker local 0 -> world 2
}

// =============================================================================
// Empty Group Tests
// =============================================================================

TEST_F(InterGroupCommTest, EmptyGroup) {
    auto result = translate_to_world_rank(*empty_group_, 0);
    EXPECT_TRUE(result.has_error());

    auto result2 = inter_group_dest_rank(*empty_group_, 0);
    EXPECT_TRUE(result2.has_error());
}

// =============================================================================
// Single Member Group Tests
// =============================================================================

TEST_F(InterGroupCommTest, SingleMemberGroup) {
    // Single member group has world rank 5 at local rank 0
    auto result = translate_to_world_rank(*single_group_, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);

    // Local rank 1 is out of range for single-member group
    auto invalid = translate_to_world_rank(*single_group_, 1);
    EXPECT_TRUE(invalid.has_error());
}

// =============================================================================
// Group Membership Check Tests
// =============================================================================

TEST_F(InterGroupCommTest, GroupMembershipCheck) {
    // World rank 0 is in coordinator group
    EXPECT_TRUE(coord_group_->contains(0));
    EXPECT_TRUE(coord_group_->contains(1));
    EXPECT_FALSE(coord_group_->contains(2));

    // World rank 2 is in worker group
    EXPECT_TRUE(worker_group_->contains(2));
    EXPECT_FALSE(worker_group_->contains(0));
}

// =============================================================================
// Full Setup With Role Manager Tests
// =============================================================================

TEST_F(InterGroupCommTest, InterGroupCommSetup) {
    MockCommunicator comm(0, 8);
    role_manager mgr;
    setup_worker_coordinator(mgr, 2);  // 2 coordinators, 6 workers

    auto result = mgr.initialize(comm);
    ASSERT_TRUE(result.has_value());

    auto* coord = mgr.get_group(node_role::coordinator);
    auto* worker = mgr.get_group(node_role::worker);

    ASSERT_NE(coord, nullptr);
    ASSERT_NE(worker, nullptr);

    EXPECT_EQ(coord->size(), 2);
    EXPECT_EQ(worker->size(), 6);

    // Translate coordinator local 0 -> world 0
    auto world_rank = translate_to_world_rank(*coord, 0);
    ASSERT_TRUE(world_rank.has_value());
    EXPECT_EQ(world_rank.value(), 0);

    // Translate worker local 0 -> world 2
    world_rank = translate_to_world_rank(*worker, 0);
    ASSERT_TRUE(world_rank.has_value());
    EXPECT_EQ(world_rank.value(), 2);
}

// =============================================================================
// World Rank / Local Rank Conversion Tests
// =============================================================================

TEST_F(InterGroupCommTest, WorldRankFromLocalRank) {
    // Worker group: [2, 3, 4, 5, 6, 7]
    for (rank_t local = 0; local < 6; ++local) {
        auto result = translate_to_world_rank(*worker_group_, local);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), local + 2);
    }
}

TEST_F(InterGroupCommTest, LocalRankFromWorldRank) {
    // Worker group: [2, 3, 4, 5, 6, 7]
    for (rank_t world = 2; world < 8; ++world) {
        auto result = translate_to_local_rank(*worker_group_, world);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), world - 2);
    }

    // World rank 0 is not in worker group
    auto invalid = translate_to_local_rank(*worker_group_, 0);
    EXPECT_TRUE(invalid.has_error());
}

// =============================================================================
// Multi-Group Cross Communication Tests
// =============================================================================

TEST_F(InterGroupCommTest, MultiGroupCrossComm) {
    // Set up channels between coordinator and worker groups
    auto chan_coord_to_worker = make_channel(
        *coord_group_, 0, *worker_group_, 0);
    auto chan_worker_to_coord = make_channel(
        *worker_group_, 0, *coord_group_, 0);

    EXPECT_TRUE(chan_coord_to_worker.valid());
    EXPECT_TRUE(chan_worker_to_coord.valid());

    // Verify reverse direction
    auto src1 = chan_coord_to_worker.src_world_rank();
    auto dst1 = chan_coord_to_worker.dest_world_rank();
    auto src2 = chan_worker_to_coord.src_world_rank();
    auto dst2 = chan_worker_to_coord.dest_world_rank();

    ASSERT_TRUE(src1.has_value());
    ASSERT_TRUE(dst1.has_value());
    ASSERT_TRUE(src2.has_value());
    ASSERT_TRUE(dst2.has_value());

    // First channel: coord[0]=world 0 -> worker[0]=world 2
    EXPECT_EQ(src1.value(), 0);
    EXPECT_EQ(dst1.value(), 2);

    // Second channel: worker[0]=world 2 -> coord[0]=world 0
    EXPECT_EQ(src2.value(), 2);
    EXPECT_EQ(dst2.value(), 0);
}

// =============================================================================
// Tag Preservation Tests
// =============================================================================

TEST_F(InterGroupCommTest, TagPreservation) {
    auto endpoint = make_endpoint(*coord_group_, 0, 99);
    EXPECT_EQ(endpoint.tag, 99);

    auto channel = make_channel(*coord_group_, 0, *worker_group_, 0, 42);
    EXPECT_EQ(channel.source.tag, 42);
    EXPECT_EQ(channel.destination.tag, 42);
}

// =============================================================================
// Bidirectional Communication Tests
// =============================================================================

TEST_F(InterGroupCommTest, BidirectionalComm) {
    // Forward: coord local 1 -> worker local 2
    auto fwd_dst = inter_group_dest_rank(*worker_group_, 2);
    ASSERT_TRUE(fwd_dst.has_value());
    EXPECT_EQ(fwd_dst.value(), 4);  // Worker local 2 -> world 4

    // Reverse: worker local 2 -> coord local 1
    auto rev_dst = inter_group_dest_rank(*coord_group_, 1);
    ASSERT_TRUE(rev_dst.has_value());
    EXPECT_EQ(rev_dst.value(), 1);  // Coord local 1 -> world 1
}

// =============================================================================
// Broadcast Root Tests
// =============================================================================

TEST_F(InterGroupCommTest, BroadcastRoot) {
    // Coordinator local 0 as broadcast root -> world 0
    auto root = inter_group_broadcast_root(*coord_group_, 0);
    ASSERT_TRUE(root.has_value());
    EXPECT_EQ(root.value(), 0);

    // Worker local 3 as broadcast root -> world 5
    root = inter_group_broadcast_root(*worker_group_, 3);
    ASSERT_TRUE(root.has_value());
    EXPECT_EQ(root.value(), 5);
}

// =============================================================================
// Group/Role Mapping Tests
// =============================================================================

TEST_F(InterGroupCommTest, GroupRoleMapping) {
    MockCommunicator comm(0, 4);
    role_manager mgr;
    setup_worker_coordinator(mgr, 1);
    mgr.initialize(comm);

    auto* coord = mgr.get_group(node_role::coordinator);
    auto* worker = mgr.get_group(node_role::worker);

    ASSERT_NE(coord, nullptr);
    ASSERT_NE(worker, nullptr);

    EXPECT_EQ(coord->role(), node_role::coordinator);
    EXPECT_EQ(worker->role(), node_role::worker);

    // Verify the mapping from role -> group is correct
    EXPECT_EQ(coord->name(), "coordinator");
    EXPECT_EQ(worker->name(), "worker");
}

// =============================================================================
// Invalid Group Operations Tests
// =============================================================================

TEST_F(InterGroupCommTest, InvalidGroupOps) {
    // Out of range local rank in worker group (6 members, local 6 invalid)
    auto result = translate_to_world_rank(*worker_group_, 6);
    EXPECT_TRUE(result.has_error());

    // Negative local rank (cast to large value via rank_t = int)
    auto result2 = translate_to_world_rank(*worker_group_, -1);
    EXPECT_TRUE(result2.has_error());
}

// =============================================================================
// Endpoint / Channel Validation Tests
// =============================================================================

TEST_F(InterGroupCommTest, NullEndpointHandling) {
    inter_group_endpoint ep;
    EXPECT_FALSE(ep.valid());

    auto world_rank = ep.world_rank();
    EXPECT_TRUE(world_rank.has_error());
}

TEST_F(InterGroupCommTest, ValidEndpoint) {
    auto ep = make_endpoint(*coord_group_, 0, 7);
    EXPECT_TRUE(ep.valid());
    EXPECT_EQ(ep.tag, 7);

    auto wr = ep.world_rank();
    ASSERT_TRUE(wr.has_value());
    EXPECT_EQ(wr.value(), 0);
}

}  // namespace dtl::test

// =============================================================================
// Main (standalone test binary)
// =============================================================================
// Note: Uses gtest_main from the unit test target (no custom main needed)
