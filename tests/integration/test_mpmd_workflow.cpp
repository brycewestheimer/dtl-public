// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpmd_workflow.cpp
/// @brief Integration test for complete MPMD workflow
/// @details Tests Phase 11: Complete MPMD workflow including role management,
///          inter-group communication via simulated mailbox, and barrier ops.
///          All tests run in single-process mode (no MPI required).

#include <dtl/mpmd/mpmd.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Mock Communicator
// =============================================================================

class WorkflowMockCommunicator : public dtl::communicator_base {
public:
    WorkflowMockCommunicator(dtl::rank_t rank, dtl::rank_t size)
        : rank_(rank), size_(size) {}

    [[nodiscard]] dtl::rank_t rank() const noexcept override { return rank_; }
    [[nodiscard]] dtl::rank_t size() const noexcept override { return size_; }

    [[nodiscard]] dtl::communicator_properties properties() const noexcept override {
        return dtl::communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "WorkflowMockCommunicator"
        };
    }

private:
    dtl::rank_t rank_;
    dtl::rank_t size_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class MpmdWorkflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear mailbox between tests to avoid cross-test contamination
        mpmd::inter_group_communicator::clear_mailbox();
    }

    void TearDown() override {
        mpmd::inter_group_communicator::clear_mailbox();
    }
};

// =============================================================================
// T05: Complete MPMD Workflow Test
// =============================================================================

TEST_F(MpmdWorkflowTest, CompleteWorkflow) {
    // Step 1: Create role manager and register two roles
    mpmd::role_manager mgr;

    auto coord_reg = mgr.register_role(
        mpmd::node_role::coordinator,
        "coordinator",
        mpmd::role_assignment::first_n_ranks(2));
    ASSERT_TRUE(coord_reg.has_value());

    auto worker_reg = mgr.register_role(
        mpmd::node_role::worker,
        "worker",
        [](rank_t rank, rank_t) { return rank >= 2; });
    ASSERT_TRUE(worker_reg.has_value());

    // Step 2: Initialize with mock communicator (rank 0, size 8)
    WorkflowMockCommunicator comm(0, 8);
    auto init_result = mgr.initialize(comm);
    ASSERT_TRUE(init_result.has_value());

    // Step 3: Get groups
    auto* coord_group = mgr.get_group(mpmd::node_role::coordinator);
    auto* worker_group = mgr.get_group(mpmd::node_role::worker);
    ASSERT_NE(coord_group, nullptr);
    ASSERT_NE(worker_group, nullptr);

    EXPECT_EQ(coord_group->size(), 2u);
    EXPECT_EQ(worker_group->size(), 6u);

    // Step 4: Create inter-group communicator
    auto igc_result = mpmd::make_inter_group_communicator(*coord_group, *worker_group);
    ASSERT_TRUE(igc_result.has_value());
    auto igc = igc_result.value();
    EXPECT_TRUE(igc.valid());

    // Step 5: Leader send/recv via simulated mailbox
    // In single-process testing, we simulate being the leader of both groups
    // by explicitly setting local_rank. (In a real multi-process run,
    // different processes would call send and recv.)
    coord_group->set_local_rank(0);  // Simulate being coordinator leader
    auto send_result = igc.leader_send(42, /*tag=*/1);
    ASSERT_TRUE(send_result.has_value());

    worker_group->set_local_rank(0);  // Simulate being worker leader
    auto recv_result = igc.leader_recv<int>(/*tag=*/1);
    ASSERT_TRUE(recv_result.has_value());
    EXPECT_EQ(recv_result.value(), 42);

    // Step 6: Barrier operations
    auto barrier_result = igc.barrier();
    ASSERT_TRUE(barrier_result.has_value());

    auto barrier_all_result = igc.barrier_all();
    ASSERT_TRUE(barrier_all_result.has_value());
}

// =============================================================================
// T01: Leader Send/Recv Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, LeaderSendRecvInt) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto send = igc.leader_send(100, 5);
    ASSERT_TRUE(send.has_value());

    auto recv = igc.leader_recv<int>(5);
    ASSERT_TRUE(recv.has_value());
    EXPECT_EQ(recv.value(), 100);
}

TEST_F(MpmdWorkflowTest, LeaderSendRecvString) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    std::string msg = "hello from coordinator";
    auto send = igc.leader_send(msg, 10);
    ASSERT_TRUE(send.has_value());

    auto recv = igc.leader_recv<std::string>(10);
    ASSERT_TRUE(recv.has_value());
    EXPECT_EQ(recv.value(), "hello from coordinator");
}

TEST_F(MpmdWorkflowTest, LeaderSendRecvVector) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    auto send = igc.leader_send(data, 20);
    ASSERT_TRUE(send.has_value());

    auto recv = igc.leader_recv<std::vector<double>>(20);
    ASSERT_TRUE(recv.has_value());
    EXPECT_EQ(recv.value().size(), 4u);
    EXPECT_DOUBLE_EQ(recv.value()[0], 1.0);
    EXPECT_DOUBLE_EQ(recv.value()[3], 4.0);
}

TEST_F(MpmdWorkflowTest, NonLeaderSendIsNoop) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.add_member(1);
    src.set_local_rank(1);  // Not the leader

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(2);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    // Non-leader send is a no-op (no data stored)
    auto send = igc.leader_send(999, 30);
    ASSERT_TRUE(send.has_value());

    // Recv should fail since nothing was stored
    auto recv = igc.leader_recv<int>(30);
    EXPECT_TRUE(recv.has_error());
}

TEST_F(MpmdWorkflowTest, NonLeaderRecvReturnsDefault) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.add_member(2);
    dst.set_local_rank(1);  // Not the leader

    mpmd::inter_group_communicator igc(&src, &dst);

    auto send = igc.leader_send(42, 40);
    ASSERT_TRUE(send.has_value());

    // Non-leader recv returns default-constructed T
    auto recv = igc.leader_recv<int>(40);
    ASSERT_TRUE(recv.has_value());
    EXPECT_EQ(recv.value(), 0);  // Default int
}

TEST_F(MpmdWorkflowTest, MultipleSendRecvWithDifferentTags) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    // Send multiple messages with different tags
    igc.leader_send(100, 1);
    igc.leader_send(200, 2);
    igc.leader_send(300, 3);

    // Receive in different order
    auto r2 = igc.leader_recv<int>(2);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2.value(), 200);

    auto r1 = igc.leader_recv<int>(1);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1.value(), 100);

    auto r3 = igc.leader_recv<int>(3);
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(r3.value(), 300);
}

// =============================================================================
// T02: Collective Operation Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, BroadcastReturnsData) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.broadcast(42);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(MpmdWorkflowTest, BroadcastString) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.broadcast(std::string("broadcast message"));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "broadcast message");
}

TEST_F(MpmdWorkflowTest, ScatterReturnsElement) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    std::vector<int> data = {10, 20, 30, 40};
    auto result = igc.scatter(data);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 10);  // Element at local index 0
}

TEST_F(MpmdWorkflowTest, ScatterEmptyDataFails) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    std::vector<int> empty_data;
    auto result = igc.scatter(empty_data);
    EXPECT_TRUE(result.has_error());
}

TEST_F(MpmdWorkflowTest, GatherReturnsVector) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.gather(42);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 1u);
    EXPECT_EQ(result.value()[0], 42);
}

TEST_F(MpmdWorkflowTest, ReduceAppliesOp) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.reduce(42, std::plus<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

// =============================================================================
// T03: Barrier Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, BarrierSucceeds) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.barrier();
    ASSERT_TRUE(result.has_value());
}

TEST_F(MpmdWorkflowTest, BarrierAllSucceeds) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "src");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::worker, "dst");
    dst.add_member(1);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    auto result = igc.barrier_all();
    ASSERT_TRUE(result.has_value());
}

// =============================================================================
// Invalid Communicator Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, InvalidCommunicatorOperations) {
    mpmd::inter_group_communicator igc;  // Default — invalid

    EXPECT_FALSE(igc.valid());

    auto send = igc.leader_send(42, 0);
    EXPECT_TRUE(send.has_error());

    auto recv = igc.leader_recv<int>(0);
    EXPECT_TRUE(recv.has_error());

    auto bcast = igc.broadcast(42);
    EXPECT_TRUE(bcast.has_error());

    auto barrier = igc.barrier();
    EXPECT_TRUE(barrier.has_error());

    auto barrier_all = igc.barrier_all();
    EXPECT_TRUE(barrier_all.has_error());

    std::vector<int> sdata = {1, 2, 3};
    auto scat = igc.scatter(sdata);
    EXPECT_TRUE(scat.has_error());

    auto gath = igc.gather(42);
    EXPECT_TRUE(gath.has_error());

    auto red = igc.reduce(42, std::plus<int>{});
    EXPECT_TRUE(red.has_error());
}

// =============================================================================
// T04: Role Manager Sub-Communicator Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, GetRoleCommunicator) {
    WorkflowMockCommunicator comm(0, 4);
    mpmd::role_manager mgr;
    mpmd::setup_worker_coordinator(mgr, 1);

    auto result = mgr.initialize(comm);
    ASSERT_TRUE(result.has_value());

    // Without a dedicated sub-communicator, falls back to world communicator
    auto* coord_comm = mgr.get_role_communicator(mpmd::node_role::coordinator);
    EXPECT_NE(coord_comm, nullptr);
    EXPECT_EQ(coord_comm->rank(), 0);

    auto* worker_comm = mgr.get_role_communicator(mpmd::node_role::worker);
    EXPECT_NE(worker_comm, nullptr);

    // Non-existent role returns nullptr
    auto* none = mgr.get_role_communicator(mpmd::node_role::monitor);
    EXPECT_EQ(none, nullptr);
}

TEST_F(MpmdWorkflowTest, GetRoleCommunicatorConst) {
    WorkflowMockCommunicator comm(0, 4);
    mpmd::role_manager mgr;
    mpmd::setup_worker_coordinator(mgr, 1);
    mgr.initialize(comm);

    const auto& const_mgr = mgr;

    const auto* coord_comm = const_mgr.get_role_communicator(mpmd::node_role::coordinator);
    EXPECT_NE(coord_comm, nullptr);

    const auto* none = const_mgr.get_role_communicator(mpmd::node_role::monitor);
    EXPECT_EQ(none, nullptr);
}

// =============================================================================
// Pipeline Integration Test
// =============================================================================

TEST_F(MpmdWorkflowTest, PipelineWorkflow) {
    // Create 3 groups for a pipeline
    mpmd::rank_group input_group(0, mpmd::node_role::io_handler, "input");
    input_group.add_member(0);
    input_group.set_local_rank(0);

    mpmd::rank_group compute_group(1, mpmd::node_role::worker, "compute");
    compute_group.add_member(1);
    compute_group.add_member(2);
    compute_group.set_local_rank(0);

    mpmd::rank_group output_group(2, mpmd::node_role::aggregator, "output");
    output_group.add_member(3);
    output_group.set_local_rank(0);

    // Build pipeline
    mpmd::group_pipeline pipeline;
    pipeline.add_stage(&input_group, "input");
    pipeline.add_stage(&compute_group, "compute");
    pipeline.add_stage(&output_group, "output");

    EXPECT_EQ(pipeline.size(), 3u);

    // Forward data through pipeline
    auto result = pipeline.forward(42);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

// =============================================================================
// Multi-Group Communication Test
// =============================================================================

TEST_F(MpmdWorkflowTest, MulticastToMultipleGroups) {
    mpmd::rank_group src(0, mpmd::node_role::coordinator, "coord");
    src.add_member(0);
    src.set_local_rank(0);

    mpmd::rank_group dst1(1, mpmd::node_role::worker, "workers1");
    dst1.add_member(1);
    dst1.set_local_rank(0);

    mpmd::rank_group dst2(2, mpmd::node_role::io_handler, "io");
    dst2.add_member(2);
    dst2.set_local_rank(0);

    std::vector<mpmd::rank_group*> dests = {&dst1, &dst2};
    auto result = mpmd::multicast(src, dests, 42);
    ASSERT_TRUE(result.has_value());
}

// =============================================================================
// SendToLeader / RecvFromLeader Tests
// =============================================================================

TEST_F(MpmdWorkflowTest, SendToLeaderRecvFromLeader) {
    mpmd::rank_group src(0, mpmd::node_role::worker, "src");
    src.add_member(0);
    src.add_member(1);
    src.set_local_rank(0);

    mpmd::rank_group dst(1, mpmd::node_role::coordinator, "dst");
    dst.add_member(2);
    dst.set_local_rank(0);

    mpmd::inter_group_communicator igc(&src, &dst);

    // Send from local rank 0 in source to leader
    auto send = igc.send_to_leader(99, 0, 50);
    ASSERT_TRUE(send.has_value());

    // Receive at local rank 0 in destination
    auto recv = igc.recv_from_leader<int>(0, 50);
    ASSERT_TRUE(recv.has_value());
    EXPECT_EQ(recv.value(), 99);
}

}  // namespace dtl::test