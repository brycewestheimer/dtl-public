// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpmd.hpp
/// @brief Master include for DTL MPMD (Multiple Program Multiple Data) support
/// @details Provides single-header access to all MPMD types.
/// @since 0.1.0

#pragma once

// Node roles and role descriptors
#include <dtl/mpmd/node_role.hpp>

// Rank groups
#include <dtl/mpmd/rank_group.hpp>

// Role management
#include <dtl/mpmd/role_manager.hpp>

// Inter-group communication
#include <dtl/mpmd/inter_group.hpp>
#include <dtl/mpmd/inter_group_comm.hpp>

namespace dtl {

// ============================================================================
// Re-export dtl::mpmd:: types into dtl:: for backward compatibility
// ============================================================================

// node_role.hpp types
using mpmd::node_role;
using mpmd::role_id;
using mpmd::role_properties;
using mpmd::role_name;
using mpmd::is_predefined_role;
using mpmd::is_custom_role;
using mpmd::make_custom_role;
using mpmd::custom_role_id;
using mpmd::role_descriptor;
namespace role_assignment = mpmd::role_assignment;

// rank_group.hpp types
using mpmd::rank_group;
using mpmd::group_membership;
using mpmd::rank_info;

// role_manager.hpp types
using mpmd::role_manager_config;
using mpmd::role_manager;
using mpmd::setup_worker_coordinator;
using mpmd::setup_with_io_handlers;

// inter_group.hpp types
using mpmd::inter_group_communicator;
using mpmd::make_inter_group_communicator;
using mpmd::multicast_result;
using mpmd::multicast;
using mpmd::multi_receive;
using mpmd::pipeline_stage;
using mpmd::group_pipeline;

// inter_group_comm.hpp types
using mpmd::translate_to_world_rank;
using mpmd::translate_to_local_rank;
using mpmd::inter_group_dest_rank;
using mpmd::inter_group_src_rank;
using mpmd::inter_group_broadcast_root;
using mpmd::inter_group_endpoint;
using mpmd::make_endpoint;
using mpmd::inter_group_channel;
using mpmd::make_channel;

// ============================================================================
// MPMD Module Summary
// ============================================================================
//
// The MPMD (Multiple Program Multiple Data) module provides abstractions for
// organizing distributed computations where different ranks perform different
// roles. This enables patterns like:
//
// - Worker/Coordinator: Coordinator distributes work to workers
// - Producer/Consumer: Producers generate data, consumers process it
// - Pipeline: Data flows through stages of processing
// - Hybrid: Ranks serve multiple roles simultaneously
//
// ============================================================================
// Node Roles
// ============================================================================
//
// Predefined roles in node_role enum:
// - worker: Performs main computation
// - coordinator: Orchestrates work distribution
// - io_handler: Manages file/network I/O
// - aggregator: Collects and reduces results
// - monitor: Tracks progress and health
// - checkpoint_handler: Manages fault tolerance
// - load_balancer: Redistributes work
// - gateway: Bridges between groups
//
// Custom roles can be created with:
// @code
// auto my_role = dtl::make_custom_role(42);
// @endcode
//
// ============================================================================
// Role Properties
// ============================================================================
//
// role_properties describes role requirements:
// - name, description: Human-readable identifiers
// - min_ranks, max_ranks: Cardinality constraints
// - required: Whether role must have assigned ranks
// - prefer_separate_nodes: Scheduling hint
// - requires_gpu: GPU access requirement
// - memory_hint: Memory requirement hint
//
// ============================================================================
// Role Assignment
// ============================================================================
//
// Predefined assignment strategies in role_assignment namespace:
// - first_rank_only(): Assign to rank 0
// - last_rank_only(): Assign to last rank
// - all_ranks(): Assign to all ranks
// - specific_ranks({0, 1, 2}): Assign to listed ranks
// - rank_range(start, end): Assign to range
// - first_n_ranks(n): First N ranks
// - last_n_ranks(n): Last N ranks
// - every_nth_rank(n, offset): Strided assignment
// - fraction_of_ranks(f): Fraction of total
//
// ============================================================================
// Rank Groups
// ============================================================================
//
// rank_group represents a subset of ranks with shared role:
// - id(), role(), name(): Group identity
// - members(): World ranks in group
// - size(), empty(): Size queries
// - contains(rank): Membership check
// - local_rank(): Rank within group
// - is_member(), is_leader(): Role checks
// - to_local_rank(), to_world_rank(): Rank translation
// - communicator(): Sub-communicator for group
//
// ============================================================================
// Role Manager
// ============================================================================
//
// role_manager coordinates role assignments:
//
// @code
// dtl::role_manager manager;
//
// // Register roles
// manager.register_role(
//     dtl::node_role::coordinator,
//     "coordinator",
//     dtl::role_assignment::first_rank_only());
//
// manager.register_role(
//     dtl::node_role::worker,
//     "worker",
//     [](rank_t r, rank_t) { return r > 0; });
//
// // Initialize (collective)
// manager.initialize(world_comm);
//
// // Query groups
// if (manager.has_role(dtl::node_role::coordinator)) {
//     // Coordinator logic
// }
//
// auto* workers = manager.get_group(dtl::node_role::worker);
// @endcode
//
// ============================================================================
// Inter-Group Communication
// ============================================================================
//
// inter_group_communicator enables communication between groups:
//
// Point-to-point:
// - leader_send(data): Send from source leader to dest leader
// - leader_recv<T>(): Receive at dest leader from source leader
// - send_to_leader(data, local): Any source rank to dest leader
// - recv_from_leader<T>(local): Source leader to any dest rank
//
// Collective:
// - broadcast(data): Source leader to all dest members
// - scatter(data): Source leader to dest members (partitioned)
// - gather(data): Source members to dest leader
// - reduce(data, op): Reduce across source, result at dest
// - transfer(data): 1:1 transfer between groups
//
// Synchronization:
// - barrier(): Synchronize leaders
// - barrier_all(): Synchronize all members of both groups
//
// ============================================================================
// Multi-Group Patterns
// ============================================================================
//
// Multicast to multiple groups:
// @code
// dtl::multicast(source, {dest1, dest2, dest3}, data);
// @endcode
//
// Receive from multiple groups:
// @code
// auto result = dtl::multi_receive<T>({src1, src2}, dest);
// for (const auto& val : result.data) {
//     // Process data from each source
// }
// @endcode
//
// ============================================================================
// Pipeline Patterns
// ============================================================================
//
// group_pipeline for staged processing:
//
// @code
// dtl::group_pipeline pipeline;
// pipeline.add_stage(producers, "produce");
// pipeline.add_stage(processors, "process");
// pipeline.add_stage(consumers, "consume");
//
// // Forward data through pipeline
// auto result = pipeline.forward(initial_data);
// @endcode
//
// ============================================================================
// Convenience Setup Functions
// ============================================================================
//
// @code
// // Simple worker/coordinator setup
// dtl::setup_worker_coordinator(manager, 1);  // 1 coordinator
//
// // Workers with I/O handlers
// dtl::setup_with_io_handlers(manager, 2);  // 2 I/O handlers
// @endcode
//
// ============================================================================
// Usage Examples
// ============================================================================
//
// @code
// #include <dtl/mpmd/mpmd.hpp>
//
// int main(int argc, char** argv) {
//     // Initialize communication (MPI, etc.)
//     dtl::mpi_environment env(argc, argv);
//     auto& world = dtl::world_communicator();
//
//     // Set up role manager
//     dtl::role_manager manager;
//     dtl::setup_worker_coordinator(manager, 1);
//     manager.initialize(world);
//
//     // Execute based on role
//     if (manager.has_role(dtl::node_role::coordinator)) {
//         // Coordinator gathers results
//         auto* workers = manager.get_group(dtl::node_role::worker);
//         auto comm = dtl::make_inter_group_communicator(
//             *workers, *manager.get_group(dtl::node_role::coordinator));
//
//         auto results = comm.value().gather<double>(0.0);
//         // Process results...
//     } else {
//         // Workers compute and send to coordinator
//         double result = compute();
//         auto* coord = manager.get_group(dtl::node_role::coordinator);
//         auto comm = dtl::make_inter_group_communicator(
//             *manager.get_group(dtl::node_role::worker), *coord);
//
//         comm.value().leader_send(result);
//     }
//
//     return 0;
// }
// @endcode
//
// ============================================================================
// Advanced: Custom Roles
// ============================================================================
//
// @code
// // Define custom roles
// constexpr auto data_source = dtl::make_custom_role(0);
// constexpr auto transformer = dtl::make_custom_role(1);
// constexpr auto data_sink = dtl::make_custom_role(2);
//
// // Register with custom properties
// dtl::role_properties source_props;
// source_props.name = "data_source";
// source_props.min_ranks = 1;
// source_props.max_ranks = 4;
// source_props.requires_gpu = true;
//
// manager.register_role(dtl::role_descriptor(
//     data_source, source_props,
//     dtl::role_assignment::first_n_ranks(4)));
// @endcode
//
// ============================================================================

}  // namespace dtl
