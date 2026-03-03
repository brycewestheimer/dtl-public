// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hybrid_backend.cpp
/// @brief Hierarchical communication: MPI between nodes, shared memory within
/// @details Demonstrates a hybrid approach where different communication
///          mechanisms are used at different scales. This is common in
///          modern HPC where nodes have many cores sharing memory.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./hybrid_backend
///
/// Expected output (4 ranks):
///   DTL Hybrid Backend Example
///   ==========================
///
///   Configuration:
///     Total ranks: 4
///     Simulated nodes: 2
///     Ranks per node: 2
///
///   --- Topology Discovery ---
///   Rank 0: Node 0, local rank 0 (node leader)
///   Rank 1: Node 0, local rank 1
///   Rank 2: Node 1, local rank 0 (node leader)
///   Rank 3: Node 1, local rank 1
///
///   --- Intra-Node Reduction (Shared Memory Simulation) ---
///   Each node reduces locally before inter-node communication.
///   Node 0 local sum: 300 (ranks 0,1)
///   Node 1 local sum: 700 (ranks 2,3)
///
///   --- Inter-Node Reduction (MPI) ---
///   Only node leaders participate in cross-node communication.
///   Global sum: 1000
///
///   --- Broadcast Back to Non-Leaders ---
///   All ranks now have global result.
///
///   SUCCESS: Hybrid backend example completed!

// NOTE: This example intentionally uses direct MPI calls (#include <mpi.h>)
// because it requires MPI_Comm_split to create sub-communicators (node-local
// and inter-node leader communicators). DTL does not yet provide a
// sub-communicator creation API. See KNOWN_ISSUES.md for details.

#include <dtl/dtl.hpp>
#include <iostream>
#include <vector>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    dtl::rank_t my_rank = ctx.rank();
    dtl::rank_t num_ranks = ctx.size();

    // Simulate node topology: assume 2 ranks per node
    const dtl::rank_t ranks_per_node = 2;
    dtl::rank_t num_nodes = (num_ranks + ranks_per_node - 1) / ranks_per_node;
    dtl::rank_t my_node = my_rank / ranks_per_node;
    dtl::rank_t local_rank = my_rank % ranks_per_node;
    bool is_node_leader = (local_rank == 0);

    if (my_rank == 0) {
        std::cout << "DTL Hybrid Backend Example\n";
        std::cout << "==========================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Total ranks: " << num_ranks << "\n";
        std::cout << "  Simulated nodes: " << num_nodes << "\n";
        std::cout << "  Ranks per node: " << ranks_per_node << "\n\n";
    }

#if DTL_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // =========================================================================
    // Topology Discovery
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Topology Discovery ---\n";
    }

#if DTL_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << ": Node " << my_node
                      << ", local rank " << local_rank;
            if (is_node_leader) {
                std::cout << " (node leader)";
            }
            std::cout << "\n";
        }
#if DTL_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    // Each rank has some data (rank * 100 for easy verification)
    int my_value = my_rank * 100;

#if DTL_ENABLE_MPI
    // =========================================================================
    // Create node-local communicator
    // =========================================================================
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_node, local_rank, &node_comm);

    // Create inter-node communicator (only leaders)
    MPI_Comm leader_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_node_leader ? 0 : MPI_UNDEFINED,
                   my_node, &leader_comm);

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // =========================================================================
    // Intra-Node Reduction (would use shared memory in real implementation)
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Intra-Node Reduction (Shared Memory Simulation) ---\n";
        std::cout << "Each node reduces locally before inter-node communication.\n";
    }

#if DTL_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int node_sum = 0;

#if DTL_ENABLE_MPI
    // Reduce within node to leader
    MPI_Reduce(&my_value, &node_sum, 1, MPI_INT, MPI_SUM, 0, node_comm);
#else
    node_sum = my_value;
#endif

    // Print node sums (only leaders have valid values)
#if DTL_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for (dtl::rank_t n = 0; n < num_nodes; ++n) {
        if (my_node == n && is_node_leader) {
            std::cout << "Node " << n << " local sum: " << node_sum
                      << " (ranks " << (n * ranks_per_node);
            if (ranks_per_node > 1) {
                std::cout << "," << (n * ranks_per_node + 1);
            }
            std::cout << ")\n";
        }
#if DTL_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    // =========================================================================
    // Inter-Node Reduction (MPI between leaders only)
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Inter-Node Reduction (MPI) ---\n";
        std::cout << "Only node leaders participate in cross-node communication.\n";
    }

#if DTL_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int global_sum = 0;

#if DTL_ENABLE_MPI
    if (is_node_leader && leader_comm != MPI_COMM_NULL) {
        MPI_Allreduce(&node_sum, &global_sum, 1, MPI_INT, MPI_SUM, leader_comm);
    }
#else
    global_sum = node_sum;
#endif

    if (my_rank == 0) {
        std::cout << "Global sum: " << global_sum << "\n";
    }

    // =========================================================================
    // Broadcast Back to Non-Leaders
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Broadcast Back to Non-Leaders ---\n";
        std::cout << "All ranks now have global result.\n";
    }

#if DTL_ENABLE_MPI
    // Broadcast from node leader to other ranks in node
    MPI_Bcast(&global_sum, 1, MPI_INT, 0, node_comm);

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Verify all ranks have the correct global sum
    // Expected: 0*100 + 1*100 + 2*100 + 3*100 = 600 (for 4 ranks)
    int expected = 0;
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        expected += r * 100;
    }

    bool correct = (global_sum == expected);

#if DTL_ENABLE_MPI
    int local_correct = correct ? 1 : 0;
    int all_correct = 0;
    MPI_Allreduce(&local_correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    correct = (all_correct == 1);
#endif

    if (my_rank == 0) {
        std::cout << "\n--- Verification ---\n";
        std::cout << "Expected global sum: " << expected << "\n";
        std::cout << "Computed global sum: " << global_sum << "\n";
        std::cout << "All ranks correct: " << std::boolalpha << correct << "\n\n";
        std::cout << "SUCCESS: Hybrid backend example completed!\n";
    }

#if DTL_ENABLE_MPI
    // Cleanup
    MPI_Comm_free(&node_comm);
    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }
#endif

    return correct ? 0 : 1;
}
