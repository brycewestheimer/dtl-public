// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file point_to_point_dtl.cpp
/// @brief Point-to-point ring communication using DTL
///
/// Demonstrates:
/// - dtl::environment + make_world_context() initialization
/// - comm.send() / comm.recv() for blocking P2P
/// - Deadlock-free ring pattern with odd/even ordering
///
/// Run:
///   mpirun -np 4 ./point_to_point_dtl

#include <dtl/dtl.hpp>

#include <iostream>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    auto rank = ctx.rank();
    auto size = ctx.size();

    if (size < 2) {
        if (rank == 0) std::cout << "This example requires at least 2 ranks.\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "DTL Point-to-Point Ring Example\n";
        std::cout << "================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }
    comm.barrier();

    // Ring communication: send to next, receive from previous
    dtl::rank_t next = (rank + 1) % size;
    dtl::rank_t prev = (rank + size - 1) % size;

    int send_val = rank * 100;
    int recv_val = -1;
    const int tag = 0;

    // Deadlock-free ordering: even ranks send first, odd ranks recv first
    if (rank % 2 == 0) {
        comm.send(&send_val, sizeof(int), next, tag);
        comm.recv(&recv_val, sizeof(int), prev, tag);
    } else {
        comm.recv(&recv_val, sizeof(int), prev, tag);
        comm.send(&send_val, sizeof(int), next, tag);
    }

    // Print results in rank order
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << rank
                      << ": sent " << send_val << " -> rank " << next
                      << ", received " << recv_val << " <- rank " << prev << "\n";
        }
        comm.barrier();
    }

    // Verify: each rank should have received prev*100
    int expected = static_cast<int>(prev) * 100;
    bool ok = (recv_val == expected);

    bool all_ok = comm.allreduce_land_value(ok);

    if (rank == 0) {
        std::cout << "\n" << (all_ok ? "SUCCESS" : "FAILURE")
                  << ": Ring communication " << (all_ok ? "correct" : "incorrect") << "\n";
    }

    return 0;
}
