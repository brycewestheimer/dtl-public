// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file role_manager_dtl.cpp
/// @brief MPMD Role Manager using DTL abstractions
///
/// Demonstrates:
/// - dtl::mpmd::role_manager for role assignment
/// - Coordinator (rank 0) distributes tasks to workers
/// - All P2P communication through DTL comm adapter
///
/// Run:
///   mpirun -np 4 ./role_manager_dtl

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>

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
        std::cout << "DTL MPMD Role Manager Example\n";
        std::cout << "==============================\n";
        std::cout << "Ranks: " << size << "\n";
        std::cout << "Coordinator: rank 0\n";
        std::cout << "Workers: ranks 1.." << (size - 1) << "\n\n";
    }
    comm.barrier();

    const int task_tag = 1;
    const int result_tag = 2;

    bool ok = true;
    if (rank == 0) {
        // --- Coordinator ---
        // Distribute tasks: each worker gets a number to square
        for (dtl::rank_t w = 1; w < size; ++w) {
            int task_data = static_cast<int>(w) * 10;
            comm.send(&task_data, sizeof(int), w, task_tag);
            std::cout << "Coordinator: sent task " << task_data
                      << " to worker " << w << "\n";
        }

        // Collect results
        std::cout << "\nResults:\n";
        for (dtl::rank_t w = 1; w < size; ++w) {
            int result = 0;
            comm.recv(&result, sizeof(int), w, result_tag);
            std::cout << "  Worker " << w << " returned: " << result << "\n";

            const int task_data = static_cast<int>(w) * 10;
            const int expected = task_data * task_data;
            ok = ok && (result == expected);
        }

        std::cout << "\nDone!\n";
    } else {
        // --- Worker ---
        int task_data = 0;
        comm.recv(&task_data, sizeof(int), 0, task_tag);

        // Compute: square the input
        int result = task_data * task_data;
        ok = (result == task_data * task_data);

        std::cout << "Worker " << rank << ": received " << task_data
                  << ", computed " << result << "\n";

        comm.send(&result, sizeof(int), 0, result_tag);
    }

    const bool all_ok = comm.allreduce_land_value(ok);

    comm.barrier();

    if (rank == 0) {
        std::cout << "\n" << (all_ok ? "SUCCESS" : "FAILURE") << "\n";
    }

    return all_ok ? 0 : 1;
}
