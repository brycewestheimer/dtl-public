// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file coordinator_worker.cpp
/// @brief Classic coordinator/worker pattern using DTL's MPMD support
/// @details Demonstrates the coordinator/worker (master/slave) pattern where
///          one rank coordinates work distribution while others execute tasks.
///          This is a fundamental pattern for dynamic load balancing.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./coordinator_worker
///
/// Expected output (4 ranks):
///   DTL Coordinator/Worker Pattern Example
///   ======================================
///
///   Configuration:
///     Total ranks: 4
///     Coordinator: Rank 0
///     Workers: Ranks 1-3
///     Tasks: 12
///
///   --- Role Assignment ---
///   Rank 0: COORDINATOR
///   Rank 1: WORKER
///   Rank 2: WORKER
///   Rank 3: WORKER
///
///   --- Task Distribution ---
///   [Coordinator] Distributing 12 tasks to 3 workers...
///   [Worker 1] Received task 0, computing...
///   [Worker 2] Received task 1, computing...
///   [Worker 3] Received task 2, computing...
///   [Worker 1] Completed task 0, result: 0
///   ...
///   [Coordinator] All tasks completed!
///
///   --- Results ---
///   Total results collected: 12
///   Sum of all results: 66
///   Expected (0+1+...+11): 66
///
///   SUCCESS: Coordinator/worker example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <queue>

// Message tags
const int TAG_TASK = 1;
const int TAG_RESULT = 2;

// Sentinel value: task_id == -1 means "no more work"
const int TASK_DONE = -1;

// Simulated computation: returns task_id (for verification)
int compute_task(int task_id) {
    return task_id;
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    const int num_tasks = 12;
    const dtl::rank_t coordinator_rank = 0;
    const bool is_coordinator = (my_rank == coordinator_rank);
    const int num_workers = num_ranks - 1;

    if (my_rank == 0) {
        std::cout << "DTL Coordinator/Worker Pattern Example\n";
        std::cout << "======================================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Total ranks: " << num_ranks << "\n";
        std::cout << "  Coordinator: Rank " << coordinator_rank << "\n";
        std::cout << "  Workers: Ranks 1-" << (num_ranks - 1) << "\n";
        std::cout << "  Tasks: " << num_tasks << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // Role Assignment
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Role Assignment ---\n";
    }

    comm.barrier();

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << ": " << (is_coordinator ? "COORDINATOR" : "WORKER") << "\n";
        }
        comm.barrier();
    }

    if (num_workers == 0) {
        if (my_rank == 0) {
            std::cout << "\nNeed at least 2 ranks for coordinator/worker pattern.\n";
            std::cout << "Run with: mpirun -np 4 ./coordinator_worker\n";
        }
        return 1;
    }

    // =========================================================================
    // Task Distribution (Coordinator)
    // =========================================================================
    // Note: DTL adapter does not support MPI_ANY_SOURCE. The coordinator polls
    // each worker by posting irecv() per worker and testing in round-robin.
    // See KNOWN_ISSUES.md for details on the MPI_ANY_SOURCE gap.
    std::vector<int> all_results;

    if (is_coordinator) {
        std::cout << "\n--- Task Distribution ---\n";
        std::cout << "[Coordinator] Distributing " << num_tasks << " tasks to "
                  << num_workers << " workers...\n";

        std::queue<int> pending_tasks;
        for (int i = 0; i < num_tasks; ++i) {
            pending_tasks.push(i);
        }

        // Per-worker receive buffers and request handles
        std::vector<int> recv_bufs(static_cast<size_t>(num_ranks), 0);
        std::vector<dtl::request_handle> recv_reqs(static_cast<size_t>(num_ranks));
        std::vector<bool> worker_active(static_cast<size_t>(num_ranks), false);
        int active_workers = 0;

        // Initial distribution: send one task to each worker and post irecv
        for (dtl::rank_t w = 1; w < num_ranks && !pending_tasks.empty(); ++w) {
            int task = pending_tasks.front();
            pending_tasks.pop();
            comm.send(&task, sizeof(int), w, TAG_TASK);
            recv_reqs[static_cast<size_t>(w)] = comm.irecv(
                &recv_bufs[static_cast<size_t>(w)], sizeof(int), w, TAG_RESULT);
            worker_active[static_cast<size_t>(w)] = true;
            active_workers++;
        }

        // Poll workers in round-robin for completed results
        while (active_workers > 0) {
            for (dtl::rank_t w = 1; w < num_ranks; ++w) {
                if (!worker_active[static_cast<size_t>(w)]) continue;

                if (comm.test(recv_reqs[static_cast<size_t>(w)])) {
                    int result = recv_bufs[static_cast<size_t>(w)];
                    all_results.push_back(result);
                    std::cout << "[Coordinator] Received result " << result
                              << " from Worker " << w << "\n";

                    // Send next task or done signal
                    if (!pending_tasks.empty()) {
                        int task = pending_tasks.front();
                        pending_tasks.pop();
                        comm.send(&task, sizeof(int), w, TAG_TASK);
                        recv_reqs[static_cast<size_t>(w)] = comm.irecv(
                            &recv_bufs[static_cast<size_t>(w)], sizeof(int),
                            w, TAG_RESULT);
                    } else {
                        int done_signal = TASK_DONE;
                        comm.send(&done_signal, sizeof(int), w, TAG_TASK);
                        worker_active[static_cast<size_t>(w)] = false;
                        active_workers--;
                    }
                }
            }
        }

        std::cout << "[Coordinator] All tasks completed!\n";

    } else {
        // =========================================================================
        // Task Execution (Workers)
        // =========================================================================
        // Workers receive tasks on TAG_TASK. A sentinel value of TASK_DONE (-1)
        // signals completion (avoids needing MPI_ANY_TAG).
        while (true) {
            int task;
            comm.recv(&task, sizeof(int), coordinator_rank, TAG_TASK);

            if (task == TASK_DONE) {
                break;  // No more tasks
            }

            // Process task
            int result = compute_task(task);

            // Send result back
            comm.send(&result, sizeof(int), coordinator_rank, TAG_RESULT);
        }
    }

    comm.barrier();

    // =========================================================================
    // Results
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Results ---\n";
        std::cout << "Total results collected: " << all_results.size() << "\n";

        int sum = 0;
        for (int r : all_results) {
            sum += r;
        }
        std::cout << "Sum of all results: " << sum << "\n";

        int expected = num_tasks * (num_tasks - 1) / 2;
        std::cout << "Expected (0+1+...+" << (num_tasks - 1) << "): " << expected << "\n\n";

        if (sum == expected && static_cast<int>(all_results.size()) == num_tasks) {
            std::cout << "SUCCESS: Coordinator/worker example completed!\n";
        } else {
            std::cout << "FAILURE: Results don't match expected!\n";
        }
    }

    return 0;
}
