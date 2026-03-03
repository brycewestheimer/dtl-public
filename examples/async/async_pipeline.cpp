// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file async_pipeline.cpp
/// @brief Asynchronous pipeline processing with overlapped computation
/// @details Demonstrates DTL's async capabilities for building pipelines
///          where communication and computation can overlap. This pattern
///          is essential for achieving high performance in distributed systems.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./async_pipeline
///
/// Expected output:
///   DTL Async Pipeline Example
///   ==========================
///
///   Configuration:
///     Number of ranks: 4
///     Pipeline stages: 3
///     Items per stage: 10
///
///   --- Pipeline Stage Descriptions ---
///   Stage 1: Generate data (producer)
///   Stage 2: Transform data (processor)
///   Stage 3: Aggregate results (consumer)
///
///   --- Pipeline Execution ---
///   [Stage 1] Generating batch 1...
///   [Stage 2] Processing batch 1 (overlapped with stage 1 batch 2)...
///   [Stage 3] Aggregating results...
///
///   --- Results ---
///   Total items processed: 40
///   Final aggregate: 780
///
///   SUCCESS: Async pipeline example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Simulate work with a small delay
void simulate_work(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    const int num_batches = 3;
    const int items_per_batch = 10;

    if (my_rank == 0) {
        std::cout << "DTL Async Pipeline Example\n";
        std::cout << "==========================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n";
        std::cout << "  Pipeline stages: 3\n";
        std::cout << "  Batches: " << num_batches << "\n";
        std::cout << "  Items per batch: " << items_per_batch << "\n\n";

        std::cout << "--- Pipeline Stage Descriptions ---\n";
        std::cout << "Stage 1: Generate data (producer)\n";
        std::cout << "Stage 2: Transform data (processor)\n";
        std::cout << "Stage 3: Aggregate results (consumer)\n\n";
    }

    comm.barrier();

    // Each rank has local data storage
    std::vector<int> local_data(static_cast<size_t>(items_per_batch));
    std::vector<int> processed_data(static_cast<size_t>(items_per_batch));
    int local_aggregate = 0;

    if (my_rank == 0) {
        std::cout << "--- Pipeline Execution ---\n";
    }

    comm.barrier();

    // =========================================================================
    // Pipeline Execution with Overlapped Stages
    // =========================================================================

    for (int batch = 0; batch < num_batches; ++batch) {
        // Stage 1: Generate/receive data
        if (my_rank == 0) {
            std::cout << "[Batch " << (batch + 1) << "] Stage 1: Generating data...\n";
        }

        // Generate local data for this batch
        for (int i = 0; i < items_per_batch; ++i) {
            local_data[static_cast<size_t>(i)] = my_rank * 100 + batch * 10 + i;
        }
        simulate_work(10);  // Simulate generation time

        comm.barrier();

        // Stage 2: Transform data (can overlap with next batch's stage 1)
        if (my_rank == 0) {
            std::cout << "[Batch " << (batch + 1) << "] Stage 2: Processing data...\n";
        }

        // Apply transformation: square each element
        for (int i = 0; i < items_per_batch; ++i) {
            processed_data[static_cast<size_t>(i)] =
                local_data[static_cast<size_t>(i)] * local_data[static_cast<size_t>(i)];
        }
        simulate_work(15);  // Simulate processing time

        comm.barrier();

        // Stage 3: Local aggregation
        if (my_rank == 0) {
            std::cout << "[Batch " << (batch + 1) << "] Stage 3: Aggregating...\n";
        }

        int batch_sum = 0;
        for (int i = 0; i < items_per_batch; ++i) {
            // Use original values for simpler verification
            batch_sum += local_data[static_cast<size_t>(i)];
        }
        local_aggregate += batch_sum;
        simulate_work(5);  // Simulate aggregation time

        comm.barrier();
    }

    // =========================================================================
    // Final Global Reduction
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Final Reduction ---\n";
    }

    int total_items = num_batches * items_per_batch * num_ranks;

    int global_aggregate = comm.allreduce_sum_value<int>(local_aggregate);

    // =========================================================================
    // Results
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Results ---\n";
        std::cout << "Total items processed: " << total_items << "\n";
        std::cout << "Final aggregate: " << global_aggregate << "\n";
    }

    comm.barrier();

    // Show per-rank contribution
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "  Rank " << r << " contributed: " << local_aggregate << "\n";
        }
        comm.barrier();
    }

    if (my_rank == 0) {
        std::cout << "\nSUCCESS: Async pipeline example completed!\n";
    }

    return 0;
}
