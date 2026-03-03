// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file heterogeneous_roles.cpp
/// @brief Multiple role types in a distributed application
/// @details Demonstrates DTL's MPMD support for applications where different
///          ranks serve different purposes: producers generate data, processors
///          transform it, and consumers aggregate results.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 6 ./heterogeneous_roles
///
/// Expected output (6 ranks):
///   DTL Heterogeneous Roles Example
///   ===============================
///
///   Configuration:
///     Total ranks: 6
///     Producers: 2 (ranks 0-1)
///     Processors: 2 (ranks 2-3)
///     Consumers: 2 (ranks 4-5)
///
///   --- Role Assignment ---
///   Rank 0: PRODUCER
///   Rank 1: PRODUCER
///   Rank 2: PROCESSOR
///   Rank 3: PROCESSOR
///   Rank 4: CONSUMER
///   Rank 5: CONSUMER
///
///   --- Data Flow ---
///   [Producer 0] Generated 10 items
///   [Producer 1] Generated 10 items
///   [Processor 2] Received 10 items, transforming...
///   [Processor 3] Received 10 items, transforming...
///   [Consumer 4] Received 10 transformed items
///   [Consumer 5] Received 10 transformed items
///
///   --- Final Aggregation ---
///   Consumer 4 aggregate: 90
///   Consumer 5 aggregate: 190
///   Global total: 280
///
///   SUCCESS: Heterogeneous roles example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <numeric>

// Role definitions
enum class Role {
    PRODUCER,
    PROCESSOR,
    CONSUMER
};

const char* role_name(Role r) {
    switch (r) {
        case Role::PRODUCER: return "PRODUCER";
        case Role::PROCESSOR: return "PROCESSOR";
        case Role::CONSUMER: return "CONSUMER";
    }
    return "UNKNOWN";
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    // Determine role based on rank
    // For 6 ranks: 0-1 producers, 2-3 processors, 4-5 consumers
    // For other sizes, divide into thirds
    int third = (num_ranks + 2) / 3;
    Role my_role;
    if (my_rank < third) {
        my_role = Role::PRODUCER;
    } else if (my_rank < 2 * third) {
        my_role = Role::PROCESSOR;
    } else {
        my_role = Role::CONSUMER;
    }

    // Count ranks per role
    int num_producers = third;
    int num_processors = third;
    int num_consumers = num_ranks - 2 * third;
    if (num_consumers < 1) num_consumers = 1;

    const int items_per_producer = 10;

    if (my_rank == 0) {
        std::cout << "DTL Heterogeneous Roles Example\n";
        std::cout << "===============================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Total ranks: " << num_ranks << "\n";
        std::cout << "  Producers: " << num_producers << " (ranks 0-" << (num_producers - 1) << ")\n";
        std::cout << "  Processors: " << num_processors << " (ranks " << num_producers << "-"
                  << (num_producers + num_processors - 1) << ")\n";
        std::cout << "  Consumers: " << num_consumers << " (ranks " << (2 * third) << "-"
                  << (num_ranks - 1) << ")\n\n";
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
            Role r_role;
            if (r < third) r_role = Role::PRODUCER;
            else if (r < 2 * third) r_role = Role::PROCESSOR;
            else r_role = Role::CONSUMER;
            std::cout << "Rank " << r << ": " << role_name(r_role) << "\n";
        }
        comm.barrier();
    }

    if (my_rank == 0) {
        std::cout << "\n--- Data Flow ---\n";
    }

    comm.barrier();

    // Data storage
    std::vector<int> my_data;
    int my_aggregate = 0;

    // =========================================================================
    // Producer Phase
    // =========================================================================
    if (my_role == Role::PRODUCER) {
        // Generate data
        my_data.resize(static_cast<size_t>(items_per_producer));
        for (int i = 0; i < items_per_producer; ++i) {
            my_data[static_cast<size_t>(i)] = my_rank * items_per_producer + i;
        }
        std::cout << "[Producer " << my_rank << "] Generated " << items_per_producer << " items\n";

        // Send to corresponding processor
        int target_processor = third + (my_rank % num_processors);
        comm.send(my_data.data(),
                  static_cast<dtl::size_type>(items_per_producer) * sizeof(int),
                  static_cast<dtl::rank_t>(target_processor), 1);
    }

    comm.barrier();

    // =========================================================================
    // Processor Phase
    // =========================================================================
    if (my_role == Role::PROCESSOR) {
        // Receive from producer
        my_data.resize(static_cast<size_t>(items_per_producer));
        int source_producer = (my_rank - third) % num_producers;
        comm.recv(my_data.data(),
                  static_cast<dtl::size_type>(items_per_producer) * sizeof(int),
                  static_cast<dtl::rank_t>(source_producer), 1);

        std::cout << "[Processor " << my_rank << "] Received " << items_per_producer
                  << " items, transforming...\n";

        // Transform: square each value
        for (auto& val : my_data) {
            val = val * val;
        }

        // Send to corresponding consumer
        int target_consumer = 2 * third + ((my_rank - third) % num_consumers);
        comm.send(my_data.data(),
                  static_cast<dtl::size_type>(items_per_producer) * sizeof(int),
                  static_cast<dtl::rank_t>(target_consumer), 2);
    }

    comm.barrier();

    // =========================================================================
    // Consumer Phase
    // =========================================================================
    if (my_role == Role::CONSUMER) {
        // Receive from processor
        my_data.resize(static_cast<size_t>(items_per_producer));
        int source_processor = third + ((my_rank - 2 * third) % num_processors);
        comm.recv(my_data.data(),
                  static_cast<dtl::size_type>(items_per_producer) * sizeof(int),
                  static_cast<dtl::rank_t>(source_processor), 2);

        std::cout << "[Consumer " << my_rank << "] Received " << items_per_producer
                  << " transformed items\n";

        // Aggregate
        my_aggregate = std::accumulate(my_data.begin(), my_data.end(), 0);
    }

    comm.barrier();

    // =========================================================================
    // Final Aggregation
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Final Aggregation ---\n";
    }

    comm.barrier();

    // Print consumer aggregates
    for (dtl::rank_t r = 2 * third; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Consumer " << r << " aggregate: " << my_aggregate << "\n";
        }
        comm.barrier();
    }

    // Global reduction of consumer aggregates
    int local_contrib = (my_role == Role::CONSUMER) ? my_aggregate : 0;
    int global_total = comm.allreduce_sum_value<int>(local_contrib);

    if (my_rank == 0) {
        std::cout << "Global total: " << global_total << "\n\n";
        std::cout << "SUCCESS: Heterogeneous roles example completed!\n";
    }

    return 0;
}
