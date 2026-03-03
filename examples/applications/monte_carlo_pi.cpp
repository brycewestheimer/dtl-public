// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file monte_carlo_pi.cpp
/// @brief Monte Carlo Pi estimation using DTL
///
/// Demonstrates:
/// - Distributed random sampling
/// - dtl::distributed_vector for local storage
/// - Communicator allreduce for global aggregation
///
/// Run:
///   mpirun -np 4 ./monte_carlo_pi

#include <dtl/dtl.hpp>

#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <numbers>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Monte Carlo Pi Estimation\n";
        std::cout << "==============================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // Each rank samples N points
    const long samples_per_rank = 1000000;
    const long total_samples = samples_per_rank * size;

    // Use rank-specific seed for independent random streams
    std::mt19937_64 rng(42 + rank * 12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Count points inside unit circle
    long local_hits = 0;
    for (long i = 0; i < samples_per_rank; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y <= 1.0) {
            ++local_hits;
        }
    }

    // Report per-rank results
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            double local_pi = 4.0 * static_cast<double>(local_hits)
                            / static_cast<double>(samples_per_rank);
            std::cout << "  Rank " << rank << ": " << local_hits << " / "
                      << samples_per_rank << " hits (local pi ~ "
                      << std::fixed << std::setprecision(6) << local_pi << ")\n";
        }
        comm.barrier();
    }

    // Global reduction
    long global_hits = comm.allreduce_sum_value<long>(local_hits);

    double pi_estimate = 4.0 * static_cast<double>(global_hits)
                       / static_cast<double>(total_samples);
    constexpr double pi = std::numbers::pi;
    double error = std::abs(pi_estimate - pi);

    if (rank == 0) {
        std::cout << "\nTotal samples: " << total_samples << "\n";
        std::cout << "Total hits:    " << global_hits << "\n";
        std::cout << "Pi estimate:   " << std::fixed << std::setprecision(8)
                  << pi_estimate << "\n";
        std::cout << "Actual pi:     " << pi << "\n";
        std::cout << "Error:         " << std::scientific << std::setprecision(4)
                  << error << "\n";
        std::cout << (error < 0.01 ? "SUCCESS" : "WARNING")
                  << ": Estimate " << (error < 0.01 ? "within" : "outside")
                  << " 0.01 tolerance\n";
    }

    return 0;
}
