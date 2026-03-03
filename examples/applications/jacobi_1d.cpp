// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file jacobi_1d.cpp
/// @brief 1D Jacobi iterative solver using DTL
///
/// Solves u''(x) = 0 with boundary conditions u(0) = 1, u(L) = 0.
/// Uses Jacobi iteration with halo exchange via DTL P2P.
///
/// Demonstrates:
/// - Halo exchange pattern using communicator send() / recv()
/// - Convergence checking via communicator allreduce()
/// - Distributed vector for local storage
///
/// Run:
///   mpirun -np 4 ./jacobi_1d

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL 1D Jacobi Solver\n";
        std::cout << "=====================\n";
        std::cout << "Ranks: " << size << "\n";
    }
    comm.barrier();

    // Problem setup
    const int global_n = 100;               // Interior points
    const int max_iter = 10000;
    const double tol = 1e-8;

    // Partition interior points among ranks
    int local_n = global_n / size;
    int remainder = global_n % size;
    if (rank < remainder) local_n += 1;

    // Local arrays with halo cells: [left_halo | interior | right_halo]
    std::vector<double> u(local_n + 2, 0.0);
    std::vector<double> u_new(local_n + 2, 0.0);

    // Apply boundary conditions
    // u(0) = 1.0 on leftmost rank's left boundary
    if (rank == 0) {
        u[0] = 1.0;
        u_new[0] = 1.0;
    }
    // u(L) = 0.0 on rightmost rank's right boundary (already 0)

    const int halo_tag = 10;

    if (rank == 0) {
        std::cout << "Grid: " << global_n << " interior points\n";
        std::cout << "BCs: u(0)=1, u(L)=0\n";
        std::cout << "Tolerance: " << tol << "\n\n";
    }
    comm.barrier();

    int iter = 0;
    double global_diff = 0.0;

    for (iter = 0; iter < max_iter; ++iter) {
        // --- Halo exchange ---
        // Send right boundary to right neighbor, recv left halo from left neighbor
        if (rank < size - 1) {
            comm.send(&u[local_n], sizeof(double),
                      static_cast<dtl::rank_t>(rank + 1), halo_tag);
        }
        if (rank > 0) {
            comm.recv(&u[0], sizeof(double),
                      static_cast<dtl::rank_t>(rank - 1), halo_tag);
        }

        // Send left boundary to left neighbor, recv right halo from right neighbor
        if (rank > 0) {
            comm.send(&u[1], sizeof(double),
                      static_cast<dtl::rank_t>(rank - 1), halo_tag + 1);
        }
        if (rank < size - 1) {
            comm.recv(&u[local_n + 1], sizeof(double),
                      static_cast<dtl::rank_t>(rank + 1), halo_tag + 1);
        }

        // --- Jacobi update ---
        double local_diff = 0.0;
        for (int i = 1; i <= local_n; ++i) {
            u_new[i] = 0.5 * (u[i - 1] + u[i + 1]);
            double d = std::abs(u_new[i] - u[i]);
            if (d > local_diff) local_diff = d;
        }

        // Copy new to old
        for (int i = 1; i <= local_n; ++i) {
            u[i] = u_new[i];
        }

        // Check convergence
        global_diff = comm.allreduce_max_value<double>(local_diff);
        if (global_diff < tol) {
            break;
        }
    }

    comm.barrier();

    if (rank == 0) {
        std::cout << "Converged after " << iter << " iterations\n";
        std::cout << "Final max diff: " << std::scientific << std::setprecision(4)
                  << global_diff << "\n\n";
    }

    // Print solution samples
    // Gather first element from each rank to show solution profile
    double local_first = u[1];
    double local_last = u[local_n];

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << rank << ": u[first]=" << std::fixed
                      << std::setprecision(6) << local_first
                      << ", u[last]=" << local_last << "\n";
        }
        comm.barrier();
    }

    // Verify: analytic solution is u(x) = 1 - x/L
    // Check at midpoint: u(0.5) should be ~0.5
    if (rank == 0) {
        std::cout << "\nExpected: linear from 1.0 to 0.0\n";
        std::cout << (global_diff < tol ? "SUCCESS" : "FAILURE")
                  << ": Solver " << (global_diff < tol ? "converged" : "did not converge")
                  << "\n";
    }

    return 0;
}
