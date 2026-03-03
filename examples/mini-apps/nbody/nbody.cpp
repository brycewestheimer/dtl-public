// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file nbody.cpp
/// @brief 2D gravitational N-body simulation using DTL
///
/// Direct-summation O(N^2/P) N-body with leapfrog (velocity Verlet) integration.
/// Particles interact via softened gravitational force. Energy conservation is
/// checked each timestep as a correctness diagnostic.
///
/// Demonstrates:
/// - dtl::environment + env.make_world_context() — RAII backend lifecycle
/// - dtl::distributed_vector<double> — distributed particle state (x, y, vx, vy, mass)
/// - local_view() — local force computation and integration
/// - comm.allgather() — share all positions each timestep (all-to-all comm pattern)
/// - comm.allreduce_sum_value<double>() — global energy computation
/// - global_offset() / local_size() — partition-aware indexing
///
/// Build (in-tree):
///   cmake .. -DDTL_BUILD_EXAMPLES=ON && make nbody
///
/// Build (standalone, against installed DTL):
///   mkdir build && cd build
///   cmake .. -DCMAKE_PREFIX_PATH=/path/to/dtl/install && make
///
/// Run:
///   ./nbody                    # single rank
///   mpirun -np 4 ./nbody       # 4 ranks

#include <dtl/dtl.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

/// Compute total kinetic + potential energy across all particles.
/// Each rank computes partial KE for its own particles and partial PE for
/// unique pairs where the lower-index particle is local.
static double compute_energy(
    dtl::mpi::mpi_comm_adapter& comm,
    const double* all_x, const double* all_y,
    const double* all_mass,
    const double* local_vx, const double* local_vy,
    dtl::size_type local_n, dtl::size_type global_n,
    dtl::index_t global_off, double softening)
{
    // Kinetic energy: 0.5 * m * v^2 (local particles only)
    double local_ke = 0.0;
    for (dtl::size_type i = 0; i < local_n; ++i) {
        dtl::size_type gi = static_cast<dtl::size_type>(global_off) + i;
        double v2 = local_vx[i] * local_vx[i] + local_vy[i] * local_vy[i];
        local_ke += 0.5 * all_mass[gi] * v2;
    }

    // Potential energy: -G * m_i * m_j / r_ij for unique pairs (i < j)
    // Each rank handles pairs where i is in its local range
    double local_pe = 0.0;
    for (dtl::size_type i = 0; i < local_n; ++i) {
        dtl::size_type gi = static_cast<dtl::size_type>(global_off) + i;
        for (dtl::size_type j = gi + 1; j < global_n; ++j) {
            double dx = all_x[j] - all_x[gi];
            double dy = all_y[j] - all_y[gi];
            double r = std::sqrt(dx * dx + dy * dy + softening * softening);
            local_pe -= all_mass[gi] * all_mass[j] / r;
        }
    }

    double total = comm.allreduce_sum_value<double>(local_ke + local_pe);
    return total;
}

int main(int argc, char** argv) {
    // --- DTL initialization ---
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    // --- Simulation parameters ---
    const dtl::size_type global_n = 128;  // Total particles
    const int num_steps = 100;
    const double dt = 0.001;
    const double softening = 0.01;  // Gravitational softening length

    if (rank == 0) {
        std::cout << "DTL N-Body Gravitational Simulation (2D)\n";
        std::cout << "========================================\n";
        std::cout << "Particles: " << global_n << "\n";
        std::cout << "Timesteps: " << num_steps << "\n";
        std::cout << "dt:        " << dt << "\n";
        std::cout << "Softening: " << softening << "\n";
        std::cout << "Ranks:     " << size << "\n\n";
    }
    comm.barrier();

    // --- Distributed particle state ---
    // Each rank owns global_n/size particles (block partition)
    dtl::distributed_vector<double> px_vec(global_n, ctx);  // position x
    dtl::distributed_vector<double> py_vec(global_n, ctx);  // position y
    dtl::distributed_vector<double> vx_vec(global_n, ctx);  // velocity x
    dtl::distributed_vector<double> vy_vec(global_n, ctx);  // velocity y
    dtl::distributed_vector<double> m_vec(global_n, ctx);   // mass

    auto px_local = px_vec.local_view();
    auto py_local = py_vec.local_view();
    auto vx_local = vx_vec.local_view();
    auto vy_local = vy_vec.local_view();
    auto m_local  = m_vec.local_view();

    dtl::size_type local_n = px_vec.local_size();
    dtl::index_t global_off = px_vec.global_offset();

    // Print partition info
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << rank << ": particles ["
                      << global_off << ", "
                      << (global_off + static_cast<dtl::index_t>(local_n))
                      << ") — " << local_n << " particles\n";
        }
        comm.barrier();
    }
    if (rank == 0) std::cout << "\n";
    comm.barrier();

    // --- Initialize particles in a ring configuration ---
    // Particles are placed on a circle with tangential velocities,
    // providing a stable-ish initial condition where energy is roughly conserved.
    constexpr double pi = 3.14159265358979323846;
    const double radius = 1.0;
    const double v_orbit = 0.5;  // Orbital speed

    for (dtl::size_type i = 0; i < local_n; ++i) {
        dtl::size_type gi = static_cast<dtl::size_type>(global_off) + i;
        double angle = 2.0 * pi * static_cast<double>(gi) / static_cast<double>(global_n);

        px_local[i] = radius * std::cos(angle);
        py_local[i] = radius * std::sin(angle);
        // Tangential velocity (perpendicular to radius)
        vx_local[i] = -v_orbit * std::sin(angle);
        vy_local[i] =  v_orbit * std::cos(angle);
        m_local[i]  = 1.0 / static_cast<double>(global_n);  // Equal mass, total = 1
    }

    // --- Global buffers for allgathered positions and masses ---
    std::vector<double> all_x(global_n);
    std::vector<double> all_y(global_n);
    std::vector<double> all_mass(global_n);

    // Allgather masses (constant, only need to do once)
    comm.allgather(&m_local[0], all_mass.data(),
                   local_n * sizeof(double));

    // Allgather initial positions for energy computation
    comm.allgather(&px_local[0], all_x.data(),
                   local_n * sizeof(double));
    comm.allgather(&py_local[0], all_y.data(),
                   local_n * sizeof(double));

    double E0 = compute_energy(comm, all_x.data(), all_y.data(), all_mass.data(),
                               &vx_local[0], &vy_local[0],
                               local_n, global_n, global_off, softening);

    if (rank == 0) {
        std::cout << "Initial energy: " << std::scientific << std::setprecision(6)
                  << E0 << "\n\n";
    }

    // --- Force buffers (local only) ---
    std::vector<double> fx(local_n, 0.0);
    std::vector<double> fy(local_n, 0.0);

    // --- Leapfrog (velocity Verlet) time integration ---
    for (int step = 0; step < num_steps; ++step) {
        // 1. Allgather current positions
        comm.allgather(&px_local[0], all_x.data(),
                       local_n * sizeof(double));
        comm.allgather(&py_local[0], all_y.data(),
                       local_n * sizeof(double));

        // 2. Compute forces on local particles from all particles
        for (dtl::size_type i = 0; i < local_n; ++i) {
            fx[i] = 0.0;
            fy[i] = 0.0;
            dtl::size_type gi = static_cast<dtl::size_type>(global_off) + i;

            for (dtl::size_type j = 0; j < global_n; ++j) {
                if (j == gi) continue;

                double dx = all_x[j] - all_x[gi];
                double dy = all_y[j] - all_y[gi];
                double r2 = dx * dx + dy * dy + softening * softening;
                double inv_r3 = 1.0 / (r2 * std::sqrt(r2));

                // F = G * m_i * m_j / r^2 * r_hat = m_j / r^3 * dr
                // (G = 1, acceleration = F / m_i = m_j / r^3 * dr)
                fx[i] += all_mass[j] * dx * inv_r3;
                fy[i] += all_mass[j] * dy * inv_r3;
            }
        }

        // 3. Leapfrog integration
        for (dtl::size_type i = 0; i < local_n; ++i) {
            // Update velocity by half step (kick)
            vx_local[i] += 0.5 * dt * fx[i];
            vy_local[i] += 0.5 * dt * fy[i];

            // Update position (drift)
            px_local[i] += dt * vx_local[i];
            py_local[i] += dt * vy_local[i];
        }

        // Recompute forces at new positions for second velocity half-step
        comm.allgather(&px_local[0], all_x.data(),
                       local_n * sizeof(double));
        comm.allgather(&py_local[0], all_y.data(),
                       local_n * sizeof(double));

        for (dtl::size_type i = 0; i < local_n; ++i) {
            double fx2 = 0.0;
            double fy2 = 0.0;
            dtl::size_type gi = static_cast<dtl::size_type>(global_off) + i;

            for (dtl::size_type j = 0; j < global_n; ++j) {
                if (j == gi) continue;

                double dx = all_x[j] - all_x[gi];
                double dy = all_y[j] - all_y[gi];
                double r2 = dx * dx + dy * dy + softening * softening;
                double inv_r3 = 1.0 / (r2 * std::sqrt(r2));

                fx2 += all_mass[j] * dx * inv_r3;
                fy2 += all_mass[j] * dy * inv_r3;
            }

            // Second velocity half-step (kick)
            vx_local[i] += 0.5 * dt * fx2;
            vy_local[i] += 0.5 * dt * fy2;
        }

        // 4. Periodic energy check
        if ((step + 1) % 10 == 0 || step == 0 || step == num_steps - 1) {
            double E = compute_energy(comm, all_x.data(), all_y.data(),
                                      all_mass.data(),
                                      &vx_local[0], &vy_local[0],
                                      local_n, global_n, global_off, softening);
            double dE = std::abs((E - E0) / E0);

            if (rank == 0) {
                std::cout << "  Step " << std::setw(4) << (step + 1)
                          << ": E = " << std::scientific << std::setprecision(6) << E
                          << "  |dE/E0| = " << std::setprecision(2) << dE << "\n";
            }
        }
    }

    // --- Final energy and verification ---
    comm.allgather(&px_local[0], all_x.data(),
                   local_n * sizeof(double));
    comm.allgather(&py_local[0], all_y.data(),
                   local_n * sizeof(double));

    double E_final = compute_energy(comm, all_x.data(), all_y.data(),
                                    all_mass.data(),
                                    &vx_local[0], &vy_local[0],
                                    local_n, global_n, global_off, softening);
    double relative_energy_drift = std::abs((E_final - E0) / E0);

    if (rank == 0) {
        std::cout << "\nFinal energy:    " << std::scientific << std::setprecision(6)
                  << E_final << "\n";
        std::cout << "Initial energy:  " << E0 << "\n";
        std::cout << "Relative drift:  " << std::setprecision(4)
                  << relative_energy_drift << "\n";

        // With leapfrog and small dt, energy should be conserved to ~O(dt^2)
        bool ok = relative_energy_drift < 0.01;
        std::cout << (ok ? "SUCCESS" : "FAILURE")
                  << ": Energy " << (ok ? "conserved" : "not conserved")
                  << " (< 1% drift)\n";
    }

    return 0;
}
