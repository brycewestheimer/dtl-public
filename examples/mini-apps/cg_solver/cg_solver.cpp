// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cg_solver.cpp
/// @brief Conjugate Gradient solver for a 1D Laplacian system using DTL
///
/// Solves Ax = b where A is the 1D discrete Laplacian (tridiagonal: -1, 2, -1)
/// and b is chosen so the exact solution is x_i = sin(pi * i / (N+1)).
/// Uses the Conjugate Gradient method with nearest-neighbor halo exchange
/// for the sparse matrix-vector product.
///
/// Demonstrates:
/// - dtl::environment + env.make_world_context() — RAII backend lifecycle
/// - dtl::distributed_vector<double> — distributed data with block partitioning
/// - local_view() — STL-compatible access for local computation
/// - comm.allreduce_sum_value<double>() — global dot products
/// - comm.send() / comm.recv() — halo exchange for the tridiagonal stencil
/// - global_offset() / local_size() — partition-aware indexing
///
/// Build (in-tree):
///   cmake .. -DDTL_BUILD_EXAMPLES=ON && make cg_solver
///
/// Build (standalone, against installed DTL):
///   mkdir build && cd build
///   cmake .. -DCMAKE_PREFIX_PATH=/path/to/dtl/install && make
///
/// Run:
///   ./cg_solver                    # single rank
///   mpirun -np 4 ./cg_solver       # 4 ranks

#include <dtl/dtl.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

/// Compute the 1D Laplacian stencil y = A*x with halo exchange.
///
/// A is the (global_n x global_n) tridiagonal matrix with entries (-1, 2, -1).
/// Each rank owns a contiguous block of rows. The stencil requires one halo
/// element from each neighbor (left and right).
///
/// @param comm        MPI communicator for halo exchange
/// @param x_local     local view of the input vector (size = local_n)
/// @param y_local     local view of the output vector (size = local_n)
/// @param global_n    total problem size
/// @param global_off  global index of this rank's first element
/// @param rank        this rank's ID
/// @param size        total number of ranks
static void spmv_laplacian(
    dtl::mpi::mpi_comm_adapter& comm,
    const double* x_local, double* y_local,
    dtl::size_type local_n, dtl::size_type global_n,
    dtl::index_t global_off,
    dtl::rank_t rank, dtl::rank_t size)
{
    const int tag_left = 100;
    const int tag_right = 101;

    // Halo values from left and right neighbors
    double halo_left = 0.0;
    double halo_right = 0.0;

    // Exchange halos: send my leftmost to left neighbor, rightmost to right
    // Ordered to avoid deadlock: even ranks send first, odd ranks recv first
    // (but since send/recv are blocking, we use the send-right-then-send-left
    // pattern matching the existing jacobi_1d example)

    // Send rightmost element to right neighbor; recv left halo from left neighbor
    if (rank < size - 1) {
        comm.send(&x_local[local_n - 1], sizeof(double),
                  static_cast<dtl::rank_t>(rank + 1), tag_left);
    }
    if (rank > 0) {
        comm.recv(&halo_left, sizeof(double),
                  static_cast<dtl::rank_t>(rank - 1), tag_left);
    }

    // Send leftmost element to left neighbor; recv right halo from right neighbor
    if (rank > 0) {
        comm.send(&x_local[0], sizeof(double),
                  static_cast<dtl::rank_t>(rank - 1), tag_right);
    }
    if (rank < size - 1) {
        comm.recv(&halo_right, sizeof(double),
                  static_cast<dtl::rank_t>(rank + 1), tag_right);
    }

    // Apply the stencil: y_i = -x_{i-1} + 2*x_i - x_{i+1}
    for (dtl::size_type i = 0; i < local_n; ++i) {
        dtl::index_t gi = global_off + static_cast<dtl::index_t>(i);

        double left  = (i > 0)           ? x_local[i - 1] : halo_left;
        double right = (i < local_n - 1) ? x_local[i + 1] : halo_right;

        // At global boundaries, the missing neighbor is implicitly 0
        // (Dirichlet BCs: x_0 = x_{N+1} = 0)
        if (gi == 0)                                        left = 0.0;
        if (gi == static_cast<dtl::index_t>(global_n) - 1)  right = 0.0;

        y_local[i] = -left + 2.0 * x_local[i] - right;
    }
}

/// Compute dot(a, b) locally, then allreduce to get the global dot product.
static double global_dot(dtl::mpi::mpi_comm_adapter& comm,
                         const double* a, const double* b,
                         dtl::size_type n)
{
    double local_sum = 0.0;
    for (dtl::size_type i = 0; i < n; ++i) {
        local_sum += a[i] * b[i];
    }
    return comm.allreduce_sum_value<double>(local_sum);
}

/// axpy: y = y + alpha * x
static void axpy(double alpha, const double* x, double* y, dtl::size_type n) {
    for (dtl::size_type i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

int main(int argc, char** argv) {
    // --- DTL initialization ---
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    // --- Problem parameters ---
    const dtl::size_type global_n = 1000;  // Interior grid points
    const int max_iter = 2000;
    const double tol = 1e-10;

    if (rank == 0) {
        std::cout << "DTL Conjugate Gradient Solver\n";
        std::cout << "=============================\n";
        std::cout << "Problem: 1D Laplacian (-1, 2, -1), N = " << global_n << "\n";
        std::cout << "Ranks:   " << size << "\n";
        std::cout << "Tol:     " << std::scientific << tol << "\n\n";
    }
    comm.barrier();

    // --- Set up distributed vectors ---
    // x (solution), b (RHS), r (residual), p (search direction), Ap (matrix-vector product)
    dtl::distributed_vector<double> x_vec(global_n, ctx);
    dtl::distributed_vector<double> b_vec(global_n, ctx);
    dtl::distributed_vector<double> r_vec(global_n, ctx);
    dtl::distributed_vector<double> p_vec(global_n, ctx);
    dtl::distributed_vector<double> ap_vec(global_n, ctx);

    auto x_local  = x_vec.local_view();
    auto b_local  = b_vec.local_view();
    auto r_local  = r_vec.local_view();
    auto p_local  = p_vec.local_view();
    auto ap_local = ap_vec.local_view();

    dtl::size_type local_n = x_vec.local_size();
    dtl::index_t global_off = x_vec.global_offset();

    // Print partition info
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << rank << ": elements ["
                      << global_off << ", " << (global_off + static_cast<dtl::index_t>(local_n))
                      << ") — " << local_n << " elements\n";
        }
        comm.barrier();
    }
    if (rank == 0) std::cout << "\n";
    comm.barrier();

    // --- Initialize ---
    // Exact solution: x_exact_i = sin(pi * (i+1) / (N+1))
    // RHS: b = A * x_exact (so CG should recover x_exact)
    // Initial guess: x = 0
    constexpr double pi = std::numbers::pi;
    std::vector<double> x_exact(local_n);

    for (dtl::size_type i = 0; i < local_n; ++i) {
        dtl::index_t gi = global_off + static_cast<dtl::index_t>(i);
        x_exact[i] = std::sin(pi * static_cast<double>(gi + 1)
                              / static_cast<double>(global_n + 1));
        x_local[i] = 0.0;  // initial guess
    }

    // Compute b = A * x_exact via the stencil
    spmv_laplacian(comm, x_exact.data(), &b_local[0],
                   local_n, global_n, global_off, rank, size);

    // Initial residual: r = b - A*x = b (since x=0)
    // Initial search direction: p = r
    for (dtl::size_type i = 0; i < local_n; ++i) {
        r_local[i] = b_local[i];
        p_local[i] = r_local[i];
    }

    double r_dot_r = global_dot(comm, &r_local[0], &r_local[0], local_n);

    // --- CG iteration ---
    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p
        spmv_laplacian(comm, &p_local[0], &ap_local[0],
                       local_n, global_n, global_off, rank, size);

        // alpha = (r, r) / (p, Ap)
        double p_dot_ap = global_dot(comm, &p_local[0], &ap_local[0], local_n);
        double alpha = r_dot_r / p_dot_ap;

        // x = x + alpha * p
        axpy(alpha, &p_local[0], &x_local[0], local_n);

        // r = r - alpha * Ap
        axpy(-alpha, &ap_local[0], &r_local[0], local_n);

        // Check convergence
        double r_dot_r_new = global_dot(comm, &r_local[0], &r_local[0], local_n);
        double residual_norm = std::sqrt(r_dot_r_new);

        if (rank == 0 && (iter < 5 || (iter + 1) % 100 == 0)) {
            std::cout << "  Iter " << std::setw(4) << (iter + 1)
                      << ": ||r|| = " << std::scientific << std::setprecision(4)
                      << residual_norm << "\n";
        }

        if (residual_norm < tol) {
            ++iter;  // count this iteration
            break;
        }

        // beta = (r_new, r_new) / (r_old, r_old)
        double beta = r_dot_r_new / r_dot_r;
        r_dot_r = r_dot_r_new;

        // p = r + beta * p
        for (dtl::size_type i = 0; i < local_n; ++i) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }

    comm.barrier();

    // --- Verify against exact solution ---
    double local_max_err = 0.0;
    for (dtl::size_type i = 0; i < local_n; ++i) {
        double err = std::abs(x_local[i] - x_exact[i]);
        if (err > local_max_err) local_max_err = err;
    }
    double global_max_err = comm.allreduce_max_value<double>(local_max_err);

    if (rank == 0) {
        std::cout << "\nConverged in " << iter << " iterations\n";
        std::cout << "Max error vs exact solution: " << std::scientific
                  << std::setprecision(4) << global_max_err << "\n";
        bool ok = global_max_err < 1e-8;
        std::cout << (ok ? "SUCCESS" : "FAILURE")
                  << ": CG solver " << (ok ? "matches" : "does not match")
                  << " exact solution\n";
    }

    return 0;
}
