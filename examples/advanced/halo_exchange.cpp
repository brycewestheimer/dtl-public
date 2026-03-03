// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file halo_exchange.cpp
/// @brief Stencil computation with halo/ghost cell exchange
/// @details Demonstrates the common HPC pattern of halo exchange for
///          stencil computations on distributed data. Each rank needs
///          boundary values from neighbors to compute its local region.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./halo_exchange
///
/// Expected output (4 ranks):
///   DTL Halo Exchange Example
///   =========================
///
///   Configuration:
///     Global size: 100
///     Halo width: 1
///     Number of ranks: 4
///
///   --- Initial State ---
///   Each cell initialized with its global index.
///
///   --- After Halo Exchange ---
///   Rank 0: local [0-24], halo: left=N/A, right=25
///   Rank 1: local [25-49], halo: left=24, right=50
///   ...
///
///   --- 3-Point Stencil Computation ---
///   Computing new[i] = 0.25*old[i-1] + 0.5*old[i] + 0.25*old[i+1]
///
///   --- Verification ---
///   Interior points computed correctly using neighbor values.
///
///   SUCCESS: Halo exchange example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    // Extract communicator for explicit P2P and collective MPI operations
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    const dtl::size_type global_size = 100;
    const int halo_width = 1;

    if (my_rank == 0) {
        std::cout << "DTL Halo Exchange Example\n";
        std::cout << "=========================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Global size: " << global_size << "\n";
        std::cout << "  Halo width: " << halo_width << "\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n\n";
    }

    // Create distributed vector for main data
    dtl::distributed_vector<double> vec(global_size, ctx);
    auto local = vec.local_view();

    // Initialize with global indices
    dtl::index_t offset = vec.global_offset();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<double>(offset + static_cast<dtl::index_t>(i));
    }

    // Create halo buffers
    // left_halo: receives from rank-1
    // right_halo: receives from rank+1
    std::vector<double> left_halo(static_cast<size_t>(halo_width), 0.0);
    std::vector<double> right_halo(static_cast<size_t>(halo_width), 0.0);

    // Determine neighbors
    dtl::rank_t left_neighbor = (my_rank > 0) ? my_rank - 1 : -1;
    dtl::rank_t right_neighbor = (my_rank < num_ranks - 1) ? my_rank + 1 : -1;

    comm.barrier();

    if (my_rank == 0) {
        std::cout << "--- Initial State ---\n";
        std::cout << "Each cell initialized with its global index.\n\n";
    }

    // =========================================================================
    // Halo Exchange
    // =========================================================================
    std::vector<dtl::request_handle> requests;

    // Send my leftmost values to left neighbor, receive their rightmost
    if (left_neighbor >= 0) {
        requests.push_back(comm.isend(&local[0],
            static_cast<dtl::size_type>(halo_width) * sizeof(double),
            left_neighbor, 0));
        requests.push_back(comm.irecv(left_halo.data(),
            static_cast<dtl::size_type>(halo_width) * sizeof(double),
            left_neighbor, 1));
    }

    // Send my rightmost values to right neighbor, receive their leftmost
    if (right_neighbor >= 0) {
        dtl::size_type send_start = local.size() - static_cast<dtl::size_type>(halo_width);
        requests.push_back(comm.isend(&local[send_start],
            static_cast<dtl::size_type>(halo_width) * sizeof(double),
            right_neighbor, 1));
        requests.push_back(comm.irecv(right_halo.data(),
            static_cast<dtl::size_type>(halo_width) * sizeof(double),
            right_neighbor, 0));
    }

    // Wait for all exchanges to complete
    for (auto& req : requests) {
        comm.wait(req);
    }

    comm.barrier();

    if (my_rank == 0) {
        std::cout << "--- After Halo Exchange ---\n";
    }

    comm.barrier();

    // Print halo info for each rank
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << ": local [" << offset << "-"
                      << (offset + static_cast<dtl::index_t>(local.size()) - 1) << "], halo: ";

            if (left_neighbor >= 0) {
                std::cout << "left=" << left_halo[0];
            } else {
                std::cout << "left=N/A";
            }

            std::cout << ", ";

            if (right_neighbor >= 0) {
                std::cout << "right=" << right_halo[0];
            } else {
                std::cout << "right=N/A";
            }
            std::cout << "\n";
        }
        comm.barrier();
    }

    // =========================================================================
    // 3-Point Stencil Computation
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- 3-Point Stencil Computation ---\n";
        std::cout << "Computing new[i] = 0.25*old[i-1] + 0.5*old[i] + 0.25*old[i+1]\n\n";
    }

    comm.barrier();

    // Create output buffer
    std::vector<double> result(local.size());

    // Apply stencil
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        double left_val, center_val, right_val;

        center_val = local[i];

        // Get left value
        if (i == 0) {
            if (left_neighbor >= 0) {
                left_val = left_halo[0];  // From halo
            } else {
                left_val = local[i];  // Boundary: reflect
            }
        } else {
            left_val = local[i - 1];
        }

        // Get right value
        if (i == local.size() - 1) {
            if (right_neighbor >= 0) {
                right_val = right_halo[0];  // From halo
            } else {
                right_val = local[i];  // Boundary: reflect
            }
        } else {
            right_val = local[i + 1];
        }

        result[i] = 0.25 * left_val + 0.5 * center_val + 0.25 * right_val;
    }

    // =========================================================================
    // Verification
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Verification ---\n";
    }

    comm.barrier();

    // Check an interior point
    bool correct = true;
    if (local.size() > 2) {
        // Check index 1 (has both neighbors locally)
        double expected = 0.25 * local[0] + 0.5 * local[1] + 0.25 * local[2];
        if (std::abs(result[1] - expected) > 1e-10) {
            correct = false;
        }
    }

    int local_correct = correct ? 1 : 0;
    int global_correct = comm.allreduce_min_value<int>(local_correct);
    correct = (global_correct == 1);

    if (my_rank == 0) {
        std::cout << "Interior points computed correctly using neighbor values: "
                  << std::boolalpha << correct << "\n\n";
        std::cout << "SUCCESS: Halo exchange example completed!\n";
    }

    return correct ? 0 : 1;
}
