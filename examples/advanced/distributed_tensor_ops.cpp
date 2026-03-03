// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_tensor_ops.cpp
/// @brief Distributed tensor ND operations using DTL
///
/// Demonstrates:
/// - distributed_tensor<T, 2> with 2D ND extents
/// - Partition along dimension 0
/// - Local view fill and Frobenius norm computation
/// - Cross-rank allreduce for global norm
///
/// Run:
///   mpirun -np 4 ./distributed_tensor_ops

#include <dtl/dtl.hpp>

#include <iostream>
#include <cmath>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    auto rank = ctx.rank();
    auto size = ctx.size();

    // Extract communicator for barrier and allreduce operations
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    if (rank == 0) {
        std::cout << "DTL Distributed Tensor ND Operations\n";
        std::cout << "======================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // Create a 2D tensor (100 rows x 64 cols), partitioned along dim 0
    const dtl::size_type nrows = 100;
    const dtl::size_type ncols = 64;
    dtl::nd_extent<2> global_shape{nrows, ncols};

    dtl::distributed_tensor<double, 2> tensor(global_shape, ctx);

    // Report partition info
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            auto local_rows = tensor.local_size() / ncols;
            std::cout << "Rank " << rank << ": "
                      << local_rows << " rows x " << ncols << " cols"
                      << " (" << tensor.local_size() << " elements)\n";
        }
        comm.barrier();
    }

    // Fill tensor: each element = row * 100 + col
    // global_offset() returns nd_index<2>; for dim-0 partition, [0] is the row offset
    auto local = tensor.local_view();
    auto g_offset = tensor.global_offset();
    auto row_offset = g_offset[0];  // starting global row for this rank
    for (dtl::size_type i = 0; i < tensor.local_size(); ++i) {
        auto local_row = static_cast<dtl::index_t>(i / ncols);
        auto col = static_cast<dtl::index_t>(i % ncols);
        auto global_row = row_offset + local_row;
        local[i] = static_cast<double>(global_row * 100 + col);
    }

    // Compute local Frobenius norm contribution: sum of squares
    double local_sum_sq = 0.0;
    for (dtl::size_type i = 0; i < tensor.local_size(); ++i) {
        local_sum_sq += local[i] * local[i];
    }

    // Global sum of squares
    double global_sum_sq = comm.allreduce_sum_value<double>(local_sum_sq);

    double frobenius_norm = std::sqrt(global_sum_sq);

    // Report
    if (rank == 0) {
        std::cout << "\nTensor shape: " << nrows << " x " << ncols << "\n";
        std::cout << "Total elements: " << (nrows * ncols) << "\n";
        std::cout << "Frobenius norm: " << frobenius_norm << "\n";

        // Compute expected norm analytically
        // sum_{r=0}^{99} sum_{c=0}^{63} (100*r + c)^2
        double expected_sq = 0.0;
        for (dtl::size_type r = 0; r < nrows; ++r) {
            for (dtl::size_type c = 0; c < ncols; ++c) {
                double val = static_cast<double>(r * 100 + c);
                expected_sq += val * val;
            }
        }
        double expected_norm = std::sqrt(expected_sq);

        std::cout << "Expected norm: " << expected_norm << "\n";
        bool ok = std::abs(frobenius_norm - expected_norm) < 1e-6;
        std::cout << (ok ? "SUCCESS" : "FAILURE") << "\n";
    }

    return 0;
}
