// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file collective_ops_dtl.cpp
/// @brief Collective operations using DTL environment and context
///
/// Demonstrates:
/// - dtl::environment + make_world_context() initialization
/// - Broadcast via comm.broadcast()
/// - Allreduce via comm.allreduce_sum_value<T>()
/// - Gather via comm.gather()
/// - Scatter via comm.scatter()
///
/// Run:
///   mpirun -np 4 ./collective_ops_dtl

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Collective Operations Example\n";
        std::cout << "==================================\n";
        std::cout << "Running with " << size << " ranks\n\n";
    }
    comm.barrier();

    // 1. Broadcast: rank 0 sends value 42 to all
    {
        if (rank == 0) std::cout << "1. Broadcast:\n";
        comm.barrier();

        int value = (rank == 0) ? 42 : 0;
        std::cout << "  Rank " << rank << " before: " << value << "\n";

        comm.broadcast(&value, 1, 0);

        std::cout << "  Rank " << rank << " after:  " << value << "\n";
        comm.barrier();
    }

    // 2. Allreduce: sum of (rank + 1) across all ranks
    {
        if (rank == 0) std::cout << "\n2. Allreduce (sum):\n";
        comm.barrier();

        int local_val = rank + 1;
        int global_sum = comm.allreduce_sum_value<int>(local_val);

        int expected = size * (size + 1) / 2;
        std::cout << "  Rank " << rank << ": local=" << local_val
                  << ", global_sum=" << global_sum << "\n";
        if (rank == 0) {
            std::cout << "  Expected: " << expected
                      << " -> " << (global_sum == expected ? "OK" : "FAIL") << "\n";
        }
        comm.barrier();
    }

    // 3. Gather: each rank sends rank*10 to root
    {
        if (rank == 0) std::cout << "\n3. Gather:\n";
        comm.barrier();

        int send_val = rank * 10;
        std::vector<int> recv_buf(rank == 0 ? size : 0);

        comm.gather(&send_val, recv_buf.data(), 1, 0);

        if (rank == 0) {
            std::cout << "  Root gathered: [";
            for (int i = 0; i < size; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << recv_buf[i];
            }
            std::cout << "]\n";
        }
        comm.barrier();
    }

    // 4. Scatter: root distributes values to each rank
    {
        if (rank == 0) std::cout << "\n4. Scatter:\n";
        comm.barrier();

        std::vector<int> send_buf;
        if (rank == 0) {
            send_buf.resize(size);
            for (int i = 0; i < size; ++i) {
                send_buf[i] = (i + 1) * 100;
            }
            std::cout << "  Root scattering: [";
            for (int i = 0; i < size; ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << send_buf[i];
            }
            std::cout << "]\n";
        }

        int recv_val = 0;
        comm.scatter(send_buf.data(), &recv_val, 1, 0);

        std::cout << "  Rank " << rank << " received: " << recv_val << "\n";
        comm.barrier();
    }

    // 5. Allgather: each rank shares its value with all
    {
        if (rank == 0) std::cout << "\n5. Allgather:\n";
        comm.barrier();

        int my_val = rank;
        std::vector<int> all_vals(size);

        comm.allgather(&my_val, all_vals.data(), 1);

        std::cout << "  Rank " << rank << ": [";
        for (int i = 0; i < size; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << all_vals[i];
        }
        std::cout << "]\n";
        comm.barrier();
    }

    if (rank == 0) std::cout << "\nDone!\n";

    return 0;
}
