// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_put_get_demo.cpp
/// @brief Demonstrates SHMEM put/get operations using DTL
/// @details Shows one-sided communication with symmetric memory.
///
/// @par Build:
/// cmake -DDTL_ENABLE_SHMEM=ON ..
/// make shmem_put_get_demo
///
/// @par Run:
/// oshrun -np 2 ./shmem_put_get_demo
/// oshrun -np 4 ./shmem_put_get_demo

#include <dtl/core/config.hpp>

#if DTL_ENABLE_SHMEM

#include <backends/shmem/shmem_communicator.hpp>
#include <backends/shmem/shmem_memory_space.hpp>
#include <backends/shmem/shmem_memory_window_impl.hpp>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

int main() {
    // Initialize SHMEM
    dtl::shmem::scoped_shmem_environment env;

    auto& comm = dtl::shmem::world_communicator();
    const auto rank = comm.rank();
    const auto size = comm.size();

    std::printf("PE %d of %d: SHMEM Put/Get Demo\n", rank, size);

    if (size < 2) {
        std::printf("This demo requires at least 2 PEs\n");
        return 1;
    }

    // =========================================================================
    // Example 1: Simple Put/Get with Memory Window
    // =========================================================================
    std::printf("\n=== Example 1: Simple Put/Get ===\n");
    {
        constexpr dtl::size_type buffer_size = sizeof(int) * 10;

        // Create SHMEM memory window (allocates symmetric memory)
        auto window_result = dtl::shmem::make_shmem_window(buffer_size);
        if (!window_result) {
            std::printf("PE %d: Failed to create window\n", rank);
            return 1;
        }
        auto& window = *window_result.value();

        // Get typed access to the buffer
        int* data = static_cast<int*>(window.base());

        // Initialize local data
        for (int i = 0; i < 10; ++i) {
            data[i] = rank * 100 + i;
        }

        // Synchronize before communication
        window.barrier();

        // PE 0 puts data to PE 1
        if (rank == 0) {
            std::printf("PE 0: Putting data to PE 1...\n");

            // Prepare data to send
            std::vector<int> send_data(10);
            std::iota(send_data.begin(), send_data.end(), 1);  // 1, 2, 3, ..., 10

            // Put to PE 1's window
            window.put(send_data.data(), sizeof(int) * 10, 1, 0);

            // Ensure operation completes
            window.flush_all();
        }

        // Synchronize
        window.barrier();

        // PE 1 verifies received data
        if (rank == 1) {
            std::printf("PE 1: Received data: ");
            for (int i = 0; i < 10; ++i) {
                std::printf("%d ", data[i]);
            }
            std::printf("\n");

            // Verify
            bool correct = true;
            for (int i = 0; i < 10; ++i) {
                if (data[i] != i + 1) correct = false;
            }
            std::printf("PE 1: Data %s\n", correct ? "CORRECT" : "INCORRECT");
        }
    }

    // =========================================================================
    // Example 2: Get Operation (Read from Remote PE)
    // =========================================================================
    std::printf("\n=== Example 2: Get Operation ===\n");
    {
        constexpr dtl::size_type buffer_size = sizeof(long) * 4;

        auto window_result = dtl::shmem::make_shmem_window(buffer_size);
        if (!window_result) return 1;
        auto& window = *window_result.value();

        long* data = static_cast<long*>(window.base());

        // Each PE initializes with unique values
        for (int i = 0; i < 4; ++i) {
            data[i] = static_cast<long>(rank * 1000 + i);
        }

        window.barrier();

        // PE 0 gets data from PE 1
        if (rank == 0) {
            std::vector<long> local_buffer(4);

            std::printf("PE 0: Getting data from PE 1...\n");
            window.get(local_buffer.data(), sizeof(long) * 4, 1, 0);

            std::printf("PE 0: Got data from PE 1: ");
            for (int i = 0; i < 4; ++i) {
                std::printf("%ld ", local_buffer[i]);
            }
            std::printf("\n");

            // Verify (PE 1's data should be 1000, 1001, 1002, 1003)
            bool correct = true;
            for (int i = 0; i < 4; ++i) {
                if (local_buffer[i] != 1000 + i) correct = false;
            }
            std::printf("PE 0: Data %s\n", correct ? "CORRECT" : "INCORRECT");
        }

        window.barrier();
    }

    // =========================================================================
    // Example 3: Ring Communication Pattern
    // =========================================================================
    std::printf("\n=== Example 3: Ring Communication ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(int));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        int* value = static_cast<int*>(window.base());
        *value = 0;

        window.barrier();

        // Each PE sends its rank to the next PE in a ring
        int next_pe = (rank + 1) % size;
        int my_rank = rank;

        std::printf("PE %d: Sending value %d to PE %d\n", rank, my_rank, next_pe);

        window.put(&my_rank, sizeof(int), next_pe, 0);
        window.flush_all();
        window.barrier();

        // Check received value (should be from previous PE)
        int expected = (rank == 0) ? size - 1 : rank - 1;
        std::printf("PE %d: Received %d (expected %d) - %s\n",
                    rank, *value, expected,
                    (*value == expected) ? "CORRECT" : "INCORRECT");
    }

    // =========================================================================
    // Example 4: Non-Blocking Put
    // =========================================================================
    std::printf("\n=== Example 4: Non-Blocking Put ===\n");
    {
        constexpr dtl::size_type count = 100;
        constexpr dtl::size_type buffer_size = sizeof(double) * count;

        auto window_result = dtl::shmem::make_shmem_window(buffer_size);
        if (!window_result) return 1;
        auto& window = *window_result.value();

        double* data = static_cast<double*>(window.base());

        // Initialize
        for (dtl::size_type i = 0; i < count; ++i) {
            data[i] = 0.0;
        }

        window.barrier();

        if (rank == 0) {
            std::printf("PE 0: Starting non-blocking put of %zu doubles...\n", count);

            std::vector<double> send_data(count);
            for (dtl::size_type i = 0; i < count; ++i) {
                send_data[i] = static_cast<double>(i) * 1.5;
            }

            // Non-blocking put
            dtl::memory_window_impl::rma_request_handle request;
            window.async_put(send_data.data(), buffer_size, 1, 0, request);

            std::printf("PE 0: Put initiated, doing other work...\n");

            // Simulate other work
            volatile double x = 0.0;
            for (int i = 0; i < 1000; ++i) {
                x += static_cast<double>(i);
            }

            // Wait for completion
            window.wait_async(request);
            std::printf("PE 0: Put completed\n");
        }

        window.barrier();

        if (rank == 1) {
            // Verify received data
            double sum = 0.0;
            for (dtl::size_type i = 0; i < count; ++i) {
                sum += data[i];
            }
            // Expected sum: sum of (i * 1.5) for i = 0..99 = 1.5 * (99 * 100 / 2) = 7425
            std::printf("PE 1: Received sum = %.1f (expected 7425.0)\n", sum);
        }
    }

    std::printf("\nPE %d: Demo complete\n", rank);
    return 0;
}

#else  // !DTL_ENABLE_SHMEM

#include <cstdio>

int main() {
    std::printf("SHMEM support not enabled. Build with -DDTL_ENABLE_SHMEM=ON\n");
    return 0;
}

#endif  // DTL_ENABLE_SHMEM
