// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_atomics_demo.cpp
/// @brief Demonstrates SHMEM atomic operations using DTL
/// @details Shows fetch-add, compare-swap, and atomic set operations.
///
/// @par Build:
/// cmake -DDTL_ENABLE_SHMEM=ON ..
/// make shmem_atomics_demo
///
/// @par Run:
/// oshrun -np 4 ./shmem_atomics_demo

#include <dtl/core/config.hpp>

#if DTL_ENABLE_SHMEM

#include <backends/shmem/shmem_communicator.hpp>
#include <backends/shmem/shmem_memory_space.hpp>
#include <backends/shmem/shmem_memory_window_impl.hpp>

#include <cstdio>
#include <vector>

int main() {
    // Initialize SHMEM
    dtl::shmem::scoped_shmem_environment env;

    auto& comm = dtl::shmem::world_communicator();
    const auto rank = comm.rank();
    const auto size = comm.size();

    std::printf("PE %d of %d: SHMEM Atomics Demo\n", rank, size);

    // =========================================================================
    // Example 1: Atomic Fetch-Add (Global Counter)
    // =========================================================================
    std::printf("\n=== Example 1: Atomic Fetch-Add (Global Counter) ===\n");
    {
        // Create window for a single counter on PE 0
        auto window_result = dtl::shmem::make_shmem_window(sizeof(int));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        int* counter = static_cast<int*>(window.base());
        *counter = 0;

        window.barrier();

        // All PEs atomically increment the counter on PE 0
        int increment = 1;
        int old_value = 0;

        auto result = window.fetch_and_op(
            &increment, &old_value, sizeof(int),
            0,  // target PE
            0,  // offset
            dtl::rma_reduce_op::sum);

        if (result) {
            std::printf("PE %d: fetch_add returned old value = %d\n", rank, old_value);
        }

        window.flush_all();
        window.barrier();

        // PE 0 reports final count
        if (rank == 0) {
            std::printf("PE 0: Final counter value = %d (expected %d)\n",
                        *counter, size);
        }
    }

    // =========================================================================
    // Example 2: Compare-and-Swap (Mutex-like Pattern)
    // =========================================================================
    std::printf("\n=== Example 2: Compare-and-Swap ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(int));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        int* lock = static_cast<int*>(window.base());
        *lock = -1;  // -1 means unlocked

        window.barrier();

        // All PEs try to acquire the lock on PE 0
        int compare = -1;       // Expect unlocked
        int new_value = rank;   // Set to our rank
        int old_value = 0;

        auto result = window.compare_and_swap(
            &new_value, &compare, &old_value, sizeof(int),
            0,  // target PE
            0); // offset

        if (result) {
            if (old_value == -1) {
                std::printf("PE %d: WON the lock (was unlocked)\n", rank);
            } else {
                std::printf("PE %d: Lost the lock (already held by PE %d)\n",
                            rank, old_value);
            }
        }

        window.barrier();

        // Winner releases the lock
        if (rank == 0) {
            std::printf("PE 0: Lock is now held by PE %d\n", *lock);
        }
    }

    // =========================================================================
    // Example 3: Atomic Swap (Exchange Values)
    // =========================================================================
    std::printf("\n=== Example 3: Atomic Swap ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(long));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        long* value = static_cast<long*>(window.base());
        *value = 0;

        window.barrier();

        // Each PE swaps in its rank * 1000, getting back previous value
        long my_value = static_cast<long>(rank * 1000);
        long old_value = 0;

        auto result = window.fetch_and_op(
            &my_value, &old_value, sizeof(long),
            0,  // target PE
            0,  // offset
            dtl::rma_reduce_op::replace);  // swap operation

        if (result) {
            std::printf("PE %d: Swapped in %ld, got back %ld\n",
                        rank, my_value, old_value);
        }

        window.barrier();

        if (rank == 0) {
            std::printf("PE 0: Final value = %ld\n", *value);
        }
    }

    // =========================================================================
    // Example 4: Atomic Fetch (Read-Only)
    // =========================================================================
    std::printf("\n=== Example 4: Atomic Fetch ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(int) * 4);
        if (!window_result) return 1;
        auto& window = *window_result.value();

        int* data = static_cast<int*>(window.base());

        // Each PE sets its value
        data[rank % 4] = rank * 10;

        window.barrier();

        // All PEs atomically read from PE 0
        for (int target_pe = 0; target_pe < std::min(4, size); ++target_pe) {
            int dummy = 0;  // No-op doesn't need origin value
            int fetched = 0;

            window.fetch_and_op(
                &dummy, &fetched, sizeof(int),
                target_pe,                    // target PE
                0,                            // offset
                dtl::rma_reduce_op::no_op);   // just fetch

            std::printf("PE %d: Fetched value %d from PE %d\n",
                        rank, fetched, target_pe);
        }

        window.barrier();
    }

    // =========================================================================
    // Example 5: Distributed Sum Reduction (Using Atomics)
    // =========================================================================
    std::printf("\n=== Example 5: Distributed Sum Reduction ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(long));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        long* sum = static_cast<long*>(window.base());
        *sum = 0;

        window.barrier();

        // Each PE contributes rank + 1 to the sum on PE 0
        long contribution = static_cast<long>(rank + 1);
        long old_sum = 0;

        window.fetch_and_op(
            &contribution, &old_sum, sizeof(long),
            0, 0, dtl::rma_reduce_op::sum);

        window.flush_all();
        window.barrier();

        // PE 0 reports the total
        if (rank == 0) {
            // Expected: 1 + 2 + ... + size = size * (size + 1) / 2
            long expected = static_cast<long>(size) * (size + 1) / 2;
            std::printf("PE 0: Sum = %ld (expected %ld) - %s\n",
                        *sum, expected,
                        (*sum == expected) ? "CORRECT" : "INCORRECT");
        }
    }

    // =========================================================================
    // Example 6: Concurrent Updates with Long Type
    // =========================================================================
    std::printf("\n=== Example 6: Concurrent Long Updates ===\n");
    {
        auto window_result = dtl::shmem::make_shmem_window(sizeof(long));
        if (!window_result) return 1;
        auto& window = *window_result.value();

        long* counter = static_cast<long*>(window.base());
        *counter = 0;

        window.barrier();

        // Each PE does 10 atomic increments
        const int iterations = 10;
        long increment = 1;
        long old_value = 0;

        for (int i = 0; i < iterations; ++i) {
            window.fetch_and_op(
                &increment, &old_value, sizeof(long),
                0, 0, dtl::rma_reduce_op::sum);
        }

        window.flush_all();
        window.barrier();

        if (rank == 0) {
            long expected = static_cast<long>(size * iterations);
            std::printf("PE 0: Counter = %ld (expected %ld) - %s\n",
                        *counter, expected,
                        (*counter == expected) ? "CORRECT" : "INCORRECT");
        }
    }

    std::printf("\nPE %d: Atomics demo complete\n", rank);
    return 0;
}

#else  // !DTL_ENABLE_SHMEM

#include <cstdio>

int main() {
    std::printf("SHMEM support not enabled. Build with -DDTL_ENABLE_SHMEM=ON\n");
    return 0;
}

#endif  // DTL_ENABLE_SHMEM
