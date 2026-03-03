// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_map_example.cpp
/// @brief Demonstrates distributed_map usage for key-value storage
/// @details Shows how DTL's distributed_map provides a distributed associative
///          container (analogous to std::unordered_map) where key-value pairs
///          are partitioned across ranks based on key hashes.
///
///          Key concepts demonstrated:
///          - Creating a distributed_map with context (communicator)
///          - Inserting key-value pairs (local and potentially remote keys)
///          - Explicit remote mutation APIs for non-local ownership
///          - Looking up values by key
///          - Checking key ownership with is_local() and owner()
///          - Iterating over local entries
///          - Erasing entries
///          - Querying sizes and statistics
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./distributed_map_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./distributed_map_example

#include <dtl/dtl.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Map Example\n";
        std::cout << "===========================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Create a distributed map
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 1. Creating a distributed_map<int, double> ---\n";
    }
    comm.barrier();

    dtl::distributed_map<int, double> map(ctx);

    if (rank == 0) {
        std::cout << "Empty map created. Local size: " << map.local_size() << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 2. Key ownership: which rank owns which key?
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 2. Key Ownership (hash-based distribution) ---\n";
        for (int k = 0; k < 10; ++k) {
            std::cout << "Key " << k << " -> owner rank " << map.owner(k)
                      << (map.is_local(k) ? " (local)" : " (remote)") << "\n";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Insert local key-value pairs
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 3. Inserting Key-Value Pairs ---\n";
    }
    comm.barrier();

    // Each rank inserts keys that hash to itself
    int inserted_count = 0;
    for (int k = 0; k < 100; ++k) {
        if (map.is_local(k)) {
            map.insert(k, static_cast<double>(k) * 1.5);
            inserted_count++;
        }
    }

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << " inserted " << inserted_count
                      << " local entries, local_size=" << map.local_size() << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 3b. Explicit remote mutation path (Phase 06 owner-aware contract)
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 3b. Explicit Remote Mutation APIs ---\n";
    }
    comm.barrier();

    // Rank 0 enqueues one remote-owned key explicitly.
    // Other ranks perform the same collective flush and receive/apply as needed.
    if (rank == 0) {
        int remote_key = -1;
        for (int k = 0; k < 1000; ++k) {
            if (!map.is_local(k)) {
                remote_key = k;
                break;
            }
        }

        if (remote_key >= 0) {
            auto remote_res = map.insert_or_assign_remote(remote_key, 4242.0);
            if (remote_res.has_value()) {
                std::cout << "Rank 0 queued explicit remote upsert for key "
                          << remote_key << " (owner=" << map.owner(remote_key) << ")\n";
            } else {
                std::cout << "Rank 0 remote upsert failed: " << remote_res.error().message() << "\n";
            }
        }
    }

    auto flush_remote = map.flush_pending_with_comm(comm);
    if (!flush_remote.has_value()) {
        std::cout << "Rank " << rank << " flush_pending_with_comm failed: "
                  << flush_remote.error().message() << "\n";
    }

    if (rank == 0) {
        if (map.has_legacy_ownership_diagnostic()) {
            auto diag = map.legacy_ownership_diagnostic();
            if (diag.has_value()) {
                std::cout << "Migration diagnostic: " << *diag << "\n";
            }
            map.clear_legacy_ownership_diagnostic();
        } else {
            std::cout << "No legacy implicit-remote diagnostics observed in this run.\n";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. Looking up values
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 4. Looking Up Values ---\n";
    }
    comm.barrier();

    // Each rank checks a few keys it owns
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << " local lookups: ";
            int shown = 0;
            for (int k = 0; k < 100 && shown < 5; ++k) {
                if (map.is_local(k) && map.contains(k)) {
                    auto ref = map[k];
                    auto val = ref.get();
                    if (val) {
                        std::cout << "[" << k << "]=" << val.value() << " ";
                        shown++;
                    }
                }
            }
            std::cout << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 5. Iterating over local entries
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 5. Iterating Local Entries ---\n";
    }
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << " local entries (first 5): ";
            int count = 0;
            for (auto it = map.begin(); it != map.end() && count < 5; ++it, ++count) {
                std::cout << "{" << it->first << ": " << it->second << "} ";
            }
            std::cout << "(total: " << map.local_size() << ")\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 6. Erase some entries
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 6. Erasing Entries ---\n";
    }
    comm.barrier();

    dtl::size_type before_size = map.local_size();

    // Erase even-numbered local keys
    int erased_count = 0;
    for (int k = 0; k < 100; k += 2) {
        if (map.is_local(k)) {
            auto res = map.erase(k);
            if (res && res.value() > 0) {
                erased_count++;
            }
        }
    }

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": erased " << erased_count
                      << " entries, size " << before_size
                      << " -> " << map.local_size() << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 7. Using contains() and count()
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 7. Membership Queries ---\n";
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "Checking keys 0-9 on rank 0:\n";
        for (int k = 0; k < 10; ++k) {
            if (map.is_local(k)) {
                std::cout << "  Key " << k << ": contains=" << std::boolalpha
                          << map.contains(k) << ", count=" << map.count(k) << "\n";
            }
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 8. Using insert_or_assign for upserts
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 8. Insert-or-Assign (Upsert) ---\n";
    }
    comm.barrier();

    // Insert or update: set all local keys 0-9 to 999.0
    for (int k = 0; k < 10; ++k) {
        if (map.is_local(k)) {
            map.insert_or_assign(k, 999.0);
        }
    }

    if (rank == 0) {
        std::cout << "After insert_or_assign(k, 999.0) for keys 0-9:\n";
        for (int k = 0; k < 10; ++k) {
            if (map.is_local(k)) {
                auto ref = map[k];
                auto val = ref.get();
                if (val) {
                    std::cout << "  [" << k << "] = " << val.value() << "\n";
                }
            }
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 9. Global size using communicator
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 9. Global Statistics ---\n";
    }
    comm.barrier();

    auto global_size = map.global_size_with_comm(comm);

    if (rank == 0) {
        std::cout << "Global map size (all ranks): " << global_size << "\n";
        std::cout << "Load factor: " << map.load_factor() << "\n";
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Distributed map example completed!\n";
    }

    return 0;
}
