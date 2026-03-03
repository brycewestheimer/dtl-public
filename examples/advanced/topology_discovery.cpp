// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file topology_discovery.cpp
/// @brief Hardware topology discovery using DTL
///
/// Demonstrates:
/// - dtl::topology::local_topology() for hardware info
/// - CPU count, NUMA nodes, GPU enumeration
/// - Per-rank hardware view reporting
///
/// Run:
///   mpirun -np 2 ./topology_discovery

#include <dtl/dtl.hpp>
#include <dtl/topology/hardware.hpp>
#include <dtl/topology/topology.hpp>

#include <iostream>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    auto rank = ctx.rank();
    auto size = ctx.size();

    // Extract communicator for barrier operations
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    if (rank == 0) {
        std::cout << "DTL Hardware Topology Discovery\n";
        std::cout << "================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // Query local topology
    const auto& topo = dtl::topology::local_topology();

    // Print per-rank topology info
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << rank << " topology:\n";
            std::cout << "  Hostname:    " << topo.hostname << "\n";
            std::cout << "  Total CPUs:  " << topo.total_cpus << "\n";
            std::cout << "  Total cores: " << topo.total_cores << "\n";
            std::cout << "  Sockets:     " << topo.sockets.size() << "\n";
            std::cout << "  NUMA nodes:  " << topo.num_numa_nodes() << "\n";
            std::cout << "  GPUs:        " << topo.num_gpus() << "\n";

            if (topo.has_gpus()) {
                for (const auto& gpu : topo.gpus) {
                    std::cout << "    GPU " << gpu.id << ": " << gpu.name
                              << " (" << (gpu.memory_bytes / (1024*1024)) << " MB)\n";
                }
            }

            std::cout << "\n";
        }
        comm.barrier();
    }

    // Summary using topology query functions
    if (rank == 0) {
        std::cout << "Quick topology queries:\n";
        std::cout << "  cpu_count():       " << dtl::topology::cpu_count() << "\n";
        std::cout << "  numa_node_count(): " << dtl::topology::numa_node_count() << "\n";
        std::cout << "  gpu_count():       " << dtl::topology::gpu_count() << "\n";
        std::cout << "\nDone!\n";
    }

    return 0;
}
