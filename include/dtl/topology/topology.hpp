// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file topology.hpp
/// @brief Master include for DTL topology support
/// @details Provides single-header access to hardware discovery, proximity
///          matrices, affinity control, and network topology.
/// @since 0.1.0

#pragma once

// Hardware discovery
#include <dtl/topology/hardware.hpp>

// Platform-specific implementations
#include <dtl/topology/detail/linux.hpp>
#include <dtl/topology/detail/cuda.hpp>

// Proximity matrices
#include <dtl/topology/proximity.hpp>

// CPU affinity
#include <dtl/topology/affinity.hpp>

// Network topology
#include <dtl/topology/network.hpp>

namespace dtl::topology {

// ============================================================================
// Hardware Discovery Implementation
// ============================================================================

/// @brief Discover local hardware topology
inline hardware_topology discover_local() {
    hardware_topology topo;

#if defined(__linux__)
    topo = detail::discover_linux();
#else
    // Fallback for unsupported platforms
    topo.hostname = "unknown";
    topo.total_cpus = 1;
    topo.total_cores = 1;

    numa_node node;
    node.id = 0;
    node.cpu_ids = {0};
    topo.numa_nodes.push_back(node);
#endif

    // Add GPU information
    detail::discover_cuda_gpus(topo);

    return topo;
}

/// @brief Get cached local topology
inline const hardware_topology& local_topology() {
    static hardware_topology topo = discover_local();
    return topo;
}

/// @brief Refresh cached local topology
/// @details Rebuilds the same static variable used by local_topology()
inline const hardware_topology& refresh_local_topology() {
    // Use a lambda to get a mutable reference to the static in local_topology().
    // C++ guarantees a function-local static has exactly one instance.
    auto& topo = const_cast<hardware_topology&>(local_topology());
    topo = discover_local();
    return topo;
}

// ============================================================================
// Query Function Implementations
// ============================================================================

inline std::uint32_t cpu_count() {
    return local_topology().total_cpus;
}

inline std::uint32_t numa_node_count() {
    return static_cast<std::uint32_t>(local_topology().numa_nodes.size());
}

inline std::uint32_t gpu_count() {
    return static_cast<std::uint32_t>(local_topology().gpus.size());
}

inline std::vector<std::uint32_t> cpus_on_numa_node(std::uint32_t node) {
    const auto& topo = local_topology();
    for (const auto& n : topo.numa_nodes) {
        if (n.id == node) {
            return n.cpu_ids;
        }
    }
    return {};
}

inline std::uint32_t numa_node_for_cpu(std::uint32_t cpu) {
    const auto& topo = local_topology();
    for (const auto& node : topo.numa_nodes) {
        for (auto c : node.cpu_ids) {
            if (c == cpu) return node.id;
        }
    }
    return 0;
}

inline std::vector<std::uint32_t> gpus_on_numa_node(std::uint32_t node) {
    const auto& topo = local_topology();
    for (const auto& n : topo.numa_nodes) {
        if (n.id == node) {
            return n.gpu_ids;
        }
    }
    return {};
}

// ============================================================================
// Topology Module Summary
// ============================================================================
//
// The topology module provides hardware and network topology awareness for DTL:
//
// 1. **Hardware Discovery**
//    - CPU topology (sockets, cores, threads)
//    - NUMA topology (nodes, memory, CPU affinity)
//    - GPU topology (devices, memory, NUMA affinity)
//
// 2. **Proximity Matrices**
//    - CPU proximity (NUMA-aware distances)
//    - GPU proximity (P2P capability-based)
//    - Rank proximity (network-based)
//
// 3. **Affinity Control**
//    - Get/set CPU affinity
//    - NUMA binding
//    - RAII scoped affinity
//
// 4. **Network Topology**
//    - Rank-to-host mapping
//    - Co-location detection
//    - Proximity-based rank ordering
//
// ============================================================================
// Quick Start
// ============================================================================
//
// 1. Query hardware topology:
//
//    const auto& topo = dtl::topology::local_topology();
//    std::cout << "CPUs: " << topo.total_cpus << "\n";
//    std::cout << "NUMA nodes: " << topo.num_numa_nodes() << "\n";
//    std::cout << "GPUs: " << topo.num_gpus() << "\n";
//
// 2. Build proximity matrices:
//
//    auto cpu_prox = dtl::topology::build_cpu_proximity(topo);
//    auto nearest = cpu_prox.nearest(0);  // Nearest CPU to CPU 0
//
// 3. Control affinity:
//
//    // Bind to specific CPU
//    dtl::topology::bind_to_cpu(0);
//
//    // Bind to NUMA node
//    dtl::topology::bind_to_numa_node(0);
//
//    // RAII scoped binding
//    {
//        dtl::topology::scoped_affinity guard(cpu_set(0, 3));
//        // Work on CPUs 0-3...
//    }  // Original affinity restored
//
// 4. Network topology (with MPI):
//
//    auto host_map = dtl::topology::gather_hostnames(my_rank, num_ranks);
//
//    if (host_map.same_node(my_rank, other_rank)) {
//        // Use shared memory
//    } else {
//        // Use network
//    }
//
// ============================================================================

}  // namespace dtl::topology
