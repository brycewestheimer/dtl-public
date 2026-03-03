// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hardware.hpp
/// @brief Hardware topology discovery
/// @details Provides types and functions for discovering local hardware
///          topology including CPUs, NUMA nodes, and GPUs.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dtl::topology {

// ============================================================================
// Hardware Types
// ============================================================================

/// @brief CPU socket information
struct cpu_socket {
    std::uint32_t id = 0;               ///< Socket ID
    std::uint32_t num_cores = 0;        ///< Physical cores
    std::uint32_t num_threads = 0;      ///< Logical threads (with SMT)
    std::vector<std::uint32_t> numa_nodes;  ///< Associated NUMA nodes
};

/// @brief NUMA node information
struct numa_node {
    std::uint32_t id = 0;               ///< NUMA node ID
    std::uint64_t memory_bytes = 0;     ///< Total memory in bytes
    std::vector<std::uint32_t> cpu_ids; ///< CPUs on this node
    std::vector<std::uint32_t> gpu_ids; ///< GPUs near this node
};

/// @brief GPU device information
struct gpu_device {
    std::uint32_t id = 0;               ///< GPU device ID
    std::string name;                   ///< Device name
    std::uint64_t memory_bytes = 0;     ///< Device memory in bytes
    std::optional<std::uint32_t> numa_node;  ///< Nearest NUMA node (if known)
    std::uint32_t compute_capability_major = 0;  ///< CUDA compute capability major
    std::uint32_t compute_capability_minor = 0;  ///< CUDA compute capability minor
};

/// @brief Complete hardware topology
struct hardware_topology {
    std::string hostname;               ///< System hostname
    std::uint32_t total_cpus = 0;       ///< Total logical CPUs
    std::uint32_t total_cores = 0;      ///< Total physical cores
    std::vector<cpu_socket> sockets;    ///< CPU sockets
    std::vector<numa_node> numa_nodes;  ///< NUMA nodes
    std::vector<gpu_device> gpus;       ///< GPU devices

    /// @brief Check if topology is empty/uninitialized
    [[nodiscard]] bool empty() const noexcept {
        return total_cpus == 0;
    }

    /// @brief Get number of NUMA nodes
    [[nodiscard]] size_type num_numa_nodes() const noexcept {
        return numa_nodes.size();
    }

    /// @brief Get number of GPUs
    [[nodiscard]] size_type num_gpus() const noexcept {
        return gpus.size();
    }

    /// @brief Check if system has NUMA topology
    [[nodiscard]] bool has_numa() const noexcept {
        return numa_nodes.size() > 1;
    }

    /// @brief Check if system has GPUs
    [[nodiscard]] bool has_gpus() const noexcept {
        return !gpus.empty();
    }

    /// @brief NUMA distance matrix from sysfs (empty if unavailable)
    /// @details numa_distances[i][j] is the distance from NUMA node i to node j.
    ///          Populated from /sys/devices/system/node/nodeN/distance.
    std::vector<std::vector<std::uint32_t>> numa_distances;
};

// ============================================================================
// Discovery Functions
// ============================================================================

/// @brief Discover local hardware topology
/// @return Hardware topology information
/// @details Platform-specific discovery:
///          - Linux: Parses /sys/devices/system/cpu and /sys/devices/system/node
///          - CUDA: Uses cudaGetDeviceCount/cudaGetDeviceProperties
hardware_topology discover_local();

/// @brief Get cached local topology (singleton)
/// @return Reference to cached topology
/// @details Calls discover_local() on first access, then returns cached result.
const hardware_topology& local_topology();

/// @brief Refresh the cached local topology
/// @return Reference to newly discovered topology
const hardware_topology& refresh_local_topology();

// ============================================================================
// Query Functions
// ============================================================================

/// @brief Get total number of logical CPUs
/// @return CPU count
[[nodiscard]] std::uint32_t cpu_count();

/// @brief Get number of NUMA nodes
/// @return NUMA node count
[[nodiscard]] std::uint32_t numa_node_count();

/// @brief Get number of GPU devices
/// @return GPU count
[[nodiscard]] std::uint32_t gpu_count();

/// @brief Get CPUs on a NUMA node
/// @param node NUMA node ID
/// @return Vector of CPU IDs, empty if node doesn't exist
[[nodiscard]] std::vector<std::uint32_t> cpus_on_numa_node(std::uint32_t node);

/// @brief Get NUMA node for a CPU
/// @param cpu CPU ID
/// @return NUMA node ID, or 0 if not found
[[nodiscard]] std::uint32_t numa_node_for_cpu(std::uint32_t cpu);

/// @brief Get GPUs near a NUMA node
/// @param node NUMA node ID
/// @return Vector of GPU IDs
[[nodiscard]] std::vector<std::uint32_t> gpus_on_numa_node(std::uint32_t node);

}  // namespace dtl::topology
