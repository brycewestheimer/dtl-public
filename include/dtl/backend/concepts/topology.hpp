// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file topology.hpp
/// @brief Topology concept for hardware layout discovery
/// @details Defines requirements for querying system topology.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace dtl {

// ============================================================================
// Topology Node Types
// ============================================================================

/// @brief Type of topology node
enum class topology_node_type {
    machine,    ///< Physical machine/node
    socket,     ///< CPU socket
    numa_node,  ///< NUMA domain
    core,       ///< CPU core
    pu,         ///< Processing unit (hardware thread)
    gpu,        ///< GPU device
    nic,        ///< Network interface
    memory,     ///< Memory region
    unknown     ///< Unknown node type
};

// ============================================================================
// Node Properties
// ============================================================================

/// @brief Properties of a topology node
struct topology_node_properties {
    /// @brief Node type
    topology_node_type type = topology_node_type::unknown;

    /// @brief Logical index within parent
    size_type index = 0;

    /// @brief OS-assigned index (if applicable)
    int os_index = -1;

    /// @brief Human-readable name
    std::string name;

    /// @brief Memory capacity in bytes (for memory nodes)
    size_type memory_size = 0;
};

// ============================================================================
// Topology Concept
// ============================================================================

/// @brief Core topology concept for hardware discovery
/// @details Defines minimum requirements for querying system topology.
///
/// @par Required Operations:
/// - num_nodes(): Get number of compute nodes
/// - num_sockets(): Get number of CPU sockets
/// - num_cores(): Get total CPU cores
/// - num_gpus(): Get number of GPU devices
template <typename T>
concept Topology = requires(const T& topo) {
    // Node counts
    { topo.num_nodes() } -> std::same_as<size_type>;
    { topo.num_sockets() } -> std::same_as<size_type>;
    { topo.num_cores() } -> std::same_as<size_type>;
    { topo.num_pus() } -> std::same_as<size_type>;
    { topo.num_gpus() } -> std::same_as<size_type>;

    // Hostname/identifier
    { topo.hostname() } -> std::convertible_to<std::string>;
};

// ============================================================================
// NUMA-Aware Topology Concept
// ============================================================================

/// @brief Topology with NUMA awareness
template <typename T>
concept NumaTopology = Topology<T> &&
    requires(const T& topo, size_type numa_node, size_type core) {
    // NUMA queries
    { topo.num_numa_nodes() } -> std::same_as<size_type>;
    { topo.numa_node_of_core(core) } -> std::same_as<size_type>;
    { topo.cores_in_numa_node(numa_node) } -> std::convertible_to<std::vector<size_type>>;
    { topo.memory_of_numa_node(numa_node) } -> std::same_as<size_type>;
};

// ============================================================================
// GPU Topology Concept
// ============================================================================

/// @brief Topology with GPU awareness
template <typename T>
concept GpuTopology = Topology<T> &&
    requires(const T& topo, size_type gpu_id) {
    // GPU queries
    { topo.gpu_name(gpu_id) } -> std::convertible_to<std::string>;
    { topo.gpu_memory(gpu_id) } -> std::same_as<size_type>;
    { topo.gpu_compute_capability(gpu_id) } -> std::convertible_to<std::pair<int, int>>;

    // GPU-CPU affinity
    { topo.gpus_near_numa_node(size_type{}) } -> std::convertible_to<std::vector<size_type>>;
    { topo.closest_numa_node_to_gpu(gpu_id) } -> std::same_as<size_type>;
};

// ============================================================================
// Network Topology Concept
// ============================================================================

/// @brief Topology with network awareness
template <typename T>
concept NetworkTopology = Topology<T> &&
    requires(const T& topo, rank_t rank1, rank_t rank2) {
    // Network queries
    { topo.num_nics() } -> std::same_as<size_type>;
    { topo.is_same_node(rank1, rank2) } -> std::same_as<bool>;
    { topo.hops_between(rank1, rank2) } -> std::same_as<int>;
    { topo.bandwidth_between(rank1, rank2) } -> std::same_as<double>;
};

// ============================================================================
// Topology Traits
// ============================================================================

/// @brief Traits for topology types
template <typename Topo>
struct topology_traits {
    /// @brief Whether topology provides NUMA information
    static constexpr bool has_numa = false;

    /// @brief Whether topology provides GPU information
    static constexpr bool has_gpu = false;

    /// @brief Whether topology provides network information
    static constexpr bool has_network = false;

    /// @brief Whether topology is dynamically updated
    static constexpr bool is_dynamic = false;
};

// ============================================================================
// Basic Topology Implementation
// ============================================================================

/// @brief Basic topology that queries standard system information
/// @details Uses real system calls to discover hardware topology
///          instead of returning hardcoded stub values.
class basic_topology {
public:
    /// @brief Get number of nodes (always 1 for basic)
    [[nodiscard]] static constexpr size_type num_nodes() noexcept {
        return 1;
    }

    /// @brief Get number of sockets
    [[nodiscard]] static size_type num_sockets() noexcept {
        static const size_type value = discover_sockets();
        return value;
    }

    /// @brief Get number of cores
    [[nodiscard]] static size_type num_cores() noexcept {
        // Approximate: use hardware_concurrency as a proxy for cores
        // (may include hyperthreads, but better than returning 1)
        static const size_type value = []() -> size_type {
            auto n = std::thread::hardware_concurrency();
            return (n > 0) ? static_cast<size_type>(n) : 1;
        }();
        return value;
    }

    /// @brief Get number of processing units (hardware threads)
    [[nodiscard]] static size_type num_pus() noexcept {
        static const size_type value = []() -> size_type {
            auto n = std::thread::hardware_concurrency();
            return (n > 0) ? static_cast<size_type>(n) : 1;
        }();
        return value;
    }

    /// @brief Get number of GPUs
    [[nodiscard]] static size_type num_gpus() noexcept {
        static const size_type value = discover_gpus();
        return value;
    }

    /// @brief Get hostname
    [[nodiscard]] static std::string hostname() {
        static const std::string value = discover_hostname();
        return value;
    }

private:
    /// @brief Discover actual hostname
    static std::string discover_hostname() {
#if defined(__linux__) || defined(__APPLE__)
        char buf[256];
        if (::gethostname(buf, sizeof(buf)) == 0) {
            buf[sizeof(buf) - 1] = 0;
            return std::string(buf);
        }
#endif
        return "localhost";
    }

    /// @brief Discover number of CPU sockets
    static size_type discover_sockets() {
#if defined(__linux__)
        // Count unique physical_package_id values
        try {
            std::set<std::uint32_t> packages;
            for (std::uint32_t cpu = 0; cpu < 4096; ++cpu) {
                std::string path = "/sys/devices/system/cpu/cpu"
                                   + std::to_string(cpu)
                                   + "/topology/physical_package_id";
                std::ifstream file(path);
                if (!file) break;
                std::uint32_t pkg = 0;
                if (file >> pkg) {
                    packages.insert(pkg);
                }
            }
            if (!packages.empty()) {
                return packages.size();
            }
        } catch (...) {}
#endif
        return 1;
    }

    /// @brief Discover number of GPU devices
    static size_type discover_gpus() {
        // No GPU runtime linked in basic_topology — return 0.
        // CUDA/HIP-aware topology is in dtl::topology::detail.
        return 0;
    }
};

/// @brief Traits specialization for basic_topology
template <>
struct topology_traits<basic_topology> {
    static constexpr bool has_numa = false;
    static constexpr bool has_gpu = false;
    static constexpr bool has_network = false;
    static constexpr bool is_dynamic = false;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Topology<basic_topology>, "basic_topology must satisfy Topology concept");

// ============================================================================
// Topology Utilities
// ============================================================================

/// @brief Get the default topology for the current system
/// @return Reference to singleton topology object
[[nodiscard]] inline basic_topology& get_default_topology() {
    static basic_topology topo;
    return topo;
}

/// @brief Check if two ranks are on the same physical node
/// @tparam Topo Topology type
/// @param topo The topology
/// @param rank1 First rank
/// @param rank2 Second rank
/// @return true if on same node
template <Topology Topo>
[[nodiscard]] bool same_node(const Topo& topo, rank_t rank1, rank_t rank2) {
    if constexpr (NetworkTopology<Topo>) {
        return topo.is_same_node(rank1, rank2);
    } else {
        // Without network topology, assume same if same rank
        return rank1 == rank2;
    }
}

}  // namespace dtl
