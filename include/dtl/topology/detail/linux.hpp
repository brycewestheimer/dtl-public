// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file detail/linux.hpp
/// @brief Linux-specific hardware topology discovery
/// @details Parses /sys/devices/system/cpu and related files for topology info.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/topology/hardware.hpp>

#if defined(__linux__)

#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <set>
#include <unistd.h>

namespace dtl::topology::detail {

/// @brief Read a single integer from a sysfs file
/// @param path Path to the file
/// @return Value or 0 if not readable
inline std::uint32_t read_sysfs_uint(const std::string& path) {
    std::ifstream file(path);
    if (!file) return 0;

    std::uint32_t value = 0;
    file >> value;
    return value;
}

/// @brief Read hostname
inline std::string read_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}

/// @brief Parse a CPU list string like "0-3,5,7-9"
/// @param list_str CPU list string
/// @return Set of CPU IDs
inline std::set<std::uint32_t> parse_cpu_list(const std::string& list_str) {
    std::set<std::uint32_t> cpus;

    std::istringstream iss(list_str);
    std::string token;

    while (std::getline(iss, token, ',')) {
        // Check for range (e.g., "0-3")
        auto dash = token.find('-');
        if (dash != std::string::npos) {
            std::uint32_t start = static_cast<std::uint32_t>(std::stoul(token.substr(0, dash)));
            std::uint32_t end = static_cast<std::uint32_t>(std::stoul(token.substr(dash + 1)));
            for (std::uint32_t i = start; i <= end; ++i) {
                cpus.insert(i);
            }
        } else if (!token.empty()) {
            cpus.insert(static_cast<std::uint32_t>(std::stoul(token)));
        }
    }

    return cpus;
}

/// @brief Read CPU list from a sysfs file
inline std::set<std::uint32_t> read_cpu_list(const std::string& path) {
    std::ifstream file(path);
    if (!file) return {};

    std::string line;
    std::getline(file, line);
    return parse_cpu_list(line);
}

/// @brief Discover CPU topology on Linux
inline void discover_cpus_linux(hardware_topology& topo) {
    namespace fs = std::filesystem;

    const std::string cpu_base = "/sys/devices/system/cpu";

    // Count online CPUs
    auto online_cpus = read_cpu_list(cpu_base + "/online");
    topo.total_cpus = static_cast<std::uint32_t>(online_cpus.size());

    if (topo.total_cpus == 0) {
        // Fallback to sysconf
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        if (nprocs > 0) {
            topo.total_cpus = static_cast<std::uint32_t>(nprocs);
        }
    }

    // Discover socket topology
    std::set<std::uint32_t> seen_packages;
    for (std::uint32_t cpu : online_cpus) {
        std::string cpu_path = cpu_base + "/cpu" + std::to_string(cpu);

        // Read package (socket) ID
        std::uint32_t package_id = read_sysfs_uint(
            cpu_path + "/topology/physical_package_id");

        if (seen_packages.find(package_id) == seen_packages.end()) {
            seen_packages.insert(package_id);

            cpu_socket socket;
            socket.id = package_id;

            // Count cores on this socket
            auto core_siblings = read_cpu_list(cpu_path + "/topology/core_siblings_list");
            socket.num_threads = static_cast<std::uint32_t>(core_siblings.size());

            // Count physical cores (unique core IDs)
            std::set<std::uint32_t> core_ids;
            for (std::uint32_t sibling : core_siblings) {
                std::string sibling_path = cpu_base + "/cpu" + std::to_string(sibling);
                std::uint32_t core_id = read_sysfs_uint(sibling_path + "/topology/core_id");
                core_ids.insert(core_id);
            }
            socket.num_cores = static_cast<std::uint32_t>(core_ids.size());

            topo.sockets.push_back(std::move(socket));
        }
    }

    // Calculate total cores
    topo.total_cores = 0;
    for (const auto& socket : topo.sockets) {
        topo.total_cores += socket.num_cores;
    }
}

/// @brief Discover NUMA topology on Linux
inline void discover_numa_linux(hardware_topology& topo) {
    namespace fs = std::filesystem;

    const std::string numa_base = "/sys/devices/system/node";

    if (!fs::exists(numa_base)) {
        // No NUMA - create single node with all CPUs
        numa_node node;
        node.id = 0;
        node.memory_bytes = static_cast<std::uint64_t>(sysconf(_SC_PHYS_PAGES))
                          * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
        for (std::uint32_t i = 0; i < topo.total_cpus; ++i) {
            node.cpu_ids.push_back(i);
        }
        topo.numa_nodes.push_back(std::move(node));
        return;
    }

    // Enumerate NUMA nodes
    for (const auto& entry : fs::directory_iterator(numa_base)) {
        std::string name = entry.path().filename().string();
        if (name.substr(0, 4) != "node") continue;

        std::uint32_t node_id = static_cast<std::uint32_t>(std::stoul(name.substr(4)));

        numa_node node;
        node.id = node_id;

        // Read memory info
        std::ifstream meminfo(entry.path() / "meminfo");
        if (meminfo) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") != std::string::npos) {
                    // Format: "Node X MemTotal: NNNN kB"
                    std::istringstream iss(line);
                    std::string tok;
                    std::uint64_t kb = 0;
                    while (iss >> tok) {
                        try {
                            kb = std::stoull(tok);
                        } catch (...) {}
                    }
                    node.memory_bytes = kb * 1024;
                    break;
                }
            }
        }

        // Read CPUs on this node
        auto cpulist = read_cpu_list((entry.path() / "cpulist").string());
        node.cpu_ids.assign(cpulist.begin(), cpulist.end());

        topo.numa_nodes.push_back(std::move(node));
    }

    // Sort by ID
    std::sort(topo.numa_nodes.begin(), topo.numa_nodes.end(),
              [](const numa_node& a, const numa_node& b) { return a.id < b.id; });

    // Read NUMA distance matrix from sysfs
    topo.numa_distances.resize(topo.numa_nodes.size());
    bool distances_ok = true;
    for (std::size_t i = 0; i < topo.numa_nodes.size(); ++i) {
        std::string distance_path = numa_base + "/node" +
            std::to_string(topo.numa_nodes[i].id) + "/distance";
        std::ifstream dist_file(distance_path);
        if (!dist_file) {
            distances_ok = false;
            break;
        }
        std::string dist_line;
        if (!std::getline(dist_file, dist_line)) {
            distances_ok = false;
            break;
        }
        std::istringstream diss(dist_line);
        std::uint32_t d;
        while (diss >> d) {
            topo.numa_distances[i].push_back(d);
        }
    }
    if (!distances_ok) {
        topo.numa_distances.clear();
    }

    // Associate sockets with NUMA nodes by checking physical_package_id
    const std::string cpu_base_path = "/sys/devices/system/cpu";
    for (auto& socket : topo.sockets) {
        for (const auto& node : topo.numa_nodes) {
            // Check if any CPU in this NUMA node belongs to this socket
            bool belongs_to_socket = false;
            for (auto cpu_id : node.cpu_ids) {
                std::uint32_t package_id = read_sysfs_uint(
                    cpu_base_path + "/cpu" + std::to_string(cpu_id) +
                    "/topology/physical_package_id");
                if (package_id == socket.id) {
                    belongs_to_socket = true;
                    break;
                }
            }
            if (belongs_to_socket) {
                socket.numa_nodes.push_back(node.id);
            }
        }
    }
}

/// @brief Full Linux hardware discovery
inline hardware_topology discover_linux() {
    hardware_topology topo;

    topo.hostname = read_hostname();
    discover_cpus_linux(topo);
    discover_numa_linux(topo);

    return topo;
}

}  // namespace dtl::topology::detail

#endif  // __linux__
