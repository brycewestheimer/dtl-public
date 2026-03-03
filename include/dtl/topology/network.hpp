// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file network.hpp
/// @brief Network topology discovery for distributed systems
/// @details Provides functions for discovering rank-to-host mapping and
///          rank proximity based on network topology.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/topology/proximity.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

namespace dtl::topology {

// ============================================================================
// Hostname Utilities
// ============================================================================

/// @brief Get local hostname
/// @return Hostname string
[[nodiscard]] inline std::string get_hostname() {
#if defined(__linux__)
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
#endif
    return "unknown";
}

// ============================================================================
// Rank-to-Host Mapping
// ============================================================================

/// @brief Rank-to-hostname mapping
class rank_host_map {
public:
    /// @brief Default constructor (empty map)
    rank_host_map() = default;

    /// @brief Construct from vector of hostnames indexed by rank
    explicit rank_host_map(std::vector<std::string> hostnames)
        : hostnames_(std::move(hostnames)) {
        build_host_to_ranks();
    }

    /// @brief Get hostname for a rank
    /// @param r Rank
    /// @return Hostname, or empty string if unknown
    [[nodiscard]] std::string hostname_for_rank(rank_t r) const {
        if (r < 0 || static_cast<size_type>(r) >= hostnames_.size()) {
            return "";
        }
        return hostnames_[static_cast<size_type>(r)];
    }

    /// @brief Check if two ranks are on the same node
    /// @param a First rank
    /// @param b Second rank
    /// @return true if same hostname
    [[nodiscard]] bool same_node(rank_t a, rank_t b) const {
        auto ha = hostname_for_rank(a);
        auto hb = hostname_for_rank(b);
        return !ha.empty() && !hb.empty() && ha == hb;
    }

    /// @brief Get all ranks on the same node as a given rank
    /// @param r Reference rank
    /// @return Vector of ranks (including r itself)
    [[nodiscard]] std::vector<rank_t> ranks_on_same_node(rank_t r) const {
        auto hostname = hostname_for_rank(r);
        if (hostname.empty()) return {r};

        auto it = host_to_ranks_.find(hostname);
        if (it != host_to_ranks_.end()) {
            return it->second;
        }
        return {r};
    }

    /// @brief Get ranks sorted by proximity to a given rank
    /// @param r Reference rank
    /// @return Vector of all ranks, sorted by proximity (nearest first)
    [[nodiscard]] std::vector<rank_t> ranks_by_proximity(rank_t r) const {
        if (hostnames_.empty()) return {};

        std::vector<rank_t> result;
        result.reserve(hostnames_.size());

        // First: ranks on same node
        auto same_node_ranks = ranks_on_same_node(r);
        for (rank_t rank : same_node_ranks) {
            result.push_back(rank);
        }

        // Then: ranks on other nodes
        for (size_type i = 0; i < hostnames_.size(); ++i) {
            rank_t rank = static_cast<rank_t>(i);
            if (!same_node(r, rank)) {
                result.push_back(rank);
            }
        }

        return result;
    }

    /// @brief Get total number of ranks
    [[nodiscard]] size_type size() const noexcept {
        return hostnames_.size();
    }

    /// @brief Get unique hostnames
    [[nodiscard]] std::vector<std::string> unique_hosts() const {
        std::vector<std::string> result;
        for (const auto& [host, ranks] : host_to_ranks_) {
            result.push_back(host);
        }
        return result;
    }

    /// @brief Get number of unique hosts
    [[nodiscard]] size_type num_hosts() const noexcept {
        return host_to_ranks_.size();
    }

private:
    void build_host_to_ranks() {
        host_to_ranks_.clear();
        for (size_type i = 0; i < hostnames_.size(); ++i) {
            host_to_ranks_[hostnames_[i]].push_back(static_cast<rank_t>(i));
        }
    }

    std::vector<std::string> hostnames_;
    std::unordered_map<std::string, std::vector<rank_t>> host_to_ranks_;
};

// ============================================================================
// Rank Proximity
// ============================================================================

/// @brief Build rank proximity matrix from host mapping
/// @param host_map Rank-to-host mapping
/// @return Proximity matrix for ranks
[[nodiscard]] inline proximity_matrix build_rank_proximity(const rank_host_map& host_map) {
    size_type n = host_map.size();
    if (n == 0) return {};

    proximity_matrix prox(n);

    for (size_type i = 0; i < n; ++i) {
        for (size_type j = i + 1; j < n; ++j) {
            rank_t ri = static_cast<rank_t>(i);
            rank_t rj = static_cast<rank_t>(j);

            distance_t dist = host_map.same_node(ri, rj)
                            ? adjacent_distance
                            : network_distance;

            prox.set_distance(i, j, dist);
        }
    }

    return prox;
}

// ============================================================================
// MPI Integration Stubs
// ============================================================================

/// @brief Maximum hostname length for MPI gather
inline constexpr size_type max_hostname_length = 256;

/// @brief Gather hostnames from all MPI ranks
/// @param my_rank Local rank
/// @param num_ranks Total number of ranks
/// @return Rank-to-host mapping
/// @note Uses MPI_Allgather when MPI is enabled
[[nodiscard]] inline rank_host_map gather_hostnames(
    [[maybe_unused]] rank_t my_rank,
    [[maybe_unused]] rank_t num_ranks) {
    std::vector<std::string> hostnames;
    hostnames.resize(static_cast<size_type>(num_ranks));

    std::string local_hostname = get_hostname();

    // Single-rank case: just return local hostname
    if (num_ranks <= 1) {
        hostnames[0] = local_hostname;
        return rank_host_map(std::move(hostnames));
    }

#if DTL_ENABLE_MPI && defined(__has_include)
#if __has_include(<mpi.h>)
    // Use MPI_Allgather to collect hostnames from all ranks
    // Each rank sends a fixed-size buffer
    std::vector<char> send_buf(max_hostname_length, '\0');
    std::vector<char> recv_buf(max_hostname_length * static_cast<size_type>(num_ranks), '\0');

    // Copy local hostname to send buffer (null-terminated)
    std::strncpy(send_buf.data(), local_hostname.c_str(),
                 max_hostname_length - 1);

    // Gather all hostnames
    MPI_Allgather(send_buf.data(), static_cast<int>(max_hostname_length), MPI_CHAR,
                  recv_buf.data(), static_cast<int>(max_hostname_length), MPI_CHAR,
                  MPI_COMM_WORLD);

    // Extract hostnames from receive buffer
    for (rank_t r = 0; r < num_ranks; ++r) {
        const char* hostname_ptr = recv_buf.data() +
                                   static_cast<size_type>(r) * max_hostname_length;
        hostnames[static_cast<size_type>(r)] = std::string(hostname_ptr);
    }
#else
    // MPI not available - assume all ranks on same host
    for (rank_t r = 0; r < num_ranks; ++r) {
        hostnames[static_cast<size_type>(r)] = local_hostname;
    }
#endif
#else
    // MPI disabled - assume all ranks on same host
    for (rank_t r = 0; r < num_ranks; ++r) {
        hostnames[static_cast<size_type>(r)] = local_hostname;
    }
#endif

    return rank_host_map(std::move(hostnames));
}

/// @brief Gather hostnames using a communicator adapter
/// @tparam Communicator Communicator type with allgather support
/// @param comm Communicator adapter
/// @return Rank-to-host mapping
template <typename Communicator>
[[nodiscard]] rank_host_map gather_hostnames_via_comm(Communicator& comm) {
    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    std::vector<std::string> hostnames;
    hostnames.resize(static_cast<size_type>(num_ranks));

    std::string local_hostname = get_hostname();

    if (num_ranks <= 1) {
        hostnames[0] = local_hostname;
        return rank_host_map(std::move(hostnames));
    }

    // Prepare buffers
    std::vector<char> send_buf(max_hostname_length, '\0');
    std::vector<char> recv_buf(max_hostname_length * static_cast<size_type>(num_ranks), '\0');

    std::strncpy(send_buf.data(), local_hostname.c_str(),
                 max_hostname_length - 1);

    // Use communicator's allgather
    comm.allgather(send_buf.data(), recv_buf.data(), max_hostname_length);

    // Extract hostnames
    for (rank_t r = 0; r < num_ranks; ++r) {
        const char* hostname_ptr = recv_buf.data() +
                                   static_cast<size_type>(r) * max_hostname_length;
        hostnames[static_cast<size_type>(r)] = std::string(hostname_ptr);
    }

    return rank_host_map(std::move(hostnames));
}

/// @brief Build topology info for multi-host systems
/// @param host_map Rank-to-host mapping
/// @return Topology summary
struct topology_summary {
    size_type num_hosts;
    size_type num_ranks;
    size_type ranks_per_host_min;
    size_type ranks_per_host_max;
    std::vector<std::string> host_names;
    std::vector<size_type> ranks_per_host;
};

[[nodiscard]] inline topology_summary analyze_topology(const rank_host_map& host_map) {
    topology_summary summary;
    summary.num_ranks = host_map.size();
    summary.host_names = host_map.unique_hosts();
    summary.num_hosts = summary.host_names.size();

    // Count ranks per host
    summary.ranks_per_host.reserve(summary.num_hosts);
    summary.ranks_per_host_min = summary.num_ranks;
    summary.ranks_per_host_max = 0;

    for (const auto& host : summary.host_names) {
        size_type count = 0;
        for (size_type r = 0; r < summary.num_ranks; ++r) {
            if (host_map.hostname_for_rank(static_cast<rank_t>(r)) == host) {
                ++count;
            }
        }
        summary.ranks_per_host.push_back(count);
        summary.ranks_per_host_min = std::min(summary.ranks_per_host_min, count);
        summary.ranks_per_host_max = std::max(summary.ranks_per_host_max, count);
    }

    return summary;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Check if two ranks are on the same physical node
/// @param host_map Rank-to-host mapping
/// @param a First rank
/// @param b Second rank
/// @return true if same node
[[nodiscard]] inline bool ranks_colocated(
    const rank_host_map& host_map, rank_t a, rank_t b) {
    return host_map.same_node(a, b);
}

/// @brief Get local ranks (ranks on same node as given rank)
/// @param host_map Rank-to-host mapping
/// @param r Reference rank
/// @return Vector of local ranks
[[nodiscard]] inline std::vector<rank_t> local_ranks(
    const rank_host_map& host_map, rank_t r) {
    return host_map.ranks_on_same_node(r);
}

/// @brief Get remote ranks (ranks on different nodes than given rank)
/// @param host_map Rank-to-host mapping
/// @param r Reference rank
/// @return Vector of remote ranks
[[nodiscard]] inline std::vector<rank_t> remote_ranks(
    const rank_host_map& host_map, rank_t r) {
    std::vector<rank_t> result;
    for (size_type i = 0; i < host_map.size(); ++i) {
        rank_t rank = static_cast<rank_t>(i);
        if (!host_map.same_node(r, rank)) {
            result.push_back(rank);
        }
    }
    return result;
}

}  // namespace dtl::topology
