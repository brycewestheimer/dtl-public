// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_topology.hpp
/// @brief MPI topology discovery and node-awareness utilities
/// @details Provides utilities for discovering network topology,
///          identifying shared-memory domains, and optimizing communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <numeric>
#include <string>
#include <vector>
#include <unordered_map>

namespace dtl {
namespace mpi {

// ============================================================================
// Node Information
// ============================================================================

/// @brief Information about a compute node
struct node_info {
    /// @brief Unique node identifier (usually hostname hash)
    size_type node_id = 0;

    /// @brief Human-readable node name (hostname)
    std::string node_name;

    /// @brief Ranks on this node
    std::vector<rank_t> local_ranks;

    /// @brief Number of ranks on this node
    [[nodiscard]] rank_t num_local_ranks() const noexcept {
        return static_cast<rank_t>(local_ranks.size());
    }
};

// ============================================================================
// Topology Information
// ============================================================================

/// @brief Network topology information for a communicator
struct topology_info {
    /// @brief Total number of ranks
    rank_t world_size = 0;

    /// @brief Number of unique nodes
    size_type num_nodes = 0;

    /// @brief Information for each node
    std::vector<node_info> nodes;

    /// @brief Map from world rank to node index
    std::vector<size_type> rank_to_node;

    /// @brief Map from world rank to local rank within node
    std::vector<rank_t> rank_to_local;
};

// ============================================================================
// MPI Topology Discovery
// ============================================================================

/// @brief Topology discovery utilities for MPI communicators
class mpi_topology {
public:
    /// @brief Default constructor
    mpi_topology() = default;

#if DTL_ENABLE_MPI
    /// @brief Construct from MPI communicator
    /// @param comm MPI communicator to analyze
    explicit mpi_topology(MPI_Comm comm)
        : comm_(comm) {
        discover_topology();
    }
#endif

    /// @brief Destructor
    ~mpi_topology() = default;

    // Non-copyable
    mpi_topology(const mpi_topology&) = delete;
    mpi_topology& operator=(const mpi_topology&) = delete;

    // Movable
    mpi_topology(mpi_topology&&) = default;
    mpi_topology& operator=(mpi_topology&&) = default;

    // ------------------------------------------------------------------------
    // Topology Queries
    // ------------------------------------------------------------------------

    /// @brief Get full topology information
    [[nodiscard]] const topology_info& info() const noexcept { return info_; }

    /// @brief Get number of nodes
    [[nodiscard]] size_type num_nodes() const noexcept { return info_.num_nodes; }

    /// @brief Get world size
    [[nodiscard]] rank_t world_size() const noexcept { return info_.world_size; }

    /// @brief Get node ID for a rank
    /// @param rank The world rank to query
    /// @return Node index
    [[nodiscard]] size_type node_of(rank_t rank) const noexcept {
        if (rank < 0) {
            return 0;
        }
        const size_type idx = static_cast<size_type>(rank);
        if (idx >= info_.rank_to_node.size()) {
            return 0;
        }
        return info_.rank_to_node[idx];
    }

    /// @brief Get local rank within node
    /// @param rank The world rank to query
    /// @return Local rank on the node (0-based)
    [[nodiscard]] rank_t local_rank_of(rank_t rank) const noexcept {
        if (rank < 0) {
            return 0;
        }
        const size_type idx = static_cast<size_type>(rank);
        if (idx >= info_.rank_to_local.size()) {
            return 0;
        }
        return info_.rank_to_local[idx];
    }

    /// @brief Check if two ranks are on the same node
    /// @param rank1 First rank
    /// @param rank2 Second rank
    /// @return true if ranks share a node
    [[nodiscard]] bool same_node(rank_t rank1, rank_t rank2) const noexcept {
        return node_of(rank1) == node_of(rank2);
    }

    /// @brief Get all ranks on a specific node
    /// @param node_idx Node index
    /// @return Vector of ranks on that node
    [[nodiscard]] const std::vector<rank_t>& ranks_on_node(size_type node_idx) const {
        static const std::vector<rank_t> empty;
        if (node_idx >= info_.nodes.size()) return empty;
        return info_.nodes[node_idx].local_ranks;
    }

    /// @brief Get node information
    /// @param node_idx Node index
    /// @return Node info structure
    [[nodiscard]] const node_info& get_node_info(size_type node_idx) const {
        static const node_info empty;
        if (node_idx >= info_.nodes.size()) return empty;
        return info_.nodes[node_idx];
    }

    // ------------------------------------------------------------------------
    // Split Operations
    // ------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Create a communicator for ranks on the same node
    /// @return Intra-node communicator or error
    [[nodiscard]] result<MPI_Comm> create_node_comm() const {
        MPI_Comm node_comm;
        int result = MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0,
                                          MPI_INFO_NULL, &node_comm);
        if (result != MPI_SUCCESS) {
            return make_error<MPI_Comm>(status_code::communication_error,
                                        "MPI_Comm_split_type failed");
        }
        return node_comm;
    }

    /// @brief Create a communicator for leaders of each node
    /// @return Inter-node communicator (only valid for node leaders) or error
    [[nodiscard]] result<MPI_Comm> create_leader_comm() const {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        // Node leaders have local rank 0
        int color = (info_.rank_to_local[rank] == 0) ? 0 : MPI_UNDEFINED;

        MPI_Comm leader_comm;
        int result = MPI_Comm_split(comm_, color, rank, &leader_comm);
        if (result != MPI_SUCCESS) {
            return make_error<MPI_Comm>(status_code::communication_error,
                                        "MPI_Comm_split failed");
        }
        return leader_comm;
    }

    /// @brief Get the underlying MPI communicator
    [[nodiscard]] MPI_Comm native_handle() const noexcept { return comm_; }
#endif

    // ------------------------------------------------------------------------
    // Optimization Hints
    // ------------------------------------------------------------------------

    /// @brief Check if communication between ranks can use shared memory
    /// @param src Source rank
    /// @param dst Destination rank
    /// @return true if shared memory is available
    [[nodiscard]] bool can_use_shared_memory(rank_t src, rank_t dst) const noexcept {
        return same_node(src, dst);
    }

    /// @brief Estimate relative communication cost
    /// @param src Source rank
    /// @param dst Destination rank
    /// @return Relative cost (1 = intra-node, higher = inter-node)
    [[nodiscard]] double comm_cost_estimate(rank_t src, rank_t dst) const noexcept {
        if (same_node(src, dst)) {
            return 1.0;  // Intra-node (shared memory possible)
        }
        return 10.0;  // Inter-node (network required)
    }

private:
    void discover_topology() {
#if DTL_ENABLE_MPI
        int rank, size;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);

        info_.world_size = size;

        // Get processor name (hostname)
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        std::string my_hostname(processor_name, name_len);

        // Gather all hostnames
        std::vector<char> all_names(size * MPI_MAX_PROCESSOR_NAME);
        MPI_Allgather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm_);

        // Build node map
        std::unordered_map<std::string, size_type> hostname_to_node;
        info_.rank_to_node.resize(size);
        info_.rank_to_local.resize(size);

        for (rank_t r = 0; r < size; ++r) {
            std::string hostname(&all_names[r * MPI_MAX_PROCESSOR_NAME]);

            auto it = hostname_to_node.find(hostname);
            if (it == hostname_to_node.end()) {
                // New node
                size_type node_idx = info_.nodes.size();
                hostname_to_node[hostname] = node_idx;

                node_info ni;
                ni.node_id = std::hash<std::string>{}(hostname);
                ni.node_name = hostname;
                ni.local_ranks.push_back(r);
                info_.nodes.push_back(std::move(ni));

                info_.rank_to_node[r] = node_idx;
                info_.rank_to_local[r] = 0;
            } else {
                // Existing node
                size_type node_idx = it->second;
                info_.rank_to_local[r] = static_cast<rank_t>(
                    info_.nodes[node_idx].local_ranks.size());
                info_.nodes[node_idx].local_ranks.push_back(r);
                info_.rank_to_node[r] = node_idx;
            }
        }

        info_.num_nodes = info_.nodes.size();
#endif
    }

#if DTL_ENABLE_MPI
    MPI_Comm comm_ = MPI_COMM_NULL;
#endif
    topology_info info_;
};

// ============================================================================
// Factory Functions
// ============================================================================

#if DTL_ENABLE_MPI
/// @brief Discover topology for MPI_COMM_WORLD
/// @return Topology for world communicator
[[nodiscard]] inline mpi_topology discover_world_topology() {
    return mpi_topology(MPI_COMM_WORLD);
}
#endif

// ============================================================================
// Topology-Aware Utilities
// ============================================================================

/// @brief Suggest optimal rank ordering for a given communication pattern
/// @details Currently returns identity ordering. A future implementation may
///          use topology-aware reordering (e.g., Cuthill-McKee or graph
///          partitioning) to minimize inter-node communication.
/// @param topo Topology information
/// @param pattern Communication pattern (pairs of communicating ranks)
/// @return Suggested rank reordering (currently identity permutation)
[[nodiscard]] inline std::vector<rank_t> optimize_rank_ordering(
    const mpi_topology& topo,
    const std::vector<std::pair<rank_t, rank_t>>& pattern) {
    (void)pattern;  // unused in stub
    // Stub: return identity ordering
    if (topo.world_size() <= 0) {
        return {};
    }
    std::vector<rank_t> ordering(static_cast<size_type>(topo.world_size()));
    std::iota(ordering.begin(), ordering.end(), rank_t{0});
    return ordering;
}

/// @brief Calculate communication volume matrix
/// @param topo Topology information
/// @return Matrix of communication costs (num_nodes x num_nodes)
[[nodiscard]] inline std::vector<std::vector<double>> comm_volume_matrix(
    const mpi_topology& topo) {
    size_type n = topo.num_nodes();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

    for (size_type i = 0; i < n; ++i) {
        for (size_type j = 0; j < n; ++j) {
            matrix[i][j] = (i == j) ? 1.0 : 10.0;
        }
    }

    return matrix;
}

}  // namespace mpi
}  // namespace dtl
