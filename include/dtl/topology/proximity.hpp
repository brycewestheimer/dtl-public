// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file proximity.hpp
/// @brief Proximity matrices for topology-aware placement
/// @details Provides distance metrics between hardware components.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/topology/hardware.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace dtl::topology {

// ============================================================================
// Distance Types
// ============================================================================

/// @brief Distance type (lower is closer)
using distance_t = std::uint32_t;

/// @brief Maximum distance (unreachable)
inline constexpr distance_t max_distance = std::numeric_limits<distance_t>::max();

/// @brief Local (same) distance
inline constexpr distance_t local_distance = 0;

/// @brief Adjacent distance (e.g., same NUMA node)
inline constexpr distance_t adjacent_distance = 10;

/// @brief Remote distance (e.g., different NUMA node, same machine)
inline constexpr distance_t remote_distance = 20;

/// @brief Network distance (e.g., different machine)
inline constexpr distance_t network_distance = 100;

// ============================================================================
// Proximity Matrix
// ============================================================================

/// @brief NxN proximity matrix for distance queries
class proximity_matrix {
public:
    /// @brief Default constructor (empty matrix)
    proximity_matrix() = default;

    /// @brief Construct with size
    /// @param n Number of elements
    explicit proximity_matrix(size_type n)
        : size_(n)
        , distances_(n * n, max_distance) {
        // Initialize diagonal to 0
        for (size_type i = 0; i < n; ++i) {
            set_distance(i, i, local_distance);
        }
    }

    /// @brief Get distance between two elements
    /// @param i First element index
    /// @param j Second element index
    /// @return Distance (lower is closer)
    [[nodiscard]] distance_t distance(size_type i, size_type j) const noexcept {
        if (i >= size_ || j >= size_) return max_distance;
        return distances_[i * size_ + j];
    }

    /// @brief Set distance between two elements (symmetric)
    /// @param i First element index
    /// @param j Second element index
    /// @param dist Distance value
    void set_distance(size_type i, size_type j, distance_t dist) {
        if (i >= size_ || j >= size_) return;
        distances_[i * size_ + j] = dist;
        distances_[j * size_ + i] = dist;  // Symmetric
    }

    /// @brief Get nearest neighbor to element i (excluding itself)
    /// @param i Element index
    /// @return Index of nearest neighbor
    [[nodiscard]] size_type nearest(size_type i) const noexcept {
        if (size_ <= 1 || i >= size_) return i;

        size_type best = (i == 0) ? 1 : 0;
        distance_t best_dist = distance(i, best);

        for (size_type j = 0; j < size_; ++j) {
            if (j == i) continue;
            distance_t d = distance(i, j);
            if (d < best_dist) {
                best_dist = d;
                best = j;
            }
        }

        return best;
    }

    /// @brief Get k nearest neighbors to element i
    /// @param i Element index
    /// @param k Number of neighbors
    /// @return Sorted vector of indices (nearest first)
    [[nodiscard]] std::vector<size_type> k_nearest(size_type i, size_type k) const {
        if (i >= size_) return {};

        // Build pairs of (distance, index)
        std::vector<std::pair<distance_t, size_type>> pairs;
        pairs.reserve(size_ - 1);

        for (size_type j = 0; j < size_; ++j) {
            if (j != i) {
                pairs.emplace_back(distance(i, j), j);
            }
        }

        // Partial sort to get k nearest
        size_type count = std::min(k, pairs.size());
        auto mid = pairs.begin() + static_cast<typename decltype(pairs)::difference_type>(count);
        std::partial_sort(pairs.begin(), mid, pairs.end());

        // Extract indices
        std::vector<size_type> result;
        result.reserve(count);
        for (size_type n = 0; n < count; ++n) {
            result.push_back(pairs[n].second);
        }

        return result;
    }

    /// @brief Get matrix size
    [[nodiscard]] size_type size() const noexcept { return size_; }

    /// @brief Check if empty
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// @brief Get all elements within a distance threshold
    /// @param i Element index
    /// @param threshold Maximum distance
    /// @return Vector of indices within threshold
    [[nodiscard]] std::vector<size_type> within_distance(
        size_type i, distance_t threshold) const {

        std::vector<size_type> result;
        for (size_type j = 0; j < size_; ++j) {
            if (distance(i, j) <= threshold) {
                result.push_back(j);
            }
        }
        return result;
    }

private:
    size_type size_ = 0;
    std::vector<distance_t> distances_;
};

// ============================================================================
// Proximity Builders
// ============================================================================

/// @brief Build CPU proximity matrix from NUMA distances
/// @param topo Hardware topology
/// @return Proximity matrix for CPUs
[[nodiscard]] inline proximity_matrix build_cpu_proximity(const hardware_topology& topo) {
    if (topo.total_cpus == 0) {
        return proximity_matrix{};
    }

    proximity_matrix prox(topo.total_cpus);

    // Simple model: CPUs on same NUMA node have adjacent distance,
    // CPUs on different NUMA nodes have remote distance
    for (const auto& node : topo.numa_nodes) {
        // Same node - adjacent
        for (size_type i = 0; i < node.cpu_ids.size(); ++i) {
            for (size_type j = i + 1; j < node.cpu_ids.size(); ++j) {
                prox.set_distance(node.cpu_ids[i], node.cpu_ids[j], adjacent_distance);
            }
        }
    }

    // Different nodes - remote
    for (size_type ni = 0; ni < topo.numa_nodes.size(); ++ni) {
        for (size_type nj = ni + 1; nj < topo.numa_nodes.size(); ++nj) {
            const auto& node_i = topo.numa_nodes[ni];
            const auto& node_j = topo.numa_nodes[nj];

            for (std::uint32_t cpu_i : node_i.cpu_ids) {
                for (std::uint32_t cpu_j : node_j.cpu_ids) {
                    prox.set_distance(cpu_i, cpu_j, remote_distance);
                }
            }
        }
    }

    return prox;
}

/// @brief Build GPU proximity matrix
/// @param topo Hardware topology
/// @return Proximity matrix for GPUs
[[nodiscard]] inline proximity_matrix build_gpu_proximity(const hardware_topology& topo) {
    if (topo.gpus.empty()) {
        return proximity_matrix{};
    }

    proximity_matrix prox(topo.gpus.size());

    // Model: GPUs on same NUMA node are adjacent, others are remote
    for (size_type i = 0; i < topo.gpus.size(); ++i) {
        for (size_type j = i + 1; j < topo.gpus.size(); ++j) {
            const auto& gpu_i = topo.gpus[i];
            const auto& gpu_j = topo.gpus[j];

            distance_t dist = remote_distance;
            if (gpu_i.numa_node && gpu_j.numa_node &&
                *gpu_i.numa_node == *gpu_j.numa_node) {
                dist = adjacent_distance;
            }

            prox.set_distance(i, j, dist);
        }
    }

    return prox;
}

/// @brief Build NUMA node proximity matrix
/// @param topo Hardware topology
/// @return Proximity matrix for NUMA nodes
[[nodiscard]] inline proximity_matrix build_numa_proximity(const hardware_topology& topo) {
    if (topo.numa_nodes.empty()) {
        return proximity_matrix{};
    }

    proximity_matrix prox(topo.numa_nodes.size());

    // Use real NUMA distances from sysfs if available
    if (!topo.numa_distances.empty() &&
        topo.numa_distances.size() == topo.numa_nodes.size()) {
        for (size_type i = 0; i < topo.numa_nodes.size(); ++i) {
            for (size_type j = i + 1; j < topo.numa_nodes.size(); ++j) {
                if (j < topo.numa_distances[i].size()) {
                    prox.set_distance(i, j,
                        static_cast<distance_t>(topo.numa_distances[i][j]));
                } else {
                    prox.set_distance(i, j, remote_distance);
                }
            }
        }
        return prox;
    }

    // Fallback: heuristic distances (all non-local are remote)
    for (size_type i = 0; i < topo.numa_nodes.size(); ++i) {
        for (size_type j = i + 1; j < topo.numa_nodes.size(); ++j) {
            prox.set_distance(i, j, remote_distance);
        }
    }

    return prox;
}

}  // namespace dtl::topology
