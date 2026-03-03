// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file sample_sort_detail.hpp
/// @brief Helper functions for distributed sample sort algorithm
/// @details Provides local-only helper utilities for sample sort:
///          sampling, pivot selection, bucket partitioning, and
///          alltoallv parameter computation. These are building blocks
///          for the full distributed sample sort (which requires MPI).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace dtl::detail {

// =============================================================================
// Local Sampling
// =============================================================================

/// @brief Sample evenly-spaced elements from a sorted range
/// @tparam Iterator Random access iterator type
/// @tparam Compare Comparison function type
/// @param first Iterator to the beginning of the range
/// @param last Iterator past the end of the range
/// @param count Number of samples to select
/// @param comp Comparison function (unused but part of API for consistency)
/// @return Vector of sampled elements, evenly spaced across the range
/// @note The input range does not need to be sorted for sampling,
///       but sorted input produces better pivot quality.
template <typename Iterator, typename Compare>
std::vector<typename std::iterator_traits<Iterator>::value_type>
sample_local(Iterator first, Iterator last, size_type count, [[maybe_unused]] Compare comp) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    std::vector<T> samples;
    auto n = std::distance(first, last);
    if (n == 0 || count == 0) return samples;
    samples.reserve(count);
    for (size_type i = 0; i < count; ++i) {
        auto idx = static_cast<decltype(n)>((i * static_cast<size_type>(n)) / count);
        samples.push_back(*(first + idx));
    }
    return samples;
}

// =============================================================================
// Pivot Selection
// =============================================================================

/// @brief Select p-1 evenly-spaced pivots from a gathered sample array
/// @tparam T Element type
/// @tparam Compare Comparison function type
/// @param all_samples Combined samples from all ranks
/// @param num_ranks Number of ranks participating in the sort
/// @param comp Comparison function for sorting samples
/// @return Vector of p-1 pivots in sorted order
/// @details The pivot selection algorithm:
///          1. Sort all gathered samples
///          2. Pick p-1 evenly-spaced elements as pivots
///          These pivots define bucket boundaries for redistribution.
template <typename T, typename Compare>
std::vector<T> select_pivots(const std::vector<T>& all_samples,
                              rank_t num_ranks, Compare comp) {
    if (num_ranks <= 1 || all_samples.empty()) return {};

    // Sort all samples to find good split points
    auto sorted_samples = all_samples;
    std::sort(sorted_samples.begin(), sorted_samples.end(), comp);

    // Pick p-1 pivots evenly spaced through the sorted samples
    std::vector<T> pivots;
    pivots.reserve(static_cast<size_type>(num_ranks - 1));
    size_type n = sorted_samples.size();
    for (rank_t i = 1; i < num_ranks; ++i) {
        size_type idx = (static_cast<size_type>(i) * n) / static_cast<size_type>(num_ranks);
        if (idx >= n) idx = n - 1;
        pivots.push_back(sorted_samples[idx]);
    }
    return pivots;
}

// =============================================================================
// Bucket Partitioning
// =============================================================================

/// @brief Partition local data into p buckets based on pivot values
/// @tparam Iterator Random access iterator type
/// @tparam T Element type (must match iterator value type)
/// @tparam Compare Comparison function type
/// @param first Iterator to the beginning of the local data
/// @param last Iterator past the end of the local data
/// @param pivots p-1 pivot values defining bucket boundaries
/// @param comp Comparison function
/// @return Vector of p buckets, where bucket[i] holds elements
///         in [pivot[i-1], pivot[i]) (with boundary cases for first/last bucket)
/// @details Each element is placed in the bucket corresponding to its
///          position relative to the pivot array using lower_bound.
template <typename Iterator, typename T, typename Compare>
std::vector<std::vector<T>> partition_by_pivots(
    Iterator first, Iterator last,
    const std::vector<T>& pivots, Compare comp) {
    rank_t num_buckets = static_cast<rank_t>(pivots.size()) + 1;
    std::vector<std::vector<T>> buckets(static_cast<size_type>(num_buckets));
    for (auto it = first; it != last; ++it) {
        auto pivot_it = std::lower_bound(pivots.begin(), pivots.end(), *it, comp);
        rank_t bucket_idx = static_cast<rank_t>(std::distance(pivots.begin(), pivot_it));
        buckets[static_cast<size_type>(bucket_idx)].push_back(*it);
    }
    return buckets;
}

// =============================================================================
// Alltoallv Parameter Computation
// =============================================================================

/// @brief Parameters for MPI_Alltoallv: counts and displacements
struct alltoallv_params {
    std::vector<int> send_counts;   ///< Number of elements to send to each rank
    std::vector<int> send_displs;   ///< Displacement (offset) for data sent to each rank
    std::vector<int> recv_counts;   ///< Number of elements to receive from each rank
    std::vector<int> recv_displs;   ///< Displacement (offset) for data received from each rank
    size_type total_recv = 0;       ///< Total number of elements to receive
};

/// @brief Compute alltoallv send parameters from bucket contents
/// @tparam T Element type
/// @param buckets Vector of buckets (one per target rank)
/// @return alltoallv_params with send_counts and send_displs filled in;
///         recv_counts and recv_displs are initialized to zero
///         (to be filled after MPI_Alltoall of send_counts)
template <typename T>
alltoallv_params compute_alltoallv_params(
    const std::vector<std::vector<T>>& buckets) {
    alltoallv_params params;
    size_type num_ranks = buckets.size();
    params.send_counts.resize(num_ranks);
    params.send_displs.resize(num_ranks);
    int disp = 0;
    for (size_type i = 0; i < num_ranks; ++i) {
        params.send_counts[i] = static_cast<int>(buckets[i].size());
        params.send_displs[i] = disp;
        disp += params.send_counts[i];
    }
    // recv_counts and recv_displs are filled after alltoall of send_counts
    params.recv_counts.resize(num_ranks, 0);
    params.recv_displs.resize(num_ranks, 0);
    return params;
}

// =============================================================================
// Bucket Flattening
// =============================================================================

/// @brief Flatten buckets into a single contiguous send buffer
/// @tparam T Element type
/// @param buckets Vector of buckets to flatten
/// @return Contiguous vector with all bucket contents in order
/// @details Elements are concatenated in bucket index order, matching
///         the layout expected by MPI_Alltoallv with the computed displacements.
template <typename T>
std::vector<T> flatten_buckets(const std::vector<std::vector<T>>& buckets) {
    size_type total = 0;
    for (const auto& b : buckets) total += b.size();
    std::vector<T> flat;
    flat.reserve(total);
    for (const auto& b : buckets) {
        flat.insert(flat.end(), b.begin(), b.end());
    }
    return flat;
}

}  // namespace dtl::detail
