// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file index.hpp
/// @brief Master include for DTL index utilities
/// @details Provides single-header access to all index types and operations.
/// @since 0.1.0

#pragma once

// Global index types
#include <dtl/index/global_index.hpp>

// Local index types
#include <dtl/index/local_index.hpp>

// Index translation utilities
#include <dtl/index/index_translation.hpp>

// Partition map
#include <dtl/index/partition_map.hpp>

namespace dtl {

// ============================================================================
// Index Module Summary
// ============================================================================
//
// The index module provides strongly-typed index abstractions for distributed
// containers. It distinguishes between global indices (logical position in the
// distributed address space) and local indices (position within a rank's
// partition), preventing common programming errors.
//
// ============================================================================
// Global Index
// ============================================================================
//
// global_index<T> represents a position in the global distributed space:
//
// @code
// dtl::global_index<> idx(42);  // Global position 42
// auto value = idx.value();      // Get raw value
// bool valid = idx.valid();      // Check if valid
//
// // Arithmetic
// ++idx;                         // Increment
// auto next = idx + 1;           // Offset
// auto diff = idx2 - idx1;       // Difference
//
// // Iteration
// for (auto i : dtl::make_global_range(0, 100)) {
//     // i is global_index<>
// }
// @endcode
//
// ============================================================================
// Local Index
// ============================================================================
//
// local_index<T> represents a position within a rank's local partition:
//
// @code
// dtl::local_index<> idx(10);   // Local position 10
// auto value = idx.value();      // Get raw value
//
// // With rank information
// dtl::ranked_local_index<> rli(10, 2);  // Local index 10 on rank 2
// rli.rank;   // 2
// rli.index;  // local_index<>(10)
// @endcode
//
// ============================================================================
// Multi-Dimensional Indices
// ============================================================================
//
// md_global_index<N> and md_local_index<N> for ND containers:
//
// @code
// dtl::md_global_index<3> idx(10, 20, 30);  // 3D index
// auto x = idx[0];  // 10
// auto y = idx[1];  // 20
// auto z = idx[2];  // 30
//
// // Linearization
// size_t extents[3] = {100, 50, 40};
// auto linear = idx.linearize(extents);
//
// // From linear
// auto idx2 = dtl::md_global_index<3>::from_linear(linear, extents);
// @endcode
//
// ============================================================================
// Index Ranges
// ============================================================================
//
// Ranges support iteration and membership testing:
//
// @code
// auto range = dtl::make_global_range(0, 100);
// range.size();         // 100
// range.contains(50);   // true
// range.contains(150);  // false
//
// for (auto idx : range) {
//     // Process each global index
// }
// @endcode
//
// ============================================================================
// Index Translation
// ============================================================================
//
// Convert between global and local indices based on partition policy:
//
// Block Partition (contiguous chunks):
// @code
// using namespace dtl::block_partition_translation;
//
// // Which rank owns global index 42?
// rank_t owner_rank = owner(42, global_size, num_ranks);
//
// // Is global index 42 local to this rank?
// bool local = is_local(42, global_size, num_ranks, my_rank);
//
// // Convert global to local
// auto local_idx = to_local(42, global_size, num_ranks, my_rank);
//
// // Convert local to global
// auto global_idx = to_global(10, global_size, num_ranks, my_rank);
//
// // Get local size for a rank
// auto size = local_size(global_size, num_ranks, my_rank);
//
// // Get global range owned by a rank
// auto range = owned_range(global_size, num_ranks, my_rank);
// @endcode
//
// Cyclic Partition (round-robin):
// @code
// using namespace dtl::cyclic_partition_translation;
//
// rank_t owner_rank = owner(42, num_ranks);  // Note: no global_size needed
// auto local_idx = to_local(42, num_ranks, my_rank);
// auto global_idx = to_global(10, num_ranks, my_rank);
// @endcode
//
// ============================================================================
// Index Translator Class
// ============================================================================
//
// For repeated translations, use the translator class:
//
// @code
// auto translator = dtl::make_block_translator(global_size, num_ranks, my_rank);
//
// auto owner = translator.owner(global_index<>(42));
// bool local = translator.is_local(global_index<>(42));
// auto local_idx = translator.to_local(global_index<>(42));
// auto global_idx = translator.to_global(local_index<>(10));
// auto my_size = translator.local_size();
// auto their_size = translator.local_size(other_rank);
// @endcode
//
// ============================================================================
// Type Safety Benefits
// ============================================================================
//
// Strong typing prevents common errors:
//
// @code
// void process_local(dtl::local_index<> idx);
// void process_global(dtl::global_index<> idx);
//
// dtl::local_index<> li(10);
// dtl::global_index<> gi(100);
//
// process_local(li);   // OK
// process_global(gi);  // OK
// process_local(gi);   // Compile error! Type mismatch
// process_global(li);  // Compile error! Type mismatch
// @endcode
//
// ============================================================================
// Usage Examples
// ============================================================================
//
// @code
// #include <dtl/index/index.hpp>
//
// // Distributed iteration pattern
// void process_distributed_data(
//     std::vector<double>& local_data,
//     size_t global_size,
//     int num_ranks,
//     int my_rank) {
//
//     auto translator = dtl::make_block_translator(global_size, num_ranks, my_rank);
//
//     // Process only local data
//     for (size_t i = 0; i < local_data.size(); ++i) {
//         auto local_idx = dtl::make_local_index(i);
//         auto global_idx = translator.to_global(local_idx);
//
//         // global_idx is the logical position in the distributed array
//         local_data[i] = compute(global_idx.value());
//     }
// }
//
// // Find owner of specific indices
// void print_owners(size_t global_size, int num_ranks) {
//     using namespace dtl::block_partition_translation;
//
//     for (size_t i = 0; i < global_size; ++i) {
//         auto owner_rank = owner(i, global_size, num_ranks);
//         std::cout << "Index " << i << " owned by rank " << owner_rank << "\n";
//     }
// }
//
// // Multi-dimensional indexing
// void process_3d_tensor(
//     double* local_data,
//     size_t global_extents[3],
//     size_t local_extents[3],
//     size_t local_offset[3]) {
//
//     for (size_t i = 0; i < local_extents[0]; ++i) {
//         for (size_t j = 0; j < local_extents[1]; ++j) {
//             for (size_t k = 0; k < local_extents[2]; ++k) {
//                 dtl::md_local_index<3> local_idx(i, j, k);
//                 dtl::md_global_index<3> global_idx(
//                     i + local_offset[0],
//                     j + local_offset[1],
//                     k + local_offset[2]);
//
//                 auto linear_local = local_idx.linearize(local_extents);
//                 local_data[linear_local] = compute(global_idx);
//             }
//         }
//     }
// }
// @endcode
//
// ============================================================================
// Integration with Containers
// ============================================================================
//
// DTL containers use these index types internally:
//
// @code
// dtl::distributed_vector<double> vec(1000, ctx);
//
// // Local view uses local_index internally
// auto local = vec.local_view();
// for (dtl::local_index<> i(0); i < local.size(); ++i) {
//     local[i] = compute(i.value());
// }
//
// // Global view uses global_index and returns remote_ref for remote elements
// auto global = vec.global_view();
// for (dtl::global_index<> i(0); i < vec.size(); ++i) {
//     // global[i] returns remote_ref<double> - explicit access required
// }
// @endcode
//
// ============================================================================

}  // namespace dtl
