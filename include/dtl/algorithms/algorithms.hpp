// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file algorithms.hpp
/// @brief Master include for all DTL distributed algorithms
/// @details Provides single-header access to all distributed algorithm types.
/// @since 0.1.0

#pragma once

// Algorithm infrastructure
#include <dtl/algorithms/algorithm_traits.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>

// Non-modifying algorithms
#include <dtl/algorithms/non_modifying/for_each.hpp>
#include <dtl/algorithms/non_modifying/find.hpp>
#include <dtl/algorithms/non_modifying/count.hpp>
#include <dtl/algorithms/non_modifying/predicates.hpp>

// Modifying algorithms
#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/algorithms/modifying/copy.hpp>
#include <dtl/algorithms/modifying/fill.hpp>
#include <dtl/algorithms/modifying/replace.hpp>
#include <dtl/algorithms/modifying/adjacent_difference.hpp>
#include <dtl/algorithms/modifying/iota.hpp>
#include <dtl/algorithms/modifying/rotate.hpp>
#include <dtl/algorithms/modifying/partition_algorithm.hpp>

// Reduction algorithms
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/reductions/transform_reduce.hpp>
#include <dtl/algorithms/reductions/minmax.hpp>
#include <dtl/algorithms/reductions/accumulate.hpp>
#include <dtl/algorithms/reductions/scan.hpp>  // V1.1: prefix scans
#include <dtl/algorithms/reductions/inner_product.hpp>

// Sorting algorithms
#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/algorithms/sorting/partial_sort.hpp>
#include <dtl/algorithms/sorting/nth_element.hpp>
#include <dtl/algorithms/sorting/unique.hpp>

// Canonical semantic domain namespaces (depends on algorithm declarations above)
#include <dtl/algorithms/domain_namespaces.hpp>

namespace dtl {

// ============================================================================
// Algorithm Summary
// ============================================================================
//
// DTL provides distributed versions of common algorithms. Each algorithm
// comes in several variants:
//
// 1. Standard (collective): Operates on entire distributed container
//    - Requires all ranks to participate
//    - Result is consistent across all ranks
//    - Example: dtl::global_reduce(vec, 0, std::plus<>{}, comm)
//
// 2. Local (no communication): Operates on local partition only
//    - Does not require collective participation
//    - Result is rank-local
//    - Example: dtl::local_reduce(vec, 0, std::plus<>{})
//
// 3. Async: Non-blocking version returning future
//    - Returns immediately, completes in background
//    - Use .get() or .then() to access result
//    - Example: auto fut = dtl::async_reduce(vec, 0, std::plus<>{})
//
// ============================================================================
// Execution Policies
// ============================================================================
//
// Most algorithms accept an execution policy as first parameter:
//
// - dtl::seq{}        - Sequential execution (single-threaded local work)
// - dtl::par{}        - Parallel execution (multi-threaded local work)
// - dtl::async{} - Asynchronous execution (returns future)
//
// Example:
//   dtl::for_each(dtl::par{}, vec, f);  // Parallel local processing
//   dtl::for_each(dtl::seq{}, vec, f);  // Sequential local processing
//
// ============================================================================
// Recommended Patterns
// ============================================================================
//
// For local processing (no communication):
//   auto local = vec.local_view();
//   std::for_each(local.begin(), local.end(), f);  // Use STL directly
//
// For distributed iteration (segmented):
//   dtl::segmented_for_each(dtl::par{}, vec, f);
//
// For reductions:
//   auto sum = dtl::global_reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);
//
// For sorting:
//   dtl::sort(dtl::par{}, vec);  // Global sort
//   dtl::local_sort(vec);        // Local only
//
// ============================================================================
// Collective vs Non-Collective
// ============================================================================
//
// COLLECTIVE algorithms require all ranks to participate:
// - reduce, transform_reduce, accumulate
// - min_element, max_element, minmax_element
// - find (global), count (global)
// - sort, unique
// - all_of, any_of, none_of (global)
//
// NON-COLLECTIVE algorithms (local_* variants) do not communicate:
// - local_reduce, local_transform_reduce
// - local_find, local_count
// - local_sort, local_unique
// - local_all_of, local_any_of, local_none_of
// - for_each, transform, fill, replace (always local unless redistributing)
//
// ============================================================================

}  // namespace dtl
