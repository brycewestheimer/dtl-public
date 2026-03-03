// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file iterators.hpp
/// @brief Master include for all DTL iterators
/// @details Provides single-header access to all iterator types.
/// @since 0.1.0

#pragma once

// Core iterator types
#include <dtl/iterators/local_iterator.hpp>
#include <dtl/iterators/global_iterator.hpp>
#include <dtl/iterators/device_iterator.hpp>

// Iterator traits and utilities
#include <dtl/iterators/iterator_traits.hpp>

namespace dtl {

// ============================================================================
// Iterator Summary
// ============================================================================
//
// DTL provides three main iterator types:
//
// 1. local_iterator<Container>
//    - Random access iterator for local partition
//    - Never communicates
//    - STL-compatible (use with std::sort, etc.)
//    - Obtained from: container.local_view().begin()
//
// 2. global_iterator<Container>
//    - Forward iterator for global indexing
//    - Dereference returns remote_ref<T> (may communicate)
//    - NOT recommended for tight loops
//    - Obtained from: container.global_view().begin()
//
// 3. device_iterator<T>
//    - Random access iterator for GPU memory
//    - Host/device callable
//    - Use in kernels for parallel access
//    - Obtained from: container.device_begin()
//
// Recommended patterns:
//
// For local processing (fastest):
//   auto local = container.local_view();
//   std::for_each(local.begin(), local.end(), f);
//
// For distributed algorithms (bulk operations):
//   for (auto& segment : container.segmented_view()) {
//       if (segment.is_local()) {
//           std::for_each(segment.begin(), segment.end(), f);
//       }
//   }
//
// For global random access (sparse, careful!):
//   auto ref = container.global_view()[global_idx];
//   if (!ref.is_local()) {
//       auto val = ref.get().value();  // Explicit remote access
//   }
//
// ============================================================================

}  // namespace dtl
