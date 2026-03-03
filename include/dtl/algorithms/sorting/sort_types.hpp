// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file sort_types.hpp
/// @brief Common types for distributed sort algorithms
/// @since 0.1.0

#pragma once

#include <dtl/core/types.hpp>

namespace dtl {

/// @brief Configuration for distributed sort algorithm
struct distributed_sort_config {
    /// @brief Number of samples per rank (for pivot selection)
    size_type oversampling_factor = 3;

    /// @brief Whether to use parallel local sort
    bool use_parallel_local_sort = true;

    /// @brief Whether to use parallel merge after exchange
    bool use_parallel_merge = true;
};

/// @brief Result type for distributed sort operations
struct distributed_sort_result {
    /// @brief Whether the operation succeeded
    bool success = true;

    /// @brief Number of elements sent to other ranks
    size_type elements_sent = 0;

    /// @brief Number of elements received from other ranks
    size_type elements_received = 0;
};

}  // namespace dtl
