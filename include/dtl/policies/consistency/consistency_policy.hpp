// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file consistency_policy.hpp
/// @brief Base consistency policy concept and interface
/// @details Defines the consistency model for distributed operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/concepts.hpp>

namespace dtl {

// Note: consistency_policy_tag is defined in dtl/core/traits.hpp
// Note: is_consistency_policy_v is defined in dtl/core/traits.hpp
// Note: ConsistencyPolicy concept is defined in dtl/core/concepts.hpp

/// @brief Synchronization points available in the system
enum class sync_point {
    none,       ///< No synchronization
    barrier,    ///< Full barrier synchronization
    fence,      ///< Memory fence only
    epoch       ///< Epoch-based synchronization
};

/// @brief Memory ordering for distributed operations
enum class memory_ordering {
    relaxed,            ///< No ordering guarantees
    acquire,            ///< Acquire semantics
    release,            ///< Release semantics
    acquire_release,    ///< Both acquire and release
    sequential          ///< Sequential consistency
};

/// @brief Traits for consistency policy inspection
template <typename Policy>
struct consistency_traits {
    /// @brief Check if policy requires explicit barriers
    static constexpr bool requires_barrier = false;

    /// @brief Check if policy allows overlap of computation and communication
    static constexpr bool allows_overlap = true;

    /// @brief Get the default memory ordering
    static constexpr memory_ordering default_ordering = memory_ordering::relaxed;
};

}  // namespace dtl
