// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file placement_policy.hpp
/// @brief Base placement policy concept and interface
/// @details Defines the placement policy tag and requirements for
///          memory placement (host, device, unified).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

namespace dtl {

// =============================================================================
// Placement Policy Tag
// =============================================================================

// placement_policy_tag is defined in core/traits.hpp

// =============================================================================
// Memory Location Enumeration
// =============================================================================

/// @brief Enumeration of possible memory locations
enum class memory_location {
    host,           ///< CPU/host memory
    device,         ///< GPU/accelerator device memory
    unified,        ///< Unified/managed memory accessible from host and device
    remote,         ///< Remote memory (on another rank)
    unknown         ///< Unknown or invalid location
};

// =============================================================================
// Placement Policy Concept Requirements
// =============================================================================

/// @brief Concept defining requirements for placement policies
/// @details A placement policy must provide:
///          - policy_category type alias = placement_policy_tag
///          - preferred_location() -> memory_location
///          - is_host_accessible() -> bool
///          - is_device_accessible() -> bool
template <typename P>
concept PlacementPolicyConcept =
    std::same_as<typename P::policy_category, placement_policy_tag> &&
    requires {
        { P::preferred_location() } -> std::convertible_to<memory_location>;
        { P::is_host_accessible() } -> std::convertible_to<bool>;
        { P::is_device_accessible() } -> std::convertible_to<bool>;
    };

// =============================================================================
// Default Placement Policy
// =============================================================================

/// @brief Forward declaration of host_only
struct host_only;

/// @brief Default placement policy is host_only
using default_placement_policy = host_only;

}  // namespace dtl
