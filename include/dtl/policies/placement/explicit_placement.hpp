// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file explicit_placement.hpp
/// @brief User-controlled explicit placement policy
/// @details Allows per-element or per-partition placement specification.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/placement/placement_policy.hpp>

#include <functional>

namespace dtl {

/// @brief Explicit placement with user-defined location mapping
/// @tparam PlacementMap Type of the placement mapping function
/// @details Allows fine-grained control over memory placement by providing
///          a function that determines the location for each element or partition.
template <typename PlacementMap>
struct explicit_placement {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief The placement mapping function type
    using placement_function = PlacementMap;

    /// @brief Construct with placement mapping function
    /// @param map Function that maps index/rank to memory_location
    explicit explicit_placement(PlacementMap map) : placement_map_{std::move(map)} {}

    /// @brief Get the preferred memory location for an index
    /// @param idx The element index
    /// @param rank The rank
    /// @return The memory location for this element
    [[nodiscard]] memory_location location_for(index_t idx, rank_t rank) const {
        return placement_map_(idx, rank);
    }

    /// @brief Get the default preferred location (fallback)
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::host;  // Default fallback
    }

    /// @brief Check if memory is host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return true;  // May be, depending on placement
    }

    /// @brief Check if memory is device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return true;  // May be, depending on placement
    }

    /// @brief Check if this uses heterogeneous placement
    [[nodiscard]] static constexpr bool is_heterogeneous() noexcept {
        return true;
    }

private:
    PlacementMap placement_map_;
};

/// @brief Factory function to create explicit placement
/// @tparam Fn Placement function type
/// @param fn Function mapping (index, rank) -> memory_location
/// @return explicit_placement<Fn> instance
template <typename Fn>
[[nodiscard]] auto make_explicit_placement(Fn&& fn) {
    return explicit_placement<std::decay_t<Fn>>{std::forward<Fn>(fn)};
}

/// @brief Type-erased explicit placement using std::function
using dynamic_explicit_placement = explicit_placement<
    std::function<memory_location(index_t, rank_t)>>;

/// @brief Simple per-rank placement specification
struct per_rank_placement {
    /// @brief Policy category tag
    using policy_category = placement_policy_tag;

    /// @brief Construct with location for each rank
    /// @param locations Vector of memory_location, one per rank
    explicit per_rank_placement(std::vector<memory_location> locations)
        : locations_{std::move(locations)} {}

    /// @brief Get location for a rank
    [[nodiscard]] memory_location location_for_rank(rank_t rank) const {
        if (rank >= 0 && static_cast<size_type>(rank) < locations_.size()) {
            return locations_[static_cast<size_type>(rank)];
        }
        return memory_location::host;  // Default
    }

    /// @brief Get the default preferred location
    [[nodiscard]] static constexpr memory_location preferred_location() noexcept {
        return memory_location::host;
    }

    /// @brief Check if host accessible
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return true;
    }

    /// @brief Check if device accessible
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return true;
    }

private:
    std::vector<memory_location> locations_;
};

}  // namespace dtl
