// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file node_role.hpp
/// @brief Role enumeration and role descriptors for MPMD patterns
/// @details Defines roles for organizing ranks in MPMD programs.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <functional>

namespace dtl::mpmd {

// ============================================================================
// Node Role Enumeration
// ============================================================================

/// @brief Predefined role types for MPMD programs
/// @details These are common roles; applications can define custom roles
///          using role_id values above custom_base.
enum class node_role : std::uint32_t {
    /// @brief Worker role - performs main computation
    worker = 0,

    /// @brief Coordinator role - orchestrates work distribution
    coordinator = 1,

    /// @brief I/O handler role - manages file/network I/O
    io_handler = 2,

    /// @brief Aggregator role - collects and reduces results
    aggregator = 3,

    /// @brief Monitor role - tracks progress and health
    monitor = 4,

    /// @brief Checkpoint handler role - manages fault tolerance
    checkpoint_handler = 5,

    /// @brief Load balancer role - redistributes work
    load_balancer = 6,

    /// @brief Gateway role - bridges between groups
    gateway = 7,

    /// @brief Any role - matches any role (for queries)
    any = 0xFFFFFFFE,

    /// @brief Undefined/unassigned role
    undefined = 0xFFFFFFFF,

    /// @brief Base value for custom application-defined roles
    custom_base = 0x10000
};

/// @brief Type alias for role identifiers
using role_id = std::uint32_t;

// ============================================================================
// Role Properties
// ============================================================================

/// @brief Properties describing a role's capabilities and requirements
struct role_properties {
    /// @brief Human-readable role name
    std::string name{};

    /// @brief Role description
    std::string description{};

    /// @brief Minimum number of ranks required for this role
    size_type min_ranks = 0;

    /// @brief Maximum number of ranks for this role (0 = unlimited)
    size_type max_ranks = 0;

    /// @brief Whether this role is required (at least min_ranks must exist)
    bool required = false;

    /// @brief Whether ranks with this role should be on separate nodes
    bool prefer_separate_nodes = false;

    /// @brief Whether this role needs GPU access
    bool requires_gpu = false;

    /// @brief Memory requirement hint in bytes (0 = no preference)
    size_type memory_hint = 0;
};

// ============================================================================
// Role Traits
// ============================================================================

/// @brief Get the name of a predefined role
/// @param role The role to name
/// @return String view of the role name
[[nodiscard]] constexpr std::string_view role_name(node_role role) noexcept {
    switch (role) {
        case node_role::worker: return "worker";
        case node_role::coordinator: return "coordinator";
        case node_role::io_handler: return "io_handler";
        case node_role::aggregator: return "aggregator";
        case node_role::monitor: return "monitor";
        case node_role::checkpoint_handler: return "checkpoint_handler";
        case node_role::load_balancer: return "load_balancer";
        case node_role::gateway: return "gateway";
        case node_role::any: return "any";
        case node_role::undefined: return "undefined";
        default: return "custom";
    }
}

/// @brief Check if a role is a predefined role
/// @param role The role to check
/// @return true if predefined, false if custom
[[nodiscard]] constexpr bool is_predefined_role(node_role role) noexcept {
    return static_cast<role_id>(role) < static_cast<role_id>(node_role::custom_base);
}

/// @brief Check if a role is a custom role
/// @param role The role to check
/// @return true if custom, false if predefined
[[nodiscard]] constexpr bool is_custom_role(node_role role) noexcept {
    auto id = static_cast<role_id>(role);
    return id >= static_cast<role_id>(node_role::custom_base) &&
           id < static_cast<role_id>(node_role::any);
}

/// @brief Create a custom role from an ID
/// @param id Custom role identifier (added to custom_base)
/// @return The custom role
[[nodiscard]] constexpr node_role make_custom_role(role_id id) noexcept {
    return static_cast<node_role>(static_cast<role_id>(node_role::custom_base) + id);
}

/// @brief Get the custom ID from a custom role
/// @param role The custom role
/// @return The custom ID portion
[[nodiscard]] constexpr role_id custom_role_id(node_role role) noexcept {
    return static_cast<role_id>(role) - static_cast<role_id>(node_role::custom_base);
}

// ============================================================================
// Role Descriptor
// ============================================================================

/// @brief Complete descriptor for a role including assignment function
class role_descriptor {
public:
    using assignment_function = std::function<bool(rank_t rank, rank_t num_ranks)>;

    /// @brief Construct with role and properties
    /// @param role The role being described
    /// @param props Role properties
    role_descriptor(node_role role, role_properties props)
        : role_(role)
        , properties_(std::move(props))
        , assigner_([](rank_t, rank_t) { return false; }) {}

    /// @brief Construct with role, properties, and assignment function
    /// @param role The role being described
    /// @param props Role properties
    /// @param assigner Function that returns true if rank should have this role
    role_descriptor(node_role role, role_properties props, assignment_function assigner)
        : role_(role)
        , properties_(std::move(props))
        , assigner_(std::move(assigner)) {}

    /// @brief Get the role
    [[nodiscard]] node_role role() const noexcept { return role_; }

    /// @brief Get the properties
    [[nodiscard]] const role_properties& properties() const noexcept { return properties_; }

    /// @brief Check if a rank should be assigned this role
    /// @param rank The rank to check
    /// @param num_ranks Total number of ranks
    /// @return true if the rank should have this role
    [[nodiscard]] bool should_assign(rank_t rank, rank_t num_ranks) const {
        return assigner_(rank, num_ranks);
    }

private:
    node_role role_;
    role_properties properties_;
    assignment_function assigner_;
};

// ============================================================================
// Predefined Assignment Strategies
// ============================================================================

namespace role_assignment {

/// @brief Assign role to rank 0 only
/// @return Assignment function
[[nodiscard]] inline auto first_rank_only() {
    return [](rank_t rank, rank_t) { return rank == 0; };
}

/// @brief Assign role to the last rank only
/// @return Assignment function
[[nodiscard]] inline auto last_rank_only() {
    return [](rank_t rank, rank_t num_ranks) { return rank == num_ranks - 1; };
}

/// @brief Assign role to all ranks
/// @return Assignment function
[[nodiscard]] inline auto all_ranks() {
    return [](rank_t, rank_t) { return true; };
}

/// @brief Assign role to no ranks (explicit exclusion)
/// @return Assignment function
[[nodiscard]] inline auto no_ranks() {
    return [](rank_t, rank_t) { return false; };
}

/// @brief Assign role to specific ranks
/// @param ranks List of ranks to assign
/// @return Assignment function
[[nodiscard]] inline auto specific_ranks(std::vector<rank_t> ranks) {
    return [r = std::move(ranks)](rank_t rank, rank_t) {
        return std::find(r.begin(), r.end(), rank) != r.end();
    };
}

/// @brief Assign role to ranks in a range
/// @param start First rank (inclusive)
/// @param end Last rank (exclusive)
/// @return Assignment function
[[nodiscard]] inline auto rank_range(rank_t start, rank_t end) {
    return [start, end](rank_t rank, rank_t) {
        return rank >= start && rank < end;
    };
}

/// @brief Assign role to first N ranks
/// @param count Number of ranks
/// @return Assignment function
[[nodiscard]] inline auto first_n_ranks(rank_t count) {
    return [count](rank_t rank, rank_t) { return rank < count; };
}

/// @brief Assign role to last N ranks
/// @param count Number of ranks
/// @return Assignment function
[[nodiscard]] inline auto last_n_ranks(rank_t count) {
    return [count](rank_t rank, rank_t num_ranks) {
        return rank >= num_ranks - count;
    };
}

/// @brief Assign role to every Nth rank
/// @param n Stride
/// @param offset Starting offset
/// @return Assignment function
[[nodiscard]] inline auto every_nth_rank(rank_t n, rank_t offset = 0) {
    return [n, offset](rank_t rank, rank_t) {
        return (rank - offset) % n == 0 && rank >= offset;
    };
}

/// @brief Assign role based on a fraction of total ranks
/// @param fraction Fraction of ranks (0.0 to 1.0)
/// @return Assignment function
[[nodiscard]] inline auto fraction_of_ranks(double fraction) {
    return [fraction](rank_t rank, rank_t num_ranks) {
        return rank < static_cast<rank_t>(num_ranks * fraction);
    };
}

}  // namespace role_assignment

}  // namespace dtl::mpmd
