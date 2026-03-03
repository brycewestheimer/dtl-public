// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file role_manager.hpp
/// @brief Role assignment and group management for MPMD patterns
/// @details Manages role descriptors, group creation, and membership.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/mpmd/node_role.hpp>
#include <dtl/mpmd/rank_group.hpp>
#include <dtl/communication/communicator_base.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dtl::mpmd {

// Import communicator_base from parent dtl:: namespace
using dtl::communicator_base;

// ============================================================================
// Role Manager Configuration
// ============================================================================

/// @brief Configuration for role manager initialization
struct role_manager_config {
    /// @brief Whether to create sub-communicators for groups
    bool create_communicators = true;

    /// @brief Whether to allow multiple roles per rank
    bool allow_multiple_roles = true;

    /// @brief Whether to validate role requirements on finalize
    bool validate_requirements = true;

    /// @brief Default role for ranks without explicit assignment
    node_role default_role = node_role::worker;
};

// ============================================================================
// Role Manager
// ============================================================================

/// @brief Manages role assignments and rank groups for MPMD patterns
/// @details The role_manager coordinates the assignment of roles to ranks
///          and manages the creation of rank groups with associated
///          sub-communicators. It provides a central registry for
///          MPMD program structure.
///
/// Usage pattern:
/// 1. Register role descriptors with register_role()
/// 2. Call initialize() to perform role assignment
/// 3. Query groups with get_group() or get_my_groups()
/// 4. Use group communicators for intra-group operations
class role_manager {
public:
    /// @brief Default constructor
    role_manager() = default;

    /// @brief Construct with configuration
    /// @param config Manager configuration
    explicit role_manager(role_manager_config config)
        : config_(std::move(config)) {}

    /// @brief Destructor
    ~role_manager() = default;

    // Non-copyable
    role_manager(const role_manager&) = delete;
    role_manager& operator=(const role_manager&) = delete;

    // Movable
    role_manager(role_manager&&) = default;
    role_manager& operator=(role_manager&&) = default;

    // ------------------------------------------------------------------------
    // Role Registration
    // ------------------------------------------------------------------------

    /// @brief Register a role descriptor
    /// @param descriptor The role descriptor
    /// @return Success or error
    result<void> register_role(role_descriptor descriptor) {
        if (initialized_) {
            return make_error<void>(status_code::invalid_state,
                                    "Cannot register roles after initialization");
        }
        descriptors_.push_back(std::move(descriptor));
        return {};
    }

    /// @brief Register a simple role with assignment function
    /// @param role The role to register
    /// @param name Role name
    /// @param assigner Function determining role assignment
    /// @return Success or error
    result<void> register_role(
        node_role role,
        std::string name,
        std::function<bool(rank_t, rank_t)> assigner) {
        role_properties props;
        props.name = std::move(name);
        return register_role(role_descriptor(role, std::move(props), std::move(assigner)));
    }

    /// @brief Get all registered role descriptors
    [[nodiscard]] const std::vector<role_descriptor>& descriptors() const noexcept {
        return descriptors_;
    }

    // ------------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------------

    /// @brief Initialize role assignments and create groups
    /// @param world_comm The world communicator
    /// @return Success or error
    /// @note This is a collective operation. All descriptors must be
    ///       deterministic functions of (rank, size) so that every rank
    ///       can independently compute the full role assignment.
    result<void> initialize(communicator_base& world_comm) {
        if (initialized_) {
            return make_error<void>(status_code::invalid_state,
                                    "Already initialized");
        }

        world_comm_ = &world_comm;
        const rank_t my_rank = world_comm.rank();
        const rank_t world_size = world_comm.size();

        // Step 1: Determine which roles each rank gets by evaluating
        //         all descriptors for all ranks. Descriptors are deterministic
        //         functions of (rank, size), so every rank computes the same
        //         result without communication.

        // For each descriptor, collect the set of assigned world ranks.
        // Also track which roles this rank (my_rank) is assigned.
        struct role_members {
            node_role role;
            std::string name;
            std::vector<rank_t> members;
        };

        std::vector<role_members> role_member_list;
        role_member_list.reserve(descriptors_.size());

        // Track which roles this rank has been assigned
        std::vector<bool> my_role_assigned(descriptors_.size(), false);

        for (size_type d = 0; d < descriptors_.size(); ++d) {
            const auto& desc = descriptors_[d];
            role_members rm;
            rm.role = desc.role();
            rm.name = desc.properties().name;

            for (rank_t r = 0; r < world_size; ++r) {
                if (desc.should_assign(r, world_size)) {
                    rm.members.push_back(r);
                    if (r == my_rank) {
                        my_role_assigned[d] = true;
                    }
                }
            }

            role_member_list.push_back(std::move(rm));
        }

        // Step 2: Check if this rank was assigned any role at all.
        //         If not, and a default role is configured, assign it.
        bool has_any_role = false;
        for (size_type d = 0; d < descriptors_.size(); ++d) {
            if (my_role_assigned[d]) {
                has_any_role = true;
                break;
            }
        }

        if (!has_any_role && config_.default_role != node_role::undefined) {
            // Check if we already have a group for the default role
            bool found_default_group = false;
            for (auto& rm : role_member_list) {
                if (rm.role == config_.default_role) {
                    // Add this rank to the existing default role group
                    rm.members.push_back(my_rank);
                    // Sort to maintain order
                    std::sort(rm.members.begin(), rm.members.end());
                    found_default_group = true;
                    break;
                }
            }
            if (!found_default_group) {
                // Note: In a fully distributed setting, all ranks without
                // any role would need the same default group. Since descriptors
                // are deterministic, we compute default assignment for all ranks.
                role_members rm;
                rm.role = config_.default_role;
                rm.name = std::string(role_name(config_.default_role));
                // Add all unassigned ranks to the default role
                for (rank_t r = 0; r < world_size; ++r) {
                    bool r_has_role = false;
                    for (size_type d = 0; d < descriptors_.size(); ++d) {
                        if (descriptors_[d].should_assign(r, world_size)) {
                            r_has_role = true;
                            break;
                        }
                    }
                    if (!r_has_role) {
                        rm.members.push_back(r);
                    }
                }
                role_member_list.push_back(std::move(rm));
            }
        }

        // Step 3: Validate role requirements if configured
        if (config_.validate_requirements) {
            for (size_type d = 0; d < descriptors_.size(); ++d) {
                const auto& desc = descriptors_[d];
                const auto& props = desc.properties();
                const auto& members = role_member_list[d].members;
                size_type count = members.size();

                if (props.min_ranks > 0 && count < props.min_ranks) {
                    return make_error<void>(status_code::precondition_failed,
                        "Role '" + props.name + "' requires at least " +
                        std::to_string(props.min_ranks) + " ranks but got " +
                        std::to_string(count));
                }

                if (props.max_ranks > 0 && count > props.max_ranks) {
                    return make_error<void>(status_code::precondition_failed,
                        "Role '" + props.name + "' allows at most " +
                        std::to_string(props.max_ranks) + " ranks but got " +
                        std::to_string(count));
                }
            }
        }

        // Step 4: Create rank_group objects for each unique role with members.
        //         Merge groups with the same role into one group.
        std::map<node_role, size_type> role_index;  // role -> index into groups_

        for (const auto& rm : role_member_list) {
            if (rm.members.empty()) {
                continue;  // Skip roles with no members
            }

            auto it = role_index.find(rm.role);
            if (it != role_index.end()) {
                // Merge into existing group
                auto& existing = groups_[it->second];
                for (rank_t r : rm.members) {
                    if (!existing.contains(r)) {
                        existing.add_member(r);
                    }
                }
            } else {
                auto id = static_cast<rank_group::group_id>(groups_.size());
                groups_.emplace_back(id, rm.role, rm.name);
                auto& group = groups_.back();
                for (rank_t r : rm.members) {
                    group.add_member(r);
                }
                role_index[rm.role] = groups_.size() - 1;
            }
        }

        // Step 5: Build role_to_group_ mapping (pointers to groups).
        //         Must be done after all groups are created since vector
        //         reallocation invalidates pointers.
        role_to_group_.clear();
        for (auto& [role, idx] : role_index) {
            role_to_group_[role] = &groups_[idx];
        }

        // Step 6: Set local ranks for my_rank in each group, and build
        //         my_rank_info_ with primary_role and memberships.
        my_rank_info_ = rank_info{};
        my_rank_info_.world_rank = my_rank;
        my_groups_.clear();

        for (auto& group : groups_) {
            rank_t local_rank = group.to_local_rank(my_rank);
            if (local_rank != no_rank) {
                group.set_local_rank(local_rank);

                group_membership membership;
                membership.group = &group;
                membership.local_rank = local_rank;
                my_rank_info_.memberships.push_back(membership);

                my_groups_.push_back(&group);
            }
        }

        // Step 7: Determine primary role. Use the first non-worker role
        //         if available; otherwise use worker; otherwise use the
        //         first role assigned.
        if (!my_rank_info_.memberships.empty()) {
            my_rank_info_.primary_role = my_rank_info_.memberships[0].group->role();
            for (const auto& m : my_rank_info_.memberships) {
                if (m.group->role() != node_role::worker) {
                    my_rank_info_.primary_role = m.group->role();
                    break;
                }
            }
        } else {
            my_rank_info_.primary_role = node_role::undefined;
        }

        initialized_ = true;

        return {};
    }

    /// @brief Check if the manager has been initialized
    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

    /// @brief Reset the manager to uninitialized state
    void reset() {
        groups_.clear();
        my_groups_.clear();
        role_to_group_.clear();
        my_rank_info_ = rank_info{};
        initialized_ = false;
        world_comm_ = nullptr;
    }

    // ------------------------------------------------------------------------
    // Group Access
    // ------------------------------------------------------------------------

    /// @brief Get a group by role
    /// @param role The role to find
    /// @return Pointer to group, or nullptr if not found
    [[nodiscard]] rank_group* get_group(node_role role) noexcept {
        auto it = role_to_group_.find(role);
        if (it != role_to_group_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /// @brief Get a group by role (const version)
    [[nodiscard]] const rank_group* get_group(node_role role) const noexcept {
        auto it = role_to_group_.find(role);
        if (it != role_to_group_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /// @brief Get a group by ID
    /// @param id The group ID
    /// @return Pointer to group, or nullptr if not found
    [[nodiscard]] rank_group* get_group_by_id(rank_group::group_id id) noexcept {
        for (auto& group : groups_) {
            if (group.id() == id) return &group;
        }
        return nullptr;
    }

    /// @brief Get all groups
    [[nodiscard]] const std::vector<rank_group>& groups() const noexcept {
        return groups_;
    }

    /// @brief Get groups that this rank belongs to
    [[nodiscard]] const std::vector<rank_group*>& my_groups() const noexcept {
        return my_groups_;
    }

    /// @brief Get rank info for the calling process
    [[nodiscard]] const rank_info& my_info() const noexcept {
        return my_rank_info_;
    }

    // ------------------------------------------------------------------------
    // Role Queries
    // ------------------------------------------------------------------------

    /// @brief Check if this rank has a specific role
    /// @param role The role to check
    /// @return true if this rank has the role
    [[nodiscard]] bool has_role(node_role role) const noexcept {
        return my_rank_info_.has_role(role);
    }

    /// @brief Get the primary role of this rank
    /// @return The primary role
    [[nodiscard]] node_role primary_role() const noexcept {
        return my_rank_info_.primary_role;
    }

    /// @brief Get all roles assigned to this rank
    [[nodiscard]] std::vector<node_role> my_roles() const {
        std::vector<node_role> roles;
        for (const auto& m : my_rank_info_.memberships) {
            if (m.group) {
                roles.push_back(m.group->role());
            }
        }
        return roles;
    }

    // ------------------------------------------------------------------------
    // Group Creation
    // ------------------------------------------------------------------------

    /// @brief Manually create a group (for advanced usage)
    /// @param role Role for the group
    /// @param name Group name
    /// @param members World ranks to include
    /// @return Reference to created group, or error
    result<rank_group*> create_group(
        node_role role,
        std::string name,
        const std::vector<rank_t>& members) {
        auto id = static_cast<rank_group::group_id>(groups_.size());
        groups_.emplace_back(id, role, std::move(name));
        auto& group = groups_.back();

        for (rank_t r : members) {
            group.add_member(r);
        }

        role_to_group_[role] = &group;
        return &group;
    }

    // ------------------------------------------------------------------------
    // Inter-Group Communication
    // ------------------------------------------------------------------------

    /// @brief Create an inter-communicator between two groups
    /// @param group_a First group
    /// @param group_b Second group
    /// @return Not yet supported
    /// @note This is a placeholder for future MPI_Intercomm_create support.
    ///       In a real MPI deployment, this would use MPI_Intercomm_create
    ///       to enable direct communication between disjoint groups.
    result<void> create_inter_communicator(
        [[maybe_unused]] const rank_group& group_a,
        [[maybe_unused]] const rank_group& group_b) {
        return make_error<void>(status_code::not_supported,
                                "Inter-communicator creation not yet supported");
    }

    /// @brief Get the communicator for a specific role group
    /// @param role The role to get communicator for
    /// @return Pointer to communicator, or nullptr if role not found
    /// @note In a real MPI environment, this would return a sub-communicator
    ///       created via MPI_Comm_split during initialization. Each role
    ///       group would have its own communicator restricted to group
    ///       members, enabling efficient intra-group collectives.
    ///       In single-process simulation, sub-communicators are logical
    ///       only — this returns the group's communicator if set, or
    ///       falls back to the world communicator.
    [[nodiscard]] communicator_base* get_role_communicator(node_role role) noexcept {
        auto* group = get_group(role);
        if (!group) return nullptr;
        // If the group has a dedicated sub-communicator, return it
        if (group->has_communicator()) {
            return group->communicator();
        }
        // Fall back to world communicator for single-process testing.
        // In a real MPI deployment, MPI_Comm_split would create a
        // proper sub-communicator for each role group during initialize().
        return world_comm_;
    }

    /// @brief Get the communicator for a specific role group (const)
    /// @param role The role to get communicator for
    /// @return Pointer to communicator, or nullptr if role not found
    [[nodiscard]] const communicator_base* get_role_communicator(node_role role) const noexcept {
        auto* group = get_group(role);
        if (!group) return nullptr;
        if (group->has_communicator()) {
            return group->communicator();
        }
        return world_comm_;
    }

    // ------------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------------

    /// @brief Get the configuration
    [[nodiscard]] const role_manager_config& config() const noexcept {
        return config_;
    }

    /// @brief Modify configuration (before initialization)
    /// @param config New configuration
    /// @return Success or error if already initialized
    result<void> set_config(role_manager_config config) {
        if (initialized_) {
            return make_error<void>(status_code::invalid_state,
                                    "Cannot change config after initialization");
        }
        config_ = std::move(config);
        return {};
    }

private:
    role_manager_config config_;
    std::vector<role_descriptor> descriptors_;
    std::vector<rank_group> groups_;
    std::map<node_role, rank_group*> role_to_group_;
    std::vector<rank_group*> my_groups_;
    rank_info my_rank_info_;
    communicator_base* world_comm_ = nullptr;
    bool initialized_ = false;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Create a simple worker/coordinator MPMD setup
/// @param manager The role manager to configure
/// @param coordinator_count Number of coordinator ranks (default 1)
/// @return Success or error
inline result<void> setup_worker_coordinator(
    role_manager& manager,
    rank_t coordinator_count = 1) {

    // Coordinators are the first N ranks
    auto coord_result = manager.register_role(
        node_role::coordinator,
        "coordinator",
        role_assignment::first_n_ranks(coordinator_count));
    if (!coord_result) return coord_result;

    // Workers are all other ranks
    auto worker_result = manager.register_role(
        node_role::worker,
        "worker",
        [coordinator_count](rank_t rank, rank_t) {
            return rank >= coordinator_count;
        });

    return worker_result;
}

/// @brief Create a setup with dedicated I/O handlers
/// @param manager The role manager to configure
/// @param io_count Number of I/O handler ranks
/// @return Success or error
inline result<void> setup_with_io_handlers(
    role_manager& manager,
    rank_t io_count = 1) {

    // I/O handlers are the last N ranks
    auto io_result = manager.register_role(
        node_role::io_handler,
        "io_handler",
        role_assignment::last_n_ranks(io_count));
    if (!io_result) return io_result;

    // Workers are all other ranks
    auto worker_result = manager.register_role(
        node_role::worker,
        "worker",
        [io_count](rank_t rank, rank_t num_ranks) {
            return rank < num_ranks - io_count;
        });

    return worker_result;
}

}  // namespace dtl::mpmd
