// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rank_group.hpp
/// @brief Rank group for MPMD patterns - subset of ranks with shared role
/// @details Provides grouping and sub-communicator management for MPMD.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/mpmd/node_role.hpp>

#include <memory>
#include <string>
#include <vector>
#include <optional>

// Forward declare communicator_base in dtl:: namespace
namespace dtl { class communicator_base; }

namespace dtl::mpmd {

// Import communicator_base from parent dtl:: namespace
using dtl::communicator_base;

// ============================================================================
// Rank Group
// ============================================================================

/// @brief A group of ranks sharing a common role
/// @details Rank groups enable MPMD patterns by organizing processes into
///          logical subsets that can communicate independently. Each group
///          has its own sub-communicator for intra-group operations.
class rank_group {
public:
    /// @brief Group identifier type
    using group_id = std::uint32_t;

    /// @brief Invalid group ID sentinel
    static constexpr group_id invalid_group_id = static_cast<group_id>(-1);

    /// @brief Default constructor (creates invalid group)
    rank_group() = default;

    /// @brief Construct a rank group
    /// @param id Unique group identifier
    /// @param role The role assigned to this group
    /// @param name Human-readable group name
    rank_group(group_id id, node_role role, std::string name = "")
        : id_(id)
        , role_(role)
        , name_(std::move(name))
        , local_rank_(no_rank)
        , group_size_(0) {}

    // ------------------------------------------------------------------------
    // Group Identity
    // ------------------------------------------------------------------------

    /// @brief Get the group identifier
    [[nodiscard]] group_id id() const noexcept { return id_; }

    /// @brief Get the role associated with this group
    [[nodiscard]] node_role role() const noexcept { return role_; }

    /// @brief Get the group name
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// @brief Check if this is a valid group
    [[nodiscard]] bool valid() const noexcept {
        return id_ != invalid_group_id;
    }

    /// @brief Boolean conversion (true if valid)
    explicit operator bool() const noexcept { return valid(); }

    // ------------------------------------------------------------------------
    // Membership
    // ------------------------------------------------------------------------

    /// @brief Get ranks in this group (world ranks)
    [[nodiscard]] const std::vector<rank_t>& members() const noexcept {
        return members_;
    }

    /// @brief Get number of ranks in this group
    [[nodiscard]] size_type size() const noexcept { return group_size_; }

    /// @brief Check if the group is empty
    [[nodiscard]] bool empty() const noexcept { return group_size_ == 0; }

    /// @brief Check if a world rank is a member of this group
    /// @param world_rank The rank to check (in world communicator)
    /// @return true if the rank is a member
    [[nodiscard]] bool contains(rank_t world_rank) const noexcept {
        for (rank_t r : members_) {
            if (r == world_rank) return true;
        }
        return false;
    }

    /// @brief Get the local rank within this group
    /// @return Local rank, or no_rank if not a member
    [[nodiscard]] rank_t local_rank() const noexcept { return local_rank_; }

    /// @brief Check if the calling process is a member
    [[nodiscard]] bool is_member() const noexcept { return local_rank_ != no_rank; }

    /// @brief Convert world rank to group-local rank
    /// @param world_rank The world rank to convert
    /// @return Local rank, or no_rank if not a member
    [[nodiscard]] rank_t to_local_rank(rank_t world_rank) const noexcept {
        for (size_type i = 0; i < members_.size(); ++i) {
            if (members_[i] == world_rank) {
                return static_cast<rank_t>(i);
            }
        }
        return no_rank;
    }

    /// @brief Convert group-local rank to world rank
    /// @param local_rank The local rank to convert
    /// @return World rank, or no_rank if out of range
    [[nodiscard]] rank_t to_world_rank(rank_t local_rank) const noexcept {
        if (local_rank >= 0 && static_cast<size_type>(local_rank) < members_.size()) {
            return members_[static_cast<size_type>(local_rank)];
        }
        return no_rank;
    }

    /// @brief Get the group leader (rank 0 in group)
    /// @return World rank of group leader, or no_rank if empty
    [[nodiscard]] rank_t leader() const noexcept {
        return members_.empty() ? no_rank : members_[0];
    }

    /// @brief Check if calling process is the group leader
    [[nodiscard]] bool is_leader() const noexcept { return local_rank_ == 0; }

    // ------------------------------------------------------------------------
    // Communication
    // ------------------------------------------------------------------------

    /// @brief Get the sub-communicator for this group
    /// @return Pointer to communicator, or nullptr if not available
    /// @note The communicator is only valid for group members
    [[nodiscard]] communicator_base* communicator() const noexcept {
        return communicator_.get();
    }

    /// @brief Check if group has an associated communicator
    [[nodiscard]] bool has_communicator() const noexcept {
        return communicator_ != nullptr;
    }

    // ------------------------------------------------------------------------
    // Modification (used by role_manager)
    // ------------------------------------------------------------------------

    /// @brief Add a member to this group
    /// @param world_rank The world rank to add
    void add_member(rank_t world_rank) {
        members_.push_back(world_rank);
        group_size_ = members_.size();
    }

    /// @brief Set the local rank for the calling process
    /// @param rank The local rank within this group
    void set_local_rank(rank_t rank) noexcept { local_rank_ = rank; }

    /// @brief Set the sub-communicator for this group
    /// @param comm The communicator
    void set_communicator(std::shared_ptr<communicator_base> comm) {
        communicator_ = std::move(comm);
    }

    /// @brief Clear all members
    void clear_members() {
        members_.clear();
        group_size_ = 0;
        local_rank_ = no_rank;
    }

private:
    group_id id_ = invalid_group_id;
    node_role role_ = node_role::undefined;
    std::string name_;
    std::vector<rank_t> members_;
    rank_t local_rank_ = no_rank;
    size_type group_size_ = 0;
    std::shared_ptr<communicator_base> communicator_;
};

// ============================================================================
// Group Query Results
// ============================================================================

/// @brief Result of group membership query
struct group_membership {
    /// @brief The group
    rank_group* group = nullptr;

    /// @brief Local rank within the group
    rank_t local_rank = no_rank;

    /// @brief Check if membership is valid
    [[nodiscard]] bool valid() const noexcept {
        return group != nullptr && local_rank != no_rank;
    }

    /// @brief Boolean conversion
    explicit operator bool() const noexcept { return valid(); }
};

/// @brief Information about a rank's group memberships
struct rank_info {
    /// @brief World rank
    rank_t world_rank = no_rank;

    /// @brief Groups this rank belongs to
    std::vector<group_membership> memberships;

    /// @brief Primary role (first non-worker role, or worker)
    node_role primary_role = node_role::undefined;

    /// @brief Check if rank has a specific role
    [[nodiscard]] bool has_role(node_role role) const noexcept {
        for (const auto& m : memberships) {
            if (m.group && m.group->role() == role) return true;
        }
        return false;
    }

    /// @brief Get group for a specific role
    /// @param role The role to find
    /// @return Pointer to group, or nullptr if not found
    [[nodiscard]] rank_group* group_for_role(node_role role) const noexcept {
        for (const auto& m : memberships) {
            if (m.group && m.group->role() == role) return m.group;
        }
        return nullptr;
    }
};

// ============================================================================
// Comparison Operators
// ============================================================================

/// @brief Equality comparison for rank groups
[[nodiscard]] inline bool operator==(const rank_group& lhs, const rank_group& rhs) noexcept {
    return lhs.id() == rhs.id();
}

/// @brief Inequality comparison for rank groups
[[nodiscard]] inline bool operator!=(const rank_group& lhs, const rank_group& rhs) noexcept {
    return !(lhs == rhs);
}

/// @brief Less-than comparison (for ordered containers)
[[nodiscard]] inline bool operator<(const rank_group& lhs, const rank_group& rhs) noexcept {
    return lhs.id() < rhs.id();
}

}  // namespace dtl::mpmd
