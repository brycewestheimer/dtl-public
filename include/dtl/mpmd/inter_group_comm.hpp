// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file inter_group_comm.hpp
/// @brief Inter-group communication utilities for MPMD patterns
/// @details Provides rank translation and communication helpers for sending
///          data between rank groups using the world communicator.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/mpmd/rank_group.hpp>
#include <dtl/mpmd/role_manager.hpp>
#include <dtl/communication/communicator_base.hpp>

namespace dtl::mpmd {

// Import communicator_base from parent dtl:: namespace
using dtl::communicator_base;

// ============================================================================
// Rank Translation Helpers
// ============================================================================

/// @brief Translate a local rank within a group to a world rank
/// @param group The rank group
/// @param local_rank The local rank within the group
/// @return The world rank, or error if local_rank is invalid
[[nodiscard]] inline result<rank_t> translate_to_world_rank(
    const rank_group& group, rank_t local_rank) {
    if (group.empty()) {
        return make_error<rank_t>(status_code::invalid_argument,
                                  "Cannot translate rank in empty group");
    }
    rank_t world_rank = group.to_world_rank(local_rank);
    if (world_rank == no_rank) {
        return make_error<rank_t>(status_code::invalid_rank,
                                  "Invalid local rank " +
                                  std::to_string(local_rank) +
                                  " for group '" + group.name() +
                                  "' of size " +
                                  std::to_string(group.size()));
    }
    return world_rank;
}

/// @brief Translate a world rank to a local rank within a group
/// @param group The rank group
/// @param world_rank The world rank to translate
/// @return The local rank, or error if world_rank is not a member
[[nodiscard]] inline result<rank_t> translate_to_local_rank(
    const rank_group& group, rank_t world_rank) {
    rank_t local_rank = group.to_local_rank(world_rank);
    if (local_rank == no_rank) {
        return make_error<rank_t>(status_code::invalid_rank,
                                  "World rank " +
                                  std::to_string(world_rank) +
                                  " is not a member of group '" +
                                  group.name() + "'");
    }
    return local_rank;
}

// ============================================================================
// Inter-Group Point-to-Point Communication
// ============================================================================

/// @brief Compute the world rank for an inter-group send operation
/// @details Translates a destination local rank within a target group to the
///          corresponding world rank. This is the core operation needed for
///          inter-group communication: the caller can then use the world
///          communicator with the returned world rank.
/// @param dest_group The destination rank group
/// @param dest_local_rank Local rank within dest_group
/// @return The world rank to use for communication, or error
[[nodiscard]] inline result<rank_t> inter_group_dest_rank(
    const rank_group& dest_group, rank_t dest_local_rank) {
    return translate_to_world_rank(dest_group, dest_local_rank);
}

/// @brief Compute the world rank for an inter-group receive operation
/// @details Translates a source local rank within a source group to the
///          corresponding world rank.
/// @param src_group The source rank group
/// @param src_local_rank Local rank within src_group
/// @return The world rank to use for communication, or error
[[nodiscard]] inline result<rank_t> inter_group_src_rank(
    const rank_group& src_group, rank_t src_local_rank) {
    return translate_to_world_rank(src_group, src_local_rank);
}

/// @brief Compute the world rank for an inter-group broadcast root
/// @details Translates a root local rank within a root group to the
///          corresponding world rank for use as broadcast root.
/// @param root_group The group containing the broadcast root
/// @param root_local_rank Local rank within root_group
/// @return The world rank to use as broadcast root, or error
[[nodiscard]] inline result<rank_t> inter_group_broadcast_root(
    const rank_group& root_group, rank_t root_local_rank) {
    return translate_to_world_rank(root_group, root_local_rank);
}

// ============================================================================
// Inter-Group Communication Descriptor
// ============================================================================

/// @brief Describes an inter-group communication endpoint
/// @details Encapsulates the information needed to identify a rank in
///          another group for inter-group communication.
struct inter_group_endpoint {
    /// @brief The rank group this endpoint belongs to
    const rank_group* group = nullptr;

    /// @brief The local rank within the group
    rank_t local_rank = no_rank;

    /// @brief Message tag for this endpoint
    int tag = 0;

    /// @brief Check if this endpoint is valid
    [[nodiscard]] bool valid() const noexcept {
        return group != nullptr && local_rank != no_rank && !group->empty();
    }

    /// @brief Get the world rank for this endpoint
    /// @return World rank, or error if invalid
    [[nodiscard]] result<rank_t> world_rank() const {
        if (!group) {
            return make_error<rank_t>(status_code::invalid_argument,
                                      "Null group in endpoint");
        }
        return translate_to_world_rank(*group, local_rank);
    }
};

/// @brief Create an inter-group endpoint from a group and local rank
/// @param group The rank group
/// @param local_rank Local rank within the group
/// @param tag Message tag (default 0)
/// @return The endpoint descriptor
[[nodiscard]] inline inter_group_endpoint make_endpoint(
    const rank_group& group, rank_t local_rank, int tag = 0) {
    return inter_group_endpoint{&group, local_rank, tag};
}

// ============================================================================
// Inter-Group Channel
// ============================================================================

/// @brief Represents a bidirectional communication channel between two groups
/// @details Captures the source and destination endpoints for inter-group
///          communication, simplifying repeated send/recv patterns.
struct inter_group_channel {
    /// @brief Source endpoint (local group/rank)
    inter_group_endpoint source;

    /// @brief Destination endpoint (remote group/rank)
    inter_group_endpoint destination;

    /// @brief Check if the channel is valid
    [[nodiscard]] bool valid() const noexcept {
        return source.valid() && destination.valid();
    }

    /// @brief Get the destination world rank
    /// @return World rank, or error
    [[nodiscard]] result<rank_t> dest_world_rank() const {
        return destination.world_rank();
    }

    /// @brief Get the source world rank
    /// @return World rank, or error
    [[nodiscard]] result<rank_t> src_world_rank() const {
        return source.world_rank();
    }
};

/// @brief Create a channel between two groups
/// @param src_group Source group
/// @param src_local_rank Local rank in source group
/// @param dst_group Destination group
/// @param dst_local_rank Local rank in destination group
/// @param tag Message tag (default 0)
/// @return The channel descriptor
[[nodiscard]] inline inter_group_channel make_channel(
    const rank_group& src_group, rank_t src_local_rank,
    const rank_group& dst_group, rank_t dst_local_rank,
    int tag = 0) {
    return inter_group_channel{
        make_endpoint(src_group, src_local_rank, tag),
        make_endpoint(dst_group, dst_local_rank, tag)
    };
}

}  // namespace dtl::mpmd
