// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file message_status.hpp
/// @brief Message status structure for communication operations
/// @details Provides status information about received messages.
/// @since 0.1.0

#pragma once

#include <dtl/core/types.hpp>

namespace dtl {

// ============================================================================
// Message Status
// ============================================================================

/// @brief Status of a received message
struct message_status {
    /// @brief Source rank of the message
    rank_t source = no_rank;

    /// @brief Tag of the message
    int tag = 0;

    /// @brief Number of elements received
    size_type count = 0;

    /// @brief Whether the receive was cancelled
    bool cancelled = false;

    /// @brief Error code (0 = success)
    int error = 0;
};

}  // namespace dtl
