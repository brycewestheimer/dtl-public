// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file nccl_group_ops.hpp
/// @brief NCCL group operation wrappers
/// @details Provides RAII wrappers and helper functions for batching
///          multiple NCCL operations within ncclGroupStart/ncclGroupEnd.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_NCCL
#include <nccl.h>
#endif

namespace dtl {
namespace nccl {

// ============================================================================
// Group Operation Functions
// ============================================================================

/// @brief Start a NCCL group operation
/// @return Success or error
[[nodiscard]] inline result<void> group_start() {
#if DTL_ENABLE_NCCL
    ncclResult_t res = ncclGroupStart();
    if (res != ncclSuccess) {
        return make_error<void>(status_code::backend_error,
                               "ncclGroupStart failed");
    }
    return {};
#else
    return make_error<void>(status_code::not_supported,
                           "NCCL support not enabled");
#endif
}

/// @brief End a NCCL group operation
/// @return Success or error
[[nodiscard]] inline result<void> group_end() {
#if DTL_ENABLE_NCCL
    ncclResult_t res = ncclGroupEnd();
    if (res != ncclSuccess) {
        return make_error<void>(status_code::backend_error,
                               "ncclGroupEnd failed");
    }
    return {};
#else
    return make_error<void>(status_code::not_supported,
                           "NCCL support not enabled");
#endif
}

// ============================================================================
// Scoped Group Operations
// ============================================================================

/// @brief RAII wrapper for NCCL group operations
/// @details Calls ncclGroupStart() on construction and ncclGroupEnd()
///          on destruction. All NCCL calls between construction and
///          destruction are batched as a single group operation.
class scoped_group_ops {
public:
    /// @brief Begin a group operation
    scoped_group_ops() {
#if DTL_ENABLE_NCCL
        ncclResult_t res = ncclGroupStart();
        valid_ = (res == ncclSuccess);
#endif
    }

    /// @brief End the group operation
    ~scoped_group_ops() {
#if DTL_ENABLE_NCCL
        if (valid_) {
            ncclGroupEnd();
        }
#endif
    }

    // Non-copyable, non-movable
    scoped_group_ops(const scoped_group_ops&) = delete;
    scoped_group_ops& operator=(const scoped_group_ops&) = delete;
    scoped_group_ops(scoped_group_ops&&) = delete;
    scoped_group_ops& operator=(scoped_group_ops&&) = delete;

    /// @brief Check if group was started successfully
    [[nodiscard]] bool valid() const noexcept { return valid_; }

private:
    bool valid_ = false;
};

}  // namespace nccl
}  // namespace dtl
