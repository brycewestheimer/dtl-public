// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_mpmd.cpp
 * @brief DTL C bindings - MPMD role manager implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_mpmd.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_types.h>

#include "dtl_internal.hpp"

#include <cstring>
#include <string>
#include <vector>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// Internal Structures
// ============================================================================

/**
 * Role manager implementation
 *
 * Manages named role groups for MPMD execution. Ranks are assigned
 * sequentially to roles in the order they are added: first N ranks
 * to the first role, next M to the second, and so on.
 */
struct dtl_role_manager_s {
    // Reference to the bound context
    dtl_context_t ctx;

    // Role definitions (parallel arrays)
    std::vector<std::string> role_names;
    std::vector<dtl_size_t> role_sizes;

    // Whether initialize() has been called
    bool initialized;

    // This rank's global MPI rank and total communicator size
    dtl_rank_t my_rank;
    dtl_size_t num_ranks;

    // This rank's assigned role (set after initialize)
    std::string my_role;

    // This rank's rank within its role group (set after initialize)
    dtl_rank_t role_rank;

    // Validation magic
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xDEAD3030;
};

// ============================================================================
// Validation Helpers
// ============================================================================

static bool is_valid_role_manager(dtl_role_manager_t mgr) {
    return mgr && mgr->magic == dtl_role_manager_s::VALID_MAGIC;
}

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Find the index of a role by name.
 * Returns -1 if not found.
 */
static int find_role_index(dtl_role_manager_t mgr, const char* role_name) {
    for (size_t i = 0; i < mgr->role_names.size(); ++i) {
        if (mgr->role_names[i] == role_name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/**
 * Compute the global rank offset for a given role index.
 * The offset is the sum of all role sizes before this role.
 */
static dtl_rank_t role_global_offset(dtl_role_manager_t mgr, int role_idx) {
    dtl_rank_t offset = 0;
    for (int i = 0; i < role_idx; ++i) {
        offset += static_cast<dtl_rank_t>(mgr->role_sizes[i]);
    }
    return offset;
}

// ============================================================================
// MPI Datatype Mapping
// ============================================================================

#ifdef DTL_HAS_MPI

static MPI_Datatype mpmd_dtype_to_mpi(dtl_dtype dtype) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    return MPI_INT8_T;
        case DTL_DTYPE_INT16:   return MPI_INT16_T;
        case DTL_DTYPE_INT32:   return MPI_INT32_T;
        case DTL_DTYPE_INT64:   return MPI_INT64_T;
        case DTL_DTYPE_UINT8:   return MPI_UINT8_T;
        case DTL_DTYPE_UINT16:  return MPI_UINT16_T;
        case DTL_DTYPE_UINT32:  return MPI_UINT32_T;
        case DTL_DTYPE_UINT64:  return MPI_UINT64_T;
        case DTL_DTYPE_FLOAT32: return MPI_FLOAT;
        case DTL_DTYPE_FLOAT64: return MPI_DOUBLE;
        case DTL_DTYPE_BYTE:    return MPI_BYTE;
        case DTL_DTYPE_BOOL:    return MPI_UINT8_T;
        default:                return MPI_DATATYPE_NULL;
    }
}

#endif  // DTL_HAS_MPI

// ============================================================================
// Role Manager Lifecycle
// ============================================================================

extern "C" {

dtl_status dtl_role_manager_create(dtl_context_t ctx,
                                    dtl_role_manager_t* mgr) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!mgr) {
        return DTL_ERROR_NULL_POINTER;
    }

    dtl_role_manager_s* impl = nullptr;
    try {
        impl = new dtl_role_manager_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->ctx = ctx;
    impl->initialized = false;
    impl->my_rank = ctx->rank;
    impl->num_ranks = static_cast<dtl_size_t>(ctx->size);
    impl->role_rank = DTL_NO_RANK;
    impl->magic = dtl_role_manager_s::VALID_MAGIC;

    *mgr = impl;
    return DTL_SUCCESS;
}

void dtl_role_manager_destroy(dtl_role_manager_t mgr) {
    if (!is_valid_role_manager(mgr)) {
        return;
    }

    mgr->magic = 0;
    delete mgr;
}

// ============================================================================
// Role Configuration
// ============================================================================

dtl_status dtl_role_manager_add_role(dtl_role_manager_t mgr,
                                      const char* role_name,
                                      dtl_size_t num_ranks) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!role_name || role_name[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (num_ranks == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    // Check for duplicate role name
    if (find_role_index(mgr, role_name) >= 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    try {
        mgr->role_names.emplace_back(role_name);
        mgr->role_sizes.push_back(num_ranks);
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    return DTL_SUCCESS;
}

dtl_status dtl_role_manager_initialize(dtl_role_manager_t mgr) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }
    if (mgr->role_names.empty()) {
        return DTL_ERROR_INVALID_STATE;
    }

    // Verify total ranks matches communicator size
    dtl_size_t total = 0;
    for (size_t i = 0; i < mgr->role_sizes.size(); ++i) {
        total += mgr->role_sizes[i];
    }
    if (total != mgr->num_ranks) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Assign this rank to a role sequentially
    dtl_rank_t offset = 0;
    for (size_t i = 0; i < mgr->role_names.size(); ++i) {
        dtl_rank_t role_end = offset + static_cast<dtl_rank_t>(mgr->role_sizes[i]);
        if (mgr->my_rank >= offset && mgr->my_rank < role_end) {
            mgr->my_role = mgr->role_names[i];
            mgr->role_rank = mgr->my_rank - offset;
            break;
        }
        offset = role_end;
    }

    mgr->initialized = true;
    return DTL_SUCCESS;
}

// ============================================================================
// Role Queries
// ============================================================================

dtl_status dtl_role_manager_has_role(dtl_role_manager_t mgr,
                                      const char* role_name,
                                      int* has_role) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!role_name || role_name[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!has_role) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    *has_role = (mgr->my_role == role_name) ? 1 : 0;
    return DTL_SUCCESS;
}

dtl_status dtl_role_manager_role_size(dtl_role_manager_t mgr,
                                       const char* role_name,
                                       dtl_size_t* size) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!role_name || role_name[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!size) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    int idx = find_role_index(mgr, role_name);
    if (idx < 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    *size = mgr->role_sizes[idx];
    return DTL_SUCCESS;
}

dtl_status dtl_role_manager_role_rank(dtl_role_manager_t mgr,
                                       const char* role_name,
                                       dtl_rank_t* rank) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!role_name || role_name[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!rank) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    // This rank must belong to the specified role
    if (mgr->my_role != role_name) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    *rank = mgr->role_rank;
    return DTL_SUCCESS;
}

// ============================================================================
// Inter-Group Communication
// ============================================================================

dtl_status dtl_intergroup_send(dtl_role_manager_t mgr,
                                const char* target_role,
                                dtl_rank_t target_rank,
                                const void* buf,
                                dtl_size_t count,
                                dtl_dtype dtype,
                                dtl_tag_t tag) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!target_role || target_role[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buf && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    // Find target role
    int role_idx = find_role_index(mgr, target_role);
    if (role_idx < 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Validate target_rank within role
    if (target_rank < 0 || static_cast<dtl_size_t>(target_rank) >= mgr->role_sizes[role_idx]) {
        return DTL_ERROR_INVALID_RANK;
    }

    // Compute global rank of target
    dtl_rank_t global_target = role_global_offset(mgr, role_idx) + target_rank;

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = mpmd_dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    if (!checked_size_to_int(count, &mpi_count)) {
        return DTL_ERROR_OUT_OF_RANGE;
    }

    int err = MPI_Send(buf, mpi_count, mpi_type,
                       global_target, tag, mgr->ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_SEND_FAILED;
    }
    return DTL_SUCCESS;
#else
    (void)global_target;
    (void)count;
    (void)dtype;
    (void)tag;
    return DTL_ERROR_NOT_SUPPORTED;
#endif
}

dtl_status dtl_intergroup_recv(dtl_role_manager_t mgr,
                                const char* source_role,
                                dtl_rank_t source_rank,
                                void* buf,
                                dtl_size_t count,
                                dtl_dtype dtype,
                                dtl_tag_t tag) {
    if (!is_valid_role_manager(mgr)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!source_role || source_role[0] == '\0') {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buf && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!mgr->initialized) {
        return DTL_ERROR_INVALID_STATE;
    }

    // Find source role
    int role_idx = find_role_index(mgr, source_role);
    if (role_idx < 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Validate source_rank within role
    if (source_rank < 0 || static_cast<dtl_size_t>(source_rank) >= mgr->role_sizes[role_idx]) {
        return DTL_ERROR_INVALID_RANK;
    }

    // Compute global rank of source
    dtl_rank_t global_source = role_global_offset(mgr, role_idx) + source_rank;

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = mpmd_dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    if (!checked_size_to_int(count, &mpi_count)) {
        return DTL_ERROR_OUT_OF_RANGE;
    }

    int err = MPI_Recv(buf, mpi_count, mpi_type,
                       global_source, tag, mgr->ctx->comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_RECV_FAILED;
    }
    return DTL_SUCCESS;
#else
    (void)global_source;
    (void)count;
    (void)dtype;
    (void)tag;
    return DTL_ERROR_NOT_SUPPORTED;
#endif
}

}  // extern "C"
