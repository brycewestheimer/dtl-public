// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_mpmd.h
 * @brief DTL C bindings - MPMD role manager operations
 * @since 0.1.0
 *
 * This header provides C bindings for Multiple Program Multiple Data
 * (MPMD) support, including role management and inter-group communication.
 * Roles partition MPI ranks into named groups, enabling distinct programs
 * or phases to communicate across group boundaries.
 */

#ifndef DTL_MPMD_H
#define DTL_MPMD_H

#include "dtl_config.h"
#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"

DTL_C_BEGIN

/* ==========================================================================
 * Opaque Handle Types
 * ========================================================================== */

/** @brief Forward declaration for role manager implementation */
struct dtl_role_manager_s;

/**
 * @brief Opaque handle to MPMD role manager
 *
 * A role manager partitions MPI ranks into named groups (roles).
 * Each rank belongs to exactly one role after initialization.
 * Inter-group communication is supported between ranks in different roles.
 */
typedef struct dtl_role_manager_s* dtl_role_manager_t;

/* ==========================================================================
 * Role Manager Lifecycle
 * ========================================================================== */

/**
 * @brief Create a new MPMD role manager
 *
 * Creates an uninitialized role manager bound to the given context.
 * After creation, roles must be added with dtl_role_manager_add_role()
 * and the manager must be initialized with dtl_role_manager_initialize()
 * before any queries or communication can be performed.
 *
 * @param ctx The context to bind the role manager to
 * @param[out] mgr Pointer to receive the created role manager handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre mgr must not be NULL
 * @post On success, *mgr contains a valid role manager handle
 * @post On failure, *mgr is unchanged
 *
 * @note The caller must call dtl_role_manager_destroy() when done.
 *
 * @code
 * dtl_context_t ctx;
 * dtl_context_create_default(&ctx);
 *
 * dtl_role_manager_t mgr;
 * dtl_status status = dtl_role_manager_create(ctx, &mgr);
 * if (status != DTL_SUCCESS) {
 *     fprintf(stderr, "Failed: %s\n", dtl_status_message(status));
 *     return 1;
 * }
 *
 * dtl_role_manager_add_role(mgr, "producer", 2);
 * dtl_role_manager_add_role(mgr, "consumer", 2);
 * dtl_role_manager_initialize(mgr);
 *
 * // Use role manager...
 * dtl_role_manager_destroy(mgr);
 * dtl_context_destroy(ctx);
 * @endcode
 */
DTL_API dtl_status dtl_role_manager_create(dtl_context_t ctx,
                                            dtl_role_manager_t* mgr);

/**
 * @brief Destroy a role manager
 *
 * Releases all resources associated with the role manager.
 *
 * @param mgr The role manager to destroy (may be NULL, which is a no-op)
 *
 * @post mgr is invalid and must not be used
 *
 * @note It is safe to call with NULL.
 * @note The associated context must still be valid when this is called.
 */
DTL_API void dtl_role_manager_destroy(dtl_role_manager_t mgr);

/* ==========================================================================
 * Role Configuration
 * ========================================================================== */

/**
 * @brief Add a role to the role manager
 *
 * Defines a named role with a specified number of ranks. Roles must
 * be added before calling dtl_role_manager_initialize(). The total
 * number of ranks across all roles must equal the communicator size.
 *
 * @param mgr The role manager
 * @param role_name Name of the role (must not be NULL or empty)
 * @param num_ranks Number of ranks to assign to this role
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, uninitialized role manager
 * @pre role_name must be a non-empty null-terminated string
 * @pre num_ranks must be > 0
 * @pre The role manager must not yet be initialized
 *
 * @note Role names must be unique within a role manager.
 * @note Roles are assigned ranks sequentially in the order they are added.
 */
DTL_API dtl_status dtl_role_manager_add_role(dtl_role_manager_t mgr,
                                              const char* role_name,
                                              dtl_size_t num_ranks);

/**
 * @brief Initialize the role manager
 *
 * Assigns ranks to roles sequentially: the first N ranks go to the
 * first role added, the next M ranks to the second role, and so on.
 * The total number of ranks across all roles must equal the context's
 * communicator size.
 *
 * @param mgr The role manager
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, uninitialized role manager
 * @pre At least one role must have been added
 * @pre The total ranks across all roles must equal the communicator size
 * @post On success, the role manager is initialized and ready for queries
 *
 * @note This function must be called by all ranks.
 */
DTL_API dtl_status dtl_role_manager_initialize(dtl_role_manager_t mgr);

/* ==========================================================================
 * Role Queries
 * ========================================================================== */

/**
 * @brief Query whether this rank has a specific role
 *
 * Checks if the calling rank belongs to the named role group.
 *
 * @param mgr The role manager
 * @param role_name Name of the role to check
 * @param[out] has_role Set to 1 if this rank has the role, 0 otherwise
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, initialized role manager
 * @pre role_name must be a non-empty null-terminated string
 * @pre has_role must not be NULL
 */
DTL_API dtl_status dtl_role_manager_has_role(dtl_role_manager_t mgr,
                                              const char* role_name,
                                              int* has_role);

/**
 * @brief Get the number of ranks in a role
 *
 * Returns the number of ranks assigned to the named role group.
 *
 * @param mgr The role manager
 * @param role_name Name of the role to query
 * @param[out] size Pointer to receive the number of ranks in the role
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, initialized role manager
 * @pre role_name must be a non-empty null-terminated string
 * @pre size must not be NULL
 */
DTL_API dtl_status dtl_role_manager_role_size(dtl_role_manager_t mgr,
                                               const char* role_name,
                                               dtl_size_t* size);

/**
 * @brief Get this rank's rank within a role group
 *
 * Returns the local rank index of the calling process within the
 * named role group. Only valid if this rank belongs to the role.
 *
 * @param mgr The role manager
 * @param role_name Name of the role to query
 * @param[out] rank Pointer to receive the rank within the role group
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, initialized role manager
 * @pre role_name must be a non-empty null-terminated string
 * @pre rank must not be NULL
 * @pre This rank must belong to the specified role
 *
 * @note Returns DTL_ERROR_INVALID_ARGUMENT if this rank does not
 *       belong to the specified role.
 */
DTL_API dtl_status dtl_role_manager_role_rank(dtl_role_manager_t mgr,
                                               const char* role_name,
                                               dtl_rank_t* rank);

/* ==========================================================================
 * Inter-Group Communication
 * ========================================================================== */

/**
 * @brief Send data to a rank in another role group
 *
 * Sends count elements to a specific rank within the target role group.
 * The target_rank is the local rank index within the target role, not
 * the global MPI rank. The implementation translates the role-local rank
 * to the corresponding global rank.
 *
 * @param mgr The role manager
 * @param target_role Name of the target role group
 * @param target_rank Rank within the target role group (0 to role_size-1)
 * @param buf Pointer to send buffer (must not be NULL)
 * @param count Number of elements to send
 * @param dtype Data type of elements
 * @param tag Message tag
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, initialized role manager
 * @pre target_role must be a valid role name
 * @pre target_rank must be a valid rank within the target role
 * @pre buf must point to at least count * dtype_size(dtype) bytes
 *
 * @note Requires MPI backend. Returns DTL_ERROR_NOT_SUPPORTED without MPI.
 *
 * @code
 * // Producer sends data to consumer rank 0
 * double data[100];
 * dtl_intergroup_send(mgr, "consumer", 0, data, 100,
 *                     DTL_DTYPE_FLOAT64, 42);
 * @endcode
 */
DTL_API dtl_status dtl_intergroup_send(dtl_role_manager_t mgr,
                                        const char* target_role,
                                        dtl_rank_t target_rank,
                                        const void* buf,
                                        dtl_size_t count,
                                        dtl_dtype dtype,
                                        dtl_tag_t tag);

/**
 * @brief Receive data from a rank in another role group
 *
 * Receives count elements from a specific rank within the source role
 * group. The source_rank is the local rank index within the source role,
 * not the global MPI rank. The implementation translates the role-local
 * rank to the corresponding global rank.
 *
 * @param mgr The role manager
 * @param source_role Name of the source role group
 * @param source_rank Rank within the source role group (0 to role_size-1)
 * @param buf Pointer to receive buffer (must not be NULL)
 * @param count Number of elements to receive
 * @param dtype Data type of elements
 * @param tag Message tag
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre mgr must be a valid, initialized role manager
 * @pre source_role must be a valid role name
 * @pre source_rank must be a valid rank within the source role
 * @pre buf must point to at least count * dtype_size(dtype) bytes
 *
 * @note Requires MPI backend. Returns DTL_ERROR_NOT_SUPPORTED without MPI.
 *
 * @code
 * // Consumer receives data from producer rank 0
 * double data[100];
 * dtl_intergroup_recv(mgr, "producer", 0, data, 100,
 *                     DTL_DTYPE_FLOAT64, 42);
 * @endcode
 */
DTL_API dtl_status dtl_intergroup_recv(dtl_role_manager_t mgr,
                                        const char* source_role,
                                        dtl_rank_t source_rank,
                                        void* buf,
                                        dtl_size_t count,
                                        dtl_dtype dtype,
                                        dtl_tag_t tag);

DTL_C_END

/* Mark header as available for master include */
#define DTL_MPMD_H_AVAILABLE

#endif /* DTL_MPMD_H */
