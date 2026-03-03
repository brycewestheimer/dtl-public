// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_policies.h
 * @brief DTL C bindings - Policy definitions and container options
 * @since 0.1.0
 *
 * This header provides C API access to DTL's policy system, allowing
 * containers to be created with specific partition, placement, and
 * execution policies.
 */

#ifndef DTL_POLICIES_H
#define DTL_POLICIES_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_config.h"

DTL_C_BEGIN

/* ==========================================================================
 * Partition Policy Enumeration
 * ========================================================================== */

/**
 * @brief Partition policy types
 *
 * Determines how data is distributed across ranks.
 */
typedef enum dtl_partition_policy {
    /** @brief Block partition - contiguous chunks per rank (default) */
    DTL_PARTITION_BLOCK = 0,

    /** @brief Cyclic partition - round-robin distribution */
    DTL_PARTITION_CYCLIC = 1,

    /** @brief Block-cyclic partition - blocks distributed cyclically */
    DTL_PARTITION_BLOCK_CYCLIC = 2,

    /** @brief Hash partition - hash-based distribution (for maps) */
    DTL_PARTITION_HASH = 3,

    /** @brief Replicated - full copy on each rank */
    DTL_PARTITION_REPLICATED = 4,

    /** @brief Number of partition policies */
    DTL_PARTITION_COUNT = 5
} dtl_partition_policy;

/**
 * @brief Get the name of a partition policy
 * @param policy The partition policy
 * @return String name (e.g., "block"), or "unknown" for invalid policy
 */
DTL_API const char* dtl_partition_policy_name(dtl_partition_policy policy);

/* ==========================================================================
 * Placement Policy Enumeration
 * ========================================================================== */

/**
 * @brief Placement policy types
 *
 * Determines where data is stored (CPU vs GPU memory).
 */
typedef enum dtl_placement_policy {
    /** @brief Host-only memory (CPU, default) */
    DTL_PLACEMENT_HOST = 0,

    /** @brief Device-only memory (GPU) - requires DTL_HAS_CUDA */
    DTL_PLACEMENT_DEVICE = 1,

    /** @brief Unified/managed memory - requires DTL_HAS_CUDA */
    DTL_PLACEMENT_UNIFIED = 2,

    /** @brief Device-preferred with host fallback - requires DTL_HAS_CUDA */
    DTL_PLACEMENT_DEVICE_PREFERRED = 3,

    /** @brief Number of placement policies */
    DTL_PLACEMENT_COUNT = 4
} dtl_placement_policy;

/**
 * @brief Get the name of a placement policy
 * @param policy The placement policy
 * @return String name (e.g., "host"), or "unknown" for invalid policy
 */
DTL_API const char* dtl_placement_policy_name(dtl_placement_policy policy);

/**
 * @brief Check if a placement policy is available in this build
 *
 * Some placements (device, unified, device_preferred) require CUDA support.
 *
 * @param policy The placement policy to check
 * @return 1 if available, 0 if not available
 */
DTL_API int dtl_placement_available(dtl_placement_policy policy);

/* ==========================================================================
 * Execution Policy Enumeration
 * ========================================================================== */

/**
 * @brief Execution policy types
 *
 * Determines how algorithm operations are executed.
 */
typedef enum dtl_execution_policy {
    /** @brief Sequential execution (blocking, single-threaded) */
    DTL_EXEC_SEQ = 0,

    /** @brief Parallel execution (blocking, multi-threaded) */
    DTL_EXEC_PAR = 1,

    /** @brief Asynchronous execution (non-blocking, returns future) */
    DTL_EXEC_ASYNC = 2,

    /** @brief Number of execution policies */
    DTL_EXEC_COUNT = 3
} dtl_execution_policy;

/**
 * @brief Get the name of an execution policy
 * @param policy The execution policy
 * @return String name (e.g., "seq"), or "unknown" for invalid policy
 */
DTL_API const char* dtl_execution_policy_name(dtl_execution_policy policy);

/* ==========================================================================
 * Consistency Policy Enumeration
 * ========================================================================== */

/**
 * @brief Consistency policy types
 *
 * Determines memory consistency guarantees for distributed operations.
 * @since 0.1.0
 */
typedef enum dtl_consistency_policy {
    /** @brief Bulk-synchronous consistency (default) - all operations complete at barriers */
    DTL_CONSISTENCY_BULK_SYNCHRONOUS = 0,

    /** @brief Relaxed consistency - no ordering guarantees between operations */
    DTL_CONSISTENCY_RELAXED = 1,

    /** @brief Release-acquire consistency - synchronizes at release/acquire pairs */
    DTL_CONSISTENCY_RELEASE_ACQUIRE = 2,

    /** @brief Sequential consistency - total order of all operations */
    DTL_CONSISTENCY_SEQUENTIAL = 3,

    /** @brief Number of consistency policies */
    DTL_CONSISTENCY_COUNT = 4
} dtl_consistency_policy;

/**
 * @brief Get the name of a consistency policy
 * @param policy The consistency policy
 * @return String name (e.g., "bulk_synchronous"), or "unknown" for invalid policy
 * @since 0.1.0
 */
DTL_API const char* dtl_consistency_policy_name(dtl_consistency_policy policy);

/* ==========================================================================
 * Error Policy Enumeration
 * ========================================================================== */

/**
 * @brief Error policy types
 *
 * Determines how errors are reported and handled in C API operations.
 * @since 0.1.0
 */
typedef enum dtl_error_policy {
    /** @brief Return status codes (default) - caller checks return values */
    DTL_ERROR_POLICY_RETURN_STATUS = 0,

    /** @brief Invoke callback on error - callback is called before returning status */
    DTL_ERROR_POLICY_CALLBACK = 1,

    /** @brief Terminate on error - calls abort() on any error */
    DTL_ERROR_POLICY_TERMINATE = 2,

    /** @brief Number of error policies */
    DTL_ERROR_POLICY_COUNT = 3
} dtl_error_policy;

/**
 * @brief Get the name of an error policy
 * @param policy The error policy
 * @return String name (e.g., "return_status"), or "unknown" for invalid policy
 * @since 0.1.0
 */
DTL_API const char* dtl_error_policy_name(dtl_error_policy policy);

/**
 * @brief Error handler callback type
 *
 * Called when an error occurs and error policy is DTL_ERROR_POLICY_CALLBACK.
 *
 * @param status The error status code
 * @param message Human-readable error message (may be NULL)
 * @param user_data User-provided context pointer
 * @since 0.1.0
 */
typedef void (*dtl_error_handler_t)(dtl_status status, const char* message, void* user_data);

/**
 * @brief Set the error handler callback for a context
 *
 * When error policy is DTL_ERROR_POLICY_CALLBACK, this handler is invoked
 * on any error before the status is returned to the caller.
 *
 * @param ctx The context
 * @param handler The error handler callback (NULL to clear)
 * @param user_data User data passed to the handler
 * @return DTL_SUCCESS on success, error code otherwise
 * @since 0.1.0
 */
DTL_API dtl_status dtl_context_set_error_handler(
    dtl_context_t ctx,
    dtl_error_handler_t handler,
    void* user_data);

/* ==========================================================================
 * Container Options
 * ========================================================================== */

/**
 * @brief Options for policy-aware container creation
 *
 * Pass to dtl_vector_create_with_options() or dtl_array_create_with_options()
 * to specify policies at creation time.
 *
 * @note Use dtl_container_options_init() to initialize with defaults.
 * @note The reserved fields store consistency (reserved[0]) and error (reserved[1])
 *       policies. Use the provided accessor macros for clarity.
 *
 * @since 0.1.0 (consistency/error added in 1.2.0)
 */
typedef struct dtl_container_options {
    /** @brief Partition policy (default: DTL_PARTITION_BLOCK) */
    dtl_partition_policy partition;

    /** @brief Placement policy (default: DTL_PLACEMENT_HOST) */
    dtl_placement_policy placement;

    /** @brief Execution policy (default: DTL_EXEC_SEQ) */
    dtl_execution_policy execution;

    /** @brief GPU device ID (only used with device placements, default: 0) */
    int device_id;

    /** @brief Block size for block-cyclic partition (default: 1) */
    dtl_size_t block_size;

    /**
     * @brief Reserved fields for ABI-compatible extension
     *
     * reserved[0]: Consistency policy (dtl_consistency_policy, default: 0 = BULK_SYNCHRONOUS)
     * reserved[1]: Error policy (dtl_error_policy, default: 0 = RETURN_STATUS)
     * reserved[2]: Reserved for future use (must be 0)
     */
    int reserved[3];
} dtl_container_options;

/**
 * @brief Get consistency policy from container options
 * @param opts Pointer to options structure
 * @return The consistency policy
 * @since 0.1.0
 */
#define dtl_container_options_consistency(opts) \
    ((dtl_consistency_policy)((opts)->reserved[0]))

/**
 * @brief Set consistency policy in container options
 * @param opts Pointer to options structure
 * @param policy The consistency policy to set
 * @since 0.1.0
 */
#define dtl_container_options_set_consistency(opts, policy) \
    ((opts)->reserved[0] = (int)(policy))

/**
 * @brief Get error policy from container options
 * @param opts Pointer to options structure
 * @return The error policy
 * @since 0.1.0
 */
#define dtl_container_options_error(opts) \
    ((dtl_error_policy)((opts)->reserved[1]))

/**
 * @brief Set error policy in container options
 * @param opts Pointer to options structure
 * @param policy The error policy to set
 * @since 0.1.0
 */
#define dtl_container_options_set_error(opts, policy) \
    ((opts)->reserved[1] = (int)(policy))

/**
 * @brief Initialize container options to default values
 *
 * Sets all options to their default values:
 * - partition: DTL_PARTITION_BLOCK
 * - placement: DTL_PLACEMENT_HOST
 * - execution: DTL_EXEC_SEQ
 * - device_id: 0
 * - block_size: 1
 *
 * @param opts Pointer to options structure
 */
DTL_API void dtl_container_options_init(dtl_container_options* opts);

/* ==========================================================================
 * Policy-Aware Container Creation
 * ========================================================================== */

/**
 * @brief Create a distributed vector with specified policies
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param global_size Total number of elements
 * @param opts Container options (or NULL for defaults)
 * @param[out] vec Pointer to receive the vector handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @code
 * dtl_container_options opts;
 * dtl_container_options_init(&opts);
 * opts.partition = DTL_PARTITION_CYCLIC;
 * opts.placement = DTL_PLACEMENT_HOST;
 *
 * dtl_vector_t vec;
 * dtl_status status = dtl_vector_create_with_options(
 *     ctx, DTL_DTYPE_FLOAT64, 1000, &opts, &vec);
 * @endcode
 */
DTL_API dtl_status dtl_vector_create_with_options(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t global_size,
    const dtl_container_options* opts,
    dtl_vector_t* vec);

/**
 * @brief Create a distributed array with specified policies
 *
 * @param ctx The context
 * @param dtype Data type of elements
 * @param size Total number of elements (fixed)
 * @param opts Container options (or NULL for defaults)
 * @param[out] arr Pointer to receive the array handle
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_array_create_with_options(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t size,
    const dtl_container_options* opts,
    dtl_array_t* arr);

/* ==========================================================================
 * Policy Query Functions
 * ========================================================================== */

/**
 * @brief Get the partition policy of a vector
 *
 * @param vec The vector
 * @return Partition policy, or -1 on error
 */
DTL_API dtl_partition_policy dtl_vector_partition_policy(dtl_vector_t vec);

/**
 * @brief Get the placement policy of a vector
 *
 * @param vec The vector
 * @return Placement policy, or -1 on error
 */
DTL_API dtl_placement_policy dtl_vector_placement_policy(dtl_vector_t vec);

/**
 * @brief Get the partition policy of an array
 *
 * @param arr The array
 * @return Partition policy, or -1 on error
 */
DTL_API dtl_partition_policy dtl_array_partition_policy(dtl_array_t arr);

/**
 * @brief Get the placement policy of an array
 *
 * @param arr The array
 * @return Placement policy, or -1 on error
 */
DTL_API dtl_placement_policy dtl_array_placement_policy(dtl_array_t arr);

/**
 * @brief Get the execution policy of a vector
 *
 * @param vec The vector
 * @return Execution policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_execution_policy dtl_vector_execution_policy(dtl_vector_t vec);

/**
 * @brief Get the device ID of a vector
 *
 * @param vec The vector
 * @return Device ID (0 for host, or GPU device ID for device placements), or -1 on error
 * @since 0.1.0
 */
DTL_API int dtl_vector_device_id(dtl_vector_t vec);

/**
 * @brief Get the consistency policy of a vector
 *
 * @param vec The vector
 * @return Consistency policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_consistency_policy dtl_vector_consistency_policy(dtl_vector_t vec);

/**
 * @brief Get the error policy of a vector
 *
 * @param vec The vector
 * @return Error policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_error_policy dtl_vector_error_policy(dtl_vector_t vec);

/**
 * @brief Get the execution policy of an array
 *
 * @param arr The array
 * @return Execution policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_execution_policy dtl_array_execution_policy(dtl_array_t arr);

/**
 * @brief Get the device ID of an array
 *
 * @param arr The array
 * @return Device ID, or -1 on error
 * @since 0.1.0
 */
DTL_API int dtl_array_device_id(dtl_array_t arr);

/**
 * @brief Get the consistency policy of an array
 *
 * @param arr The array
 * @return Consistency policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_consistency_policy dtl_array_consistency_policy(dtl_array_t arr);

/**
 * @brief Get the error policy of an array
 *
 * @param arr The array
 * @return Error policy, or -1 on error
 * @since 0.1.0
 */
DTL_API dtl_error_policy dtl_array_error_policy(dtl_array_t arr);

/* ==========================================================================
 * Copy Helpers for Device Placement
 * ========================================================================== */

/**
 * @brief Copy local vector data to a host buffer
 *
 * For device-only placement, this performs a device-to-host copy.
 * For host placement, this is equivalent to memcpy.
 *
 * @param vec The vector
 * @param host_buffer Destination buffer on host
 * @param count Number of elements to copy (0 = copy all local elements)
 * @return DTL_SUCCESS on success, error code otherwise
 * @since 0.1.0
 */
DTL_API dtl_status dtl_vector_copy_to_host(
    dtl_vector_t vec,
    void* host_buffer,
    dtl_size_t count);

/**
 * @brief Copy data from a host buffer to local vector data
 *
 * For device-only placement, this performs a host-to-device copy.
 * For host placement, this is equivalent to memcpy.
 *
 * @param vec The vector
 * @param host_buffer Source buffer on host
 * @param count Number of elements to copy (0 = copy all local elements)
 * @return DTL_SUCCESS on success, error code otherwise
 * @since 0.1.0
 */
DTL_API dtl_status dtl_vector_copy_from_host(
    dtl_vector_t vec,
    const void* host_buffer,
    dtl_size_t count);

/**
 * @brief Copy local array data to a host buffer
 *
 * @param arr The array
 * @param host_buffer Destination buffer on host
 * @param count Number of elements to copy (0 = copy all local elements)
 * @return DTL_SUCCESS on success, error code otherwise
 * @since 0.1.0
 */
DTL_API dtl_status dtl_array_copy_to_host(
    dtl_array_t arr,
    void* host_buffer,
    dtl_size_t count);

/**
 * @brief Copy data from a host buffer to local array data
 *
 * @param arr The array
 * @param host_buffer Source buffer on host
 * @param count Number of elements to copy (0 = copy all local elements)
 * @return DTL_SUCCESS on success, error code otherwise
 * @since 0.1.0
 */
DTL_API dtl_status dtl_array_copy_from_host(
    dtl_array_t arr,
    const void* host_buffer,
    dtl_size_t count);

DTL_C_END

/* Mark header as available for master include */
#define DTL_POLICIES_H_AVAILABLE

#endif /* DTL_POLICIES_H */
