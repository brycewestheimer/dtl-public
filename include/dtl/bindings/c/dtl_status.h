// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_status.h
 * @brief DTL C bindings - Status codes and error handling
 * @since 0.1.0
 *
 * This header defines status codes for DTL operations and provides
 * functions for converting status codes to human-readable messages.
 */

#ifndef DTL_STATUS_H
#define DTL_STATUS_H

#include "dtl_config.h"

DTL_C_BEGIN

/* ==========================================================================
 * Status Code Definitions
 * ========================================================================== */

/** @defgroup status_codes Status Codes
 *  @brief Status codes returned by DTL functions
 *  @{
 */

/* Success */
/** @brief Operation completed successfully */
#define DTL_SUCCESS                      0

/* Non-error sentinel values */
/** @brief Key/element not found (non-error sentinel) */
#define DTL_NOT_FOUND                    1
/** @brief Iterator past-the-end (non-error sentinel) */
#define DTL_END                          2

/* Communication errors (100-199) */
/** @brief Generic communication error */
#define DTL_ERROR_COMMUNICATION        100
/** @brief Send operation failed */
#define DTL_ERROR_SEND_FAILED          101
/** @brief Receive operation failed */
#define DTL_ERROR_RECV_FAILED          102
/** @brief Broadcast operation failed */
#define DTL_ERROR_BROADCAST_FAILED     103
/** @brief Reduce operation failed */
#define DTL_ERROR_REDUCE_FAILED        104
/** @brief Barrier synchronization failed */
#define DTL_ERROR_BARRIER_FAILED       105
/** @brief Operation timed out */
#define DTL_ERROR_TIMEOUT              106
/** @brief Operation was canceled */
#define DTL_ERROR_CANCELED             107
/** @brief Connection to peer lost */
#define DTL_ERROR_CONNECTION_LOST      108
/** @brief Remote rank has failed */
#define DTL_ERROR_RANK_FAILURE         109
/** @brief Collective operation failed */
#define DTL_ERROR_COLLECTIVE_FAILED    110
/** @brief Collective participation contract violated */
#define DTL_ERROR_COLLECTIVE_PARTICIPATION 111

/* Memory errors (200-299) */
/** @brief Generic memory error */
#define DTL_ERROR_MEMORY               200
/** @brief Memory allocation failed */
#define DTL_ERROR_ALLOCATION_FAILED    201
/** @brief Out of memory */
#define DTL_ERROR_OUT_OF_MEMORY        202
/** @brief Invalid memory pointer */
#define DTL_ERROR_INVALID_POINTER      203
/** @brief Host-device memory transfer failed */
#define DTL_ERROR_TRANSFER_FAILED      204
/** @brief Device memory error */
#define DTL_ERROR_DEVICE_MEMORY        205

/* Serialization errors (300-399) */
/** @brief Generic serialization error */
#define DTL_ERROR_SERIALIZATION        300
/** @brief Failed to serialize data */
#define DTL_ERROR_SERIALIZE_FAILED     301
/** @brief Failed to deserialize data */
#define DTL_ERROR_DESERIALIZE_FAILED   302
/** @brief Buffer too small for operation */
#define DTL_ERROR_BUFFER_TOO_SMALL     303
/** @brief Invalid data format */
#define DTL_ERROR_INVALID_FORMAT       304

/* Bounds/argument errors (400-499) */
/** @brief Generic bounds error */
#define DTL_ERROR_BOUNDS               400
/** @brief Index out of bounds */
#define DTL_ERROR_OUT_OF_BOUNDS        401
/** @brief Invalid index value */
#define DTL_ERROR_INVALID_INDEX        402
/** @brief Invalid rank identifier */
#define DTL_ERROR_INVALID_RANK         403
/** @brief Dimension count mismatch */
#define DTL_ERROR_DIMENSION_MISMATCH   404
/** @brief Extent size mismatch */
#define DTL_ERROR_EXTENT_MISMATCH      405
/** @brief Key not found in container */
#define DTL_ERROR_KEY_NOT_FOUND        406
/** @brief Value out of valid range */
#define DTL_ERROR_OUT_OF_RANGE         407
/** @brief Invalid argument provided */
#define DTL_ERROR_INVALID_ARGUMENT     410
/** @brief Null pointer passed where not allowed */
#define DTL_ERROR_NULL_POINTER         411
/** @brief Operation not supported */
#define DTL_ERROR_NOT_SUPPORTED        420

/* Backend errors (500-599) */
/** @brief Generic backend error */
#define DTL_ERROR_BACKEND              500
/** @brief Requested backend not available */
#define DTL_ERROR_BACKEND_UNAVAILABLE  501
/** @brief Backend initialization failed */
#define DTL_ERROR_BACKEND_INIT_FAILED  502
/** @brief Backend is invalid for requested operation */
#define DTL_ERROR_BACKEND_INVALID      503
/** @brief CUDA-specific error */
#define DTL_ERROR_CUDA                 510
/** @brief HIP-specific error */
#define DTL_ERROR_HIP                  520
/** @brief MPI-specific error */
#define DTL_ERROR_MPI                  530
/** @brief NCCL-specific error */
#define DTL_ERROR_NCCL                 540
/** @brief SHMEM-specific error */
#define DTL_ERROR_SHMEM                550

/* Algorithm errors (600-699) */
/** @brief Generic algorithm error */
#define DTL_ERROR_ALGORITHM            600
/** @brief Alias for algorithm error */
#define DTL_ERROR_OPERATION_FAILED     600
/** @brief Algorithm precondition not met */
#define DTL_ERROR_PRECONDITION_FAILED  601
/** @brief Algorithm postcondition not met */
#define DTL_ERROR_POSTCONDITION_FAILED 602
/** @brief Iterative algorithm failed to converge */
#define DTL_ERROR_CONVERGENCE_FAILED   603

/* Consistency errors (700-799) */
/** @brief Generic consistency error */
#define DTL_ERROR_CONSISTENCY          700
/** @brief Consistency policy violated */
#define DTL_ERROR_CONSISTENCY_VIOLATION 701
/** @brief Structure invalidated during operation */
#define DTL_ERROR_STRUCTURAL_INVALIDATION 702

/* Internal errors (900-999) */
/** @brief Internal DTL error */
#define DTL_ERROR_INTERNAL             900
/** @brief Feature not implemented */
#define DTL_ERROR_NOT_IMPLEMENTED      901
/** @brief Object in invalid state */
#define DTL_ERROR_INVALID_STATE        902
/** @brief Unknown error occurred */
#define DTL_ERROR_UNKNOWN              999

/** @} */ /* end of status_codes group */

/* ==========================================================================
 * Status Query Functions
 * ========================================================================== */

/**
 * @brief Check if status indicates success
 * @param status The status code to check
 * @return 1 if success (DTL_SUCCESS), 0 otherwise
 */
DTL_API int dtl_status_ok(dtl_status status);

/**
 * @brief Check if status indicates an error
 * @param status The status code to check
 * @return 1 if error (not DTL_SUCCESS), 0 otherwise
 */
DTL_API int dtl_status_is_error(dtl_status status);

/**
 * @brief Get human-readable message for status code
 * @param status The status code
 * @return Static string describing the status (never NULL)
 *
 * @note The returned string is statically allocated and must not be freed.
 */
DTL_API const char* dtl_status_message(dtl_status status);

/**
 * @brief Get short name for status code
 * @param status The status code
 * @return Static string with code name (e.g., "ok", "communication_error")
 *
 * @note The returned string is statically allocated and must not be freed.
 */
DTL_API const char* dtl_status_name(dtl_status status);

/**
 * @brief Get category name for status code
 * @param status The status code
 * @return Static string with category (e.g., "success", "communication", "memory")
 *
 * @note The returned string is statically allocated and must not be freed.
 */
DTL_API const char* dtl_status_category(dtl_status status);

/**
 * @brief Get category code for status
 * @param status The status code
 * @return Category as integer (status / 100)
 *
 * Categories:
 * - 0: Success
 * - 1: Communication
 * - 2: Memory
 * - 3: Serialization
 * - 4: Bounds/argument
 * - 5: Backend
 * - 6: Algorithm
 * - 7: Consistency
 * - 9: Internal
 */
DTL_API int dtl_status_category_code(dtl_status status);

/**
 * @brief Check if status is in a specific category
 * @param status The status code
 * @param category_code The category code (0-9)
 * @return 1 if status is in the category, 0 otherwise
 */
DTL_API int dtl_status_is_category(dtl_status status, int category_code);

/* ==========================================================================
 * Status Category Constants
 * ========================================================================== */

/** @brief Category code for success */
#define DTL_CATEGORY_SUCCESS        0
/** @brief Category code for communication errors */
#define DTL_CATEGORY_COMMUNICATION  1
/** @brief Category code for memory errors */
#define DTL_CATEGORY_MEMORY         2
/** @brief Category code for serialization errors */
#define DTL_CATEGORY_SERIALIZATION  3
/** @brief Category code for bounds/argument errors */
#define DTL_CATEGORY_BOUNDS         4
/** @brief Category code for backend errors */
#define DTL_CATEGORY_BACKEND        5
/** @brief Category code for algorithm errors */
#define DTL_CATEGORY_ALGORITHM      6
/** @brief Category code for consistency errors */
#define DTL_CATEGORY_CONSISTENCY    7
/** @brief Category code for internal errors */
#define DTL_CATEGORY_INTERNAL       9

DTL_C_END

#endif /* DTL_STATUS_H */
