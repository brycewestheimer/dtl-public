// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file policy_matrix.hpp
 * @brief Supported policy matrix and dispatch tables for C ABI
 * @since 0.1.0
 *
 * This header defines the supported policy matrix for C API container creation.
 * It provides validation functions and dispatch tables for mapping C enums
 * to C++ policy types.
 */

#ifndef DTL_C_DETAIL_POLICY_MATRIX_HPP
#define DTL_C_DETAIL_POLICY_MATRIX_HPP

#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/core/config.hpp>

namespace dtl::c::detail {

// ============================================================================
// Supported Policy Matrix
// ============================================================================

/**
 * @brief Default supported matrix
 *
 * Default instantiation budget:
 * - dtypes: 10 (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64)
 * - partitions: 3 (block, cyclic, replicated)
 * - placements: 3 (host, unified, device) - device/unified require CUDA
 * - executions: 2 (seq, par)
 * - consistency: 1 (bulk_synchronous only by default)
 * - error: 1 (return_status only by default)
 *
 * Total default instantiations: 10 × 3 × 3 × 2 × 1 × 1 = 180
 * (Reduced from full 600+ by limiting consistency/error policies)
 */

// ============================================================================
// Partition Policy Support
// ============================================================================

/**
 * @brief Check if a partition policy is supported in the default matrix
 */
constexpr bool is_partition_supported(dtl_partition_policy partition) noexcept {
    switch (partition) {
        case DTL_PARTITION_BLOCK:
        case DTL_PARTITION_CYCLIC:
        case DTL_PARTITION_REPLICATED:
            return true;

        case DTL_PARTITION_BLOCK_CYCLIC:
#ifdef DTL_C_ABI_ENABLE_BLOCK_CYCLIC
            return true;
#else
            return false;  // Not in default matrix
#endif

        case DTL_PARTITION_HASH:
            return false;  // Hash is for maps only

        default:
            return false;
    }
}

// ============================================================================
// Placement Policy Support
// ============================================================================

/**
 * @brief Check if a placement policy is available (build-time check)
 */
constexpr bool is_placement_available(dtl_placement_policy placement) noexcept {
    switch (placement) {
        case DTL_PLACEMENT_HOST:
            return true;  // Always available

        case DTL_PLACEMENT_DEVICE:
        case DTL_PLACEMENT_UNIFIED:
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
            return true;
#else
            return false;
#endif

        case DTL_PLACEMENT_DEVICE_PREFERRED:
#ifdef DTL_C_ABI_ENABLE_DEVICE_PREFERRED
            return true;
#else
            return false;  // Not in default matrix
#endif

        default:
            return false;
    }
}

/**
 * @brief Check if a placement policy is supported in the default matrix
 */
constexpr bool is_placement_supported(dtl_placement_policy placement) noexcept {
    // Supported ⊆ Available
    switch (placement) {
        case DTL_PLACEMENT_HOST:
            return true;

        case DTL_PLACEMENT_DEVICE:
        case DTL_PLACEMENT_UNIFIED:
            return is_placement_available(placement);

        case DTL_PLACEMENT_DEVICE_PREFERRED:
            return false;  // Opt-in only

        default:
            return false;
    }
}

// ============================================================================
// Execution Policy Support
// ============================================================================

/**
 * @brief Check if an execution policy is supported
 */
constexpr bool is_execution_supported(dtl_execution_policy execution) noexcept {
    switch (execution) {
        case DTL_EXEC_SEQ:
        case DTL_EXEC_PAR:
            return true;

        case DTL_EXEC_ASYNC:
#ifdef DTL_C_ABI_ENABLE_ASYNC
            return true;
#else
            return false;  // Async not in default C ABI
#endif

        default:
            return false;
    }
}

// ============================================================================
// Consistency Policy Support
// ============================================================================

/**
 * @brief Check if a consistency policy is supported
 */
constexpr bool is_consistency_supported(dtl_consistency_policy consistency) noexcept {
    switch (consistency) {
        case DTL_CONSISTENCY_BULK_SYNCHRONOUS:
            return true;  // Default, always supported

        case DTL_CONSISTENCY_RELAXED:
#ifdef DTL_C_ABI_ENABLE_RELAXED_CONSISTENCY
            return true;
#else
            return false;
#endif

        case DTL_CONSISTENCY_RELEASE_ACQUIRE:
        case DTL_CONSISTENCY_SEQUENTIAL:
#ifdef DTL_C_ABI_ENABLE_FULL_CONSISTENCY
            return true;
#else
            return false;
#endif

        default:
            return false;
    }
}

// ============================================================================
// Error Policy Support
// ============================================================================

/**
 * @brief Check if an error policy is supported
 */
constexpr bool is_error_policy_supported(dtl_error_policy error) noexcept {
    switch (error) {
        case DTL_ERROR_POLICY_RETURN_STATUS:
            return true;  // Default, always supported

        case DTL_ERROR_POLICY_CALLBACK:
            return true;  // Callback is always available

        case DTL_ERROR_POLICY_TERMINATE:
#ifdef DTL_C_ABI_ENABLE_TERMINATE_ON_ERROR
            return true;
#else
            return true;  // Actually, terminate is safe to enable by default
#endif

        default:
            return false;
    }
}

// ============================================================================
// Combined Validation
// ============================================================================

/**
 * @brief Validate a complete policy combination
 * @return DTL_SUCCESS if valid, DTL_ERROR_NOT_SUPPORTED otherwise
 */
inline dtl_status validate_policy_combination(
    dtl_partition_policy partition,
    dtl_placement_policy placement,
    dtl_execution_policy execution,
    dtl_consistency_policy consistency,
    dtl_error_policy error) noexcept {

    if (!is_partition_supported(partition)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!is_placement_available(placement)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!is_placement_supported(placement)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!is_execution_supported(execution)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!is_consistency_supported(consistency)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
    if (!is_error_policy_supported(error)) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

    // Additional cross-policy validation
    // Device placement with async execution requires special handling
    if (placement == DTL_PLACEMENT_DEVICE && execution == DTL_EXEC_ASYNC) {
#ifndef DTL_C_ABI_ENABLE_DEVICE_ASYNC
        return DTL_ERROR_NOT_SUPPORTED;
#endif
    }

    return DTL_SUCCESS;
}

/**
 * @brief Validate device_id against placement
 * @return DTL_SUCCESS if valid, error code otherwise
 */
inline dtl_status validate_device_id(dtl_placement_policy placement, int device_id) noexcept {
    // device_id is only meaningful for device placements
    if (placement == DTL_PLACEMENT_DEVICE ||
        placement == DTL_PLACEMENT_UNIFIED ||
        placement == DTL_PLACEMENT_DEVICE_PREFERRED) {

        // device_id -1 means "use current context device"
        // Other values must be non-negative
        if (device_id < -1) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }
    }
    return DTL_SUCCESS;
}

/**
 * @brief Validate block_size for block-cyclic partition
 */
inline dtl_status validate_block_size(dtl_partition_policy partition, dtl_size_t block_size) noexcept {
    if (partition == DTL_PARTITION_BLOCK_CYCLIC) {
        if (block_size == 0) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }
    }
    return DTL_SUCCESS;
}

}  // namespace dtl::c::detail

#endif /* DTL_C_DETAIL_POLICY_MATRIX_HPP */
