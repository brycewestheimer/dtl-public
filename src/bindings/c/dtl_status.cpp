// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_status.cpp
 * @brief DTL C bindings - Status code implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_config.h>

// Define DTL_C_EXPORTS to export symbols
#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

extern "C" {

int dtl_status_ok(dtl_status status) {
    // DTL_SUCCESS, DTL_NOT_FOUND, and DTL_END are all non-error
    return (status < 100) ? 1 : 0;
}

int dtl_status_is_error(dtl_status status) {
    // Non-error sentinels: DTL_SUCCESS (0), DTL_NOT_FOUND (1), DTL_END (2)
    return (status >= 100) ? 1 : 0;
}

const char* dtl_status_message(dtl_status status) {
    switch (status) {
        case DTL_SUCCESS:
            return "Operation completed successfully";
        case DTL_NOT_FOUND:
            return "Key or element not found";
        case DTL_END:
            return "Iterator past the end";

        // Communication errors
        case DTL_ERROR_COMMUNICATION:
            return "Communication error";
        case DTL_ERROR_SEND_FAILED:
            return "Send operation failed";
        case DTL_ERROR_RECV_FAILED:
            return "Receive operation failed";
        case DTL_ERROR_BROADCAST_FAILED:
            return "Broadcast operation failed";
        case DTL_ERROR_REDUCE_FAILED:
            return "Reduce operation failed";
        case DTL_ERROR_BARRIER_FAILED:
            return "Barrier synchronization failed";
        case DTL_ERROR_TIMEOUT:
            return "Operation timed out";
        case DTL_ERROR_CANCELED:
            return "Operation was canceled";
        case DTL_ERROR_CONNECTION_LOST:
            return "Connection to peer lost";
        case DTL_ERROR_RANK_FAILURE:
            return "Remote rank has failed";
        case DTL_ERROR_COLLECTIVE_FAILED:
            return "Collective operation failed";
        case DTL_ERROR_COLLECTIVE_PARTICIPATION:
            return "Collective participation contract violated";

        // Memory errors
        case DTL_ERROR_MEMORY:
            return "Memory error";
        case DTL_ERROR_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case DTL_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case DTL_ERROR_INVALID_POINTER:
            return "Invalid memory pointer";
        case DTL_ERROR_TRANSFER_FAILED:
            return "Host-device memory transfer failed";
        case DTL_ERROR_DEVICE_MEMORY:
            return "Device memory error";

        // Serialization errors
        case DTL_ERROR_SERIALIZATION:
            return "Serialization error";
        case DTL_ERROR_SERIALIZE_FAILED:
            return "Failed to serialize data";
        case DTL_ERROR_DESERIALIZE_FAILED:
            return "Failed to deserialize data";
        case DTL_ERROR_BUFFER_TOO_SMALL:
            return "Buffer too small for operation";
        case DTL_ERROR_INVALID_FORMAT:
            return "Invalid data format";

        // Bounds/argument errors
        case DTL_ERROR_BOUNDS:
            return "Bounds error";
        case DTL_ERROR_OUT_OF_BOUNDS:
            return "Index out of bounds";
        case DTL_ERROR_INVALID_INDEX:
            return "Invalid index value";
        case DTL_ERROR_INVALID_RANK:
            return "Invalid rank identifier";
        case DTL_ERROR_DIMENSION_MISMATCH:
            return "Dimension count mismatch";
        case DTL_ERROR_EXTENT_MISMATCH:
            return "Extent size mismatch";
        case DTL_ERROR_KEY_NOT_FOUND:
            return "Key not found in container";
        case DTL_ERROR_OUT_OF_RANGE:
            return "Value out of valid range";
        case DTL_ERROR_INVALID_ARGUMENT:
            return "Invalid argument provided";
        case DTL_ERROR_NULL_POINTER:
            return "Null pointer passed where not allowed";
        case DTL_ERROR_NOT_SUPPORTED:
            return "Operation not supported";

        // Backend errors
        case DTL_ERROR_BACKEND:
            return "Backend error";
        case DTL_ERROR_BACKEND_UNAVAILABLE:
            return "Requested backend not available";
        case DTL_ERROR_BACKEND_INIT_FAILED:
            return "Backend initialization failed";
        case DTL_ERROR_BACKEND_INVALID:
            return "Backend is invalid for requested operation";
        case DTL_ERROR_CUDA:
            return "CUDA error";
        case DTL_ERROR_HIP:
            return "HIP error";
        case DTL_ERROR_MPI:
            return "MPI error";
        case DTL_ERROR_NCCL:
            return "NCCL error";
        case DTL_ERROR_SHMEM:
            return "SHMEM error";

        // Algorithm errors
        case DTL_ERROR_ALGORITHM:
            return "Algorithm error";
        case DTL_ERROR_PRECONDITION_FAILED:
            return "Algorithm precondition not met";
        case DTL_ERROR_POSTCONDITION_FAILED:
            return "Algorithm postcondition not met";
        case DTL_ERROR_CONVERGENCE_FAILED:
            return "Iterative algorithm failed to converge";

        // Consistency errors
        case DTL_ERROR_CONSISTENCY:
            return "Consistency error";
        case DTL_ERROR_CONSISTENCY_VIOLATION:
            return "Consistency policy violated";
        case DTL_ERROR_STRUCTURAL_INVALIDATION:
            return "Structure invalidated during operation";

        // Internal errors
        case DTL_ERROR_INTERNAL:
            return "Internal DTL error";
        case DTL_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case DTL_ERROR_INVALID_STATE:
            return "Object in invalid state";
        case DTL_ERROR_UNKNOWN:
            return "Unknown error occurred";

        default:
            return "Unrecognized error code";
    }
}

const char* dtl_status_name(dtl_status status) {
    switch (status) {
        case DTL_SUCCESS:
            return "ok";
        case DTL_NOT_FOUND:
            return "not_found";
        case DTL_END:
            return "end_iterator";

        // Communication
        case DTL_ERROR_COMMUNICATION:
            return "communication_error";
        case DTL_ERROR_SEND_FAILED:
            return "send_failed";
        case DTL_ERROR_RECV_FAILED:
            return "recv_failed";
        case DTL_ERROR_BROADCAST_FAILED:
            return "broadcast_failed";
        case DTL_ERROR_REDUCE_FAILED:
            return "reduce_failed";
        case DTL_ERROR_BARRIER_FAILED:
            return "barrier_failed";
        case DTL_ERROR_TIMEOUT:
            return "timeout";
        case DTL_ERROR_CANCELED:
            return "canceled";
        case DTL_ERROR_CONNECTION_LOST:
            return "connection_lost";
        case DTL_ERROR_RANK_FAILURE:
            return "rank_failure";
        case DTL_ERROR_COLLECTIVE_FAILED:
            return "collective_failure";
        case DTL_ERROR_COLLECTIVE_PARTICIPATION:
            return "collective_participation_error";

        // Memory
        case DTL_ERROR_MEMORY:
            return "memory_error";
        case DTL_ERROR_ALLOCATION_FAILED:
            return "allocation_failed";
        case DTL_ERROR_OUT_OF_MEMORY:
            return "out_of_memory";
        case DTL_ERROR_INVALID_POINTER:
            return "invalid_pointer";
        case DTL_ERROR_TRANSFER_FAILED:
            return "memory_transfer_failed";
        case DTL_ERROR_DEVICE_MEMORY:
            return "device_memory_error";

        // Serialization
        case DTL_ERROR_SERIALIZATION:
            return "serialization_error";
        case DTL_ERROR_SERIALIZE_FAILED:
            return "serialize_failed";
        case DTL_ERROR_DESERIALIZE_FAILED:
            return "deserialize_failed";
        case DTL_ERROR_BUFFER_TOO_SMALL:
            return "buffer_too_small";
        case DTL_ERROR_INVALID_FORMAT:
            return "invalid_format";

        // Bounds
        case DTL_ERROR_BOUNDS:
            return "bounds_error";
        case DTL_ERROR_OUT_OF_BOUNDS:
            return "out_of_bounds";
        case DTL_ERROR_INVALID_INDEX:
            return "invalid_index";
        case DTL_ERROR_INVALID_RANK:
            return "invalid_rank";
        case DTL_ERROR_DIMENSION_MISMATCH:
            return "dimension_mismatch";
        case DTL_ERROR_EXTENT_MISMATCH:
            return "extent_mismatch";
        case DTL_ERROR_KEY_NOT_FOUND:
            return "key_not_found";
        case DTL_ERROR_OUT_OF_RANGE:
            return "out_of_range";
        case DTL_ERROR_INVALID_ARGUMENT:
            return "invalid_argument";
        case DTL_ERROR_NULL_POINTER:
            return "null_pointer";
        case DTL_ERROR_NOT_SUPPORTED:
            return "not_supported";

        // Backend
        case DTL_ERROR_BACKEND:
            return "backend_error";
        case DTL_ERROR_BACKEND_UNAVAILABLE:
            return "backend_not_available";
        case DTL_ERROR_BACKEND_INIT_FAILED:
            return "backend_init_failed";
        case DTL_ERROR_BACKEND_INVALID:
            return "backend_invalid";
        case DTL_ERROR_CUDA:
            return "cuda_error";
        case DTL_ERROR_HIP:
            return "hip_error";
        case DTL_ERROR_MPI:
            return "mpi_error";
        case DTL_ERROR_NCCL:
            return "nccl_error";
        case DTL_ERROR_SHMEM:
            return "shmem_error";

        // Algorithm
        case DTL_ERROR_ALGORITHM:
            return "algorithm_error";
        case DTL_ERROR_PRECONDITION_FAILED:
            return "precondition_failed";
        case DTL_ERROR_POSTCONDITION_FAILED:
            return "postcondition_failed";
        case DTL_ERROR_CONVERGENCE_FAILED:
            return "convergence_failed";

        // Consistency
        case DTL_ERROR_CONSISTENCY:
            return "consistency_error";
        case DTL_ERROR_CONSISTENCY_VIOLATION:
            return "consistency_violation";
        case DTL_ERROR_STRUCTURAL_INVALIDATION:
            return "structural_invalidation";

        // Internal
        case DTL_ERROR_INTERNAL:
            return "internal_error";
        case DTL_ERROR_NOT_IMPLEMENTED:
            return "not_implemented";
        case DTL_ERROR_INVALID_STATE:
            return "invalid_state";
        case DTL_ERROR_UNKNOWN:
            return "unknown_error";

        default:
            return "unknown";
    }
}

const char* dtl_status_category(dtl_status status) {
    if (status >= 0 && status < 100) {
        return "success";
    }

    int category = status / 100;
    switch (category) {
        case 1:
            return "communication";
        case 2:
            return "memory";
        case 3:
            return "serialization";
        case 4:
            return "bounds";
        case 5:
            return "backend";
        case 6:
            return "algorithm";
        case 7:
            return "consistency";
        case 9:
            return "internal";
        default:
            return "unknown";
    }
}

int dtl_status_category_code(dtl_status status) {
    if (status >= 0 && status < 100) {
        return 0;
    }
    return status / 100;
}

int dtl_status_is_category(dtl_status status, int category_code) {
    return dtl_status_category_code(status) == category_code ? 1 : 0;
}

}  // extern "C"
