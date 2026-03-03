// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_types.cpp
 * @brief DTL C bindings - Type helper implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_config.h>

#include <cstring>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

extern "C" {

// ============================================================================
// Version Query Functions
// ============================================================================

int dtl_version_major(void) {
    return DTL_VERSION_MAJOR;
}

int dtl_version_minor(void) {
    return DTL_VERSION_MINOR;
}

int dtl_version_patch(void) {
    return DTL_VERSION_PATCH;
}

int dtl_abi_version(void) {
    return DTL_ABI_VERSION;
}

const char* dtl_version_string(void) {
    return DTL_VERSION_STRING;
}

// ============================================================================
// Feature Query Functions
// ============================================================================

int dtl_has_mpi(void) {
#ifdef DTL_HAS_MPI
    return 1;
#else
    return 0;
#endif
}

int dtl_has_cuda(void) {
#if DTL_ENABLE_CUDA
    return 1;
#else
    return 0;
#endif
}

int dtl_has_hip(void) {
#if DTL_ENABLE_HIP
    return 1;
#else
    return 0;
#endif
}

int dtl_has_nccl(void) {
#if DTL_ENABLE_NCCL
    return 1;
#else
    return 0;
#endif
}

int dtl_has_shmem(void) {
#if DTL_ENABLE_SHMEM
    return 1;
#else
    return 0;
#endif
}

// ============================================================================
// Data Type Functions
// ============================================================================

dtl_size_t dtl_dtype_size(dtl_dtype dtype) {
    switch (dtype) {
        case DTL_DTYPE_INT8:
            return sizeof(int8_t);
        case DTL_DTYPE_INT16:
            return sizeof(int16_t);
        case DTL_DTYPE_INT32:
            return sizeof(int32_t);
        case DTL_DTYPE_INT64:
            return sizeof(int64_t);
        case DTL_DTYPE_UINT8:
            return sizeof(uint8_t);
        case DTL_DTYPE_UINT16:
            return sizeof(uint16_t);
        case DTL_DTYPE_UINT32:
            return sizeof(uint32_t);
        case DTL_DTYPE_UINT64:
            return sizeof(uint64_t);
        case DTL_DTYPE_FLOAT32:
            return sizeof(float);
        case DTL_DTYPE_FLOAT64:
            return sizeof(double);
        case DTL_DTYPE_BYTE:
            return sizeof(unsigned char);
        case DTL_DTYPE_BOOL:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

const char* dtl_dtype_name(dtl_dtype dtype) {
    switch (dtype) {
        case DTL_DTYPE_INT8:
            return "int8";
        case DTL_DTYPE_INT16:
            return "int16";
        case DTL_DTYPE_INT32:
            return "int32";
        case DTL_DTYPE_INT64:
            return "int64";
        case DTL_DTYPE_UINT8:
            return "uint8";
        case DTL_DTYPE_UINT16:
            return "uint16";
        case DTL_DTYPE_UINT32:
            return "uint32";
        case DTL_DTYPE_UINT64:
            return "uint64";
        case DTL_DTYPE_FLOAT32:
            return "float32";
        case DTL_DTYPE_FLOAT64:
            return "float64";
        case DTL_DTYPE_BYTE:
            return "byte";
        case DTL_DTYPE_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}

// ============================================================================
// Reduction Operation Functions
// ============================================================================

const char* dtl_reduce_op_name(dtl_reduce_op op) {
    switch (op) {
        case DTL_OP_SUM:
            return "sum";
        case DTL_OP_PROD:
            return "prod";
        case DTL_OP_MIN:
            return "min";
        case DTL_OP_MAX:
            return "max";
        case DTL_OP_LAND:
            return "land";
        case DTL_OP_LOR:
            return "lor";
        case DTL_OP_BAND:
            return "band";
        case DTL_OP_BOR:
            return "bor";
        case DTL_OP_LXOR:
            return "lxor";
        case DTL_OP_BXOR:
            return "bxor";
        case DTL_OP_MINLOC:
            return "minloc";
        case DTL_OP_MAXLOC:
            return "maxloc";
        default:
            return "unknown";
    }
}

// ============================================================================
// Context Options Functions
// ============================================================================

void dtl_context_options_init(dtl_context_options* opts) {
    if (opts) {
        opts->device_id = -1;     // CPU only by default
        opts->init_mpi = 1;       // Initialize MPI if needed
        opts->finalize_mpi = 0;   // Don't finalize MPI by default
        std::memset(opts->reserved, 0, sizeof(opts->reserved));
        opts->reserved[0] = DTL_DETERMINISM_THROUGHPUT;
        opts->reserved[1] = DTL_REDUCTION_SCHEDULE_IMPLEMENTATION_DEFINED;
        opts->reserved[2] = DTL_PROGRESS_ORDERING_IMPLEMENTATION_DEFINED;
    }
}

// ============================================================================
// Shape Functions
// ============================================================================

dtl_shape dtl_shape_1d(dtl_size_t dim0) {
    dtl_shape shape;
    shape.ndim = 1;
    shape.dims[0] = dim0;
    // Zero out remaining dims for safety
    for (int i = 1; i < DTL_MAX_TENSOR_RANK; ++i) {
        shape.dims[i] = 0;
    }
    return shape;
}

dtl_shape dtl_shape_2d(dtl_size_t dim0, dtl_size_t dim1) {
    dtl_shape shape;
    shape.ndim = 2;
    shape.dims[0] = dim0;
    shape.dims[1] = dim1;
    for (int i = 2; i < DTL_MAX_TENSOR_RANK; ++i) {
        shape.dims[i] = 0;
    }
    return shape;
}

dtl_shape dtl_shape_3d(dtl_size_t dim0, dtl_size_t dim1, dtl_size_t dim2) {
    dtl_shape shape;
    shape.ndim = 3;
    shape.dims[0] = dim0;
    shape.dims[1] = dim1;
    shape.dims[2] = dim2;
    for (int i = 3; i < DTL_MAX_TENSOR_RANK; ++i) {
        shape.dims[i] = 0;
    }
    return shape;
}

dtl_shape dtl_shape_nd(int ndim, const dtl_size_t* dims) {
    dtl_shape shape;
    shape.ndim = (ndim > DTL_MAX_TENSOR_RANK) ? DTL_MAX_TENSOR_RANK : ndim;
    if (ndim < 0) {
        shape.ndim = 0;
    }

    for (int i = 0; i < shape.ndim; ++i) {
        shape.dims[i] = dims ? dims[i] : 0;
    }
    for (int i = shape.ndim; i < DTL_MAX_TENSOR_RANK; ++i) {
        shape.dims[i] = 0;
    }
    return shape;
}

dtl_size_t dtl_shape_size(const dtl_shape* shape) {
    if (!shape || shape->ndim <= 0) {
        return 0;
    }

    dtl_size_t size = 1;
    for (int i = 0; i < shape->ndim; ++i) {
        size *= shape->dims[i];
    }
    return size;
}

}  // extern "C"
