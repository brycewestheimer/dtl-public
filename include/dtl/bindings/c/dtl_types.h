// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_types.h
 * @brief DTL C bindings - Core type definitions
 * @since 0.1.0
 *
 * This header defines the core types used throughout the DTL C API,
 * including opaque handle types, enumerations, and structure definitions.
 */

#ifndef DTL_TYPES_H
#define DTL_TYPES_H

#include "dtl_config.h"

DTL_C_BEGIN

/* ==========================================================================
 * Opaque Handle Types
 * ========================================================================== */

/** @brief Forward declaration for environment implementation */
struct dtl_environment_s;

/** @brief Forward declaration for context implementation */
struct dtl_context_s;

/** @brief Forward declaration for communicator implementation */
struct dtl_communicator_s;

/** @brief Forward declaration for vector implementation */
struct dtl_vector_s;

/** @brief Forward declaration for array implementation */
struct dtl_array_s;

/** @brief Forward declaration for tensor implementation */
struct dtl_tensor_s;

/** @brief Forward declaration for distributed span implementation */
struct dtl_span_s;

/** @brief Forward declaration for map implementation */
struct dtl_map_s;

/** @brief Forward declaration for async request implementation */
struct dtl_request_s;

/** @brief Forward declaration for memory window implementation */
struct dtl_window_s;

/**
 * @brief Opaque handle to DTL environment
 *
 * An environment manages the lifecycle of all DTL backends (MPI, CUDA, etc.).
 * Multiple environment handles may coexist; the first creation initializes
 * backends and the last destruction finalizes them (reference counted).
 * @since 0.1.0
 */
typedef struct dtl_environment_s* dtl_environment_t;

/**
 * @brief Opaque handle to DTL context
 *
 * A context encapsulates the MPI communicator, device selection,
 * and other execution environment state.
 */
typedef struct dtl_context_s* dtl_context_t;

/**
 * @brief Opaque handle to DTL communicator
 *
 * Wraps communication operations. Usually obtained from a context.
 */
typedef struct dtl_communicator_s* dtl_communicator_t;

/**
 * @brief Opaque handle to distributed vector
 *
 * A 1D distributed container similar to std::vector.
 */
typedef struct dtl_vector_s* dtl_vector_t;

/**
 * @brief Opaque handle to distributed array
 *
 * A fixed-size 1D distributed container similar to std::array.
 * Unlike vector, arrays cannot be resized after creation.
 */
typedef struct dtl_array_s* dtl_array_t;

/**
 * @brief Opaque handle to distributed tensor
 *
 * An N-dimensional distributed container.
 */
typedef struct dtl_tensor_s* dtl_tensor_t;

/**
 * @brief Opaque handle to distributed span
 *
 * A non-owning distributed view over rank-local contiguous memory
 * with associated global-size and rank metadata.
 */
typedef struct dtl_span_s* dtl_span_t;

/**
 * @brief Opaque handle to distributed map
 *
 * A hash-based distributed associative container.
 * @since 0.1.0
 */
typedef struct dtl_map_s* dtl_map_t;

/**
 * @brief Opaque handle to asynchronous request
 *
 * Represents a pending asynchronous operation.
 */
typedef struct dtl_request_s* dtl_request_t;

/**
 * @brief Opaque handle to memory window
 *
 * Represents an RMA memory window for one-sided communication.
 */
typedef struct dtl_window_s* dtl_window_t;

/* ==========================================================================
 * Data Type Enumeration
 * ========================================================================== */

/**
 * @brief Enumeration of supported data types
 *
 * These correspond to C/C++ scalar types and are used for type-erased
 * container operations and communication.
 */
typedef enum dtl_dtype {
    DTL_DTYPE_INT8     = 0,   /**< Signed 8-bit integer (int8_t) */
    DTL_DTYPE_INT16    = 1,   /**< Signed 16-bit integer (int16_t) */
    DTL_DTYPE_INT32    = 2,   /**< Signed 32-bit integer (int32_t) */
    DTL_DTYPE_INT64    = 3,   /**< Signed 64-bit integer (int64_t) */
    DTL_DTYPE_UINT8    = 4,   /**< Unsigned 8-bit integer (uint8_t) */
    DTL_DTYPE_UINT16   = 5,   /**< Unsigned 16-bit integer (uint16_t) */
    DTL_DTYPE_UINT32   = 6,   /**< Unsigned 32-bit integer (uint32_t) */
    DTL_DTYPE_UINT64   = 7,   /**< Unsigned 64-bit integer (uint64_t) */
    DTL_DTYPE_FLOAT32  = 8,   /**< 32-bit floating point (float) */
    DTL_DTYPE_FLOAT64  = 9,   /**< 64-bit floating point (double) */
    DTL_DTYPE_BYTE     = 10,  /**< Raw byte (unsigned char) */
    DTL_DTYPE_BOOL     = 11,  /**< Boolean (stored as uint8_t) */
    DTL_DTYPE_COUNT    = 12   /**< Number of supported dtypes */
} dtl_dtype;

/**
 * @brief Get the size in bytes for a data type
 * @param dtype The data type
 * @return Size in bytes, or 0 for invalid dtype
 */
DTL_API dtl_size_t dtl_dtype_size(dtl_dtype dtype);

/**
 * @brief Get the name of a data type
 * @param dtype The data type
 * @return String name (e.g., "int32"), or "unknown" for invalid dtype
 */
DTL_API const char* dtl_dtype_name(dtl_dtype dtype);

/* ==========================================================================
 * Reduction Operation Enumeration
 * ========================================================================== */

/**
 * @brief Enumeration of reduction operations
 *
 * These correspond to MPI reduction operations and are used
 * in collective reduce/allreduce calls.
 */
typedef enum dtl_reduce_op {
    DTL_OP_SUM   = 0,  /**< Sum of elements */
    DTL_OP_PROD  = 1,  /**< Product of elements */
    DTL_OP_MIN   = 2,  /**< Minimum element */
    DTL_OP_MAX   = 3,  /**< Maximum element */
    DTL_OP_LAND  = 4,  /**< Logical AND */
    DTL_OP_LOR   = 5,  /**< Logical OR */
    DTL_OP_BAND  = 6,  /**< Bitwise AND */
    DTL_OP_BOR   = 7,  /**< Bitwise OR */
    DTL_OP_LXOR  = 8,  /**< Logical XOR */
    DTL_OP_BXOR  = 9,  /**< Bitwise XOR */
    DTL_OP_MINLOC = 10, /**< Min with location */
    DTL_OP_MAXLOC = 11, /**< Max with location */
    DTL_OP_COUNT  = 12  /**< Number of operations */
} dtl_reduce_op;

/* ==========================================================================
 * Determinism Policy Enumerations (ARC-0010)
 * ========================================================================== */

/** @brief Determinism mode for algorithm/runtime behavior */
typedef enum dtl_determinism_mode {
    /** @brief Prefer throughput (default) */
    DTL_DETERMINISM_THROUGHPUT = 0,
    /** @brief Enforce deterministic behavior where supported */
    DTL_DETERMINISM_DETERMINISTIC = 1
} dtl_determinism_mode;

/** @brief Reduction scheduling control under deterministic mode */
typedef enum dtl_reduction_schedule_policy {
    /** @brief Implementation-defined reduction scheduling (default) */
    DTL_REDUCTION_SCHEDULE_IMPLEMENTATION_DEFINED = 0,
    /** @brief Fixed reduction tree where supported */
    DTL_REDUCTION_SCHEDULE_FIXED_TREE = 1
} dtl_reduction_schedule_policy;

/** @brief Progress ordering control under deterministic mode */
typedef enum dtl_progress_ordering_policy {
    /** @brief Implementation-defined progress ordering (default) */
    DTL_PROGRESS_ORDERING_IMPLEMENTATION_DEFINED = 0,
    /** @brief Rank-ordered progress where supported */
    DTL_PROGRESS_ORDERING_RANK_ORDERED = 1
} dtl_progress_ordering_policy;

/**
 * @brief Get the name of a reduction operation
 * @param op The reduction operation
 * @return String name (e.g., "sum"), or "unknown" for invalid op
 */
DTL_API const char* dtl_reduce_op_name(dtl_reduce_op op);

/* ==========================================================================
 * Context Options
 * ========================================================================== */

/**
 * @brief Options for context creation
 *
 * All fields are optional. Set to 0/NULL for defaults.
 */
typedef struct dtl_context_options {
    /**
     * @brief Device ID for GPU operations (-1 for CPU only)
     *
     * Default: -1 (CPU only)
     */
    int device_id;

    /**
     * @brief Whether to initialize MPI if not already initialized
     *
     * Default: 1 (true)
     * If set to 0 and MPI is not initialized, context creation will fail.
     */
    int init_mpi;

    /**
     * @brief Whether to finalize MPI on context destruction
     *
     * Default: 0 (false)
     * Only relevant if this context initialized MPI.
     */
    int finalize_mpi;

    /**
     * @brief ABI-stable extension fields
     *
     * reserved[0]: dtl_determinism_mode
     * reserved[1]: dtl_reduction_schedule_policy
     * reserved[2]: dtl_progress_ordering_policy
     * reserved[3]: reserved for future use
     */
    int reserved[4];
} dtl_context_options;

#define DTL_CONTEXT_OPT_DETERMINISM_MODE(opts) ((opts).reserved[0])
#define DTL_CONTEXT_OPT_REDUCTION_SCHEDULE(opts) ((opts).reserved[1])
#define DTL_CONTEXT_OPT_PROGRESS_ORDERING(opts) ((opts).reserved[2])

/**
 * @brief Initialize context options to default values
 * @param opts Pointer to options structure
 */
DTL_API void dtl_context_options_init(dtl_context_options* opts);

/* ==========================================================================
 * Tensor Shape/Extents
 * ========================================================================== */

/** @brief Maximum tensor rank (dimensions) supported */
#define DTL_MAX_TENSOR_RANK 8

/**
 * @brief Tensor shape descriptor
 *
 * Describes the shape of an N-dimensional tensor.
 */
typedef struct dtl_shape {
    /** @brief Number of dimensions (1 to DTL_MAX_TENSOR_RANK) */
    int ndim;

    /** @brief Size along each dimension */
    dtl_size_t dims[DTL_MAX_TENSOR_RANK];
} dtl_shape;

/**
 * @brief Create a 1D shape
 * @param dim0 Size of dimension 0
 * @return Shape descriptor
 */
DTL_API dtl_shape dtl_shape_1d(dtl_size_t dim0);

/**
 * @brief Create a 2D shape
 * @param dim0 Size of dimension 0 (rows)
 * @param dim1 Size of dimension 1 (cols)
 * @return Shape descriptor
 */
DTL_API dtl_shape dtl_shape_2d(dtl_size_t dim0, dtl_size_t dim1);

/**
 * @brief Create a 3D shape
 * @param dim0 Size of dimension 0
 * @param dim1 Size of dimension 1
 * @param dim2 Size of dimension 2
 * @return Shape descriptor
 */
DTL_API dtl_shape dtl_shape_3d(dtl_size_t dim0, dtl_size_t dim1, dtl_size_t dim2);

/**
 * @brief Create an N-dimensional shape
 * @param ndim Number of dimensions
 * @param dims Array of dimension sizes
 * @return Shape descriptor
 */
DTL_API dtl_shape dtl_shape_nd(int ndim, const dtl_size_t* dims);

/**
 * @brief Get total number of elements in shape
 * @param shape Shape descriptor
 * @return Total element count
 */
DTL_API dtl_size_t dtl_shape_size(const dtl_shape* shape);

DTL_C_END

#endif /* DTL_TYPES_H */
