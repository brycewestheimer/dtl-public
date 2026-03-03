// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_environment.h
 * @brief C ABI for dtl::environment lifecycle management
 * @since 0.1.0, updated 1.4.0 (instance-based queries, from_comm)
 *
 * This header defines the DTL environment, which manages backend
 * initialization and finalization (MPI, CUDA, HIP, NCCL, SHMEM)
 * using reference-counted RAII semantics. The first create call
 * initializes backends; the last destroy finalizes them.
 *
 * V1.4.0 changes:
 * - Backend query functions now also accept an environment handle
 * - Zero-arg query functions are kept for backward compatibility
 * - Added dtl_environment_domain() for named domain access
 */

#ifndef DTL_ENVIRONMENT_H
#define DTL_ENVIRONMENT_H

#include "dtl_types.h"
#include "dtl_status.h"

DTL_C_BEGIN

/* dtl_environment_t is defined in dtl_types.h (included above) */

/* ==========================================================================
 * Environment Lifecycle
 * ========================================================================== */

/** @defgroup environment Environment Lifecycle
 *  @{
 */

/**
 * @brief Create a new DTL environment (no argc/argv).
 *
 * Initializes all configured backends on first call. Subsequent calls
 * increment the internal reference count.
 *
 * @param[out] env  Pointer to receive the environment handle.
 * @return DTL_SUCCESS on success, DTL_ERROR_NULL_POINTER if env is NULL,
 *         DTL_ERROR_ALLOCATION_FAILED on memory failure.
 *
 * @pre env must not be NULL
 * @post On success, *env contains a valid environment handle
 *
 * @note The caller must call dtl_environment_destroy() when done.
 * @note Some MPI implementations may not work correctly without argc/argv.
 *       For multi-rank MPI programs, prefer dtl_environment_create_with_args().
 *
 * @code
 * dtl_environment_t env;
 * dtl_status status = dtl_environment_create(&env);
 * if (status != DTL_SUCCESS) {
 *     fprintf(stderr, "Failed: %s\n", dtl_status_message(status));
 *     return 1;
 * }
 * // Use environment...
 * dtl_environment_destroy(env);
 * @endcode
 */
DTL_API dtl_status dtl_environment_create(dtl_environment_t* env);

/**
 * @brief Create a new DTL environment with command-line arguments.
 *
 * Passes argc/argv to MPI_Init_thread on first call. Subsequent calls
 * increment the internal reference count.
 *
 * @param[out] env   Pointer to receive the environment handle.
 * @param[in]  argc  Pointer to argument count (may be modified by MPI_Init).
 * @param[in]  argv  Pointer to argument vector (may be modified by MPI_Init).
 * @return DTL_SUCCESS on success, DTL_ERROR_NULL_POINTER if env is NULL,
 *         DTL_ERROR_ALLOCATION_FAILED on memory failure.
 *
 * @pre env must not be NULL
 * @post On success, *env contains a valid environment handle
 *
 * @note The caller must call dtl_environment_destroy() when done.
 *
 * @code
 * int main(int argc, char** argv) {
 *     dtl_environment_t env;
 *     dtl_status status = dtl_environment_create_with_args(&env, &argc, &argv);
 *     if (status != DTL_SUCCESS) {
 *         return 1;
 *     }
 *     // Use environment...
 *     dtl_environment_destroy(env);
 *     return 0;
 * }
 * @endcode
 */
DTL_API dtl_status dtl_environment_create_with_args(dtl_environment_t* env,
                                                      int* argc, char*** argv);

/**
 * @brief Destroy a DTL environment and potentially finalize backends.
 *
 * Decrements the reference count. When the last handle is destroyed,
 * backends are finalized in reverse order (SHMEM -> NCCL -> HIP -> CUDA -> MPI).
 *
 * @param env  Environment handle to destroy (may be NULL, which is a no-op).
 *
 * @post env is invalid and must not be used.
 *
 * @note It is safe to call with NULL.
 * @note All contexts created from this environment should be destroyed first.
 */
DTL_API void dtl_environment_destroy(dtl_environment_t env);

/** @} */ /* end of environment group */

/* ==========================================================================
 * Environment State Queries
 * ========================================================================== */

/** @defgroup environment_state Environment State Queries
 *  @{
 */

/**
 * @brief Check if DTL environment is currently initialized.
 *
 * Returns non-zero if at least one environment handle exists (reference count > 0).
 *
 * @return Non-zero if initialized, 0 otherwise.
 */
DTL_API int dtl_environment_is_initialized(void);

/**
 * @brief Get the current environment reference count.
 *
 * @return Reference count (0 if not initialized).
 */
DTL_API dtl_size_t dtl_environment_ref_count(void);

/**
 * @brief Get the named domain label for an environment.
 *
 * @param env  Environment handle.
 * @return Domain name string, or "unknown" if env is invalid.
 * @since 0.1.0
 */
DTL_API const char* dtl_environment_domain(dtl_environment_t env);

/** @} */ /* end of environment_state group */

/* ==========================================================================
 * Backend Availability
 * ========================================================================== */

/** @defgroup environment_backends Backend Availability
 *  @{
 */

/**
 * @brief Check if MPI backend is available and was initialized.
 * @return Non-zero if MPI is available, 0 otherwise.
 *
 * @note Returns runtime state (whether MPI was actually initialized),
 *       not compile-time availability. Use dtl_has_mpi() for compile-time.
 */
DTL_API int dtl_environment_has_mpi(void);

/**
 * @brief Check if CUDA backend is available and was initialized.
 * @return Non-zero if CUDA is available, 0 otherwise.
 */
DTL_API int dtl_environment_has_cuda(void);

/**
 * @brief Check if HIP backend is available and was initialized.
 * @return Non-zero if HIP is available, 0 otherwise.
 */
DTL_API int dtl_environment_has_hip(void);

/**
 * @brief Check if NCCL backend is available.
 * @return Non-zero if NCCL is available, 0 otherwise.
 */
DTL_API int dtl_environment_has_nccl(void);

/**
 * @brief Check if SHMEM backend is available and was initialized.
 * @return Non-zero if SHMEM is available, 0 otherwise.
 */
DTL_API int dtl_environment_has_shmem(void);

/**
 * @brief Get the MPI thread level provided.
 * @return MPI thread level (MPI_THREAD_SINGLE=0, FUNNELED=1, SERIALIZED=2,
 *         MULTIPLE=3), or -1 if MPI is not available.
 */
DTL_API int dtl_environment_mpi_thread_level(void);

/** @} */ /* end of environment_backends group */

/* ==========================================================================
 * Context Factory Methods
 * ========================================================================== */

/** @defgroup environment_context Context Factories
 *  @{
 */

/**
 * @brief Create a world context from the environment.
 *
 * Creates a context that spans all MPI ranks with CPU domain.
 * This is the primary way to get a context for distributed operations.
 *
 * @param[in]  env  Environment handle.
 * @param[out] ctx  Pointer to receive the context handle.
 * @return DTL_SUCCESS on success, DTL_ERROR_NULL_POINTER if ctx is NULL,
 *         DTL_ERROR_INVALID_ARGUMENT if env is invalid.
 *
 * @pre env must be a valid environment handle
 * @pre ctx must not be NULL
 * @pre MPI should be available for multi-rank operation
 */
DTL_API dtl_status dtl_environment_make_world_context(dtl_environment_t env,
                                                        dtl_context_t* ctx);

/**
 * @brief Create a GPU-enabled world context from the environment.
 *
 * Creates a context that spans all MPI ranks with CPU and CUDA domains.
 *
 * @param[in]  env        Environment handle.
 * @param[in]  device_id  GPU device ID.
 * @param[out] ctx        Pointer to receive the context handle.
 * @return DTL_SUCCESS on success, DTL_ERROR_NULL_POINTER if ctx is NULL,
 *         DTL_ERROR_INVALID_ARGUMENT if env is invalid, or
 *         DTL_ERROR_BACKEND_UNAVAILABLE if CUDA is not available.
 *
 * @pre env must be a valid environment handle
 * @pre ctx must not be NULL
 * @pre CUDA should be available for GPU operations
 */
DTL_API dtl_status dtl_environment_make_world_context_gpu(dtl_environment_t env,
                                                            int device_id,
                                                            dtl_context_t* ctx);

/**
 * @brief Create a CPU-only context from the environment.
 *
 * Creates a single-process CPU-only context (no MPI).
 * Useful for local testing or non-distributed code.
 *
 * @param[in]  env  Environment handle.
 * @param[out] ctx  Pointer to receive the context handle.
 * @return DTL_SUCCESS on success, DTL_ERROR_NULL_POINTER if ctx is NULL,
 *         DTL_ERROR_INVALID_ARGUMENT if env is invalid.
 *
 * @pre env must be a valid environment handle
 * @pre ctx must not be NULL
 */
DTL_API dtl_status dtl_environment_make_cpu_context(dtl_environment_t env,
                                                      dtl_context_t* ctx);

/** @} */ /* end of environment_context group */

DTL_C_END

#endif /* DTL_ENVIRONMENT_H */
