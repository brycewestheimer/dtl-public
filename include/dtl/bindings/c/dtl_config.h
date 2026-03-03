// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_config.h
 * @brief DTL C bindings - Version and configuration
 * @since 0.1.0
 *
 * This header provides version information, platform detection,
 * and feature query macros for the DTL C API.
 */

#ifndef DTL_CONFIG_H
#define DTL_CONFIG_H

#include <dtl/generated/version_config.h>
#include <stdint.h>

/* ==========================================================================
 * Version Information
 * ========================================================================== */

/* ==========================================================================
 * Platform Detection
 * ========================================================================== */

/* Operating System */
#if defined(_WIN32) || defined(_WIN64)
    #define DTL_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define DTL_PLATFORM_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
    #define DTL_PLATFORM_MACOS 1
#elif defined(__FreeBSD__)
    #define DTL_PLATFORM_FREEBSD 1
#else
    #define DTL_PLATFORM_UNKNOWN 1
#endif

/* Compiler Detection */
#if defined(__clang__)
    #define DTL_COMPILER_CLANG 1
#elif defined(__GNUC__) || defined(__GNUG__)
    #define DTL_COMPILER_GCC 1
#elif defined(_MSC_VER)
    #define DTL_COMPILER_MSVC 1
#else
    #define DTL_COMPILER_UNKNOWN 1
#endif

/* Architecture */
#if defined(__x86_64__) || defined(_M_X64)
    #define DTL_ARCH_X64 1
#elif defined(__i386__) || defined(_M_IX86)
    #define DTL_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DTL_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
    #define DTL_ARCH_ARM 1
#else
    #define DTL_ARCH_UNKNOWN 1
#endif

/* ==========================================================================
 * Export/Import Macros
 * ========================================================================== */

#if defined(DTL_PLATFORM_WINDOWS)
    #ifdef DTL_C_EXPORTS
        #define DTL_API __declspec(dllexport)
    #else
        #define DTL_API __declspec(dllimport)
    #endif
    #define DTL_HIDDEN
#else
    #if defined(DTL_COMPILER_GCC) || defined(DTL_COMPILER_CLANG)
        #define DTL_API __attribute__((visibility("default")))
        #define DTL_HIDDEN __attribute__((visibility("hidden")))
    #else
        #define DTL_API
        #define DTL_HIDDEN
    #endif
#endif

/* ==========================================================================
 * C++ Compatibility
 * ========================================================================== */

#ifdef __cplusplus
    #define DTL_C_BEGIN extern "C" {
    #define DTL_C_END   }
#else
    #define DTL_C_BEGIN
    #define DTL_C_END
#endif

/* ==========================================================================
 * Scalar Type Definitions
 * ========================================================================== */

DTL_C_BEGIN

/** @brief MPI rank type (signed for MPI compatibility) */
typedef int32_t dtl_rank_t;

/** @brief Size type (unsigned, for counts and sizes) */
typedef uint64_t dtl_size_t;

/** @brief Index type (signed, for offsets and indexing) */
typedef int64_t dtl_index_t;

/** @brief Status/error code type */
typedef int32_t dtl_status;

/** @brief Tag type for message passing */
typedef int32_t dtl_tag_t;

/** @brief Special rank value indicating no rank / invalid */
#define DTL_NO_RANK ((dtl_rank_t)-1)

/** @brief Special rank value for any source */
#define DTL_ANY_SOURCE ((dtl_rank_t)-2)

/** @brief Special tag for any tag */
#define DTL_ANY_TAG ((dtl_tag_t)-1)

/* ==========================================================================
 * Version Query Functions
 * ========================================================================== */

/**
 * @brief Get DTL major version number
 * @return Major version number
 */
DTL_API int dtl_version_major(void);

/**
 * @brief Get DTL minor version number
 * @return Minor version number
 */
DTL_API int dtl_version_minor(void);

/**
 * @brief Get DTL patch version number
 * @return Patch version number
 */
DTL_API int dtl_version_patch(void);

/**
 * @brief Get DTL ABI version number
 * @return ABI version number
 */
DTL_API int dtl_abi_version(void);

/**
 * @brief Get DTL version as string
 * @return Version string (e.g., "1.0.0")
 */
DTL_API const char* dtl_version_string(void);

/* ==========================================================================
 * Feature Query Functions
 * ========================================================================== */

/**
 * @brief Check if MPI backend is available
 * @return 1 if MPI is available, 0 otherwise
 */
DTL_API int dtl_has_mpi(void);

/**
 * @brief Check if CUDA backend is available
 * @return 1 if CUDA is available, 0 otherwise
 */
DTL_API int dtl_has_cuda(void);

/**
 * @brief Check if HIP backend is available
 * @return 1 if HIP is available, 0 otherwise
 */
DTL_API int dtl_has_hip(void);

/**
 * @brief Check if NCCL is available
 * @return 1 if NCCL is available, 0 otherwise
 */
DTL_API int dtl_has_nccl(void);

/**
 * @brief Check if OpenSHMEM backend is available
 * @return 1 if SHMEM is available, 0 otherwise
 */
DTL_API int dtl_has_shmem(void);

DTL_C_END

#endif /* DTL_CONFIG_H */
