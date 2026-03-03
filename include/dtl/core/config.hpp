// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file config.hpp
/// @brief Compile-time configuration and feature detection macros
/// @details Defines feature macros based on CMake configuration and provides
///          compile-time feature detection utilities.
/// @since 0.1.0

#pragma once

// =============================================================================
// Platform Detection
// =============================================================================

// Compiler detection
#if defined(__clang__)
    #define DTL_COMPILER_CLANG 1
    #define DTL_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__) || defined(__GNUG__)
    #define DTL_COMPILER_GCC 1
    #define DTL_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
    #define DTL_COMPILER_MSVC 1
    #define DTL_COMPILER_VERSION _MSC_VER
#elif defined(__NVCC__)
    #define DTL_COMPILER_NVCC 1
    #define DTL_COMPILER_VERSION (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100)
#else
    #define DTL_COMPILER_UNKNOWN 1
    #define DTL_COMPILER_VERSION 0
#endif

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define DTL_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define DTL_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define DTL_PLATFORM_MACOS 1
#else
    #define DTL_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// C++20 Feature Detection
// =============================================================================

// Concepts support
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    #define DTL_HAS_CONCEPTS 1
#else
    #error "DTL requires C++20 concepts support"
#endif

// Three-way comparison
#if defined(__cpp_impl_three_way_comparison) && __cpp_impl_three_way_comparison >= 201907L
    #define DTL_HAS_SPACESHIP 1
#endif

// Constexpr features
#if defined(__cpp_constexpr) && __cpp_constexpr >= 201907L
    #define DTL_HAS_CONSTEXPR_DYNAMIC_ALLOC 1
#endif

// Ranges
#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
    #define DTL_HAS_RANGES 1
#endif

// std::expected (C++23, but checking for backports)
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
    #define DTL_HAS_STD_EXPECTED 1
#endif

// =============================================================================
// Backend Feature Macros (configured by CMake)
// =============================================================================

// These are defined by CMake based on build configuration:
// #cmakedefine DTL_ENABLE_CUDA
// #cmakedefine DTL_ENABLE_HIP
// #cmakedefine DTL_ENABLE_NCCL
// #cmakedefine DTL_ENABLE_SHMEM

// For header-only usage without CMake configuration, provide defaults
#ifndef DTL_ENABLE_CUDA
    #define DTL_ENABLE_CUDA 0
#endif

#ifndef DTL_ENABLE_HIP
    #define DTL_ENABLE_HIP 0
#endif

#ifndef DTL_ENABLE_NCCL
    #define DTL_ENABLE_NCCL 0
#endif

#ifndef DTL_ENABLE_MPI
    #define DTL_ENABLE_MPI 0
#endif

#ifndef DTL_ENABLE_SHMEM
    #define DTL_ENABLE_SHMEM 0
#endif

#ifndef DTL_ENABLE_UCX
    #define DTL_ENABLE_UCX 0
#endif

#ifndef DTL_ENABLE_GASNET
    #define DTL_ENABLE_GASNET 0
#endif

#ifndef DTL_ENABLE_SYCL
    #define DTL_ENABLE_SYCL 0
#endif

// =============================================================================
// Attribute Macros
// =============================================================================

// [[nodiscard]] with message (C++20)
#define DTL_NODISCARD [[nodiscard]]
#define DTL_NODISCARD_MSG(msg) [[nodiscard(msg)]]

// [[likely]] and [[unlikely]] (C++20)
#define DTL_LIKELY [[likely]]
#define DTL_UNLIKELY [[unlikely]]

// [[no_unique_address]] (C++20)
#define DTL_NO_UNIQUE_ADDRESS [[no_unique_address]]

// Deprecation
#define DTL_DEPRECATED [[deprecated]]
#define DTL_DEPRECATED_MSG(msg) [[deprecated(msg)]]

// =============================================================================
// Device/Host Execution Space Macros
// =============================================================================

/// Mark function as callable from host and device
/// Expands to __host__ __device__ on CUDA/HIP, empty otherwise
#if defined(__CUDACC__) || defined(__HIPCC__)
    #define DTL_HOST_DEVICE __host__ __device__
    #define DTL_DEVICE __device__
    #define DTL_HOST __host__
#else
    #define DTL_HOST_DEVICE
    #define DTL_DEVICE
    #define DTL_HOST
#endif

// =============================================================================
// Debug and Assert Configuration
// =============================================================================

#ifdef NDEBUG
    #define DTL_DEBUG 0
#else
    #define DTL_DEBUG 1
#endif

// DTL assertion macro
#if DTL_DEBUG
    #include <cassert>
    #define DTL_ASSERT(cond) assert(cond)
    #define DTL_ASSERT_MSG(cond, msg) assert((cond) && (msg))
#else
    #define DTL_ASSERT(cond) ((void)0)
    #define DTL_ASSERT_MSG(cond, msg) ((void)0)
#endif

// Unreachable code marker
#if defined(DTL_COMPILER_GCC) || defined(DTL_COMPILER_CLANG)
    #define DTL_UNREACHABLE() __builtin_unreachable()
#elif defined(DTL_COMPILER_MSVC)
    #define DTL_UNREACHABLE() __assume(0)
#else
    #define DTL_UNREACHABLE() ((void)0)
#endif

// =============================================================================
// ABI and Visibility
// =============================================================================

#if defined(DTL_PLATFORM_WINDOWS)
    #define DTL_EXPORT __declspec(dllexport)
    #define DTL_IMPORT __declspec(dllimport)
    #define DTL_HIDDEN
#else
    #define DTL_EXPORT __attribute__((visibility("default")))
    #define DTL_IMPORT __attribute__((visibility("default")))
    #define DTL_HIDDEN __attribute__((visibility("hidden")))
#endif

// For header-only library, typically no import/export needed
#ifndef DTL_API
#define DTL_API
#endif

// =============================================================================
// Inline and Constexpr Hints
// =============================================================================

#define DTL_INLINE inline
#define DTL_FORCE_INLINE inline

#if defined(DTL_COMPILER_GCC) || defined(DTL_COMPILER_CLANG)
    #undef DTL_FORCE_INLINE
    #define DTL_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(DTL_COMPILER_MSVC)
    #undef DTL_FORCE_INLINE
    #define DTL_FORCE_INLINE __forceinline
#endif

namespace dtl {

/// @brief Compile-time configuration information
struct config {
    /// @brief Check if MPI backend is enabled
    static constexpr bool mpi_enabled = DTL_ENABLE_MPI;

    /// @brief Check if CUDA backend is enabled
    static constexpr bool cuda_enabled = DTL_ENABLE_CUDA;

    /// @brief Check if HIP backend is enabled
    static constexpr bool hip_enabled = DTL_ENABLE_HIP;

    /// @brief Check if NCCL communicator is enabled
    static constexpr bool nccl_enabled = DTL_ENABLE_NCCL;

    /// @brief Check if OpenSHMEM backend is enabled
    static constexpr bool shmem_enabled = DTL_ENABLE_SHMEM;

    /// @brief Check if UCX backend is enabled
    static constexpr bool ucx_enabled = DTL_ENABLE_UCX;

    /// @brief Check if GASNet-EX backend is enabled
    static constexpr bool gasnet_enabled = DTL_ENABLE_GASNET;

    /// @brief Check if SYCL backend is enabled
    static constexpr bool sycl_enabled = DTL_ENABLE_SYCL;

    /// @brief Check if debug mode is enabled
    static constexpr bool debug_enabled = DTL_DEBUG;

    /// @brief Check if concepts are supported
    static constexpr bool concepts_supported = DTL_HAS_CONCEPTS;
};

}  // namespace dtl
