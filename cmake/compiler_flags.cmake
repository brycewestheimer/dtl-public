# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Compiler Flags Configuration
# =============================================================================
# Provides compiler-specific flags for DTL builds.
# =============================================================================

include_guard(GLOBAL)

# -----------------------------------------------------------------------------
# Compiler Detection
# -----------------------------------------------------------------------------
set(DTL_COMPILER_IS_GNU     FALSE)
set(DTL_COMPILER_IS_CLANG   FALSE)
set(DTL_COMPILER_IS_MSVC    FALSE)
set(DTL_COMPILER_IS_INTEL   FALSE)
set(DTL_COMPILER_IS_NVCC    FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(DTL_COMPILER_IS_GNU TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DTL_COMPILER_IS_CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(DTL_COMPILER_IS_MSVC TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(DTL_COMPILER_IS_INTEL TRUE)
endif()

# -----------------------------------------------------------------------------
# C++20 Concepts Support Check
# -----------------------------------------------------------------------------
function(dtl_check_concepts_support)
    include(CheckCXXSourceCompiles)

    set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    check_cxx_source_compiles("
        #include <concepts>
        template<typename T>
        concept Numeric = std::integral<T> || std::floating_point<T>;

        template<Numeric T>
        T add(T a, T b) { return a + b; }

        int main() { return add(1, 2); }
    " DTL_HAS_CONCEPTS)

    if(NOT DTL_HAS_CONCEPTS)
        message(FATAL_ERROR "C++20 concepts support is required but not available")
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Warning Flags
# -----------------------------------------------------------------------------
function(dtl_set_warning_flags target)
    if(DTL_COMPILER_IS_GNU OR DTL_COMPILER_IS_CLANG)
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wcast-qual
            -Wformat=2
            -Wundef
            -Werror=return-type
            -Wno-unused-parameter
        )

        if(DTL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()

    elseif(DTL_COMPILER_IS_MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /Zc:__cplusplus
        )

        if(DTL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Optimization Flags
# -----------------------------------------------------------------------------
function(dtl_set_optimization_flags target)
    if(DTL_COMPILER_IS_GNU OR DTL_COMPILER_IS_CLANG)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-O0 -g>
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
            $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
        )
    elseif(DTL_COMPILER_IS_MSVC)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /DNDEBUG>
            $<$<CONFIG:MinSizeRel>:/O1 /DNDEBUG>
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# NVCC Flags (for CUDA builds)
# -----------------------------------------------------------------------------
function(dtl_set_cuda_flags target)
    if(DTL_ENABLE_CUDA)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )

        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-extended-lambda
                --expt-relaxed-constexpr
                -Xcompiler=-Wall
            >
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Combined Flag Setup
# -----------------------------------------------------------------------------
function(dtl_configure_target target)
    dtl_set_warning_flags(${target})
    dtl_set_optimization_flags(${target})

    if(DTL_ENABLE_CUDA)
        dtl_set_cuda_flags(${target})
    endif()
endfunction()
