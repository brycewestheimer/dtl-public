# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Compiler warning configuration
# =============================================================================
# Provides dtl_set_compiler_warnings() function to configure strict compiler
# warnings for DTL targets.
# =============================================================================

#[[
  dtl_set_compiler_warnings(target)

  Sets compiler warnings for the given target based on the compiler being used.
  Supports GCC, Clang, and MSVC.

  Parameters:
    target - The CMake target to apply warnings to

  Options controlled by CMake cache variables:
    DTL_WARNINGS_AS_ERRORS - Treat warnings as errors (default OFF)
]]
function(dtl_set_compiler_warnings target)
    # Common warning flags for all compilers
    set(CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )

    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wuseless-cast
    )

    set(MSVC_WARNINGS
        /W4
        /permissive-
        /w14242  # 'identifier': conversion from 'type1' to 'type2', possible loss of data
        /w14254  # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits'
        /w14263  # 'function': member function does not override any base class virtual member function
        /w14265  # 'class': class has virtual functions, but destructor is not virtual
        /w14287  # 'operator': unsigned/negative constant mismatch
        /we4289  # 'variable': loop control variable declared in the for-loop is used outside the for-loop scope
        /w14296  # 'operator': expression is always 'boolean_value'
        /w14311  # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545  # expression before comma evaluates to a function which is missing an argument list
        /w14546  # function call before comma missing argument list
        /w14547  # 'operator': operator before comma has no effect
        /w14549  # 'operator': operator before comma has no effect
        /w14555  # expression has no effect
        /w14619  # pragma warning: there is no warning number 'number'
        /w14640  # thread un-safe static member initialization
        /w14826  # Conversion from 'type1' to 'type2' is sign-extended
        /w14905  # wide string literal cast to 'LPSTR'
        /w14906  # string literal cast to 'LPWSTR'
        /w14928  # illegal copy-initialization
    )

    # Add warnings-as-errors if enabled
    if(DTL_WARNINGS_AS_ERRORS)
        list(APPEND CLANG_WARNINGS -Werror)
        list(APPEND GCC_WARNINGS -Werror)
        list(APPEND MSVC_WARNINGS /WX)
    endif()

    # Detect compiler and apply appropriate warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PROJECT_WARNINGS ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_WARNINGS ${GCC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(PROJECT_WARNINGS ${MSVC_WARNINGS})
    else()
        message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}, no warnings configured")
        set(PROJECT_WARNINGS "")
    endif()

    # Apply warnings to target
    # For INTERFACE libraries (header-only), use INTERFACE scope
    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
        target_compile_options(${target} INTERFACE ${PROJECT_WARNINGS})
    else()
        target_compile_options(${target} PRIVATE ${PROJECT_WARNINGS})
    endif()
endfunction()

#[[
  dtl_suppress_warnings(target)

  Suppresses all compiler warnings for the given target.
  Useful for third-party code or generated code.
]]
function(dtl_suppress_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -w)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /w)
    endif()
endfunction()
