# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Fortran Bindings Examples
# =============================================================================
# These examples demonstrate the native dtl module from bindings/fortran/dtl.f90
# They use the dtl_fortran library target, not the examples/fortran module.

# Example 1: Vector Demo - Basic context and vector operations
add_executable(dtl_fortran_vector_demo
    fortran_vector_demo.f90
)
target_link_libraries(dtl_fortran_vector_demo PRIVATE dtl_fortran)

# Example 2: Hello World - Minimal DTL usage
add_executable(dtl_fortran_hello
    fortran_hello_dtl.f90
)
target_link_libraries(dtl_fortran_hello PRIVATE dtl_fortran)

# Set output directory
set_target_properties(dtl_fortran_vector_demo dtl_fortran_hello PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/fortran
)
