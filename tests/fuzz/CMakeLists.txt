# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Fuzz testing targets for DTL
# Requires Clang with -fsanitize=fuzzer support

if(NOT DTL_BUILD_C_BINDINGS)
    message(STATUS "Fuzz targets require DTL_BUILD_C_BINDINGS=ON, skipping")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Fuzz targets require Clang compiler, skipping")
    return()
endif()

# C API fuzz target
add_executable(fuzz_c_api fuzz_c_api.cpp)
target_compile_features(fuzz_c_api PRIVATE cxx_std_20)
target_link_libraries(fuzz_c_api PRIVATE DTL::dtl dtl_c)
target_compile_options(fuzz_c_api PRIVATE -fsanitize=fuzzer,address)
target_link_options(fuzz_c_api PRIVATE -fsanitize=fuzzer,address)

# Create corpus directory for seed inputs
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus)
