# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Standalone build for the N-body mini-app
# =============================================================================
# Build against an installed DTL:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/dtl/install
#   make
#
# Alternative: build with pkg-config
#   mpicxx -std=c++20 $(pkg-config --cflags dtl) nbody.cpp -o nbody $(pkg-config --libs dtl)
# =============================================================================

cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES CXX)

find_package(DTL REQUIRED)
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)
target_compile_features(nbody PRIVATE cxx_std_20)
target_link_libraries(nbody PRIVATE
    DTL::dtl
    MPI::MPI_CXX
)
