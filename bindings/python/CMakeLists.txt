# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Python Bindings
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Fall back to FetchContent
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# -----------------------------------------------------------------------------
# Native Extension Module
# -----------------------------------------------------------------------------

pybind11_add_module(_dtl MODULE
    src/dtl/_dtl.cpp
    src/dtl/core.cpp
    src/dtl/policies.cpp
    src/dtl/containers.cpp
    src/dtl/collective.cpp
    src/dtl/algorithms.cpp
    src/dtl/rma.cpp
    src/dtl/mpmd.cpp
    src/dtl/topology.cpp
    src/dtl/futures.cpp
)

# Link against C bindings library
target_link_libraries(_dtl PRIVATE dtl_c)

# Include directories
target_include_directories(_dtl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(_dtl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dtl
)

# MPI support
if(DTL_MPI_AVAILABLE)
    target_link_libraries(_dtl PRIVATE MPI::MPI_CXX)
    target_compile_definitions(_dtl PRIVATE DTL_HAS_MPI)

    # Find mpi4py headers
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
        OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(MPI4PY_INCLUDE_DIR)
        target_include_directories(_dtl PRIVATE ${MPI4PY_INCLUDE_DIR})
        target_compile_definitions(_dtl PRIVATE DTL_HAS_MPI4PY)
    endif()
endif()

# CUDA support
if(DTL_ENABLE_CUDA)
    target_compile_definitions(_dtl PRIVATE DTL_ENABLE_CUDA)
endif()

# -----------------------------------------------------------------------------
# Python Package Files
# -----------------------------------------------------------------------------

# Copy Python source files to build directory
file(GLOB PYTHON_SOURCES "src/dtl/*.py")
foreach(SOURCE ${PYTHON_SOURCES})
    get_filename_component(FILENAME ${SOURCE} NAME)
    configure_file(${SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/dtl/${FILENAME} COPYONLY)
endforeach()

# Copy type stub files
file(GLOB STUB_FILES "src/dtl/*.pyi")
foreach(STUB ${STUB_FILES})
    get_filename_component(FILENAME ${STUB} NAME)
    configure_file(${STUB} ${CMAKE_CURRENT_BINARY_DIR}/dtl/${FILENAME} COPYONLY)
endforeach()

# Create py.typed marker for PEP 561
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dtl/py.typed "")

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

# Install to Python site-packages when building standalone
if(NOT SKBUILD)
    install(TARGETS _dtl
        LIBRARY DESTINATION ${Python3_SITEARCH}/dtl
    )

    install(FILES ${PYTHON_SOURCES}
        DESTINATION ${Python3_SITEARCH}/dtl
    )

    install(FILES ${STUB_FILES}
        DESTINATION ${Python3_SITEARCH}/dtl
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dtl/py.typed
        DESTINATION ${Python3_SITEARCH}/dtl
    )
endif()
