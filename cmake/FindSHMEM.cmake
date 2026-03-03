# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# FindSHMEM.cmake
# =============================================================================
# Find OpenSHMEM installation
#
# This module finds if OpenSHMEM is installed and determines where the include
# files and libraries are. This code sets the following variables:
#
#  SHMEM_FOUND         - True if SHMEM was found
#  SHMEM_INCLUDE_DIRS  - Include directories for SHMEM
#  SHMEM_LIBRARIES     - Libraries to link against
#  SHMEM_VERSION       - Version of SHMEM (if available)
#
# Environment variables that may affect the search:
#  SHMEM_ROOT          - Root directory of SHMEM installation
#  OSHMEM_ROOT         - Root directory for OpenMPI's OSHMEM
#  SOS_ROOT            - Root directory for Sandia OpenSHMEM
#
# Imported targets:
#  SHMEM::shmem        - Interface target for OpenSHMEM
# =============================================================================

# Look for the shmem header
find_path(SHMEM_INCLUDE_DIR
    NAMES shmem.h
    HINTS
        ${SHMEM_ROOT}
        ${OSHMEM_ROOT}
        ${SOS_ROOT}
        ENV SHMEM_ROOT
        ENV OSHMEM_ROOT
        ENV SOS_ROOT
    PATH_SUFFIXES
        include
        include/shmem
)

# Look for the OpenSHMEM library
# Try various naming conventions used by different implementations
find_library(SHMEM_LIBRARY
    NAMES
        oshmem        # OpenMPI's OpenSHMEM
        shmem         # Generic SHMEM
        openshmem     # Some implementations
        sos           # Sandia OpenSHMEM
    HINTS
        ${SHMEM_ROOT}
        ${OSHMEM_ROOT}
        ${SOS_ROOT}
        ENV SHMEM_ROOT
        ENV OSHMEM_ROOT
        ENV SOS_ROOT
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64-linux-gnu
)

# Also look for the oshrun executable (OpenMPI's SHMEM launcher)
find_program(SHMEM_OSHRUN_EXECUTABLE
    NAMES oshrun oshcc
    HINTS
        ${SHMEM_ROOT}
        ${OSHMEM_ROOT}
        ENV SHMEM_ROOT
        ENV OSHMEM_ROOT
    PATH_SUFFIXES
        bin
)

# Try to determine the SHMEM version
if(SHMEM_INCLUDE_DIR)
    # Check for OpenMPI's OSHMEM version
    if(EXISTS "${SHMEM_INCLUDE_DIR}/oshmem/version.h")
        file(STRINGS "${SHMEM_INCLUDE_DIR}/oshmem/version.h" _shmem_version_str
             REGEX "^#define[ \t]+OSHMEM_VERSION_STRING[ \t]+\"[^\"]*\"")
        if(_shmem_version_str)
            string(REGEX REPLACE ".*\"([^\"]*)\".*" "\\1" SHMEM_VERSION "${_shmem_version_str}")
        endif()
    endif()

    # If still no version, try to extract from shmem.h
    if(NOT SHMEM_VERSION AND EXISTS "${SHMEM_INCLUDE_DIR}/shmem.h")
        file(STRINGS "${SHMEM_INCLUDE_DIR}/shmem.h" _shmem_version_major
             REGEX "^#define[ \t]+SHMEM_MAJOR_VERSION[ \t]+[0-9]+")
        file(STRINGS "${SHMEM_INCLUDE_DIR}/shmem.h" _shmem_version_minor
             REGEX "^#define[ \t]+SHMEM_MINOR_VERSION[ \t]+[0-9]+")
        if(_shmem_version_major AND _shmem_version_minor)
            string(REGEX REPLACE ".*([0-9]+).*" "\\1" _major "${_shmem_version_major}")
            string(REGEX REPLACE ".*([0-9]+).*" "\\1" _minor "${_shmem_version_minor}")
            set(SHMEM_VERSION "${_major}.${_minor}")
        endif()
    endif()
endif()

# Use FindPackageHandleStandardArgs to handle success/failure
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SHMEM
    REQUIRED_VARS SHMEM_LIBRARY SHMEM_INCLUDE_DIR
    VERSION_VAR SHMEM_VERSION
)

# Set output variables
if(SHMEM_FOUND)
    set(SHMEM_INCLUDE_DIRS ${SHMEM_INCLUDE_DIR})
    set(SHMEM_LIBRARIES ${SHMEM_LIBRARY})

    # Create imported target if it doesn't exist
    if(NOT TARGET SHMEM::shmem)
        add_library(SHMEM::shmem UNKNOWN IMPORTED)
        set_target_properties(SHMEM::shmem PROPERTIES
            IMPORTED_LOCATION "${SHMEM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SHMEM_INCLUDE_DIR}"
        )
    endif()
endif()

# Mark as advanced
mark_as_advanced(
    SHMEM_INCLUDE_DIR
    SHMEM_LIBRARY
    SHMEM_OSHRUN_EXECUTABLE
)
