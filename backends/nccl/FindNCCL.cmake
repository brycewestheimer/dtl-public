# FindNCCL.cmake
# ---------------
# Find NVIDIA NCCL library
#
# This module defines:
#   NCCL_FOUND        - True if NCCL was found
#   NCCL_INCLUDE_DIRS - NCCL include directories
#   NCCL_LIBRARIES    - NCCL libraries to link
#   NCCL_VERSION      - NCCL version string (e.g. "2.18.5")
#
# And the imported target:
#   NCCL::NCCL        - Imported interface target
#
# Hints:
#   NCCL_ROOT         - Root directory to search
#   NCCL_INCLUDE_DIR  - Explicit include directory
#   NCCL_LIBRARY      - Explicit library path

include(FindPackageHandleStandardArgs)

# Search for nccl.h
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}
        ENV NCCL_ROOT
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
    PATH_SUFFIXES include
)

# Search for libnccl
find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        ${NCCL_ROOT}
        ENV NCCL_ROOT
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/cuda/lib64
    PATH_SUFFIXES lib lib64
)

# Extract version from nccl.h
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_major
         REGEX "^#define NCCL_MAJOR[ \t]+[0-9]+")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_minor
         REGEX "^#define NCCL_MINOR[ \t]+[0-9]+")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_patch
         REGEX "^#define NCCL_PATCH[ \t]+[0-9]+")

    if(_nccl_major AND _nccl_minor AND _nccl_patch)
        string(REGEX REPLACE "^#define NCCL_MAJOR[ \t]+([0-9]+)" "\\1"
               NCCL_VERSION_MAJOR "${_nccl_major}")
        string(REGEX REPLACE "^#define NCCL_MINOR[ \t]+([0-9]+)" "\\1"
               NCCL_VERSION_MINOR "${_nccl_minor}")
        string(REGEX REPLACE "^#define NCCL_PATCH[ \t]+([0-9]+)" "\\1"
               NCCL_VERSION_PATCH "${_nccl_patch}")
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()

    unset(_nccl_major)
    unset(_nccl_minor)
    unset(_nccl_patch)
endif()

find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})

    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
