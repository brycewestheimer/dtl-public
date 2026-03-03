# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# FindCereal.cmake - CMake module for finding the Cereal serialization library
# =============================================================================
# This module attempts to find the Cereal C++ serialization library.
#
# Result variables:
#   Cereal_FOUND        - True if Cereal was found
#   Cereal_INCLUDE_DIRS - Include directories for Cereal
#   Cereal_VERSION      - Version string (if detectable)
#
# Imported targets:
#   Cereal::Cereal      - Interface target for linking
#
# Hints:
#   CEREAL_ROOT         - Root directory to search for Cereal
#   CEREAL_INCLUDE_DIR  - Include directory containing cereal/cereal.hpp
# =============================================================================

include(FindPackageHandleStandardArgs)

# Try to find Cereal using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_Cereal QUIET cereal)
endif()

# Look for the main Cereal header
find_path(Cereal_INCLUDE_DIR
    NAMES cereal/cereal.hpp
    PATHS
        ${CEREAL_ROOT}
        ${PC_Cereal_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
        /opt/local/include
        $ENV{CEREAL_ROOT}
        $ENV{HOME}/.local/include
    PATH_SUFFIXES include
)

# Try to detect version from cereal.hpp or version.hpp
if(Cereal_INCLUDE_DIR)
    # Cereal doesn't have a dedicated version header, but we can check for presence
    if(EXISTS "${Cereal_INCLUDE_DIR}/cereal/version.hpp")
        file(STRINGS "${Cereal_INCLUDE_DIR}/cereal/version.hpp" _cereal_version_line
             REGEX "^#define[ \t]+CEREAL_VERSION[ \t]+")
        if(_cereal_version_line)
            string(REGEX REPLACE "^#define[ \t]+CEREAL_VERSION[ \t]+\"([^\"]+)\".*" "\\1"
                   Cereal_VERSION "${_cereal_version_line}")
        endif()
    endif()
    
    # If version not found, just mark as "unknown"
    if(NOT Cereal_VERSION)
        set(Cereal_VERSION "unknown")
    endif()
endif()

# Standard CMake package handling
find_package_handle_standard_args(Cereal
    REQUIRED_VARS Cereal_INCLUDE_DIR
    VERSION_VAR Cereal_VERSION
)

# Create imported target
if(Cereal_FOUND AND NOT TARGET Cereal::Cereal)
    add_library(Cereal::Cereal INTERFACE IMPORTED)
    set_target_properties(Cereal::Cereal PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Cereal_INCLUDE_DIR}"
    )
endif()

# Set output variables
if(Cereal_FOUND)
    set(Cereal_INCLUDE_DIRS ${Cereal_INCLUDE_DIR})
endif()

mark_as_advanced(Cereal_INCLUDE_DIR)
