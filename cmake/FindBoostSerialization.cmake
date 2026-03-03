# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# FindBoostSerialization.cmake - CMake module for finding Boost.Serialization
# =============================================================================
# This module attempts to find the Boost.Serialization library.
# It wraps the standard FindBoost module with specific component detection.
#
# Result variables:
#   BoostSerialization_FOUND         - True if Boost.Serialization was found
#   BoostSerialization_INCLUDE_DIRS  - Include directories
#   BoostSerialization_LIBRARIES     - Libraries to link
#   BoostSerialization_VERSION       - Boost version string
#
# Imported targets:
#   BoostSerialization::BoostSerialization - Target for linking
#
# Hints:
#   BOOST_ROOT                      - Root directory for Boost
#   Boost_NO_SYSTEM_PATHS           - Disable system path search
# =============================================================================

include(FindPackageHandleStandardArgs)

# Use CMake's built-in Boost finder with serialization component
find_package(Boost QUIET COMPONENTS serialization)

if(Boost_SERIALIZATION_FOUND OR Boost_serialization_FOUND)
    set(BoostSerialization_FOUND TRUE)
    set(BoostSerialization_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    set(BoostSerialization_LIBRARIES ${Boost_SERIALIZATION_LIBRARY})
    set(BoostSerialization_VERSION ${Boost_VERSION_STRING})
    
    # Create imported target
    if(NOT TARGET BoostSerialization::BoostSerialization)
        add_library(BoostSerialization::BoostSerialization INTERFACE IMPORTED)
        set_target_properties(BoostSerialization::BoostSerialization PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}"
        )
        if(TARGET Boost::serialization)
            set_target_properties(BoostSerialization::BoostSerialization PROPERTIES
                INTERFACE_LINK_LIBRARIES "Boost::serialization"
            )
        else()
            set_target_properties(BoostSerialization::BoostSerialization PROPERTIES
                INTERFACE_LINK_LIBRARIES "${Boost_SERIALIZATION_LIBRARY}"
            )
        endif()
    endif()
else()
    set(BoostSerialization_FOUND FALSE)
endif()

# Standard package handling for output messages
find_package_handle_standard_args(BoostSerialization
    REQUIRED_VARS BoostSerialization_FOUND
    VERSION_VAR BoostSerialization_VERSION
    FAIL_MESSAGE "Could not find Boost.Serialization. Set BOOST_ROOT or install libboost-serialization-dev."
)

mark_as_advanced(
    BoostSerialization_INCLUDE_DIRS
    BoostSerialization_LIBRARIES
)
