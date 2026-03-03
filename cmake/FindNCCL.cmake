# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# FindNCCL.cmake
# =============================================================================
# Finds NVIDIA NCCL and defines:
#   NCCL_FOUND
#   NCCL_INCLUDE_DIRS
#   NCCL_LIBRARIES
#   NCCL_VERSION
# and imported target:
#   NCCL::NCCL
#
# Supports NCCL_ROOT as either a CMake variable or environment variable.
# =============================================================================

include(FindPackageHandleStandardArgs)

set(_NCCL_HINTS)
if(NCCL_ROOT)
	list(APPEND _NCCL_HINTS "${NCCL_ROOT}")
endif()
if(DEFINED ENV{NCCL_ROOT})
	list(APPEND _NCCL_HINTS "$ENV{NCCL_ROOT}")
endif()

find_path(NCCL_INCLUDE_DIR
	NAMES nccl.h
	HINTS ${_NCCL_HINTS}
	PATH_SUFFIXES include include/nccl
)

find_library(NCCL_LIBRARY
	NAMES nccl
	HINTS ${_NCCL_HINTS}
	PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
	file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_major
		 REGEX "^#define NCCL_MAJOR[ \t]+[0-9]+")
	file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_minor
		 REGEX "^#define NCCL_MINOR[ \t]+[0-9]+")
	file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_patch
		 REGEX "^#define NCCL_PATCH[ \t]+[0-9]+")

	string(REGEX REPLACE ".*NCCL_MAJOR[ \t]+([0-9]+).*" "\\1"
		   _nccl_major "${_nccl_version_major}")
	string(REGEX REPLACE ".*NCCL_MINOR[ \t]+([0-9]+).*" "\\1"
		   _nccl_minor "${_nccl_version_minor}")
	string(REGEX REPLACE ".*NCCL_PATCH[ \t]+([0-9]+).*" "\\1"
		   _nccl_patch "${_nccl_version_patch}")

	if(_nccl_major MATCHES "^[0-9]+$" AND _nccl_minor MATCHES "^[0-9]+$" AND _nccl_patch MATCHES "^[0-9]+$")
		set(NCCL_VERSION "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
	endif()
endif()

find_package_handle_standard_args(NCCL
	REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
	VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
	set(NCCL_LIBRARIES "${NCCL_LIBRARY}")
	set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")

	if(NOT TARGET NCCL::NCCL)
		add_library(NCCL::NCCL UNKNOWN IMPORTED)
		set_target_properties(NCCL::NCCL PROPERTIES
			IMPORTED_LOCATION "${NCCL_LIBRARY}"
			INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
		)
	endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
