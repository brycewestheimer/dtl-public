# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# NCCL backend build configuration

if(DTL_ENABLE_NCCL)
    find_package(NCCL QUIET)

    add_library(dtl_nccl_backend INTERFACE)
    target_include_directories(dtl_nccl_backend INTERFACE
        ${CMAKE_SOURCE_DIR}/backends/nccl
    )

    if(NCCL_FOUND)
        target_link_libraries(dtl_nccl_backend INTERFACE ${NCCL_LIBRARIES})
        target_include_directories(dtl_nccl_backend INTERFACE ${NCCL_INCLUDE_DIRS})
        target_compile_definitions(dtl_nccl_backend INTERFACE DTL_ENABLE_NCCL)
    else()
        # Fallback: find NCCL manually (FindNCCL.cmake may be empty)
        find_library(NCCL_LIBRARY NAMES nccl HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
        find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS /usr/include /usr/local/include)
        if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
            message(STATUS "NCCL found manually: ${NCCL_LIBRARY}")
            target_link_libraries(dtl_nccl_backend INTERFACE ${NCCL_LIBRARY})
            target_include_directories(dtl_nccl_backend INTERFACE ${NCCL_INCLUDE_DIR})
            target_compile_definitions(dtl_nccl_backend INTERFACE DTL_ENABLE_NCCL)
        else()
            message(WARNING "NCCL not found - NCCL backend will not work")
        endif()
    endif()

    add_library(DTL::nccl ALIAS dtl_nccl_backend)
endif()
