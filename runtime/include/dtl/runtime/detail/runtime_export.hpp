// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file runtime_export.hpp
/// @brief Export/import macro for the DTL runtime shared library
/// @details Defines DTL_RUNTIME_API for symbol visibility in libdtl_runtime.so.
///          When building the DSO, symbols are exported; when consuming, imported.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#if defined(DTL_PLATFORM_WINDOWS)
    #ifdef DTL_RUNTIME_EXPORTS
        #define DTL_RUNTIME_API __declspec(dllexport)
    #else
        #define DTL_RUNTIME_API __declspec(dllimport)
    #endif
#else
    #ifdef DTL_RUNTIME_EXPORTS
        #define DTL_RUNTIME_API __attribute__((visibility("default")))
    #else
        #define DTL_RUNTIME_API
    #endif
#endif
