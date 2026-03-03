// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file error_code.hpp
/// @brief Legacy error code enumeration (see status.hpp for preferred API)
/// @details Provides backward compatibility with code using error_code.
///          New code should use status_code from status.hpp.
/// @since 0.1.0
/// @deprecated Use status_code from status.hpp instead

#pragma once

#include <dtl/error/status.hpp>

namespace dtl {

/// @brief Alias for status_code for backward compatibility
/// @deprecated Use status_code directly
using error_code = status_code;

}  // namespace dtl
