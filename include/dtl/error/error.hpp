// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file error.hpp
/// @brief Master include for error handling module
/// @details Includes all error handling components: status codes, result types,
///          and error handling utilities.
/// @since 0.1.0

#pragma once

// Core error types
#include <dtl/error/status.hpp>
#include <dtl/error/result.hpp>

// Additional error handling utilities
#include <dtl/error/error_code.hpp>
#include <dtl/error/error_type.hpp>
#include <dtl/error/collective_error.hpp>
#include <dtl/error/failure_handler.hpp>
