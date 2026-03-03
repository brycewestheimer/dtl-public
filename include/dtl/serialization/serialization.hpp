// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file serialization.hpp
/// @brief Master include for serialization module
/// @details Includes all serialization components: serializer trait,
///          trivial serialization, and integration utilities.
/// @since 0.1.0

#pragma once

// Core serializer trait
#include <dtl/serialization/serializer.hpp>

// Trivial type optimizations
#include <dtl/serialization/trivial_serializer.hpp>

// Serialization traits and detection
#include <dtl/serialization/serialization_traits.hpp>

// Member function extension points
#include <dtl/serialization/member_serialization.hpp>

// Built-in STL type serializers
#include <dtl/serialization/stl_serializers.hpp>

// Optional library integration
#include <dtl/serialization/library_integration.hpp>
