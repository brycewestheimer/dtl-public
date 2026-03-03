// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file containers.hpp
/// @brief Master include for all DTL containers
/// @details Provides single-header access to all container types.
/// @since 0.1.0

#pragma once

// Sequence containers
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/distributed_tensor.hpp>

// Non-owning views
#include <dtl/containers/distributed_span.hpp>

// Associative containers
#include <dtl/containers/distributed_map.hpp>

// Container traits are defined in:
// - dtl/core/traits.hpp: Base templates for is_distributed_*
// - Individual container headers: Specializations
//
// Container concepts are defined in:
// - dtl/core/concepts.hpp: DistributedContainer concept
