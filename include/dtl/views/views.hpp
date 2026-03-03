// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file views.hpp
/// @brief Master include for all DTL views
/// @details Provides single-header access to all view types.
/// @since 0.1.0

#pragma once

// Core view types
#include <dtl/views/local_view.hpp>
#include <dtl/views/global_view.hpp>
#include <dtl/views/segmented_view.hpp>

// Remote element access
#include <dtl/views/remote_ref.hpp>

// Derived views
#include <dtl/views/subview.hpp>
#include <dtl/views/strided_view.hpp>

// Composition utilities
#include <dtl/views/composed_view.hpp>

// Batching views (for throughput-oriented patterns)
#include <dtl/views/chunk_view.hpp>
#include <dtl/views/tile_view.hpp>
#include <dtl/views/chunk_by_view.hpp>
#include <dtl/views/window_view.hpp>

// Core traits and concepts are in:
// - dtl/core/traits.hpp: is_local_view, is_global_view, is_segmented_view, is_remote_ref
// - dtl/core/concepts.hpp: LocalView, GlobalView, SegmentedView

namespace dtl {

// ============================================================================
// Communication Traits (unique to views)
// ============================================================================

/// @brief Check if a view may communicate
template <typename T>
struct may_communicate : std::false_type {};

template <typename Container>
struct may_communicate<global_view<Container>> : std::true_type {};

template <typename T>
struct may_communicate<remote_ref<T>> : std::true_type {};

template <typename T>
inline constexpr bool may_communicate_v = may_communicate<T>::value;

/// @brief Check if a view is safe for STL algorithms (no communication)
template <typename T>
struct is_stl_safe : std::true_type {};

template <typename Container>
struct is_stl_safe<global_view<Container>> : std::false_type {};

template <typename T>
inline constexpr bool is_stl_safe_v = is_stl_safe<T>::value;

// ============================================================================
// Batching View Type Traits
// ============================================================================

/// @brief Check if a type is a batching view (chunk, tile, window, chunk_by)
template <typename T>
struct is_batching_view : std::false_type {};

template <typename Range>
struct is_batching_view<chunk_view<Range>> : std::true_type {};

template <typename MDRange, std::size_t N>
struct is_batching_view<tile_view<MDRange, N>> : std::true_type {};

template <typename Range, typename Predicate>
struct is_batching_view<chunk_by_view<Range, Predicate>> : std::true_type {};

template <typename Range>
struct is_batching_view<window_view<Range>> : std::true_type {};

template <typename T>
inline constexpr bool is_batching_view_v = is_batching_view<T>::value;

// ============================================================================
// View Concepts (unique to views)
// ============================================================================

/// @brief Concept for distributed views
template <typename T>
concept DistributedView = requires(T t) {
    { t.begin() };
    { t.end() };
    { t.size() } -> std::convertible_to<size_type>;
};

}  // namespace dtl
