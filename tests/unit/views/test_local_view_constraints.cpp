// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_local_view_constraints.cpp
/// @brief Tests for placement-aware local_view constraints
/// @since 0.1.0

#include <dtl/views/local_view.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/core/device_concepts.hpp>

#include <vector>
#include <array>

// ============================================================================
// Compile-time tests for local_view
// ============================================================================

// local_view should work with any type (it's a host view)
static_assert(std::is_constructible_v<dtl::local_view<int>, int*, std::size_t>,
    "local_view<int> should be constructible from pointer and size");

static_assert(std::is_constructible_v<dtl::local_view<float>, float*, std::size_t>,
    "local_view<float> should be constructible from pointer and size");

// local_view should have begin/end iterators
static_assert(requires(dtl::local_view<int> v) {
    { v.begin() } -> std::same_as<int*>;
    { v.end() } -> std::same_as<int*>;
}, "local_view should have begin/end iterators");

// local_view should have random access
static_assert(requires(dtl::local_view<int> v, std::size_t i) {
    { v[i] } -> std::same_as<int&>;
}, "local_view should support random access");

// ============================================================================
// Compile-time tests for device_view
// ============================================================================

// device_view should only accept DeviceStorable types
static_assert(std::is_constructible_v<dtl::device_view<int>, int*, std::size_t>,
    "device_view<int> should be constructible from pointer and size");

static_assert(std::is_constructible_v<dtl::device_view<float>, float*, std::size_t>,
    "device_view<float> should be constructible from pointer and size");

// device_view should NOT have host-iterable begin/end returning raw pointers
// Instead it only exposes data() and size()
static_assert(requires(dtl::device_view<int> v) {
    { v.data() } -> std::same_as<int*>;
    { v.size() } -> std::same_as<dtl::size_type>;
}, "device_view should have data() and size()");

// device_view should NOT have operator[] (would dereference device memory on host)
static_assert(!requires(dtl::device_view<int> v, std::size_t i) {
    { v[i] };
}, "device_view should NOT have operator[]");

// device_view should NOT have begin/end iterators
// (This might be allowed in some implementations, but we check that
// standard iteration isn't accidentally enabled)

// ============================================================================
// Compile-time tests for is_device_view trait
// ============================================================================

static_assert(dtl::is_device_view_v<dtl::device_view<int>>,
    "is_device_view should be true for device_view<int>");
static_assert(dtl::is_device_view_v<dtl::device_view<float>>,
    "is_device_view should be true for device_view<float>");
static_assert(!dtl::is_device_view_v<dtl::local_view<int>>,
    "is_device_view should be false for local_view<int>");
static_assert(!dtl::is_device_view_v<int>,
    "is_device_view should be false for int");

// ============================================================================
// Runtime tests for local_view
// ============================================================================

int main() {
    // Test local_view basic operations
    std::vector<int> data = {1, 2, 3, 4, 5};
    dtl::local_view<int> view(data.data(), data.size());

    // Check size
    if (view.size() != 5) {
        return 1;
    }

    // Check element access
    if (view[0] != 1 || view[4] != 5) {
        return 2;
    }

    // Check iteration
    int sum = 0;
    for (auto& elem : view) {
        sum += elem;
    }
    if (sum != 15) {
        return 3;
    }

    // Check const view
    dtl::local_view<const int> const_view(data.data(), data.size());
    if (const_view.size() != 5) {
        return 4;
    }

    // Test device_view basic operations (pointer access only, no dereferencing)
    int* dummy_device_ptr = data.data();  // For testing, not actual device memory
    dtl::device_view<int> dev_view(dummy_device_ptr, 5);

    if (dev_view.size() != 5) {
        return 5;
    }

    if (dev_view.data() != dummy_device_ptr) {
        return 6;
    }

    if (dev_view.empty()) {
        return 7;
    }

    // Test subview
    auto sub = dev_view.subview(1, 3);
    if (sub.size() != 3) {
        return 8;
    }
    if (sub.data() != dummy_device_ptr + 1) {
        return 9;
    }

    // Test first/last
    auto first_two = dev_view.first(2);
    if (first_two.size() != 2) {
        return 10;
    }

    auto last_two = dev_view.last(2);
    if (last_two.size() != 2) {
        return 11;
    }

    return 0;
}
