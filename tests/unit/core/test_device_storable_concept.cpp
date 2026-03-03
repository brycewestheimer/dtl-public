// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_storable_concept.cpp
/// @brief Tests for DeviceStorable concept and type constraints
/// @since 0.1.0

#include <dtl/core/device_concepts.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/policies/placement/unified_memory.hpp>

#include <vector>
#include <string>
#include <array>
#include <cstdint>

// ============================================================================
// Compile-time tests for DeviceStorable concept
// ============================================================================

// Positive cases: types that should be DeviceStorable
static_assert(dtl::DeviceStorable<int>, "int should be DeviceStorable");
static_assert(dtl::DeviceStorable<float>, "float should be DeviceStorable");
static_assert(dtl::DeviceStorable<double>, "double should be DeviceStorable");
static_assert(dtl::DeviceStorable<char>, "char should be DeviceStorable");
static_assert(dtl::DeviceStorable<unsigned int>, "unsigned int should be DeviceStorable");
static_assert(dtl::DeviceStorable<std::int64_t>, "int64_t should be DeviceStorable");
static_assert(dtl::DeviceStorable<std::uint8_t>, "uint8_t should be DeviceStorable");

// POD structs should be DeviceStorable
struct SimplePod {
    int x;
    float y;
    double z;
};
static_assert(dtl::DeviceStorable<SimplePod>, "SimplePod should be DeviceStorable");

// std::array of trivial types should be DeviceStorable
static_assert(dtl::DeviceStorable<std::array<int, 10>>,
    "std::array<int, 10> should be DeviceStorable");

// Negative cases: types that should NOT be DeviceStorable
static_assert(!dtl::DeviceStorable<std::string>,
    "std::string should NOT be DeviceStorable");
static_assert(!dtl::DeviceStorable<std::vector<int>>,
    "std::vector<int> should NOT be DeviceStorable");

// Class with virtual functions should NOT be DeviceStorable
class VirtualClass {
public:
    virtual ~VirtualClass() = default;
    virtual void foo() {}
};
static_assert(!dtl::DeviceStorable<VirtualClass>,
    "VirtualClass should NOT be DeviceStorable");

// Class with non-trivial constructor should NOT be DeviceStorable
class NonTrivialConstruct {
public:
    NonTrivialConstruct() : value(42) {}
    int value;
};
static_assert(!dtl::DeviceStorable<NonTrivialConstruct>,
    "NonTrivialConstruct should NOT be DeviceStorable");

// ============================================================================
// Compile-time tests for is_device_storable trait
// ============================================================================

static_assert(dtl::is_device_storable_v<int>, "is_device_storable_v<int> should be true");
static_assert(!dtl::is_device_storable_v<std::string>,
    "is_device_storable_v<std::string> should be false");

// ============================================================================
// Compile-time tests for requires_device_storable
// ============================================================================

static_assert(!dtl::requires_device_storable_v<dtl::host_only>,
    "host_only should NOT require DeviceStorable");
static_assert(dtl::requires_device_storable_v<dtl::device_only<0>>,
    "device_only<0> should require DeviceStorable");
static_assert(dtl::requires_device_storable_v<dtl::device_only<1>>,
    "device_only<1> should require DeviceStorable");
static_assert(dtl::requires_device_storable_v<dtl::device_only_runtime>,
    "device_only_runtime should require DeviceStorable");
static_assert(dtl::requires_device_storable_v<dtl::unified_memory>,
    "unified_memory should require DeviceStorable");

// ============================================================================
// Compile-time tests for ValidElementForPlacement
// ============================================================================

// int with any placement should be valid
static_assert(dtl::ValidElementForPlacement<int, dtl::host_only>,
    "int with host_only should be valid");
static_assert(dtl::ValidElementForPlacement<int, dtl::device_only<0>>,
    "int with device_only<0> should be valid");
static_assert(dtl::ValidElementForPlacement<int, dtl::device_only_runtime>,
    "int with device_only_runtime should be valid");
static_assert(dtl::ValidElementForPlacement<int, dtl::unified_memory>,
    "int with unified_memory should be valid");

// std::string with host_only should be valid
static_assert(dtl::ValidElementForPlacement<std::string, dtl::host_only>,
    "std::string with host_only should be valid");

// std::string with device placements should NOT be valid
static_assert(!dtl::ValidElementForPlacement<std::string, dtl::device_only<0>>,
    "std::string with device_only<0> should NOT be valid");
static_assert(!dtl::ValidElementForPlacement<std::string, dtl::device_only_runtime>,
    "std::string with device_only_runtime should NOT be valid");
static_assert(!dtl::ValidElementForPlacement<std::string, dtl::unified_memory>,
    "std::string with unified_memory should NOT be valid");

// ============================================================================
// Main function (tests are compile-time, so this is minimal)
// ============================================================================

int main() {
    // All tests are compile-time static_asserts
    // If we reach here, all compile-time checks passed
    return 0;
}
