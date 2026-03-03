// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_memory_space_compatibility.cpp
/// @brief Verify spaces_compatible, get_transfer_direction, and transfer_supported
/// @details Tests the memory space compatibility logic, transfer direction
///          inference, and transfer support predicates.

#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/concepts/memory_transfer.hpp>
#include <dtl/memory/host_memory_space.hpp>
#include <gtest/gtest.h>

#include <cstdlib>

namespace dtl::test {

// =============================================================================
// Mock Memory Spaces for Testing
// =============================================================================

/// @brief Mock device memory space
struct mock_device_space {
    using pointer = void*;
    using size_type = dtl::size_type;

    void* allocate(size_type sz) { return std::malloc(sz); }
    void* allocate(size_type sz, size_type) { return std::malloc(sz); }
    void deallocate(void* p, size_type) { std::free(p); }
    memory_space_properties properties() const {
        return {.host_accessible = false, .device_accessible = true};
    }
    const char* name() const { return "mock_device"; }
};

/// @brief Mock unified memory space
struct mock_unified_space {
    using pointer = void*;
    using size_type = dtl::size_type;

    void* allocate(size_type sz) { return std::malloc(sz); }
    void* allocate(size_type sz, size_type) { return std::malloc(sz); }
    void deallocate(void* p, size_type) { std::free(p); }
    memory_space_properties properties() const {
        return {.host_accessible = true, .device_accessible = true, .unified = true};
    }
    const char* name() const { return "mock_unified"; }
};

/// @brief Unclassified mock space (no traits specialization)
struct mock_plain_space {
    using pointer = void*;
    using size_type = dtl::size_type;

    void* allocate(size_type sz) { return std::malloc(sz); }
    void* allocate(size_type sz, size_type) { return std::malloc(sz); }
    void deallocate(void* p, size_type) { std::free(p); }
    memory_space_properties properties() const { return {}; }
    const char* name() const { return "mock_plain"; }
};

}  // namespace dtl::test

namespace dtl {

// Traits specializations for mock spaces

template <>
struct memory_space_traits<test::mock_device_space> {
    static constexpr bool is_host_space = false;
    static constexpr bool is_device_space = true;
    static constexpr bool is_unified_space = false;
    static constexpr bool is_thread_safe = true;
};

template <>
struct memory_space_traits<test::mock_unified_space> {
    static constexpr bool is_host_space = false;
    static constexpr bool is_device_space = false;
    static constexpr bool is_unified_space = true;
    static constexpr bool is_thread_safe = true;
};

// mock_plain_space intentionally has NO traits specialization (uses defaults)

}  // namespace dtl

namespace dtl::test {

// =============================================================================
// spaces_compatible Tests
// =============================================================================

TEST(MemorySpaceCompatibility, HostHostCompatible) {
    EXPECT_TRUE((spaces_compatible<host_memory_space, host_memory_space>()));
}

TEST(MemorySpaceCompatibility, DeviceDeviceCompatible) {
    EXPECT_TRUE((spaces_compatible<mock_device_space, mock_device_space>()));
}

TEST(MemorySpaceCompatibility, HostDeviceIncompatible) {
    EXPECT_FALSE((spaces_compatible<host_memory_space, mock_device_space>()));
    EXPECT_FALSE((spaces_compatible<mock_device_space, host_memory_space>()));
}

TEST(MemorySpaceCompatibility, UnifiedCompatibleWithHost) {
    EXPECT_TRUE((spaces_compatible<mock_unified_space, host_memory_space>()));
    EXPECT_TRUE((spaces_compatible<host_memory_space, mock_unified_space>()));
}

TEST(MemorySpaceCompatibility, UnifiedCompatibleWithDevice) {
    EXPECT_TRUE((spaces_compatible<mock_unified_space, mock_device_space>()));
    EXPECT_TRUE((spaces_compatible<mock_device_space, mock_unified_space>()));
}

TEST(MemorySpaceCompatibility, UnifiedCompatibleWithUnified) {
    EXPECT_TRUE((spaces_compatible<mock_unified_space, mock_unified_space>()));
}

TEST(MemorySpaceCompatibility, UnclassifiedSpacesDefaultCompatible) {
    // Unclassified (no traits specialization) spaces default to compatible
    // to avoid false negatives for user-defined or mock spaces
    EXPECT_TRUE((spaces_compatible<mock_plain_space, mock_plain_space>()));
}

// =============================================================================
// get_transfer_direction Tests
// =============================================================================

TEST(TransferDirection, HostToHost) {
    EXPECT_EQ((get_transfer_direction<host_memory_space, host_memory_space>()),
              transfer_direction::host_to_host);
}

TEST(TransferDirection, HostToDevice) {
    EXPECT_EQ((get_transfer_direction<host_memory_space, mock_device_space>()),
              transfer_direction::host_to_device);
}

TEST(TransferDirection, DeviceToHost) {
    EXPECT_EQ((get_transfer_direction<mock_device_space, host_memory_space>()),
              transfer_direction::device_to_host);
}

TEST(TransferDirection, DeviceToDevice) {
    EXPECT_EQ((get_transfer_direction<mock_device_space, mock_device_space>()),
              transfer_direction::device_to_device);
}

TEST(TransferDirection, UnifiedIsUnknown) {
    // Unified spaces don't classify as host or device in traits
    EXPECT_EQ((get_transfer_direction<mock_unified_space, host_memory_space>()),
              transfer_direction::unknown);
}

TEST(TransferDirection, UnclassifiedIsUnknown) {
    EXPECT_EQ((get_transfer_direction<mock_plain_space, mock_plain_space>()),
              transfer_direction::unknown);
}

// =============================================================================
// transfer_supported Tests
// =============================================================================

TEST(TransferSupported, AllCombinationsSupported) {
    EXPECT_TRUE((transfer_supported<host_memory_space, host_memory_space>()));
    EXPECT_TRUE((transfer_supported<host_memory_space, mock_device_space>()));
    EXPECT_TRUE((transfer_supported<mock_device_space, host_memory_space>()));
    EXPECT_TRUE((transfer_supported<mock_device_space, mock_device_space>()));
    EXPECT_TRUE((transfer_supported<mock_unified_space, host_memory_space>()));
    EXPECT_TRUE((transfer_supported<mock_unified_space, mock_device_space>()));
}

}  // namespace dtl::test
