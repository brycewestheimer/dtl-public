// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_remote_ref.cpp
/// @brief Unit tests for remote_ref
/// @details CRITICAL: Tests that remote_ref has NO implicit conversions

#include <dtl/views/remote_ref.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// NO Implicit Conversions (CRITICAL)
// =============================================================================

TEST(RemoteRefTest, NoImplicitConversionToValueRef) {
    // CRITICAL: remote_ref<int> must NOT convert to int&
    static_assert(!std::is_convertible_v<remote_ref<int>, int&>,
                  "Violation: remote_ref<int> converts to int&");
}

TEST(RemoteRefTest, NoImplicitConversionToConstValueRef) {
    // CRITICAL: remote_ref<int> must NOT convert to const int&
    static_assert(!std::is_convertible_v<remote_ref<int>, const int&>,
                  "Violation: remote_ref<int> converts to const int&");
}

TEST(RemoteRefTest, NoImplicitConversionToValuePtr) {
    // CRITICAL: remote_ref<int> must NOT convert to int*
    static_assert(!std::is_convertible_v<remote_ref<int>, int*>,
                  "Violation: remote_ref<int> converts to int*");
}

TEST(RemoteRefTest, NoImplicitConversionToConstValuePtr) {
    // CRITICAL: remote_ref<int> must NOT convert to const int*
    static_assert(!std::is_convertible_v<remote_ref<int>, const int*>,
                  "Violation: remote_ref<int> converts to const int*");
}

TEST(RemoteRefTest, NoImplicitConversionToBool) {
    // CRITICAL: remote_ref<int> must NOT convert to bool
    static_assert(!std::is_convertible_v<remote_ref<int>, bool>,
                  "Violation: remote_ref<int> converts to bool");
}

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(RemoteRefTest, ConstructWithLocalPointer) {
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
    EXPECT_EQ(ref.remote_capability(), remote_access_capability::local_only);
    EXPECT_EQ(ref.owner_rank(), 0);
    EXPECT_EQ(ref.global_index(), 0);
}

TEST(RemoteRefTest, ConstructWithNullPointer) {
    remote_ref<int> ref(1, 100, nullptr);

    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
    EXPECT_EQ(ref.remote_capability(),
              remote_access_capability::remote_transport_unavailable);
    EXPECT_EQ(ref.owner_rank(), 1);
    EXPECT_EQ(ref.global_index(), 100);
}

// =============================================================================
// Explicit Access Tests (get/put)
// =============================================================================

TEST(RemoteRefTest, GetLocalValue) {
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(RemoteRefTest, PutLocalValue) {
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    auto result = ref.put(100);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(value, 100);
}

TEST(RemoteRefTest, PutLocalValueMove) {
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    auto result = ref.put(std::move(200));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(value, 200);
}

TEST(RemoteRefTest, GetRemoteValueReturnsError) {
    // Remote get should return error (no communicator)
    remote_ref<int> ref(1, 100, nullptr);

    auto result = ref.get();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::not_supported);
}

TEST(RemoteRefTest, PutRemoteValueReturnsError) {
    // Remote put should return error (no communicator)
    remote_ref<int> ref(1, 100, nullptr);

    auto result = ref.put(42);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::not_supported);
}

TEST(RemoteRefTest, RemoteTransportAvailabilityFlagCanGateRemoteAccess) {
    remote_ref<int> ref(1, 3, nullptr, false);

    EXPECT_FALSE(ref.remote_transport_available());
    EXPECT_EQ(ref.remote_capability(),
              remote_access_capability::remote_transport_unavailable);

    auto get_result = ref.get();
    EXPECT_FALSE(get_result.has_value());
    EXPECT_EQ(get_result.error().code(), status_code::not_supported);
}

// =============================================================================
// Const Remote Ref Tests
// =============================================================================

TEST(RemoteRefTest, ConstRemoteRefNoImplicitConversions) {
    // Const version should also have no implicit conversions
    static_assert(!std::is_convertible_v<remote_ref<const int>, const int&>,
                  "Violation: remote_ref<const int> converts to const int&");
    static_assert(!std::is_convertible_v<remote_ref<const int>, const int*>,
                  "Violation: remote_ref<const int> converts to const int*");
    static_assert(!std::is_convertible_v<remote_ref<const int>, bool>,
                  "Violation: remote_ref<const int> converts to bool");
}

TEST(RemoteRefTest, ConstRemoteRefGet) {
    const int value = 42;
    remote_ref<const int> ref(0, 0, &value);

    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

// =============================================================================
// Type Alias Tests
// =============================================================================

TEST(RemoteRefTest, ValueTypeAlias) {
    static_assert(std::is_same_v<remote_ref<int>::value_type, int>);
    static_assert(std::is_same_v<remote_ref<const int>::value_type, int>);
    static_assert(std::is_same_v<remote_ref<double>::value_type, double>);
}

TEST(RemoteRefTest, ElementTypeAlias) {
    static_assert(std::is_same_v<remote_ref<int>::element_type, int>);
    static_assert(std::is_same_v<remote_ref<const int>::element_type, const int>);
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(RemoteRefTest, IsRemoteRefTrait) {
    static_assert(is_remote_ref_v<remote_ref<int>>);
    static_assert(is_remote_ref_v<remote_ref<const int>>);
    static_assert(is_remote_ref_v<remote_ref<double>>);

    static_assert(!is_remote_ref_v<int>);
    static_assert(!is_remote_ref_v<int&>);
    static_assert(!is_remote_ref_v<int*>);
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

TEST(RemoteRefTest, CorrectUsagePattern) {
    // Demonstrate the correct explicit access pattern
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    // Check locality first
    if (ref.is_local()) {
        // Explicit get - this is the intended usage
        auto result = ref.get();
        if (result.has_value()) {
            int x = result.value();
            EXPECT_EQ(x, 42);
        }

        // Explicit put
        ref.put(100);
        EXPECT_EQ(value, 100);
    }
}

}  // namespace dtl::test
