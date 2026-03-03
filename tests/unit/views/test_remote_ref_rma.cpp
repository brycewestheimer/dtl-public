// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_remote_ref_rma.cpp
/// @brief Unit tests for remote_ref RMA window extensions (Phase 12B)
/// @details Tests the new 5-argument constructor, has_window(), and
///          window-aware get/put behavior added for RMA support.

#include <dtl/views/remote_ref.hpp>
#include <dtl/communication/memory_window.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <memory>

namespace dtl::test {

// =============================================================================
// Backward Compatibility: 3-Argument Constructor
// =============================================================================

TEST(RemoteRefRmaTest, LocalGetWithoutWindow) {
    // Existing 3-arg constructor, local get works as before
    int value = 42;
    remote_ref<int> ref(0, 0, &value);

    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 42);
}

TEST(RemoteRefRmaTest, LocalPutWithoutWindow) {
    // Existing 3-arg constructor, local put works as before
    int value = 0;
    remote_ref<int> ref(0, 0, &value);

    auto res = ref.put(99);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(value, 99);
}

TEST(RemoteRefRmaTest, IsLocalWithPtr) {
    int value = 10;
    remote_ref<int> ref(0, 5, &value);
    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
}

TEST(RemoteRefRmaTest, IsRemoteWithoutPtr) {
    remote_ref<int> ref(2, 500, nullptr);
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
}

TEST(RemoteRefRmaTest, HasWindowDefault) {
    // 3-arg constructor should leave window_ as null
    int value = 1;
    remote_ref<int> ref(0, 0, &value);
    EXPECT_FALSE(ref.has_window());
}

// =============================================================================
// Extended 5-Argument Constructor
// =============================================================================

TEST(RemoteRefRmaTest, HasWindowExtended) {
    // 5-arg constructor with non-null window should report has_window true
    int value = 7;
    auto window_impl = std::make_unique<null_window_impl>(&value, sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    remote_ref<int> ref(0, 10, &value, window_ptr, 128);
    EXPECT_TRUE(ref.has_window());
}

TEST(RemoteRefRmaTest, ExtendedConstructor) {
    // Verify all members are set correctly by the 5-arg constructor
    int value = 55;
    auto window_impl = std::make_unique<null_window_impl>(&value, sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    remote_ref<int> ref(3, 42, &value, window_ptr, 256);

    EXPECT_EQ(ref.owner_rank(), 3);
    EXPECT_EQ(ref.global_index(), 42);
    EXPECT_TRUE(ref.is_local());
    EXPECT_TRUE(ref.has_window());

    // Local get should still work even with window set
    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 55);
}

TEST(RemoteRefRmaTest, BackwardCompat3Arg) {
    // 3-arg constructor still works and window_ is null
    remote_ref<int> ref(1, 100, nullptr);
    EXPECT_EQ(ref.owner_rank(), 1);
    EXPECT_EQ(ref.global_index(), 100);
    EXPECT_TRUE(ref.is_remote());
    EXPECT_FALSE(ref.has_window());
}

// =============================================================================
// Remote Get/Put with and without Window
// =============================================================================

TEST(RemoteRefRmaTest, RemoteGetWithoutWindow) {
    // Remote ref without window: get returns explicit unsupported status
    remote_ref<int> ref(5, 200, nullptr);
    ASSERT_FALSE(ref.has_window());

    auto res = ref.get();
    EXPECT_FALSE(res.has_value());
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
}

TEST(RemoteRefRmaTest, RemoteGetWithWindow) {
    // Remote ref with window: get should now work via null_window_impl
    int data = 42;
    auto window_impl = std::make_unique<null_window_impl>(&data, sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    // Create remote ref pointing to rank 0 (null window assumes rank 0)
    remote_ref<int> ref(0, 200, nullptr, window_ptr, 0);
    ASSERT_TRUE(ref.has_window());

    auto res = ref.get();
    // Should succeed via null_window_impl (single-rank only)
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 42);
}

TEST(RemoteRefRmaTest, RemotePutWithoutWindow) {
    // Remote ref without window: put returns explicit unsupported status
    remote_ref<int> ref(5, 200, nullptr);

    auto res = ref.put(42);
    EXPECT_FALSE(res.has_value());
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
}

TEST(RemoteRefRmaTest, RemotePutWithWindow) {
    // Remote ref with window: put should now work via null_window_impl
    int data = 0;
    auto window_impl = std::make_unique<null_window_impl>(&data, sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    // Create remote ref pointing to rank 0 (null window assumes rank 0)
    remote_ref<int> ref(0, 200, nullptr, window_ptr, 0);

    auto res = ref.put(42);
    // Should succeed via null_window_impl (single-rank only)
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(data, 42);  // Verify data was written
}

// =============================================================================
// Query Operations
// =============================================================================

TEST(RemoteRefRmaTest, OwnerRank) {
    remote_ref<int> ref(7, 300, nullptr);
    EXPECT_EQ(ref.owner_rank(), 7);
}

TEST(RemoteRefRmaTest, GlobalIndex) {
    remote_ref<int> ref(0, 12345, nullptr);
    EXPECT_EQ(ref.global_index(), 12345);
}

// =============================================================================
// Const Remote Ref Tests (const T specialization)
// =============================================================================

TEST(RemoteRefRmaTest, ConstRemoteRef) {
    // const version works with 5-arg constructor
    const int value = 77;
    auto window_impl = std::make_unique<null_window_impl>(
        const_cast<int*>(&value), sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    remote_ref<const int> ref(2, 50, &value, window_ptr, 32);

    EXPECT_EQ(ref.owner_rank(), 2);
    EXPECT_EQ(ref.global_index(), 50);
    EXPECT_TRUE(ref.is_local());
    EXPECT_TRUE(ref.has_window());
}

TEST(RemoteRefRmaTest, ConstRefGet) {
    // const ref get() on local data works
    const int value = 88;
    remote_ref<const int> ref(0, 0, &value);

    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 88);
}

TEST(RemoteRefRmaTest, ConstRefHasWindow) {
    // const ref has_window() with 3-arg and 5-arg constructors
    const int value = 1;
    remote_ref<const int> ref_no_win(0, 0, &value);
    EXPECT_FALSE(ref_no_win.has_window());

    auto window_impl = std::make_unique<null_window_impl>(
        const_cast<int*>(&value), sizeof(int), false);
    memory_window_impl* window_ptr = window_impl.get();

    remote_ref<const int> ref_win(0, 0, &value, window_ptr, 0);
    EXPECT_TRUE(ref_win.has_window());
}

// =============================================================================
// Async Operations
// =============================================================================

TEST(RemoteRefRmaTest, AsyncGetLocal) {
    // async_get on local data returns value (falls back to sync get)
    int value = 33;
    remote_ref<int> ref(0, 0, &value);

    auto res = ref.async_get().get_result();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 33);
}

TEST(RemoteRefRmaTest, AsyncGetRemote) {
    // async_get on remote data returns error (no communicator)
    remote_ref<int> ref(5, 200, nullptr);

    auto res = ref.async_get().get_result();
    EXPECT_FALSE(res.has_value());
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// Deleted Conversions
// =============================================================================

TEST(RemoteRefRmaTest, DeletedConversions) {
    // Verify that implicit conversions remain deleted after extensions
    static_assert(!std::is_convertible_v<remote_ref<int>, int&>,
                  "Violation: remote_ref<int> converts to int&");
    static_assert(!std::is_convertible_v<remote_ref<int>, const int&>,
                  "Violation: remote_ref<int> converts to const int&");
    static_assert(!std::is_convertible_v<remote_ref<int>, int*>,
                  "Violation: remote_ref<int> converts to int*");
    static_assert(!std::is_convertible_v<remote_ref<int>, const int*>,
                  "Violation: remote_ref<int> converts to const int*");
    static_assert(!std::is_convertible_v<remote_ref<int>, bool>,
                  "Violation: remote_ref<int> converts to bool");

    // Also check the const specialization
    static_assert(!std::is_convertible_v<remote_ref<const int>, const int&>,
                  "Violation: remote_ref<const int> converts to const int&");
    static_assert(!std::is_convertible_v<remote_ref<const int>, const int*>,
                  "Violation: remote_ref<const int> converts to const int*");
    static_assert(!std::is_convertible_v<remote_ref<const int>, bool>,
                  "Violation: remote_ref<const int> converts to bool");
}

}  // namespace dtl::test
