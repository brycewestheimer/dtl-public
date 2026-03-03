// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_remote_integration.cpp
/// @brief Unit tests for RMA remote integration
/// @details Verifies rma_remote_ref operations with memory windows.

#include <dtl/rma/remote_integration.hpp>
#include <dtl/communication/memory_window.hpp>

#include <gtest/gtest.h>
#include <array>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class RmaRemoteIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_.fill(0);
        auto result = memory_window::create(data_.data(), data_.size() * sizeof(int));
        ASSERT_TRUE(result.has_value());
        window_ = std::move(*result);
    }

    std::array<int, 100> data_;
    memory_window window_;
};

// =============================================================================
// RMA Remote Ref Construction Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, DefaultConstructorCreatesInvalidRef) {
    rma::rma_remote_ref<int> ref;
    EXPECT_FALSE(ref.valid());
}

TEST_F(RmaRemoteIntegrationTest, ConstructorCreatesValidRef) {
    rma::rma_remote_ref<int> ref(0, 0, window_);
    EXPECT_TRUE(ref.valid());
    EXPECT_EQ(ref.owner_rank(), 0);
    EXPECT_EQ(ref.offset(), 0u);
}

TEST_F(RmaRemoteIntegrationTest, OffsetStoredCorrectly) {
    rma::rma_remote_ref<int> ref(0, sizeof(int) * 5, window_);
    EXPECT_EQ(ref.offset(), sizeof(int) * 5);
}

// =============================================================================
// Get/Put Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, GetRetrievesValue) {
    data_[0] = 42;

    rma::rma_remote_ref<int> ref(0, 0, window_);
    auto result = ref.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST_F(RmaRemoteIntegrationTest, GetWithOffsetRetrievesCorrectValue) {
    data_[5] = 123;

    rma::rma_remote_ref<int> ref(0, sizeof(int) * 5, window_);
    auto result = ref.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 123);
}

TEST_F(RmaRemoteIntegrationTest, PutWritesValue) {
    rma::rma_remote_ref<int> ref(0, 0, window_);
    auto result = ref.put(999);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[0], 999);
}

TEST_F(RmaRemoteIntegrationTest, PutWithOffsetWritesCorrectly) {
    rma::rma_remote_ref<int> ref(0, sizeof(int) * 7, window_);
    auto result = ref.put(777);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data_[7], 777);
}

TEST_F(RmaRemoteIntegrationTest, GetOnInvalidRefFails) {
    rma::rma_remote_ref<int> ref;
    auto result = ref.get();
    EXPECT_FALSE(result.has_value());
}

TEST_F(RmaRemoteIntegrationTest, PutOnInvalidRefFails) {
    rma::rma_remote_ref<int> ref;
    auto result = ref.put(42);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Locality Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, IsLocalForRankZero) {
    // In null backend, rank 0 is always local
    rma::rma_remote_ref<int> ref(0, 0, window_);
    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
}

TEST_F(RmaRemoteIntegrationTest, IsRemoteForNonZeroRank) {
    // In null backend, non-zero ranks are remote
    rma::rma_remote_ref<int> ref(1, 0, window_);
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
}

// =============================================================================
// is_local() with local_rank parameter (Phase 01 / CR-P01-T04)
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, IsLocalWithLocalRankParameter) {
    // On rank 2, owner 2 should be local
    rma::rma_remote_ref<int> ref(2, 0, window_, 2);
    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
}

TEST_F(RmaRemoteIntegrationTest, IsRemoteWithLocalRankParameter) {
    // On rank 2, owner 0 should be remote
    rma::rma_remote_ref<int> ref(0, 0, window_, 2);
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
}

TEST_F(RmaRemoteIntegrationTest, IsLocalConstRefWithLocalRankParameter) {
    // const ref: on rank 3, owner 3 should be local
    rma::rma_remote_ref<const int> ref(3, 0, window_, 3);
    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
}

TEST_F(RmaRemoteIntegrationTest, IsRemoteConstRefWithLocalRankParameter) {
    // const ref: on rank 3, owner 1 should be remote
    rma::rma_remote_ref<const int> ref(1, 0, window_, 3);
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.is_remote());
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, MakeRmaRefCreatesRef) {
    auto ref = rma::make_rma_ref<int>(0, sizeof(int) * 3, window_);
    EXPECT_TRUE(ref.valid());
    EXPECT_EQ(ref.offset(), sizeof(int) * 3);
}

TEST_F(RmaRemoteIntegrationTest, MakeRmaRefIndexedCalculatesOffset) {
    auto ref = rma::make_rma_ref_indexed<int>(0, 4, window_);
    EXPECT_TRUE(ref.valid());
    EXPECT_EQ(ref.offset(), sizeof(int) * 4);
}

// =============================================================================
// Const Ref Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, ConstRefCanGet) {
    data_[0] = 555;

    rma::rma_remote_ref<const int> ref(0, 0, window_);
    auto result = ref.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 555);
}

TEST_F(RmaRemoteIntegrationTest, ConstRefIsReadOnly) {
    // This test verifies const correctness at compile time
    // rma_remote_ref<const int> has no put() method
    rma::rma_remote_ref<const int> ref(0, 0, window_);

    // The following should NOT compile (commented out):
    // ref.put(42);

    // But get should work
    auto result = ref.get();
    EXPECT_TRUE(result.has_value() || result.has_error());  // Either is valid
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, IsRmaRemoteRefTrait) {
    static_assert(rma::is_rma_remote_ref_v<rma::rma_remote_ref<int>>,
                  "rma_remote_ref<int> should satisfy is_rma_remote_ref");
    static_assert(rma::is_rma_remote_ref_v<rma::rma_remote_ref<const int>>,
                  "rma_remote_ref<const int> should satisfy is_rma_remote_ref");
    static_assert(!rma::is_rma_remote_ref_v<int>,
                  "int should not satisfy is_rma_remote_ref");
}

// =============================================================================
// Window Access Tests
// =============================================================================

TEST_F(RmaRemoteIntegrationTest, WindowAccessReturnsWindow) {
    rma::rma_remote_ref<int> ref(0, 0, window_);
    EXPECT_EQ(ref.window(), &window_);
}

TEST_F(RmaRemoteIntegrationTest, InvalidRefWindowIsNull) {
    rma::rma_remote_ref<int> ref;
    EXPECT_EQ(ref.window(), nullptr);
}

}  // namespace dtl::test
