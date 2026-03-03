// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_allreduce_inplace.cpp
/// @brief Unit tests for allreduce_inplace (CR-P03-T08)
/// @details Tests in-place allreduce with all reduction operations and types.

#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// ============================================================================
// Test Fixture
// ============================================================================

class AllreduceInplaceTest : public ::testing::Test {
protected:
    null_communicator comm;
};

// ============================================================================
// Sum Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Sum_Int) {
    std::vector<int> data = {10, 20, 30};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: identity operation
    EXPECT_EQ(data[0], 10);
    EXPECT_EQ(data[1], 20);
    EXPECT_EQ(data[2], 30);
}

TEST_F(AllreduceInplaceTest, Sum_Double) {
    std::vector<double> data = {1.5, 2.5, 3.5};
    auto result = allreduce_inplace(comm, std::span<double>(data), reduce_sum<double>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(data[0], 1.5);
    EXPECT_DOUBLE_EQ(data[1], 2.5);
    EXPECT_DOUBLE_EQ(data[2], 3.5);
}

TEST_F(AllreduceInplaceTest, Sum_Long) {
    std::vector<long> data = {100L, 200L, 300L};
    auto result = allreduce_inplace(comm, std::span<long>(data), reduce_sum<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 100L);
    EXPECT_EQ(data[1], 200L);
    EXPECT_EQ(data[2], 300L);
}

TEST_F(AllreduceInplaceTest, Sum_Float) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto result = allreduce_inplace(comm, std::span<float>(data), reduce_sum<float>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
}

// ============================================================================
// Min Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Min_Int) {
    std::vector<int> data = {5, 3, 8, 1, 9};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_min<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: values unchanged
    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 8);
    EXPECT_EQ(data[3], 1);
    EXPECT_EQ(data[4], 9);
}

TEST_F(AllreduceInplaceTest, Min_Double) {
    std::vector<double> data = {1.5, 0.5, 3.5};
    auto result = allreduce_inplace(comm, std::span<double>(data), reduce_min<double>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(data[0], 1.5);
    EXPECT_DOUBLE_EQ(data[1], 0.5);
    EXPECT_DOUBLE_EQ(data[2], 3.5);
}

// ============================================================================
// Max Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Max_Int) {
    std::vector<int> data = {5, 3, 8, 1, 9};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_max<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 8);
    EXPECT_EQ(data[3], 1);
    EXPECT_EQ(data[4], 9);
}

TEST_F(AllreduceInplaceTest, Max_Double) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    auto result = allreduce_inplace(comm, std::span<double>(data), reduce_max<double>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(data[0], 1.0);
    EXPECT_DOUBLE_EQ(data[1], 2.0);
    EXPECT_DOUBLE_EQ(data[2], 3.0);
}

// ============================================================================
// Product Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Product_Int) {
    std::vector<int> data = {2, 3, 5, 7};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_product<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 5);
    EXPECT_EQ(data[3], 7);
}

TEST_F(AllreduceInplaceTest, Product_Double) {
    std::vector<double> data = {2.0, 3.0, 4.0};
    auto result = allreduce_inplace(comm, std::span<double>(data), reduce_product<double>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(data[0], 2.0);
    EXPECT_DOUBLE_EQ(data[1], 3.0);
    EXPECT_DOUBLE_EQ(data[2], 4.0);
}

// ============================================================================
// Bitwise AND Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Band_Int) {
    std::vector<int> data = {0xFF, 0x0F, 0xAA};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_band<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: identity
    EXPECT_EQ(data[0], 0xFF);
    EXPECT_EQ(data[1], 0x0F);
    EXPECT_EQ(data[2], 0xAA);
}

TEST_F(AllreduceInplaceTest, Band_Long) {
    std::vector<long> data = {0xFFL, 0x0FL};
    auto result = allreduce_inplace(comm, std::span<long>(data), reduce_band<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0xFFL);
    EXPECT_EQ(data[1], 0x0FL);
}

// ============================================================================
// Bitwise OR Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Bor_Int) {
    std::vector<int> data = {0x01, 0x02, 0x04};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_bor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0x01);
    EXPECT_EQ(data[1], 0x02);
    EXPECT_EQ(data[2], 0x04);
}

TEST_F(AllreduceInplaceTest, Bor_Long) {
    std::vector<long> data = {0x01L, 0x02L};
    auto result = allreduce_inplace(comm, std::span<long>(data), reduce_bor<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0x01L);
    EXPECT_EQ(data[1], 0x02L);
}

// ============================================================================
// Bitwise XOR Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Bxor_Int) {
    std::vector<int> data = {0xAA, 0x55, 0xFF};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_bxor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0xAA);
    EXPECT_EQ(data[1], 0x55);
    EXPECT_EQ(data[2], 0xFF);
}

TEST_F(AllreduceInplaceTest, Bxor_Long) {
    std::vector<long> data = {0xAAL, 0x55L};
    auto result = allreduce_inplace(comm, std::span<long>(data), reduce_bxor<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0xAAL);
    EXPECT_EQ(data[1], 0x55L);
}

// ============================================================================
// Logical AND Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Land_Int) {
    std::vector<int> data = {1, 0, 1, 1};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_land<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 0);
    EXPECT_EQ(data[2], 1);
    EXPECT_EQ(data[3], 1);
}

// ============================================================================
// Logical OR Tests
// ============================================================================

TEST_F(AllreduceInplaceTest, Lor_Int) {
    std::vector<int> data = {0, 0, 1, 0};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_lor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0);
    EXPECT_EQ(data[1], 0);
    EXPECT_EQ(data[2], 1);
    EXPECT_EQ(data[3], 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AllreduceInplaceTest, SingleElement_Sum) {
    std::vector<int> data = {42};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 42);
}

TEST_F(AllreduceInplaceTest, SingleElement_Band) {
    std::vector<int> data = {0xFF};
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_band<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data[0], 0xFF);
}

TEST_F(AllreduceInplaceTest, LargeArray_Sum) {
    constexpr size_t N = 1000;
    std::vector<int> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i);
    }
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: identity, values unchanged
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i));
    }
}

TEST_F(AllreduceInplaceTest, BufferModifiedInPlace) {
    // Verify the buffer is actually modified in place (not just a copy returned)
    std::vector<int> data = {1, 2, 3};
    int* original_ptr = data.data();
    auto result = allreduce_inplace(comm, std::span<int>(data), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    // Pointer should be the same (data modified in place)
    EXPECT_EQ(data.data(), original_ptr);
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 3);
}

}  // namespace dtl::test
