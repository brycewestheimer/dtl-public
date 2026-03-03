// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_trivial_serialization.cpp
/// @brief Unit tests for trivial type serialization
/// @details Tests for Phase 11.5: serialization of trivially copyable types

#include <dtl/serialization/serializer.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

namespace dtl::test {

// =============================================================================
// Basic Type Serialization Tests
// =============================================================================

TEST(TrivialSerializationTest, IntRoundtrip) {
    int original = 42;
    std::byte buffer[sizeof(int)];

    size_type written = serialize(original, buffer);
    EXPECT_EQ(written, sizeof(int));

    int recovered = deserialize<int>(buffer, sizeof(int));
    EXPECT_EQ(recovered, original);
}

TEST(TrivialSerializationTest, DoubleRoundtrip) {
    double original = 3.14159265358979;
    std::byte buffer[sizeof(double)];

    size_type written = serialize(original, buffer);
    EXPECT_EQ(written, sizeof(double));

    double recovered = deserialize<double>(buffer, sizeof(double));
    EXPECT_DOUBLE_EQ(recovered, original);
}

TEST(TrivialSerializationTest, FloatRoundtrip) {
    float original = 2.71828f;
    std::byte buffer[sizeof(float)];

    serialize(original, buffer);
    float recovered = deserialize<float>(buffer, sizeof(float));

    EXPECT_FLOAT_EQ(recovered, original);
}

TEST(TrivialSerializationTest, CharRoundtrip) {
    char original = 'X';
    std::byte buffer[sizeof(char)];

    serialize(original, buffer);
    char recovered = deserialize<char>(buffer, sizeof(char));

    EXPECT_EQ(recovered, original);
}

TEST(TrivialSerializationTest, BoolRoundtrip) {
    bool original_true = true;
    bool original_false = false;
    std::byte buffer[sizeof(bool)];

    serialize(original_true, buffer);
    EXPECT_EQ(deserialize<bool>(buffer, sizeof(bool)), true);

    serialize(original_false, buffer);
    EXPECT_EQ(deserialize<bool>(buffer, sizeof(bool)), false);
}

TEST(TrivialSerializationTest, Int64Roundtrip) {
    std::int64_t original = 0x123456789ABCDEF0LL;
    std::byte buffer[sizeof(std::int64_t)];

    serialize(original, buffer);
    std::int64_t recovered = deserialize<std::int64_t>(buffer, sizeof(std::int64_t));

    EXPECT_EQ(recovered, original);
}

TEST(TrivialSerializationTest, Uint32Roundtrip) {
    std::uint32_t original = 0xDEADBEEF;
    std::byte buffer[sizeof(std::uint32_t)];

    serialize(original, buffer);
    std::uint32_t recovered = deserialize<std::uint32_t>(buffer, sizeof(std::uint32_t));

    EXPECT_EQ(recovered, original);
}

// =============================================================================
// Serialized Size Tests
// =============================================================================

TEST(TrivialSerializationTest, SerializedSizeBasicTypes) {
    EXPECT_EQ(serialized_size(42), sizeof(int));
    EXPECT_EQ(serialized_size(3.14), sizeof(double));
    EXPECT_EQ(serialized_size(1.0f), sizeof(float));
    EXPECT_EQ(serialized_size('a'), sizeof(char));
    EXPECT_EQ(serialized_size(true), sizeof(bool));
}

TEST(TrivialSerializationTest, SerializedSizeNoInstance) {
    EXPECT_EQ(serializer<int>::serialized_size(), sizeof(int));
    EXPECT_EQ(serializer<double>::serialized_size(), sizeof(double));
    EXPECT_EQ(serializer<float>::serialized_size(), sizeof(float));
}

// =============================================================================
// POD Struct Serialization Tests
// =============================================================================

TEST(TrivialSerializationTest, SimplePODRoundtrip) {
    struct Point { int x, y; };
    Point original{10, 20};
    std::byte buffer[sizeof(Point)];

    serialize(original, buffer);
    Point recovered = deserialize<Point>(buffer, sizeof(Point));

    EXPECT_EQ(recovered.x, 10);
    EXPECT_EQ(recovered.y, 20);
}

TEST(TrivialSerializationTest, LargerPODRoundtrip) {
    struct Color { float r, g, b, a; };
    Color original{1.0f, 0.5f, 0.25f, 1.0f};
    std::byte buffer[sizeof(Color)];

    serialize(original, buffer);
    Color recovered = deserialize<Color>(buffer, sizeof(Color));

    EXPECT_FLOAT_EQ(recovered.r, 1.0f);
    EXPECT_FLOAT_EQ(recovered.g, 0.5f);
    EXPECT_FLOAT_EQ(recovered.b, 0.25f);
    EXPECT_FLOAT_EQ(recovered.a, 1.0f);
}

TEST(TrivialSerializationTest, NestedPODRoundtrip) {
    struct Inner { int a, b; };
    struct Outer { Inner in; double d; };

    Outer original{{1, 2}, 3.14};
    std::byte buffer[sizeof(Outer)];

    serialize(original, buffer);
    Outer recovered = deserialize<Outer>(buffer, sizeof(Outer));

    EXPECT_EQ(recovered.in.a, 1);
    EXPECT_EQ(recovered.in.b, 2);
    EXPECT_DOUBLE_EQ(recovered.d, 3.14);
}

// =============================================================================
// std::array Serialization Tests
// =============================================================================

TEST(TrivialSerializationTest, StdArrayIntRoundtrip) {
    std::array<int, 4> original = {1, 2, 3, 4};
    std::byte buffer[sizeof(std::array<int, 4>)];

    serialize(original, buffer);
    auto recovered = deserialize<std::array<int, 4>>(buffer, sizeof(std::array<int, 4>));

    EXPECT_EQ(recovered[0], 1);
    EXPECT_EQ(recovered[1], 2);
    EXPECT_EQ(recovered[2], 3);
    EXPECT_EQ(recovered[3], 4);
}

TEST(TrivialSerializationTest, StdArrayDoubleRoundtrip) {
    std::array<double, 3> original = {1.1, 2.2, 3.3};
    std::byte buffer[sizeof(std::array<double, 3>)];

    serialize(original, buffer);
    auto recovered = deserialize<std::array<double, 3>>(buffer, sizeof(std::array<double, 3>));

    EXPECT_DOUBLE_EQ(recovered[0], 1.1);
    EXPECT_DOUBLE_EQ(recovered[1], 2.2);
    EXPECT_DOUBLE_EQ(recovered[2], 3.3);
}

TEST(TrivialSerializationTest, StdArraySerializedSize) {
    using arr10 = std::array<int, 10>;
    arr10 arr{};
    EXPECT_EQ(serialized_size(arr), sizeof(arr10));
    EXPECT_EQ(serializer<arr10>::serialized_size(), sizeof(arr10));
}

// =============================================================================
// std::pair Serialization Tests
// =============================================================================

TEST(TrivialSerializationTest, StdPairIntIntRoundtrip) {
    std::pair<int, int> original{42, 99};
    std::byte buffer[sizeof(int) + sizeof(int)];

    serialize(original, buffer);
    auto recovered = deserialize<std::pair<int, int>>(buffer, sizeof(int) + sizeof(int));

    EXPECT_EQ(recovered.first, 42);
    EXPECT_EQ(recovered.second, 99);
}

TEST(TrivialSerializationTest, StdPairMixedRoundtrip) {
    std::pair<int, double> original{10, 3.14};
    std::byte buffer[sizeof(int) + sizeof(double)];

    serialize(original, buffer);
    auto recovered = deserialize<std::pair<int, double>>(buffer, sizeof(int) + sizeof(double));

    EXPECT_EQ(recovered.first, 10);
    EXPECT_DOUBLE_EQ(recovered.second, 3.14);
}

// =============================================================================
// Enum Serialization Tests
// =============================================================================

TEST(TrivialSerializationTest, EnumClassRoundtrip) {
    enum class Color : int { Red = 1, Green = 2, Blue = 3 };

    Color original = Color::Green;
    std::byte buffer[sizeof(Color)];

    serialize(original, buffer);
    Color recovered = deserialize<Color>(buffer, sizeof(Color));

    EXPECT_EQ(recovered, Color::Green);
}

TEST(TrivialSerializationTest, OldStyleEnumRoundtrip) {
    enum OldEnum { A = 10, B = 20, C = 30 };

    OldEnum original = B;
    std::byte buffer[sizeof(OldEnum)];

    serialize(original, buffer);
    OldEnum recovered = deserialize<OldEnum>(buffer, sizeof(OldEnum));

    EXPECT_EQ(recovered, B);
}

// =============================================================================
// is_trivial Tests
// =============================================================================

TEST(TrivialSerializationTest, IsTrivialMethod) {
    using arr4 = std::array<int, 4>;
    using pair_ii = std::pair<int, int>;

    EXPECT_TRUE(serializer<int>::is_trivial());
    EXPECT_TRUE(serializer<double>::is_trivial());
    EXPECT_TRUE(serializer<arr4>::is_trivial());
    EXPECT_TRUE(serializer<pair_ii>::is_trivial());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(TrivialSerializationTest, ZeroValue) {
    int original = 0;
    std::byte buffer[sizeof(int)];

    serialize(original, buffer);
    int recovered = deserialize<int>(buffer, sizeof(int));

    EXPECT_EQ(recovered, 0);
}

TEST(TrivialSerializationTest, NegativeValue) {
    int original = -12345;
    std::byte buffer[sizeof(int)];

    serialize(original, buffer);
    int recovered = deserialize<int>(buffer, sizeof(int));

    EXPECT_EQ(recovered, -12345);
}

TEST(TrivialSerializationTest, MaxValue) {
    int original = std::numeric_limits<int>::max();
    std::byte buffer[sizeof(int)];

    serialize(original, buffer);
    int recovered = deserialize<int>(buffer, sizeof(int));

    EXPECT_EQ(recovered, std::numeric_limits<int>::max());
}

TEST(TrivialSerializationTest, MinValue) {
    int original = std::numeric_limits<int>::min();
    std::byte buffer[sizeof(int)];

    serialize(original, buffer);
    int recovered = deserialize<int>(buffer, sizeof(int));

    EXPECT_EQ(recovered, std::numeric_limits<int>::min());
}

}  // namespace dtl::test
