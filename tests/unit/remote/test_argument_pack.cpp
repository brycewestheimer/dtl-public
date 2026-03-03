// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_argument_pack.cpp
/// @brief Unit tests for dtl/remote/argument_pack.hpp
/// @details Tests argument pack serialization and deserialization.

#include <dtl/remote/argument_pack.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dtl::remote::test {

// =============================================================================
// Basic Serialization Tests
// =============================================================================

TEST(ArgumentPackTest, EmptyPack) {
    using pack = argument_pack<>;

    EXPECT_EQ(pack::serialized_size(), 0u);

    auto data = pack::serialize_to_vector();
    EXPECT_TRUE(data.empty());

    auto result = pack::deserialize(nullptr, 0);
    static_assert(std::is_same_v<decltype(result), std::tuple<>>);
}

TEST(ArgumentPackTest, SingleInt) {
    using pack = argument_pack<int>;

    int value = 42;
    size_type size = pack::serialized_size(value);
    EXPECT_EQ(size, sizeof(int));

    auto data = pack::serialize_to_vector(value);
    EXPECT_EQ(data.size(), sizeof(int));

    auto result = pack::deserialize(data.data(), data.size());
    EXPECT_EQ(std::get<0>(result), 42);
}

TEST(ArgumentPackTest, TwoInts) {
    using pack = argument_pack<int, int>;

    int a = 10, b = 20;
    size_type size = pack::serialized_size(a, b);
    EXPECT_EQ(size, 2 * sizeof(int));

    auto data = pack::serialize_to_vector(a, b);
    EXPECT_EQ(data.size(), 2 * sizeof(int));

    auto result = pack::deserialize(data.data(), data.size());
    EXPECT_EQ(std::get<0>(result), 10);
    EXPECT_EQ(std::get<1>(result), 20);
}

TEST(ArgumentPackTest, MixedTrivialTypes) {
    using pack = argument_pack<int, double, char>;

    int i = 100;
    double d = 3.14;
    char c = 'X';

    auto data = pack::serialize_to_vector(i, d, c);
    size_type expected = sizeof(int) + sizeof(double) + sizeof(char);
    EXPECT_EQ(data.size(), expected);

    auto result = pack::deserialize(data.data(), data.size());
    EXPECT_EQ(std::get<0>(result), 100);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 3.14);
    EXPECT_EQ(std::get<2>(result), 'X');
}

// =============================================================================
// Tuple Serialization Tests
// =============================================================================

TEST(ArgumentPackTest, SerializeFromTuple) {
    using pack = argument_pack<int, int>;
    using tuple_t = std::tuple<int, int>;

    tuple_t args{5, 10};

    size_type size = pack::serialized_size(args);
    EXPECT_EQ(size, 2 * sizeof(int));

    auto data = pack::serialize_to_vector(args);

    auto result = pack::deserialize(data.data(), data.size());
    EXPECT_EQ(result, args);
}

// =============================================================================
// Buffer Serialization Tests
// =============================================================================

TEST(ArgumentPackTest, SerializeToBuffer) {
    using pack = argument_pack<int, int>;

    int a = 1, b = 2;
    std::vector<std::byte> buffer(pack::serialized_size(a, b));

    size_type written = pack::serialize(a, b, buffer.data());
    EXPECT_EQ(written, buffer.size());

    auto result = pack::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(std::get<0>(result), 1);
    EXPECT_EQ(std::get<1>(result), 2);
}

// =============================================================================
// Array Serialization Tests
// =============================================================================

TEST(ArgumentPackTest, StdArray) {
    using pack = argument_pack<std::array<int, 3>>;

    std::array<int, 3> arr{1, 2, 3};

    auto data = pack::serialize_to_vector(arr);

    auto result = pack::deserialize(data.data(), data.size());
    std::array<int, 3> out = std::get<0>(result);

    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 3);
}

// =============================================================================
// Pair Serialization Tests
// =============================================================================

TEST(ArgumentPackTest, StdPair) {
    using pack = argument_pack<std::pair<int, double>>;

    std::pair<int, double> p{42, 2.718};

    auto data = pack::serialize_to_vector(p);

    auto result = pack::deserialize(data.data(), data.size());
    auto out = std::get<0>(result);

    EXPECT_EQ(out.first, 42);
    EXPECT_DOUBLE_EQ(out.second, 2.718);
}

// =============================================================================
// Multiple Arguments with Different Types
// =============================================================================

TEST(ArgumentPackTest, FourArguments) {
    using pack = argument_pack<std::int32_t, std::int64_t, float, double>;

    std::int32_t a = 100;
    std::int64_t b = 1000000000000LL;
    float c = 1.5f;
    double d = 2.5;

    auto data = pack::serialize_to_vector(a, b, c, d);

    auto result = pack::deserialize(data.data(), data.size());
    EXPECT_EQ(std::get<0>(result), 100);
    EXPECT_EQ(std::get<1>(result), 1000000000000LL);
    EXPECT_FLOAT_EQ(std::get<2>(result), 1.5f);
    EXPECT_DOUBLE_EQ(std::get<3>(result), 2.5);
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

TEST(ArgumentPackTest, RoundtripTrivial) {
    using pack = argument_pack<int>;

    for (int val : {0, 1, -1, 100, -100, INT_MAX, INT_MIN}) {
        auto data = pack::serialize_to_vector(val);
        auto result = pack::deserialize(data.data(), data.size());
        EXPECT_EQ(std::get<0>(result), val);
    }
}

TEST(ArgumentPackTest, RoundtripDouble) {
    using pack = argument_pack<double>;

    for (double val : {0.0, 1.0, -1.0, 3.14159265359, 1e308, -1e308}) {
        auto data = pack::serialize_to_vector(val);
        auto result = pack::deserialize(data.data(), data.size());
        EXPECT_DOUBLE_EQ(std::get<0>(result), val);
    }
}

// =============================================================================
// Pack Traits Tests
// =============================================================================

TEST(ArgumentPackTraitsTest, AllSerializable) {
    static_assert(all_serializable_v<int>);
    static_assert(all_serializable_v<int, double>);
    static_assert(all_serializable_v<int, double, char>);
    static_assert(all_serializable_v<std::array<int, 5>>);
    static_assert(all_serializable_v<std::pair<int, double>>);
}

TEST(ArgumentPackTraitsTest, SerializableArgsConcept) {
    static_assert(SerializableArgs<int>);
    static_assert(SerializableArgs<int, int>);
    static_assert(SerializableArgs<double, float, int>);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ArgumentPackTest, BoolValues) {
    using pack = argument_pack<bool, bool>;

    auto data = pack::serialize_to_vector(true, false);
    auto result = pack::deserialize(data.data(), data.size());

    EXPECT_TRUE(std::get<0>(result));
    EXPECT_FALSE(std::get<1>(result));
}

TEST(ArgumentPackTest, CharValues) {
    using pack = argument_pack<char, char, char>;

    auto data = pack::serialize_to_vector('A', 'B', 'C');
    auto result = pack::deserialize(data.data(), data.size());

    EXPECT_EQ(std::get<0>(result), 'A');
    EXPECT_EQ(std::get<1>(result), 'B');
    EXPECT_EQ(std::get<2>(result), 'C');
}

}  // namespace dtl::remote::test
