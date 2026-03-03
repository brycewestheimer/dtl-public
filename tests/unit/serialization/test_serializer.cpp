// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_serializer.cpp
/// @brief Unit tests for dtl/serialization/serializer.hpp
/// @details Tests serializer trait, trivial serialization, and STL specializations.

#include <dtl/serialization/serializer.hpp>
#include <dtl/serialization/serialization_traits.hpp>

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Concept Tests
// =============================================================================

TEST(SerializerConceptTest, TriviallySerializablePrimitives) {
    static_assert(TriviallySerializable<int>);
    static_assert(TriviallySerializable<double>);
    static_assert(TriviallySerializable<float>);
    static_assert(TriviallySerializable<char>);
    static_assert(TriviallySerializable<std::uint64_t>);
}

TEST(SerializerConceptTest, NotTriviallySerializable) {
    static_assert(!TriviallySerializable<std::string>);
    static_assert(!TriviallySerializable<std::vector<int>>);
}

TEST(SerializerConceptTest, SerializablePrimitives) {
    static_assert(Serializable<int>);
    static_assert(Serializable<double>);
    static_assert(Serializable<float>);
}

TEST(SerializerConceptTest, SerializableArray) {
    static_assert(Serializable<std::array<int, 5>>);
    static_assert(Serializable<std::array<double, 10>>);
}

TEST(SerializerConceptTest, SerializablePair) {
    static_assert(Serializable<std::pair<int, double>>);
    static_assert(Serializable<std::pair<float, char>>);
}

TEST(SerializerConceptTest, FixedSizeSerializer) {
    static_assert(FixedSizeSerializer<int>);
    static_assert(FixedSizeSerializer<std::array<int, 5>>);
    static_assert(FixedSizeSerializer<std::pair<int, int>>);
}

// =============================================================================
// has_serializer Trait Tests
// =============================================================================

TEST(HasSerializerTest, PrimitivesHaveSerializer) {
    static_assert(has_serializer_v<int>);
    static_assert(has_serializer_v<double>);
    static_assert(has_serializer_v<float>);
}

TEST(HasSerializerTest, ArraysHaveSerializer) {
    static_assert(has_serializer_v<std::array<int, 3>>);
    static_assert(has_serializer_v<std::array<double, 10>>);
}

TEST(HasSerializerTest, PairsHaveSerializer) {
    static_assert(has_serializer_v<std::pair<int, int>>);
    static_assert(has_serializer_v<std::pair<float, double>>);
}

// =============================================================================
// Primitive Serialization Tests
// =============================================================================

TEST(SerializerTest, SerializeInt) {
    int value = 42;
    std::array<std::byte, sizeof(int)> buffer{};

    size_type written = serializer<int>::serialize(value, buffer.data());
    EXPECT_EQ(written, sizeof(int));

    int deserialized = serializer<int>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(deserialized, value);
}

TEST(SerializerTest, SerializeDouble) {
    double value = 3.14159265358979;
    std::array<std::byte, sizeof(double)> buffer{};

    size_type written = serializer<double>::serialize(value, buffer.data());
    EXPECT_EQ(written, sizeof(double));

    double deserialized = serializer<double>::deserialize(buffer.data(), buffer.size());
    EXPECT_DOUBLE_EQ(deserialized, value);
}

TEST(SerializerTest, SerializeNegativeInt) {
    int value = -12345;
    std::array<std::byte, sizeof(int)> buffer{};

    serializer<int>::serialize(value, buffer.data());
    int deserialized = serializer<int>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(deserialized, value);
}

TEST(SerializerTest, SerializedSizeConstexpr) {
    constexpr size_type int_size = serializer<int>::serialized_size();
    constexpr size_type double_size = serializer<double>::serialized_size();

    static_assert(int_size == sizeof(int));
    static_assert(double_size == sizeof(double));
}

// =============================================================================
// std::array Serialization Tests
// =============================================================================

TEST(SerializerTest, SerializeArray) {
    std::array<int, 5> value = {1, 2, 3, 4, 5};
    std::array<std::byte, sizeof(value)> buffer{};

    size_type written = serializer<std::array<int, 5>>::serialize(value, buffer.data());
    EXPECT_EQ(written, sizeof(value));

    auto deserialized = serializer<std::array<int, 5>>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(deserialized, value);
}

TEST(SerializerTest, SerializeArrayDouble) {
    std::array<double, 3> value = {1.1, 2.2, 3.3};
    std::array<std::byte, sizeof(value)> buffer{};

    serializer<std::array<double, 3>>::serialize(value, buffer.data());
    auto deserialized = serializer<std::array<double, 3>>::deserialize(buffer.data(), buffer.size());

    EXPECT_DOUBLE_EQ(deserialized[0], 1.1);
    EXPECT_DOUBLE_EQ(deserialized[1], 2.2);
    EXPECT_DOUBLE_EQ(deserialized[2], 3.3);
}

TEST(SerializerTest, ArraySerializedSize) {
    using array_type = std::array<int, 10>;
    constexpr size_type size = serializer<array_type>::serialized_size();
    static_assert(size == sizeof(array_type));

    array_type arr{};
    EXPECT_EQ(serializer<array_type>::serialized_size(arr), sizeof(array_type));
}

// =============================================================================
// std::pair Serialization Tests
// =============================================================================

TEST(SerializerTest, SerializePair) {
    std::pair<int, double> value = {42, 3.14};
    std::array<std::byte, sizeof(int) + sizeof(double)> buffer{};

    size_type written = serializer<std::pair<int, double>>::serialize(value, buffer.data());
    EXPECT_EQ(written, sizeof(int) + sizeof(double));

    auto deserialized = serializer<std::pair<int, double>>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(deserialized.first, 42);
    EXPECT_DOUBLE_EQ(deserialized.second, 3.14);
}

TEST(SerializerTest, SerializePairSameType) {
    std::pair<int, int> value = {100, 200};
    std::array<std::byte, sizeof(int) * 2> buffer{};

    serializer<std::pair<int, int>>::serialize(value, buffer.data());
    auto deserialized = serializer<std::pair<int, int>>::deserialize(buffer.data(), buffer.size());

    EXPECT_EQ(deserialized.first, 100);
    EXPECT_EQ(deserialized.second, 200);
}

TEST(SerializerTest, PairSerializedSize) {
    using pair_type = std::pair<int, double>;
    constexpr size_type size = serializer<pair_type>::serialized_size();
    static_assert(size == sizeof(int) + sizeof(double));
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(SerializerTest, SerializeHelperFunction) {
    int value = 42;
    std::array<std::byte, sizeof(int)> buffer{};

    size_type written = dtl::serialize(value, buffer.data());
    EXPECT_EQ(written, sizeof(int));
}

TEST(SerializerTest, DeserializeHelperFunction) {
    int original = 42;
    std::array<std::byte, sizeof(int)> buffer{};

    dtl::serialize(original, buffer.data());
    int deserialized = dtl::deserialize<int>(buffer.data(), buffer.size());

    EXPECT_EQ(deserialized, original);
}

TEST(SerializerTest, SerializedSizeHelperFunction) {
    int value = 42;
    EXPECT_EQ(dtl::serialized_size(value), sizeof(int));
}

// =============================================================================
// POD Struct Serialization Tests
// =============================================================================

TEST(SerializerTest, SerializePODStruct) {
    struct Point {
        int x;
        int y;
    };

    static_assert(TriviallySerializable<Point>);
    static_assert(Serializable<Point>);

    Point value = {10, 20};
    std::array<std::byte, sizeof(Point)> buffer{};

    serializer<Point>::serialize(value, buffer.data());
    Point deserialized = serializer<Point>::deserialize(buffer.data(), buffer.size());

    EXPECT_EQ(deserialized.x, 10);
    EXPECT_EQ(deserialized.y, 20);
}

TEST(SerializerTest, SerializeComplexPOD) {
    struct ComplexPOD {
        int id;
        double values[3];
        char tag;
    };

    static_assert(TriviallySerializable<ComplexPOD>);

    ComplexPOD value = {42, {1.1, 2.2, 3.3}, 'X'};
    std::array<std::byte, sizeof(ComplexPOD)> buffer{};

    serializer<ComplexPOD>::serialize(value, buffer.data());
    ComplexPOD deserialized = serializer<ComplexPOD>::deserialize(buffer.data(), buffer.size());

    EXPECT_EQ(deserialized.id, 42);
    EXPECT_DOUBLE_EQ(deserialized.values[0], 1.1);
    EXPECT_DOUBLE_EQ(deserialized.values[1], 2.2);
    EXPECT_DOUBLE_EQ(deserialized.values[2], 3.3);
    EXPECT_EQ(deserialized.tag, 'X');
}

// =============================================================================
// is_serializable Trait Tests
// =============================================================================

TEST(SerializationTraitsTest, IsSerializable) {
    static_assert(is_serializable_v<int>);
    static_assert(is_serializable_v<double>);
    static_assert(is_serializable_v<std::array<int, 5>>);
    static_assert(is_serializable_v<std::pair<int, double>>);
}

TEST(SerializationTraitsTest, SerializationStrategy) {
    EXPECT_EQ(get_serialization_strategy<int>(), serialization_strategy::trivial);
    EXPECT_EQ(get_serialization_strategy<double>(), serialization_strategy::trivial);

    struct POD { int x; };
    EXPECT_EQ(get_serialization_strategy<POD>(), serialization_strategy::trivial);
}

// =============================================================================
// Roundtrip Tests (Full Integration)
// =============================================================================

TEST(SerializerTest, RoundtripAllTypes) {
    // Test that all types can roundtrip correctly
    std::vector<std::byte> buffer(1024);

    // Int
    {
        int original = -12345;
        dtl::serialize(original, buffer.data());
        EXPECT_EQ(dtl::deserialize<int>(buffer.data(), sizeof(int)), original);
    }

    // Double
    {
        double original = 3.14159265358979;
        dtl::serialize(original, buffer.data());
        EXPECT_DOUBLE_EQ(dtl::deserialize<double>(buffer.data(), sizeof(double)), original);
    }

    // Array
    {
        std::array<int, 4> original = {1, 2, 3, 4};
        dtl::serialize(original, buffer.data());
        EXPECT_EQ((dtl::deserialize<std::array<int, 4>>(buffer.data(), sizeof(original))), original);
    }

    // Pair
    {
        std::pair<int, float> original = {42, 2.5f};
        dtl::serialize(original, buffer.data());
        auto result = dtl::deserialize<std::pair<int, float>>(buffer.data(), sizeof(int) + sizeof(float));
        EXPECT_EQ(result.first, 42);
        EXPECT_FLOAT_EQ(result.second, 2.5f);
    }
}

}  // namespace dtl::test
