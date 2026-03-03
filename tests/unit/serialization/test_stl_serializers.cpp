// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_stl_serializers.cpp
/// @brief Unit tests for STL type serializers (string, vector, optional)
/// @details Phase 13 T05: Tests serialization round-trips for built-in
///          STL type serializers.

// Suppress false-positive from GCC 13 in Release mode: aggressive inlining
// causes memcpy bounds analysis to misfire on empty container serialization.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#endif

#include <dtl/serialization/serialization.hpp>
#include <dtl/serialization/stl_serializers.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// std::string Serializer Tests
// =============================================================================

TEST(StringSerializerTest, EmptyString) {
    std::string original;
    auto size = serializer<std::string>::serialized_size(original);
    EXPECT_EQ(size, sizeof(size_type));

    std::vector<std::byte> buffer(size);
    auto written = serializer<std::string>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);

    auto result = serializer<std::string>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(StringSerializerTest, NonEmptyString) {
    std::string original = "Hello, DTL!";
    auto size = serializer<std::string>::serialized_size(original);
    EXPECT_EQ(size, sizeof(size_type) + original.size());

    std::vector<std::byte> buffer(size);
    auto written = serializer<std::string>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);

    auto result = serializer<std::string>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(StringSerializerTest, LongString) {
    std::string original(10000, 'x');
    auto size = serializer<std::string>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::string>::serialize(original, buffer.data());

    auto result = serializer<std::string>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(StringSerializerTest, SpecialCharacters) {
    std::string original = "null\0embedded";
    original.resize(14);  // Ensure null is retained
    auto size = serializer<std::string>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::string>::serialize(original, buffer.data());

    auto result = serializer<std::string>::deserialize(buffer.data(), size);
    EXPECT_EQ(result.size(), original.size());
}

TEST(StringSerializerTest, IsNotTrivial) {
    EXPECT_FALSE(serializer<std::string>::is_trivial());
}

// Helper: use free-function interface
TEST(StringSerializerTest, FreeFunctionInterface) {
    std::string original = "Free functions work";
    auto size = dtl::serialized_size(original);
    std::vector<std::byte> buffer(size);
    dtl::serialize(original, buffer.data());
    auto result = dtl::deserialize<std::string>(buffer.data(), size);
    EXPECT_EQ(result, original);
}

// =============================================================================
// std::vector<T> Serializer Tests (Trivial Elements)
// =============================================================================

TEST(VectorSerializerTrivialTest, EmptyVector) {
    std::vector<int> original;
    auto size = serializer<std::vector<int>>::serialized_size(original);
    EXPECT_EQ(size, sizeof(size_type));

    std::vector<std::byte> buffer(size);
    auto written = serializer<std::vector<int>>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);

    auto result = serializer<std::vector<int>>::deserialize(buffer.data(), size);
    EXPECT_TRUE(result.empty());
}

TEST(VectorSerializerTrivialTest, IntVector) {
    std::vector<int> original = {1, 2, 3, 4, 5};
    auto size = serializer<std::vector<int>>::serialized_size(original);
    EXPECT_EQ(size, sizeof(size_type) + 5 * sizeof(int));

    std::vector<std::byte> buffer(size);
    serializer<std::vector<int>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<int>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(VectorSerializerTrivialTest, DoubleVector) {
    std::vector<double> original = {1.5, 2.7, 3.14159};
    auto size = serializer<std::vector<double>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::vector<double>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<double>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(VectorSerializerTrivialTest, LargeVector) {
    std::vector<int> original(10000);
    for (int i = 0; i < 10000; ++i) {
        original[static_cast<size_type>(i)] = i;
    }

    auto size = serializer<std::vector<int>>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    serializer<std::vector<int>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<int>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

// =============================================================================
// std::vector<T> Serializer Tests (Non-Trivial Elements)
// =============================================================================

TEST(VectorSerializerNonTrivialTest, StringVector) {
    std::vector<std::string> original = {"hello", "world", "dtl"};
    auto size = serializer<std::vector<std::string>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::vector<std::string>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<std::string>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(VectorSerializerNonTrivialTest, EmptyStringVector) {
    std::vector<std::string> original = {"", "", ""};
    auto size = serializer<std::vector<std::string>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::vector<std::string>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<std::string>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(VectorSerializerNonTrivialTest, NestedVector) {
    std::vector<std::vector<int>> original = {{1, 2}, {3, 4, 5}, {6}};
    auto size = serializer<std::vector<std::vector<int>>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::vector<std::vector<int>>>::serialize(original, buffer.data());

    auto result = serializer<std::vector<std::vector<int>>>::deserialize(buffer.data(), size);
    EXPECT_EQ(result, original);
}

TEST(VectorSerializerTest, IsNotTrivial) {
    EXPECT_FALSE(serializer<std::vector<int>>::is_trivial());
}

// =============================================================================
// std::optional<T> Serializer Tests
// =============================================================================

TEST(OptionalSerializerTest, EmptyOptional) {
    std::optional<int> original;
    auto size = serializer<std::optional<int>>::serialized_size(original);
    EXPECT_EQ(size, 1u);

    std::vector<std::byte> buffer(size);
    serializer<std::optional<int>>::serialize(original, buffer.data());

    auto result = serializer<std::optional<int>>::deserialize(buffer.data(), size);
    EXPECT_FALSE(result.has_value());
}

TEST(OptionalSerializerTest, ValueOptional) {
    std::optional<int> original = 42;
    auto size = serializer<std::optional<int>>::serialized_size(original);
    EXPECT_EQ(size, 1u + sizeof(int));

    std::vector<std::byte> buffer(size);
    serializer<std::optional<int>>::serialize(original, buffer.data());

    auto result = serializer<std::optional<int>>::deserialize(buffer.data(), size);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST(OptionalSerializerTest, StringOptional) {
    std::optional<std::string> original = std::string("hello");
    auto size = serializer<std::optional<std::string>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::optional<std::string>>::serialize(original, buffer.data());

    auto result = serializer<std::optional<std::string>>::deserialize(buffer.data(), size);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, "hello");
}

TEST(OptionalSerializerTest, EmptyStringOptional) {
    std::optional<std::string> original;
    auto size = serializer<std::optional<std::string>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::optional<std::string>>::serialize(original, buffer.data());

    auto result = serializer<std::optional<std::string>>::deserialize(buffer.data(), size);
    EXPECT_FALSE(result.has_value());
}

TEST(OptionalSerializerTest, DoubleOptional) {
    std::optional<double> original = 3.14159;
    auto size = serializer<std::optional<double>>::serialized_size(original);

    std::vector<std::byte> buffer(size);
    serializer<std::optional<double>>::serialize(original, buffer.data());

    auto result = serializer<std::optional<double>>::deserialize(buffer.data(), size);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 3.14159);
}

TEST(OptionalSerializerTest, IsNotTrivial) {
    EXPECT_FALSE(serializer<std::optional<int>>::is_trivial());
}

// =============================================================================
// Serialization Trait Detection Tests
// =============================================================================

TEST(SerializationTraitsTest, STLTypesAreSerializable) {
    EXPECT_TRUE(has_serializer_v<std::string>);
    EXPECT_TRUE(has_serializer_v<std::vector<int>>);
    EXPECT_TRUE(has_serializer_v<std::optional<int>>);
    EXPECT_TRUE(has_serializer_v<std::vector<std::string>>);
    EXPECT_TRUE(has_serializer_v<std::optional<std::string>>);
}

TEST(SerializationTraitsTest, Serializable) {
    static_assert(Serializable<std::string>);
    static_assert(Serializable<std::vector<int>>);
    static_assert(Serializable<std::optional<int>>);
}

}  // namespace dtl::test
