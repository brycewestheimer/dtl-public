// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_custom_serialization.cpp
/// @brief Unit tests for custom type serialization
/// @details Tests for Phase 11.5: custom serializer trait specialization

#include <dtl/serialization/serializer.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// has_serializer Trait Tests
// =============================================================================

TEST(CustomSerializationTest, HasSerializerTrivialTypes) {
    // Trivial types have the default serializer
    static_assert(has_serializer_v<int>);
    static_assert(has_serializer_v<double>);
    static_assert(has_serializer_v<float>);

    EXPECT_TRUE(has_serializer_v<int>);
}

TEST(CustomSerializationTest, HasSerializerStdArray) {
    using arr4 = std::array<int, 4>;
    using arr10 = std::array<double, 10>;

    static_assert(has_serializer_v<arr4>);
    static_assert(has_serializer_v<arr10>);

    EXPECT_TRUE(has_serializer_v<arr4>);
}

TEST(CustomSerializationTest, HasSerializerStdPair) {
    using pair_ii = std::pair<int, int>;
    using pair_id = std::pair<int, double>;

    static_assert(has_serializer_v<pair_ii>);
    static_assert(has_serializer_v<pair_id>);

    EXPECT_TRUE(has_serializer_v<pair_ii>);
}

// =============================================================================
// Serializable Concept Tests
// =============================================================================

TEST(CustomSerializationTest, SerializableConceptTrivial) {
    using arr4 = std::array<int, 4>;

    static_assert(Serializable<int>);
    static_assert(Serializable<double>);
    static_assert(Serializable<arr4>);

    EXPECT_TRUE(Serializable<int>);
}

TEST(CustomSerializationTest, FixedSizeSerializerConcept) {
    using arr4 = std::array<int, 4>;
    using pair_ii = std::pair<int, int>;

    // Trivial types have fixed-size serializers
    static_assert(FixedSizeSerializer<int>);
    static_assert(FixedSizeSerializer<double>);
    static_assert(FixedSizeSerializer<arr4>);
    static_assert(FixedSizeSerializer<pair_ii>);

    EXPECT_TRUE(FixedSizeSerializer<int>);
}

// =============================================================================
// Custom Serializer Definition Tests
// =============================================================================

// Define a custom type that needs a custom serializer
struct dynamic_string {
    std::string data;
};

}  // namespace dtl::test

// Custom serializer specialization in dtl namespace
namespace dtl {

template <>
struct serializer<test::dynamic_string> {
    static size_type serialized_size(const test::dynamic_string& value) {
        // Size prefix + string data
        return sizeof(size_type) + value.data.size();
    }

    static size_type serialize(const test::dynamic_string& value, std::byte* buffer) {
        size_type len = value.data.size();
        std::memcpy(buffer, &len, sizeof(size_type));
        std::memcpy(buffer + sizeof(size_type), value.data.data(), len);
        return sizeof(size_type) + len;
    }

    static test::dynamic_string deserialize(const std::byte* buffer, size_type size) {
        (void)size;
        size_type len;
        std::memcpy(&len, buffer, sizeof(size_type));
        test::dynamic_string result;
        result.data = std::string(reinterpret_cast<const char*>(buffer + sizeof(size_type)), len);
        return result;
    }

    static constexpr bool is_trivial() noexcept {
        return false;
    }
};

}  // namespace dtl

namespace dtl::test {

TEST(CustomSerializationTest, CustomTypeHasSerializer) {
    static_assert(has_serializer_v<dynamic_string>);

    EXPECT_TRUE(has_serializer_v<dynamic_string>);
}

TEST(CustomSerializationTest, CustomTypeSerializable) {
    static_assert(Serializable<dynamic_string>);

    EXPECT_TRUE(Serializable<dynamic_string>);
}

TEST(CustomSerializationTest, CustomTypeNotFixedSize) {
    // dynamic_string is variable-size (no serialized_size() without instance)
    static_assert(!FixedSizeSerializer<dynamic_string>);

    EXPECT_FALSE(FixedSizeSerializer<dynamic_string>);
}

TEST(CustomSerializationTest, CustomTypeRoundtrip) {
    dynamic_string original;
    original.data = "Hello, World!";

    size_type size = serialized_size(original);
    std::vector<std::byte> buffer(size);

    serialize(original, buffer.data());
    dynamic_string recovered = deserialize<dynamic_string>(buffer.data(), size);

    EXPECT_EQ(recovered.data, "Hello, World!");
}

TEST(CustomSerializationTest, CustomTypeEmptyString) {
    dynamic_string original;
    original.data = "";

    size_type size = serialized_size(original);
    std::vector<std::byte> buffer(size);

    serialize(original, buffer.data());
    dynamic_string recovered = deserialize<dynamic_string>(buffer.data(), size);

    EXPECT_EQ(recovered.data, "");
}

TEST(CustomSerializationTest, CustomTypeLongString) {
    dynamic_string original;
    original.data = std::string(1000, 'X');

    size_type size = serialized_size(original);
    std::vector<std::byte> buffer(size);

    serialize(original, buffer.data());
    dynamic_string recovered = deserialize<dynamic_string>(buffer.data(), size);

    EXPECT_EQ(recovered.data.size(), 1000);
    EXPECT_EQ(recovered.data, std::string(1000, 'X'));
}

TEST(CustomSerializationTest, CustomTypeIsTrivialFalse) {
    EXPECT_FALSE(serializer<dynamic_string>::is_trivial());
}

// =============================================================================
// is_std_array and is_std_pair Trait Tests
// =============================================================================

TEST(CustomSerializationTest, IsStdArrayTrait) {
    using arr4 = std::array<int, 4>;
    using arr100 = std::array<double, 100>;
    using pair_ii = std::pair<int, int>;

    static_assert(is_std_array_v<arr4>);
    static_assert(is_std_array_v<arr100>);

    static_assert(!is_std_array_v<int>);
    static_assert(!is_std_array_v<std::vector<int>>);
    static_assert(!is_std_array_v<pair_ii>);

    EXPECT_TRUE(is_std_array_v<arr4>);
}

TEST(CustomSerializationTest, IsStdPairTrait) {
    using pair_ii = std::pair<int, int>;
    using pair_df = std::pair<double, float>;
    using arr2 = std::array<int, 2>;

    static_assert(is_std_pair_v<pair_ii>);
    static_assert(is_std_pair_v<pair_df>);

    static_assert(!is_std_pair_v<int>);
    static_assert(!is_std_pair_v<std::vector<int>>);
    static_assert(!is_std_pair_v<arr2>);

    EXPECT_TRUE(is_std_pair_v<pair_ii>);
}

// =============================================================================
// POD Struct with Custom Layout Tests
// =============================================================================

TEST(CustomSerializationTest, PODWithPadding) {
    // A struct that might have padding
    struct PaddedStruct {
        char c;
        // Potential 3 or 7 bytes of padding here
        double d;
        // Potential padding
        int i;
    };

    // Should still be trivially serializable
    static_assert(is_trivially_serializable_v<PaddedStruct>);

    PaddedStruct original{'A', 3.14, 42};
    std::byte buffer[sizeof(PaddedStruct)];

    serialize(original, buffer);
    PaddedStruct recovered = deserialize<PaddedStruct>(buffer, sizeof(PaddedStruct));

    EXPECT_EQ(recovered.c, 'A');
    EXPECT_DOUBLE_EQ(recovered.d, 3.14);
    EXPECT_EQ(recovered.i, 42);
}

}  // namespace dtl::test
