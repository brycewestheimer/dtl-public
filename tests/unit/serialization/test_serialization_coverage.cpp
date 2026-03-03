// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_serialization_coverage.cpp
/// @brief Expanded unit tests for the DTL serialization module
/// @details Phase 14 T07: serialization_traits, serializer, trivial_buffer.

#include <dtl/serialization/serialization.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dtl::test {

// =============================================================================
// Serialization Traits Tests
// =============================================================================

TEST(SerializationTraitsTest, IntIsTriviallySerializable) {
    EXPECT_TRUE(is_trivially_serializable_v<int>);
    EXPECT_TRUE(is_serializable_v<int>);
}

TEST(SerializationTraitsTest, DoubleIsTriviallySerializable) {
    EXPECT_TRUE(is_trivially_serializable_v<double>);
    EXPECT_TRUE(is_serializable_v<double>);
}

TEST(SerializationTraitsTest, FloatIsTriviallySerializable) {
    EXPECT_TRUE(is_trivially_serializable_v<float>);
}

TEST(SerializationTraitsTest, CharIsTriviallySerializable) {
    EXPECT_TRUE(is_trivially_serializable_v<char>);
}

TEST(SerializationTraitsTest, StrategyForInt) {
    auto s = get_serialization_strategy<int>();
    EXPECT_EQ(s, serialization_strategy::trivial);
}

TEST(SerializationTraitsTest, StrategyForDouble) {
    auto s = get_serialization_strategy<double>();
    EXPECT_EQ(s, serialization_strategy::trivial);
}

// A type with no serialization support
struct non_serializable_type {
    std::vector<int> data;  // not trivially copyable
};

TEST(SerializationTraitsTest, NonSerializableTypeDetection) {
    // std::vector is not trivially serializable
    EXPECT_FALSE(is_trivially_serializable_v<std::vector<int>>);
}

// =============================================================================
// Serializer<trivial> Tests
// =============================================================================

TEST(TrivialSerializerTest, SerializedSizeInt) {
    int x = 42;
    auto sz = serializer<int>::serialized_size(x);
    EXPECT_EQ(sz, sizeof(int));
}

TEST(TrivialSerializerTest, SerializedSizeNoInstance) {
    auto sz = serializer<int>::serialized_size();
    EXPECT_EQ(sz, sizeof(int));
}

TEST(TrivialSerializerTest, SerializeDeserializeInt) {
    int original = 12345;
    std::byte buffer[sizeof(int)];

    auto written = serializer<int>::serialize(original, buffer);
    EXPECT_EQ(written, sizeof(int));

    auto restored = serializer<int>::deserialize(buffer, sizeof(int));
    EXPECT_EQ(restored, original);
}

TEST(TrivialSerializerTest, SerializeDeserializeDouble) {
    double original = 3.14159265358979;
    std::byte buffer[sizeof(double)];

    serializer<double>::serialize(original, buffer);
    auto restored = serializer<double>::deserialize(buffer, sizeof(double));
    EXPECT_DOUBLE_EQ(restored, original);
}

TEST(TrivialSerializerTest, IsTrivial) {
    EXPECT_TRUE(serializer<int>::is_trivial());
    EXPECT_TRUE(serializer<double>::is_trivial());
}

TEST(TrivialSerializerTest, RoundtripUint64) {
    std::uint64_t original = 0xDEADBEEFCAFEBABEULL;
    std::byte buffer[sizeof(std::uint64_t)];

    serializer<std::uint64_t>::serialize(original, buffer);
    auto restored = serializer<std::uint64_t>::deserialize(buffer, sizeof(std::uint64_t));
    EXPECT_EQ(restored, original);
}

// =============================================================================
// Serializer<std::array> Tests
// =============================================================================

TEST(ArraySerializerTest, SerializeDeserialize) {
    std::array<int, 4> original = {1, 2, 3, 4};
    std::byte buffer[sizeof(std::array<int, 4>)];

    auto written = serializer<std::array<int, 4>>::serialize(original, buffer);
    EXPECT_EQ(written, sizeof(std::array<int, 4>));

    auto restored = serializer<std::array<int, 4>>::deserialize(buffer, sizeof(std::array<int, 4>));
    EXPECT_EQ(restored, original);
}

TEST(ArraySerializerTest, SerializedSize) {
    std::array<double, 3> arr = {1.0, 2.0, 3.0};
    EXPECT_EQ((serializer<std::array<double, 3>>::serialized_size(arr)),
              sizeof(std::array<double, 3>));
}

TEST(ArraySerializerTest, IsTrivial) {
    EXPECT_TRUE((serializer<std::array<int, 5>>::is_trivial()));
}

// =============================================================================
// Serializer<std::pair> Tests
// =============================================================================

TEST(PairSerializerTest, SerializeDeserialize) {
    std::pair<int, double> original{42, 3.14};
    std::byte buffer[sizeof(int) + sizeof(double)];

    auto written = serializer<std::pair<int, double>>::serialize(original, buffer);
    EXPECT_EQ(written, sizeof(int) + sizeof(double));

    auto restored = serializer<std::pair<int, double>>::deserialize(
        buffer, sizeof(int) + sizeof(double));
    EXPECT_EQ(restored.first, 42);
    EXPECT_DOUBLE_EQ(restored.second, 3.14);
}

TEST(PairSerializerTest, SerializedSize) {
    std::pair<int, int> p{1, 2};
    EXPECT_EQ((serializer<std::pair<int, int>>::serialized_size(p)),
              sizeof(int) + sizeof(int));
}

// =============================================================================
// trivial_buffer Tests
// =============================================================================

TEST(TrivialBufferTest, DefaultConstruction) {
    trivial_buffer buf;
    EXPECT_EQ(buf.position(), 0u);
    EXPECT_GT(buf.capacity(), 0u);
}

TEST(TrivialBufferTest, ConstructWithCapacity) {
    trivial_buffer buf(4096);
    EXPECT_EQ(buf.position(), 0u);
    EXPECT_GE(buf.capacity(), 4096u);
}

TEST(TrivialBufferTest, WriteSingleValue) {
    trivial_buffer buf;
    int value = 42;
    buf.write(value);
    EXPECT_EQ(buf.position(), sizeof(int));
}

TEST(TrivialBufferTest, WriteMultipleValues) {
    trivial_buffer buf;
    buf.write(1);
    buf.write(2.0);
    char ch = 'A';
    buf.write(ch);
    EXPECT_EQ(buf.position(), sizeof(int) + sizeof(double) + sizeof(char));
}

TEST(TrivialBufferTest, WriteRange) {
    trivial_buffer buf;
    int data[] = {10, 20, 30};
    buf.write_range(data, 3);
    EXPECT_EQ(buf.position(), 3 * sizeof(int));
}

TEST(TrivialBufferTest, Reset) {
    trivial_buffer buf;
    buf.write(42);
    EXPECT_GT(buf.position(), 0u);
    buf.reset();
    EXPECT_EQ(buf.position(), 0u);
}

TEST(TrivialBufferTest, Clear) {
    trivial_buffer buf;
    buf.write(42);
    buf.clear();
    EXPECT_EQ(buf.position(), 0u);
}

TEST(TrivialBufferTest, Reserve) {
    trivial_buffer buf(64);
    buf.reserve(8192);
    EXPECT_GE(buf.capacity(), 8192u);
}

TEST(TrivialBufferTest, WriteAndReadBack) {
    trivial_buffer buf;
    int original = 99;
    buf.write(original);
    // Read back from buffer data
    int restored;
    std::memcpy(&restored, buf.data(), sizeof(int));
    EXPECT_EQ(restored, original);
}

TEST(TrivialBufferTest, AutoGrow) {
    trivial_buffer buf(16);  // Small initial capacity
    // Write enough to force growth
    for (int i = 0; i < 100; ++i) {
        buf.write(i);
    }
    EXPECT_EQ(buf.position(), 100 * sizeof(int));
    // Verify data integrity
    int first;
    std::memcpy(&first, buf.data(), sizeof(int));
    EXPECT_EQ(first, 0);
}

// =============================================================================
// Bulk Trivial Serialization Tests
// =============================================================================

TEST(BulkTrivialTest, SerializeRange) {
    int data[] = {1, 2, 3, 4, 5};
    std::byte buffer[5 * sizeof(int)];
    auto written = serialize_trivial_range(data, 5, buffer);
    EXPECT_EQ(written, 5 * sizeof(int));

    int restored[5];
    auto read = deserialize_trivial_range(buffer, 5, restored);
    EXPECT_EQ(read, 5 * sizeof(int));
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(restored[i], i + 1);
    }
}

TEST(BulkTrivialTest, DeserializeVector) {
    double data[] = {1.1, 2.2, 3.3};
    std::byte buffer[3 * sizeof(double)];
    serialize_trivial_range(data, 3, buffer);

    auto vec = deserialize_trivial_vector<double>(buffer, 3);
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_DOUBLE_EQ(vec[0], 1.1);
    EXPECT_DOUBLE_EQ(vec[1], 2.2);
    EXPECT_DOUBLE_EQ(vec[2], 3.3);
}

TEST(BulkTrivialTest, CanUseTrivialSerialization) {
    EXPECT_TRUE(can_use_trivial_serialization<int>());
    EXPECT_TRUE(can_use_trivial_serialization<double>());
    EXPECT_TRUE(can_use_trivial_serialization<char>());
}

TEST(BulkTrivialTest, AlignedSize) {
    EXPECT_EQ(aligned_size(1), 8u);
    EXPECT_EQ(aligned_size(8), 8u);
    EXPECT_EQ(aligned_size(9), 16u);
    EXPECT_EQ(aligned_size(15, 16), 16u);
    EXPECT_EQ(aligned_size(17, 16), 32u);
}

// =============================================================================
// Member Serialization Detection Tests
// =============================================================================

struct simple_struct {
    int a;
    double b;
};

TEST(MemberSerializationTest, SimpleStructNotMemberSerializable) {
    EXPECT_FALSE(has_member_serialize_v<simple_struct>);
    EXPECT_FALSE(has_member_deserialize_v<simple_struct>);
    EXPECT_FALSE(has_complete_member_serialization_v<simple_struct>);
}

TEST(MemberSerializationTest, IntNotMemberSerializable) {
    EXPECT_FALSE(has_member_serialize_v<int>);
}

// =============================================================================
// Serialization Strategy Tests
// =============================================================================

TEST(SerializationStrategyTest, EnumValues) {
    EXPECT_NE(serialization_strategy::trivial, serialization_strategy::member_functions);
    EXPECT_NE(serialization_strategy::member_functions, serialization_strategy::specialization);
    EXPECT_NE(serialization_strategy::specialization, serialization_strategy::not_serializable);
}

}  // namespace dtl::test
