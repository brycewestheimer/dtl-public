// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_aggregate_serializer.cpp
/// @brief Unit tests for DTL_SERIALIZABLE macro

#include <dtl/serialization/serialization.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace dtl::test {

// ============================================================================
// Test Aggregate Types
// ============================================================================

struct SimpleAggregate {
    int x;
    double y;
};

struct MixedAggregate {
    int id;
    std::string name;
    std::vector<double> data;
};

struct SingleField {
    int value;
};

struct NestedAggregate {
    int tag;
    std::string label;
};

struct OuterAggregate {
    int id;
    NestedAggregate inner;
};

}  // namespace dtl::test

// Macro invocations must be at namespace scope (outside dtl::test)
DTL_SERIALIZABLE(dtl::test::SimpleAggregate, x, y);
DTL_SERIALIZABLE(dtl::test::MixedAggregate, id, name, data);
DTL_SERIALIZABLE(dtl::test::SingleField, value);
DTL_SERIALIZABLE(dtl::test::NestedAggregate, tag, label);
DTL_SERIALIZABLE(dtl::test::OuterAggregate, id, inner);

namespace dtl::test {

// ============================================================================
// Roundtrip Tests
// ============================================================================

TEST(AggregateSerializerTest, SimpleAggregateRoundtrip) {
    SimpleAggregate original{42, 3.14};

    auto sz = dtl::serializer<SimpleAggregate>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<SimpleAggregate>::serialize(original, buffer.data());

    auto restored = dtl::serializer<SimpleAggregate>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.x, 42);
    EXPECT_DOUBLE_EQ(restored.y, 3.14);
}

TEST(AggregateSerializerTest, MixedAggregateRoundtrip) {
    MixedAggregate original{
        99,
        "hello world",
        {1.0, 2.0, 3.0, 4.5}
    };

    auto sz = dtl::serializer<MixedAggregate>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<MixedAggregate>::serialize(original, buffer.data());

    auto restored = dtl::serializer<MixedAggregate>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.id, 99);
    EXPECT_EQ(restored.name, "hello world");
    ASSERT_EQ(restored.data.size(), 4u);
    EXPECT_DOUBLE_EQ(restored.data[0], 1.0);
    EXPECT_DOUBLE_EQ(restored.data[3], 4.5);
}

TEST(AggregateSerializerTest, SingleFieldRoundtrip) {
    SingleField original{-7};

    auto sz = dtl::serializer<SingleField>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<SingleField>::serialize(original, buffer.data());

    auto restored = dtl::serializer<SingleField>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.value, -7);
}

TEST(AggregateSerializerTest, EmptyStringAndVector) {
    MixedAggregate original{0, "", {}};

    auto sz = dtl::serializer<MixedAggregate>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<MixedAggregate>::serialize(original, buffer.data());

    auto restored = dtl::serializer<MixedAggregate>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.id, 0);
    EXPECT_TRUE(restored.name.empty());
    EXPECT_TRUE(restored.data.empty());
}

TEST(AggregateSerializerTest, NestedAggregateRoundtrip) {
    OuterAggregate original{
        10,
        {42, "nested_label"}
    };

    auto sz = dtl::serializer<OuterAggregate>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<OuterAggregate>::serialize(original, buffer.data());

    auto restored = dtl::serializer<OuterAggregate>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.id, 10);
    EXPECT_EQ(restored.inner.tag, 42);
    EXPECT_EQ(restored.inner.label, "nested_label");
}

// ============================================================================
// Trait Tests
// ============================================================================

TEST(AggregateSerializerTest, HasSerializer) {
    EXPECT_TRUE(dtl::has_serializer_v<SimpleAggregate>);
    EXPECT_TRUE(dtl::has_serializer_v<MixedAggregate>);
    EXPECT_TRUE(dtl::has_serializer_v<SingleField>);
}

TEST(AggregateSerializerTest, IsNotTrivial) {
    EXPECT_FALSE(dtl::serializer<MixedAggregate>::is_trivial());
    EXPECT_FALSE(dtl::serializer<SimpleAggregate>::is_trivial());
}

TEST(AggregateSerializerTest, SatisfiesSerializableConcept) {
    static_assert(dtl::Serializable<SimpleAggregate>);
    static_assert(dtl::Serializable<MixedAggregate>);
    static_assert(dtl::Serializable<SingleField>);
    static_assert(dtl::Serializable<OuterAggregate>);
    SUCCEED();
}

// ============================================================================
// Large Data Roundtrip
// ============================================================================

TEST(AggregateSerializerTest, LargeVectorField) {
    MixedAggregate original;
    original.id = 123;
    original.name = "large_test";
    original.data.resize(10000);
    for (size_t i = 0; i < original.data.size(); ++i) {
        original.data[i] = static_cast<double>(i) * 0.1;
    }

    auto sz = dtl::serializer<MixedAggregate>::serialized_size(original);
    std::vector<std::byte> buffer(sz);
    dtl::serializer<MixedAggregate>::serialize(original, buffer.data());

    auto restored = dtl::serializer<MixedAggregate>::deserialize(buffer.data(), sz);
    EXPECT_EQ(restored.id, 123);
    EXPECT_EQ(restored.name, "large_test");
    ASSERT_EQ(restored.data.size(), 10000u);
    EXPECT_DOUBLE_EQ(restored.data[0], 0.0);
    EXPECT_DOUBLE_EQ(restored.data[9999], 999.9);
}

}  // namespace dtl::test
