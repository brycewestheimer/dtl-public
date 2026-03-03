// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_boost_serialization_adapter.cpp
/// @brief Unit tests for Boost.Serialization adapter
/// @details Tests the integration between Boost.Serialization and dtl::serializer
///          when Boost is available. Tests are conditionally compiled based on
///          DTL_HAS_BOOST_SERIALIZATION.

#include <dtl/serialization/adapters/boost_serialization.hpp>
#include <dtl/serialization/serializer.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Tests that always run (detection and trait tests)
// =============================================================================

TEST(BoostSerializationAdapterTest, HasBoostSerializationMacroIsDefined) {
    // DTL_HAS_BOOST_SERIALIZATION should always be defined (0 or 1)
#if defined(DTL_HAS_BOOST_SERIALIZATION)
    SUCCEED() << "DTL_HAS_BOOST_SERIALIZATION is defined with value " << DTL_HAS_BOOST_SERIALIZATION;
#else
    FAIL() << "DTL_HAS_BOOST_SERIALIZATION should always be defined";
#endif
}

#if DTL_HAS_BOOST_SERIALIZATION

// =============================================================================
// Test Types with Boost Serialization
// =============================================================================

/// @brief Simple test struct with Boost serialize method
struct BoostSimpleData {
    int id;
    double value;
    
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & id;
        ar & value;
    }
    
    bool operator==(const BoostSimpleData& other) const {
        return id == other.id && value == other.value;
    }
};

/// @brief Struct with nested types
struct BoostNestedData {
    std::string name;
    std::vector<int> values;
    BoostSimpleData inner;
    
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & name;
        ar & values;
        ar & inner;
    }
    
    bool operator==(const BoostNestedData& other) const {
        return name == other.name && values == other.values && inner == other.inner;
    }
};

/// @brief Struct with separate save/load members
struct BoostSaveLoadData {
    int x;
    int y;
    
    template <class Archive>
    void save(Archive& ar, const unsigned int /*version*/) const {
        ar & x;
        ar & y;
    }
    
    template <class Archive>
    void load(Archive& ar, const unsigned int /*version*/) {
        ar & x;
        ar & y;
    }
    
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    
    bool operator==(const BoostSaveLoadData& other) const {
        return x == other.x && y == other.y;
    }
};

// Opt-in types to use Boost adapter
template <>
struct use_boost_adapter<BoostSimpleData> : std::true_type {};

template <>
struct use_boost_adapter<BoostNestedData> : std::true_type {};

template <>
struct use_boost_adapter<BoostSaveLoadData> : std::true_type {};

// =============================================================================
// Trait Detection Tests
// =============================================================================

TEST(BoostSerializationAdapterTest, DetectsBoostSerializableTypes) {
    static_assert(detail::has_boost_serialization_v<BoostSimpleData>);
    static_assert(detail::has_boost_serialization_v<BoostNestedData>);
    static_assert(detail::has_boost_serialization_v<BoostSaveLoadData>);
}

TEST(BoostSerializationAdapterTest, UseAdapterTraitWorks) {
    static_assert(use_boost_adapter_v<BoostSimpleData>);
    static_assert(use_boost_adapter_v<BoostNestedData>);
    static_assert(use_boost_adapter_v<BoostSaveLoadData>);
    
    // Non-opted-in types should return false
    static_assert(!use_boost_adapter_v<int>);
    static_assert(!use_boost_adapter_v<std::vector<int>>);
}

TEST(BoostSerializationAdapterTest, IsBoostSerializableWorks) {
    static_assert(is_boost_serializable_v<BoostSimpleData>);
    static_assert(is_boost_serializable_v<BoostNestedData>);
}

// =============================================================================
// Serializer Integration Tests
// =============================================================================

TEST(BoostSerializationAdapterTest, HasSerializerForOptedInType) {
    static_assert(has_serializer_v<BoostSimpleData>);
    static_assert(Serializable<BoostSimpleData>);
}

TEST(BoostSerializationAdapterTest, RoundtripSimpleData) {
    BoostSimpleData original{42, 3.14159};
    
    // Get serialized size
    size_type size = serializer<BoostSimpleData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    size_type written = serializer<BoostSimpleData>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);
    
    // Deserialize
    BoostSimpleData restored = serializer<BoostSimpleData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(BoostSerializationAdapterTest, RoundtripNestedData) {
    BoostNestedData original{
        "test_name",
        {1, 2, 3, 4, 5},
        {100, 2.71828}
    };
    
    // Get serialized size
    size_type size = serializer<BoostNestedData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    size_type written = serializer<BoostNestedData>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);
    
    // Deserialize
    BoostNestedData restored = serializer<BoostNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(BoostSerializationAdapterTest, RoundtripSaveLoadData) {
    BoostSaveLoadData original{10, 20};
    
    // Get serialized size
    size_type size = serializer<BoostSaveLoadData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    serializer<BoostSaveLoadData>::serialize(original, buffer.data());
    
    // Deserialize
    BoostSaveLoadData restored = serializer<BoostSaveLoadData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST(BoostSerializationAdapterTest, SerializeToVectorUtility) {
    BoostSimpleData original{99, 1.5};
    
    std::vector<std::byte> buffer = boost_serialize_to_vector(original);
    ASSERT_GT(buffer.size(), 0u);
    
    BoostSimpleData restored = boost_deserialize_from_buffer<BoostSimpleData>(
        buffer.data(), static_cast<size_type>(buffer.size()));
    EXPECT_EQ(restored, original);
}

TEST(BoostSerializationAdapterTest, EmptyVectorRoundtrip) {
    BoostNestedData original{"empty", {}, {0, 0.0}};
    
    size_type size = serializer<BoostNestedData>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    serializer<BoostNestedData>::serialize(original, buffer.data());
    
    BoostNestedData restored = serializer<BoostNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(BoostSerializationAdapterTest, LargeDataRoundtrip) {
    BoostNestedData original{
        std::string(1000, 'x'),
        std::vector<int>(1000, 42),
        {12345, 98765.4321}
    };
    
    size_type size = serializer<BoostNestedData>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    serializer<BoostNestedData>::serialize(original, buffer.data());
    
    BoostNestedData restored = serializer<BoostNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(BoostSerializationAdapterTest, IsTrivialReturnsFalse) {
    EXPECT_FALSE(serializer<BoostSimpleData>::is_trivial());
}

#else  // !DTL_HAS_BOOST_SERIALIZATION

// =============================================================================
// Tests when Boost.Serialization is NOT available
// =============================================================================

TEST(BoostSerializationAdapterTest, TraitsInactiveWhenUnavailable) {
    static_assert(!use_boost_adapter_v<int>);
    static_assert(!is_boost_serializable_v<int>);
}

TEST(BoostSerializationAdapterTest, BoostNotAvailableMessage) {
    GTEST_SKIP() << "Boost.Serialization is not available in this build "
                 << "(DTL_HAS_BOOST_SERIALIZATION=0). "
                 << "Install libboost-serialization-dev and rebuild with "
                 << "-DDTL_ENABLE_BOOST_SERIALIZATION=ON to enable tests.";
}

#endif  // DTL_HAS_BOOST_SERIALIZATION

}  // namespace dtl::test
