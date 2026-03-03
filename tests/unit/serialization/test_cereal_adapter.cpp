// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cereal_adapter.cpp
/// @brief Unit tests for Cereal serialization adapter
/// @details Tests the integration between Cereal and dtl::serializer when Cereal is available.
///          Tests are conditionally compiled based on DTL_HAS_CEREAL.

#include <dtl/serialization/adapters/cereal.hpp>
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

TEST(CerealAdapterTest, HasCerealMacroIsDefined) {
    // DTL_HAS_CEREAL should always be defined (0 or 1)
#if defined(DTL_HAS_CEREAL)
    SUCCEED() << "DTL_HAS_CEREAL is defined with value " << DTL_HAS_CEREAL;
#else
    FAIL() << "DTL_HAS_CEREAL should always be defined";
#endif
}

#if DTL_HAS_CEREAL

// =============================================================================
// Test Types with Cereal Serialization
// =============================================================================

/// @brief Simple test struct with Cereal serialize method
struct CerealSimpleData {
    int id;
    double value;
    
    template <class Archive>
    void serialize(Archive& ar) {
        ar(id, value);
    }
    
    bool operator==(const CerealSimpleData& other) const {
        return id == other.id && value == other.value;
    }
};

/// @brief Struct with nested types
struct CerealNestedData {
    std::string name;
    std::vector<int> values;
    CerealSimpleData inner;
    
    template <class Archive>
    void serialize(Archive& ar) {
        ar(name, values, inner);
    }
    
    bool operator==(const CerealNestedData& other) const {
        return name == other.name && values == other.values && inner == other.inner;
    }
};

/// @brief Struct with save/load members
struct CerealSaveLoadData {
    int x;
    int y;
    
    template <class Archive>
    void save(Archive& ar) const {
        ar(x, y);
    }
    
    template <class Archive>
    void load(Archive& ar) {
        ar(x, y);
    }
    
    bool operator==(const CerealSaveLoadData& other) const {
        return x == other.x && y == other.y;
    }
};

// Opt-in types to use Cereal adapter
template <>
struct use_cereal_adapter<CerealSimpleData> : std::true_type {};

template <>
struct use_cereal_adapter<CerealNestedData> : std::true_type {};

template <>
struct use_cereal_adapter<CerealSaveLoadData> : std::true_type {};

// =============================================================================
// Trait Detection Tests
// =============================================================================

TEST(CerealAdapterTest, DetectsCerealSerializableTypes) {
    static_assert(detail::has_cereal_serialization_v<CerealSimpleData>);
    static_assert(detail::has_cereal_serialization_v<CerealNestedData>);
    static_assert(detail::has_cereal_serialization_v<CerealSaveLoadData>);
}

TEST(CerealAdapterTest, UseAdapterTraitWorks) {
    static_assert(use_cereal_adapter_v<CerealSimpleData>);
    static_assert(use_cereal_adapter_v<CerealNestedData>);
    static_assert(use_cereal_adapter_v<CerealSaveLoadData>);
    
    // Non-opted-in types should return false
    static_assert(!use_cereal_adapter_v<int>);
    static_assert(!use_cereal_adapter_v<std::vector<int>>);
}

TEST(CerealAdapterTest, IsCerealSerializableWorks) {
    static_assert(is_cereal_serializable_v<CerealSimpleData>);
    static_assert(is_cereal_serializable_v<CerealNestedData>);
}

// =============================================================================
// Serializer Integration Tests
// =============================================================================

TEST(CerealAdapterTest, HasSerializerForOptedInType) {
    static_assert(has_serializer_v<CerealSimpleData>);
    static_assert(Serializable<CerealSimpleData>);
}

TEST(CerealAdapterTest, RoundtripSimpleData) {
    CerealSimpleData original{42, 3.14159};
    
    // Get serialized size
    size_type size = serializer<CerealSimpleData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    size_type written = serializer<CerealSimpleData>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);
    
    // Deserialize
    CerealSimpleData restored = serializer<CerealSimpleData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(CerealAdapterTest, RoundtripNestedData) {
    CerealNestedData original{
        "test_name",
        {1, 2, 3, 4, 5},
        {100, 2.71828}
    };
    
    // Get serialized size
    size_type size = serializer<CerealNestedData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    size_type written = serializer<CerealNestedData>::serialize(original, buffer.data());
    EXPECT_EQ(written, size);
    
    // Deserialize
    CerealNestedData restored = serializer<CerealNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(CerealAdapterTest, RoundtripSaveLoadData) {
    CerealSaveLoadData original{10, 20};
    
    // Get serialized size
    size_type size = serializer<CerealSaveLoadData>::serialized_size(original);
    ASSERT_GT(size, 0u);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    serializer<CerealSaveLoadData>::serialize(original, buffer.data());
    
    // Deserialize
    CerealSaveLoadData restored = serializer<CerealSaveLoadData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST(CerealAdapterTest, SerializeToVectorUtility) {
    CerealSimpleData original{99, 1.5};
    
    std::vector<std::byte> buffer = cereal_serialize_to_vector(original);
    ASSERT_GT(buffer.size(), 0u);
    
    CerealSimpleData restored = cereal_deserialize_from_buffer<CerealSimpleData>(
        buffer.data(), static_cast<size_type>(buffer.size()));
    EXPECT_EQ(restored, original);
}

TEST(CerealAdapterTest, EmptyVectorRoundtrip) {
    CerealNestedData original{"empty", {}, {0, 0.0}};
    
    size_type size = serializer<CerealNestedData>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    serializer<CerealNestedData>::serialize(original, buffer.data());
    
    CerealNestedData restored = serializer<CerealNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(CerealAdapterTest, LargeDataRoundtrip) {
    CerealNestedData original{
        std::string(1000, 'x'),
        std::vector<int>(1000, 42),
        {12345, 98765.4321}
    };
    
    size_type size = serializer<CerealNestedData>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    serializer<CerealNestedData>::serialize(original, buffer.data());
    
    CerealNestedData restored = serializer<CerealNestedData>::deserialize(buffer.data(), buffer.size());
    EXPECT_EQ(restored, original);
}

TEST(CerealAdapterTest, IsTrivialReturnsFalse) {
    EXPECT_FALSE(serializer<CerealSimpleData>::is_trivial());
}

#else  // !DTL_HAS_CEREAL

// =============================================================================
// Tests when Cereal is NOT available
// =============================================================================

TEST(CerealAdapterTest, TraitsInactiveWhenUnavailable) {
    static_assert(!use_cereal_adapter_v<int>);
    static_assert(!is_cereal_serializable_v<int>);
}

TEST(CerealAdapterTest, CerealNotAvailableMessage) {
    GTEST_SKIP() << "Cereal is not available in this build (DTL_HAS_CEREAL=0). "
                 << "Install Cereal and rebuild with -DDTL_ENABLE_CEREAL=ON to enable tests.";
}

#endif  // DTL_HAS_CEREAL

}  // namespace dtl::test
