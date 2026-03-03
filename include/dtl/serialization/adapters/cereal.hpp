// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cereal.hpp
/// @brief Cereal serialization library adapter for dtl::serializer
/// @details Provides integration with Cereal's serialization framework,
///          allowing types with Cereal serialization support to work with
///          DTL's serialization and RPC infrastructure.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/serialization/serializer.hpp>

// =============================================================================
// Cereal Detection and Configuration
// =============================================================================

// Check if Cereal is available at compile time
#if defined(DTL_HAS_CEREAL) && DTL_HAS_CEREAL
    // Already defined via CMake
#elif __has_include(<cereal/cereal.hpp>)
    #ifndef DTL_HAS_CEREAL
        #define DTL_HAS_CEREAL 1
    #endif
#else
    #ifndef DTL_HAS_CEREAL
        #define DTL_HAS_CEREAL 0
    #endif
#endif

#if DTL_HAS_CEREAL

// Include Cereal headers
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/details/traits.hpp>

#include <sstream>
#include <type_traits>

namespace dtl {

// =============================================================================
// Cereal Serialization Trait Detection
// =============================================================================

namespace detail {

/// @brief Detect if type T has cereal serialize() member function
template <typename T, typename Archive, typename = void>
struct has_cereal_serialize_member : std::false_type {};

template <typename T, typename Archive>
struct has_cereal_serialize_member<T, Archive,
    std::void_t<decltype(std::declval<T&>().serialize(std::declval<Archive&>()))>>
    : std::true_type {};

/// @brief Detect if type T has cereal save()/load() member functions
template <typename T, typename Archive, typename = void>
struct has_cereal_save_load_members : std::false_type {};

template <typename T, typename Archive>
struct has_cereal_save_load_members<T, Archive,
    std::void_t<decltype(std::declval<const T&>().save(std::declval<Archive&>())),
                decltype(std::declval<T&>().load(std::declval<Archive&>()))>>
    : std::true_type {};

/// @brief Detect if type T has non-intrusive cereal serialization
template <typename T, typename Archive, typename = void>
struct has_cereal_external_serialize : std::false_type {};

template <typename T, typename Archive>
struct has_cereal_external_serialize<T, Archive,
    std::void_t<decltype(cereal::serialize(std::declval<Archive&>(), std::declval<T&>()))>>
    : std::true_type {};

/// @brief Detect if type T has non-intrusive cereal save/load functions
template <typename T, typename Archive, typename = void>
struct has_cereal_external_save_load : std::false_type {};

template <typename T, typename Archive>
struct has_cereal_external_save_load<T, Archive,
    std::void_t<decltype(cereal::save(std::declval<Archive&>(), std::declval<const T&>())),
                decltype(cereal::load(std::declval<Archive&>(), std::declval<T&>()))>>
    : std::true_type {};

/// @brief Check if type has any form of cereal serialization support
/// Uses cereal's binary archive as the detection archive type
template <typename T>
struct has_cereal_serialization {
    using InputArchive = cereal::BinaryInputArchive;
    using OutputArchive = cereal::BinaryOutputArchive;

    static constexpr bool value =
        has_cereal_serialize_member<T, OutputArchive>::value ||
        has_cereal_save_load_members<T, OutputArchive>::value ||
        has_cereal_external_serialize<T, OutputArchive>::value ||
        has_cereal_external_save_load<T, OutputArchive>::value ||
        cereal::traits::is_input_serializable<T, InputArchive>::value;
};

template <typename T>
inline constexpr bool has_cereal_serialization_v = has_cereal_serialization<T>::value;

}  // namespace detail

// =============================================================================
// Cereal Adapter Configuration
// =============================================================================

/// @brief Marker trait to opt a type into Cereal-based DTL serialization
/// @tparam T The type to use Cereal serialization for
/// @details Specialize this trait to std::true_type for types that should
///          use Cereal for DTL serialization. This allows explicit opt-in
///          rather than automatic detection.
///
/// @par Example usage:
/// @code
/// struct MyData {
///     int value;
///     std::string name;
///
///     template <class Archive>
///     void serialize(Archive& ar) {
///         ar(value, name);
///     }
/// };
///
/// template <>
/// struct dtl::use_cereal_adapter<MyData> : std::true_type {};
/// @endcode
template <typename T>
struct use_cereal_adapter : std::false_type {};

/// @brief Helper variable for use_cereal_adapter
template <typename T>
inline constexpr bool use_cereal_adapter_v = use_cereal_adapter<T>::value;

// =============================================================================
// Cereal Archive Configuration
// =============================================================================

/// @brief Configuration options for Cereal serialization
struct cereal_config {
    /// @brief Whether to use binary format (vs portable binary)
    bool use_binary = true;
    
    /// @brief Buffer initial capacity hint
    size_type initial_capacity = 1024;
};

/// @brief Get default Cereal configuration
[[nodiscard]] inline cereal_config default_cereal_config() noexcept {
    return cereal_config{};
}

// =============================================================================
// Cereal Serializer Adapter Implementation
// =============================================================================

/// @brief Serializer specialization for types using Cereal
/// @tparam T A type with Cereal serialization support and use_cereal_adapter<T> = true
/// @details This adapter bridges Cereal's archive-based serialization to
///          DTL's buffer-based serializer interface.
template <typename T>
    requires use_cereal_adapter_v<T> && 
             detail::has_cereal_serialization_v<T> &&
             (!is_trivially_serializable_v<T>)
struct serializer<T, void> {
    /// @brief Get serialized size by performing a trial serialization
    /// @param value The value to measure
    /// @return Size in bytes when serialized
    /// @note This performs a full serialization to determine size.
    ///       For performance-critical code with fixed-size types, consider
    ///       implementing a custom serializer instead.
    [[nodiscard]] static size_type serialized_size(const T& value) {
        std::ostringstream oss(std::ios::binary);
        {
            cereal::BinaryOutputArchive archive(oss);
            archive(value);
        }
        return static_cast<size_type>(oss.str().size());
    }

    /// @brief Serialize value to buffer using Cereal
    /// @param value The value to serialize
    /// @param buffer Destination buffer (must have sufficient space)
    /// @return Number of bytes written
    static size_type serialize(const T& value, std::byte* buffer) {
        std::ostringstream oss(std::ios::binary);
        {
            cereal::BinaryOutputArchive archive(oss);
            archive(value);
        }
        const std::string data = oss.str();
        std::memcpy(buffer, data.data(), data.size());
        return static_cast<size_type>(data.size());
    }

    /// @brief Deserialize value from buffer using Cereal
    /// @param buffer Source buffer
    /// @param size Buffer size in bytes
    /// @return The deserialized value
    [[nodiscard]] static T deserialize(const std::byte* buffer, size_type size) {
        std::string data(reinterpret_cast<const char*>(buffer), size);
        std::istringstream iss(data, std::ios::binary);
        
        T value;
        {
            cereal::BinaryInputArchive archive(iss);
            archive(value);
        }
        return value;
    }

    /// @brief Cereal-based serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

// =============================================================================
// Cereal Utility Functions
// =============================================================================

/// @brief Serialize a value to a byte vector using Cereal
/// @tparam T Type with Cereal serialization support
/// @param value The value to serialize
/// @return Vector containing serialized bytes
template <typename T>
    requires detail::has_cereal_serialization_v<T>
[[nodiscard]] inline std::vector<std::byte> cereal_serialize_to_vector(const T& value) {
    std::ostringstream oss(std::ios::binary);
    {
        cereal::BinaryOutputArchive archive(oss);
        archive(value);
    }
    const std::string data = oss.str();
    std::vector<std::byte> result(data.size());
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

/// @brief Deserialize a value from a byte span using Cereal
/// @tparam T Type with Cereal serialization support
/// @param buffer Source buffer
/// @param size Buffer size
/// @return The deserialized value
template <typename T>
    requires detail::has_cereal_serialization_v<T>
[[nodiscard]] inline T cereal_deserialize_from_buffer(const std::byte* buffer, size_type size) {
    std::string data(reinterpret_cast<const char*>(buffer), size);
    std::istringstream iss(data, std::ios::binary);
    
    T value;
    {
        cereal::BinaryInputArchive archive(iss);
        archive(value);
    }
    return value;
}

// =============================================================================
// Cereal Traits for DTL Integration
// =============================================================================

/// @brief Check if a type can use Cereal serialization with DTL
/// @tparam T The type to check
template <typename T>
struct is_cereal_serializable 
    : std::bool_constant<detail::has_cereal_serialization_v<T>> {};

/// @brief Helper variable for is_cereal_serializable
template <typename T>
inline constexpr bool is_cereal_serializable_v = is_cereal_serializable<T>::value;

}  // namespace dtl

#else  // !DTL_HAS_CEREAL

// =============================================================================
// Stub Declarations When Cereal Is Not Available
// =============================================================================

namespace dtl {

/// @brief Marker trait (inactive when Cereal unavailable)
template <typename T>
struct use_cereal_adapter : std::false_type {};

template <typename T>
inline constexpr bool use_cereal_adapter_v = false;

/// @brief Check if Cereal serialization is available (always false)
template <typename T>
struct is_cereal_serializable : std::false_type {};

template <typename T>
inline constexpr bool is_cereal_serializable_v = false;

namespace detail {
template <typename T>
inline constexpr bool has_cereal_serialization_v = false;
}

}  // namespace dtl

#endif  // DTL_HAS_CEREAL
