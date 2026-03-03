// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file boost_serialization.hpp
/// @brief Boost.Serialization library adapter for dtl::serializer
/// @details Provides integration with Boost.Serialization framework,
///          allowing types with Boost serialization support to work with
///          DTL's serialization and RPC infrastructure.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/serialization/serializer.hpp>

// =============================================================================
// Boost.Serialization Detection and Configuration
// =============================================================================

// Check if Boost.Serialization is available at compile time
#if defined(DTL_HAS_BOOST_SERIALIZATION) && DTL_HAS_BOOST_SERIALIZATION
    // Already defined via CMake
#elif __has_include(<boost/serialization/serialization.hpp>)
    #ifndef DTL_HAS_BOOST_SERIALIZATION
        #define DTL_HAS_BOOST_SERIALIZATION 1
    #endif
#else
    #ifndef DTL_HAS_BOOST_SERIALIZATION
        #define DTL_HAS_BOOST_SERIALIZATION 0
    #endif
#endif

#if DTL_HAS_BOOST_SERIALIZATION

// Include Boost.Serialization headers
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <sstream>
#include <type_traits>

namespace dtl {

// =============================================================================
// Boost.Serialization Trait Detection
// =============================================================================

namespace detail {

/// @brief Detect if type T has intrusive serialize() member function
template <typename T, typename Archive, typename = void>
struct has_boost_serialize_member : std::false_type {};

template <typename T, typename Archive>
struct has_boost_serialize_member<T, Archive,
    std::void_t<decltype(std::declval<T&>().serialize(
        std::declval<Archive&>(), 
        std::declval<unsigned int>()))>>
    : std::true_type {};

/// @brief Detect if type T has intrusive save()/load() member functions
template <typename T, typename Archive, typename = void>
struct has_boost_save_load_members : std::false_type {};

template <typename T, typename Archive>
struct has_boost_save_load_members<T, Archive,
    std::void_t<decltype(std::declval<const T&>().save(
                    std::declval<Archive&>(), 
                    std::declval<unsigned int>())),
                decltype(std::declval<T&>().load(
                    std::declval<Archive&>(), 
                    std::declval<unsigned int>()))>>
    : std::true_type {};

/// @brief Detect if type T has non-intrusive serialize function
template <typename T, typename Archive, typename = void>
struct has_boost_external_serialize : std::false_type {};

template <typename T, typename Archive>
struct has_boost_external_serialize<T, Archive,
    std::void_t<decltype(serialize(
        std::declval<Archive&>(), 
        std::declval<T&>(), 
        std::declval<unsigned int>()))>>
    : std::true_type {};

/// @brief Detect if type T has non-intrusive save/load functions
template <typename T, typename Archive, typename = void>
struct has_boost_external_save_load : std::false_type {};

template <typename T, typename Archive>
struct has_boost_external_save_load<T, Archive,
    std::void_t<decltype(save(
                    std::declval<Archive&>(), 
                    std::declval<const T&>(), 
                    std::declval<unsigned int>())),
                decltype(load(
                    std::declval<Archive&>(), 
                    std::declval<T&>(), 
                    std::declval<unsigned int>()))>>
    : std::true_type {};

/// @brief Check if type has any form of Boost serialization support
template <typename T>
struct has_boost_serialization {
    using InputArchive = boost::archive::binary_iarchive;
    using OutputArchive = boost::archive::binary_oarchive;

    static constexpr bool value =
        has_boost_serialize_member<T, OutputArchive>::value ||
        has_boost_save_load_members<T, OutputArchive>::value ||
        has_boost_external_serialize<T, OutputArchive>::value ||
        has_boost_external_save_load<T, OutputArchive>::value;
};

template <typename T>
inline constexpr bool has_boost_serialization_v = has_boost_serialization<T>::value;

}  // namespace detail

// =============================================================================
// Boost Adapter Configuration
// =============================================================================

/// @brief Marker trait to opt a type into Boost.Serialization-based DTL serialization
/// @tparam T The type to use Boost serialization for
/// @details Specialize this trait to std::true_type for types that should
///          use Boost.Serialization for DTL serialization.
///
/// @par Example usage:
/// @code
/// struct MyData {
///     int value;
///     std::string name;
///
///     template <class Archive>
///     void serialize(Archive& ar, const unsigned int version) {
///         ar & value;
///         ar & name;
///     }
/// };
///
/// template <>
/// struct dtl::use_boost_adapter<MyData> : std::true_type {};
/// @endcode
template <typename T>
struct use_boost_adapter : std::false_type {};

/// @brief Helper variable for use_boost_adapter
template <typename T>
inline constexpr bool use_boost_adapter_v = use_boost_adapter<T>::value;

// =============================================================================
// Boost Archive Configuration
// =============================================================================

/// @brief Configuration options for Boost serialization
struct boost_serialization_config {
    /// @brief Serialization flags (e.g., no_header)
    unsigned int flags = 0;
    
    /// @brief Buffer initial capacity hint
    size_type initial_capacity = 1024;
};

/// @brief Get default Boost serialization configuration
[[nodiscard]] inline boost_serialization_config default_boost_config() noexcept {
    return boost_serialization_config{};
}

// =============================================================================
// Boost Serializer Adapter Implementation
// =============================================================================

/// @brief Serializer specialization for types using Boost.Serialization
/// @tparam T A type with Boost serialization support and use_boost_adapter<T> = true
/// @details This adapter bridges Boost's archive-based serialization to
///          DTL's buffer-based serializer interface.
template <typename T>
    requires use_boost_adapter_v<T> && 
             detail::has_boost_serialization_v<T> &&
             (!is_trivially_serializable_v<T>)
struct serializer<T, void> {
    /// @brief Get serialized size by performing a trial serialization
    /// @param value The value to measure
    /// @return Size in bytes when serialized
    /// @note This performs a full serialization to determine size.
    [[nodiscard]] static size_type serialized_size(const T& value) {
        std::ostringstream oss(std::ios::binary);
        {
            boost::archive::binary_oarchive archive(oss);
            archive << value;
        }
        return oss.str().size();
    }

    /// @brief Serialize value to buffer using Boost.Serialization
    /// @param value The value to serialize
    /// @param buffer Destination buffer (must have sufficient space)
    /// @return Number of bytes written
    static size_type serialize(const T& value, std::byte* buffer) {
        std::ostringstream oss(std::ios::binary);
        {
            boost::archive::binary_oarchive archive(oss);
            archive << value;
        }
        const std::string data = oss.str();
        std::memcpy(buffer, data.data(), data.size());
        return data.size();
    }

    /// @brief Deserialize value from buffer using Boost.Serialization
    /// @param buffer Source buffer
    /// @param size Buffer size in bytes
    /// @return The deserialized value
    [[nodiscard]] static T deserialize(const std::byte* buffer, size_type size) {
        std::string data(reinterpret_cast<const char*>(buffer), size);
        std::istringstream iss(data, std::ios::binary);
        
        T value;
        {
            boost::archive::binary_iarchive archive(iss);
            archive >> value;
        }
        return value;
    }

    /// @brief Boost-based serialization is not trivial
    [[nodiscard]] static constexpr bool is_trivial() noexcept {
        return false;
    }
};

// =============================================================================
// Boost.Serialization Utility Functions
// =============================================================================

/// @brief Serialize a value to a byte vector using Boost.Serialization
/// @tparam T Type with Boost serialization support
/// @param value The value to serialize
/// @return Vector containing serialized bytes
template <typename T>
    requires detail::has_boost_serialization_v<T>
[[nodiscard]] inline std::vector<std::byte> boost_serialize_to_vector(const T& value) {
    std::ostringstream oss(std::ios::binary);
    {
        boost::archive::binary_oarchive archive(oss);
        archive << value;
    }
    const std::string data = oss.str();
    std::vector<std::byte> result(data.size());
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

/// @brief Deserialize a value from a byte span using Boost.Serialization
/// @tparam T Type with Boost serialization support
/// @param buffer Source buffer
/// @param size Buffer size
/// @return The deserialized value
template <typename T>
    requires detail::has_boost_serialization_v<T>
[[nodiscard]] inline T boost_deserialize_from_buffer(const std::byte* buffer, size_type size) {
    std::string data(reinterpret_cast<const char*>(buffer), size);
    std::istringstream iss(data, std::ios::binary);
    
    T value;
    {
        boost::archive::binary_iarchive archive(iss);
        archive >> value;
    }
    return value;
}

// =============================================================================
// Boost Traits for DTL Integration
// =============================================================================

/// @brief Check if a type can use Boost serialization with DTL
/// @tparam T The type to check
template <typename T>
struct is_boost_serializable 
    : std::bool_constant<detail::has_boost_serialization_v<T>> {};

/// @brief Helper variable for is_boost_serializable
template <typename T>
inline constexpr bool is_boost_serializable_v = is_boost_serializable<T>::value;

}  // namespace dtl

#else  // !DTL_HAS_BOOST_SERIALIZATION

// =============================================================================
// Stub Declarations When Boost.Serialization Is Not Available
// =============================================================================

namespace dtl {

/// @brief Marker trait (inactive when Boost unavailable)
template <typename T>
struct use_boost_adapter : std::false_type {};

template <typename T>
inline constexpr bool use_boost_adapter_v = false;

/// @brief Check if Boost serialization is available (always false)
template <typename T>
struct is_boost_serializable : std::false_type {};

template <typename T>
inline constexpr bool is_boost_serializable_v = false;

namespace detail {
template <typename T>
inline constexpr bool has_boost_serialization_v = false;
}

}  // namespace dtl

#endif  // DTL_HAS_BOOST_SERIALIZATION
