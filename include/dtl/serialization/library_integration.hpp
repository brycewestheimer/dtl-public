// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file library_integration.hpp
/// @brief Optional integration with external serialization libraries
/// @details Provides adapters for Cereal, Boost.Serialization, and other
///          serialization frameworks.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/serialization/serializer.hpp>

// =============================================================================
// External Library Detection
// =============================================================================

// Detect Cereal
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

// Detect Boost.Serialization
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

// =============================================================================
// Include Adapter Headers (when libraries are available)
// =============================================================================

#if DTL_HAS_CEREAL
#include <dtl/serialization/adapters/cereal.hpp>
#endif

#if DTL_HAS_BOOST_SERIALIZATION
#include <dtl/serialization/adapters/boost_serialization.hpp>
#endif

namespace dtl {

// =============================================================================
// Legacy Aliases for Backwards Compatibility
// =============================================================================
// The actual adapter implementations are now in the adapter headers.
// These aliases maintain compatibility with code that used the old trait names.

#if DTL_HAS_CEREAL

/// @brief Legacy alias for use_cereal_adapter (backwards compatibility)
/// @deprecated Use use_cereal_adapter instead
template <typename T>
using use_cereal_serialization = use_cereal_adapter<T>;

/// @brief Legacy helper variable (backwards compatibility)
template <typename T>
inline constexpr bool use_cereal_serialization_v = use_cereal_adapter_v<T>;

#else

/// @brief Marker trait (inactive when Cereal unavailable)
template <typename T>
struct use_cereal_serialization : std::false_type {};

template <typename T>
inline constexpr bool use_cereal_serialization_v = false;

#endif  // DTL_HAS_CEREAL

#if DTL_HAS_BOOST_SERIALIZATION

/// @brief Legacy alias for use_boost_adapter (backwards compatibility)
/// @deprecated Use use_boost_adapter instead
template <typename T>
using use_boost_serialization = use_boost_adapter<T>;

/// @brief Legacy helper variable (backwards compatibility)
template <typename T>
inline constexpr bool use_boost_serialization_v = use_boost_adapter_v<T>;

#else

/// @brief Marker trait (inactive when Boost unavailable)
template <typename T>
struct use_boost_serialization : std::false_type {};

template <typename T>
inline constexpr bool use_boost_serialization_v = false;

#endif  // DTL_HAS_BOOST_SERIALIZATION

// =============================================================================
// Serialization Format Registry (Extensibility Point)
// =============================================================================

/// @brief Format identifier for serialization
enum class serialization_format {
    native,           ///< DTL native binary format
    cereal_binary,    ///< Cereal binary archive
    cereal_json,      ///< Cereal JSON archive
    boost_binary,     ///< Boost.Serialization binary
    boost_text        ///< Boost.Serialization text
};

/// @brief Check if a serialization format is available
/// @param format The format to check
/// @return true if the format is supported in this build
[[nodiscard]] inline bool is_format_available(serialization_format format) noexcept {
    switch (format) {
        case serialization_format::native:
            return true;
#if DTL_HAS_CEREAL
        case serialization_format::cereal_binary:
        case serialization_format::cereal_json:
            return true;
#endif
#if DTL_HAS_BOOST_SERIALIZATION
        case serialization_format::boost_binary:
        case serialization_format::boost_text:
            return true;
#endif
        default:
            return false;
    }
}

}  // namespace dtl
