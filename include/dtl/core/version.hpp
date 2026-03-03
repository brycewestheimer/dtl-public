// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file version.hpp
/// @brief Version information and compile-time version utilities
/// @since 0.1.0

#pragma once

#include <dtl/generated/version_config.hpp>

#include <cstdint>
#include <string_view>

namespace dtl {

/// @brief Compile-time version information
/// @details Provides structured access to DTL version information at compile time.
struct version {
    /// @brief Major version number (incompatible API changes)
    static constexpr std::uint32_t major = DTL_VERSION_MAJOR;

    /// @brief Minor version number (backwards-compatible additions)
    static constexpr std::uint32_t minor = DTL_VERSION_MINOR;

    /// @brief Patch version number (backwards-compatible fixes)
    static constexpr std::uint32_t patch = DTL_VERSION_PATCH;

    /// @brief Encoded version as single integer for comparisons
    /// @details Format: MAJOR * 10000 + MINOR * 100 + PATCH
    static constexpr std::uint32_t encoded = DTL_VERSION;

    /// @brief Version string (e.g., "0.1.0")
    static constexpr std::string_view string = DTL_VERSION_STRING;

    /// @brief Prerelease suffix (empty for stable releases)
    static constexpr std::string_view prerelease = DTL_VERSION_PRERELEASE;

    /// @brief Full version string including prerelease
    static constexpr std::string_view full = DTL_VERSION_FULL;

    /// @brief Check if this version is at least the specified version
    /// @param req_major Required major version
    /// @param req_minor Required minor version (default: 0)
    /// @param req_patch Required patch version (default: 0)
    /// @return true if current version >= required version
    static constexpr bool at_least(std::uint32_t req_major,
                                   std::uint32_t req_minor = 0,
                                   std::uint32_t req_patch = 0) noexcept {
        const std::uint32_t required = req_major * 10000 + req_minor * 100 + req_patch;
        return encoded >= required;
    }

    /// @brief Check if this is a prerelease version
    /// @return true if prerelease suffix is non-empty
    static constexpr bool is_prerelease() noexcept {
        return !prerelease.empty();
    }

    /// @brief Check if this is a stable release
    /// @return true if not a prerelease
    static constexpr bool is_stable() noexcept {
        return prerelease.empty();
    }
};

/// @brief Get runtime version string
/// @return The DTL version string
/// @note Primarily useful for logging and diagnostics
[[nodiscard]] inline constexpr std::string_view get_version_string() noexcept {
    return version::full;
}

/// @brief Get encoded version number
/// @return The DTL version as a single integer
[[nodiscard]] inline constexpr std::uint32_t get_version() noexcept {
    return version::encoded;
}

}  // namespace dtl
