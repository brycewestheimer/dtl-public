// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file backend_discovery.hpp
/// @brief Runtime backend discovery and capability querying
/// @details Provides runtime introspection of available DTL backends and their
///          capabilities. Combines compile-time backend trait data with runtime
///          availability information from the runtime registry.
/// @since 0.1.0

#pragma once

#include <dtl/runtime/detail/runtime_export.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dtl::runtime {

// =============================================================================
// Backend Capability Bitmask
// =============================================================================

/// @brief Bitmask flags for backend capabilities
/// @details Mirrors the boolean fields in backend_traits<T> as a runtime
///          bitmask, enabling efficient capability querying without templates.
enum class backend_capability : uint32_t {
    none              = 0,
    point_to_point    = 1u << 0,
    collectives       = 1u << 1,
    rma               = 1u << 2,
    gpu_aware         = 1u << 3,
    async_operations  = 1u << 4,
    thread_multiple   = 1u << 5,
    rdma              = 1u << 6,
    device_execution  = 1u << 7,
    memory_management = 1u << 8,
};

/// @brief Runtime maturity level for a backend implementation
enum class backend_maturity : uint8_t {
    stub = 0,
    partial = 1,
    production = 2,
};

/// @brief Per-feature truth level for a backend capability
enum class capability_level : uint8_t {
    unavailable = 0,
    compiled = 1,
    runtime_available = 2,
    functional = 3,
};

/// @brief Per-capability truth descriptor
struct capability_descriptor {
    backend_capability capability{backend_capability::none};
    capability_level level{capability_level::unavailable};
};

/// @brief Bitwise OR for capability flags
[[nodiscard]] inline constexpr backend_capability operator|(
    backend_capability lhs, backend_capability rhs) noexcept {
    return static_cast<backend_capability>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

/// @brief Bitwise AND for capability flags
[[nodiscard]] inline constexpr backend_capability operator&(
    backend_capability lhs, backend_capability rhs) noexcept {
    return static_cast<backend_capability>(
        static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

/// @brief Check if a capability set contains a specific capability
/// @param set The capability set to check
/// @param cap The capability to look for
/// @return true if set contains cap
[[nodiscard]] inline constexpr bool has_capability(
    backend_capability set, backend_capability cap) noexcept {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(cap)) != 0;
}

// =============================================================================
// Backend Descriptor
// =============================================================================

/// @brief Runtime descriptor for a DTL backend
/// @details Combines compile-time trait information with runtime availability
///          from the runtime registry.
struct backend_descriptor {
    std::string name;                    ///< Backend name (e.g., "MPI", "CUDA")
    std::string version;                 ///< Backend version string (if known)
    backend_capability capabilities{};   ///< Bitmask of supported capabilities
    bool available{false};               ///< Runtime availability (registry says ready)
    bool compiled{false};                ///< Compile-time availability (DTL_ENABLE_* was set)

    // Extended truth model (Phase 07)
    backend_maturity maturity{backend_maturity::stub};
    backend_capability compiled_capabilities{};
    backend_capability runtime_capabilities{};
    backend_capability functional_capabilities{};
    std::vector<capability_descriptor> capability_levels;
};

// =============================================================================
// Discovery API
// =============================================================================

/// @brief Query all known backends with their capabilities and availability
/// @return Vector of backend descriptors for all known backends
DTL_RUNTIME_API std::vector<backend_descriptor> available_backends();

/// @brief Query a specific backend by name
/// @param name Backend name (case-insensitive: "mpi", "cuda", "hip", "nccl", "shmem")
/// @return Descriptor for the named backend (available=false, compiled=false if unknown)
DTL_RUNTIME_API backend_descriptor query_backend(std::string_view name);

/// @brief Check if any GPU-capable backend is available at runtime
/// @return true if CUDA, HIP, or NCCL is available
DTL_RUNTIME_API bool has_any_gpu_backend() noexcept;

/// @brief Check if any communication-capable backend is available at runtime
/// @return true if MPI, NCCL, or SHMEM is available
DTL_RUNTIME_API bool has_any_comm_backend() noexcept;

}  // namespace dtl::runtime
