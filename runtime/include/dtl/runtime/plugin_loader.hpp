// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file plugin_loader.hpp
/// @brief Plugin loading and registration
/// @details Provides infrastructure for dynamically loading DTL backend plugins
///          at runtime via dlopen (POSIX) or LoadLibrary (Windows).
///          Each plugin DSO must export a C function `dtl_plugin_register`
///          that fills a `dtl_plugin_info` struct with name, version, ABI
///          version, and optional init/fini callbacks.
/// @since 0.1.0

#pragma once

#include <dtl/runtime/detail/runtime_export.hpp>
#include <dtl/error/result.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dtl::runtime {

// =============================================================================
// ABI Version
// =============================================================================

/// @brief Plugin ABI version for compatibility checking
/// @details Plugins built against a different ABI version will be rejected.
inline constexpr uint32_t plugin_abi_version = 1;

/// @brief Well-known symbol name for plugin registration entry point
/// @details Each plugin DSO must export a C function with this name and the
///          signature: `int dtl_plugin_register(dtl_plugin_info*)`.
inline constexpr const char* plugin_register_symbol = "dtl_plugin_register";

// =============================================================================
// Plugin Info (C ABI struct filled by plugin)
// =============================================================================

/// @brief C-compatible struct that plugins fill during registration
/// @details Passed to the `dtl_plugin_register` entry point. The plugin
///          fills in its name, version, ABI version, and optional callbacks.
struct dtl_plugin_info {
    const char* name{nullptr};       ///< Plugin name (static string, not freed)
    const char* version{nullptr};    ///< Plugin version (static string, not freed)
    uint32_t abi_version{0};         ///< ABI version the plugin was built against
    int (*init)(){nullptr};          ///< Init callback (may be null), returns 0 on success
    void (*fini)(){nullptr};         ///< Fini callback (may be null)
};

/// @brief Type of the plugin registration entry point
using plugin_register_fn = int(*)(dtl_plugin_info*);

// =============================================================================
// Plugin Descriptor
// =============================================================================

/// @brief Describes a loaded plugin
struct plugin_descriptor {
    std::string name;             ///< Plugin name
    std::string version;          ///< Plugin version string
    uint32_t abi_version{0};      ///< ABI version the plugin was built against
    std::string path;             ///< Filesystem path to the plugin DSO

    /// @brief Plugin initialization function pointer
    /// @details Called after loading. Returns 0 on success.
    using init_fn = int(*)();

    /// @brief Plugin finalization function pointer
    /// @details Called before unloading.
    using fini_fn = void(*)();

    init_fn init{nullptr};        ///< Init function (may be null)
    fini_fn fini{nullptr};        ///< Fini function (may be null)
    void* dl_handle{nullptr};     ///< Opaque DSO handle (dlopen/LoadLibrary)
};

// =============================================================================
// Plugin Registry
// =============================================================================

/// @brief Singleton registry for dynamically loaded plugins
/// @details Uses the same Meyer's singleton pattern as runtime_registry.
///          Thread-safe: all public methods are internally synchronized.
///          Query methods return metadata snapshots detached from the internal
///          registry, so subsequent load/unload operations do not invalidate
///          the returned object itself.
class plugin_registry {
public:
    /// @brief Get the singleton instance
    DTL_RUNTIME_API static plugin_registry& instance();

    /// @brief Attempt to load a plugin from a filesystem path
    /// @param path Path to the plugin shared library (.so / .dll)
    /// @return The plugin name on success, or an error
    DTL_RUNTIME_API dtl::result<std::string> load_plugin(std::string_view path);

    /// @brief Unload a previously loaded plugin by name
    /// @param name The plugin name (as returned by load_plugin)
    /// @return Success or error
    DTL_RUNTIME_API dtl::result<void> unload_plugin(std::string_view name);

    /// @brief Find a loaded plugin by name
    /// @param name The plugin name
    /// @return Metadata snapshot, or std::nullopt if not found
    /// @note Returned snapshots clear live callback and DSO-handle fields.
    DTL_RUNTIME_API std::optional<plugin_descriptor> find_plugin(std::string_view name) const;

    /// @brief Get metadata snapshots for all loaded plugins
    /// @note Returned snapshots clear live callback and DSO-handle fields.
    DTL_RUNTIME_API std::vector<plugin_descriptor> loaded_plugins() const;

    /// @brief Unload all plugins (called during shutdown)
    DTL_RUNTIME_API void unload_all();

    /// @brief Number of currently loaded plugins
    [[nodiscard]] DTL_RUNTIME_API size_t count() const noexcept;

    // Non-copyable, non-movable
    plugin_registry(const plugin_registry&) = delete;
    plugin_registry& operator=(const plugin_registry&) = delete;
    plugin_registry(plugin_registry&&) = delete;
    plugin_registry& operator=(plugin_registry&&) = delete;

private:
    plugin_registry() = default;
    ~plugin_registry() = default;
};

}  // namespace dtl::runtime
