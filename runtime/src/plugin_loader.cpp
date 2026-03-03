// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file plugin_loader.cpp
/// @brief Plugin loader implementation using dlopen (POSIX) / LoadLibrary (Windows)
/// @details Loads plugin DSOs, resolves the dtl_plugin_register symbol, validates
///          ABI version, calls init, and registers into the singleton registry.
/// @since 0.1.0

#include <dtl/runtime/plugin_loader.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>

#include <mutex>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace dtl::runtime {

// =============================================================================
// Platform abstraction for dynamic library loading
// =============================================================================

namespace {

#ifdef _WIN32

void* platform_dlopen(const char* path) {
    return static_cast<void*>(LoadLibraryA(path));
}

void* platform_dlsym(void* handle, const char* symbol) {
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), symbol));
}

int platform_dlclose(void* handle) {
    return FreeLibrary(static_cast<HMODULE>(handle)) ? 0 : -1;
}

std::string platform_dlerror() {
    DWORD err = GetLastError();
    if (err == 0) return "unknown error";
    LPSTR buf = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                   nullptr, err, 0, reinterpret_cast<LPSTR>(&buf), 0, nullptr);
    std::string msg(buf ? buf : "unknown error");
    LocalFree(buf);
    return msg;
}

#else  // POSIX

void* platform_dlopen(const char* path) {
    return ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void* platform_dlsym(void* handle, const char* symbol) {
    return ::dlsym(handle, symbol);
}

int platform_dlclose(void* handle) {
    return ::dlclose(handle);
}

std::string platform_dlerror() {
    const char* err = ::dlerror();
    return err ? std::string(err) : "unknown error";
}

#endif

}  // anonymous namespace

// =============================================================================
// Implementation detail — hidden in the .cpp to keep the header clean
// =============================================================================

namespace {

plugin_descriptor make_snapshot(const plugin_descriptor& desc) {
    plugin_descriptor snapshot = desc;
    snapshot.init = nullptr;
    snapshot.fini = nullptr;
    snapshot.dl_handle = nullptr;
    return snapshot;
}

struct registry_state {
    mutable std::mutex mtx;
    std::unordered_map<std::string, plugin_descriptor> plugins;
};

registry_state& state() {
    static registry_state s;
    return s;
}

}  // anonymous namespace

// =============================================================================
// Singleton
// =============================================================================

plugin_registry& plugin_registry::instance() {
    static plugin_registry r;
    return r;
}

// =============================================================================
// Load / Unload
// =============================================================================

dtl::result<std::string> plugin_registry::load_plugin(std::string_view path) {
    auto& s = state();
    std::lock_guard lock(s.mtx);

    // 1. Open the shared library
    std::string path_str(path);
    void* handle = platform_dlopen(path_str.c_str());
    if (!handle) {
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_argument,
            "failed to load plugin '" + path_str + "': " + platform_dlerror());
    }

    // 2. Look up the registration symbol
    void* sym = platform_dlsym(handle, plugin_register_symbol);
    if (!sym) {
        std::string err = platform_dlerror();
        platform_dlclose(handle);
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_argument,
            "plugin '" + path_str + "' does not export '" +
            plugin_register_symbol + "': " + err);
    }

    // 3. Call the registration function
    auto register_fn = reinterpret_cast<plugin_register_fn>(sym);
    dtl_plugin_info info{};
    int reg_result = register_fn(&info);
    if (reg_result != 0) {
        platform_dlclose(handle);
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_state,
            "plugin registration function returned error code " +
            std::to_string(reg_result));
    }

    // 4. Validate required fields
    if (!info.name || info.name[0] == '\0') {
        platform_dlclose(handle);
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_argument,
            "plugin at '" + path_str + "' has empty name");
    }

    // 5. Copy name/version now (before any dlclose that would invalidate info pointers)
    std::string name(info.name);
    std::string version(info.version ? info.version : "");

    // 6. Validate ABI version
    if (info.abi_version != plugin_abi_version) {
        platform_dlclose(handle);
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_argument,
            "plugin '" + name + "' ABI version " +
            std::to_string(info.abi_version) + " does not match host ABI version " +
            std::to_string(plugin_abi_version));
    }

    // 7. Check for duplicate
    if (s.plugins.count(name)) {
        platform_dlclose(handle);
        return dtl::make_error<std::string>(
            dtl::status_code::invalid_argument,
            "plugin '" + name + "' is already loaded");
    }

    // 7. Call init if provided
    if (info.init) {
        int init_result = info.init();
        if (init_result != 0) {
            platform_dlclose(handle);
            return dtl::make_error<std::string>(
                dtl::status_code::invalid_state,
                "plugin '" + name + "' init failed with code " +
                std::to_string(init_result));
        }
    }

    // 8. Register
    plugin_descriptor desc;
    desc.name = name;
    desc.version = std::move(version);
    desc.abi_version = info.abi_version;
    desc.path = path_str;
    desc.init = info.init;
    desc.fini = info.fini;
    desc.dl_handle = handle;

    s.plugins.emplace(name, std::move(desc));
    return dtl::result<std::string>{name};
}

dtl::result<void> plugin_registry::unload_plugin(std::string_view name) {
    plugin_descriptor local_desc;

    {
        auto& s = state();
        std::lock_guard lock(s.mtx);

        auto it = s.plugins.find(std::string(name));
        if (it == s.plugins.end()) {
            return dtl::make_error<void>(
                dtl::status_code::not_found,
                "plugin '" + std::string(name) + "' is not loaded");
        }

        // Copy descriptor and remove from registry under lock
        local_desc = std::move(it->second);
        s.plugins.erase(it);
    }

    // Call fini and dlclose outside the lock to avoid deadlock
    // if fini() re-enters the registry (e.g., calls find_plugin())
    if (local_desc.fini) {
        local_desc.fini();
    }
    if (local_desc.dl_handle) {
        platform_dlclose(local_desc.dl_handle);
    }

    return dtl::result<void>{};
}

// =============================================================================
// Query
// =============================================================================

std::optional<plugin_descriptor> plugin_registry::find_plugin(std::string_view name) const {
    auto& s = state();
    std::lock_guard lock(s.mtx);
    auto it = s.plugins.find(std::string(name));
    if (it != s.plugins.end()) {
        return make_snapshot(it->second);
    }
    return std::nullopt;
}

std::vector<plugin_descriptor> plugin_registry::loaded_plugins() const {
    auto& s = state();
    std::lock_guard lock(s.mtx);
    std::vector<plugin_descriptor> result;
    result.reserve(s.plugins.size());
    for (const auto& [name, desc] : s.plugins) {
        result.push_back(make_snapshot(desc));
    }
    return result;
}

void plugin_registry::unload_all() {
    std::unordered_map<std::string, plugin_descriptor> local_plugins;

    {
        auto& s = state();
        std::lock_guard lock(s.mtx);
        // Swap plugins into local copy and clear under lock
        local_plugins.swap(s.plugins);
    }

    // Call fini and dlclose outside the lock to avoid deadlock
    // if fini() re-enters the registry (e.g., calls find_plugin() or count())
    for (auto& [name, desc] : local_plugins) {
        if (desc.fini) {
            desc.fini();
        }
        if (desc.dl_handle) {
            platform_dlclose(desc.dl_handle);
        }
    }
}

size_t plugin_registry::count() const noexcept {
    auto& s = state();
    std::lock_guard lock(s.mtx);
    return s.plugins.size();
}

}  // namespace dtl::runtime
