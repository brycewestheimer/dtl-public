// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_plugin.cpp
/// @brief Minimal test plugin for plugin_loader unit tests
/// @details Exports the dtl_plugin_register symbol with correct ABI version.
///          Used by test_plugin_loader.cpp to verify real dlopen/dlsym loading.

#include <dtl/runtime/plugin_loader.hpp>

static bool g_initialized = false;
static bool g_finalized = false;

static int test_init() {
    g_initialized = true;
    return 0;
}

static void test_fini() {
    g_finalized = true;
}

extern "C" {

/// @brief Plugin registration entry point
int dtl_plugin_register(dtl::runtime::dtl_plugin_info* info) {
    if (!info) return -1;
    info->name = "test_plugin";
    info->version = "1.0.0";
    info->abi_version = dtl::runtime::plugin_abi_version;
    info->init = test_init;
    info->fini = test_fini;
    return 0;
}

}  // extern "C"
