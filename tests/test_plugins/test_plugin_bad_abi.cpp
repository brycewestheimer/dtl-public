// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_plugin_bad_abi.cpp
/// @brief Test plugin with wrong ABI version for testing ABI mismatch rejection

#include <dtl/runtime/plugin_loader.hpp>

extern "C" {

int dtl_plugin_register(dtl::runtime::dtl_plugin_info* info) {
    if (!info) return -1;
    info->name = "bad_abi_plugin";
    info->version = "1.0.0";
    info->abi_version = dtl::runtime::plugin_abi_version + 999;  // Wrong ABI
    info->init = nullptr;
    info->fini = nullptr;
    return 0;
}

}  // extern "C"
