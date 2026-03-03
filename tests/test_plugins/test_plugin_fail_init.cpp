// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_plugin_fail_init.cpp
/// @brief Test plugin whose init function returns failure

#include <dtl/runtime/plugin_loader.hpp>

static int failing_init() {
    return -1;  // Signal failure
}

extern "C" {

int dtl_plugin_register(dtl::runtime::dtl_plugin_info* info) {
    if (!info) return -1;
    info->name = "fail_init_plugin";
    info->version = "1.0.0";
    info->abi_version = dtl::runtime::plugin_abi_version;
    info->init = failing_init;
    info->fini = nullptr;
    return 0;
}

}  // extern "C"
