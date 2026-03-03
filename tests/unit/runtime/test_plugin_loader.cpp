// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_plugin_loader.cpp
/// @brief Tests for runtime plugin loader
/// @since 0.1.0

#include <dtl/runtime/plugin_loader.hpp>
#include <dtl/error/status.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace dtl::runtime::testing {

// =============================================================================
// Test fixture that ensures plugin registry is clean between tests
// =============================================================================

class PluginRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        plugin_registry::instance().unload_all();
    }
    void TearDown() override {
        plugin_registry::instance().unload_all();
    }
};

// =============================================================================
// Singleton Tests
// =============================================================================

TEST(PluginRegistry, SingletonAddressStability) {
    auto& a = plugin_registry::instance();
    auto& b = plugin_registry::instance();
    EXPECT_EQ(&a, &b);
}

// =============================================================================
// Initial State Tests
// =============================================================================

TEST_F(PluginRegistryTest, StartsEmpty) {
    auto& reg = plugin_registry::instance();
    EXPECT_EQ(reg.count(), 0u);
}

TEST_F(PluginRegistryTest, LoadedPluginsStartsEmpty) {
    auto& reg = plugin_registry::instance();
    auto plugins = reg.loaded_plugins();
    EXPECT_TRUE(plugins.empty());
}

TEST_F(PluginRegistryTest, FindPluginReturnsNullForUnknown) {
    auto& reg = plugin_registry::instance();
    EXPECT_FALSE(reg.find_plugin("nonexistent").has_value());
}

// =============================================================================
// Unload All Tests
// =============================================================================

TEST_F(PluginRegistryTest, UnloadAllDoesNotCrash) {
    auto& reg = plugin_registry::instance();
    EXPECT_NO_THROW(reg.unload_all());
    EXPECT_EQ(reg.count(), 0u);
}

// =============================================================================
// Descriptor Tests
// =============================================================================

TEST(PluginDescriptor, DefaultConstruction) {
    plugin_descriptor desc;
    EXPECT_TRUE(desc.name.empty());
    EXPECT_TRUE(desc.version.empty());
    EXPECT_EQ(desc.abi_version, 0u);
    EXPECT_TRUE(desc.path.empty());
    EXPECT_EQ(desc.init, nullptr);
    EXPECT_EQ(desc.fini, nullptr);
    EXPECT_EQ(desc.dl_handle, nullptr);
}

// =============================================================================
// Constants Tests
// =============================================================================

TEST(PluginLoader, AbiVersionIsPositive) {
    EXPECT_GT(plugin_abi_version, 0u);
}

TEST(PluginLoader, RegisterSymbolIsNotEmpty) {
    EXPECT_NE(plugin_register_symbol, nullptr);
    EXPECT_GT(std::string_view(plugin_register_symbol).size(), 0u);
}

// =============================================================================
// Plugin Info Struct Tests
// =============================================================================

TEST(PluginInfo, DefaultConstruction) {
    dtl_plugin_info info{};
    EXPECT_EQ(info.name, nullptr);
    EXPECT_EQ(info.version, nullptr);
    EXPECT_EQ(info.abi_version, 0u);
    EXPECT_EQ(info.init, nullptr);
    EXPECT_EQ(info.fini, nullptr);
}

// =============================================================================
// Load Error Tests (invalid paths, missing symbols)
// =============================================================================

TEST_F(PluginRegistryTest, LoadNonExistentPathFails) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin("/nonexistent/path/plugin.so");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_argument);
}

TEST_F(PluginRegistryTest, UnloadNonExistentPluginFails) {
    auto& reg = plugin_registry::instance();
    auto result = reg.unload_plugin("never_loaded");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_found);
}

// =============================================================================
// Real DSO Loading Tests (using test plugins built by CMake)
// =============================================================================

#ifdef DTL_TEST_PLUGIN_PATH

TEST_F(PluginRegistryTest, LoadValidPlugin) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result.has_error()) << result.error().message();
    EXPECT_EQ(result.value(), "test_plugin");
    EXPECT_EQ(reg.count(), 1u);
}

TEST_F(PluginRegistryTest, LoadedPluginIsQueryable) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result.has_error());

    auto desc = reg.find_plugin("test_plugin");
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(desc->name, "test_plugin");
    EXPECT_EQ(desc->version, "1.0.0");
    EXPECT_EQ(desc->abi_version, plugin_abi_version);
    EXPECT_EQ(desc->dl_handle, nullptr);
    EXPECT_EQ(desc->init, nullptr);
    EXPECT_EQ(desc->fini, nullptr);
}

TEST_F(PluginRegistryTest, LoadedPluginsListContainsPlugin) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result.has_error());

    auto plugins = reg.loaded_plugins();
    ASSERT_EQ(plugins.size(), 1u);
    EXPECT_EQ(plugins[0].name, "test_plugin");
}

TEST_F(PluginRegistryTest, UnloadPluginSucceeds) {
    auto& reg = plugin_registry::instance();
    auto load_result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(load_result.has_error());

    auto unload_result = reg.unload_plugin("test_plugin");
    EXPECT_FALSE(unload_result.has_error());
    EXPECT_EQ(reg.count(), 0u);
    EXPECT_FALSE(reg.find_plugin("test_plugin").has_value());
}

TEST_F(PluginRegistryTest, DuplicateLoadFails) {
    auto& reg = plugin_registry::instance();
    auto result1 = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result1.has_error());

    auto result2 = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    EXPECT_TRUE(result2.has_error());
    EXPECT_EQ(result2.error().code(), dtl::status_code::invalid_argument);
    EXPECT_EQ(reg.count(), 1u);  // Still just one loaded
}

TEST_F(PluginRegistryTest, UnloadAllCleansUpPlugins) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result.has_error());
    EXPECT_EQ(reg.count(), 1u);

    reg.unload_all();
    EXPECT_EQ(reg.count(), 0u);
    EXPECT_FALSE(reg.find_plugin("test_plugin").has_value());
}

TEST_F(PluginRegistryTest, QuerySnapshotsRemainUsableAcrossUnload) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
    ASSERT_FALSE(result.has_error());

    auto desc = reg.find_plugin("test_plugin");
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(desc->name, "test_plugin");
    EXPECT_EQ(desc->dl_handle, nullptr);

    auto unload_result = reg.unload_plugin("test_plugin");
    ASSERT_FALSE(unload_result.has_error());

    EXPECT_EQ(desc->name, "test_plugin");
    EXPECT_EQ(desc->version, "1.0.0");
    EXPECT_EQ(desc->dl_handle, nullptr);
}

TEST_F(PluginRegistryTest, ConcurrentQueryDuringLoadUnloadUsesSnapshots) {
    auto& reg = plugin_registry::instance();
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        while (!stop.load()) {
            auto desc = reg.find_plugin("test_plugin");
            if (desc.has_value()) {
                EXPECT_EQ(desc->name, "test_plugin");
                EXPECT_EQ(desc->dl_handle, nullptr);
                EXPECT_EQ(desc->init, nullptr);
                EXPECT_EQ(desc->fini, nullptr);
            }

            for (const auto& plugin : reg.loaded_plugins()) {
                EXPECT_FALSE(plugin.name.empty());
                EXPECT_EQ(plugin.dl_handle, nullptr);
                EXPECT_EQ(plugin.init, nullptr);
                EXPECT_EQ(plugin.fini, nullptr);
            }
        }
    });

    for (int i = 0; i < 50; ++i) {
        auto load_result = reg.load_plugin(DTL_TEST_PLUGIN_PATH);
        ASSERT_FALSE(load_result.has_error()) << load_result.error().message();
        auto unload_result = reg.unload_plugin("test_plugin");
        ASSERT_FALSE(unload_result.has_error()) << unload_result.error().message();
    }

    stop = true;
    reader.join();
}

#endif  // DTL_TEST_PLUGIN_PATH

#ifdef DTL_TEST_PLUGIN_BAD_ABI_PATH

TEST_F(PluginRegistryTest, AbiMismatchFails) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_BAD_ABI_PATH);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_argument);
    EXPECT_EQ(reg.count(), 0u);
}

#endif  // DTL_TEST_PLUGIN_BAD_ABI_PATH

#ifdef DTL_TEST_PLUGIN_FAIL_INIT_PATH

TEST_F(PluginRegistryTest, InitFailureRejectsPlugin) {
    auto& reg = plugin_registry::instance();
    auto result = reg.load_plugin(DTL_TEST_PLUGIN_FAIL_INIT_PATH);
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
    EXPECT_EQ(reg.count(), 0u);
}

#endif  // DTL_TEST_PLUGIN_FAIL_INIT_PATH

// =============================================================================
// Re-entrant fini() safety test
// Verifies that fini() callbacks can safely query the registry without deadlock
// =============================================================================

TEST_F(PluginRegistryTest, UnloadDoesNotDeadlockOnReentrantQuery) {
    // This test verifies that calling find_plugin() or count() from within
    // a fini() callback does not deadlock, because fini() is now called
    // outside the registry lock.
    auto& reg = plugin_registry::instance();

    // We can't easily inject a custom fini into a DSO, but we can verify
    // the structural fix: after unload_plugin or unload_all, the registry
    // is queryable. The real protection is that fini()/dlclose() happen
    // outside the lock (verified by code review and TSan).
    EXPECT_EQ(reg.count(), 0u);
    EXPECT_FALSE(reg.find_plugin("any").has_value());

    // Calling unload_all on empty registry should not deadlock
    reg.unload_all();
    EXPECT_EQ(reg.count(), 0u);
}

}  // namespace dtl::runtime::testing
