# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Test Plugin DSOs for plugin_loader unit tests
# =============================================================================
# These are minimal shared libraries that export dtl_plugin_register.
# They are NOT installed — only used during testing.

# Good plugin: correct ABI, init/fini callbacks
add_library(dtl_test_plugin SHARED test_plugin.cpp)
target_link_libraries(dtl_test_plugin PRIVATE dtl_runtime)
set_target_properties(dtl_test_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dtl_test_plugin"
)

# Bad ABI plugin: wrong ABI version (should be rejected)
add_library(dtl_test_plugin_bad_abi SHARED test_plugin_bad_abi.cpp)
target_link_libraries(dtl_test_plugin_bad_abi PRIVATE dtl_runtime)
set_target_properties(dtl_test_plugin_bad_abi PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dtl_test_plugin_bad_abi"
)

# Fail init plugin: init returns error (should be rejected)
add_library(dtl_test_plugin_fail_init SHARED test_plugin_fail_init.cpp)
target_link_libraries(dtl_test_plugin_fail_init PRIVATE dtl_runtime)
set_target_properties(dtl_test_plugin_fail_init PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dtl_test_plugin_fail_init"
)

# Make test plugins available to the test executable via compile definitions
# that provide the DSO paths
target_compile_definitions(dtl_unit_tests PRIVATE
    DTL_TEST_PLUGIN_PATH="$<TARGET_FILE:dtl_test_plugin>"
    DTL_TEST_PLUGIN_BAD_ABI_PATH="$<TARGET_FILE:dtl_test_plugin_bad_abi>"
    DTL_TEST_PLUGIN_FAIL_INIT_PATH="$<TARGET_FILE:dtl_test_plugin_fail_init>"
)

# Ensure test plugins are built before unit tests run
add_dependencies(dtl_unit_tests
    dtl_test_plugin
    dtl_test_plugin_bad_abi
    dtl_test_plugin_fail_init
)
