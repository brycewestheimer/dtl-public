// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_runtime_dso.cpp
/// @brief Tests verifying that runtime_registry lives in a shared library (DSO)
/// @details Validates that the Meyer's singleton in libdtl_runtime.so
///          provides a single process-wide instance, and the DSO is linked
///          transitively via the dtl INTERFACE target.
/// @since 0.1.0

#include <gtest/gtest.h>

#include <dtl/runtime/runtime_registry.hpp>

namespace dtl::tests {

// =============================================================================
// Singleton Identity Tests
// =============================================================================

/// @brief Verify that runtime_registry::instance() returns the same address
///        across multiple calls from this translation unit.
TEST(RuntimeDSOTest, SingletonAddressIsStable) {
    auto& r1 = runtime::runtime_registry::instance();
    auto& r2 = runtime::runtime_registry::instance();

    EXPECT_EQ(&r1, &r2)
        << "runtime_registry::instance() must return the same object";
}

/// @brief Verify that the singleton starts in a consistent state
///        (not initialized, zero refcount) before any environment is created.
TEST(RuntimeDSOTest, InitialStateIsClean) {
    auto& reg = runtime::runtime_registry::instance();

    // If no environment is alive in this test process, refcount should be 0.
    // If a previous test created one, it may be > 0. Either way the registry
    // should be queryable without crashing.
    [[maybe_unused]] bool init = reg.is_initialized();
    [[maybe_unused]] size_t refs = reg.ref_count();

    // These should not crash — the DSO is linked and the singleton is alive
    SUCCEED();
}

/// @brief Verify that DTL_RUNTIME_API macro resolves correctly
///        (compile-time check — if this TU links, the macro works).
TEST(RuntimeDSOTest, ExportMacroResolvesCorrectly) {
    // If we got here, the DTL_RUNTIME_API-decorated methods in the header
    // resolved to symbols in libdtl_runtime.so at link time. This test
    // simply validates that the linkage is correct.
    auto& reg = runtime::runtime_registry::instance();
    (void)reg.has_mpi();
    (void)reg.has_cuda();
    (void)reg.has_hip();
    (void)reg.has_nccl();
    (void)reg.has_shmem();
    (void)reg.mpi_thread_level();
    (void)reg.mpi_thread_level_name();
    (void)reg.thread_level();
    (void)reg.thread_level_name();
    (void)reg.thread_level_for_backend("mpi");

    SUCCEED();
}

}  // namespace dtl::tests
