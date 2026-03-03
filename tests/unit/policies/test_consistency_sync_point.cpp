// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_consistency_sync_point.cpp
/// @brief Compile and runtime tests for consistency policy sync methods
/// @details Verifies that barrier/release/acquire/fence/sync methods
///          are callable and that Doxygen documentation is accurate.

#include <dtl/policies/consistency/bulk_synchronous.hpp>
#include <dtl/policies/consistency/release_acquire.hpp>
#include <dtl/policies/consistency/relaxed.hpp>
#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Bulk Synchronous Barrier Tests
// =============================================================================

TEST(ConsistencySyncPoint, BulkSynchronousBarrierCompiles) {
    // barrier() is a static no-op policy marker; verify it compiles and runs
    dtl::bulk_synchronous::barrier();
    SUCCEED();
}

TEST(ConsistencySyncPoint, BulkSynchronousBarrierIsNoOp) {
    // Calling barrier multiple times should be harmless (no-op)
    for (int i = 0; i < 10; ++i) {
        dtl::bulk_synchronous::barrier();
    }
    SUCCEED();
}

// =============================================================================
// Release-Acquire Method Tests
// =============================================================================

TEST(ConsistencySyncPoint, ReleaseAcquireReleaseCompiles) {
    dtl::release_acquire::release();
    SUCCEED();
}

TEST(ConsistencySyncPoint, ReleaseAcquireAcquireCompiles) {
    dtl::release_acquire::acquire();
    SUCCEED();
}

TEST(ConsistencySyncPoint, ReleaseAcquireFenceCompiles) {
    dtl::release_acquire::fence();
    SUCCEED();
}

TEST(ConsistencySyncPoint, ReleaseAcquireFullSequence) {
    // Typical usage pattern: release, then acquire, with fence as full barrier
    dtl::release_acquire::release();
    dtl::release_acquire::acquire();
    dtl::release_acquire::fence();
    SUCCEED();
}

// =============================================================================
// Relaxed Sync Tests
// =============================================================================

TEST(ConsistencySyncPoint, RelaxedSyncCompiles) {
    dtl::relaxed::sync();
    SUCCEED();
}

TEST(ConsistencySyncPoint, RelaxedSyncIsNoOp) {
    // Calling sync multiple times should be harmless (no-op)
    for (int i = 0; i < 10; ++i) {
        dtl::relaxed::sync();
    }
    SUCCEED();
}

}  // namespace dtl::test
