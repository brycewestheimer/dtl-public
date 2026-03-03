// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_mpmd.cpp
 * @brief Unit tests for DTL C bindings MPMD operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>

// ============================================================================
// Role Manager Lifecycle Tests
// ============================================================================

TEST(CBindingsMpmd, CreateRoleManagerSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_status status = dtl_role_manager_create(ctx, &mgr);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(mgr, nullptr);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, CreateWithNullContextFails) {
    dtl_role_manager_t mgr = nullptr;
    dtl_status status = dtl_role_manager_create(nullptr, &mgr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsMpmd, CreateWithNullOutputFails) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_status status = dtl_role_manager_create(ctx, nullptr);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, DestroyNullIsSafe) {
    dtl_role_manager_destroy(nullptr);
}

// ============================================================================
// Role Configuration Tests
// ============================================================================

TEST(CBindingsMpmd, AddRoleSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_status status = dtl_role_manager_add_role(mgr, "worker", 1);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, AddRoleWithNullNameFails) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_status status = dtl_role_manager_add_role(mgr, nullptr, 1);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, AddRoleWithNullManagerFails) {
    dtl_status status = dtl_role_manager_add_role(nullptr, "worker", 1);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Role Manager Initialize Tests
// ============================================================================

TEST(CBindingsMpmd, InitializeSingleRoleSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_size_t num_ranks = static_cast<dtl_size_t>(dtl_context_size(ctx));
    dtl_role_manager_add_role(mgr, "all", num_ranks);

    dtl_status status = dtl_role_manager_initialize(mgr);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, InitializeWithNullFails) {
    dtl_status status = dtl_role_manager_initialize(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Role Query Tests
// ============================================================================

TEST(CBindingsMpmd, HasRoleAfterInitialize) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_size_t num_ranks = static_cast<dtl_size_t>(dtl_context_size(ctx));
    dtl_role_manager_add_role(mgr, "worker", num_ranks);
    dtl_role_manager_initialize(mgr);

    int has_role = 0;
    dtl_status status = dtl_role_manager_has_role(mgr, "worker", &has_role);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(has_role, 1);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, HasRoleReturnsFalseForUnknown) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_size_t num_ranks = static_cast<dtl_size_t>(dtl_context_size(ctx));
    dtl_role_manager_add_role(mgr, "worker", num_ranks);
    dtl_role_manager_initialize(mgr);

    int has_role = 1;
    dtl_status status = dtl_role_manager_has_role(mgr, "nonexistent", &has_role);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(has_role, 0);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, RoleSizeReturnsCorrectCount) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_size_t num_ranks = static_cast<dtl_size_t>(dtl_context_size(ctx));
    dtl_role_manager_add_role(mgr, "worker", num_ranks);
    dtl_role_manager_initialize(mgr);

    dtl_size_t size = 0;
    dtl_status status = dtl_role_manager_role_size(mgr, "worker", &size);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(size, num_ranks);

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}

TEST(CBindingsMpmd, RoleRankIsValid) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_role_manager_t mgr = nullptr;
    dtl_role_manager_create(ctx, &mgr);

    dtl_size_t num_ranks = static_cast<dtl_size_t>(dtl_context_size(ctx));
    dtl_role_manager_add_role(mgr, "worker", num_ranks);
    dtl_role_manager_initialize(mgr);

    dtl_rank_t rank = -1;
    dtl_status status = dtl_role_manager_role_rank(mgr, "worker", &rank);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, static_cast<dtl_rank_t>(num_ranks));

    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);
}
