// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_context_registry.cpp
/// @brief Unit tests for context_registry
/// @details Verifies context_registry lifecycle and thread-safe operations.

#include <dtl/backend/common/backend_context.hpp>
#include <dtl/backend/common/backend_traits.hpp>

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace dtl::test {

// =============================================================================
// Test Backend Tag
// =============================================================================

/// @brief Test backend tag for registry testing
struct test_backend_tag {};

}  // namespace dtl::test

namespace dtl {
/// @brief Traits specialization for test backend
template <>
struct backend_traits<test::test_backend_tag> {
    static constexpr bool supports_point_to_point = false;
    static constexpr bool supports_collectives = false;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = false;
    static constexpr bool supports_async = false;
    static constexpr bool supports_thread_multiple = false;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "test";
};
}  // namespace dtl

namespace dtl::test {

// =============================================================================
// Type Aliases
// =============================================================================

using test_context = backend_context<test_backend_tag>;

// =============================================================================
// Singleton Tests
// =============================================================================

TEST(ContextRegistryTest, InstanceReturnsSingleton) {
    auto& registry1 = context_registry::instance();
    auto& registry2 = context_registry::instance();

    EXPECT_EQ(&registry1, &registry2);
}

// =============================================================================
// Registration Tests
// =============================================================================

TEST(ContextRegistryTest, RegisterAndGetContext) {
    auto& registry = context_registry::instance();

    test_context ctx;
    ctx.initialize();

    registry.register_context("test_ctx", &ctx);

    backend_context_base* retrieved = registry.get_context("test_ctx");
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved, &ctx);

    registry.unregister_context("test_ctx");
}

TEST(ContextRegistryTest, GetNonExistentContextReturnsNullptr) {
    auto& registry = context_registry::instance();

    backend_context_base* retrieved = registry.get_context("nonexistent");
    EXPECT_EQ(retrieved, nullptr);
}

TEST(ContextRegistryTest, ContainsRegisteredContext) {
    auto& registry = context_registry::instance();

    test_context ctx;
    ctx.initialize();

    registry.register_context("test_ctx_2", &ctx);

    EXPECT_TRUE(registry.contains("test_ctx_2"));
    EXPECT_FALSE(registry.contains("nonexistent"));

    registry.unregister_context("test_ctx_2");
}

TEST(ContextRegistryTest, UnregisterRemovesContext) {
    auto& registry = context_registry::instance();

    test_context ctx;
    ctx.initialize();

    registry.register_context("test_ctx_3", &ctx);
    EXPECT_TRUE(registry.contains("test_ctx_3"));

    registry.unregister_context("test_ctx_3");
    EXPECT_FALSE(registry.contains("test_ctx_3"));

    backend_context_base* retrieved = registry.get_context("test_ctx_3");
    EXPECT_EQ(retrieved, nullptr);
}

// =============================================================================
// Size Tests
// =============================================================================

TEST(ContextRegistryTest, SizeTracksRegistrations) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    EXPECT_EQ(registry.size(), 0);

    test_context ctx1, ctx2, ctx3;
    ctx1.initialize();
    ctx2.initialize();
    ctx3.initialize();

    registry.register_context("size_test_1", &ctx1);
    EXPECT_EQ(registry.size(), 1);

    registry.register_context("size_test_2", &ctx2);
    EXPECT_EQ(registry.size(), 2);

    registry.register_context("size_test_3", &ctx3);
    EXPECT_EQ(registry.size(), 3);

    registry.unregister_context("size_test_2");
    EXPECT_EQ(registry.size(), 2);

    registry.unregister_context("size_test_1");
    EXPECT_EQ(registry.size(), 1);

    registry.unregister_context("size_test_3");
    EXPECT_EQ(registry.size(), 0);
}

// =============================================================================
// Finalize All Tests
// =============================================================================

TEST(ContextRegistryTest, FinalizeAllFinalizesActiveContexts) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context ctx1, ctx2;
    ctx1.initialize();
    ctx2.initialize();

    EXPECT_TRUE(ctx1.is_active());
    EXPECT_TRUE(ctx2.is_active());

    registry.register_context("finalize_test_1", &ctx1);
    registry.register_context("finalize_test_2", &ctx2);

    registry.finalize_all();

    EXPECT_FALSE(ctx1.is_active());
    EXPECT_FALSE(ctx2.is_active());
    EXPECT_EQ(registry.size(), 0);
}

TEST(ContextRegistryTest, FinalizeAllClearsRegistry) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context ctx1, ctx2;
    ctx1.initialize();
    ctx2.initialize();

    registry.register_context("clear_test_1", &ctx1);
    registry.register_context("clear_test_2", &ctx2);

    EXPECT_EQ(registry.size(), 2);

    registry.finalize_all();

    EXPECT_EQ(registry.size(), 0);
    EXPECT_FALSE(registry.contains("clear_test_1"));
    EXPECT_FALSE(registry.contains("clear_test_2"));
}

TEST(ContextRegistryTest, FinalizeAllSkipsInactiveContexts) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context ctx1, ctx2;
    ctx1.initialize();
    // ctx2 remains uninitialized

    registry.register_context("inactive_test_1", &ctx1);
    registry.register_context("inactive_test_2", &ctx2);

    EXPECT_TRUE(ctx1.is_active());
    EXPECT_FALSE(ctx2.is_active());

    // Should finalize ctx1 but skip ctx2 (already inactive)
    registry.finalize_all();

    EXPECT_FALSE(ctx1.is_active());
    EXPECT_FALSE(ctx2.is_active());
    EXPECT_EQ(registry.size(), 0);
}

// =============================================================================
// Overwrite Tests
// =============================================================================

TEST(ContextRegistryTest, RegisterSameNameOverwrites) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context ctx1, ctx2;
    ctx1.initialize();
    ctx2.initialize();

    registry.register_context("overwrite_test", &ctx1);
    backend_context_base* retrieved1 = registry.get_context("overwrite_test");
    EXPECT_EQ(retrieved1, &ctx1);

    // Register with same name, should overwrite
    registry.register_context("overwrite_test", &ctx2);
    backend_context_base* retrieved2 = registry.get_context("overwrite_test");
    EXPECT_EQ(retrieved2, &ctx2);

    // Size should still be 1
    EXPECT_EQ(registry.size(), 1);

    registry.unregister_context("overwrite_test");
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(ContextRegistryTest, ConcurrentRegistrationThreadSafe) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    constexpr int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<test_context> contexts(static_cast<size_t>(num_threads));

    // Initialize all contexts
    for (int i = 0; i < num_threads; ++i) {
        contexts[static_cast<size_t>(i)].initialize();
    }

    // Register from multiple threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&registry, &contexts, i]() {
            std::string name = "thread_test_" + std::to_string(i);
            registry.register_context(name.c_str(), &contexts[static_cast<size_t>(i)]);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All contexts should be registered
    EXPECT_EQ(registry.size(), num_threads);

    for (int i = 0; i < num_threads; ++i) {
        std::string name = "thread_test_" + std::to_string(i);
        EXPECT_TRUE(registry.contains(name.c_str()));
    }

    registry.finalize_all();
}

TEST(ContextRegistryTest, ConcurrentGetThreadSafe) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context ctx;
    ctx.initialize();
    registry.register_context("concurrent_get_test", &ctx);

    constexpr int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<backend_context_base*> results(static_cast<size_t>(num_threads), nullptr);

    // Get from multiple threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&registry, &results, i]() {
            results[static_cast<size_t>(i)] = registry.get_context("concurrent_get_test");
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should have retrieved the same context
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_EQ(results[static_cast<size_t>(i)], &ctx);
    }

    registry.unregister_context("concurrent_get_test");
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(ContextRegistryTest, UnregisterNonExistentContextNoOp) {
    auto& registry = context_registry::instance();

    // Should not crash or throw
    registry.unregister_context("does_not_exist");

    EXPECT_FALSE(registry.contains("does_not_exist"));
}

TEST(ContextRegistryTest, RegisterNullptrContext) {
    auto& registry = context_registry::instance();

    // Register nullptr (allowed, but not useful)
    registry.register_context("nullptr_test", nullptr);

    EXPECT_TRUE(registry.contains("nullptr_test"));

    backend_context_base* retrieved = registry.get_context("nullptr_test");
    EXPECT_EQ(retrieved, nullptr);

    registry.unregister_context("nullptr_test");
}

TEST(ContextRegistryTest, EmptyStringName) {
    auto& registry = context_registry::instance();

    test_context ctx;
    ctx.initialize();

    registry.register_context("", &ctx);

    EXPECT_TRUE(registry.contains(""));

    backend_context_base* retrieved = registry.get_context("");
    EXPECT_EQ(retrieved, &ctx);

    registry.unregister_context("");
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ContextRegistryTest, MultipleContextTypes) {
    auto& registry = context_registry::instance();

    // Clear any existing contexts
    registry.finalize_all();

    test_context test_ctx;
    backend_context<mpi_backend_tag> mpi_ctx;

    test_ctx.initialize();
    mpi_ctx.initialize();

    registry.register_context("test", &test_ctx);
    registry.register_context("mpi", &mpi_ctx);

    EXPECT_EQ(registry.size(), 2);

    backend_context_base* retrieved_test = registry.get_context("test");
    backend_context_base* retrieved_mpi = registry.get_context("mpi");

    EXPECT_EQ(retrieved_test, &test_ctx);
    EXPECT_EQ(retrieved_mpi, &mpi_ctx);

    EXPECT_STREQ(retrieved_test->backend_name(), "test");
    EXPECT_STREQ(retrieved_mpi->backend_name(), "MPI");

    registry.finalize_all();
}

}  // namespace dtl::test
