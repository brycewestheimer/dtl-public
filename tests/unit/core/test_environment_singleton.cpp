// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_environment_singleton.cpp
/// @brief Unit tests for the RAII environment class (handle/view pattern)
/// @details Tests reference counting, lifecycle management, thread safety,
///          instance-based query methods, move semantics, and named domains.
///          Updated for V1.4.0 runtime_registry refactor.

#include <dtl/core/environment.hpp>
#include <dtl/core/environment_options.hpp>
#include <dtl/runtime/runtime_registry.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace dtl::test {

// Helper to query registry directly (avoids deprecated static methods)
static bool registry_initialized() {
    return runtime::runtime_registry::instance().is_initialized();
}

static size_t registry_ref_count() {
    return runtime::runtime_registry::instance().ref_count();
}

// =============================================================================
// Type Traits Tests
// =============================================================================

TEST(EnvironmentSingletonTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<environment>,
                  "environment must not be copy constructible");
    static_assert(!std::is_copy_assignable_v<environment>,
                  "environment must not be copy assignable");
    SUCCEED();
}

TEST(EnvironmentSingletonTest, MoveOnly) {
    static_assert(std::is_move_constructible_v<environment>,
                  "environment must be move constructible");
    static_assert(std::is_move_assignable_v<environment>,
                  "environment must be move assignable");
    SUCCEED();
}

// =============================================================================
// Pre-Initialization Queries
// =============================================================================

TEST(EnvironmentSingletonTest, StaticQueryBeforeInit) {
    // Before any environment is constructed, registry should not be initialized
    EXPECT_EQ(registry_ref_count(), size_t{0});
    EXPECT_FALSE(registry_initialized());
}

// =============================================================================
// Basic Construction and Destruction
// =============================================================================

TEST(EnvironmentSingletonTest, DefaultConstruction) {
    {
        environment env{environment_options::minimal()};
        EXPECT_TRUE(registry_initialized());
        EXPECT_EQ(registry_ref_count(), size_t{1});
    }
    EXPECT_FALSE(registry_initialized());
    EXPECT_EQ(registry_ref_count(), size_t{0});
}

TEST(EnvironmentSingletonTest, InstanceQueryAfterInit) {
    environment env{environment_options::minimal()};
    EXPECT_TRUE(registry_initialized());
}

TEST(EnvironmentSingletonTest, RefCountAfterInit) {
    environment env{environment_options::minimal()};
    EXPECT_EQ(registry_ref_count(), size_t{1});
}

// =============================================================================
// Reference Counting Tests
// =============================================================================

TEST(EnvironmentSingletonTest, RefCounting) {
    EXPECT_EQ(registry_ref_count(), size_t{0});

    auto env1 = std::make_unique<environment>(environment_options::minimal());
    EXPECT_EQ(registry_ref_count(), size_t{1});

    auto env2 = std::make_unique<environment>(environment_options::minimal());
    EXPECT_EQ(registry_ref_count(), size_t{2});

    env1.reset();
    EXPECT_EQ(registry_ref_count(), size_t{1});
    EXPECT_TRUE(registry_initialized());

    env2.reset();
    EXPECT_EQ(registry_ref_count(), size_t{0});
    EXPECT_FALSE(registry_initialized());
}

TEST(EnvironmentSingletonTest, MultipleInstances) {
    auto env1 = std::make_unique<environment>(environment_options::minimal());
    auto env2 = std::make_unique<environment>(environment_options::minimal());
    auto env3 = std::make_unique<environment>(environment_options::minimal());

    EXPECT_EQ(registry_ref_count(), size_t{3});
    EXPECT_TRUE(registry_initialized());
}

TEST(EnvironmentSingletonTest, DestructionDecrements) {
    auto env1 = std::make_unique<environment>(environment_options::minimal());
    auto env2 = std::make_unique<environment>(environment_options::minimal());

    EXPECT_EQ(registry_ref_count(), size_t{2});

    env2.reset();
    EXPECT_EQ(registry_ref_count(), size_t{1});
}

TEST(EnvironmentSingletonTest, LastDestructionFinalizes) {
    {
        environment env1{environment_options::minimal()};
        {
            environment env2{environment_options::minimal()};
            EXPECT_EQ(registry_ref_count(), size_t{2});
        }
        EXPECT_TRUE(registry_initialized());
        EXPECT_EQ(registry_ref_count(), size_t{1});
    }
    EXPECT_FALSE(registry_initialized());
    EXPECT_EQ(registry_ref_count(), size_t{0});
}

// =============================================================================
// Nesting and Scope Tests
// =============================================================================

TEST(EnvironmentSingletonTest, Nesting) {
    {
        environment outer{environment_options::minimal()};
        EXPECT_EQ(registry_ref_count(), size_t{1});

        {
            environment inner{environment_options::minimal()};
            EXPECT_EQ(registry_ref_count(), size_t{2});
        }

        EXPECT_TRUE(registry_initialized());
        EXPECT_EQ(registry_ref_count(), size_t{1});
    }
    EXPECT_FALSE(registry_initialized());
}

TEST(EnvironmentSingletonTest, NestedScopes) {
    {
        environment env1{environment_options::minimal()};
        {
            environment env2{environment_options::minimal()};
            {
                environment env3{environment_options::minimal()};
                EXPECT_EQ(registry_ref_count(), size_t{3});
            }
            EXPECT_EQ(registry_ref_count(), size_t{2});
            EXPECT_TRUE(registry_initialized());
        }
        EXPECT_EQ(registry_ref_count(), size_t{1});
        EXPECT_TRUE(registry_initialized());
    }
    EXPECT_EQ(registry_ref_count(), size_t{0});
    EXPECT_FALSE(registry_initialized());
}

// =============================================================================
// Re-initialization Tests
// =============================================================================

TEST(EnvironmentSingletonTest, ReInitAfterFullDestruct) {
    {
        environment env{environment_options::minimal()};
        EXPECT_TRUE(registry_initialized());
    }
    EXPECT_FALSE(registry_initialized());

    {
        environment env{environment_options::minimal()};
        EXPECT_TRUE(registry_initialized());
        EXPECT_EQ(registry_ref_count(), size_t{1});
    }
    EXPECT_FALSE(registry_initialized());
}

// =============================================================================
// Options Tests
// =============================================================================

TEST(EnvironmentSingletonTest, DefaultOptions) {
    environment env{environment_options::defaults()};
    EXPECT_TRUE(registry_initialized());
}

TEST(EnvironmentSingletonTest, CustomOptions) {
    environment_options opts = environment_options::minimal();
    opts.verbose = true;

    environment env{std::move(opts)};
    EXPECT_TRUE(registry_initialized());
}

TEST(EnvironmentSingletonTest, OptionsPreserved) {
    auto env1 = std::make_unique<environment>(environment_options::minimal());
    EXPECT_TRUE(registry_initialized());

    auto env2 = std::make_unique<environment>(environment_options::defaults());
    EXPECT_EQ(registry_ref_count(), size_t{2});

    env2.reset();
    env1.reset();
    EXPECT_FALSE(registry_initialized());
}

// =============================================================================
// Backend Query Tests (Instance-Based)
// =============================================================================

TEST(EnvironmentSingletonTest, HasMpiQuery) {
    environment env{environment_options::minimal()};
    EXPECT_FALSE(env.has_mpi());
}

TEST(EnvironmentSingletonTest, HasCudaQuery) {
    environment env{environment_options::minimal()};
    EXPECT_FALSE(env.has_cuda());
}

TEST(EnvironmentSingletonTest, MpiThreadLevel) {
    environment env{environment_options::minimal()};
    EXPECT_EQ(env.mpi_thread_level(), -1);
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(EnvironmentSingletonTest, MoveConstruction) {
    auto env1 = std::make_unique<environment>(environment_options::minimal());
    size_t count_before = registry_ref_count();

    // Move construct env2 from env1
    environment env2{std::move(*env1)};

    // env1 was moved from but still holds a registry ref until destroyed
    // env2 acquired a new registry ref in the move constructor
    // So total should be count_before + 1
    EXPECT_EQ(registry_ref_count(), count_before + 1);

    // Destroy the moved-from object
    env1.reset();
    EXPECT_EQ(registry_ref_count(), count_before);

    // env2 should still be valid
    EXPECT_TRUE(registry_initialized());
}

TEST(EnvironmentSingletonTest, MoveAssignment) {
    auto env1 = std::make_unique<environment>(environment_options::minimal());
    auto env2 = std::make_unique<environment>(environment_options::minimal());
    EXPECT_EQ(registry_ref_count(), size_t{2});

    // Move assign env1 into env2
    *env2 = std::move(*env1);

    // Both still hold registry refs
    EXPECT_EQ(registry_ref_count(), size_t{2});

    // Destroy moved-from
    env1.reset();
    EXPECT_EQ(registry_ref_count(), size_t{1});

    // env2 should still work
    EXPECT_TRUE(registry_initialized());

    env2.reset();
    EXPECT_FALSE(registry_initialized());
}

TEST(EnvironmentSingletonTest, MovedFromDoesNotCrash) {
    environment env1{environment_options::minimal()};
    environment env2{std::move(env1)};

    // env1 is moved-from; its destructor should not crash
    // (no double-free of comm, registry release is safe)
}

// =============================================================================
// Named Domain Tests
// =============================================================================

TEST(EnvironmentSingletonTest, DefaultDomainName) {
    environment env{environment_options::minimal()};
    EXPECT_EQ(env.domain(), "default");
}

TEST(EnvironmentSingletonTest, CustomDomainName) {
    environment_options opts = environment_options::minimal();
    opts.domain = "my_library";

    environment env{std::move(opts)};
    EXPECT_EQ(env.domain(), "my_library");
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(EnvironmentSingletonTest, ThreadSafety) {
    auto main_env = std::make_unique<environment>(environment_options::minimal());
    EXPECT_TRUE(registry_initialized());

    constexpr int num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([]() {
            environment thread_env{environment_options::minimal()};
            EXPECT_TRUE(registry_initialized());
            EXPECT_GE(registry_ref_count(), size_t{1});
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_TRUE(registry_initialized());
    EXPECT_EQ(registry_ref_count(), size_t{1});

    main_env.reset();
    EXPECT_FALSE(registry_initialized());
}

// =============================================================================
// Backend Detection Tests (Phase 12.5)
// =============================================================================

TEST(EnvironmentSingletonTest, HasShmemQuery) {
    environment env{environment_options::minimal()};
    EXPECT_FALSE(env.has_shmem());
}

TEST(EnvironmentSingletonTest, HasNcclQuery) {
    environment env{environment_options::minimal()};
    EXPECT_FALSE(env.has_nccl());
}

TEST(EnvironmentSingletonTest, HasHipQuery) {
    environment env{environment_options::minimal()};
    EXPECT_FALSE(env.has_hip());
}

TEST(EnvironmentSingletonTest, NcclOptionsDefault) {
    environment_options opts{};
    EXPECT_EQ(opts.nccl.ownership, backend_ownership::optional);
}

TEST(EnvironmentSingletonTest, HipOptionsDefault) {
    environment_options opts{};
    EXPECT_EQ(opts.hip.ownership, backend_ownership::optional);
    EXPECT_EQ(opts.hip.device_id, -1);
    EXPECT_FALSE(opts.hip.eager_context);
}

TEST(EnvironmentSingletonTest, BackwardCompatAlias) {
    static_assert(std::is_same_v<environment, environment_guard>,
                  "environment_guard must be an alias for environment");
    SUCCEED();
}

// =============================================================================
// Generic Thread Level Tests (Phase 1.2.2)
// =============================================================================

TEST(EnvironmentSingletonTest, ThreadSupportLevelEnum) {
    EXPECT_EQ(static_cast<int>(thread_support_level::single), 0);
    EXPECT_EQ(static_cast<int>(thread_support_level::funneled), 1);
    EXPECT_EQ(static_cast<int>(thread_support_level::serialized), 2);
    EXPECT_EQ(static_cast<int>(thread_support_level::multiple), 3);
}

TEST(EnvironmentSingletonTest, ThreadSupportLevelToString) {
    EXPECT_EQ(to_string(thread_support_level::single), "single");
    EXPECT_EQ(to_string(thread_support_level::funneled), "funneled");
    EXPECT_EQ(to_string(thread_support_level::serialized), "serialized");
    EXPECT_EQ(to_string(thread_support_level::multiple), "multiple");
}

TEST(EnvironmentSingletonTest, ThreadSupportLevelMpiConversion) {
    EXPECT_EQ(from_mpi_thread_level(0), thread_support_level::single);
    EXPECT_EQ(from_mpi_thread_level(1), thread_support_level::funneled);
    EXPECT_EQ(from_mpi_thread_level(2), thread_support_level::serialized);
    EXPECT_EQ(from_mpi_thread_level(3), thread_support_level::multiple);
    EXPECT_EQ(from_mpi_thread_level(-1), thread_support_level::single);

    EXPECT_EQ(to_mpi_thread_level(thread_support_level::single), 0);
    EXPECT_EQ(to_mpi_thread_level(thread_support_level::funneled), 1);
    EXPECT_EQ(to_mpi_thread_level(thread_support_level::serialized), 2);
    EXPECT_EQ(to_mpi_thread_level(thread_support_level::multiple), 3);
}

TEST(EnvironmentSingletonTest, GenericThreadLevel) {
    environment env{environment_options::minimal()};
    EXPECT_EQ(env.thread_level(), thread_support_level::single);
}

TEST(EnvironmentSingletonTest, GenericThreadLevelName) {
    environment env{environment_options::minimal()};
    EXPECT_EQ(env.thread_level_name(), "single");
}

TEST(EnvironmentSingletonTest, ThreadLevelForBackend) {
    environment env{environment_options::minimal()};

    EXPECT_EQ(env.thread_level_for_backend("mpi"),
              thread_support_level::single);
    EXPECT_EQ(env.thread_level_for_backend("cuda"),
              thread_support_level::single);
    EXPECT_EQ(env.thread_level_for_backend("unknown"),
              thread_support_level::single);
}

// =============================================================================
// Domain Options Test
// =============================================================================

TEST(EnvironmentSingletonTest, DomainOptionDefault) {
    environment_options opts{};
    EXPECT_EQ(opts.domain, "default");
}

}  // namespace dtl::test
