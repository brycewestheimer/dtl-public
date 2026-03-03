// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_executor_concept.cpp
/// @brief Unit tests for Executor concept
/// @details Verifies concept requirements using mock implementations.

#include <dtl/backend/concepts/executor.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <vector>

namespace dtl::test {

// =============================================================================
// Mock Executor Implementation
// =============================================================================

/// @brief Minimal mock executor that satisfies the Executor concept
class mock_executor {
public:
    /// @brief Execute callable immediately
    template <typename F>
    void execute(F&& f) {
        std::forward<F>(f)();
    }

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "mock";
    }
};

/// @brief Mock sync executor
class mock_sync_executor : public mock_executor {
public:
    /// @brief Always synchronous
    [[nodiscard]] static constexpr bool is_synchronous() noexcept {
        return true;
    }
};

/// @brief Mock parallel executor
class mock_parallel_executor : public mock_executor {
public:
    /// @brief Parallel for (sequential implementation for testing)
    template <typename F>
    void parallel_for(size_type count, F&& f) {
        for (size_type i = 0; i < count; ++i) {
            f(i);
        }
    }

    /// @brief Maximum parallelism
    [[nodiscard]] static constexpr size_type max_parallelism() noexcept {
        return 4;  // Mock value
    }

    /// @brief Suggested parallelism
    [[nodiscard]] static constexpr size_type suggested_parallelism() noexcept {
        return 2;
    }
};

/// @brief Mock bulk executor
class mock_bulk_executor : public mock_parallel_executor {
public:
    /// @brief Bulk execute with chunk boundaries
    template <typename F>
    void bulk_execute(size_type count, F&& f) {
        constexpr size_type chunk_size = 10;
        for (size_type i = 0; i < count; i += chunk_size) {
            size_type end = std::min(i + chunk_size, count);
            f(i, end);
        }
    }
};

/// @brief Type that doesn't satisfy Executor concept
struct not_an_executor {
    // Missing: execute, name
    void foo() {}
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(ExecutorConceptTest, MockExecutorSatisfiesConcept) {
    static_assert(Executor<mock_executor>);
}

TEST(ExecutorConceptTest, MockSyncSatisfiesConcept) {
    static_assert(Executor<mock_sync_executor>);
    static_assert(SyncExecutor<mock_sync_executor>);
}

TEST(ExecutorConceptTest, MockParallelSatisfiesConcept) {
    static_assert(Executor<mock_parallel_executor>);
    static_assert(ParallelExecutor<mock_parallel_executor>);
}

TEST(ExecutorConceptTest, MockBulkSatisfiesConcept) {
    static_assert(Executor<mock_bulk_executor>);
    static_assert(ParallelExecutor<mock_bulk_executor>);
    static_assert(BulkExecutor<mock_bulk_executor>);
}

TEST(ExecutorConceptTest, NonExecutorDoesNotSatisfy) {
    static_assert(!Executor<not_an_executor>);
    static_assert(!Executor<int>);
    static_assert(!Executor<void>);
}

TEST(ExecutorConceptTest, BasicExecutorDoesNotSatisfyParallel) {
    static_assert(!ParallelExecutor<mock_executor>);
}

// =============================================================================
// Standard Executor Tests
// =============================================================================

TEST(ExecutorConceptTest, InlineExecutorSatisfiesConcept) {
    static_assert(Executor<inline_executor>);
    static_assert(SyncExecutor<inline_executor>);
}

TEST(ExecutorConceptTest, SequentialExecutorSatisfiesConcept) {
    static_assert(Executor<sequential_executor>);
    static_assert(ParallelExecutor<sequential_executor>);
}

// =============================================================================
// Executor Properties Tests
// =============================================================================

TEST(ExecutorPropertiesTest, DefaultProperties) {
    executor_properties props;

    EXPECT_EQ(props.max_concurrency, 1);
    EXPECT_TRUE(props.in_order);
    EXPECT_FALSE(props.owns_threads);
    EXPECT_FALSE(props.supports_work_stealing);
}

// =============================================================================
// Execution Tests
// =============================================================================

TEST(ExecutorConceptTest, InlineExecutorExecutes) {
    inline_executor exec;

    bool executed = false;
    exec.execute([&]() { executed = true; });

    EXPECT_TRUE(executed);
}

TEST(ExecutorConceptTest, InlineExecutorIsSynchronous) {
    inline_executor exec;

    EXPECT_TRUE(exec.is_synchronous());
    EXPECT_STREQ(exec.name(), "inline");
}

TEST(ExecutorConceptTest, SequentialExecutorExecutes) {
    sequential_executor exec;

    bool executed = false;
    exec.execute([&]() { executed = true; });

    EXPECT_TRUE(executed);
    EXPECT_STREQ(exec.name(), "sequential");
}

TEST(ExecutorConceptTest, SequentialParallelFor) {
    sequential_executor exec;

    std::vector<int> results(10, 0);
    exec.parallel_for(10, [&](size_type i) {
        results[i] = static_cast<int>(i * 2);
    });

    for (size_type i = 0; i < 10; ++i) {
        EXPECT_EQ(results[i], static_cast<int>(i * 2));
    }
}

TEST(ExecutorConceptTest, SequentialExecutorParallelism) {
    sequential_executor exec;

    EXPECT_EQ(exec.max_parallelism(), 1);
    EXPECT_EQ(exec.suggested_parallelism(), 1);
}

TEST(ExecutorConceptTest, MockParallelFor) {
    mock_parallel_executor exec;

    std::vector<int> results(100, 0);
    exec.parallel_for(100, [&](size_type i) {
        results[i] = static_cast<int>(i);
    });

    for (size_type i = 0; i < 100; ++i) {
        EXPECT_EQ(results[i], static_cast<int>(i));
    }
}

TEST(ExecutorConceptTest, MockBulkExecute) {
    mock_bulk_executor exec;

    std::vector<std::pair<size_type, size_type>> chunks;
    exec.bulk_execute(35, [&](size_type start, size_type end) {
        chunks.emplace_back(start, end);
    });

    // Should have chunks: [0,10), [10,20), [20,30), [30,35)
    ASSERT_EQ(chunks.size(), 4);
    EXPECT_EQ(chunks[0], std::make_pair(size_type{0}, size_type{10}));
    EXPECT_EQ(chunks[1], std::make_pair(size_type{10}, size_type{20}));
    EXPECT_EQ(chunks[2], std::make_pair(size_type{20}, size_type{30}));
    EXPECT_EQ(chunks[3], std::make_pair(size_type{30}, size_type{35}));
}

// =============================================================================
// Tag Type Tests
// =============================================================================

TEST(ExecutorTagTest, TagsAreDistinct) {
    static_assert(!std::is_same_v<inline_executor_tag, thread_pool_executor_tag>);
    static_assert(!std::is_same_v<inline_executor_tag, single_thread_executor_tag>);
    static_assert(!std::is_same_v<inline_executor_tag, gpu_executor_tag>);
    static_assert(!std::is_same_v<thread_pool_executor_tag, gpu_executor_tag>);
}

TEST(ExecutorTagTest, InlineExecutorTag) {
    static_assert(std::is_same_v<inline_executor::tag_type, inline_executor_tag>);
}

TEST(ExecutorTagTest, SequentialExecutorTag) {
    static_assert(std::is_same_v<sequential_executor::tag_type, single_thread_executor_tag>);
}

// =============================================================================
// Traits Tests
// =============================================================================

TEST(ExecutorTraitsTest, DefaultTraits) {
    // Default traits for unknown executor type
    static_assert(!executor_traits<mock_executor>::is_sync);
    static_assert(!executor_traits<mock_executor>::is_parallel);
    static_assert(!executor_traits<mock_executor>::is_gpu);
    static_assert(executor_traits<mock_executor>::default_chunk_size == 1);
}

}  // namespace dtl::test
