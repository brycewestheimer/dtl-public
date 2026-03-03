// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file transform.hpp
/// @brief Distributed transform algorithm
/// @details Apply transformation to elements of distributed containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <algorithm>
#include <functional>
#include <memory>

namespace dtl {

// ============================================================================
// Unary Transform
// ============================================================================

/// @brief Apply unary transformation to each element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source distributed container type
/// @tparam OutputContainer Destination distributed container type
/// @tparam UnaryOp Transformation function type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param op Unary transformation function
/// @return Result indicating success or failure
///
/// @par Complexity:
/// O(n/p) local transformations where n is global size and p is ranks.
///
/// @par Requirements:
/// - Input and output containers must have compatible partitioning
/// - Local sizes must match
///
/// @par Example:
/// @code
/// distributed_vector<int> src(1000, ctx);
/// distributed_vector<int> dst(1000, ctx);
/// dtl::transform(dtl::par{}, src, dst, [](int x) { return x * 2; });
/// @endcode
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<void> transform(ExecutionPolicy&& policy,
                       const InputContainer& input,
                       OutputContainer& output,
                       UnaryOp op) {
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    dispatch_transform(std::forward<ExecutionPolicy>(policy),
                       in_local.begin(), in_local.end(),
                       out_local.begin(), op);

    return {};
}

/// @brief In-place unary transformation
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam UnaryOp Transformation function type
/// @param policy Execution policy
/// @param container Container to transform in place
/// @param op Unary transformation function
/// @return Result indicating success or failure
template <typename ExecutionPolicy, typename Container, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> transform(ExecutionPolicy&& policy,
                       Container& container,
                       UnaryOp op) {
    auto local_v = container.local_view();
    dispatch_transform(std::forward<ExecutionPolicy>(policy),
                       local_v.begin(), local_v.end(),
                       local_v.begin(), op);
    return {};
}

/// @brief Unary transform with default execution
template <typename InputContainer, typename OutputContainer, typename UnaryOp>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<void> transform(const InputContainer& input,
                       OutputContainer& output,
                       UnaryOp op) {
    return transform(seq{}, input, output, std::move(op));
}

/// @brief In-place transform with default execution
template <typename Container, typename UnaryOp>
    requires DistributedContainer<Container>
result<void> transform(Container& container, UnaryOp op) {
    return transform(seq{}, container, std::move(op));
}

// ============================================================================
// Binary Transform
// ============================================================================

/// @brief Apply binary transformation to pairs of elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Input1Container First source container type
/// @tparam Input2Container Second source container type
/// @tparam OutputContainer Destination container type
/// @tparam BinaryOp Binary transformation function type
/// @param policy Execution policy
/// @param input1 First source container
/// @param input2 Second source container
/// @param output Destination container
/// @param op Binary transformation function
/// @return Result indicating success or failure
///
/// @par Requirements:
/// - All containers must have compatible partitioning
/// - Local sizes must match
///
/// @par Example:
/// @code
/// distributed_vector<int> a(1000, ctx), b(1000, ctx), c(1000, ctx);
/// dtl::transform(dtl::par{}, a, b, c, std::plus<>{});
/// @endcode
template <typename ExecutionPolicy, typename Input1Container,
          typename Input2Container, typename OutputContainer, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Input1Container> &&
             DistributedContainer<Input2Container> &&
             DistributedContainer<OutputContainer>
result<void> transform([[maybe_unused]] ExecutionPolicy&& policy,
                       const Input1Container& input1,
                       const Input2Container& input2,
                       OutputContainer& output,
                       BinaryOp op) {
    auto in1_local = input1.local_view();
    auto in2_local = input2.local_view();
    auto out_local = output.local_view();

    // Unified dispatch: both seq and par use the same element pairing logic.
    // The execution policy affects parallelism, not correctness.
    std::transform(in1_local.begin(), in1_local.end(),
                   in2_local.begin(),
                   out_local.begin(), op);

    return {};
}

/// @brief Binary transform with default execution
template <typename Input1Container, typename Input2Container,
          typename OutputContainer, typename BinaryOp>
    requires DistributedContainer<Input1Container> &&
             DistributedContainer<Input2Container> &&
             DistributedContainer<OutputContainer>
result<void> transform(const Input1Container& input1,
                       const Input2Container& input2,
                       OutputContainer& output,
                       BinaryOp op) {
    return transform(seq{}, input1, input2, output, std::move(op));
}

// ============================================================================
// Segmented Transform
// ============================================================================

/// @brief Transform using segmented iteration pattern
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam UnaryOp Transformation function type
/// @param policy Execution policy
/// @param container Container to transform in place
/// @param op Transformation function
/// @return Result indicating success or failure
///
/// @par Design Rationale:
/// Uses segmented iteration for predictable bulk operations.
template <typename ExecutionPolicy, typename Container, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> segmented_transform(ExecutionPolicy&& policy,
                                  Container& container,
                                  UnaryOp op) {
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            dispatch_transform(std::forward<ExecutionPolicy>(policy),
                               segment.begin(), segment.end(),
                               segment.begin(), op);
        }
    }
    return {};
}

// ============================================================================
// Local-only transform (no communication)
// ============================================================================

/// @brief Transform local partition only (no communication)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam UnaryOp Transformation function type
/// @param policy Execution policy
/// @param container Container to transform
/// @param op Transformation function
/// @return Local iterator past last transformed element
///
/// @note NOT collective - transforms local data only.
template <typename ExecutionPolicy, typename Container, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto local_transform(ExecutionPolicy&& policy, Container& container, UnaryOp op) {
    auto local_v = container.local_view();
    return dispatch_transform(std::forward<ExecutionPolicy>(policy),
                              local_v.begin(), local_v.end(),
                              local_v.begin(), op);
}

/// @brief Local transform with default sequential execution
template <typename Container, typename UnaryOp>
    requires DistributedContainer<Container>
auto local_transform(Container& container, UnaryOp op) {
    return local_transform(seq{}, container, std::move(op));
}

// ============================================================================
// Async transform
// ============================================================================

/// @brief Asynchronously transform elements
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam UnaryOp Transformation function type
/// @param input Source container
/// @param output Destination container
/// @param op Transformation function
/// @return Future indicating completion
template <typename InputContainer, typename OutputContainer, typename UnaryOp>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
auto async_transform(const InputContainer& input,
                     OutputContainer& output,
                     UnaryOp op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = transform(seq{}, input, output, std::move(op));
        if (result) {
            promise->set_value();
        } else {
            promise->set_error(result.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously transform elements in-place
/// @tparam Container Distributed container type
/// @tparam UnaryOp Transformation function type
/// @param container Container to transform
/// @param op Transformation function
/// @return Future indicating completion
template <typename Container, typename UnaryOp>
    requires DistributedContainer<Container>
auto async_transform(Container& container, UnaryOp op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = transform(seq{}, container, std::move(op));
        if (result) {
            promise->set_value();
        } else {
            promise->set_error(result.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
