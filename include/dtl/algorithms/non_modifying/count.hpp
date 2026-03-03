// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file count.hpp
/// @brief Distributed count algorithms
/// @details Count elements matching criteria in distributed containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/detail/determinism_guard.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <memory>
#include <stdexcept>

namespace dtl {

namespace detail {

template <typename T>
T unwrap_or_throw(result<T>&& api_result, const char* api_name) {
    if (api_result.has_value()) {
        return std::move(api_result).value();
    }
    const auto& error = api_result.error();
    throw std::runtime_error(
        std::string(api_name) + " failed with status " +
        std::string(status_code_name(error.code())) + ": " + error.message());
}

[[nodiscard]] inline status_code map_collective_exception(const std::exception& ex) {
    const std::string message = ex.what();
    if (message.find("backend") != std::string::npos ||
        message.find("Backend") != std::string::npos) {
        return status_code::backend_invalid;
    }
    return status_code::collective_failure;
}

}  // namespace detail

// ============================================================================
// Distributed count
// ============================================================================

/// @brief Count elements equal to value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container The distributed container
/// @param value Value to count
/// @return Total count across all ranks
///
/// @par Complexity:
/// O(n/p) local comparisons, plus O(log p) allreduce communication.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result is the global count, consistent across all ranks.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// size_type total = dtl::count(dtl::par{}, vec, 0);
/// std::cout << "Found " << total << " zeros globally\n";
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type count([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, const T& value) {
    if (container.num_ranks() > 1) {
        throw std::runtime_error(
            "dtl::count requires an explicit communicator when num_ranks()>1; "
            "use dtl::global_count(..., comm) or dtl::local_count(...) for local semantics.");
    }

    // Count locally
    size_type local_count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (*it == value) {
            ++local_count;
        }
    }

    // Standalone mode: return local count only
    // For distributed count, use overload with communicator parameter
    return local_count;
}

/// @brief Count elements equal to value (default execution)
template <typename Container, typename T>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("count(container, value) is not collective for multi-rank containers; use count(..., comm) for global semantics or local_count(...) for local semantics")
size_type count(const Container& container, const T& value) {
    return count(seq{}, container, value);
}

// ============================================================================
// Communicator-Aware Distributed Count
// ============================================================================

/// @brief Count elements equal to value with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param value Value to count
/// @param comm The communicator for allreduce
/// @return Total global count across all ranks
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result is consistent across all ranks after allreduce.
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> count_result([[maybe_unused]] ExecutionPolicy&& policy,
                               const Container& container,
                               const T& value,
                               Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::count");
        !deterministic_guard) {
        return make_error<size_type>(deterministic_guard.error().code(),
                                     deterministic_guard.error().message());
    }

    try {
        size_type local_count = 0;
        auto local_view = container.local_view();
        for (auto it = local_view.begin(); it != local_view.end(); ++it) {
            if (*it == value) {
                ++local_count;
            }
        }

        return result<size_type>::success(
            comm.template allreduce_sum_value<size_type>(local_count));
    } catch (const std::exception& ex) {
        return make_error<size_type>(
            detail::map_collective_exception(ex),
            std::string("count_result collective communication failed: ") + ex.what());
    } catch (...) {
        return make_error<size_type>(
            status_code::collective_failure,
            "count_result collective communication failed with unknown exception");
    }
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
size_type count([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, const T& value, Comm& comm) {
    return detail::unwrap_or_throw(
        count_result(std::forward<ExecutionPolicy>(policy), container, value, comm),
        "dtl::count");
}

/// @brief Count elements equal to value with communicator (default execution)
template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
size_type count(const Container& container, const T& value, Comm& comm) {
    return count(seq{}, container, value, comm);
}

// ============================================================================
// Distributed count_if
// ============================================================================

/// @brief Count elements satisfying predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Total count across all ranks
///
/// @par Example:
/// @code
/// size_type positive = dtl::count_if(dtl::par{}, vec,
///     [](int x) { return x > 0; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type count_if([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    if (container.num_ranks() > 1) {
        throw std::runtime_error(
            "dtl::count_if requires an explicit communicator when num_ranks()>1; "
            "use dtl::global_count_if(..., comm) or dtl::local_count_if(...) for local semantics.");
    }

    size_type local_count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            ++local_count;
        }
    }

    // Standalone mode: return local count only
    // For distributed count, use overload with communicator parameter
    return local_count;
}

/// @brief Count elements satisfying predicate (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("count_if(container, pred) is not collective for multi-rank containers; use count_if(..., comm) for global semantics or local_count_if(...) for local semantics")
size_type count_if(const Container& container, Predicate pred) {
    return count_if(seq{}, container, std::move(pred));
}

/// @brief Count elements satisfying predicate with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @param comm The communicator for allreduce
/// @return Total global count across all ranks
template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> count_if_result([[maybe_unused]] ExecutionPolicy&& policy,
                                  const Container& container,
                                  Predicate pred,
                                  Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::count_if");
        !deterministic_guard) {
        return make_error<size_type>(deterministic_guard.error().code(),
                                     deterministic_guard.error().message());
    }

    try {
        size_type local_count = 0;
        auto local_view = container.local_view();
        for (auto it = local_view.begin(); it != local_view.end(); ++it) {
            if (pred(*it)) {
                ++local_count;
            }
        }

        return result<size_type>::success(
            comm.template allreduce_sum_value<size_type>(local_count));
    } catch (const std::exception& ex) {
        return make_error<size_type>(
            detail::map_collective_exception(ex),
            std::string("count_if_result collective communication failed: ") + ex.what());
    } catch (...) {
        return make_error<size_type>(
            status_code::collective_failure,
            "count_if_result collective communication failed with unknown exception");
    }
}

template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
size_type count_if([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    return detail::unwrap_or_throw(
        count_if_result(std::forward<ExecutionPolicy>(policy), container, std::move(pred), comm),
        "dtl::count_if");
}

/// @brief Count elements satisfying predicate with communicator (default execution)
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
size_type count_if(const Container& container, Predicate pred, Comm& comm) {
    return count_if(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Explicit global_* root APIs (Phase 02 semantic split)
// ============================================================================

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> global_count_result(ExecutionPolicy&& policy,
                                      const Container& container,
                                      const T& value,
                                      Comm& comm) {
    return count_result(std::forward<ExecutionPolicy>(policy), container, value, comm);
}

template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> global_count_result(const Container& container,
                                      const T& value,
                                      Comm& comm) {
    return global_count_result(seq{}, container, value, comm);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
size_type global_count(ExecutionPolicy&& policy,
                       const Container& container,
                       const T& value,
                       Comm& comm) {
    return detail::unwrap_or_throw(
        global_count_result(std::forward<ExecutionPolicy>(policy), container, value, comm),
        "dtl::global_count");
}

template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
size_type global_count(const Container& container,
                       const T& value,
                       Comm& comm) {
    return global_count(seq{}, container, value, comm);
}

template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> global_count_if_result(ExecutionPolicy&& policy,
                                         const Container& container,
                                         Predicate pred,
                                         Comm& comm) {
    return count_if_result(std::forward<ExecutionPolicy>(policy), container, std::move(pred), comm);
}

template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
result<size_type> global_count_if_result(const Container& container,
                                         Predicate pred,
                                         Comm& comm) {
    return global_count_if_result(seq{}, container, std::move(pred), comm);
}

template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
size_type global_count_if(ExecutionPolicy&& policy,
                          const Container& container,
                          Predicate pred,
                          Comm& comm) {
    return detail::unwrap_or_throw(
        global_count_if_result(std::forward<ExecutionPolicy>(policy), container, std::move(pred), comm),
        "dtl::global_count_if");
}

template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
size_type global_count_if(const Container& container,
                          Predicate pred,
                          Comm& comm) {
    return global_count_if(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Local-only count (no communication)
// ============================================================================

/// @brief Count in local partition only (no communication)
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container The distributed container
/// @param value Value to count
/// @return Local count only
///
/// @note NOT collective - returns local count only.
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_count(const Container& container, const T& value) {
    size_type count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (*it == value) {
            ++count;
        }
    }
    return count;
}

/// @brief Count in local partition with predicate (no communication)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
size_type local_count_if(const Container& container, Predicate pred) {
    size_type count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            ++count;
        }
    }
    return count;
}

// ============================================================================
// Async count
// ============================================================================

/// @brief Asynchronously count elements
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container The distributed container
/// @param value Value to count
/// @return Future containing the count
///
/// @note Returns immediately; use .get() to await result.
template <typename Container, typename T>
    requires DistributedContainer<Container>
auto async_count(const Container& container, const T& value)
    -> futures::distributed_future<size_type> {
    auto promise = std::make_shared<futures::distributed_promise<size_type>>();
    auto future = promise->get_future();

    try {
        auto result = count(seq{}, container, value);
        promise->set_value(result);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously count elements satisfying predicate
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Future containing the count
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto async_count_if(const Container& container, Predicate pred)
    -> futures::distributed_future<size_type> {
    auto promise = std::make_shared<futures::distributed_promise<size_type>>();
    auto future = promise->get_future();

    try {
        auto result = count_if(seq{}, container, std::move(pred));
        promise->set_value(result);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
