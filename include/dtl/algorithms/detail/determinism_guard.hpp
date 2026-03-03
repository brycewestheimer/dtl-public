// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file determinism_guard.hpp
/// @brief Determinism policy guard helpers for distributed algorithms
/// @since 0.1.0

#pragma once

#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/runtime/runtime_registry.hpp>
#include <dtl/runtime/backend_discovery.hpp>

#include <string>
#include <string_view>

namespace dtl::detail {

[[nodiscard]] inline bool deterministic_mode_enabled() {
    return runtime::runtime_registry::instance().deterministic_mode_enabled();
}

[[nodiscard]] inline result<void> require_deterministic_collective_support(
    rank_t communicator_size,
    std::string_view api_name) {
    if (!deterministic_mode_enabled() || communicator_size <= 1) {
        return result<void>::success();
    }

    const auto mpi_desc = runtime::query_backend("mpi");
    const bool mpi_collectives_functional = runtime::has_capability(
        mpi_desc.functional_capabilities,
        runtime::backend_capability::collectives);

    if (mpi_collectives_functional) {
        return result<void>::success();
    }

    return make_error<void>(
        status_code::not_supported,
        std::string(api_name) +
            " deterministic mode requires a functional collective backend");
}

[[nodiscard]] inline bool deterministic_policy_requests_fixed_reduction_schedule() {
    const auto options = runtime::runtime_registry::instance().determinism_options_config();
    return options.mode == determinism_mode::deterministic &&
           options.reduction_schedule == reduction_schedule_policy::fixed_tree;
}

}  // namespace dtl::detail
