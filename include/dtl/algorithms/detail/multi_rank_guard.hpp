// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file multi_rank_guard.hpp
/// @brief Multi-rank guard for distributed algorithms requiring explicit communicator
/// @since 0.1.0

#pragma once

#include <stdexcept>
#include <string>

namespace dtl::detail {

/// @brief Guard that enforces explicit communicator for multi-rank containers
/// @details Algorithms that lack a distributed implementation must either
///          receive an explicit communicator parameter or operate on single-rank
///          containers. This guard throws if neither condition is met.
/// @param container The distributed container being operated on
/// @param api_name The API function name for the error message
/// @throws std::runtime_error if container has more than one rank
template <typename Container>
inline void require_collective_comm_or_single_rank(const Container& container,
                                                    const char* api_name) {
    if (container.num_ranks() > 1) {
        throw std::runtime_error(
            std::string(api_name) +
            " requires an explicit communicator when num_ranks()>1; "
            "use the overload with Comm&, or call local_* API for rank-local semantics.");
    }
}

}  // namespace dtl::detail
