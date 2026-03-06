// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file placement_mapping.hpp
/// @brief Maps C ABI placement enums to C++ placement policy types
/// @since 0.1.0

#pragma once

#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/core/config.hpp>

namespace dtl::c::detail {

/// @brief Result of placement mapping: which C++ placement policy to use
enum class mapped_placement {
    host,
    device,
    unified,
    not_supported
};

/// @brief Map a C ABI placement enum to a dispatch tag
constexpr mapped_placement map_placement(dtl_placement_policy p) noexcept {
    switch (p) {
        case DTL_PLACEMENT_HOST:
            return mapped_placement::host;
        case DTL_PLACEMENT_DEVICE:
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
            return mapped_placement::device;
#else
            return mapped_placement::not_supported;
#endif
        case DTL_PLACEMENT_UNIFIED:
#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP
            return mapped_placement::unified;
#else
            return mapped_placement::not_supported;
#endif
        case DTL_PLACEMENT_DEVICE_PREFERRED:
        default:
            return mapped_placement::not_supported;
    }
}

/// @brief Check if a C ABI placement requires CUDA context
constexpr bool placement_requires_cuda(dtl_placement_policy p) noexcept {
    return p == DTL_PLACEMENT_DEVICE || p == DTL_PLACEMENT_UNIFIED;
}

}  // namespace dtl::c::detail
