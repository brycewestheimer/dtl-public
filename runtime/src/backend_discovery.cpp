// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file backend_discovery.cpp
/// @brief Implementation of runtime backend discovery service
/// @details Queries the runtime_registry singleton and combines its runtime
///          availability data with compile-time backend trait information.
/// @since 0.1.0

#include <dtl/runtime/backend_discovery.hpp>
#include <dtl/runtime/runtime_registry.hpp>
#include <dtl/backend/common/backend_traits.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace dtl::runtime {

namespace {

/// @brief Build capability bitmask from compile-time backend traits
template <typename BackendTag>
constexpr backend_capability traits_to_capabilities() {
    using traits = dtl::backend_traits<BackendTag>;
    auto caps = backend_capability::none;

    if constexpr (traits::supports_point_to_point)
        caps = caps | backend_capability::point_to_point;
    if constexpr (traits::supports_collectives)
        caps = caps | backend_capability::collectives;
    if constexpr (traits::supports_rma)
        caps = caps | backend_capability::rma;
    if constexpr (traits::supports_gpu_aware)
        caps = caps | backend_capability::gpu_aware;
    if constexpr (traits::supports_async)
        caps = caps | backend_capability::async_operations;
    if constexpr (traits::supports_thread_multiple)
        caps = caps | backend_capability::thread_multiple;
    if constexpr (traits::supports_rdma)
        caps = caps | backend_capability::rdma;

    return caps;
}

backend_maturity to_runtime_maturity(dtl::backend_maturity maturity) {
    switch (maturity) {
        case dtl::backend_maturity::production:
            return backend_maturity::production;
        case dtl::backend_maturity::partial:
            return backend_maturity::partial;
        case dtl::backend_maturity::stub:
        default:
            return backend_maturity::stub;
    }
}

std::vector<capability_descriptor> build_capability_levels(
    backend_capability compiled_caps,
    backend_capability runtime_caps,
    backend_capability functional_caps) {
    static constexpr std::array<backend_capability, 9> kCapabilities = {
        backend_capability::point_to_point,
        backend_capability::collectives,
        backend_capability::rma,
        backend_capability::gpu_aware,
        backend_capability::async_operations,
        backend_capability::thread_multiple,
        backend_capability::rdma,
        backend_capability::device_execution,
        backend_capability::memory_management,
    };

    std::vector<capability_descriptor> levels;
    levels.reserve(kCapabilities.size());

    for (auto cap : kCapabilities) {
        capability_level level = capability_level::unavailable;
        if (has_capability(compiled_caps, cap)) {
            level = capability_level::compiled;
        }
        if (has_capability(runtime_caps, cap)) {
            level = capability_level::runtime_available;
        }
        if (has_capability(functional_caps, cap)) {
            level = capability_level::functional;
        }
        levels.push_back(capability_descriptor{cap, level});
    }

    return levels;
}

backend_capability functional_capabilities_for_backend(
    std::string_view backend_name,
    backend_maturity maturity,
    bool runtime_available,
    backend_capability runtime_caps) {
    if (!runtime_available) {
        return backend_capability::none;
    }

    if (maturity == backend_maturity::stub) {
        return backend_capability::none;
    }

    if (maturity == backend_maturity::production) {
        return runtime_caps;
    }

    if (backend_name == "CUDA" || backend_name == "HIP") {
        return runtime_caps & (backend_capability::gpu_aware
                             | backend_capability::async_operations
                             | backend_capability::thread_multiple
                             | backend_capability::device_execution
                             | backend_capability::memory_management);
    }

    if (backend_name == "NCCL") {
        return runtime_caps & (backend_capability::point_to_point
                             | backend_capability::collectives
                             | backend_capability::gpu_aware
                             | backend_capability::async_operations
                             | backend_capability::rdma);
    }

    return runtime_caps;
}

void finalize_descriptor(
    backend_descriptor& desc,
    backend_capability static_caps,
    backend_capability extra_caps,
    bool runtime_available,
    bool compiled,
    backend_maturity maturity) {
    const auto declared_caps = static_caps | extra_caps;
    const auto compiled_caps = compiled ? declared_caps : backend_capability::none;
    const auto runtime_caps = (compiled && runtime_available)
        ? compiled_caps
        : backend_capability::none;
    const auto functional_caps = functional_capabilities_for_backend(
        desc.name, maturity, runtime_available, runtime_caps);

    desc.compiled = compiled;
    desc.available = runtime_available;
    desc.maturity = maturity;
    desc.compiled_capabilities = compiled_caps;
    desc.runtime_capabilities = runtime_caps;
    desc.functional_capabilities = functional_caps;
    desc.capabilities = functional_caps;
    desc.capability_levels = build_capability_levels(
        compiled_caps, runtime_caps, functional_caps);
}

/// @brief Convert string to lowercase for case-insensitive comparison
std::string to_lower(std::string_view sv) {
    std::string result(sv);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

}  // anonymous namespace

std::vector<backend_descriptor> available_backends() {
    auto& reg = runtime_registry::instance();
    std::vector<backend_descriptor> result;
    result.reserve(8);

    // MPI
    {
        backend_descriptor desc;
        desc.name = "MPI";
        const bool runtime_available = reg.has_mpi();
#if DTL_ENABLE_MPI
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::mpi_backend_tag>(),
            backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::mpi_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // CUDA
    {
        backend_descriptor desc;
        desc.name = "CUDA";
        const bool runtime_available = reg.has_cuda();
#if DTL_ENABLE_CUDA
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::cuda_backend_tag>(),
            backend_capability::device_execution | backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::cuda_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // HIP
    {
        backend_descriptor desc;
        desc.name = "HIP";
        const bool runtime_available = reg.has_hip();
#if DTL_ENABLE_HIP
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::hip_backend_tag>(),
            backend_capability::device_execution | backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::hip_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // NCCL
    {
        backend_descriptor desc;
        desc.name = "NCCL";
        const bool runtime_available = reg.has_nccl();
#if DTL_ENABLE_NCCL
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::nccl_backend_tag>(),
            backend_capability::none,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::nccl_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // SHMEM
    {
        backend_descriptor desc;
        desc.name = "SHMEM";
        const bool runtime_available = reg.has_shmem();
#if DTL_ENABLE_SHMEM
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::shmem_backend_tag>(),
            backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::shmem_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // UCX
    {
        backend_descriptor desc;
        desc.name = "UCX";
        const bool runtime_available = false;
#if DTL_ENABLE_UCX
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::ucx_backend_tag>(),
            backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::ucx_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // GASNet-EX
    {
        backend_descriptor desc;
        desc.name = "GASNet";
        const bool runtime_available = false;
#if DTL_ENABLE_GASNET
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::gasnet_backend_tag>(),
            backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::gasnet_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    // SYCL
    {
        backend_descriptor desc;
        desc.name = "SYCL";
        const bool runtime_available = false;
#if DTL_ENABLE_SYCL
        const bool compiled = true;
    #else
        const bool compiled = false;
#endif
        finalize_descriptor(
            desc,
            traits_to_capabilities<dtl::sycl_backend_tag>(),
            backend_capability::device_execution | backend_capability::memory_management,
            runtime_available,
            compiled,
            to_runtime_maturity(dtl::backend_traits<dtl::sycl_backend_tag>::maturity));
        result.push_back(std::move(desc));
    }

    return result;
}

backend_descriptor query_backend(std::string_view name) {
    auto lower_name = to_lower(name);
    auto backends = available_backends();

    for (auto& desc : backends) {
        if (to_lower(desc.name) == lower_name) {
            return desc;
        }
    }

    // Unknown backend
    backend_descriptor unknown;
    unknown.name = std::string(name);
    unknown.available = false;
    unknown.compiled = false;
    unknown.maturity = backend_maturity::stub;
    unknown.compiled_capabilities = backend_capability::none;
    unknown.runtime_capabilities = backend_capability::none;
    unknown.functional_capabilities = backend_capability::none;
    unknown.capabilities = backend_capability::none;
    unknown.capability_levels = build_capability_levels(
        backend_capability::none,
        backend_capability::none,
        backend_capability::none);
    return unknown;
}

bool has_any_gpu_backend() noexcept {
    auto& reg = runtime_registry::instance();
    return reg.has_cuda() || reg.has_hip() || reg.has_nccl();
}

bool has_any_comm_backend() noexcept {
    auto& reg = runtime_registry::instance();
    return reg.has_mpi() || reg.has_nccl() || reg.has_shmem();
}

}  // namespace dtl::runtime
