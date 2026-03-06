// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file context.hpp
/// @brief Multi-domain context template for DTL
/// @details Provides the variadic context<Domains...> template that binds multiple
///          backend domains (MPI, CUDA, NCCL, etc.) into a single type-safe handle.
///          Follows the context-bound backends design pattern.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/error/result.hpp>
#include <dtl/handle/handle.hpp>

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

namespace dtl {

// Forward declaration for use in detail namespace
template <typename... Domains>
class context;

// =============================================================================
// Context Implementation Helpers
// =============================================================================

namespace detail {

/// @brief Check if a type is in a parameter pack
template <typename T, typename... Ts>
struct contains;

template <typename T>
struct contains<T> : std::false_type {};

template <typename T, typename First, typename... Rest>
struct contains<T, First, Rest...>
    : std::conditional_t<std::is_same_v<T, First>,
                         std::true_type,
                         contains<T, Rest...>> {};

template <typename T, typename... Ts>
inline constexpr bool contains_v = contains<T, Ts...>::value;

/// @brief Get index of type in parameter pack
template <typename T, typename... Ts>
struct index_of;

template <typename T, typename First, typename... Rest>
struct index_of<T, First, Rest...>
    : std::integral_constant<size_type,
          std::is_same_v<T, First> ? 0 : 1 + index_of<T, Rest...>::value> {};

template <typename T>
struct index_of<T> : std::integral_constant<size_type, 0> {};

template <typename T, typename... Ts>
inline constexpr size_type index_of_v = index_of<T, Ts...>::value;

/// @brief Append a type to a type list
template <typename Context, typename NewDomain>
struct append_domain;

template <typename... Domains, typename NewDomain>
struct append_domain<context<Domains...>, NewDomain> {
    using type = context<Domains..., NewDomain>;
};

template <typename Context, typename NewDomain>
using append_domain_t = typename append_domain<Context, NewDomain>::type;

}  // namespace detail

// =============================================================================
// Multi-Domain Context
// =============================================================================

/// @brief Multi-domain context binding communication, execution, and memory domains
/// @tparam Domains... Domain types (mpi_domain, cpu_domain, cuda_domain, etc.)
/// @details The context provides type-safe access to multiple backend domains and
///          serves as the primary interface for distributed operations. Users
///          construct contexts from an environment and use them with containers
///          and algorithms.
///
/// @par Key Features:
/// - Compile-time domain queries via has<Domain>()
/// - Type-safe domain access via get<Domain>()
/// - Convenience queries (rank(), size(), is_root()) from primary comm domain
/// - Factory operations to create derived contexts (split, with_cuda, with_nccl)
///
/// @par Example Usage:
/// @code
/// dtl::environment env(argc, argv);
/// auto ctx = env.make_world_context();  // context<mpi_domain, cpu_domain>
///
/// // Query rank/size
/// if (ctx.is_root()) {
///     std::cout << "Running on " << ctx.size() << " ranks\n";
/// }
///
/// // Access specific domains
/// auto& mpi = ctx.get<mpi_domain>();
/// mpi.barrier();
///
/// // Add CUDA domain
/// auto gpu_ctx = ctx.with_cuda(0);  // context<mpi_domain, cpu_domain, cuda_domain>
/// auto& cuda = gpu_ctx.get<cuda_domain>();
/// @endcode
///
template <typename... Domains>
class context {
public:
    /// @brief Number of domains in this context
    static constexpr size_type domain_count = sizeof...(Domains);

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Default constructor (default-constructs all domains)
    context() = default;

    /// @brief Construct with explicit domain instances
    /// @param domains Domain instances to store
    explicit context(Domains... domains)
        : domains_(std::move(domains)...) {}

    /// @brief Construct from tuple of domains
    /// @param domains Tuple containing domain instances
    explicit context(std::tuple<Domains...> domains)
        : domains_(std::move(domains)) {}

    // Copy and move
    context(const context&) = default;
    context& operator=(const context&) = default;
    context(context&&) noexcept = default;
    context& operator=(context&&) noexcept = default;

    // =========================================================================
    // Compile-Time Domain Queries
    // =========================================================================

    /// @brief Check if context contains a specific domain type
    /// @tparam D Domain type to check for
    /// @return true if D is in Domains..., false otherwise
    template <typename D>
    [[nodiscard]] static constexpr bool has() noexcept {
        return detail::contains_v<D, Domains...>;
    }

    /// @brief Check if context has any communication domain
    [[nodiscard]] static constexpr bool has_communication_domain() noexcept {
        return (CommunicationDomain<Domains> || ...);
    }

    /// @brief Check if context has MPI domain
    [[nodiscard]] static constexpr bool has_mpi() noexcept {
        return has<mpi_domain>();
    }

    /// @brief Check if context has CUDA domain
    [[nodiscard]] static constexpr bool has_cuda() noexcept {
        return has<cuda_domain>();
    }

    /// @brief Check if context has NCCL domain
    [[nodiscard]] static constexpr bool has_nccl() noexcept {
        return has<nccl_domain>();
    }

    /// @brief Check if context has SHMEM domain
    [[nodiscard]] static constexpr bool has_shmem() noexcept {
        return has<shmem_domain>();
    }

    /// @brief Check if context has CPU domain
    [[nodiscard]] static constexpr bool has_cpu() noexcept {
        return has<cpu_domain>();
    }

    // =========================================================================
    // Domain Access
    // =========================================================================

    /// @brief Get reference to a specific domain
    /// @tparam D Domain type to retrieve
    /// @return Reference to the domain
    /// @pre has<D>() must be true
    template <typename D>
        requires (has<D>())
    [[nodiscard]] D& get() noexcept {
        return std::get<D>(domains_);
    }

    /// @brief Get const reference to a specific domain
    /// @tparam D Domain type to retrieve
    /// @return Const reference to the domain
    /// @pre has<D>() must be true
    template <typename D>
        requires (has<D>())
    [[nodiscard]] const D& get() const noexcept {
        return std::get<D>(domains_);
    }

    // =========================================================================
    // Convenience Queries (from primary communication domain)
    // =========================================================================

    /// @brief Get this process's rank from primary communication domain
    /// @return Rank in the communicator, or 0 if no comm domain
    [[nodiscard]] rank_t rank() const noexcept {
        if constexpr (has_mpi()) {
            return std::get<mpi_domain>(domains_).rank();
        } else if constexpr (has_shmem()) {
            return std::get<shmem_domain>(domains_).rank();
        } else if constexpr (has_nccl()) {
            return std::get<nccl_domain>(domains_).rank();
        } else if constexpr (has_cpu()) {
            return std::get<cpu_domain>(domains_).rank();
        } else {
            return 0;
        }
    }

    /// @brief Get total number of ranks from primary communication domain
    /// @return Number of ranks, or 1 if no comm domain
    [[nodiscard]] rank_t size() const noexcept {
        if constexpr (has_mpi()) {
            return std::get<mpi_domain>(domains_).size();
        } else if constexpr (has_shmem()) {
            return std::get<shmem_domain>(domains_).size();
        } else if constexpr (has_nccl()) {
            return std::get<nccl_domain>(domains_).size();
        } else if constexpr (has_cpu()) {
            return std::get<cpu_domain>(domains_).size();
        } else {
            return 1;
        }
    }

    /// @brief Check if this is the root rank (rank 0)
    [[nodiscard]] bool is_root() const noexcept {
        return rank() == 0;
    }

    /// @brief Check if context is valid (all domains are valid)
    [[nodiscard]] bool valid() const noexcept {
        return std::apply([](const auto&... d) {
            return (d.valid() && ...);
        }, domains_);
    }

    [[nodiscard]] handle::comm_handle communicator_handle() const {
        auto runtime = handle::runtime_handle::current();

        if constexpr (has_mpi()) {
            const auto adapter = std::get<mpi_domain>(domains_).adapter_handle();
            if (!adapter) {
                return handle::comm_handle::unbound(rank(), size(), std::move(runtime));
            }

            auto barrier_fn = [adapter]() -> result<void> {
                try {
                    adapter->barrier();
                    return result<void>::success();
                } catch (const std::exception& e) {
                    return result<void>::failure(
                        status{status_code::barrier_failed, no_rank,
                               std::string("MPI barrier failed: ") + e.what()});
                }
            };

            return handle::comm_handle{rank(), size(), std::move(barrier_fn), std::move(runtime)};
        }

        return handle::comm_handle::unbound(rank(), size(), std::move(runtime));
    }

    [[nodiscard]] handle::context_handle handle() const {
        auto runtime = handle::runtime_handle::current();
        auto determinism = runtime::runtime_registry::instance().determinism_options_config();
        return handle::context_handle{communicator_handle(), std::move(runtime), determinism};
    }

    // =========================================================================
    // Domain-Specific Queries
    // =========================================================================

    /// @brief Get rank from a specific communication domain
    /// @tparam D Communication domain type
    /// @return Rank in that domain
    template <CommunicationDomain D>
        requires (has<D>())
    [[nodiscard]] rank_t rank() const noexcept {
        return std::get<D>(domains_).rank();
    }

    /// @brief Get size from a specific communication domain
    /// @tparam D Communication domain type
    /// @return Size in that domain
    template <CommunicationDomain D>
        requires (has<D>())
    [[nodiscard]] rank_t size() const noexcept {
        return std::get<D>(domains_).size();
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// @brief Barrier synchronization on the primary communication domain
    void barrier() {
        if constexpr (has_mpi()) {
            std::get<mpi_domain>(domains_).barrier();
        } else if constexpr (has_shmem()) {
            std::get<shmem_domain>(domains_).barrier();
        }
        // CPU and execution-only domains don't need barriers
    }

    /// @brief Synchronize CUDA stream (if CUDA domain present)
    void synchronize_device() {
        if constexpr (has_cuda()) {
            std::get<cuda_domain>(domains_).synchronize();
        }
    }

    // =========================================================================
    // Factory Operations
    // =========================================================================

    /// @brief Split MPI communicator to create new context
    /// @param color Color for grouping (ranks with same color in same group)
    /// @param key Ordering key within color group (default 0)
    /// @return Result containing new context with split MPI domain
    /// @pre has_mpi() must be true
    [[nodiscard]] result<context> split_mpi(int color, int key = 0) const
        requires (has_mpi())
    {
        auto split_result = std::get<mpi_domain>(domains_).split(color, key);
        if (!split_result) {
            return result<context>::failure(split_result.error());
        }

        // Build new domain tuple with split MPI domain
        auto new_domains = replace_domain<mpi_domain>(std::move(*split_result));
        return result<context>::success(context(std::move(new_domains)));
    }

    /// @brief Add CUDA domain to create new context
    /// @param device_id CUDA device ID (default 0)
    /// @return New context with additional cuda_domain
    [[nodiscard]] auto with_cuda(int device_id = 0) const
        -> detail::append_domain_t<context, cuda_domain>
    {
        using new_context_t = detail::append_domain_t<context, cuda_domain>;
        return new_context_t(std::tuple_cat(domains_,
                                            std::make_tuple(cuda_domain(device_id))));
    }

    /// @brief Add NCCL domain from MPI to create new context
    /// @param device_id CUDA device ID for this rank
    /// @return Result containing new context with additional nccl_domain
    /// @pre has_mpi() must be true
    [[nodiscard]] auto with_nccl(int device_id) const
        -> result<detail::append_domain_t<context, nccl_domain>>
        requires (has_mpi())
    {
        using new_context_t = detail::append_domain_t<context, nccl_domain>;

        auto nccl_result = nccl_domain::from_mpi(std::get<mpi_domain>(domains_), device_id);
        if (!nccl_result) {
            return result<new_context_t>::failure(nccl_result.error());
        }

        return result<new_context_t>::success(
            new_context_t(std::tuple_cat(domains_,
                                         std::make_tuple(std::move(*nccl_result)))));
    }

    /// @brief Split both MPI and NCCL domains to create new sub-group context
    /// @param color Color for grouping (ranks with same color in same group)
    /// @param device_id CUDA device ID for this rank in the new NCCL communicator
    /// @param key Ordering key within color group (default 0)
    /// @return Result containing new context with split MPI and NCCL domains
    /// @pre has_mpi() && has_nccl() must be true
    [[nodiscard]] result<context> split_nccl(int color, int device_id, int key = 0) const
        requires (has_mpi() && has_nccl())
    {
        auto split_result = nccl_domain::split(
            std::get<mpi_domain>(domains_), color, device_id, key);
        if (!split_result) {
            return result<context>::failure(split_result.error());
        }

        auto& [new_mpi, new_nccl] = *split_result;
        auto new_domains = replace_domain<mpi_domain>(std::move(new_mpi));
        auto final_domains = std::apply([&new_nccl](auto&&... ds) {
            return std::make_tuple(
                [&new_nccl]<typename T>(T&& d) {
                    if constexpr (std::is_same_v<std::decay_t<T>, nccl_domain>) {
                        return std::move(new_nccl);
                    } else {
                        return std::forward<T>(d);
                    }
                }(std::forward<decltype(ds)>(ds))...
            );
        }, std::move(new_domains));
        return result<context>::success(context(std::move(final_domains)));
    }

    /// @brief Add CPU domain to create new context (if not already present)
    [[nodiscard]] auto with_cpu() const {
        if constexpr (has_cpu()) {
            return *this;
        } else {
            using new_context_t = detail::append_domain_t<context, cpu_domain>;
            return new_context_t(std::tuple_cat(domains_,
                                                std::make_tuple(cpu_domain{})));
        }
    }

    // =========================================================================
    // Legacy Compatibility
    // =========================================================================

    /// @brief Legacy alias for size() - returns number of ranks
    /// @deprecated Use size() instead for STL consistency
    [[deprecated("Use size() instead")]]
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return size();
    }

private:
    std::tuple<Domains...> domains_;

    /// @brief Helper to replace a domain in the tuple with a new instance
    template <typename D, typename NewD>
    [[nodiscard]] auto replace_domain(NewD&& new_domain) const {
        return std::apply([&new_domain](auto&&... ds) {
            return std::make_tuple(
                [&new_domain]<typename T>(T&& d) {
                    if constexpr (std::is_same_v<std::decay_t<T>, D>) {
                        return std::forward<NewD>(new_domain);
                    } else {
                        return std::forward<T>(d);
                    }
                }(std::forward<decltype(ds)>(ds))...
            );
        }, domains_);
    }
};

// =============================================================================
// Common Context Type Aliases
// =============================================================================

/// @brief Context with MPI and CPU domains (most common configuration)
using mpi_context = context<mpi_domain, cpu_domain>;

/// @brief Context with MPI, CPU, and CUDA domains
using mpi_cuda_context = context<mpi_domain, cpu_domain, cuda_domain>;

/// @brief Context with MPI, CPU, CUDA, and NCCL domains
using mpi_nccl_context = context<mpi_domain, cpu_domain, cuda_domain, nccl_domain>;

/// @brief Context with only CPU domain (single-process, no MPI)
using cpu_context = context<cpu_domain>;

/// @brief Context with SHMEM and CPU domains
using shmem_context = context<shmem_domain, cpu_domain>;

// =============================================================================
// Context Concept
// =============================================================================

/// @brief Concept for context types
template <typename T>
concept Context = requires(const T& ctx) {
    { ctx.rank() } -> std::same_as<rank_t>;
    { ctx.size() } -> std::same_as<rank_t>;
    { ctx.is_root() } -> std::same_as<bool>;
    { ctx.valid() } -> std::same_as<bool>;
};

// Verify common context types satisfy the concept
static_assert(Context<cpu_context>, "cpu_context must satisfy Context concept");

// =============================================================================
// Free Functions
// =============================================================================

/// @brief Create a CPU-only context
/// @return cpu_context instance
[[nodiscard]] inline cpu_context make_cpu_context() {
    return cpu_context{};
}

}  // namespace dtl
