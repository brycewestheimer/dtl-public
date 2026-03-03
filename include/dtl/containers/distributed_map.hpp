// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_map.hpp
/// @brief Distributed associative container (std::unordered_map analog)
/// @details Hash-based key distribution across ranks.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/core/sync_domain.hpp>
#include <dtl/error/result.hpp>
#include <dtl/handle/handle.hpp>
#include <dtl/policies/policies.hpp>
#include <dtl/serialization/serialization.hpp>
#include <dtl/views/remote_ref.hpp>

#include <cstring>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dtl {

// Forward declarations
template <typename T>
class remote_ref;

/// @brief Distributed hash map container
/// @tparam K Key type
/// @tparam V Value (mapped) type
/// @tparam Hash Hash function type (default: std::hash<K>)
/// @tparam KeyEqual Key equality function (default: std::equal_to<K>)
/// @tparam Policies... Policy pack
/// @details A distributed associative container that partitions key-value pairs
///          across ranks based on key hashes. Uses hash_partition by default.
///
/// @par Concept Compliance:
/// This class satisfies `DistributedAssociativeContainer` and `DistributedMap`,
/// but intentionally does NOT satisfy the `DistributedContainer` concept.
///
/// @par Why distributed_map Doesn't Satisfy DistributedContainer:
/// The `DistributedContainer` concept (defined in concepts.hpp) requires:
/// - `local_view()` returning a contiguous view of local data
/// - `global_view()` for index-based global access
/// - `segmented_view()` for segment-based iteration
/// - `size()` returning total global element count
///
/// These requirements are designed for **sequence containers** like
/// `distributed_vector` and `distributed_tensor` where:
/// - Elements have a natural linear ordering
/// - Contiguous memory layout enables efficient iteration
/// - Global indices map directly to elements
///
/// Associative containers like `distributed_map` have fundamentally different
/// semantics:
/// - **Key-based access**: Elements are accessed by key, not by index
/// - **Non-contiguous layout**: Hash tables don't have contiguous memory
/// - **No natural ordering**: Keys are distributed by hash, not position
/// - **Different iteration model**: Iterate over key-value pairs, not indices
///
/// For maps, use the `DistributedAssociativeContainer` concept instead, which
/// requires `local_size()`, `is_local(key)`, `owner(key)`, and iteration.
///
/// @par Key Distribution:
/// Keys are hashed and assigned to ranks: rank = hash(key) % num_ranks.
/// This provides good load balance for uniformly distributed keys.
///
/// @par Access Patterns:
/// - Local key access: Direct, no communication
/// - Remote key access: Returns remote_ref<V>, requires explicit get()/put()
/// - Iteration: Visits only local pairs
///
/// @par Example Usage:
/// @code
/// dtl::distributed_map<std::string, int> map(ctx);
///
/// // Insert (may be local or remote)
/// map.insert("key1", 42);
///
/// // Access
/// auto ref = map["key1"];
/// if (ref.is_local()) {
///     int val = ref.get().value();
/// }
///
/// // Iterate local entries only
/// for (auto& [key, value] : map.local_view()) {
///     process(key, value);
/// }
/// @endcode
template <typename K,
          typename V,
          typename Hash,
          typename KeyEqual,
          typename... Policies>
class distributed_map {
public:
    // ========================================================================
    // Type Aliases
    // ========================================================================

    /// @brief Key type
    using key_type = K;

    /// @brief Mapped type
    using mapped_type = V;

    /// @brief Value type (key-value pair)
    using value_type = std::pair<const K, V>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    /// @brief Hash function type
    using hasher = Hash;

    /// @brief Key equality type
    using key_equal = KeyEqual;

    /// @brief Reference type for mapped values
    using reference = V&;

    /// @brief Const reference type
    using const_reference = const V&;

    /// @brief Remote reference type (for remote access)
    using remote_reference = remote_ref<V>;

    // Local storage type
    using local_map_type = std::unordered_map<K, V, Hash, KeyEqual>;

    /// @brief Iterator type (local only)
    using iterator = typename local_map_type::iterator;

    /// @brief Const iterator type
    using const_iterator = typename local_map_type::const_iterator;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (single-rank mode)
    distributed_map() = default;

    // -------------------------------------------------------------------------
    // Context-Based Constructors (V1.3.0 - Preferred)
    // -------------------------------------------------------------------------

    /// @brief Construct with context
    /// @param ctx The execution context providing rank/size
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    explicit distributed_map(const Ctx& ctx)
        : my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    /// @brief Construct with bucket count hint and context
    /// @param bucket_count Initial bucket count hint (per rank)
    /// @param ctx The execution context providing rank/size
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    distributed_map(size_type bucket_count, const Ctx& ctx)
        : local_map_(bucket_count)
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    // -------------------------------------------------------------------------
    // Legacy Constructors (Deprecated - will be removed in V2.0.0)
    // -------------------------------------------------------------------------

    /// @brief Construct with rank info for standalone mode
    /// @param num_ranks Total number of ranks
    /// @param my_rank This rank's ID
    /// @deprecated Use distributed_map(ctx) instead
    [[deprecated("Use distributed_map(ctx) instead - will be removed in V2.0.0")]]
    distributed_map(rank_t num_ranks, rank_t my_rank)
        : my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    // ========================================================================
    // Size Queries
    // ========================================================================

    /// @brief Get local size (elements on this rank)
    [[nodiscard]] size_type local_size() const noexcept {
        return local_map_.size();
    }

    /// @brief Get global size (total elements across all ranks)
    /// @note Without communicator, returns local size only (accurate for single-rank)
    /// @deprecated For multi-rank global semantics, use global_size_with_comm()
    DTL_DEPRECATED_MSG("distributed_map::size() reports local size without communicator in multi-rank mode; use global_size_with_comm(comm) for true global size")
    [[nodiscard]] size_type size() const {
        // Without communicator, can only return local size
        // For accurate global size in multi-rank, use global_size_with_comm()
        return local_map_.size();
    }

    /// @brief Get global size using communicator
    /// @tparam Communicator Communicator type with allreduce support
    /// @param comm Communicator for reduction
    /// @return Total element count across all ranks
    /// @note This is a collective operation - all ranks must call
    template <typename Communicator>
    [[nodiscard]] size_type global_size_with_comm(Communicator& comm) const {
        return comm.template allreduce_sum_value<size_type>(local_map_.size());
    }

    /// @brief Check if locally empty
    [[nodiscard]] bool local_empty() const noexcept {
        return local_map_.empty();
    }

    /// @brief Check if globally empty
    [[nodiscard]] bool empty() const {
        return local_map_.empty();
    }

    // ========================================================================
    // Element Access
    // ========================================================================

    /// @brief Access element by key (insert-or-find for local keys)
    /// @param key The key to look up
    /// @return remote_ref<V> to the mapped value
    /// @note Returns remote_ref even for local keys to provide uniform interface
    [[nodiscard]] remote_reference operator[](const K& key) {
        rank_t owner_rank = owner(key);
        if (owner_rank != my_rank_) {
            record_legacy_ownership_diagnostic(
                "operator[]",
                "Remote key access via operator[] no longer inserts locally; use "
                "insert_or_assign_remote()/insert_remote() for explicit remote mutation");
            return remote_reference(owner_rank, 0, nullptr);
        }

        std::lock_guard<std::mutex> lock(local_mutex_);
        auto& val = local_map_[key];
        return remote_reference(owner_rank, 0, &val);
    }

    /// @brief Access element by key (const)
    [[nodiscard]] remote_ref<const V> operator[](const K& key) const {
        rank_t owner_rank = owner(key);
        if (owner_rank != my_rank_) {
            return remote_ref<const V>(owner_rank, 0, nullptr);
        }

        std::lock_guard<std::mutex> lock(local_mutex_);
        auto it = local_map_.find(key);
        if (it != local_map_.end()) {
            return remote_ref<const V>(my_rank_, 0, &it->second);
        }
        // For const access to a non-existent key, return a ref with nullptr
        return remote_ref<const V>(owner_rank, 0, nullptr);
    }

    /// @brief Find element by key
    /// @param key The key to find
    /// @return result containing iterator (local) or error (not found/remote)
    [[nodiscard]] result<iterator> find(const K& key) {
        return find_local(key);
    }

    /// @brief Find element by key in local partition only
    /// @param key The key to find
    /// @return result containing local iterator or key_not_found
    [[nodiscard]] result<iterator> find_local(const K& key) {
        auto it = local_map_.find(key);
        if (it != local_map_.end()) {
            return result<iterator>::success(it);
        }
        return result<iterator>::failure(
            status{status_code::key_not_found, no_rank, "Key not found locally"});
    }

    /// @brief Check if key exists locally
    /// @param key The key to check
    /// @return true if key exists in local partition
    /// @note For distributed query, use contains_with_comm()
    [[nodiscard]] bool contains(const K& key) const {
        return contains_local(key);
    }

    /// @brief Check if key exists in local partition only
    /// @param key The key to check
    /// @return true if key exists locally
    [[nodiscard]] bool contains_local(const K& key) const {
        // Check local storage only; for remote check use contains_with_comm()
        return local_map_.count(key) > 0;
    }

    /// @brief Check if key exists anywhere (collective)
    /// @tparam Communicator Communicator type
    /// @param key The key to check
    /// @param comm Communicator for query
    /// @return true if key exists on any rank
    /// @note This is a collective operation - all ranks must call
    template <typename Communicator>
    [[nodiscard]] bool contains_with_comm(const K& key, Communicator& comm) const {
        // Check locally first
        bool local_found = (local_map_.count(key) > 0);
        // Reduce with logical OR across all ranks
        return comm.allreduce_lor_value(local_found);
    }

    /// @brief Check if key exists on its owner rank (point-to-point)
    /// @tparam Communicator Communicator type
    /// @param key The key to check
    /// @param comm Communicator for P2P query
    /// @return Result with true if key exists, false otherwise
    /// @note Only queries the owner rank, not a collective operation
    template <typename Communicator>
    [[nodiscard]] result<bool> contains_on_owner(const K& key, Communicator& comm) const {
        rank_t owner_rank = owner(key);

        if (owner_rank == my_rank_) {
            // We own it, check locally
            return result<bool>::success(local_map_.count(key) > 0);
        }

        try {
            // Query owner rank with sendrecv pattern
            const int query_tag = 100;
            const int reply_tag = 101;

            // Send query (key hash as identifier)
            size_t key_hash = hasher{}(key);

            int found = 0;

            // Use sendrecv to avoid deadlock: we send query, receive response
            // Note: This requires owner to be listening, so typically used
            // in a bulk query pattern where all ranks participate
            comm.send(&key_hash, sizeof(size_t), owner_rank, query_tag);
            comm.recv(&found, sizeof(int), owner_rank, reply_tag);

            return result<bool>::success(found != 0);
        } catch (const std::exception& e) {
            return result<bool>::failure(
                status{status_code::operation_failed,
                       std::string("Remote contains query failed: ") + e.what()});
        }
    }

    /// @brief Count occurrences of key
    /// @param key The key to count
    /// @return 1 if exists, 0 otherwise
    [[nodiscard]] size_type count(const K& key) const {
        return contains(key) ? 1 : 0;
    }

    // ========================================================================
    // Modifiers
    // ========================================================================

    /// @brief Insert key-value pair
    /// @param key The key
    /// @param value The value
    /// @return Result indicating success or error
    /// @note For remote keys, legacy behavior queues a remote op; prefer
    ///       insert_remote() for explicit migration-safe semantics.
    result<void> insert(const K& key, const V& value) {
        if (is_local(key)) {
            return insert_local(key, value);
        }
        record_legacy_ownership_diagnostic(
            "insert",
            "Implicit remote insert path used; prefer insert_remote() for explicit ownership-aware mutation");
        return queue_remote_insert(key, value);
    }

    /// @brief Insert key-value pair (move)
    result<void> insert(K&& key, V&& value) {
        K key_copy = key;
        if (is_local(key_copy)) {
            std::lock_guard<std::mutex> lock(local_mutex_);
            local_map_.emplace(std::move(key), std::move(value));
            sync_state_.mark_local_modified();
            return result<void>::success();
        }
        record_legacy_ownership_diagnostic(
            "insert",
            "Implicit remote insert path used; prefer insert_remote() for explicit ownership-aware mutation");
        return queue_remote_insert(std::move(key), std::move(value));
    }

    /// @brief Insert key-value pair constrained to local ownership
    /// @param key The key
    /// @param value The value
    /// @return precondition_failed when key is not locally owned
    result<void> insert_local(const K& key, const V& value) {
        if (!is_local(key)) {
            return result<void>::failure(
                status{status_code::precondition_failed,
                       owner(key),
                       "insert_local requires local key ownership"});
        }
        std::lock_guard<std::mutex> lock(local_mutex_);
        local_map_.emplace(key, value);
        sync_state_.mark_local_modified();
        return result<void>::success();
    }

    /// @brief Insert key-value pair explicitly to remote owner queue
    /// @param key The key
    /// @param value The value
    /// @return precondition_failed when key is locally owned
    result<void> insert_remote(const K& key, const V& value) {
        if (is_local(key)) {
            return result<void>::failure(
                status{status_code::precondition_failed,
                       my_rank_,
                       "insert_remote requires non-local key ownership"});
        }
        return queue_remote_insert(key, value);
    }

    /// @brief Insert or assign
    /// @param key The key
    /// @param value The value to insert or assign
    template <typename M>
    result<void> insert_or_assign(const K& key, M&& value) {
        if (is_local(key)) {
            return insert_or_assign_local(key, std::forward<M>(value));
        }
        record_legacy_ownership_diagnostic(
            "insert_or_assign",
            "Implicit remote upsert path used; prefer insert_or_assign_remote() for explicit ownership-aware mutation");
        return queue_remote_insert_or_assign(key, std::forward<M>(value));
    }

    /// @brief Insert or assign constrained to local ownership
    /// @param key The key
    /// @param value The value to insert or assign
    template <typename M>
    result<void> insert_or_assign_local(const K& key, M&& value) {
        if (!is_local(key)) {
            return result<void>::failure(
                status{status_code::precondition_failed,
                       owner(key),
                       "insert_or_assign_local requires local key ownership"});
        }
        std::lock_guard<std::mutex> lock(local_mutex_);
        local_map_.insert_or_assign(key, std::forward<M>(value));
        sync_state_.mark_local_modified();
        return result<void>::success();
    }

    /// @brief Insert or assign explicitly to remote owner queue
    /// @param key The key
    /// @param value The value to insert or assign remotely
    template <typename M>
    result<void> insert_or_assign_remote(const K& key, M&& value) {
        if (is_local(key)) {
            return result<void>::failure(
                status{status_code::precondition_failed,
                       my_rank_,
                       "insert_or_assign_remote requires non-local key ownership"});
        }
        return queue_remote_insert_or_assign(key, std::forward<M>(value));
    }

    /// @brief Erase by key
    /// @param key The key to erase
    /// @return Result with count of erased elements
    result<size_type> erase(const K& key) {
        if (is_local(key)) {
            return erase_local(key);
        }
        record_legacy_ownership_diagnostic(
            "erase",
            "Implicit remote erase path used; prefer erase_remote() for explicit ownership-aware mutation");
        return queue_remote_erase(key);
    }

    /// @brief Erase constrained to local ownership
    /// @param key The key to erase
    /// @return Result with local erase count
    result<size_type> erase_local(const K& key) {
        if (!is_local(key)) {
            return result<size_type>::failure(
                status{status_code::precondition_failed,
                       owner(key),
                       "erase_local requires local key ownership"});
        }
        std::lock_guard<std::mutex> lock(local_mutex_);
        size_type erased = local_map_.erase(key);
        if (erased > 0) {
            sync_state_.mark_local_modified();
        }
        return result<size_type>::success(erased);
    }

    /// @brief Erase explicitly through remote owner queue
    /// @param key The key to erase
    /// @return precondition_failed when key is locally owned
    result<size_type> erase_remote(const K& key) {
        if (is_local(key)) {
            return result<size_type>::failure(
                status{status_code::precondition_failed,
                       my_rank_,
                       "erase_remote requires non-local key ownership"});
        }
        return queue_remote_erase(key);
    }

    /// @brief Check if implicit-remote migration diagnostics are present
    [[nodiscard]] bool has_legacy_ownership_diagnostic() const {
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        return legacy_ownership_diagnostic_.has_value();
    }

    /// @brief Get latest implicit-remote migration diagnostic
    [[nodiscard]] std::optional<std::string> legacy_ownership_diagnostic() const {
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        return legacy_ownership_diagnostic_;
    }

    /// @brief Clear implicit-remote migration diagnostic state
    void clear_legacy_ownership_diagnostic() {
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        legacy_ownership_diagnostic_.reset();
    }

    /// @brief Compile-time visibility of transport contract support
    [[nodiscard]] static constexpr bool transport_contract_satisfied() noexcept {
        constexpr bool key_supported = is_trivially_serializable_v<K> || std::is_same_v<K, std::string>;
        constexpr bool value_supported = is_trivially_serializable_v<V> || std::is_same_v<V, std::string>;
        return key_supported && value_supported;
    }

    /// @brief Flush all pending remote operations
    /// @return Result indicating success or error
    /// @note Without communicator, clears pending queues without sending
    /// @note Use flush_pending_with_comm() for actual remote delivery
    result<void> flush_pending() {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_inserts_.empty() && pending_erases_.empty() &&
            pending_insert_or_assigns_.empty()) {
            return result<void>::success();
        }

        bool had_remote_ops = !pending_inserts_.empty() ||
                              !pending_erases_.empty() ||
                              !pending_insert_or_assigns_.empty();

        pending_inserts_.clear();
        pending_erases_.clear();
        pending_insert_or_assigns_.clear();

        if (had_remote_ops) {
            return result<void>::failure(
                status{status_code::not_implemented,
                       no_rank,
                       "Remote operation batching requires communicator; use flush_pending_with_comm()"});
        }

        return result<void>::success();
    }

    /// @brief Flush all pending remote operations using communicator
    /// @tparam Communicator Communicator type with alltoallv support
    /// @param comm Communicator for operation exchange
    /// @return Result indicating success or error
    /// @note This is a collective operation - all ranks must call
    /// @details Algorithm:
    ///   1. Group pending operations by destination rank
    ///   2. Exchange operation counts via alltoall
    ///   3. Exchange serialized operations via alltoallv
    ///   4. Apply received operations locally
    template <typename Communicator>
    result<void> flush_pending_with_comm(Communicator& comm) {
        static_assert(has_serializer_v<K>,
                      "distributed_map::flush_pending_with_comm requires serializable key type K");
        static_assert(has_serializer_v<V>,
                      "distributed_map::flush_pending_with_comm requires serializable mapped type V");

        std::lock_guard<std::mutex> lock(pending_mutex_);

        // Early exit if no pending operations on any rank
        size_type local_pending = pending_inserts_.size() +
                                  pending_erases_.size() +
                                  pending_insert_or_assigns_.size();
        size_type global_pending = comm.template allreduce_sum_value<size_type>(local_pending);

        if (global_pending == 0) {
            return result<void>::success();
        }

        try {
            // Group inserts by destination rank
            std::vector<std::vector<std::pair<K, V>>> insert_by_rank(
                static_cast<size_type>(num_ranks_));
            for (const auto& [key, value] : pending_inserts_) {
                rank_t dest = owner(key);
                insert_by_rank[static_cast<size_type>(dest)].emplace_back(key, value);
            }

            // Group insert_or_assigns by destination rank
            std::vector<std::vector<std::pair<K, V>>> assign_by_rank(
                static_cast<size_type>(num_ranks_));
            for (const auto& [key, value] : pending_insert_or_assigns_) {
                rank_t dest = owner(key);
                assign_by_rank[static_cast<size_type>(dest)].emplace_back(key, value);
            }

            // Group erases by destination rank
            std::vector<std::vector<K>> erase_by_rank(
                static_cast<size_type>(num_ranks_));
            for (const auto& key : pending_erases_) {
                rank_t dest = owner(key);
                erase_by_rank[static_cast<size_type>(dest)].push_back(key);
            }

            auto insert_exchange = exchange_kv_buckets_with_comm(insert_by_rank, comm);
            if (!insert_exchange) {
                return result<void>::failure(insert_exchange.error());
            }
            auto insert_recv_flat = std::move(insert_exchange.value());

            // Apply received inserts locally
            for (const auto& [key, value] : insert_recv_flat) {
                local_map_.emplace(key, value);
            }

            auto assign_exchange = exchange_kv_buckets_with_comm(assign_by_rank, comm);
            if (!assign_exchange) {
                return result<void>::failure(assign_exchange.error());
            }
            auto assign_recv_flat = std::move(assign_exchange.value());

            for (const auto& [key, value] : assign_recv_flat) {
                local_map_.insert_or_assign(key, value);
            }

            auto erase_exchange = exchange_key_buckets_with_comm(erase_by_rank, comm);
            if (!erase_exchange) {
                return result<void>::failure(erase_exchange.error());
            }
            auto erase_recv_flat = std::move(erase_exchange.value());

            for (const auto& key : erase_recv_flat) {
                local_map_.erase(key);
            }

            // Clear pending queues
            pending_inserts_.clear();
            pending_erases_.clear();
            pending_insert_or_assigns_.clear();

            // Mark sync state
            if (!insert_recv_flat.empty() || !assign_recv_flat.empty() ||
                !erase_recv_flat.empty()) {
                sync_state_.mark_local_modified();
            }

            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::operation_failed,
                       no_rank,
                       std::string("Flush pending failed: ") + e.what()});
        }
    }

    /// @brief Apply a batch of remote insert operations (called by RPC handler)
    /// @param operations Vector of key-value pairs to insert
    void apply_remote_inserts(const std::vector<std::pair<K, V>>& operations) {
        std::lock_guard<std::mutex> lock(local_mutex_);
        for (const auto& [key, value] : operations) {
            if (is_local(key)) {
                local_map_.emplace(key, value);
            }
        }
        if (!operations.empty()) {
            sync_state_.mark_local_modified();
        }
    }

    /// @brief Apply a batch of remote erase operations (called by RPC handler)
    /// @param keys Vector of keys to erase
    void apply_remote_erases(const std::vector<K>& keys) {
        std::lock_guard<std::mutex> lock(local_mutex_);
        for (const auto& key : keys) {
            if (is_local(key)) {
                local_map_.erase(key);
            }
        }
        if (!keys.empty()) {
            sync_state_.mark_local_modified();
        }
    }

    /// @brief Clear all elements (collective)
    result<void> clear() {
        local_map_.clear();
        return result<void>::success();
    }

    // ========================================================================
    // Iterators (Local Only)
    // ========================================================================

    /// @brief Get iterator to beginning of local entries
    [[nodiscard]] iterator begin() noexcept {
        return local_map_.begin();
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator begin() const noexcept {
        return local_map_.begin();
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return local_map_.cbegin();
    }

    /// @brief Get iterator to end
    [[nodiscard]] iterator end() noexcept {
        return local_map_.end();
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator end() const noexcept {
        return local_map_.end();
    }

    /// @brief Get const iterator to end
    [[nodiscard]] const_iterator cend() const noexcept {
        return local_map_.cend();
    }

    // ========================================================================
    // Distribution Queries
    // ========================================================================

    /// @brief Check if a key is stored locally
    [[nodiscard]] bool is_local(const K& key) const {
        return owner(key) == rank();
    }

    /// @brief Get owner rank for a key
    [[nodiscard]] rank_t owner(const K& key) const {
        DTL_ASSERT(num_ranks() > 0);
        return static_cast<rank_t>(
            hasher{}(key) % static_cast<size_type>(num_ranks()));
    }

    /// @brief Get number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get current rank
    [[nodiscard]] rank_t rank() const noexcept {
        return my_rank_;
    }

    // ========================================================================
    // Hash Policy
    // ========================================================================

    /// @brief Get load factor (local)
    [[nodiscard]] float load_factor() const noexcept {
        return local_map_.load_factor();
    }

    /// @brief Get max load factor
    [[nodiscard]] float max_load_factor() const noexcept {
        return local_map_.max_load_factor();
    }

    /// @brief Set max load factor
    void max_load_factor(float ml) {
        local_map_.max_load_factor(ml);
    }

    /// @brief Rehash local map
    void rehash(size_type count) {
        local_map_.rehash(count);
    }

    /// @brief Reserve space locally
    void reserve(size_type count) {
        local_map_.reserve(count);
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// @brief Collective barrier
    result<void> barrier() {
        return comm_handle_.barrier();
    }

    [[nodiscard]] handle::comm_handle communicator_handle() const {
        return comm_handle_;
    }

    // ========================================================================
    // Sync State
    // ========================================================================

    /// @brief Check if container has local modifications
    [[nodiscard]] bool is_dirty() const noexcept {
        return sync_state_.is_dirty();
    }

    /// @brief Check if container is clean
    [[nodiscard]] bool is_clean() const noexcept {
        return sync_state_.is_clean();
    }

    /// @brief Get sync state reference
    [[nodiscard]] const sync_state& sync_state_ref() const noexcept {
        return sync_state_;
    }

    /// @brief Get sync state reference (mutable)
    [[nodiscard]] sync_state& sync_state_ref() noexcept {
        return sync_state_;
    }

    /// @brief Mark container as clean
    void mark_clean() noexcept {
        sync_state_.mark_clean();
    }

    /// @brief Mark container as locally modified
    void mark_local_modified() noexcept {
        sync_state_.mark_local_modified();
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// @brief Insert multiple key-value pairs in batch
    /// @details Groups keys by owning rank for efficient batch communication.
    ///          For single-rank mode, this is equivalent to inserating each
    ///          key-value pair individually. For multi-rank, local keys are
    ///          inserted immediately and remote keys are queued for flush.
    /// @tparam InputIterator Iterator to std::pair<K, V> (or compatible)
    /// @param first Iterator to the first key-value pair
    /// @param last Iterator past the last key-value pair
    /// @return Result containing the number of successfully inserted local keys
    template <typename InputIterator>
    result<size_type> batch_insert(InputIterator first, InputIterator last) {
        size_type local_inserted = 0;

        for (auto it = first; it != last; ++it) {
            const auto& [key, value] = *it;
            if (is_local(key)) {
                std::lock_guard<std::mutex> lock(local_mutex_);
                auto [map_it, inserted] = local_map_.emplace(key, value);
                if (inserted) {
                    ++local_inserted;
                }
            } else {
                queue_remote_insert(key, value);
            }
        }

        if (local_inserted > 0) {
            sync_state_.mark_local_modified();
        }

        return result<size_type>::success(local_inserted);
    }

    /// @brief Lookup multiple keys in batch
    /// @details Groups keys by owning rank for efficient batch lookup.
    ///          For single-rank mode, performs local lookups for all keys.
    ///          Returns std::nullopt for keys not found locally.
    /// @tparam KeyIterator Iterator to key type
    /// @param first Iterator to the first key
    /// @param last Iterator past the last key
    /// @return Result containing a vector of optional values (nullopt if not found)
    template <typename KeyIterator>
    result<std::vector<std::optional<mapped_type>>> batch_find(KeyIterator first, KeyIterator last) const {
        std::vector<std::optional<mapped_type>> results;
        results.reserve(static_cast<size_type>(std::distance(first, last)));

        for (auto it = first; it != last; ++it) {
            const auto& key = *it;
            if (is_local(key)) {
                auto map_it = local_map_.find(key);
                if (map_it != local_map_.end()) {
                    results.emplace_back(map_it->second);
                } else {
                    results.emplace_back(std::nullopt);
                }
            } else {
                // Remote key: cannot look up without communicator
                results.emplace_back(std::nullopt);
            }
        }

        return result<std::vector<std::optional<mapped_type>>>::success(std::move(results));
    }

    /// @brief Synchronize the container (flush pending + barrier)
    /// @return Result indicating success or error
    result<void> sync() {
        auto flush_result = flush_pending();
        if (!flush_result) {
            return flush_result;
        }
        auto barrier_result = barrier();
        if (!barrier_result) {
            return barrier_result;
        }
        sync_state_.mark_clean();
        return result<void>::success();
    }

private:
    local_map_type local_map_;
    mutable std::mutex local_mutex_;  // Protects local_map_

    // Distribution info
    rank_t my_rank_ = 0;
    rank_t num_ranks_ = 1;
    handle::comm_handle comm_handle_ = handle::comm_handle::local();

    // Sync state tracking
    sync_state sync_state_;

    // Pending remote operations (batched for efficiency)
    std::vector<std::pair<K, V>> pending_inserts_;
    std::vector<std::pair<K, V>> pending_insert_or_assigns_;
    std::vector<K> pending_erases_;
    mutable std::mutex pending_mutex_;  // Protects pending queues
    mutable std::optional<std::string> legacy_ownership_diagnostic_;
    mutable std::mutex diagnostics_mutex_;

    // ========================================================================
    // Private Helpers
    // ========================================================================

    result<void> queue_remote_insert(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_inserts_.emplace_back(key, value);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    result<void> queue_remote_insert(K&& key, V&& value) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_inserts_.emplace_back(std::move(key), std::move(value));
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    template <typename M>
    result<void> queue_remote_insert_or_assign(const K& key, M&& value) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_insert_or_assigns_.emplace_back(key, std::forward<M>(value));
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    result<size_type> queue_remote_erase(const K& key) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_erases_.push_back(key);
        sync_state_.mark_global_dirty();
        // Return 0 since we don't know if erase will succeed until sync
        return result<size_type>::success(0);
    }

    void record_legacy_ownership_diagnostic(std::string_view api,
                                            std::string_view message) const {
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        legacy_ownership_diagnostic_ = std::string(api) + ": " + std::string(message);
    }

    static size_type serialized_kv_pair_size(const std::pair<K, V>& entry) {
        return sizeof(size_type) + sizeof(size_type) +
               serialized_size(entry.first) + serialized_size(entry.second);
    }

    static size_type serialized_key_size(const K& key) {
        return sizeof(size_type) + serialized_size(key);
    }

    static size_type serialized_kv_bucket_size(const std::vector<std::pair<K, V>>& bucket) {
        size_type total = sizeof(size_type);
        for (const auto& entry : bucket) {
            total += serialized_kv_pair_size(entry);
        }
        return total;
    }

    static size_type serialized_key_bucket_size(const std::vector<K>& bucket) {
        size_type total = sizeof(size_type);
        for (const auto& key : bucket) {
            total += serialized_key_size(key);
        }
        return total;
    }

    static std::vector<std::byte> serialize_kv_bucket(const std::vector<std::pair<K, V>>& bucket) {
        std::vector<std::byte> bytes(serialized_kv_bucket_size(bucket));
        size_type offset = 0;
        const size_type count = bucket.size();
        std::memcpy(bytes.data() + offset, &count, sizeof(size_type));
        offset += sizeof(size_type);

        for (const auto& [key, value] : bucket) {
            const size_type key_size = serialized_size(key);
            const size_type value_size = serialized_size(value);
            std::memcpy(bytes.data() + offset, &key_size, sizeof(size_type));
            offset += sizeof(size_type);
            std::memcpy(bytes.data() + offset, &value_size, sizeof(size_type));
            offset += sizeof(size_type);
            offset += serialize(key, bytes.data() + offset);
            offset += serialize(value, bytes.data() + offset);
        }
        return bytes;
    }

    static result<std::vector<std::pair<K, V>>> deserialize_kv_bucket(std::span<const std::byte> bytes) {
        if (bytes.size() < sizeof(size_type)) {
            return result<std::vector<std::pair<K, V>>>::failure(
                status{status_code::invalid_format, no_rank,
                       "Serialized map bucket too small for count header"});
        }

        size_type offset = 0;
        size_type count = 0;
        std::memcpy(&count, bytes.data() + offset, sizeof(size_type));
        offset += sizeof(size_type);

        std::vector<std::pair<K, V>> out;
        out.reserve(count);

        for (size_type i = 0; i < count; ++i) {
            if (offset + sizeof(size_type) * 2 > bytes.size()) {
                return result<std::vector<std::pair<K, V>>>::failure(
                    status{status_code::invalid_format, no_rank,
                           "Serialized map bucket missing key/value size headers"});
            }

            size_type key_size = 0;
            size_type value_size = 0;
            std::memcpy(&key_size, bytes.data() + offset, sizeof(size_type));
            offset += sizeof(size_type);
            std::memcpy(&value_size, bytes.data() + offset, sizeof(size_type));
            offset += sizeof(size_type);

            if (offset + key_size + value_size > bytes.size()) {
                return result<std::vector<std::pair<K, V>>>::failure(
                    status{status_code::invalid_format, no_rank,
                           "Serialized map bucket payload exceeds buffer bounds"});
            }

            K key = deserialize<K>(bytes.data() + offset, key_size);
            offset += key_size;
            V value = deserialize<V>(bytes.data() + offset, value_size);
            offset += value_size;

            out.emplace_back(std::move(key), std::move(value));
        }

        return result<std::vector<std::pair<K, V>>>::success(std::move(out));
    }

    static std::vector<std::byte> serialize_key_bucket(const std::vector<K>& bucket) {
        std::vector<std::byte> bytes(serialized_key_bucket_size(bucket));
        size_type offset = 0;
        const size_type count = bucket.size();
        std::memcpy(bytes.data() + offset, &count, sizeof(size_type));
        offset += sizeof(size_type);

        for (const auto& key : bucket) {
            const size_type key_size = serialized_size(key);
            std::memcpy(bytes.data() + offset, &key_size, sizeof(size_type));
            offset += sizeof(size_type);
            offset += serialize(key, bytes.data() + offset);
        }
        return bytes;
    }

    static result<std::vector<K>> deserialize_key_bucket(std::span<const std::byte> bytes) {
        if (bytes.size() < sizeof(size_type)) {
            return result<std::vector<K>>::failure(
                status{status_code::invalid_format, no_rank,
                       "Serialized key bucket too small for count header"});
        }

        size_type offset = 0;
        size_type count = 0;
        std::memcpy(&count, bytes.data() + offset, sizeof(size_type));
        offset += sizeof(size_type);

        std::vector<K> out;
        out.reserve(count);

        for (size_type i = 0; i < count; ++i) {
            if (offset + sizeof(size_type) > bytes.size()) {
                return result<std::vector<K>>::failure(
                    status{status_code::invalid_format, no_rank,
                           "Serialized key bucket missing key size header"});
            }

            size_type key_size = 0;
            std::memcpy(&key_size, bytes.data() + offset, sizeof(size_type));
            offset += sizeof(size_type);

            if (offset + key_size > bytes.size()) {
                return result<std::vector<K>>::failure(
                    status{status_code::invalid_format, no_rank,
                           "Serialized key bucket payload exceeds buffer bounds"});
            }

            out.emplace_back(deserialize<K>(bytes.data() + offset, key_size));
            offset += key_size;
        }

        return result<std::vector<K>>::success(std::move(out));
    }

    template <typename BucketT, typename SerializeBucketFn, typename DeserializeBucketFn,
              typename ValueT, typename Communicator>
    result<std::vector<ValueT>> exchange_serialized_buckets_with_comm(
        const std::vector<BucketT>& buckets,
        Communicator& comm,
        SerializeBucketFn serialize_bucket,
        DeserializeBucketFn deserialize_bucket) {
        std::vector<std::vector<std::byte>> bucket_bytes;
        bucket_bytes.reserve(static_cast<size_type>(num_ranks_));

        std::vector<int> send_counts(static_cast<size_type>(num_ranks_));
        for (rank_t r = 0; r < num_ranks_; ++r) {
            auto bytes = serialize_bucket(buckets[static_cast<size_type>(r)]);
            send_counts[static_cast<size_type>(r)] = static_cast<int>(bytes.size());
            bucket_bytes.emplace_back(std::move(bytes));
        }

        std::vector<int> recv_counts(static_cast<size_type>(num_ranks_));
        comm.alltoall(send_counts.data(), recv_counts.data(), sizeof(int));

        std::vector<int> send_displs(static_cast<size_type>(num_ranks_));
        std::vector<int> recv_displs(static_cast<size_type>(num_ranks_));
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

        const size_type total_send = static_cast<size_type>(
            send_displs.back() + send_counts.back());
        const size_type total_recv = static_cast<size_type>(
            recv_displs.back() + recv_counts.back());

        std::vector<std::byte> send_flat(total_send);
        for (rank_t r = 0; r < num_ranks_; ++r) {
            const auto idx = static_cast<size_type>(r);
            if (!bucket_bytes[idx].empty()) {
                std::memcpy(send_flat.data() + send_displs[idx],
                            bucket_bytes[idx].data(),
                            static_cast<size_type>(send_counts[idx]));
            }
        }

        std::vector<std::byte> recv_flat(total_recv);

        if (total_send > 0 || total_recv > 0) {
            comm.alltoallv(send_flat.data(),
                           send_counts.data(), send_displs.data(),
                           recv_flat.data(),
                           recv_counts.data(), recv_displs.data(),
                           sizeof(std::byte));
        }

        std::vector<ValueT> out;
        for (rank_t r = 0; r < num_ranks_; ++r) {
            const auto idx = static_cast<size_type>(r);
            const size_type count = static_cast<size_type>(recv_counts[idx]);
            if (count == 0) {
                continue;
            }

            std::span<const std::byte> segment(
                recv_flat.data() + recv_displs[idx],
                count);
            auto decoded = deserialize_bucket(segment);
            if (!decoded) {
                return result<std::vector<ValueT>>::failure(decoded.error());
            }

            auto values = std::move(decoded.value());
            out.insert(out.end(),
                       std::make_move_iterator(values.begin()),
                       std::make_move_iterator(values.end()));
        }

        return result<std::vector<ValueT>>::success(std::move(out));
    }

    template <typename Communicator>
    result<std::vector<std::pair<K, V>>> exchange_kv_buckets_with_comm(
        const std::vector<std::vector<std::pair<K, V>>>& buckets,
        Communicator& comm) {
        return exchange_serialized_buckets_with_comm<std::vector<std::pair<K, V>>,
                                                      decltype(&serialize_kv_bucket),
                                                      decltype(&deserialize_kv_bucket),
                                                      std::pair<K, V>>(
            buckets,
            comm,
            &serialize_kv_bucket,
            &deserialize_kv_bucket);
    }

    template <typename Communicator>
    result<std::vector<K>> exchange_key_buckets_with_comm(
        const std::vector<std::vector<K>>& buckets,
        Communicator& comm) {
        return exchange_serialized_buckets_with_comm<std::vector<K>,
                                                      decltype(&serialize_key_bucket),
                                                      decltype(&deserialize_key_bucket),
                                                      K>(
            buckets,
            comm,
            &serialize_key_bucket,
            &deserialize_key_bucket);
    }
};

// =============================================================================
// Type Trait Specializations
// =============================================================================

// Note: distributed_map does NOT satisfy DistributedContainer concept
// (lacks global_view, segmented_view for index-based access).
// It satisfies DistributedAssociativeContainer instead.
// See class documentation for rationale.

template <typename K, typename V, typename Hash, typename KeyEqual, typename... Policies>
struct is_distributed_container<distributed_map<K, V, Hash, KeyEqual, Policies...>> : std::false_type {};

template <typename K, typename V, typename Hash, typename KeyEqual, typename... Policies>
struct is_distributed_map<distributed_map<K, V, Hash, KeyEqual, Policies...>> : std::true_type {};

}  // namespace dtl
