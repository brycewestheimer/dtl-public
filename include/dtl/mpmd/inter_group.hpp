// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file inter_group.hpp
/// @brief Inter-group communication for MPMD patterns
/// @details Provides operations for communication between rank groups.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/mpmd/node_role.hpp>
#include <dtl/mpmd/rank_group.hpp>

#include <any>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Forward declare communicator_base in dtl:: namespace
namespace dtl { class communicator_base; }

namespace dtl::mpmd {

// Import communicator_base from parent dtl:: namespace
using dtl::communicator_base;

// ============================================================================
// Inter-Group Communicator
// ============================================================================

/// @brief Communicator for operations between rank groups
/// @details Provides point-to-point and collective operations between
///          groups with different roles. Operations are performed between
///          group leaders or all members depending on the operation type.
class inter_group_communicator {
public:
    /// @brief Default constructor (invalid communicator)
    inter_group_communicator() = default;

    /// @brief Construct from source and destination groups
    /// @param source Source group
    /// @param dest Destination group
    inter_group_communicator(rank_group* source, rank_group* dest)
        : source_(source)
        , dest_(dest) {}

    /// @brief Check if communicator is valid
    [[nodiscard]] bool valid() const noexcept {
        return source_ != nullptr && dest_ != nullptr;
    }

    /// @brief Boolean conversion
    explicit operator bool() const noexcept { return valid(); }

    /// @brief Get the source group
    [[nodiscard]] rank_group* source() const noexcept { return source_; }

    /// @brief Get the destination group
    [[nodiscard]] rank_group* dest() const noexcept { return dest_; }

    // ------------------------------------------------------------------------
    // Point-to-Point Operations
    // ------------------------------------------------------------------------

    /// @brief Send data from source leader to destination leader
    /// @tparam T Data type
    /// @param data Data to send
    /// @param tag Message tag
    /// @return Success or error
    /// @note In a real MPI environment, this would use MPI_Send between
    ///       group leaders. This implementation uses a simulated mailbox
    ///       for single-process testing.
    template <typename T>
    result<void> leader_send(const T& data, int tag = 0) {
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                    "Invalid inter-group communicator");
        }
        // Only the source group leader stores data in the mailbox.
        // Non-leaders are no-ops (they would not participate in real MPI).
        if (source_->is_leader()) {
            mailbox::store(tag, std::any(data));
        }
        return {};
    }

    /// @brief Receive data at destination leader from source leader
    /// @tparam T Data type
    /// @param tag Message tag
    /// @return Received data or error
    /// @note In a real MPI environment, this would use MPI_Recv between
    ///       group leaders. This implementation retrieves from a simulated
    ///       mailbox for single-process testing.
    template <typename T>
    result<T> leader_recv(int tag = 0) {
        if (!valid()) {
            return make_error<T>(status_code::invalid_state,
                                "Invalid inter-group communicator");
        }
        // Only the destination group leader retrieves data.
        // Non-leaders return default-constructed T.
        if (dest_->is_leader()) {
            auto val = mailbox::retrieve<T>(tag);
            if (val.has_value()) {
                return std::move(val.value());
            }
            return make_error<T>(status_code::not_found,
                "No data in mailbox for tag " + std::to_string(tag));
        }
        return T{};
    }

    /// @brief Send data from any source rank to destination leader
    /// @tparam T Data type
    /// @param data Data to send
    /// @param source_local Local rank in source group
    /// @param tag Message tag
    /// @return Success or error
    template <typename T>
    result<void> send_to_leader(const T& data, rank_t source_local, int tag = 0) {
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                    "Invalid inter-group communicator");
        }
        // In single-process simulation, store with a composite tag
        // that incorporates the source local rank to avoid collisions.
        int composite_tag = tag * 1000 + static_cast<int>(source_local);
        mailbox::store(composite_tag, std::any(data));
        return {};
    }

    /// @brief Receive at any destination rank from source leader
    /// @tparam T Data type
    /// @param dest_local Local rank in destination group
    /// @param tag Message tag
    /// @return Received data or error
    template <typename T>
    result<T> recv_from_leader(rank_t dest_local, int tag = 0) {
        if (!valid()) {
            return make_error<T>(status_code::invalid_state,
                                "Invalid inter-group communicator");
        }
        // In single-process simulation, retrieve with composite tag.
        int composite_tag = tag * 1000 + static_cast<int>(dest_local);
        auto val = mailbox::retrieve<T>(composite_tag);
        if (val.has_value()) {
            return std::move(val.value());
        }
        return T{};
    }

    // ------------------------------------------------------------------------
    // Collective Operations
    // ------------------------------------------------------------------------

    /// @brief Broadcast from source leader to all destination members
    /// @tparam T Data type
    /// @param data Data to broadcast (only significant at source leader)
    /// @return Received data or error
    /// @note In a real MPI environment, this would use MPI_Bcast across
    ///       both groups. In single-process simulation, data is relayed
    ///       through the mailbox and returned directly.
    template <typename T>
    result<T> broadcast(const T& data) {
        if (!valid()) {
            return make_error<T>(status_code::invalid_state,
                                "Invalid inter-group communicator");
        }
        // Source leader stores data via mailbox for potential retrieval.
        // In single-process mode, we directly return the data.
        if (source_->is_leader()) {
            mailbox::store(broadcast_internal_tag_, std::any(data));
        }
        return data;
    }

    /// @brief Scatter from source leader to destination members
    /// @tparam T Element type
    /// @param data Data to scatter (only significant at source leader)
    /// @return Local portion or error
    /// @note In single-process simulation, returns element at the
    ///       destination member's local rank index.
    template <typename T>
    result<T> scatter(const std::vector<T>& data) {
        if (!valid()) {
            return make_error<T>(status_code::invalid_state,
                                "Invalid inter-group communicator");
        }
        // Validate data size: must have at least one element
        if (data.empty()) {
            return make_error<T>(status_code::invalid_argument,
                                "Scatter data must not be empty");
        }
        // Return element at the destination member's local rank.
        // In single-process simulation, use local_rank if available.
        rank_t local_idx = dest_->is_member() ? dest_->local_rank() : 0;
        if (local_idx < 0 || static_cast<size_type>(local_idx) >= data.size()) {
            local_idx = 0;
        }
        return data[static_cast<size_type>(local_idx)];
    }

    /// @brief Gather from source members to destination leader
    /// @tparam T Element type
    /// @param local_data Local data to contribute
    /// @return Gathered data at destination leader, or error
    /// @note In single-process simulation, returns a vector containing
    ///       only the local_data element.
    template <typename T>
    result<std::vector<T>> gather(const T& local_data) {
        if (!valid()) {
            return make_error<std::vector<T>>(status_code::invalid_state,
                                              "Invalid inter-group communicator");
        }
        // In single-process simulation, return vector with just local_data.
        // A real implementation would gather from all source members.
        return std::vector<T>{local_data};
    }

    /// @brief Reduce from source members to destination leader
    /// @tparam T Data type
    /// @tparam Op Reduction operation type
    /// @param local_data Local data to contribute
    /// @param op Reduction operation
    /// @return Reduced result at destination leader, or error
    /// @note In single-process simulation, returns local_data directly
    ///       since there is only one contributor.
    template <typename T, typename Op>
    result<T> reduce(const T& local_data, Op op) {
        if (!valid()) {
            return make_error<T>(status_code::invalid_state,
                                "Invalid inter-group communicator");
        }
        // With only one process, the reduction identity is just local_data.
        // The op is accepted to validate the API shape.
        (void)op;
        return local_data;
    }

    /// @brief Transfer data from source to destination (all members)
    /// @tparam T Element type
    /// @param data Data to transfer
    /// @return Transferred data or error
    /// @note Each source rank's data goes to corresponding destination rank
    template <typename T>
    result<std::vector<T>> transfer(const std::vector<T>& data) {
        if (!valid()) {
            return make_error<std::vector<T>>(status_code::invalid_state,
                                              "Invalid inter-group communicator");
        }
        return data;
    }

    // ------------------------------------------------------------------------
    // Synchronization
    // ------------------------------------------------------------------------

    /// @brief Synchronize between groups (leader barrier)
    /// @return Success or error
    /// @note In a real MPI environment, this would synchronize group
    ///       leaders via MPI_Barrier on an inter-communicator.
    ///       In single-process simulation, this is a successful no-op.
    result<void> barrier() {
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                    "Invalid inter-group communicator");
        }
        // Single-process: barrier is trivially satisfied.
        return {};
    }

    /// @brief Synchronize all members of both groups
    /// @return Success or error
    /// @note In a real MPI environment, this would synchronize all
    ///       members of both groups via MPI_Barrier.
    ///       In single-process simulation, this is a successful no-op.
    result<void> barrier_all() {
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                    "Invalid inter-group communicator");
        }
        // Single-process: barrier is trivially satisfied.
        return {};
    }

    /// @brief Clear the simulated communication mailbox
    /// @details Removes all pending messages. Useful for test cleanup.
    static void clear_mailbox() { mailbox::clear(); }

private:
    /// @brief Internal tag for broadcast operations
    static constexpr int broadcast_internal_tag_ = -100;

    /// @brief Thread-safe mailbox for simulated inter-group communication
    /// @details In a real MPI environment, leader_send/recv would use
    ///          MPI_Send/MPI_Recv between group leaders. This mailbox
    ///          simulates that for single-process testing by storing
    ///          messages in a static map keyed by tag.
    struct mailbox {
        static inline std::mutex mtx_;
        static inline std::map<int, std::any> data_;

        /// @brief Store data in the mailbox with a given tag
        static void store(int tag, std::any value) {
            std::lock_guard<std::mutex> lock(mtx_);
            data_[tag] = std::move(value);
        }

        /// @brief Retrieve and remove data from the mailbox by tag
        template <typename T>
        static std::optional<T> retrieve(int tag) {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = data_.find(tag);
            if (it != data_.end()) {
                auto val = std::any_cast<T>(it->second);
                data_.erase(it);
                return val;
            }
            return std::nullopt;
        }

        /// @brief Clear all mailbox entries
        static void clear() {
            std::lock_guard<std::mutex> lock(mtx_);
            data_.clear();
        }
    };

    rank_group* source_ = nullptr;
    rank_group* dest_ = nullptr;
};

// ============================================================================
// Inter-Group Operations
// ============================================================================

/// @brief Create an inter-group communicator
/// @param source Source group
/// @param dest Destination group
/// @return Inter-group communicator or error
[[nodiscard]] inline result<inter_group_communicator>
make_inter_group_communicator(rank_group& source, rank_group& dest) {
    if (!source.valid() || !dest.valid()) {
        return make_error<inter_group_communicator>(
            status_code::invalid_argument,
            "Invalid source or destination group");
    }
    return inter_group_communicator(&source, &dest);
}

// NOTE: The overload make_inter_group_communicator(role_manager&, node_role,
// node_role) was removed because it was declared but never defined (linker
// bomb). Use the rank_group-based overload above instead.

// ============================================================================
// Multi-Group Patterns
// ============================================================================

/// @brief Result of a multi-cast operation
/// @tparam T Data type
template <typename T>
struct multicast_result {
    /// @brief Data received from each source group
    std::vector<T> data;

    /// @brief Source group IDs
    std::vector<rank_group::group_id> sources;
};

/// @brief Send to multiple destination groups
/// @tparam T Data type
/// @param source Source group
/// @param destinations Destination groups
/// @param data Data to send
/// @return Success or error
template <typename T>
result<void> multicast(
    rank_group& source,
    const std::vector<rank_group*>& destinations,
    const T& data) {
    for (auto* dest : destinations) {
        if (!dest) continue;
        inter_group_communicator comm(&source, dest);
        auto result = comm.broadcast(data);
        if (!result) return make_error<void>(result.error().code(), result.error().message());
    }
    return {};
}

/// @brief Receive from multiple source groups
/// @tparam T Data type
/// @param sources Source groups
/// @param dest Destination group
/// @return Data from each source or error
template <typename T>
result<multicast_result<T>> multi_receive(
    const std::vector<rank_group*>& sources,
    rank_group& dest) {
    multicast_result<T> result;
    for (auto* src : sources) {
        if (!src) continue;
        inter_group_communicator comm(src, &dest);
        auto recv_result = comm.template leader_recv<T>();
        if (!recv_result) {
            return make_error<multicast_result<T>>(
                recv_result.error().code(), recv_result.error().message());
        }
        result.data.push_back(std::move(recv_result.value()));
        result.sources.push_back(src->id());
    }
    return result;
}

// ============================================================================
// Pipeline Patterns
// ============================================================================

/// @brief Stage in a pipeline
struct pipeline_stage {
    /// @brief Group for this stage
    rank_group* group = nullptr;

    /// @brief Stage name
    std::string name;

    /// @brief Stage index
    size_type index = 0;
};

/// @brief Pipeline of groups for staged processing
class group_pipeline {
public:
    /// @brief Default constructor
    group_pipeline() = default;

    /// @brief Add a stage to the pipeline
    /// @param group Group for the stage
    /// @param name Stage name
    void add_stage(rank_group* group, std::string name = "") {
        pipeline_stage stage;
        stage.group = group;
        stage.name = std::move(name);
        stage.index = stages_.size();
        stages_.push_back(std::move(stage));
    }

    /// @brief Get number of stages
    [[nodiscard]] size_type size() const noexcept { return stages_.size(); }

    /// @brief Check if pipeline is empty
    [[nodiscard]] bool empty() const noexcept { return stages_.empty(); }

    /// @brief Get stage by index
    /// @param index Stage index
    /// @return Pointer to stage, or nullptr if out of range
    [[nodiscard]] pipeline_stage* stage(size_type index) noexcept {
        if (index < stages_.size()) return &stages_[index];
        return nullptr;
    }

    /// @brief Get all stages
    [[nodiscard]] const std::vector<pipeline_stage>& stages() const noexcept {
        return stages_;
    }

    /// @brief Get communicator between adjacent stages
    /// @param from_stage Source stage index
    /// @return Inter-group communicator or error
    [[nodiscard]] result<inter_group_communicator>
    stage_communicator(size_type from_stage) {
        if (from_stage + 1 >= stages_.size()) {
            return make_error<inter_group_communicator>(
                status_code::out_of_range,
                "Invalid stage index");
        }
        return inter_group_communicator(
            stages_[from_stage].group,
            stages_[from_stage + 1].group);
    }

    /// @brief Forward data through pipeline stages
    /// @tparam T Data type
    /// @param data Initial data (at first stage)
    /// @return Final data (at last stage) or error
    template <typename T>
    result<T> forward(const T& data) {
        T current = data;
        for (size_type i = 0; i + 1 < stages_.size(); ++i) {
            auto comm_result = stage_communicator(i);
            if (!comm_result) {
                return make_error<T>(comm_result.error().code(),
                                     comm_result.error().message());
            }
            auto transfer_result = comm_result.value().broadcast(current);
            if (!transfer_result) {
                return make_error<T>(transfer_result.error().code(),
                                     transfer_result.error().message());
            }
            current = std::move(transfer_result.value());
        }
        return current;
    }

private:
    std::vector<pipeline_stage> stages_;
};

}  // namespace dtl::mpmd
