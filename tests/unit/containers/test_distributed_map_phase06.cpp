// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_map_phase06.cpp
/// @brief Phase 06 conformance tests for distributed_map ownership and transport

#include <dtl/containers/distributed_map.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace dtl::test {

namespace {

struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};

struct single_rank_collective_comm {
    template <typename T>
    [[nodiscard]] T allreduce_sum_value(const T& value) const {
        return value;
    }

    [[nodiscard]] bool allreduce_lor_value(bool value) const {
        return value;
    }

    void alltoall(const void* sendbuf, void* recvbuf, size_type count) {
        auto* recv_bytes = static_cast<std::byte*>(recvbuf);
        const auto* send_bytes = static_cast<const std::byte*>(sendbuf);

        // Emulate rank 0 contribution only: recv[0] = send[0], recv[1..] = 0
        std::memcpy(recv_bytes, send_bytes, count);
        std::memset(recv_bytes + count, 0, count);
    }

    void alltoallv(const void* sendbuf,
                   const int* sendcounts,
                   const int* senddispls,
                   void* recvbuf,
                   const int* recvcounts,
                   const int* recvdispls,
                   size_type elem_size) {
        auto* recv_bytes = static_cast<std::byte*>(recvbuf);
        const auto* send_bytes = static_cast<const std::byte*>(sendbuf);

        if (recvcounts[0] > 0 && sendcounts[0] > 0) {
            const size_type recv_offset = static_cast<size_type>(recvdispls[0]) * elem_size;
            const size_type send_offset = static_cast<size_type>(senddispls[0]) * elem_size;
            const size_type copy_count = static_cast<size_type>(std::min(recvcounts[0], sendcounts[0]));
            std::memcpy(recv_bytes + recv_offset,
                        send_bytes + send_offset,
                        copy_count * elem_size);
        }
    }
};

[[nodiscard]] int find_remote_int_key(const distributed_map<int, int>& map) {
    for (int key = 0; key < 4096; ++key) {
        if (!map.is_local(key)) {
            return key;
        }
    }
    return -1;
}

[[nodiscard]] int find_local_int_key(const distributed_map<int, int>& map) {
    for (int key = 0; key < 4096; ++key) {
        if (map.is_local(key)) {
            return key;
        }
    }
    return -1;
}

[[nodiscard]] std::string find_remote_string_key(
    const distributed_map<std::string, std::string>& map) {
    for (int i = 0; i < 4096; ++i) {
        std::string key = "remote_" + std::to_string(i);
        if (!map.is_local(key)) {
            return key;
        }
    }
    return {};
}

struct non_serializable_key {
    int value = 0;
    std::string marker;
    non_serializable_key() : marker("x") {}
};

struct non_serializable_hash {
    size_t operator()(const non_serializable_key& key) const noexcept {
        return static_cast<size_t>(key.value);
    }
};

struct non_serializable_equal {
    bool operator()(const non_serializable_key& lhs,
                    const non_serializable_key& rhs) const noexcept {
        return lhs.value == rhs.value;
    }
};

using non_serializable_map = distributed_map<
    non_serializable_key,
    int,
    non_serializable_hash,
    non_serializable_equal>;

static_assert(distributed_map<int, std::string>::transport_contract_satisfied());
static_assert(!non_serializable_map::transport_contract_satisfied());

}  // namespace

TEST(DistributedMapPhase06Test, OperatorBracketLocalKeyDefaultInserts) {
    distributed_map<int, int> map(test_context{0, 4});
    const int local_key = find_local_int_key(map);
    ASSERT_GE(local_key, 0);

    auto ref = map[local_key];
    EXPECT_TRUE(ref.is_local());

    auto value = ref.get();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), 0);
    EXPECT_TRUE(map.contains_local(local_key));
}

TEST(DistributedMapPhase06Test, OperatorBracketRemoteKeyNoLocalInsertion) {
    distributed_map<int, int> map(test_context{0, 4});
    const int remote_key = find_remote_int_key(map);
    ASSERT_GE(remote_key, 0);

    const auto local_before = map.local_size();
    auto ref = map[remote_key];

    EXPECT_TRUE(ref.is_remote());
    EXPECT_EQ(map.local_size(), local_before);
    EXPECT_FALSE(map.contains_local(remote_key));
    EXPECT_TRUE(map.has_legacy_ownership_diagnostic());

    auto diag = map.legacy_ownership_diagnostic();
    ASSERT_TRUE(diag.has_value());
    EXPECT_NE(diag->find("operator[]"), std::string::npos);
}

TEST(DistributedMapPhase06Test, LocalAndRemoteExplicitApisEnforceOwnership) {
    distributed_map<int, int> map(test_context{0, 4});
    const int local_key = find_local_int_key(map);
    const int remote_key = find_remote_int_key(map);

    ASSERT_GE(local_key, 0);
    ASSERT_GE(remote_key, 0);

    auto local_via_remote = map.insert_remote(local_key, 10);
    EXPECT_FALSE(local_via_remote.has_value());
    EXPECT_EQ(local_via_remote.error().code(), status_code::precondition_failed);

    auto remote_via_local = map.insert_local(remote_key, 20);
    EXPECT_FALSE(remote_via_local.has_value());
    EXPECT_EQ(remote_via_local.error().code(), status_code::precondition_failed);

    auto queued_remote = map.insert_remote(remote_key, 33);
    EXPECT_TRUE(queued_remote.has_value());

    auto queued_erase = map.erase_remote(remote_key);
    EXPECT_TRUE(queued_erase.has_value());
}

TEST(DistributedMapPhase06Test, LegacyImplicitRemoteMutationReportsDiagnostic) {
    distributed_map<int, int> map(test_context{0, 4});
    const int remote_key = find_remote_int_key(map);
    ASSERT_GE(remote_key, 0);

    auto insert_result = map.insert(remote_key, 101);
    EXPECT_TRUE(insert_result.has_value());

    EXPECT_TRUE(map.has_legacy_ownership_diagnostic());
    auto diag = map.legacy_ownership_diagnostic();
    ASSERT_TRUE(diag.has_value());
    EXPECT_NE(diag->find("insert"), std::string::npos);

    map.clear_legacy_ownership_diagnostic();
    EXPECT_FALSE(map.has_legacy_ownership_diagnostic());
}

TEST(DistributedMapPhase06Test, FlushPendingWithCommSupportsNonTrivialTransport) {
    distributed_map<std::string, std::string> map(test_context{0, 2});
    single_rank_collective_comm comm;

    const std::string remote_key = find_remote_string_key(map);
    ASSERT_FALSE(remote_key.empty());

    auto enqueue = map.insert_or_assign_remote(remote_key, std::string("value"));
    ASSERT_TRUE(enqueue.has_value());

    auto flush = map.flush_pending_with_comm(comm);
    EXPECT_TRUE(flush.has_value());
}

}  // namespace dtl::test
