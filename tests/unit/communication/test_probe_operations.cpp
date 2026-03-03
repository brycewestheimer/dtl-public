// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_probe_operations.cpp
/// @brief Unit tests for probe operations (MPI_Probe, MPI_Iprobe)
/// @details Tests blocking and non-blocking probe through all communication layers.
/// @since 0.1.0

#include <gtest/gtest.h>
#include <dtl/communication/point_to_point.hpp>
#include <dtl/communication/communicator_base.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

namespace dtl {
namespace {

#if DTL_ENABLE_MPI
#endif

// ============================================================================
// Null Communicator Tests (Single-Rank Behavior)
// ============================================================================

TEST(ProbeOperations, NullCommunicatorProbe) {
    null_communicator comm;

    // Probe should return a default status for single-rank
    auto status = comm.probe(0, 0);
    EXPECT_EQ(status.source, 0);
    EXPECT_EQ(status.tag, 0);
    EXPECT_EQ(status.count, 0u);
    EXPECT_FALSE(status.cancelled);
    EXPECT_EQ(status.error, 0);
}

TEST(ProbeOperations, NullCommunicatorIprobe) {
    null_communicator comm;

    // Iprobe should always return false (no messages pending in single-rank)
    auto [available, status] = comm.iprobe(0, 0);
    EXPECT_FALSE(available);
    EXPECT_EQ(status.source, no_rank);  // Default-initialized
    EXPECT_EQ(status.tag, 0);
    EXPECT_EQ(status.count, 0u);
}

TEST(ProbeOperations, NullCommunicatorProbeAnySource) {
    null_communicator comm;

    // Probe with any_source should work (though meaningless in single-rank)
    auto status = comm.probe(any_source, any_tag);
    EXPECT_EQ(status.source, 0);
    EXPECT_EQ(status.tag, 0);
}

TEST(ProbeOperations, NullCommunicatorIprobeAnySource) {
    null_communicator comm;

    // Iprobe with any_source should return false
    auto [available, status] = comm.iprobe(any_source, any_tag);
    EXPECT_FALSE(available);
}

// ============================================================================
// Free Function Template Tests (Compile-Time Verification)
// ============================================================================

TEST(ProbeOperations, FreeFunctionProbeCompiles) {
    null_communicator comm;

    // Verify free function template compiles and works
    auto result = probe(comm, 0, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 0);
}

TEST(ProbeOperations, FreeFunctionIprobeCompiles) {
    null_communicator comm;

    // Verify free function template compiles and works
    auto result = iprobe(comm, 0, 0);
    ASSERT_TRUE(result.has_value());
    auto [available, status] = *result;
    EXPECT_FALSE(available);
}

TEST(ProbeOperations, FreeFunctionProbeWithWildcards) {
    null_communicator comm;

    // Test with any_source and any_tag
    auto result = probe(comm, any_source, any_tag);
    ASSERT_TRUE(result.has_value());
}

TEST(ProbeOperations, FreeFunctionIprobeWithWildcards) {
    null_communicator comm;

    // Test with any_source and any_tag
    auto result = iprobe(comm, any_source, any_tag);
    ASSERT_TRUE(result.has_value());
    auto [available, status] = *result;
    EXPECT_FALSE(available);
}

#if DTL_ENABLE_MPI
// ============================================================================
// MPI Communicator Tests (Multi-Rank Behavior)
// ============================================================================

TEST(MpiProbeTest, MPIAdapterProbeMessageAvailable) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int tag = 42;
    const int value = 123;

    if (comm.rank() == 0) {
        // Rank 0: Send a message, then probe should find it
        comm.send(&value, sizeof(int), 1, tag);
    } else if (comm.rank() == 1) {
        // Rank 1: Probe for the message (should block until available)
        auto status = comm.probe(0, tag);
        EXPECT_EQ(status.source, 0);
        EXPECT_EQ(status.tag, tag);
        EXPECT_EQ(status.count, sizeof(int));

        // Now receive the message
        int received = 0;
        comm.recv(&received, sizeof(int), 0, tag);
        EXPECT_EQ(received, value);
    }
}

TEST(MpiProbeTest, MPIAdapterIprobeNoMessage) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    // Rank 1: Iprobe for a message that doesn't exist yet
    if (comm.rank() == 1) {
        auto [available, status] = comm.iprobe(0, 99);
        EXPECT_FALSE(available);
    }

    comm.barrier();
}

TEST(MpiProbeTest, MPIAdapterIprobeMessageAvailable) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int tag = 43;
    const int value = 456;

    if (comm.rank() == 0) {
        // Rank 0: Send a message
        comm.send(&value, sizeof(int), 1, tag);
    } else if (comm.rank() == 1) {
        // Rank 1: Poll with iprobe until message arrives
        bool found = false;
        for (int i = 0; i < 1000 && !found; ++i) {
            auto [available, status] = comm.iprobe(0, tag);
            if (available) {
                EXPECT_EQ(status.source, 0);
                EXPECT_EQ(status.tag, tag);
                EXPECT_EQ(status.count, sizeof(int));
                found = true;
            }
        }
        EXPECT_TRUE(found);

        // Receive the message
        int received = 0;
        comm.recv(&received, sizeof(int), 0, tag);
        EXPECT_EQ(received, value);
    }
}

TEST(MpiProbeTest, MPIAdapterProbeAnySource) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int tag = 44;
    const int value = 789;

    if (comm.rank() == 0) {
        // All non-root ranks send to rank 0
        if (comm.rank() != 0) {
            comm.send(&value, sizeof(int), 0, tag);
        }
    } else {
        comm.send(&value, sizeof(int), 0, tag);
    }

    if (comm.rank() == 0 && comm.size() > 1) {
        // Rank 0: Probe for message from any source
        auto status = comm.probe(any_source, tag);
        EXPECT_GE(status.source, 1);  // Message came from some rank > 0
        EXPECT_LT(status.source, comm.size());
        EXPECT_EQ(status.tag, tag);
        EXPECT_EQ(status.count, sizeof(int));

        // Receive the message
        int received = 0;
        comm.recv(&received, sizeof(int), status.source, tag);
        EXPECT_EQ(received, value);
    }

    comm.barrier();
}

TEST(MpiProbeTest, MPIAdapterProbeAnyTag) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int value = 321;
    constexpr int sent_tag = 100;

    if (comm.rank() == 0) {
        // Rank 0: Send to rank 1 with a known tag
        comm.send(&value, sizeof(int), 1, sent_tag);
    } else if (comm.rank() == 1) {
        // Rank 1: Probe for message with any tag
        auto status = comm.probe(0, any_tag);
        EXPECT_EQ(status.source, 0);
        EXPECT_EQ(status.tag, sent_tag);  // Should match the actual tag sent
        EXPECT_EQ(status.count, sizeof(int));

        // Receive the message
        int received = 0;
        comm.recv(&received, sizeof(int), 0, status.tag);
        EXPECT_EQ(received, value);
    }

    comm.barrier();
}

TEST(MpiProbeTest, FreeFunctionMPIProbe) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int tag = 45;
    const int value = 555;

    if (comm.rank() == 0) {
        comm.send(&value, sizeof(int), 1, tag);
    } else if (comm.rank() == 1) {
        // Use free function probe
        auto result = probe(comm, 0, tag);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->source, 0);
        EXPECT_EQ(result->tag, tag);

        int received = 0;
        comm.recv(&received, sizeof(int), 0, tag);
        EXPECT_EQ(received, value);
    }

    comm.barrier();
}

TEST(MpiProbeTest, FreeFunctionMPIIprobe) {
    mpi::mpi_comm_adapter comm = mpi::world_adapter();

    if (comm.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int tag = 46;
    const int value = 666;

    if (comm.rank() == 0) {
        comm.send(&value, sizeof(int), 1, tag);
    } else if (comm.rank() == 1) {
        // Use free function iprobe
        bool found = false;
        for (int i = 0; i < 1000 && !found; ++i) {
            auto result = iprobe(comm, 0, tag);
            ASSERT_TRUE(result.has_value());
            auto [available, status] = *result;
            if (available) {
                EXPECT_EQ(status.source, 0);
                EXPECT_EQ(status.tag, tag);
                found = true;
            }
        }
        EXPECT_TRUE(found);

        int received = 0;
        comm.recv(&received, sizeof(int), 0, tag);
        EXPECT_EQ(received, value);
    }

    comm.barrier();
}

#endif  // DTL_ENABLE_MPI

}  // namespace
}  // namespace dtl
