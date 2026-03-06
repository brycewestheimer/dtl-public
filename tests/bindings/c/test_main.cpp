// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#ifdef DTL_HAS_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#ifdef DTL_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);

    const bool init_mpi_in_main = !mpi_initialized && !mpi_finalized;
    if (init_mpi_in_main) {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#ifdef DTL_HAS_MPI
    mpi_initialized = 0;
    mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    MPI_Finalized(&mpi_finalized);
    if (init_mpi_in_main && mpi_initialized && !mpi_finalized) {
        MPI_Finalize();
    }
#endif

    return result;
}
