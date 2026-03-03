# MPI Backend Guide

**Status:** Production-Ready
**Since:** DTL 0.1.0-alpha.1
**Last Updated:** 2026-02-07

## Overview

The MPI (Message Passing Interface) backend provides distributed communication for DTL. It enables containers and algorithms to span multiple processes (ranks), communicating via point-to-point and collective operations. MPI is the primary communication backend and the foundation for multi-node DTL programs.

Key capabilities:

- **Point-to-point communication**: `send`, `recv`, `isend`, `irecv`, `sendrecv`
- **Collective operations**: `broadcast`, `gather`, `scatter`, `allreduce`, `alltoall`, `barrier`
- **Reduction operations**: `reduce`, `allreduce`, `scan`, `exscan` with standard ops
- **Communicator management**: Splitting, duplicating, and sub-group creation
- **Non-blocking operations**: Request-based async communication with `wait`/`test`
- **Thread safety**: Configurable via `thread_support_level`

## Requirements

- **MPI Implementation**: OpenMPI 4.0+, MPICH 3.3+, Intel MPI, or Cray MPICH
- **C++20 compiler** with MPI support
- **CMake** 3.18+

### Installing MPI

```bash
# Ubuntu/Debian
sudo apt install libopenmpi-dev openmpi-bin

# or MPICH
sudo apt install libmpich-dev mpich

# macOS (via Homebrew)
brew install open-mpi

# Verify installation
mpirun --version
```

## CMake Configuration

Enable the MPI backend:

```bash
cmake -DDTL_ENABLE_MPI=ON ..
```

CMake will auto-detect the MPI installation via `find_package(MPI)`.

### Common CMake Flags

| Flag | Default | Description |
|------|---------|-------------|
| `DTL_ENABLE_MPI` | `ON` | Enable MPI backend |
| `MPI_CXX_COMPILER` | Auto | Path to MPI C++ compiler wrapper |

## Initialization

### Using `dtl::environment`

The recommended way to initialize MPI is via DTL's `environment` RAII guard:

```cpp
#include <dtl/dtl.hpp>

int main(int argc, char** argv) {
    // Initializes MPI (and other backends) on first construction
    dtl::environment env(argc, argv);

    // Create a context spanning all ranks
    auto ctx = env.make_world_context();
    std::cout << "Rank " << ctx.rank() << " of " << ctx.size() << "\n";

    // For explicit MPI operations, extract the communicator:
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    // MPI finalized when env goes out of scope
    return 0;
}
```

### Environment Options

Configure MPI thread support and ownership:

```cpp
#include <dtl/core/environment_options.hpp>

auto opts = dtl::environment_options::defaults();

// Request multi-threaded MPI support
opts.mpi.thread_level = dtl::thread_support_level::multiple;

// Or adopt externally initialized MPI
opts.mpi.ownership = dtl::backend_ownership::adopt_external;

dtl::environment env(argc, argv, opts);
```

### Backend Ownership Modes

| Mode | Description |
|------|-------------|
| `dtl_owns` | DTL calls `MPI_Init_thread` / `MPI_Finalize` (default) |
| `adopt_external` | User has already called `MPI_Init`; DTL skips init/finalize |
| `optional` | Initialize if MPI is available, skip silently if not |
| `disabled` | Do not use MPI regardless of availability |

## Communicator Management

### World Communicator

Get the default world communicator (wraps `MPI_COMM_WORLD`):

```cpp
#include <dtl/communication/default_communicator.hpp>

auto comm = dtl::world_comm();
int rank = comm.rank();
int size = comm.size();
```

### MPI Communicator Wrapper

The `mpi_communicator` class wraps `MPI_Comm` with RAII semantics:

```cpp
#include <backends/mpi/mpi_communicator.hpp>

// Wrap MPI_COMM_WORLD (non-owning)
dtl::mpi::mpi_communicator world(MPI_COMM_WORLD, false);

// Query properties
rank_t my_rank = world.rank();
rank_t world_size = world.size();
bool valid = world.valid();
```

### Splitting Communicators

Create sub-groups for collective operations on subsets of ranks:

```cpp
// Split into even/odd groups
int color = comm.rank() % 2;
auto sub_comm = comm.split(color);

// Now sub_comm contains only ranks with the same color
// Collectives on sub_comm only involve those ranks
```

### Duplicating Communicators

Create an independent copy for isolated communication:

```cpp
auto dup_comm = comm.dup();
// dup_comm is independent — collectives on it won't interfere with comm
```

## Point-to-Point Communication

### Blocking Send/Receive

```cpp
#include <dtl/communication/point_to_point.hpp>

auto comm = dtl::world_comm();

if (comm.rank() == 0) {
    // Send data to rank 1
    std::vector<double> data = {1.0, 2.0, 3.0};
    dtl::send(comm, data.data(), data.size(), /*dest=*/1, /*tag=*/0);
} else if (comm.rank() == 1) {
    // Receive data from rank 0
    std::vector<double> data(3);
    dtl::recv(comm, data.data(), data.size(), /*source=*/0, /*tag=*/0);
}
```

### Non-Blocking Communication

```cpp
auto comm = dtl::world_comm();

std::vector<double> send_buf(100, 42.0);
std::vector<double> recv_buf(100);

// Initiate non-blocking operations
auto send_req = dtl::isend(comm, send_buf.data(), 100, /*dest=*/1, /*tag=*/0);
auto recv_req = dtl::irecv(comm, recv_buf.data(), 100, /*source=*/1, /*tag=*/0);

// ... do other work while communication proceeds ...

// Wait for completion
dtl::wait(comm, send_req);
dtl::wait(comm, recv_req);
```

### Send-Receive

Combined send and receive for exchanging data between pairs:

```cpp
std::vector<double> send_buf(100);
std::vector<double> recv_buf(100);

int partner = (comm.rank() + 1) % comm.size();

dtl::sendrecv(comm,
    send_buf.data(), 100, partner, /*send_tag=*/0,
    recv_buf.data(), 100, partner, /*recv_tag=*/0);
```

## Collective Operations

### Barrier

Synchronize all ranks:

```cpp
dtl::barrier(comm);  // Blocks until all ranks reach this point
```

### Broadcast

Send data from one rank to all:

```cpp
std::vector<double> data(100);

if (comm.rank() == 0) {
    // Root initializes data
    std::fill(data.begin(), data.end(), 3.14);
}

// All ranks receive root's data
dtl::broadcast(comm, data.data(), data.size(), /*root=*/0);
```

### Gather and Scatter

```cpp
// Gather: collect from all ranks to root
std::vector<double> local_data(10);
std::vector<double> all_data;  // Only meaningful on root

if (comm.rank() == 0) {
    all_data.resize(10 * comm.size());
}

dtl::gather(comm, local_data.data(), 10,
            all_data.data(), 10, /*root=*/0);

// Scatter: distribute from root to all ranks
dtl::scatter(comm, all_data.data(), 10,
             local_data.data(), 10, /*root=*/0);
```

### Allreduce

Reduce and distribute result to all ranks:

```cpp
double local_sum = compute_local_sum();
double global_sum;

dtl::allreduce(comm, &local_sum, &global_sum, dtl::reduce_sum<double>{});

// global_sum is the same on all ranks
```

### All-to-All

Exchange data between all pairs of ranks:

```cpp
std::vector<double> send_buf(comm.size());  // One element per rank
std::vector<double> recv_buf(comm.size());

dtl::alltoall(comm, send_buf.data(), 1, recv_buf.data(), 1);
```

## Collective Best Practices

1. **All ranks must participate** in collective operations. If one rank skips a collective, the program will deadlock.

2. **Match arguments across ranks**: All ranks must pass the same `count`, `dtype`, `root`, and `op` to collective calls.

3. **Use in-place variants** when possible to avoid extra buffer allocations:
   ```cpp
   dtl::allreduce_inplace(comm, &value, dtl::reduce_sum<double>{});
   ```

4. **Prefer allreduce over reduce + broadcast** when all ranks need the result.

5. **Batch small messages** into fewer larger messages to reduce latency overhead.

## Deadlock Avoidance

### Common Deadlock Patterns

**Mismatched collectives:**
```cpp
// DEADLOCK: rank 0 calls barrier, rank 1 calls reduce
if (comm.rank() == 0) {
    dtl::barrier(comm);
} else {
    dtl::reduce(comm, &data, &result, op, 0);
}
```

**Circular send/recv:**
```cpp
// DEADLOCK: both ranks block on send before posting recv
dtl::send(comm, data, count, partner, tag);    // Blocks!
dtl::recv(comm, data, count, partner, tag);
```

### Safe Patterns

**Use sendrecv for exchanges:**
```cpp
// SAFE: combined send/receive
dtl::sendrecv(comm,
    send_buf, count, partner, tag,
    recv_buf, count, partner, tag);
```

**Use non-blocking operations:**
```cpp
// SAFE: non-blocking send before blocking recv
auto req = dtl::isend(comm, data, count, partner, tag);
dtl::recv(comm, recv_data, count, partner, tag);
dtl::wait(comm, req);
```

**Ensure all ranks follow the same control flow for collectives:**
```cpp
// SAFE: all ranks execute the same collective
dtl::barrier(comm);
auto result = dtl::allreduce(comm, &local, &global, op);
```

## DTL Algorithm Integration

DTL algorithms use the MPI communicator for distributed phases:

```cpp
auto comm = dtl::world_comm();
dtl::distributed_vector<int> vec(100000, comm.size(), comm.rank());

// Distributed reduce (local reduce + MPI allreduce)
auto sum = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);

// Distributed sort (local sort + MPI sample exchange)
dtl::sort(dtl::par{}, vec, std::less<>{}, comm);

// Distributed scan (local scan + MPI prefix scan)
dtl::distributed_vector<int> output(100000, comm.size(), comm.rank());
dtl::inclusive_scan(dtl::par{}, vec, output, 0, std::plus<>{});
```

## Performance Tips

### Reduce Communication Volume

- Use `allreduce` instead of `reduce` + `broadcast`
- Batch multiple small messages into a single larger message
- Use variable-count variants (`gatherv`, `scatterv`) to avoid padding

### Overlap Communication and Computation

```cpp
// Post non-blocking receive
auto req = dtl::irecv(comm, recv_buf, count, source, tag);

// Do local computation while data arrives
compute_local_work();

// Wait for communication to finish
dtl::wait(comm, req);

// Process received data
process_received(recv_buf);
```

### Choose the Right MPI Implementation

| Implementation | Strengths |
|---------------|-----------|
| OpenMPI | General-purpose, good defaults, wide platform support |
| MPICH | Excellent standards compliance, good for development |
| Intel MPI | Optimized for Intel hardware, fabric-aware |
| Cray MPICH | Optimized for Cray/HPE interconnects |

### Thread Safety

For multi-threaded MPI usage, request `thread_support_level::multiple`:

```cpp
auto opts = dtl::environment_options::defaults();
opts.mpi.thread_level = dtl::thread_support_level::multiple;
dtl::environment env(argc, argv, opts);
```

### Launching MPI Programs

```bash
# OpenMPI
mpirun -np 4 ./my_dtl_app

# MPICH
mpiexec -n 4 ./my_dtl_app

# SLURM
srun -n 4 ./my_dtl_app
```

## See Also

- [CPU Backend Guide](cpu_guide.md) — Local execution policies
- [NCCL Backend](nccl_backend.md) — GPU-to-GPU collectives
- [OpenSHMEM Backend](shmem_backend.md) — PGAS one-sided communication
- [Backend Comparison](comparison.md) — Feature comparison across backends
