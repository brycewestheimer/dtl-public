# Getting Started with DTL

This guide walks you through installing DTL, building your first program, and running with MPI.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Quick Install (Header-Only)](#quick-install-header-only)
  - [Building from Source](#building-from-source)
  - [CMake Integration](#cmake-integration)
- [Hello Distributed](#hello-distributed)
- [Running with MPI](#running-with-mpi)
- [Next Steps](#next-steps)

---

## Requirements

### Compiler Support

DTL requires a C++20-compliant compiler with full `<source_location>` support:

| Compiler | Minimum Version | Notes |
|----------|-----------------|-------|
| GCC | 11.0 | Full C++20 including `<source_location>` |
| Clang | 15.0 | Full C++20 including `<source_location>` |
| MSVC | 19.29 (VS 2019 16.10) | Requires `/std:c++20` |
| NVCC | 11.4 | Requires compatible host compiler |

### Optional Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| MPI | Multi-node distribution | OpenMPI: `apt install openmpi-bin libopenmpi-dev` |
| CUDA Toolkit | GPU acceleration | [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| GTest | Unit tests | `apt install libgtest-dev` |
| Google Benchmark | Benchmarks | `apt install libbenchmark-dev` |

---

## Installation

### Quick Install (Header-Only)

DTL is header-only for most use cases. Simply copy the `include/dtl` directory to your project or install system-wide:

```bash
# Clone the repository
git clone https://github.com/brycewestheimer/dtl-public.git
cd dtl

# Install to /usr/local/include
sudo cp -r include/dtl /usr/local/include/
```

Then include in your code:

```cpp
#include <dtl/dtl.hpp>
```

### Building from Source

For tests, examples, and backends:

```bash
# Clone and create build directory
git clone https://github.com/brycewestheimer/dtl-public.git
cd dtl
mkdir build && cd build

# Configure (basic)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Installing with Spack

DTL includes a local Spack repository in [`spack/`](../spack/README.md):

```bash
git clone https://github.com/brycewestheimer/dtl-public.git
cd dtl
spack repo add ./spack
spack install dtl
spack install dtl +tests
spack install dtl +python +c_bindings
```

#### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `DTL_BUILD_TESTS` | ON | Build unit test suite |
| `DTL_BUILD_EXAMPLES` | ON | Build example programs |
| `DTL_BUILD_BENCHMARKS` | OFF | Build performance benchmarks |
| `DTL_BUILD_DOCS` | OFF | Build Doxygen documentation |
| `DTL_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `DTL_ENABLE_HIP` | OFF | Enable HIP/AMD backend |
| `DTL_ENABLE_NCCL` | OFF | Enable NCCL communicator |

#### Build with GPU Support

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DDTL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j$(nproc)
```

### CMake Integration

#### Using `find_package`

After installing DTL:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_project)

find_package(DTL REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DTL::dtl)
```

#### Using `add_subdirectory`

Add DTL as a subdirectory in your project:

```cmake
add_subdirectory(external/dtl)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DTL::dtl)
```

#### Using CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  dtl
  GIT_REPOSITORY https://github.com/brycewestheimer/dtl-public.git
  GIT_TAG v0.1.0-alpha.1
)
FetchContent_MakeAvailable(dtl)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DTL::dtl)
```

---

## Hello Distributed

Let's create a simple program that demonstrates DTL's core concepts.

### The Code

Create `hello_distributed.cpp`:

```cpp
/// hello_distributed.cpp - Your first DTL program
#include <dtl/dtl.hpp>
#include <iostream>

int main() {
    std::cout << "DTL Hello Distributed Example\n\n";

    // Create a distributed vector in standalone mode
    // (single rank, no MPI initialization required)
    const dtl::size_type global_size = 100;
    const auto ctx = dtl::make_cpu_context();

    dtl::distributed_vector<int> vec(global_size, ctx);

    std::cout << "Created distributed_vector with " << global_size << " elements\n";
    std::cout << "Rank: " << vec.rank() << ", Total ranks: " << vec.num_ranks() << "\n";
    std::cout << "Global size: " << vec.global_size()
              << ", Local size: " << vec.local_size() << "\n\n";

    // Fill with values using local view (STL-compatible, no communication)
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i);
    }

    std::cout << "Filling with values 0, 1, 2, ...\n";
    std::cout << "First 10 elements: ";
    for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    // Use DTL's for_each algorithm to transform elements
    std::cout << "Using for_each to square each element...\n";
    dtl::for_each(vec, [](int& x) { x = x * x; });

    std::cout << "First 10 elements after squaring: ";
    for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    // Use local_reduce (no MPI communication needed)
    int sum = dtl::local_reduce(vec, 0, std::plus<>{});
    std::cout << "Local sum (no MPI required): " << sum << "\n";

    // Verify: sum of i^2 for i=0..99 = 99*100*199/6 = 328350
    int expected = 99 * 100 * 199 / 6;
    std::cout << "Expected sum of squares: " << expected << "\n";

    if (sum == expected) {
        std::cout << "SUCCESS!\n";
        return 0;
    } else {
        std::cout << "FAILURE: sums don't match\n";
        return 1;
    }
}
```

### Build and Run

```bash
# Using CMake (if DTL is installed)
g++ -std=c++20 -I/usr/local/include hello_distributed.cpp -o hello_distributed
./hello_distributed
```

### Expected Output

```
DTL Hello Distributed Example

Created distributed_vector with 100 elements
Rank: 0, Total ranks: 1
Global size: 100, Local size: 100

Filling with values 0, 1, 2, ...
First 10 elements: 0 1 2 3 4 5 6 7 8 9

Using for_each to square each element...
First 10 elements after squaring: 0 1 4 9 16 25 36 49 64 81

Local sum (no MPI required): 328350
Expected sum of squares: 328350
SUCCESS!
```

### Key Concepts Demonstrated

1. **`distributed_vector<T>`** - A distributed container that partitions data across ranks
2. **`local_view()`** - Returns an STL-compatible view of locally-owned elements (no communication)
3. **`dtl::for_each`** - Applies a function to all elements (operates on local partition)
4. **`dtl::local_reduce`** - Reduces local elements without communication

### Using dtl::environment (Recommended)

For production code, use `dtl::environment` to manage backend lifecycle:

```cpp
#include <dtl/dtl.hpp>

int main(int argc, char** argv) {
    // Environment handles MPI_Init/MPI_Finalize automatically
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    dtl::distributed_vector<double> vec(1000, ctx);
    // ... use DTL ...

    return 0;  // MPI_Finalize called automatically
}
```

See the [Environment Guide](user_guide/environment.md) for details.

---

## Running with MPI

For true distributed execution, use MPI:

### MPI-Enabled Program

Create `mpi_vector_sum.cpp`:

```cpp
/// mpi_vector_sum.cpp - DTL with MPI
#include <dtl/dtl.hpp>

#include <iostream>
#include <functional>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto comm = dtl::world_comm();

    // Create distributed vector partitioned across all ranks
    const dtl::size_type global_size = 1000;
    dtl::distributed_vector<double> vec(global_size, comm);

    // Each rank fills its local partition
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<double>(vec.global_offset() + i);
    }

    // Local partial sum
    double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});

    // Global reduction using DTL communicator adapter
    double global_sum = comm.allreduce_sum_value<double>(local_sum);

    if (comm.rank() == 0) {
        double expected = (global_size - 1) * global_size / 2.0;
        std::cout << "Global sum: " << global_sum << "\n";
        std::cout << "Expected: " << expected << "\n";
        std::cout << (global_sum == expected ? "SUCCESS!" : "FAILURE!") << "\n";
    }

    return 0;
}
```

### Build with MPI

```bash
mpicxx -std=c++20 -I/usr/local/include mpi_vector_sum.cpp -o mpi_vector_sum
```

### Run with Multiple Ranks

```bash
# Run with 4 MPI ranks
mpirun -np 4 ./mpi_vector_sum

# Expected output (from rank 0):
# Global sum: 499500
# Expected: 499500
# SUCCESS!
```

---

## Platform-Specific Setup

### Ubuntu/Debian (WSL2)

```bash
# Install dependencies
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build \
  openmpi-bin libopenmpi-dev \
  libgtest-dev libbenchmark-dev

# Verify MPI
mpirun --version

# Build DTL with tests
mkdir build && cd build
cmake .. -DDTL_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

### WSL2 with CUDA

WSL2 supports CUDA through GPU passthrough (requires Windows NVIDIA driver):

```bash
# Add NVIDIA WSL repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA toolkit (NOT the driver)
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Verify
nvidia-smi   # Shows GPU info
nvcc --version  # Shows CUDA toolkit version

# Build DTL with CUDA
cmake .. -DDTL_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j$(nproc)
```

---

## Troubleshooting

### MPI Not Found

If CMake cannot find MPI:

```bash
# Verify MPI is installed
which mpicc mpicxx

# Tell CMake explicitly
cmake .. -DMPI_CXX_COMPILER=$(which mpicxx) -DMPI_C_COMPILER=$(which mpicc)
```

### Conda MPI Conflicts

If using Conda, its MPI may conflict with system MPI:

```bash
# Option 1: Use clean PATH
PATH=/usr/local/bin:/usr/bin:/bin cmake ..

# Option 2: Deactivate conda
conda deactivate
cmake ..
```

### CUDA Compilation Errors

Ensure host compiler is compatible with NVCC:

```bash
# Check NVCC's supported compilers
nvcc --help | grep -A5 "host-compiler"

# Specify a compatible host compiler
cmake .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
```

---

## Language Bindings

DTL provides bindings for languages beyond C++:

### Python

```bash
# Build Python bindings
cmake .. -DDTL_BUILD_PYTHON=ON
make _dtl
make python_install
```

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    local = vec.local_view()  # Zero-copy NumPy array
    local[:] = np.arange(len(local))

    global_sum = dtl.allreduce(ctx, np.sum(local), op=dtl.SUM)
```

### C

```bash
# Build C bindings
cmake .. -DDTL_BUILD_C_BINDINGS=ON
make dtl_c
```

```c
#include <dtl/bindings/c/dtl.h>

dtl_context_t ctx;
dtl_context_create_default(&ctx);
printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));
dtl_context_destroy(ctx);
```

See the [Language Bindings Guide](user_guide/bindings.md) for complete documentation.

---

## Next Steps

Now that you have DTL running, explore:

1. **[Containers Guide](user_guide/containers.md)** - Learn about `distributed_vector`, `distributed_array`, and `distributed_tensor`
2. **[Views Guide](user_guide/views.md)** - Understand `local_view`, `global_view`, and `remote_ref`
3. **[Algorithms Guide](user_guide/algorithms.md)** - Explore DTL's distributed algorithms
4. **[Language Bindings](user_guide/bindings.md)** - Use DTL from Python, C, or Fortran
5. **[Examples](../examples/)** - Browse more complete examples in the repository

### Example Programs

After building with `DTL_BUILD_EXAMPLES=ON`:

```bash
# Basic examples
./examples/basics/hello_distributed
./examples/basics/local_view_stl

# Algorithm examples
./examples/algorithms/parallel_reduce
./examples/algorithms/transform_reduce
./examples/algorithms/distributed_sort

# GPU examples (requires CUDA)
./examples/gpu/gpu_accelerated_transform
```
