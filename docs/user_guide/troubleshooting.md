# Legacy Deep-Dive: Troubleshooting

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [11-troubleshooting-and-diagnostics.md](11-troubleshooting-and-diagnostics.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


This guide covers common issues when building, configuring, and running DTL applications, along with their solutions.

---

## Table of Contents

- [MPI Initialization Failures](#mpi-initialization-failures)
  - [Conda MPI Conflicts](#conda-mpi-conflicts)
  - [Thread Level Mismatch](#thread-level-mismatch)
  - [MPI Already Initialized](#mpi-already-initialized)
  - [MPI Not Found During Build](#mpi-not-found-during-build)
- [CUDA Device Selection Issues](#cuda-device-selection-issues)
  - [No CUDA Devices Available](#no-cuda-devices-available)
  - [Wrong Device Selected](#wrong-device-selected)
  - [NVCC Not on PATH](#nvcc-not-on-path)
- [NCCL Communicator Creation Failures](#nccl-communicator-creation-failures)
  - [NCCL Not Found](#nccl-not-found)
  - [Single-GPU Testing](#single-gpu-testing)
- [Deadlocks in Collective Operations](#deadlocks-in-collective-operations)
  - [Mismatched Collective Calls](#mismatched-collective-calls)
  - [Barrier Placement Errors](#barrier-placement-errors)
  - [Conditional Collective Calls](#conditional-collective-calls)
- [Futures Timeout Diagnostics](#futures-timeout-diagnostics)
  - [Future Never Completes](#future-never-completes)
  - [Progress Engine Not Polled](#progress-engine-not-polled)
- [Build System Issues](#build-system-issues)
  - [FindMPI.cmake Empty File](#findmpicmake-empty-file)
  - [FindNCCL.cmake Not Found](#findncclcmake-not-found)
  - [Compiler Version Too Old](#compiler-version-too-old)
  - [libdtl_runtime.so Not Found at Runtime](#libdtl_runtimeso-not-found-at-runtime)
- [Python Binding Import Errors](#python-binding-import-errors)
  - [Module Not Found](#module-not-found)
  - [MPI Library Mismatch](#mpi-library-mismatch)
  - [NumPy Version Incompatibility](#numpy-version-incompatibility)

---

## MPI Initialization Failures

### Conda MPI Conflicts

**Symptom:** Build fails with "x86_64-conda-linux-gnu-cc: not found" or runtime crashes with MPI version mismatches.

**Cause:** Conda installs its own OpenMPI (often 5.0.x) with broken compiler wrappers. Conda's `mpicc` cannot find its expected cross-compiler, and conda's linker (`compiler_compat/ld`) cannot resolve system OpenMPI's shared libraries.

**Solution:** Always pass explicit system compiler paths to CMake when conda is on PATH:

```bash
cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DMPI_C_COMPILER=/usr/bin/mpicc \
  -DMPI_CXX_COMPILER=/usr/bin/mpicxx
```

For Python packages that link MPI (e.g., `mpi4py`), build from source against system MPI:

```bash
MPICC=/usr/bin/mpicc CC=/usr/bin/gcc LDSHARED="/usr/bin/gcc -shared" \
  pip install --no-binary mpi4py --force-reinstall mpi4py
```

**Verification:**

```bash
# Ensure system MPI is used, NOT conda's
which mpicc          # Should be /usr/bin/mpicc
mpicc --version      # Should show system GCC, not conda wrapper
mpirun --version     # Should match the expected OpenMPI version
```

### Thread Level Mismatch

**Symptom:** Runtime warning: "MPI thread level requested X but only Y provided." Or unexpected behavior with `par{}` or `async{}` execution policies.

**Cause:** The MPI implementation does not support the requested thread safety level. For example, requesting `MPI_THREAD_MULTIPLE` when the MPI library was built without thread support.

**Solution:** Check what your MPI supports and request an appropriate level:

```cpp
auto opts = dtl::environment_options::defaults();
// Check what level was actually provided
dtl::environment env(argc, argv, opts);
std::cout << "Thread level: " << env.mpi_thread_level_name() << "\n";
```

If you need `MPI_THREAD_MULTIPLE`, ensure your MPI library was configured with `--enable-mpi-thread-multiple` (OpenMPI) or similar.

### MPI Already Initialized

**Symptom:** Error "MPI_Init has already been called" or "environment construction failed."

**Cause:** Another library or framework initialized MPI before DTL's `environment` constructor.

**Solution:** Use `adopt_external` mode to let DTL use the existing MPI initialization:

```cpp
auto opts = dtl::environment_options::defaults();
opts.mpi_mode = dtl::backend_ownership::adopt_external;
dtl::environment env(argc, argv, opts);
```

Or, for library authors, inject an existing communicator:

```cpp
// Your library receives an MPI communicator from the application
auto env = dtl::environment::from_comm(app_comm);
```

### MPI Not Found During Build

**Symptom:** CMake error: "Could NOT find MPI" or MPI headers not found.

**Cause:** CMake's `FindMPI` module cannot locate your MPI installation.

**Solution:**

1. Ensure MPI is installed: `sudo apt install openmpi-bin libopenmpi-dev`
2. Check for an empty `cmake/FindMPI.cmake` file in the DTL source tree. If it exists and is empty, delete it -- it shadows CMake's built-in FindMPI module.
3. Set explicit MPI compiler paths:

```bash
cmake .. -DMPI_C_COMPILER=/usr/bin/mpicc -DMPI_CXX_COMPILER=/usr/bin/mpicxx
```

---

## CUDA Device Selection Issues

### No CUDA Devices Available

**Symptom:** `nvidia-smi` shows "No devices found" or DTL reports `has_cuda() == false`.

**Cause (WSL2):** GPU passthrough requires:
- NVIDIA driver on the Windows host (not in WSL2)
- CUDA toolkit installed in WSL2 (toolkit only, not the driver)
- The `/dev/dxg` device accessible in WSL2

**Solution (WSL2):**

1. Install NVIDIA driver on Windows (not in WSL2)
2. Install only the CUDA toolkit in WSL2:

```bash
sudo apt install cuda-toolkit-12-6
```

3. Do NOT install `nvidia-driver-*` packages inside WSL2
4. Verify:

```bash
nvidia-smi               # Should show GPU via passthrough
nvcc --version            # Should show CUDA toolkit version
```

**Cause (native Linux):** Driver not installed or not loaded.

**Solution:** Install the NVIDIA driver and verify with `nvidia-smi`.

### Wrong Device Selected

**Symptom:** Operations execute on the wrong GPU, or memory is allocated on an unexpected device.

**Cause:** The `device_only<N>` template parameter does not match the runtime CUDA device context.

**Solution:** Ensure the device ID in your placement policy matches your environment:

```cpp
// Verify device count
int device_count;
cudaGetDeviceCount(&device_count);

// Use the correct device
auto ctx = env.make_world_context(/*device_id=*/0);
dtl::distributed_vector<float, dtl::device_only<0>> vec(1000, ctx);
```

For multi-GPU nodes, map ranks to devices:

```cpp
int local_rank = get_local_rank();  // Rank within the node
int device_id = local_rank % device_count;
auto ctx = env.make_world_context(device_id);
```

### NVCC Not on PATH

**Symptom:** CMake error: "Could NOT find CUDA" or `nvcc: command not found`.

**Cause:** The CUDA toolkit is installed but its `bin/` directory is not in PATH.

**Solution:** Add CUDA to PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```

Or pass the NVCC path explicitly to CMake:

```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

---

## NCCL Communicator Creation Failures

### NCCL Not Found

**Symptom:** CMake warning: "NCCL not found" or `has_nccl() == false` at runtime.

**Solution:**

1. Install NCCL: `sudo apt install libnccl2 libnccl-dev`
2. If NCCL is installed to a non-standard location, set the CMake hint:

```bash
cmake .. -DNCCL_ROOT=/path/to/nccl
```

3. Verify at runtime:

```cpp
dtl::environment env(argc, argv);
std::cout << "NCCL available: " << env.has_nccl() << "\n";
```

### Single-GPU Testing

**Note:** NCCL supports single-GPU testing for correctness verification. Multiple MPI ranks share the same GPU, and NCCL uses shared memory transport between them. This does NOT test multi-GPU performance paths.

```bash
# Correctness test with 2 ranks on 1 GPU
mpirun -np 2 ./test_nccl_collectives
```

---

## Deadlocks in Collective Operations

### Mismatched Collective Calls

**Symptom:** Program hangs indefinitely at a collective operation (reduce, barrier, allgather, etc.).

**Cause:** Not all ranks participate in the collective, or ranks call different collectives.

**Example of the bug:**

```cpp
// DEADLOCK: only rank 0 calls reduce
if (ctx.rank() == 0) {
    auto result = dtl::reduce(vec, 0.0, std::plus<>{});  // Hangs!
}
```

**Solution:** Collective operations must be called by ALL ranks in the communicator:

```cpp
// CORRECT: all ranks participate
auto result = dtl::reduce(vec, 0.0, std::plus<>{});
if (ctx.rank() == 0) {
    std::cout << "Result: " << result << "\n";
}
```

### Barrier Placement Errors

**Symptom:** Data corruption or stale values after a collective.

**Cause:** Missing barrier between write and read phases.

```cpp
auto local = vec.local_view();
for (auto& x : local) x = compute(x);

// Missing barrier! Other ranks may not be done writing yet
auto result = dtl::reduce(vec, 0.0, std::plus<>{});  // May read stale data
```

**Solution:** Insert a barrier when needed between phases:

```cpp
auto local = vec.local_view();
for (auto& x : local) x = compute(x);

vec.barrier();  // Ensure all ranks finish writing

auto result = dtl::reduce(vec, 0.0, std::plus<>{});
```

Note: Many DTL collective algorithms include an implicit barrier. Check the algorithm documentation.

### Conditional Collective Calls

**Symptom:** Deadlock when a collective is inside a conditional that evaluates differently on different ranks.

```cpp
// DEADLOCK: condition may differ across ranks
if (local_error_detected) {
    dtl::reduce(error_counts, 0, std::plus<>{});  // Not all ranks reach this
}
```

**Solution:** Move the collective outside the conditional, or ensure the condition is the same on all ranks:

```cpp
// CORRECT: all ranks participate, then check locally
auto total_errors = dtl::reduce(error_counts, 0, std::plus<>{});
if (total_errors > 0) {
    handle_errors();
}
```

---

## Futures Timeout Diagnostics

### Future Never Completes

**Symptom:** `future.get()` blocks indefinitely or `future.is_ready()` never returns true.

**Possible causes:**

1. **Missing progress:** The underlying async operation needs polling to complete.
2. **Deadlocked collective:** The async operation wraps a collective that deadlocked.
3. **MPI progress issue:** MPI non-blocking operations need `MPI_Test` or `MPI_Wait` calls.

**Solution:**

```cpp
auto future = dtl::async_reduce(vec, 0.0, std::plus<>{});

// Ensure progress is being made
while (!future.is_ready()) {
    dtl::futures::progress_engine::instance().poll();

    // Add a timeout for debugging
    static int iterations = 0;
    if (++iterations > 1000000) {
        std::cerr << "WARNING: future not completing after 1M polls\n";
        break;
    }
}
```

### Progress Engine Not Polled

**Symptom:** Async operations appear to hang, but the progress engine is never polled.

**Cause:** DTL's async operations register callbacks with the progress engine. If nobody polls, the callbacks never execute.

**Solution:** Either:
1. Call `future.get()` (which polls internally)
2. Periodically call `dtl::futures::progress_engine::instance().poll()`
3. Use the blocking variants instead of async if you do not need overlap

---

## Build System Issues

### FindMPI.cmake Empty File

**Symptom:** MPI is not detected despite being installed. CMake silently skips MPI.

**Cause:** An empty `cmake/FindMPI.cmake` file in the DTL source tree shadows CMake's built-in `FindMPI` module.

**Solution:** Delete the empty file:

```bash
rm cmake/FindMPI.cmake  # If it exists and is empty
```

### FindNCCL.cmake Not Found

**Symptom:** CMake cannot find NCCL even though it is installed.

**Solution:** DTL provides its own `FindNCCL.cmake` module. If it is missing, ensure the `cmake/` directory is in the CMake module path:

```bash
cmake .. -DCMAKE_MODULE_PATH=/path/to/dtl/cmake
```

### Compiler Version Too Old

**Symptom:** Build errors related to C++20 features: concepts, `<source_location>`, `requires` clauses, `std::span`.

**Minimum requirements:**
- GCC 11+ (GCC 13+ recommended)
- Clang 15+
- MSVC 19.29+ (Visual Studio 2019 16.10+ with `/std:c++20`)

**Solution:** Upgrade your compiler:

```bash
# Ubuntu/Debian
sudo apt install g++-13

# Use explicitly
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-13
```

### libdtl_runtime.so Not Found at Runtime

**Symptom:** Runtime error: "error while loading shared libraries: libdtl_runtime.so: cannot open shared object file."

**Cause:** `libdtl_runtime.so` is not in the library search path.

**Solution:**

```bash
# Option 1: Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/dtl/build/runtime:$LD_LIBRARY_PATH

# Option 2: Install DTL system-wide
sudo cmake --install build/

# Option 3: Use rpath during build
cmake .. -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
```

---

## Python Binding Import Errors

### Module Not Found

**Symptom:** `import dtl` fails with `ModuleNotFoundError`.

**Cause:** The Python extension module was not built or not installed.

**Solution:**

```bash
# Build Python bindings
cmake .. -DDTL_BUILD_PYTHON=ON
make _dtl

# Install into Python environment
make python_install

# Or set PYTHONPATH
export PYTHONPATH=/path/to/dtl/build/bindings/python:$PYTHONPATH
```

### MPI Library Mismatch

**Symptom:** Python crashes on `import dtl` with a segfault, or `mpi4py` initialization fails.

**Cause:** `mpi4py` was built against a different MPI library than DTL. This commonly happens when conda provides its own MPI.

**Solution:** Rebuild `mpi4py` against the system MPI:

```bash
MPICC=/usr/bin/mpicc CC=/usr/bin/gcc LDSHARED="/usr/bin/gcc -shared" \
  pip install --no-binary mpi4py --force-reinstall mpi4py
```

**Verification:**

```bash
python3 -c "
import mpi4py
print('mpi4py version:', mpi4py.__version__)
from mpi4py import MPI
print('MPI library:', MPI.Get_library_version())
"
```

The MPI library version printed should match your system MPI (e.g., OpenMPI 4.1.6), not conda's.

### NumPy Version Incompatibility

**Symptom:** Import error related to NumPy ABI version mismatch.

**Cause:** DTL's Python bindings were built against a different NumPy version than what is currently installed.

**Solution:** Rebuild the Python bindings after updating NumPy:

```bash
pip install numpy  # Ensure desired version
cd build
cmake .. -DDTL_BUILD_PYTHON=ON
make _dtl
make python_install
```

---

## General Debugging Tips

### Enable Verbose Output

Set environment variables for verbose MPI and CUDA output:

```bash
# MPI verbose output
export OMPI_MCA_mpi_show_mca_params=all

# CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# DTL debug mode (if built with Debug)
export DTL_DEBUG=1
```

### Check Backend Availability

Print a diagnostic report at startup:

```cpp
dtl::environment env(argc, argv);
std::cout << "MPI:   " << env.has_mpi() << " (thread level: " << env.mpi_thread_level_name() << ")\n";
std::cout << "CUDA:  " << env.has_cuda() << "\n";
std::cout << "HIP:   " << env.has_hip() << "\n";
std::cout << "NCCL:  " << env.has_nccl() << "\n";
std::cout << "SHMEM: " << env.has_shmem() << "\n";
```

### Run with Address Sanitizer

For memory issues, build with sanitizers:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
```

### Run with MPI Error Handler

Enable MPI error handlers for better diagnostics:

```bash
# OpenMPI: enable error messages instead of abort
export OMPI_MCA_mpi_abort_print_stack=1
```

---

## See Also

- [Environment Guide](environment.md) -- Backend lifecycle
- [Performance Tuning Guide](performance_tuning.md) -- Optimization strategies
- [Migration Guide](migration_v1_to_v15.md) -- Upgrading from V1.0 to V1.5
- [Contributing Guide](../../CONTRIBUTING.md) -- Development environment and contribution workflow
