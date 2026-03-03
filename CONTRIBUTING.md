# Contributing to DTL

Thank you for your interest in contributing to the Distributed Template Library (DTL). This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Adding New Features](#adding-new-features)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a feature branch from `development`
5. Make your changes
6. Run tests and ensure they pass
7. Submit a pull request

## Development Setup

### Prerequisites

- **Compiler**: GCC 10+, Clang 12+, or MSVC 19.29+ (C++20 concepts support required)
- **Build system**: CMake 3.20+
- **MPI**: OpenMPI or MPICH (for distributed features)
- **Optional**: CUDA Toolkit 11.4+ (for GPU support)

### Building

```bash
# Clone and configure
git clone https://github.com/your-fork/dtl.git
cd dtl
mkdir build && cd build

# Configure with tests enabled
cmake .. -DCMAKE_BUILD_TYPE=Debug -DDTL_BUILD_TESTS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Running MPI Tests

```bash
# 2-rank test
mpirun -np 2 ./tests/dtl_unit_tests

# 4-rank test
mpirun -np 4 ./tests/dtl_unit_tests
```

## Code Style

DTL follows STL conventions for public APIs:

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Types | `snake_case` | `distributed_vector`, `block_partition` |
| Functions | `snake_case` | `local_view()`, `get_rank()` |
| Template parameters | `PascalCase` | `Container`, `Partition`, `Executor` |
| Concepts | `PascalCase` | `Communicator`, `MemorySpace` |
| Macros | `DTL_UPPER_CASE` | `DTL_VERSION_MAJOR` |
| Private members | `snake_case_` (trailing underscore) | `data_`, `rank_` |
| Constants | `snake_case` | `default_grain_size` |

### C++20 Features

- Use concepts for compile-time polymorphism instead of SFINAE
- Prefer `requires` clauses over `enable_if`
- Use `constexpr` and `consteval` where appropriate
- Use `[[nodiscard]]` for functions with important return values

### Header Organization

```cpp
/// @file filename.hpp
/// @brief Brief description
/// @since version

#pragma once

// Standard library headers (alphabetical)
#include <algorithm>
#include <vector>

// DTL headers (alphabetical)
#include <dtl/core/types.hpp>
#include <dtl/policies/partition.hpp>

namespace dtl {

// Implementation

}  // namespace dtl
```

### Formatting

- Run `./scripts/format_code.sh` before committing
- Use clang-format with the project's `.clang-format` configuration
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI configuration |
| `chore` | Other changes |

### Scopes

Common scopes: `core`, `containers`, `views`, `algorithms`, `policies`, `backend`, `mpi`, `cuda`, `rma`, `remote`, `topology`, `async`, `docs`, `tests`

### Examples

```
feat(containers): add distributed_map implementation

Implements hash-partitioned distributed map with:
- Hash-based key distribution
- Local lookup for owned keys
- Remote lookup via RPC for non-local keys

Closes #123
```

```
fix(mpi): correct allreduce buffer handling for non-contiguous data

The previous implementation assumed contiguous memory layout.
This fix properly handles strided data by using MPI derived types.
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout development
   git pull origin development
   git checkout -b feature/my-feature
   ```

2. **Make changes** following the code style guidelines

3. **Write tests** for new functionality

4. **Run the full test suite**
   ```bash
   ctest --output-on-failure
   mpirun -np 2 ./tests/dtl_unit_tests
   mpirun -np 4 ./tests/dtl_unit_tests
   ```

5. **Update documentation** if you changed public APIs

6. **Submit the PR** targeting the `development` branch

7. **Address review feedback** promptly

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] All tests pass (single-process and MPI multi-rank)
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated for any API changes
- [ ] Commit messages follow conventional commits format
- [ ] No compiler warnings introduced

## Testing Requirements

### Unit Tests

- All new public APIs must have unit tests
- Tests should cover normal operation, edge cases, and error conditions
- Use Google Test framework
- Place tests in `tests/unit/` organized by component

### MPI Tests

- Distributed features must be tested with multiple MPI ranks
- Test with at least 2 and 4 ranks
- Handle rank-dependent test logic appropriately

### Test Naming

```cpp
TEST(ComponentName, DescriptiveTestName) {
    // Test implementation
}

// Examples:
TEST(DistributedVector, LocalViewIsSTLCompatible)
TEST(BlockPartition, DistributesEvenly)
TEST(MpiCommAdapter, AllreduceWithSumOperator)
```

## Adding New Features

### Adding a New Algorithm

1. Define the algorithm in `include/dtl/algorithms/`
2. Implement local, distributed, and async variants as appropriate
3. Add tests in `tests/unit/algorithms/`
4. Update `include/dtl/algorithms.hpp` to include the new header
5. Document the algorithm in the user guide or API reference

### Adding a New Backend

1. Create backend directory in `backends/`
2. Implement required concepts (see `include/dtl/backend/concepts/`)
3. Add CMake configuration
4. Add concept compliance tests with `static_assert`
5. Document the backend in the user guide or API reference

### Adding a New Policy

1. Define in `include/dtl/policies/`
2. Ensure compatibility with existing policy compositions
3. Add tests in `tests/unit/policies/`
4. Document the policy in the user guide or API reference

## Documentation

### Code Documentation

- Use Doxygen-style comments for all public APIs
- Document parameters, return values, exceptions, and preconditions
- Include usage examples in complex APIs

```cpp
/// @brief Brief description
/// @tparam T Element type
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param op Binary operation (must be associative)
/// @return The reduced value across all ranks
/// @throws std::invalid_argument if container is empty
/// @pre Container must be in valid state
/// @note This performs MPI communication
///
/// @example
/// @code
/// distributed_vector<int> vec(100);
/// int sum = reduce(vec, 0, std::plus<>{});
/// @endcode
template<typename T, typename BinaryOp>
T reduce(const distributed_container<T>& container, T init, BinaryOp op);
```

## Questions?

- Open a [GitHub Issue](https://github.com/brycewestheimer/dtl-public/issues) for questions
- Tag issues with `question` label
- Check existing issues and documentation first

Thank you for contributing to DTL!
