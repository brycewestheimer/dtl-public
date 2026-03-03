# Globally Stable Distributed Sort

**Since:** DTL v0.1.0-alpha.1
**Header:** `<dtl/algorithms/sorting/stable_sort_global.hpp>`

## Overview

The `stable_sort_global` algorithm provides globally stable sorting for distributed containers. Unlike the standard `stable_sort` which only guarantees stability within each rank, `stable_sort_global` guarantees that equal-key elements preserve their relative ordering across the entire distributed container.

## Stability Semantics

### Standard `stable_sort`
- Guarantees stability **within each rank**
- Equal elements on the same rank maintain their relative order
- Equal elements from different ranks may be reordered relative to each other

### `stable_sort_global`
- Guarantees **global stability** across all ranks
- Equal elements maintain their relative order based on their original global position
- Original order is defined by the container's partition mapping (typically rank-major for block partitions)

## API Reference

### Basic Usage

```cpp
#include <dtl/algorithms/sorting/stable_sort_global.hpp>

// With MPI communicator (multi-rank)
mpi::mpi_comm_adapter comm;
distributed_vector<int> vec(10000, comm);

// Fill with data...

// Globally stable sort
auto result = dtl::stable_sort_global(dtl::par{}, vec, std::less<>{}, comm);
```

### Signatures

```cpp
// Single-rank version (equivalent to std::stable_sort)
template <typename ExecutionPolicy, typename Container, typename Compare = std::less<>>
result<void> stable_sort_global(ExecutionPolicy&& policy,
                                 Container& container,
                                 Compare comp = Compare{});

// Multi-rank version with communicator
template <typename ExecutionPolicy, typename Container, typename Compare, typename Comm>
stable_sort_global_result stable_sort_global(ExecutionPolicy&& policy,
                                              Container& container,
                                              Compare comp,
                                              Comm& comm);

// With sort configuration
template <typename ExecutionPolicy, typename Container, typename Compare, typename Comm>
stable_sort_global_result stable_sort_global_with_config(ExecutionPolicy&& policy,
                                                          Container& container,
                                                          Compare comp,
                                                          distributed_sort_config config,
                                                          Comm& comm);
```

### Return Type

```cpp
struct stable_sort_global_result {
    bool success;              // Whether the operation succeeded
    size_type elements_sent;   // Number of elements sent to other ranks
    size_type elements_received; // Number of elements received from other ranks
};
```

## Algorithm Details

### How It Works

The algorithm achieves global stability by augmenting each element with its original position:

1. **Augmentation**: Each element is wrapped with origin metadata `(rank, local_index)`
2. **Stable Comparator**: A comparator wrapper uses the user's comparator for primary comparison, and origin as a tie-breaker for equal keys
3. **Distributed Sort**: Standard sample sort is performed on the augmented elements
4. **Extraction**: Original values are extracted from the sorted augmented elements

### Origin Order Definition

For block-partitioned containers, the "original order" corresponds to:
- Elements on rank 0 come first (indices 0, 1, 2, ...)
- Then elements on rank 1, rank 2, etc.

This matches the natural global index interpretation.

### Tie-Breaking Logic

When two elements have equal keys (according to the user's comparator), they are ordered by:
1. Origin rank (lower rank first)
2. Origin local index (lower index first within the same rank)

This creates a strict total order that preserves the original global ordering.

## Examples

### Sorting Key-Value Pairs Stably

```cpp
#include <dtl/algorithms/sorting/stable_sort_global.hpp>

struct Record {
    int key;
    int original_rank;
    int original_index;
};

// Custom comparator that only compares keys
auto by_key = [](const Record& a, const Record& b) {
    return a.key < b.key;
};

mpi::mpi_comm_adapter comm;
distributed_vector<Record> records(1000, comm);

// Fill records with tracking data
auto local = records.local_view();
for (size_t i = 0; i < local.size(); ++i) {
    local[i] = {compute_key(i), comm.rank(), static_cast<int>(i)};
}

// Sort globally - records with same key maintain original order
dtl::stable_sort_global(dtl::par{}, records, by_key, comm);

// Verify: among records with same key, original_rank and original_index
// are in ascending order
```

### Verifying Stability After Sort

```cpp
// After stable_sort_global, you can verify stability by checking
// that elements with equal keys are ordered by their origin

bool verify_stability(const distributed_vector<Record>& sorted,
                      mpi::mpi_comm_adapter& comm) {
    auto local = sorted.local_view();
    
    for (size_t i = 1; i < local.size(); ++i) {
        if (local[i-1].key == local[i].key) {
            // Equal keys: check origin order
            if (local[i-1].original_rank > local[i].original_rank) {
                return false;
            }
            if (local[i-1].original_rank == local[i].original_rank &&
                local[i-1].original_index > local[i].original_index) {
                return false;
            }
        }
    }
    
    return true;  // Local stability verified
    // (Full verification requires boundary checks between ranks)
}
```

## Comparison with Other Sort Functions

| Function | Stability | Communication | Use Case |
|----------|-----------|---------------|----------|
| `sort` | Unstable | Yes | General distributed sorting |
| `stable_sort` | Within-rank | Yes | Stability needed only locally |
| `stable_sort_global` | Global | Yes | Full stability across all ranks |
| `local_sort` | N/A | No | Sort local partition only |
| `local_stable_sort` | Within-rank | No | Local stable sort only |

## Performance Considerations

### Memory Overhead

`stable_sort_global` wraps each element with origin metadata:
- Additional `sizeof(rank_t) + sizeof(size_type)` per element
- Typically 12-16 bytes per element overhead

### Communication Overhead

Same as standard sample sort:
- O(p × samples) for sample gathering
- O(n/p) for data redistribution
- Augmented elements are slightly larger, increasing bandwidth

### When to Use

Use `stable_sort_global` when:
- You need deterministic ordering for equal keys across ranks
- Reproducibility is important (same input always produces same output)
- You're sorting records where secondary ordering matters

Use standard `stable_sort` when:
- Stability within each rank is sufficient
- Memory efficiency is critical
- You can tolerate non-deterministic ordering of equal keys across ranks

## Thread Safety

- Collective operation: all ranks must participate
- Not thread-safe within a single rank (caller must ensure exclusive access)
- Uses parallel execution internally when `par{}` policy is specified

## Error Handling

The function returns a result indicating success or failure:

```cpp
auto result = dtl::stable_sort_global(par{}, vec, std::less<>{}, comm);
if (!result.success) {
    // Handle error
}
```

## See Also

- [sort.hpp](sort.hpp) - Standard distributed sort
- [sample_sort_detail.hpp](sample_sort_detail.hpp) - Sample sort implementation details
- [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) - Known issues and workarounds
