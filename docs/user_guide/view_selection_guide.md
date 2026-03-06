# View Selection Guide

DTL provides several view types for accessing distributed container data.
Choosing the right view is critical for both correctness and performance.

## Decision Tree

```
Do you need to access elements on other ranks?
├── NO → local_view
│   └── Need to process a subset?
│       ├── Contiguous range → subview
│       ├── Regular stride pattern → strided_view
│       ├── Fixed-size batches → chunk_view
│       └── N-dimensional tiles → tile_view
│
└── YES → Which access pattern?
    ├── Global index space → global_view
    │   └── Requires explicit .get()/.put() on remote_ref<T>
    ├── Per-rank segments → segmented_view
    │   └── Each segment is a local_view (STL-safe)
    └── Sliding window → window_view
        └── Overlapping regions across partitions
```

## View Comparison

| View | Communicates? | STL-safe? | Element type | Best for |
|------|:---:|:---:|---|---|
| `local_view` | Never | Yes | `T&` | Local computation, STL algorithms |
| `global_view` | May | No | `remote_ref<T>` | Global index access, sparse remote reads |
| `segmented_view` | May | No | Segment ranges | Map-reduce over all partitions |
| `subview` | Inherited | Inherited | Inherited | Working on a slice of data |
| `strided_view` | Inherited | Inherited | Inherited | Every Nth element, diagonal access |
| `chunk_view` | Inherited | Inherited | Inherited | Batch processing, tiled algorithms |
| `tile_view` | Inherited | Inherited | Inherited | 2D/3D blocking for cache efficiency |
| `window_view` | Inherited | Inherited | Inherited | Stencil computations, halo exchange |
| `chunk_by_view` | Inherited | Inherited | Inherited | Group by predicate |
| `composed_view` | Inherited | Inherited | Inherited | Combine multiple view transforms |

## Common Patterns

### Pattern 1: Local computation (most common)
```cpp
auto local = vec.local_view();
for (auto& elem : local) {
    elem = compute(elem);  // No communication
}
// Or use STL algorithms:
std::transform(local.begin(), local.end(), local.begin(), compute);
```

### Pattern 2: Reduction across all ranks
```cpp
// Use DTL algorithms directly — they handle communication:
auto sum = dtl::reduce(seq{}, vec, 0.0, std::plus<>{});
```

### Pattern 3: Batch processing with chunk_view
```cpp
auto local = vec.local_view();
for (auto chunk : dtl::chunk_view(local, batch_size)) {
    process_batch(chunk.begin(), chunk.end());
}
```

### Pattern 4: Stencil computation with window_view
```cpp
auto local = vec.local_view();
for (auto window : dtl::window_view(local, stencil_width)) {
    auto center = window[stencil_width / 2];
    result = apply_stencil(window);
}
```

### Pattern 5: Sparse global access
```cpp
auto global = vec.global_view();
for (auto idx : sparse_indices) {
    auto ref = global[idx];
    if (ref.is_local()) {
        process(ref.get().value());  // Free
    } else {
        auto future = ref.async_get();  // Async remote fetch
        // ... do other work ...
        process(future.get().value());
    }
}
```

### Pattern 6: View composition
```cpp
auto local = vec.local_view();
auto strided = dtl::strided_view(local, 2);     // Every other element
auto chunked = dtl::chunk_view(strided, 100);    // In batches of 100
// Or compose:
auto composed = dtl::compose(
    dtl::strided(2),
    dtl::chunk(100)
)(local);
```

## Performance Guidelines

1. **Prefer `local_view`** for all computation. It provides contiguous memory
   access with zero communication overhead.

2. **Avoid `global_view` in loops.** Each `.get()` on a non-local element
   triggers a network round-trip. Batch remote accesses using `async_get()`
   or redesign to use `segmented_view`.

3. **Use `subview` to narrow scope.** Processing a subview instead of the
   full local_view can improve cache locality.

4. **Match chunk/tile size to cache.** For `chunk_view` and `tile_view`,
   choose sizes that fit in L1/L2 cache for best performance.

5. **Compose views lazily.** View composition creates no intermediate copies.
   Chain views freely without worrying about allocation.
