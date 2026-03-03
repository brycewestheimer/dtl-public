# External Serialization Adapters

DTL provides optional adapters to integrate with popular C++ serialization libraries:
**Cereal** and **Boost.Serialization**. These adapters bridge the external library's
serialization interface to DTL's `dtl::serializer<T>` trait, enabling seamless use
of pre-existing serialization code with DTL's RPC, remote handlers, and data transfer
infrastructure.

## Overview

### Why Use External Adapters?

Many HPC and distributed computing projects already use established serialization
frameworks. Rather than requiring users to rewrite serialization code for DTL,
these adapters allow direct reuse:

| Library | Typical Use Cases | DTL Adapter Header |
|---------|------------------|-------------------|
| Cereal | Modern C++11 serialization, header-only | `<dtl/serialization/adapters/cereal.hpp>` |
| Boost.Serialization | Legacy HPC codes, versioning support | `<dtl/serialization/adapters/boost_serialization.hpp>` |

### How It Works

1. DTL detects the external library at configure time (CMake)
2. Types must **opt-in** via a marker trait specialization
3. DTL's `serializer<T>` delegates to the external library's archive system
4. Serialization works transparently with DTL RPC, remote handlers, etc.

---

## Enabling Adapters

### CMake Configuration

Adapters are **optional** and disabled by default. Enable them via CMake:

```bash
# Enable Cereal adapter (requires Cereal to be installed)
cmake -DDTL_ENABLE_CEREAL=ON ..

# Enable Boost.Serialization adapter (requires Boost)
cmake -DDTL_ENABLE_BOOST_SERIALIZATION=ON ..

# Enable both
cmake -DDTL_ENABLE_CEREAL=ON -DDTL_ENABLE_BOOST_SERIALIZATION=ON ..
```

### Automatic Detection

If you have Cereal or Boost installed in standard locations, DTL will detect them
automatically when the respective option is enabled. For non-standard locations:

```bash
# Custom Cereal location
cmake -DDTL_ENABLE_CEREAL=ON -DCEREAL_ROOT=/path/to/cereal ..

# Custom Boost location
cmake -DDTL_ENABLE_BOOST_SERIALIZATION=ON -DBOOST_ROOT=/path/to/boost ..
```

### Checking Availability

At compile time, use the detection macros:

```cpp
#include <dtl/serialization/adapters/cereal.hpp>

#if DTL_HAS_CEREAL
    // Cereal adapter is available
#endif

#if DTL_HAS_BOOST_SERIALIZATION
    // Boost.Serialization adapter is available
#endif
```

---

## Using the Cereal Adapter

### Step 1: Define Your Type with Cereal Serialization

Cereal supports several serialization patterns:

```cpp
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

// Pattern 1: Single serialize() method
struct MyData {
    int id;
    std::string name;
    std::vector<double> values;
    
    template <class Archive>
    void serialize(Archive& ar) {
        ar(id, name, values);
    }
};

// Pattern 2: Separate save/load methods
struct VersionedData {
    int version;
    std::string payload;
    
    template <class Archive>
    void save(Archive& ar) const {
        ar(version, payload);
    }
    
    template <class Archive>
    void load(Archive& ar) {
        ar(version, payload);
    }
};
```

### Step 2: Opt-In to the Adapter

Specialize the marker trait to enable DTL integration:

```cpp
#include <dtl/serialization/adapters/cereal.hpp>

template <>
struct dtl::use_cereal_adapter<MyData> : std::true_type {};

template <>
struct dtl::use_cereal_adapter<VersionedData> : std::true_type {};
```

### Step 3: Use with DTL

Now `MyData` works with all DTL serialization infrastructure:

```cpp
#include <dtl/serialization/serializer.hpp>

void example() {
    MyData original{42, "test", {1.0, 2.0, 3.0}};
    
    // Get serialized size
    dtl::size_type size = dtl::serializer<MyData>::serialized_size(original);
    
    // Serialize
    std::vector<std::byte> buffer(size);
    dtl::serializer<MyData>::serialize(original, buffer.data());
    
    // Deserialize
    MyData restored = dtl::serializer<MyData>::deserialize(buffer.data(), size);
}
```

### Utility Functions

The Cereal adapter provides helper functions for common operations:

```cpp
#include <dtl/serialization/adapters/cereal.hpp>

// Serialize directly to a vector
std::vector<std::byte> data = dtl::cereal_serialize_to_vector(my_value);

// Deserialize from a buffer
MyData restored = dtl::cereal_deserialize_from_buffer<MyData>(data.data(), data.size());
```

---

## Using the Boost.Serialization Adapter

### Step 1: Define Your Type with Boost Serialization

```cpp
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

// Intrusive serialization
struct MyBoostData {
    int id;
    std::string name;
    std::vector<double> values;
    
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & id;
        ar & name;
        ar & values;
    }
};

// Non-intrusive serialization (for external types)
struct ExternalData {
    int x, y;
};

namespace boost { namespace serialization {
template <class Archive>
void serialize(Archive& ar, ExternalData& data, const unsigned int version) {
    ar & data.x;
    ar & data.y;
}
}}
```

### Step 2: Opt-In to the Adapter

```cpp
#include <dtl/serialization/adapters/boost_serialization.hpp>

template <>
struct dtl::use_boost_adapter<MyBoostData> : std::true_type {};

template <>
struct dtl::use_boost_adapter<ExternalData> : std::true_type {};
```

### Step 3: Use with DTL

```cpp
void example() {
    MyBoostData original{42, "test", {1.0, 2.0, 3.0}};
    
    // Same interface as Cereal adapter
    dtl::size_type size = dtl::serializer<MyBoostData>::serialized_size(original);
    std::vector<std::byte> buffer(size);
    dtl::serializer<MyBoostData>::serialize(original, buffer.data());
    
    MyBoostData restored = dtl::serializer<MyBoostData>::deserialize(buffer.data(), size);
}
```

### Utility Functions

```cpp
#include <dtl/serialization/adapters/boost_serialization.hpp>

std::vector<std::byte> data = dtl::boost_serialize_to_vector(my_value);
MyBoostData restored = dtl::boost_deserialize_from_buffer<MyBoostData>(data.data(), data.size());
```

---

## Trait Detection

Both adapters provide compile-time traits for checking serialization support:

```cpp
// Check if type has Cereal serialization (regardless of opt-in)
static_assert(dtl::is_cereal_serializable_v<MyData>);

// Check if type has Boost serialization (regardless of opt-in)
static_assert(dtl::is_boost_serializable_v<MyBoostData>);

// Check if adapter is enabled for a type
static_assert(dtl::use_cereal_adapter_v<MyData>);
static_assert(dtl::use_boost_adapter_v<MyBoostData>);
```

---

## Limitations and Considerations

### Performance

External adapters serialize to an intermediate string stream before copying to the
final buffer. This adds overhead compared to DTL's native `serializer<T>` for
trivially copyable types. For performance-critical hot paths with simple data
structures, consider implementing a direct `serializer<T>` specialization.

### Size Calculation

The `serialized_size()` function performs a full trial serialization to determine
the exact buffer size needed. For types with fixed serialization size, this can be
optimized by implementing a custom `serializer<T>` with a `constexpr` size.

### Binary Compatibility

Both adapters use binary archive formats:
- Cereal: `cereal::BinaryInputArchive` / `cereal::BinaryOutputArchive`
- Boost: `boost::archive::binary_iarchive` / `boost::archive::binary_oarchive`

Binary formats are not portable across different:
- Endianness (little-endian vs big-endian machines)
- Compiler/library versions (especially Boost)
- Platform word sizes (32-bit vs 64-bit)

For cross-platform communication, ensure all nodes use compatible environments.

### Versioning

- **Cereal**: Limited versioning support via `cereal::make_nvp()` for named values
- **Boost**: Full versioning via the `version` parameter in `serialize()`

Versioning is passed through transparently by the adapters.

### Thread Safety

Serialization/deserialization operations are thread-safe for different objects.
Concurrent access to the same object requires external synchronization.

---

## Testing

Run serialization adapter tests:

```bash
# Run all serialization tests
ctest -L serialization --output-on-failure

# Run specific adapter tests
./dtl_unit_tests --gtest_filter="CerealAdapter*"
./dtl_unit_tests --gtest_filter="BoostSerializationAdapter*"
```

Tests are conditionally compiled:
- When adapter dependencies are unavailable, tests report `SKIPPED` with an
  explanation of how to enable them
- When dependencies are available, full roundtrip tests validate correctness

---

## Migration from Existing Code

If you have existing types with Cereal or Boost serialization:

1. Include the appropriate adapter header
2. Add the opt-in trait specialization (one line per type)
3. Your types now work with DTL's serialization infrastructure

No changes to your existing serialization code are required.

---

## See Also

- [Native Serializer Trait](serializer.md) - Implement custom `serializer<T>` directly
- [RPC Infrastructure](../rma/rpc.md) - Remote procedure calls using serialization
- [Dynamic Handlers](../remote/dynamic_handlers.md) - Type-erased remote handlers
