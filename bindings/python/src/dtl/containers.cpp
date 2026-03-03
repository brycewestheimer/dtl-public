// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file containers.cpp
 * @brief DTL Python bindings - Container module (DistributedVector, DistributedArray, DistributedTensor)
 * @since 0.1.0
 * @note Phase 05 (P05) adds policy parameters to container constructors
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"
#include <dtl/bindings/c/dtl_policies.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Convert DTL status to Python exception
 */
void check_status(dtl_status status) {
    ::dtl::python::check_status_or_throw(status);
}

/**
 * @brief Extract native context from Python Context object
 */
dtl_context_t get_native_context(py::object ctx_obj) {
    // Try to get native context through the Python wrapper's native() method
    if (py::hasattr(ctx_obj, "native")) {
        // Direct _dtl.Context object
        py::object native_method = ctx_obj.attr("native");
        return reinterpret_cast<dtl_context_t>(
            native_method().cast<std::uintptr_t>());
    } else if (py::hasattr(ctx_obj, "_native")) {
        // Python wrapper class
        py::object inner_ctx = ctx_obj.attr("_native");
        if (py::hasattr(inner_ctx, "native")) {
            py::object native_method = inner_ctx.attr("native");
            return reinterpret_cast<dtl_context_t>(
                native_method().cast<std::uintptr_t>());
        }
    }
    throw std::runtime_error("Invalid context object - could not extract native handle");
}

/**
 * @brief Check if a placement policy allows direct host access
 *
 * For device-only placements, local_view() is unsafe because the pointer
 * points to GPU memory.
 */
bool is_host_accessible_placement(dtl_placement_policy placement) {
    switch (placement) {
        case DTL_PLACEMENT_HOST:
        case DTL_PLACEMENT_UNIFIED:
        case DTL_PLACEMENT_DEVICE_PREFERRED:
            return true;
        case DTL_PLACEMENT_DEVICE:
            return false;
        default:
            return true;  // Default to host for safety
    }
}

/**
 * @brief Build container options from Python arguments
 */
dtl_container_options build_container_options(
    int partition,
    int placement,
    int execution,
    int device_id,
    dtl_size_t block_size)
{
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    opts.partition = static_cast<dtl_partition_policy>(partition);
    opts.placement = static_cast<dtl_placement_policy>(placement);
    opts.execution = static_cast<dtl_execution_policy>(execution);
    opts.device_id = device_id;
    opts.block_size = block_size;

    return opts;
}

/**
 * @brief Classify typed Python container wrappers for span construction
 */
enum class span_source_kind {
    vector,
    array,
    tensor,
    unsupported
};

/**
 * @brief Return true when @p value starts with @p prefix
 */
bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

/**
 * @brief Detect source kind from pybind class name
 */
span_source_kind detect_span_source_kind(const py::object& source) {
    const std::string type_name = py::str(py::type::of(source).attr("__name__"));
    if (starts_with(type_name, "DistributedVector_")) {
        return span_source_kind::vector;
    }
    if (starts_with(type_name, "DistributedArray_")) {
        return span_source_kind::array;
    }
    if (starts_with(type_name, "DistributedTensor_")) {
        return span_source_kind::tensor;
    }
    return span_source_kind::unsupported;
}

/**
 * @brief Build a native dtl_span_t from a typed Python container wrapper
 */
dtl_span_t create_native_span_from_source(const py::object& source) {
    if (!py::hasattr(source, "native")) {
        throw std::runtime_error(
            "DistributedSpan source object must provide native() handle access");
    }

    const std::uintptr_t raw = source.attr("native")().cast<std::uintptr_t>();
    if (raw == 0) {
        throw std::runtime_error("DistributedSpan source has null native handle");
    }

    dtl_span_t span = nullptr;
    dtl_status status = DTL_ERROR_INVALID_ARGUMENT;

    switch (detect_span_source_kind(source)) {
        case span_source_kind::vector:
            status = dtl_span_from_vector(reinterpret_cast<dtl_vector_t>(raw), &span);
            break;
        case span_source_kind::array:
            status = dtl_span_from_array(reinterpret_cast<dtl_array_t>(raw), &span);
            break;
        case span_source_kind::tensor:
            status = dtl_span_from_tensor(reinterpret_cast<dtl_tensor_t>(raw), &span);
            break;
        case span_source_kind::unsupported:
            throw std::runtime_error(
                "DistributedSpan source must be a typed DistributedVector_*, "
                "DistributedArray_*, or DistributedTensor_* instance");
    }

    if (status != DTL_SUCCESS) {
        if (span) {
            dtl_span_destroy(span);
            span = nullptr;
        }
        check_status(status);
    }
    if (!span || dtl_span_is_valid(span) == 0) {
        if (span) {
            dtl_span_destroy(span);
        }
        throw std::runtime_error("Failed to construct a valid native span handle");
    }

    return span;
}

}  // namespace

// ============================================================================
// DistributedVector Template Wrapper
// ============================================================================

/**
 * @brief Python wrapper for dtl_vector_t with specific element type
 *
 * @note Phase 05 (P05): Added policy support via create_with_options
 */
template <typename T, dtl_dtype DType>
class PyDistributedVector {
public:
    /**
     * @brief Create vector with given size (default policies)
     */
    PyDistributedVector(dtl_context_t ctx, dtl_size_t size)
        : placement_(DTL_PLACEMENT_HOST) {
        dtl_status status = dtl_vector_create(ctx, DType, size, &vec_);
        check_status(status);
    }

    /**
     * @brief Create vector with given size and fill value (default policies)
     */
    PyDistributedVector(dtl_context_t ctx, dtl_size_t size, T fill_value)
        : PyDistributedVector(ctx, size) {
        fill_local_data(fill_value);
    }

    /**
     * @brief Create vector with policy options (Phase 05)
     */
    PyDistributedVector(dtl_context_t ctx, dtl_size_t size,
                        int partition, int placement, int execution,
                        int device_id, dtl_size_t block_size)
        : placement_(static_cast<dtl_placement_policy>(placement)) {
        dtl_container_options opts = build_container_options(
            partition, placement, execution, device_id, block_size);
        dtl_status status = dtl_vector_create_with_options(
            ctx, DType, size, &opts, &vec_);
        check_status(status);
    }

    /**
     * @brief Create vector with policy options and fill value (Phase 05)
     */
    PyDistributedVector(dtl_context_t ctx, dtl_size_t size, T fill_value,
                        int partition, int placement, int execution,
                        int device_id, dtl_size_t block_size)
        : PyDistributedVector(ctx, size, partition, placement, execution,
                              device_id, block_size) {
        fill_local_data(fill_value);
    }

    ~PyDistributedVector() {
        if (vec_) {
            dtl_vector_destroy(vec_);
        }
    }

    // Disable copy
    PyDistributedVector(const PyDistributedVector&) = delete;
    PyDistributedVector& operator=(const PyDistributedVector&) = delete;

    // Enable move
    PyDistributedVector(PyDistributedVector&& other) noexcept
        : vec_(other.vec_), placement_(other.placement_) {
        other.vec_ = nullptr;
    }

    PyDistributedVector& operator=(PyDistributedVector&& other) noexcept {
        if (this != &other) {
            if (vec_) {
                dtl_vector_destroy(vec_);
            }
            vec_ = other.vec_;
            placement_ = other.placement_;
            other.vec_ = nullptr;
        }
        return *this;
    }

    // Properties
    dtl_size_t global_size() const { return dtl_vector_global_size(vec_); }
    dtl_size_t local_size() const { return dtl_vector_local_size(vec_); }
    dtl_index_t local_offset() const { return dtl_vector_local_offset(vec_); }

    /**
     * @brief Get the placement policy (Phase 05)
     */
    int get_placement() const { return static_cast<int>(placement_); }

    /**
     * @brief Check if local_view is safe (host-accessible placement)
     */
    bool is_host_accessible() const {
        return is_host_accessible_placement(placement_);
    }

    /**
     * @brief Get zero-copy NumPy view of local data
     *
     * @throws std::runtime_error if placement is device-only (Phase 05)
     *
     * For device-only placement, use to_numpy() instead.
     */
    py::array_t<T> local_view() {
        // Phase 05: Guard against device-only placement
        if (!is_host_accessible()) {
            throw std::runtime_error(
                "Cannot create NumPy view of device-only memory. "
                "Use to_numpy() to copy data to host.");
        }

        void* data = dtl_vector_local_data_mut(vec_);
        dtl_size_t size = dtl_vector_local_size(vec_);

        // Keep the Python wrapper alive while NumPy view references local storage.
        py::object owner = py::cast(this, py::return_value_policy::reference);

        return py::array_t<T>(
            {static_cast<py::ssize_t>(size)},
            {sizeof(T)},
            static_cast<T*>(data),
            owner
        );
    }

    /**
     * @brief Copy local data to NumPy array (Phase 05)
     *
     * Works for all placements including device-only.
     * Returns a copy of the data (not a view).
     */
    py::array_t<T> to_numpy() {
        dtl_size_t size = dtl_vector_local_size(vec_);

        // Create new NumPy array to hold the copy
        py::array_t<T> result({static_cast<py::ssize_t>(size)});
        T* dest = result.mutable_data();

        // Use C API copy function (handles device-to-host transparently)
        dtl_status status = dtl_vector_copy_to_host(vec_, dest, size);
        check_status(status);

        return result;
    }

    /**
     * @brief Copy data from NumPy array to local vector data (Phase 05)
     *
     * Works for all placements including device-only.
     *
     * @param arr Source NumPy array (must be contiguous and same dtype)
     */
    void from_numpy(py::array_t<T> arr) {
        dtl_size_t size = dtl_vector_local_size(vec_);

        // Check size matches
        if (static_cast<dtl_size_t>(arr.size()) != size) {
            throw std::runtime_error(
                "NumPy array size (" + std::to_string(arr.size()) +
                ") does not match local vector size (" + std::to_string(size) + ")");
        }

        // Ensure array is contiguous
        py::array_t<T, py::array::c_style | py::array::forcecast> contiguous_arr(arr);
        const T* src = contiguous_arr.data();

        // Use C API copy function (handles host-to-device transparently)
        dtl_status status = dtl_vector_copy_from_host(vec_, src, size);
        check_status(status);
    }

    /**
     * @brief Fill vector with value
     */
    void fill(T value) {
        dtl_status status = dtl_vector_fill_local(vec_, &value);
        check_status(status);
    }

    /**
     * @brief Check if the vector has uncommitted modifications
     */
    bool is_dirty() const {
        return dtl_vector_is_dirty(vec_) != 0;
    }

    /**
     * @brief Synchronize the vector (barrier + mark clean)
     */
    void sync() {
        dtl_status status = dtl_vector_sync(vec_);
        check_status(status);
    }

    /**
     * @brief Redistribute the vector with a new partition (collective)
     *
     * @param partition_type 0 = block, 1 = cyclic
     */
    void redistribute(int partition_type) {
        dtl_partition_type pt = static_cast<dtl_partition_type>(partition_type);
        dtl_status status = dtl_vector_redistribute(vec_, pt);
        check_status(status);
    }

    // Internal access
    dtl_vector_t native() const { return vec_; }

private:
    /**
     * @brief Helper to fill local data (used by constructors)
     */
    void fill_local_data(T fill_value) {
        // For device-only, use C API fill which handles device memory
        if (!is_host_accessible()) {
            dtl_status status = dtl_vector_fill_local(vec_, &fill_value);
            check_status(status);
        } else {
            void* data = dtl_vector_local_data_mut(vec_);
            dtl_size_t size = dtl_vector_local_size(vec_);
            T* typed_data = static_cast<T*>(data);
            for (dtl_size_t i = 0; i < size; ++i) {
                typed_data[i] = fill_value;
            }
        }
    }

    dtl_vector_t vec_ = nullptr;
    dtl_placement_policy placement_ = DTL_PLACEMENT_HOST;
};

// ============================================================================
// DistributedTensor Template Wrapper
// ============================================================================

/**
 * @brief Python wrapper for dtl_tensor_t with specific element type
 */
template <typename T, dtl_dtype DType>
class PyDistributedTensor {
public:
    /**
     * @brief Create tensor with given shape
     */
    PyDistributedTensor(dtl_context_t ctx, py::tuple shape_tuple) {
        // Convert Python tuple to dtl_shape
        dtl_shape shape;
        shape.ndim = static_cast<int>(shape_tuple.size());
        if (shape.ndim > DTL_MAX_TENSOR_RANK) {
            throw std::runtime_error("Tensor rank exceeds maximum (" +
                                     std::to_string(DTL_MAX_TENSOR_RANK) + ")");
        }

        for (int i = 0; i < shape.ndim; ++i) {
            shape.dims[i] = shape_tuple[i].cast<dtl_size_t>();
        }

        dtl_status status = dtl_tensor_create(ctx, DType, shape, &tensor_);
        check_status(status);
    }

    /**
     * @brief Create tensor with given shape and fill value
     */
    PyDistributedTensor(dtl_context_t ctx, py::tuple shape_tuple, T fill_value)
        : PyDistributedTensor(ctx, shape_tuple) {
        // Fill with value
        void* data = dtl_tensor_local_data_mut(tensor_);
        dtl_size_t local_size = dtl_tensor_local_size(tensor_);
        T* typed_data = static_cast<T*>(data);
        for (dtl_size_t i = 0; i < local_size; ++i) {
            typed_data[i] = fill_value;
        }
    }

    ~PyDistributedTensor() {
        if (tensor_) {
            dtl_tensor_destroy(tensor_);
        }
    }

    // Disable copy
    PyDistributedTensor(const PyDistributedTensor&) = delete;
    PyDistributedTensor& operator=(const PyDistributedTensor&) = delete;

    // Enable move
    PyDistributedTensor(PyDistributedTensor&& other) noexcept : tensor_(other.tensor_) {
        other.tensor_ = nullptr;
    }

    PyDistributedTensor& operator=(PyDistributedTensor&& other) noexcept {
        if (this != &other) {
            if (tensor_) {
                dtl_tensor_destroy(tensor_);
            }
            tensor_ = other.tensor_;
            other.tensor_ = nullptr;
        }
        return *this;
    }

    // Properties
    int ndim() const { return dtl_tensor_ndim(tensor_); }

    py::tuple shape() const {
        dtl_shape sh = dtl_tensor_shape(tensor_);
        py::tuple result(sh.ndim);
        for (int i = 0; i < sh.ndim; ++i) {
            result[i] = py::int_(sh.dims[i]);
        }
        return result;
    }

    py::tuple local_shape() const {
        dtl_shape sh = dtl_tensor_local_shape(tensor_);
        py::tuple result(sh.ndim);
        for (int i = 0; i < sh.ndim; ++i) {
            result[i] = py::int_(sh.dims[i]);
        }
        return result;
    }

    dtl_size_t global_size() const { return dtl_tensor_global_size(tensor_); }
    dtl_size_t local_size() const { return dtl_tensor_local_size(tensor_); }

    /**
     * @brief Get zero-copy NumPy view of local data
     */
    py::array_t<T> local_view() {
        void* data = dtl_tensor_local_data_mut(tensor_);

        // Get local shape
        dtl_shape local_sh = dtl_tensor_local_shape(tensor_);

        // Build shape and strides for NumPy
        std::vector<py::ssize_t> np_shape(local_sh.ndim);
        std::vector<py::ssize_t> np_strides(local_sh.ndim);

        // Calculate strides (row-major / C-order)
        py::ssize_t stride = sizeof(T);
        for (int i = local_sh.ndim - 1; i >= 0; --i) {
            np_shape[i] = static_cast<py::ssize_t>(local_sh.dims[i]);
            np_strides[i] = stride;
            stride *= np_shape[i];
        }

        // Keep the Python wrapper alive while NumPy view references local storage.
        py::object owner = py::cast(this, py::return_value_policy::reference);

        return py::array_t<T>(
            np_shape,
            np_strides,
            static_cast<T*>(data),
            owner
        );
    }

    /**
     * @brief Fill tensor with value
     */
    void fill(T value) {
        dtl_status status = dtl_tensor_fill_local(tensor_, &value);
        check_status(status);
        dirty_ = true;
    }

    /**
     * @brief Check if the tensor has uncommitted modifications
     *
     * Tensors use local dirty tracking since the C ABI does not expose
     * dtl_tensor_is_dirty. The dirty flag is set on fill() and cleared
     * on sync().
     */
    bool is_dirty() const {
        return dirty_;
    }

    /**
     * @brief Synchronize the tensor (barrier + mark clean)
     *
     * Performs a barrier on the tensor's communicator and clears the
     * dirty flag.
     */
    void sync() {
        dtl_status status = dtl_tensor_barrier(tensor_);
        check_status(status);
        dirty_ = false;
    }

    /**
     * @brief Redistribute the tensor (not yet supported via C ABI)
     *
     * Tensor redistribution requires reshaping along the distributed
     * dimension, which is not currently exposed in the C ABI.
     * Logs a warning and raises RuntimeError.
     */
    void redistribute(int /*partition_type*/) {
        throw std::runtime_error(
            "Tensor redistribution is not yet supported via the C ABI. "
            "Redistribute requires reallocation along the distributed dimension.");
    }

    /**
     * @brief Copy local tensor data to a new NumPy array (Phase 08 parity)
     *
     * Returns a copy (not a view) with the correct local shape.
     * Works for all placements since it reads from local_data.
     */
    py::array_t<T> to_numpy() {
        dtl_shape local_sh = dtl_tensor_local_shape(tensor_);
        const void* data = dtl_tensor_local_data(tensor_);
        dtl_size_t local_size = dtl_tensor_local_size(tensor_);

        // Build shape for NumPy array
        std::vector<py::ssize_t> np_shape(local_sh.ndim);
        for (int i = 0; i < local_sh.ndim; ++i) {
            np_shape[i] = static_cast<py::ssize_t>(local_sh.dims[i]);
        }

        // Create new NumPy array and copy data
        py::array_t<T> result(np_shape);
        T* dest = result.mutable_data();
        std::memcpy(dest, data, local_size * sizeof(T));

        return result;
    }

    /**
     * @brief Copy data from NumPy array into local tensor data (Phase 08 parity)
     *
     * The array must have the same total number of elements as the local
     * tensor partition. Shape is not checked beyond total element count.
     *
     * @param arr Source NumPy array (must be contiguous and same dtype)
     */
    void from_numpy(py::array_t<T> arr) {
        dtl_size_t local_size = dtl_tensor_local_size(tensor_);

        if (static_cast<dtl_size_t>(arr.size()) != local_size) {
            throw std::runtime_error(
                "NumPy array size (" + std::to_string(arr.size()) +
                ") does not match local tensor size (" +
                std::to_string(local_size) + ")");
        }

        py::array_t<T, py::array::c_style | py::array::forcecast> contiguous(arr);
        const T* src = contiguous.data();
        void* dest = dtl_tensor_local_data_mut(tensor_);
        std::memcpy(dest, src, local_size * sizeof(T));
        dirty_ = true;
    }

    // Internal access
    dtl_tensor_t native() const { return tensor_; }

private:
    dtl_tensor_t tensor_ = nullptr;
    bool dirty_ = false;
};

// ============================================================================
// Module Binding
// ============================================================================

/**
 * @brief Bind a DistributedVector type for a specific dtype
 *
 * @note Phase 05 (P05): Added constructors with policy parameters
 */
template <typename T, dtl_dtype DType>
void bind_vector_type(py::module_& m, const char* name) {
    using Vec = PyDistributedVector<T, DType>;

    py::class_<Vec>(m, name)
        // Original constructors (backwards compatibility)
        .def(py::init([](py::object ctx_obj, dtl_size_t size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Vec(ctx, size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             "Create a distributed vector with the given size")
        .def(py::init([](py::object ctx_obj, dtl_size_t size, T fill) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Vec(ctx, size, fill);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("fill"),
             "Create a distributed vector with fill value")
        // Phase 05: Constructor with policy options
        .def(py::init([](py::object ctx_obj, dtl_size_t size,
                         int partition, int placement, int execution,
                         int device_id, dtl_size_t block_size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Vec(ctx, size, partition, placement, execution,
                           device_id, block_size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("partition"),
             py::arg("placement"),
             py::arg("execution"),
             py::arg("device_id"),
             py::arg("block_size"),
             "Create a distributed vector with policy options")
        // Phase 05: Constructor with policy options and fill
        .def(py::init([](py::object ctx_obj, dtl_size_t size, T fill,
                         int partition, int placement, int execution,
                         int device_id, dtl_size_t block_size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Vec(ctx, size, fill, partition, placement, execution,
                           device_id, block_size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("fill"),
             py::arg("partition"),
             py::arg("placement"),
             py::arg("execution"),
             py::arg("device_id"),
             py::arg("block_size"),
             "Create a distributed vector with policy options and fill value")
        .def_property_readonly("global_size", &Vec::global_size,
                               "Total number of elements across all ranks")
        .def_property_readonly("local_size", &Vec::local_size,
                               "Number of elements on this rank")
        .def_property_readonly("local_offset", &Vec::local_offset,
                               "Global index of first local element")
        .def_property_readonly("placement", &Vec::get_placement,
                               "Placement policy (Phase 05)")
        .def_property_readonly("is_host_accessible", &Vec::is_host_accessible,
                               "True if local_view() is safe")
        .def("local_view", &Vec::local_view,
             py::return_value_policy::reference_internal,
             R"doc(
Get a NumPy array view of local data.

The returned array shares memory with the vector - modifications
to the array will modify the vector data.

Raises:
    RuntimeError: If placement is device-only. Use to_numpy() instead.

Returns:
    numpy.ndarray: Mutable view of local data
)doc")
        .def("to_numpy", &Vec::to_numpy,
             R"doc(
Copy local data to a new NumPy array.

Works for all placements including device-only. Returns a copy,
not a view.

Returns:
    numpy.ndarray: Copy of local data
)doc")
        .def("from_numpy", &Vec::from_numpy,
             py::arg("arr"),
             R"doc(
Copy data from a NumPy array to local vector data.

Works for all placements including device-only.

Args:
    arr: Source NumPy array (must match local_size)
)doc")
        .def("fill", &Vec::fill,
             py::arg("value"),
             "Fill all local elements with the given value")
        .def("is_dirty", &Vec::is_dirty,
             "Check if the vector has uncommitted modifications")
        .def("sync", &Vec::sync,
             py::call_guard<py::gil_scoped_release>(),
             "Synchronize the vector (barrier + mark clean)")
        .def("redistribute", &Vec::redistribute,
             py::arg("partition_type"),
             py::call_guard<py::gil_scoped_release>(),
             R"doc(
Redistribute the vector with a new partition (collective).

Args:
    partition_type: 0 = block, 1 = cyclic

Warning:
    This is a collective operation - all ranks must call.
    Invalidates any NumPy views from local_view().
)doc")
        .def("native", [](const Vec& v) {
            return reinterpret_cast<std::uintptr_t>(v.native());
        }, "Get native handle (for internal use by algorithm functions)")
        .def("__len__", &Vec::global_size)
        .def("__repr__", [name](const Vec& v) {
            return std::string("<") + name + " global_size=" +
                   std::to_string(v.global_size()) +
                   " local_size=" + std::to_string(v.local_size()) +
                   " placement=" + std::to_string(v.get_placement()) + ">";
        });
}

/**
 * @brief Bind a DistributedTensor type for a specific dtype
 */
template <typename T, dtl_dtype DType>
void bind_tensor_type(py::module_& m, const char* name) {
    using Tensor = PyDistributedTensor<T, DType>;

    py::class_<Tensor>(m, name)
        .def(py::init([](py::object ctx_obj, py::tuple shape) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Tensor(ctx, shape);
        }),
             py::arg("ctx"),
             py::arg("shape"),
             "Create a distributed tensor with the given shape")
        .def(py::init([](py::object ctx_obj, py::tuple shape, T fill) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Tensor(ctx, shape, fill);
        }),
             py::arg("ctx"),
             py::arg("shape"),
             py::arg("fill"),
             "Create a distributed tensor with fill value")
        .def_property_readonly("ndim", &Tensor::ndim,
                               "Number of dimensions")
        .def_property_readonly("shape", &Tensor::shape,
                               "Global shape tuple")
        .def_property_readonly("local_shape", &Tensor::local_shape,
                               "Local shape tuple on this rank")
        .def_property_readonly("global_size", &Tensor::global_size,
                               "Total number of elements across all ranks")
        .def_property_readonly("local_size", &Tensor::local_size,
                               "Number of elements on this rank")
        .def("local_view", &Tensor::local_view,
             py::return_value_policy::reference_internal,
             R"doc(
Get a NumPy array view of local data.

The returned array shares memory with the tensor - modifications
to the array will modify the tensor data.

Returns:
    numpy.ndarray: Mutable view of local data
)doc")
        .def("to_numpy", &Tensor::to_numpy,
R"doc(
Copy local tensor data to a new NumPy array.

Returns a copy of the local data (not a view) with the local shape.
Use this when you need an independent copy; use local_view() for
zero-copy access.

Returns:
    numpy.ndarray: Copy of local data with shape matching local_shape
)doc")
        .def("from_numpy", &Tensor::from_numpy,
             py::arg("arr"),
R"doc(
Copy data from a NumPy array into the local tensor partition.

The array must have the same total number of elements as the
local tensor size.

Args:
    arr: Source NumPy array (same dtype, same total element count)
)doc")
        .def("fill", &Tensor::fill,
             py::arg("value"),
             "Fill all local elements with the given value")
        .def("is_dirty", &Tensor::is_dirty,
             "Check if the tensor has uncommitted modifications")
        .def("sync", &Tensor::sync,
             py::call_guard<py::gil_scoped_release>(),
             "Synchronize the tensor (barrier + mark clean)")
        .def("redistribute", &Tensor::redistribute,
             py::arg("partition_type"),
             "Redistribute the tensor (not yet supported - raises RuntimeError)")
        .def("native", [](const Tensor& t) {
            return reinterpret_cast<std::uintptr_t>(t.native());
        }, "Get native handle (for internal use by algorithm functions)")
        .def("__repr__", [name](const Tensor& t) {
            std::string shape_str = "(";
            py::tuple shape = t.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(py::cast<dtl_size_t>(shape[i]));
            }
            shape_str += ")";
            return std::string("<") + name + " shape=" + shape_str + ">";
        });
}

// ============================================================================
// DistributedArray Template Wrapper
// ============================================================================

/**
 * @brief Python wrapper for dtl_array_t with specific element type
 *
 * Unlike DistributedVector, DistributedArray has a fixed size that
 * cannot be changed after creation.
 *
 * @note Phase 05 (P05): Added policy support via create_with_options
 */
template <typename T, dtl_dtype DType>
class PyDistributedArray {
public:
    /**
     * @brief Create array with given size (fixed, cannot be changed)
     */
    PyDistributedArray(dtl_context_t ctx, dtl_size_t size)
        : placement_(DTL_PLACEMENT_HOST) {
        dtl_status status = dtl_array_create(ctx, DType, size, &arr_);
        check_status(status);
    }

    /**
     * @brief Create array with given size and fill value
     */
    PyDistributedArray(dtl_context_t ctx, dtl_size_t size, T fill_value)
        : PyDistributedArray(ctx, size) {
        fill_local_data(fill_value);
    }

    /**
     * @brief Create array with policy options (Phase 05)
     */
    PyDistributedArray(dtl_context_t ctx, dtl_size_t size,
                       int partition, int placement, int execution,
                       int device_id, dtl_size_t block_size)
        : placement_(static_cast<dtl_placement_policy>(placement)) {
        dtl_container_options opts = build_container_options(
            partition, placement, execution, device_id, block_size);
        dtl_status status = dtl_array_create_with_options(
            ctx, DType, size, &opts, &arr_);
        check_status(status);
    }

    /**
     * @brief Create array with policy options and fill value (Phase 05)
     */
    PyDistributedArray(dtl_context_t ctx, dtl_size_t size, T fill_value,
                       int partition, int placement, int execution,
                       int device_id, dtl_size_t block_size)
        : PyDistributedArray(ctx, size, partition, placement, execution,
                             device_id, block_size) {
        fill_local_data(fill_value);
    }

    ~PyDistributedArray() {
        if (arr_) {
            dtl_array_destroy(arr_);
        }
    }

    // Disable copy
    PyDistributedArray(const PyDistributedArray&) = delete;
    PyDistributedArray& operator=(const PyDistributedArray&) = delete;

    // Enable move
    PyDistributedArray(PyDistributedArray&& other) noexcept
        : arr_(other.arr_), placement_(other.placement_), dirty_(other.dirty_) {
        other.arr_ = nullptr;
    }

    PyDistributedArray& operator=(PyDistributedArray&& other) noexcept {
        if (this != &other) {
            if (arr_) {
                dtl_array_destroy(arr_);
            }
            arr_ = other.arr_;
            placement_ = other.placement_;
            dirty_ = other.dirty_;
            other.arr_ = nullptr;
        }
        return *this;
    }

    // Properties
    dtl_size_t global_size() const { return dtl_array_global_size(arr_); }
    dtl_size_t local_size() const { return dtl_array_local_size(arr_); }
    dtl_index_t local_offset() const { return dtl_array_local_offset(arr_); }

    /**
     * @brief Get the placement policy (Phase 05)
     */
    int get_placement() const { return static_cast<int>(placement_); }

    /**
     * @brief Check if local_view is safe (host-accessible placement)
     */
    bool is_host_accessible() const {
        return is_host_accessible_placement(placement_);
    }

    /**
     * @brief Get zero-copy NumPy view of local data
     *
     * @throws std::runtime_error if placement is device-only (Phase 05)
     */
    py::array_t<T> local_view() {
        // Phase 05: Guard against device-only placement
        if (!is_host_accessible()) {
            throw std::runtime_error(
                "Cannot create NumPy view of device-only memory. "
                "Use to_numpy() to copy data to host.");
        }

        void* data = dtl_array_local_data_mut(arr_);
        dtl_size_t size = dtl_array_local_size(arr_);

        // Keep the Python wrapper alive while NumPy view references local storage.
        py::object owner = py::cast(this, py::return_value_policy::reference);

        return py::array_t<T>(
            {static_cast<py::ssize_t>(size)},
            {sizeof(T)},
            static_cast<T*>(data),
            owner
        );
    }

    /**
     * @brief Copy local data to NumPy array (Phase 05)
     */
    py::array_t<T> to_numpy() {
        dtl_size_t size = dtl_array_local_size(arr_);

        py::array_t<T> result({static_cast<py::ssize_t>(size)});
        T* dest = result.mutable_data();

        dtl_status status = dtl_array_copy_to_host(arr_, dest, size);
        check_status(status);

        return result;
    }

    /**
     * @brief Copy data from NumPy array to local array data (Phase 05)
     */
    void from_numpy(py::array_t<T> arr) {
        dtl_size_t size = dtl_array_local_size(arr_);

        if (static_cast<dtl_size_t>(arr.size()) != size) {
            throw std::runtime_error(
                "NumPy array size (" + std::to_string(arr.size()) +
                ") does not match local array size (" + std::to_string(size) + ")");
        }

        py::array_t<T, py::array::c_style | py::array::forcecast> contiguous_arr(arr);
        const T* src = contiguous_arr.data();

        dtl_status status = dtl_array_copy_from_host(arr_, src, size);
        check_status(status);
    }

    /**
     * @brief Fill array with value
     */
    void fill(T value) {
        dtl_status status = dtl_array_fill_local(arr_, &value);
        check_status(status);
        dirty_ = true;
    }

    /**
     * @brief Check if the array has uncommitted modifications
     */
    bool is_dirty() const {
        return dirty_;
    }

    /**
     * @brief Synchronize the array (barrier + mark clean)
     */
    void sync() {
        dtl_status status = dtl_array_barrier(arr_);
        check_status(status);
        dirty_ = false;
    }

    /**
     * @brief Redistribute the array (not supported - arrays have fixed size)
     */
    void redistribute(int /*partition_type*/) {
        throw std::runtime_error(
            "Array redistribution is not supported. "
            "DistributedArray has a fixed partition scheme. "
            "Use DistributedVector for redistribution capabilities.");
    }

    // Internal access
    dtl_array_t native() const { return arr_; }

private:
    /**
     * @brief Helper to fill local data (used by constructors)
     */
    void fill_local_data(T fill_value) {
        if (!is_host_accessible()) {
            dtl_status status = dtl_array_fill_local(arr_, &fill_value);
            check_status(status);
        } else {
            void* data = dtl_array_local_data_mut(arr_);
            dtl_size_t size = dtl_array_local_size(arr_);
            T* typed_data = static_cast<T*>(data);
            for (dtl_size_t i = 0; i < size; ++i) {
                typed_data[i] = fill_value;
            }
        }
    }

    dtl_array_t arr_ = nullptr;
    dtl_placement_policy placement_ = DTL_PLACEMENT_HOST;
    bool dirty_ = false;
};

/**
 * @brief Bind a DistributedArray type for a specific dtype
 *
 * @note Phase 05 (P05): Added constructors with policy parameters
 */
template <typename T, dtl_dtype DType>
void bind_array_type(py::module_& m, const char* name) {
    using Arr = PyDistributedArray<T, DType>;

    py::class_<Arr>(m, name)
        // Original constructors (backwards compatibility)
        .def(py::init([](py::object ctx_obj, dtl_size_t size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Arr(ctx, size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             "Create a distributed array with the given fixed size")
        .def(py::init([](py::object ctx_obj, dtl_size_t size, T fill) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Arr(ctx, size, fill);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("fill"),
             "Create a distributed array with fill value")
        // Phase 05: Constructor with policy options
        .def(py::init([](py::object ctx_obj, dtl_size_t size,
                         int partition, int placement, int execution,
                         int device_id, dtl_size_t block_size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Arr(ctx, size, partition, placement, execution,
                           device_id, block_size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("partition"),
             py::arg("placement"),
             py::arg("execution"),
             py::arg("device_id"),
             py::arg("block_size"),
             "Create a distributed array with policy options")
        // Phase 05: Constructor with policy options and fill
        .def(py::init([](py::object ctx_obj, dtl_size_t size, T fill,
                         int partition, int placement, int execution,
                         int device_id, dtl_size_t block_size) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            return new Arr(ctx, size, fill, partition, placement, execution,
                           device_id, block_size);
        }),
             py::arg("ctx"),
             py::arg("size"),
             py::arg("fill"),
             py::arg("partition"),
             py::arg("placement"),
             py::arg("execution"),
             py::arg("device_id"),
             py::arg("block_size"),
             "Create a distributed array with policy options and fill value")
        .def_property_readonly("global_size", &Arr::global_size,
                               "Total number of elements across all ranks (fixed)")
        .def_property_readonly("local_size", &Arr::local_size,
                               "Number of elements on this rank")
        .def_property_readonly("local_offset", &Arr::local_offset,
                               "Global index of first local element")
        .def_property_readonly("placement", &Arr::get_placement,
                               "Placement policy (Phase 05)")
        .def_property_readonly("is_host_accessible", &Arr::is_host_accessible,
                               "True if local_view() is safe")
        .def("local_view", &Arr::local_view,
             py::return_value_policy::reference_internal,
             R"doc(
Get a NumPy array view of local data.

The returned array shares memory with the array - modifications
to the NumPy array will modify the distributed array data.

Raises:
    RuntimeError: If placement is device-only. Use to_numpy() instead.

Returns:
    numpy.ndarray: Mutable view of local data
)doc")
        .def("to_numpy", &Arr::to_numpy,
             R"doc(
Copy local data to a new NumPy array.

Works for all placements including device-only. Returns a copy,
not a view.

Returns:
    numpy.ndarray: Copy of local data
)doc")
        .def("from_numpy", &Arr::from_numpy,
             py::arg("arr"),
             R"doc(
Copy data from a NumPy array to local array data.

Works for all placements including device-only.

Args:
    arr: Source NumPy array (must match local_size)
)doc")
        .def("fill", &Arr::fill,
             py::arg("value"),
             "Fill all local elements with the given value")
        .def("is_dirty", &Arr::is_dirty,
             "Check if the array has uncommitted modifications")
        .def("sync", &Arr::sync,
             py::call_guard<py::gil_scoped_release>(),
             "Synchronize the array (barrier + mark clean)")
        .def("redistribute", &Arr::redistribute,
             py::arg("partition_type"),
             "Redistribute the array (not supported - raises RuntimeError)")
        .def("native", [](const Arr& a) {
            return reinterpret_cast<std::uintptr_t>(a.native());
        }, "Get native handle (for internal use by algorithm functions)")
        .def("__len__", &Arr::global_size)
        .def("__repr__", [name](const Arr& a) {
            return std::string("<") + name + " global_size=" +
                   std::to_string(a.global_size()) +
                   " local_size=" + std::to_string(a.local_size()) +
                   " placement=" + std::to_string(a.get_placement()) +
                   " (fixed)>";
        });
}

// ============================================================================
// DistributedSpan Template Wrapper
// ============================================================================

/**
 * @brief Python wrapper for dtl_span_t with specific element type
 *
 * DistributedSpan is non-owning. The source container object is retained to
 * enforce owner lifetime while the span exists.
 */
template <typename T, dtl_dtype DType>
class PyDistributedSpan {
public:
    /**
     * @brief Construct a span from a distributed vector/array/tensor wrapper
     */
    explicit PyDistributedSpan(py::object source)
        : span_(create_native_span_from_source(source))
        , owner_(std::move(source)) {
        verify_expected_dtype();
    }

    ~PyDistributedSpan() {
        if (span_) {
            dtl_span_destroy(span_);
        }
    }

    // Disable copy
    PyDistributedSpan(const PyDistributedSpan&) = delete;
    PyDistributedSpan& operator=(const PyDistributedSpan&) = delete;

    // Enable move
    PyDistributedSpan(PyDistributedSpan&& other) noexcept
        : span_(other.span_), owner_(std::move(other.owner_)) {
        other.span_ = nullptr;
    }

    PyDistributedSpan& operator=(PyDistributedSpan&& other) noexcept {
        if (this != &other) {
            if (span_) {
                dtl_span_destroy(span_);
            }
            span_ = other.span_;
            owner_ = std::move(other.owner_);
            other.span_ = nullptr;
        }
        return *this;
    }

    // Properties
    dtl_size_t global_size() const { return dtl_span_size(span_); }
    dtl_size_t local_size() const { return dtl_span_local_size(span_); }
    dtl_size_t size_bytes() const { return dtl_span_size_bytes(span_); }
    int empty() const { return dtl_span_empty(span_); }
    int is_valid() const { return dtl_span_is_valid(span_); }
    int rank() const { return dtl_span_rank(span_); }
    int num_ranks() const { return dtl_span_num_ranks(span_); }

    /**
     * @brief Get a zero-copy NumPy local view
     */
    py::array_t<T> local_view() {
        void* data = dtl_span_data_mut(span_);
        const dtl_size_t size = dtl_span_local_size(span_);
        if (size > 0 && data == nullptr) {
            throw std::runtime_error(
                "Cannot create local_view for span without host-accessible local data");
        }

        // Keep this wrapper alive while NumPy references span storage.
        py::object owner = py::cast(this, py::return_value_policy::reference);

        return py::array_t<T>(
            {static_cast<py::ssize_t>(size)},
            {sizeof(T)},
            static_cast<T*>(data),
            owner
        );
    }

    /**
     * @brief Copy local span data to a NumPy array
     */
    py::array_t<T> to_numpy() {
        const dtl_size_t size = dtl_span_local_size(span_);
        py::array_t<T> result({static_cast<py::ssize_t>(size)});
        auto* out = result.mutable_data();

        for (dtl_size_t i = 0; i < size; ++i) {
            T value{};
            check_status(dtl_span_get_local(span_, i, &value));
            out[i] = value;
        }
        return result;
    }

    /**
     * @brief Copy NumPy data into local span storage
     */
    void from_numpy(py::array_t<T> arr) {
        const dtl_size_t size = dtl_span_local_size(span_);
        if (static_cast<dtl_size_t>(arr.size()) != size) {
            throw std::runtime_error(
                "NumPy array size (" + std::to_string(arr.size()) +
                ") does not match local span size (" + std::to_string(size) + ")");
        }

        py::array_t<T, py::array::c_style | py::array::forcecast> contiguous_arr(arr);
        const T* src = contiguous_arr.data();
        for (dtl_size_t i = 0; i < size; ++i) {
            check_status(dtl_span_set_local(span_, i, &src[i]));
        }
    }

    /**
     * @brief Read a local element
     */
    T get_local(dtl_size_t idx) const {
        T value{};
        check_status(dtl_span_get_local(span_, idx, &value));
        return value;
    }

    /**
     * @brief Write a local element
     */
    void set_local(dtl_size_t idx, T value) {
        check_status(dtl_span_set_local(span_, idx, &value));
    }

    /**
     * @brief Create first-N local subspan
     */
    PyDistributedSpan first(dtl_size_t count) const {
        dtl_span_t out = nullptr;
        check_status(dtl_span_first(span_, count, &out));
        py::object owner = py::cast(
            const_cast<PyDistributedSpan*>(this),
            py::return_value_policy::reference);
        return PyDistributedSpan(out, owner);
    }

    /**
     * @brief Create last-N local subspan
     */
    PyDistributedSpan last(dtl_size_t count) const {
        dtl_span_t out = nullptr;
        check_status(dtl_span_last(span_, count, &out));
        py::object owner = py::cast(
            const_cast<PyDistributedSpan*>(this),
            py::return_value_policy::reference);
        return PyDistributedSpan(out, owner);
    }

    /**
     * @brief Create local subspan
     */
    PyDistributedSpan subspan(dtl_size_t offset, dtl_size_t count) const {
        dtl_span_t out = nullptr;
        check_status(dtl_span_subspan(span_, offset, count, &out));
        py::object owner = py::cast(
            const_cast<PyDistributedSpan*>(this),
            py::return_value_policy::reference);
        return PyDistributedSpan(out, owner);
    }

    // Internal access
    dtl_span_t native() const { return span_; }

private:
    /**
     * @brief Internal constructor for subspan-producing methods
     */
    PyDistributedSpan(dtl_span_t span, py::object owner)
        : span_(span), owner_(std::move(owner)) {
        verify_expected_dtype();
    }

    /**
     * @brief Ensure runtime dtype agrees with template dtype
     */
    void verify_expected_dtype() {
        if (!span_ || dtl_span_is_valid(span_) == 0) {
            if (span_) {
                dtl_span_destroy(span_);
                span_ = nullptr;
            }
            throw std::runtime_error("Invalid span handle");
        }
        const dtl_dtype runtime_dtype = dtl_span_dtype(span_);
        if (runtime_dtype != DType) {
            dtl_span_destroy(span_);
            span_ = nullptr;
            throw std::runtime_error(
                "DistributedSpan dtype mismatch: expected " +
                std::string(dtl_dtype_name(DType)) + ", got " +
                std::string(dtl_dtype_name(runtime_dtype)));
        }
    }

    dtl_span_t span_ = nullptr;
    py::object owner_;
};

/**
 * @brief Bind a DistributedSpan type for a specific dtype
 */
template <typename T, dtl_dtype DType>
void bind_span_type(py::module_& m, const char* name) {
    using Span = PyDistributedSpan<T, DType>;

    py::class_<Span>(m, name)
        .def(py::init([](py::object source) {
            return new Span(source);
        }),
             py::arg("source"),
             R"doc(
Create a non-owning distributed span from a distributed container.

Args:
    source: Typed DistributedVector_*, DistributedArray_*, or DistributedTensor_* object
)doc")
        .def_property_readonly("global_size", &Span::global_size,
                               "Total number of elements represented by the span")
        .def_property_readonly("local_size", &Span::local_size,
                               "Number of local elements represented by the span")
        .def_property_readonly("size_bytes", &Span::size_bytes,
                               "Local size in bytes")
        .def_property_readonly("empty", &Span::empty,
                               "True if the span has zero global size")
        .def_property_readonly("is_valid", &Span::is_valid,
                               "True if the native span handle is valid")
        .def_property_readonly("rank", &Span::rank,
                               "Current rank id metadata")
        .def_property_readonly("num_ranks", &Span::num_ranks,
                               "Total rank count metadata")
        .def("local_view", &Span::local_view,
             py::return_value_policy::reference_internal,
             R"doc(
Get a NumPy array view of local span data.

The returned array shares memory with the span's backing owner container.
)doc")
        .def("to_numpy", &Span::to_numpy,
             "Copy local span data to a new NumPy array")
        .def("from_numpy", &Span::from_numpy,
             py::arg("arr"),
             "Copy NumPy data into local span storage")
        .def("get_local", &Span::get_local,
             py::arg("idx"),
             "Read local element by index")
        .def("set_local", &Span::set_local,
             py::arg("idx"),
             py::arg("value"),
             "Write local element by index")
        .def("first", &Span::first,
             py::arg("count"),
             "Create a span for the first local count elements")
        .def("last", &Span::last,
             py::arg("count"),
             "Create a span for the last local count elements")
        .def("subspan", &Span::subspan,
             py::arg("offset"),
             py::arg("count") = DTL_SPAN_NPOS,
             "Create a local subspan")
        .def("native", [](const Span& s) {
            return reinterpret_cast<std::uintptr_t>(s.native());
        }, "Get native span handle (for internal use)")
        .def("__len__", &Span::global_size)
        .def("__repr__", [name](const Span& s) {
            return std::string("<") + name + " global_size=" +
                   std::to_string(s.global_size()) +
                   " local_size=" + std::to_string(s.local_size()) + ">";
        });
}

void bind_containers(py::module_& m) {
    // Bind vector types for each supported dtype
    bind_vector_type<double, DTL_DTYPE_FLOAT64>(m, "DistributedVector_f64");
    bind_vector_type<float, DTL_DTYPE_FLOAT32>(m, "DistributedVector_f32");
    bind_vector_type<int64_t, DTL_DTYPE_INT64>(m, "DistributedVector_i64");
    bind_vector_type<int32_t, DTL_DTYPE_INT32>(m, "DistributedVector_i32");
    bind_vector_type<uint64_t, DTL_DTYPE_UINT64>(m, "DistributedVector_u64");
    bind_vector_type<uint32_t, DTL_DTYPE_UINT32>(m, "DistributedVector_u32");
    bind_vector_type<uint8_t, DTL_DTYPE_UINT8>(m, "DistributedVector_u8");
    bind_vector_type<int8_t, DTL_DTYPE_INT8>(m, "DistributedVector_i8");

    // Bind array types for each supported dtype
    bind_array_type<double, DTL_DTYPE_FLOAT64>(m, "DistributedArray_f64");
    bind_array_type<float, DTL_DTYPE_FLOAT32>(m, "DistributedArray_f32");
    bind_array_type<int64_t, DTL_DTYPE_INT64>(m, "DistributedArray_i64");
    bind_array_type<int32_t, DTL_DTYPE_INT32>(m, "DistributedArray_i32");
    bind_array_type<uint64_t, DTL_DTYPE_UINT64>(m, "DistributedArray_u64");
    bind_array_type<uint32_t, DTL_DTYPE_UINT32>(m, "DistributedArray_u32");
    bind_array_type<uint8_t, DTL_DTYPE_UINT8>(m, "DistributedArray_u8");
    bind_array_type<int8_t, DTL_DTYPE_INT8>(m, "DistributedArray_i8");

    // Bind distributed span types for each supported dtype
    bind_span_type<double, DTL_DTYPE_FLOAT64>(m, "DistributedSpan_f64");
    bind_span_type<float, DTL_DTYPE_FLOAT32>(m, "DistributedSpan_f32");
    bind_span_type<int64_t, DTL_DTYPE_INT64>(m, "DistributedSpan_i64");
    bind_span_type<int32_t, DTL_DTYPE_INT32>(m, "DistributedSpan_i32");
    bind_span_type<uint64_t, DTL_DTYPE_UINT64>(m, "DistributedSpan_u64");
    bind_span_type<uint32_t, DTL_DTYPE_UINT32>(m, "DistributedSpan_u32");
    bind_span_type<uint8_t, DTL_DTYPE_UINT8>(m, "DistributedSpan_u8");
    bind_span_type<int8_t, DTL_DTYPE_INT8>(m, "DistributedSpan_i8");

    // Bind tensor types for commonly used dtypes
    bind_tensor_type<double, DTL_DTYPE_FLOAT64>(m, "DistributedTensor_f64");
    bind_tensor_type<float, DTL_DTYPE_FLOAT32>(m, "DistributedTensor_f32");
    bind_tensor_type<int64_t, DTL_DTYPE_INT64>(m, "DistributedTensor_i64");
    bind_tensor_type<int32_t, DTL_DTYPE_INT32>(m, "DistributedTensor_i32");

    // ========================================================================
    // DistributedMap (Phase 16 — type-erased wrapper)
    // ========================================================================

    m.def("map_create",
        [](py::object ctx_obj, int key_dtype, int value_dtype) -> std::uintptr_t {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_map_t map = nullptr;
            dtl_status status = dtl_map_create(
                ctx,
                static_cast<dtl_dtype>(key_dtype),
                static_cast<dtl_dtype>(value_dtype),
                &map);
            check_status(status);
            return reinterpret_cast<std::uintptr_t>(map);
        },
        py::arg("ctx"),
        py::arg("key_dtype"),
        py::arg("value_dtype"),
        "Create a distributed map with specified key and value dtypes");

    m.def("map_destroy",
        [](std::uintptr_t map_ptr) {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            dtl_map_destroy(map);
        },
        py::arg("map"),
        "Destroy a distributed map");

    m.def("map_insert",
        [](std::uintptr_t map_ptr, py::object key_obj, py::object val_obj,
           int key_dtype, int val_dtype) {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);

            std::aligned_storage_t<8, 8> key_storage, val_storage;
            void* key_ptr = &key_storage;
            void* val_ptr = &val_storage;

            switch (static_cast<dtl_dtype>(key_dtype)) {
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(key_ptr) = key_obj.cast<int32_t>(); break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(key_ptr) = key_obj.cast<uint64_t>(); break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(key_ptr) = key_obj.cast<uint32_t>(); break;
                default:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
            }

            switch (static_cast<dtl_dtype>(val_dtype)) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(val_ptr) = val_obj.cast<double>(); break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(val_ptr) = val_obj.cast<float>(); break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(val_ptr) = val_obj.cast<int64_t>(); break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(val_ptr) = val_obj.cast<int32_t>(); break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(val_ptr) = val_obj.cast<uint64_t>(); break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(val_ptr) = val_obj.cast<uint32_t>(); break;
                default:
                    *static_cast<double*>(val_ptr) = val_obj.cast<double>(); break;
            }

            dtl_status status = dtl_map_insert(map, key_ptr, val_ptr);
            check_status(status);
        },
        py::arg("map"),
        py::arg("key"),
        py::arg("value"),
        py::arg("key_dtype"),
        py::arg("value_dtype"),
        "Insert a key-value pair into the distributed map");

    m.def("map_find",
        [](std::uintptr_t map_ptr, py::object key_obj,
           int key_dtype, int val_dtype) -> py::object {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);

            std::aligned_storage_t<8, 8> key_storage, val_storage;
            void* key_ptr = &key_storage;
            void* val_ptr = &val_storage;

            switch (static_cast<dtl_dtype>(key_dtype)) {
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(key_ptr) = key_obj.cast<int32_t>(); break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(key_ptr) = key_obj.cast<uint64_t>(); break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(key_ptr) = key_obj.cast<uint32_t>(); break;
                default:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
            }

            dtl_status status = dtl_map_find_local(map, key_ptr, val_ptr);
            if (status != DTL_SUCCESS) {
                return py::none();
            }

            switch (static_cast<dtl_dtype>(val_dtype)) {
                case DTL_DTYPE_FLOAT64:
                    return py::cast(*static_cast<double*>(val_ptr));
                case DTL_DTYPE_FLOAT32:
                    return py::cast(*static_cast<float*>(val_ptr));
                case DTL_DTYPE_INT64:
                    return py::cast(*static_cast<int64_t*>(val_ptr));
                case DTL_DTYPE_INT32:
                    return py::cast(*static_cast<int32_t*>(val_ptr));
                case DTL_DTYPE_UINT64:
                    return py::cast(*static_cast<uint64_t*>(val_ptr));
                case DTL_DTYPE_UINT32:
                    return py::cast(*static_cast<uint32_t*>(val_ptr));
                default:
                    return py::cast(*static_cast<double*>(val_ptr));
            }
        },
        py::arg("map"),
        py::arg("key"),
        py::arg("key_dtype"),
        py::arg("value_dtype"),
        "Find a value by key in the local map. Returns None if not found.");

    m.def("map_erase",
        [](std::uintptr_t map_ptr, py::object key_obj, int key_dtype) -> bool {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);

            std::aligned_storage_t<8, 8> key_storage;
            void* key_ptr = &key_storage;

            switch (static_cast<dtl_dtype>(key_dtype)) {
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(key_ptr) = key_obj.cast<int32_t>(); break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(key_ptr) = key_obj.cast<uint64_t>(); break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(key_ptr) = key_obj.cast<uint32_t>(); break;
                default:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
            }

            dtl_size_t erased = 0;
            dtl_status status = dtl_map_erase(map, key_ptr, &erased);
            check_status(status);
            return erased > 0;
        },
        py::arg("map"),
        py::arg("key"),
        py::arg("key_dtype"),
        "Erase a key from the map. Returns True if the key was found and erased.");

    m.def("map_contains",
        [](std::uintptr_t map_ptr, py::object key_obj, int key_dtype) -> bool {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);

            std::aligned_storage_t<8, 8> key_storage;
            void* key_ptr = &key_storage;

            switch (static_cast<dtl_dtype>(key_dtype)) {
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(key_ptr) = key_obj.cast<int32_t>(); break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(key_ptr) = key_obj.cast<uint64_t>(); break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(key_ptr) = key_obj.cast<uint32_t>(); break;
                default:
                    *static_cast<int64_t*>(key_ptr) = key_obj.cast<int64_t>(); break;
            }

            return dtl_map_contains_local(map, key_ptr) != 0;
        },
        py::arg("map"),
        py::arg("key"),
        py::arg("key_dtype"),
        "Check if a key exists locally in the map.");

    m.def("map_local_entries",
        [](std::uintptr_t map_ptr, int key_dtype, int val_dtype) -> py::list {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            dtl_map_iter_t iter = nullptr;
            dtl_status status = dtl_map_iter_begin(map, &iter);
            check_status(status);

            py::list entries;
            try {
                while (true) {
                    std::aligned_storage_t<8, 8> key_storage{};
                    std::aligned_storage_t<8, 8> val_storage{};
                    void* key_ptr = &key_storage;
                    void* val_ptr = &val_storage;

                    status = dtl_map_iter_key(iter, key_ptr);
                    if (status == DTL_ERROR_OUT_OF_BOUNDS || status == DTL_END) {
                        break;
                    }
                    check_status(status);

                    status = dtl_map_iter_value(iter, val_ptr);
                    check_status(status);

                    py::object key_obj_py;
                    py::object val_obj_py;

                    switch (static_cast<dtl_dtype>(key_dtype)) {
                        case DTL_DTYPE_INT64:
                            key_obj_py = py::cast(*static_cast<int64_t*>(key_ptr)); break;
                        case DTL_DTYPE_INT32:
                            key_obj_py = py::cast(*static_cast<int32_t*>(key_ptr)); break;
                        case DTL_DTYPE_UINT64:
                            key_obj_py = py::cast(*static_cast<uint64_t*>(key_ptr)); break;
                        case DTL_DTYPE_UINT32:
                            key_obj_py = py::cast(*static_cast<uint32_t*>(key_ptr)); break;
                        case DTL_DTYPE_INT8:
                            key_obj_py = py::cast(*static_cast<int8_t*>(key_ptr)); break;
                        case DTL_DTYPE_UINT8:
                            key_obj_py = py::cast(*static_cast<uint8_t*>(key_ptr)); break;
                        default:
                            key_obj_py = py::cast(*static_cast<int64_t*>(key_ptr)); break;
                    }

                    switch (static_cast<dtl_dtype>(val_dtype)) {
                        case DTL_DTYPE_FLOAT64:
                            val_obj_py = py::cast(*static_cast<double*>(val_ptr)); break;
                        case DTL_DTYPE_FLOAT32:
                            val_obj_py = py::cast(*static_cast<float*>(val_ptr)); break;
                        case DTL_DTYPE_INT64:
                            val_obj_py = py::cast(*static_cast<int64_t*>(val_ptr)); break;
                        case DTL_DTYPE_INT32:
                            val_obj_py = py::cast(*static_cast<int32_t*>(val_ptr)); break;
                        case DTL_DTYPE_UINT64:
                            val_obj_py = py::cast(*static_cast<uint64_t*>(val_ptr)); break;
                        case DTL_DTYPE_UINT32:
                            val_obj_py = py::cast(*static_cast<uint32_t*>(val_ptr)); break;
                        case DTL_DTYPE_INT8:
                            val_obj_py = py::cast(*static_cast<int8_t*>(val_ptr)); break;
                        case DTL_DTYPE_UINT8:
                            val_obj_py = py::cast(*static_cast<uint8_t*>(val_ptr)); break;
                        default:
                            val_obj_py = py::cast(*static_cast<double*>(val_ptr)); break;
                    }

                    entries.append(py::make_tuple(key_obj_py, val_obj_py));

                    status = dtl_map_iter_next(iter);
                    if (status == DTL_END) {
                        break;
                    }
                    check_status(status);
                }
            } catch (...) {
                if (iter) {
                    dtl_map_iter_destroy(iter);
                }
                throw;
            }

            if (iter) {
                dtl_map_iter_destroy(iter);
            }
            return entries;
        },
        py::arg("map"),
        py::arg("key_dtype"),
        py::arg("value_dtype"),
        "Return a snapshot list of local (key, value) pairs.");

    m.def("map_local_size",
        [](std::uintptr_t map_ptr) -> dtl_size_t {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            return dtl_map_local_size(map);
        },
        py::arg("map"),
        "Get the number of key-value pairs stored locally.");

    m.def("map_global_size",
        [](std::uintptr_t map_ptr) -> dtl_size_t {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            return dtl_map_global_size(map);
        },
        py::arg("map"),
        py::call_guard<py::gil_scoped_release>(),
        "Get the number of key-value pairs. In v0.1.0-alpha.1 returns the local size (no cross-rank allreduce).");

    m.def("map_flush",
        [](std::uintptr_t map_ptr) {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            dtl_status status = dtl_map_flush(map);
            check_status(status);
        },
        py::arg("map"),
        py::call_guard<py::gil_scoped_release>(),
        "Flush pending operations. In v0.1.0-alpha.1 this is a local no-op (clears dirty flag).");

    m.def("map_sync",
        [](std::uintptr_t map_ptr) {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            dtl_status status = dtl_map_sync(map);
            check_status(status);
        },
        py::arg("map"),
        py::call_guard<py::gil_scoped_release>(),
        "Synchronize the map. In v0.1.0-alpha.1 this is a local no-op (clears dirty flag).");

    m.def("map_clear",
        [](std::uintptr_t map_ptr) {
            dtl_map_t map = reinterpret_cast<dtl_map_t>(map_ptr);
            dtl_status status = dtl_map_clear(map);
            check_status(status);
        },
        py::arg("map"),
        py::call_guard<py::gil_scoped_release>(),
        "Clear all elements from the local partition. In v0.1.0-alpha.1 this clears locally only.");
}
