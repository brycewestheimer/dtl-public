// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file collective.cpp
 * @brief DTL Python bindings - Collective operations (reduce, allreduce, broadcast, etc.)
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

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
    if (py::hasattr(ctx_obj, "native")) {
        py::object native_method = ctx_obj.attr("native");
        return reinterpret_cast<dtl_context_t>(
            native_method().cast<std::uintptr_t>());
    } else if (py::hasattr(ctx_obj, "_native")) {
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
 * @brief Convert NumPy dtype to DTL dtype
 */
dtl_dtype numpy_to_dtl_dtype(py::dtype dtype) {
    if (dtype.equal(py::dtype::of<double>())) return DTL_DTYPE_FLOAT64;
    if (dtype.equal(py::dtype::of<float>())) return DTL_DTYPE_FLOAT32;
    if (dtype.equal(py::dtype::of<int64_t>())) return DTL_DTYPE_INT64;
    if (dtype.equal(py::dtype::of<int32_t>())) return DTL_DTYPE_INT32;
    if (dtype.equal(py::dtype::of<uint64_t>())) return DTL_DTYPE_UINT64;
    if (dtype.equal(py::dtype::of<uint32_t>())) return DTL_DTYPE_UINT32;
    if (dtype.equal(py::dtype::of<int8_t>())) return DTL_DTYPE_INT8;
    if (dtype.equal(py::dtype::of<uint8_t>())) return DTL_DTYPE_UINT8;
    throw std::runtime_error("Unsupported dtype for collective operation");
}

/**
 * @brief Convert string to DTL reduce operation
 */
dtl_reduce_op string_to_reduce_op(const std::string& op) {
    if (op == "sum" || op == "SUM") return DTL_OP_SUM;
    if (op == "prod" || op == "PROD") return DTL_OP_PROD;
    if (op == "min" || op == "MIN") return DTL_OP_MIN;
    if (op == "max" || op == "MAX") return DTL_OP_MAX;
    if (op == "land" || op == "LAND") return DTL_OP_LAND;
    if (op == "lor" || op == "LOR") return DTL_OP_LOR;
    if (op == "band" || op == "BAND") return DTL_OP_BAND;
    if (op == "bor" || op == "BOR") return DTL_OP_BOR;
    throw std::runtime_error("Unknown reduce operation: " + op);
}

}  // namespace

// ============================================================================
// Module Binding
// ============================================================================

void bind_collective(py::module_& m) {
    // Reduce operation enum as strings
    m.attr("SUM") = "sum";
    m.attr("PROD") = "prod";
    m.attr("MIN") = "min";
    m.attr("MAX") = "max";

    // Allreduce: reduce and distribute result to all ranks
    m.def("allreduce",
        [](py::object ctx_obj, py::object data_obj, const std::string& op) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            // Convert input to contiguous numpy array
            py::module_ np = py::module_::import("numpy");
            py::array data = py::array::ensure(np.attr("ascontiguousarray")(data_obj));

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Create output array using numpy (ensures writeability)
            py::array result = np.attr("empty_like")(data);

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_allreduce(
                    ctx,
                    data.data(),
                    result.mutable_data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    reduce_op
                );
            }
            check_status(status);

            return py::cast<py::object>(result);
        },
        py::arg("ctx"),
        py::arg("data"),
        py::arg("op") = "sum",
        R"doc(
Reduce data from all ranks and distribute result to all.

Combines elements from all ranks using the specified operation
and places the result on all ranks.

Args:
    ctx: Execution context
    data: NumPy array to reduce
    op: Reduction operation ("sum", "prod", "min", "max")

Returns:
    Reduced result (same shape as input)

Example:
    >>> local_sum = np.sum(local_data)
    >>> global_sum = dtl.allreduce(ctx, np.array([local_sum]), op="sum")
)doc");

    // Reduce: reduce to root only
    m.def("reduce",
        [](py::object ctx_obj, py::object data_obj, const std::string& op, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            // Convert input to contiguous numpy array
            py::module_ np = py::module_::import("numpy");
            py::array data = py::array::ensure(np.attr("ascontiguousarray")(data_obj));

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Create output array using numpy (ensures writeability)
            py::array result = np.attr("empty_like")(data);

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_reduce(
                    ctx,
                    data.data(),
                    result.mutable_data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    reduce_op,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return py::cast<py::object>(result);
        },
        py::arg("ctx"),
        py::arg("data"),
        py::arg("op") = "sum",
        py::arg("root") = 0,
        R"doc(
Reduce data from all ranks to root.

Combines elements from all ranks using the specified operation
and places the result on the root rank only.

Args:
    ctx: Execution context
    data: NumPy array to reduce
    op: Reduction operation ("sum", "prod", "min", "max")
    root: Root rank that receives the result (default: 0)

Returns:
    Reduced result on root, undefined on other ranks
)doc");

    // Broadcast: send from root to all
    m.def("broadcast",
        [](py::object ctx_obj, py::object data_obj, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            // Convert input to contiguous numpy array and make a writable copy
            py::module_ np = py::module_::import("numpy");
            py::array data = np.attr("array")(data_obj, "dtype"_a=py::none(), "copy"_a=true);

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Broadcast modifies the buffer in place
            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_broadcast(
                    ctx,
                    data.mutable_data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return data;
        },
        py::arg("ctx"),
        py::arg("data"),
        py::arg("root") = 0,
        R"doc(
Broadcast data from root to all ranks.

The root rank's data is sent to all other ranks.
The input array is modified in place on non-root ranks.

Args:
    ctx: Execution context
    data: NumPy array (send on root, receive on others)
    root: Root rank that broadcasts (default: 0)

Returns:
    The data array (modified in place)
)doc");

    // Gather: collect from all ranks to root
    m.def("gather",
        [](py::object ctx_obj, py::object data_obj, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_rank_t rank = dtl_context_rank(ctx);
            dtl_rank_t size = dtl_context_size(ctx);

            // Convert input to contiguous numpy array
            py::module_ np = py::module_::import("numpy");
            py::array data = py::array::ensure(np.attr("ascontiguousarray")(data_obj));

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Result array: size * data.size() elements on root
            py::array result;
            if (rank == root) {
                py::list shape_list;
                shape_list.append(size);
                for (py::ssize_t i = 0; i < data.ndim(); ++i) {
                    shape_list.append(data.shape(i));
                }
                result = np.attr("empty")(py::tuple(shape_list), "dtype"_a=data.attr("dtype"));
            } else {
                // Non-root: create dummy array
                result = np.attr("empty")(py::make_tuple(0), "dtype"_a=data.attr("dtype"));
            }

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_gather(
                    ctx,
                    data.data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    result.mutable_data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return result;
        },
        py::arg("ctx"),
        py::arg("data"),
        py::arg("root") = 0,
        R"doc(
Gather data from all ranks to root.

Each rank sends its data to the root, which collects all data.

Args:
    ctx: Execution context
    data: NumPy array to send
    root: Root rank that gathers (default: 0)

Returns:
    On root: array with shape (size, *data.shape)
    On others: empty array
)doc");

    // Scatter: distribute from root to all
    m.def("scatter",
        [](py::object ctx_obj, py::object data_obj, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_rank_t rank = dtl_context_rank(ctx);
            dtl_rank_t size = dtl_context_size(ctx);

            // Convert input to contiguous numpy array
            py::module_ np = py::module_::import("numpy");
            py::array data = py::array::ensure(np.attr("ascontiguousarray")(data_obj));

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Calculate chunk size
            py::ssize_t chunk_size;
            if (rank == root) {
                if (data.ndim() < 1 || data.shape(0) != size) {
                    throw std::runtime_error(
                        "Scatter: data first dimension must equal number of ranks");
                }
                chunk_size = data.size() / size;
            } else {
                // Non-root doesn't know the size, we need to broadcast it
                // For simplicity, assume all ranks provide same-shaped data
                chunk_size = data.size() / size;
            }

            // Result array: one chunk
            py::list result_shape_list;
            if (data.ndim() > 1) {
                for (py::ssize_t i = 1; i < data.ndim(); ++i) {
                    result_shape_list.append(data.shape(i));
                }
            } else {
                result_shape_list.append(chunk_size);
            }
            py::array result = np.attr("empty")(py::tuple(result_shape_list), "dtype"_a=data.attr("dtype"));

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_scatter(
                    ctx,
                    data.data(),
                    static_cast<dtl_size_t>(chunk_size),
                    dtype,
                    result.mutable_data(),
                    static_cast<dtl_size_t>(chunk_size),
                    dtype,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return result;
        },
        py::arg("ctx"),
        py::arg("data"),
        py::arg("root") = 0,
        R"doc(
Scatter data from root to all ranks.

Root distributes portions of its data to each rank.

Args:
    ctx: Execution context
    data: NumPy array to scatter (significant on root only)
          First dimension must equal number of ranks.
    root: Root rank that scatters (default: 0)

Returns:
    One chunk of the scattered data
)doc");

    // Allgatherv: variable-count gather to all ranks
    m.def("allgatherv",
        [](py::object ctx_obj, py::array sendbuf, py::array recvcounts_arr,
           py::array displs_arr, py::array recvbuf) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype dtype = numpy_to_dtl_dtype(sendbuf.dtype());

            // Extract recvcounts and displs as arrays of dtl_size_t
            py::array_t<int64_t> rc = py::array_t<int64_t>::ensure(recvcounts_arr);
            py::array_t<int64_t> dp = py::array_t<int64_t>::ensure(displs_arr);

            // Convert to dtl_size_t arrays
            auto rc_buf = rc.unchecked<1>();
            auto dp_buf = dp.unchecked<1>();
            std::vector<dtl_size_t> recvcounts(rc_buf.shape(0));
            std::vector<dtl_size_t> displs(dp_buf.shape(0));
            for (py::ssize_t i = 0; i < rc_buf.shape(0); ++i) {
                recvcounts[i] = static_cast<dtl_size_t>(rc_buf(i));
            }
            for (py::ssize_t i = 0; i < dp_buf.shape(0); ++i) {
                displs[i] = static_cast<dtl_size_t>(dp_buf(i));
            }

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_allgatherv(
                    ctx,
                    sendbuf.data(),
                    static_cast<dtl_size_t>(sendbuf.size()),
                    dtype,
                    recvbuf.mutable_data(),
                    recvcounts.data(),
                    displs.data()
                );
            }
            check_status(status);

            return recvbuf;
        },
        py::arg("ctx"),
        py::arg("sendbuf"),
        py::arg("recvcounts"),
        py::arg("displs"),
        py::arg("recvbuf"),
        R"doc(
Variable-count gather to all ranks.

Each rank contributes a potentially different number of elements.
All ranks receive all data.

Args:
    ctx: Execution context
    sendbuf: NumPy array with local data to send
    recvcounts: Array of receive counts (one per rank)
    displs: Array of displacements in recvbuf (one per rank)
    recvbuf: Pre-allocated receive buffer

Returns:
    The receive buffer with gathered data
)doc");

    // Alltoallv: variable-count all-to-all exchange
    m.def("alltoallv",
        [](py::object ctx_obj, py::array sendbuf, py::array sendcounts_arr,
           py::array sdispls_arr, py::array recvbuf, py::array recvcounts_arr,
           py::array rdispls_arr) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype send_dtype = numpy_to_dtl_dtype(sendbuf.dtype());
            dtl_dtype recv_dtype = numpy_to_dtl_dtype(recvbuf.dtype());

            // Extract counts and displacements as dtl_size_t arrays
            py::array_t<int64_t> sc = py::array_t<int64_t>::ensure(sendcounts_arr);
            py::array_t<int64_t> sd = py::array_t<int64_t>::ensure(sdispls_arr);
            py::array_t<int64_t> rc = py::array_t<int64_t>::ensure(recvcounts_arr);
            py::array_t<int64_t> rd = py::array_t<int64_t>::ensure(rdispls_arr);

            auto sc_buf = sc.unchecked<1>();
            auto sd_buf = sd.unchecked<1>();
            auto rc_buf = rc.unchecked<1>();
            auto rd_buf = rd.unchecked<1>();

            std::vector<dtl_size_t> sendcounts(sc_buf.shape(0));
            std::vector<dtl_size_t> sdispls(sd_buf.shape(0));
            std::vector<dtl_size_t> recvcounts(rc_buf.shape(0));
            std::vector<dtl_size_t> rdispls(rd_buf.shape(0));

            for (py::ssize_t i = 0; i < sc_buf.shape(0); ++i) {
                sendcounts[i] = static_cast<dtl_size_t>(sc_buf(i));
            }
            for (py::ssize_t i = 0; i < sd_buf.shape(0); ++i) {
                sdispls[i] = static_cast<dtl_size_t>(sd_buf(i));
            }
            for (py::ssize_t i = 0; i < rc_buf.shape(0); ++i) {
                recvcounts[i] = static_cast<dtl_size_t>(rc_buf(i));
            }
            for (py::ssize_t i = 0; i < rd_buf.shape(0); ++i) {
                rdispls[i] = static_cast<dtl_size_t>(rd_buf(i));
            }

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_alltoallv(
                    ctx,
                    sendbuf.data(),
                    sendcounts.data(),
                    sdispls.data(),
                    send_dtype,
                    recvbuf.mutable_data(),
                    recvcounts.data(),
                    rdispls.data(),
                    recv_dtype
                );
            }
            check_status(status);

            return recvbuf;
        },
        py::arg("ctx"),
        py::arg("sendbuf"),
        py::arg("sendcounts"),
        py::arg("sdispls"),
        py::arg("recvbuf"),
        py::arg("recvcounts"),
        py::arg("rdispls"),
        R"doc(
Variable-count all-to-all exchange.

Each rank sends a potentially different amount of data to every other rank.

Args:
    ctx: Execution context
    sendbuf: NumPy array with data to send
    sendcounts: Array of send counts (one per rank)
    sdispls: Array of send displacements (one per rank)
    recvbuf: Pre-allocated receive buffer
    recvcounts: Array of receive counts (one per rank)
    rdispls: Array of receive displacements (one per rank)

Returns:
    The receive buffer with exchanged data
)doc");

    // Send: blocking point-to-point send
    m.def("send",
        [](py::object ctx_obj, py::array buf, int dest, int tag) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype dtype = numpy_to_dtl_dtype(buf.dtype());

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_send(
                    ctx,
                    buf.data(),
                    static_cast<dtl_size_t>(buf.size()),
                    dtype,
                    static_cast<dtl_rank_t>(dest),
                    static_cast<dtl_tag_t>(tag)
                );
            }
            check_status(status);
        },
        py::arg("ctx"),
        py::arg("buf"),
        py::arg("dest"),
        py::arg("tag") = 0,
        R"doc(
Send data to a destination rank (blocking).

Sends the contents of the buffer to the specified destination rank.
This call blocks until the send buffer can be reused.

Args:
    ctx: Execution context
    buf: NumPy array with data to send
    dest: Destination rank (0 to size-1)
    tag: Message tag (default: 0)
)doc");

    // Recv: blocking point-to-point receive
    m.def("recv",
        [](py::object ctx_obj, py::array buf, int source, int tag) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype dtype = numpy_to_dtl_dtype(buf.dtype());

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_recv(
                    ctx,
                    buf.mutable_data(),
                    static_cast<dtl_size_t>(buf.size()),
                    dtype,
                    static_cast<dtl_rank_t>(source),
                    static_cast<dtl_tag_t>(tag)
                );
            }
            check_status(status);

            return buf;
        },
        py::arg("ctx"),
        py::arg("buf"),
        py::arg("source"),
        py::arg("tag") = 0,
        R"doc(
Receive data from a source rank (blocking).

Receives data into the provided buffer from the specified source rank.
This call blocks until the message is received.

Args:
    ctx: Execution context
    buf: Pre-allocated NumPy array to receive into
    source: Source rank (0 to size-1)
    tag: Message tag (default: 0)

Returns:
    The receive buffer with received data
)doc");

    // Sendrecv: simultaneous send and receive
    m.def("sendrecv",
        [](py::object ctx_obj, py::array sendbuf, int dest, int sendtag,
           py::array recvbuf, int source, int recvtag) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype send_dtype = numpy_to_dtl_dtype(sendbuf.dtype());
            dtl_dtype recv_dtype = numpy_to_dtl_dtype(recvbuf.dtype());

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_sendrecv(
                    ctx,
                    sendbuf.data(),
                    static_cast<dtl_size_t>(sendbuf.size()),
                    send_dtype,
                    static_cast<dtl_rank_t>(dest),
                    static_cast<dtl_tag_t>(sendtag),
                    recvbuf.mutable_data(),
                    static_cast<dtl_size_t>(recvbuf.size()),
                    recv_dtype,
                    static_cast<dtl_rank_t>(source),
                    static_cast<dtl_tag_t>(recvtag)
                );
            }
            check_status(status);

            return recvbuf;
        },
        py::arg("ctx"),
        py::arg("sendbuf"),
        py::arg("dest"),
        py::arg("sendtag"),
        py::arg("recvbuf"),
        py::arg("source"),
        py::arg("recvtag"),
        R"doc(
Send and receive data simultaneously (blocking).

Performs a combined send and receive operation. Useful for
exchanging data between pairs of ranks.

Args:
    ctx: Execution context
    sendbuf: NumPy array with data to send
    dest: Destination rank
    sendtag: Send message tag
    recvbuf: Pre-allocated NumPy array for received data
    source: Source rank
    recvtag: Receive message tag

Returns:
    The receive buffer with received data
)doc");

    // Probe: blocking probe for incoming message
    m.def("probe",
        [](py::object ctx_obj, int source, int tag) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_message_info_t info;
            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_probe(
                    ctx,
                    static_cast<dtl_rank_t>(source),
                    static_cast<dtl_tag_t>(tag),
                    DTL_DTYPE_BYTE,
                    &info
                );
            }
            check_status(status);

            py::dict result;
            result["source"] = static_cast<int>(info.source);
            result["tag"] = static_cast<int>(info.tag);
            result["count"] = static_cast<int64_t>(info.count);
            return result;
        },
        py::arg("ctx"),
        py::arg("source") = DTL_ANY_SOURCE,
        py::arg("tag") = DTL_ANY_TAG,
        R"doc(
Probe for an incoming message without receiving it (blocking).

Blocks until a matching message is available and returns information
about it.

Args:
    ctx: Execution context
    source: Source rank to match (default: any source)
    tag: Tag to match (default: any tag)

Returns:
    Dict with "source" (int), "tag" (int), "count" (int) keys
)doc");

    // Iprobe: non-blocking probe for incoming message
    m.def("iprobe",
        [](py::object ctx_obj, int source, int tag) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_message_info_t info;
            int flag = 0;
            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_iprobe(
                    ctx,
                    static_cast<dtl_rank_t>(source),
                    static_cast<dtl_tag_t>(tag),
                    DTL_DTYPE_BYTE,
                    &flag,
                    &info
                );
            }
            check_status(status);

            if (flag) {
                py::dict result;
                result["source"] = static_cast<int>(info.source);
                result["tag"] = static_cast<int>(info.tag);
                result["count"] = static_cast<int64_t>(info.count);
                return py::make_tuple(true, py::cast<py::object>(result));
            } else {
                return py::make_tuple(false, py::none());
            }
        },
        py::arg("ctx"),
        py::arg("source") = DTL_ANY_SOURCE,
        py::arg("tag") = DTL_ANY_TAG,
        R"doc(
Probe for an incoming message without receiving it (non-blocking).

Returns immediately with information about whether a matching message
is available.

Args:
    ctx: Execution context
    source: Source rank to match (default: any source)
    tag: Tag to match (default: any tag)

Returns:
    Tuple of (bool, dict_or_None). If a message is available, returns
    (True, {"source": int, "tag": int, "count": int}). Otherwise
    returns (False, None).
)doc");

    // Gatherv: variable-count gather to root
    m.def("gatherv",
        [](py::object ctx_obj, py::array sendbuf, py::array recvcounts_arr,
           py::array displs_arr, py::array recvbuf, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype send_dtype = numpy_to_dtl_dtype(sendbuf.dtype());
            dtl_dtype recv_dtype = numpy_to_dtl_dtype(recvbuf.dtype());

            // Extract recvcounts and displs as arrays of dtl_size_t
            py::array_t<int64_t> rc = py::array_t<int64_t>::ensure(recvcounts_arr);
            py::array_t<int64_t> dp = py::array_t<int64_t>::ensure(displs_arr);

            auto rc_buf = rc.unchecked<1>();
            auto dp_buf = dp.unchecked<1>();
            std::vector<dtl_size_t> recvcounts(rc_buf.shape(0));
            std::vector<dtl_size_t> displs(dp_buf.shape(0));
            for (py::ssize_t i = 0; i < rc_buf.shape(0); ++i) {
                recvcounts[i] = static_cast<dtl_size_t>(rc_buf(i));
            }
            for (py::ssize_t i = 0; i < dp_buf.shape(0); ++i) {
                displs[i] = static_cast<dtl_size_t>(dp_buf(i));
            }

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_gatherv(
                    ctx,
                    sendbuf.data(),
                    static_cast<dtl_size_t>(sendbuf.size()),
                    send_dtype,
                    recvbuf.mutable_data(),
                    recvcounts.data(),
                    displs.data(),
                    recv_dtype,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return recvbuf;
        },
        py::arg("ctx"),
        py::arg("sendbuf"),
        py::arg("recvcounts"),
        py::arg("displs"),
        py::arg("recvbuf"),
        py::arg("root") = 0,
        R"doc(
Variable-count gather to root.

Each rank sends a potentially different number of elements to the root.

Args:
    ctx: Execution context
    sendbuf: NumPy array with local data to send
    recvcounts: Array of receive counts (one per rank, significant on root)
    displs: Array of displacements in recvbuf (one per rank, significant on root)
    recvbuf: Pre-allocated receive buffer (significant on root)
    root: Root rank that gathers (default: 0)

Returns:
    The receive buffer with gathered data
)doc");

    // Scatterv: variable-count scatter from root
    m.def("scatterv",
        [](py::object ctx_obj, py::array sendbuf, py::array sendcounts_arr,
           py::array displs_arr, py::array recvbuf, int root) {
            dtl_context_t ctx = get_native_context(ctx_obj);

            dtl_dtype send_dtype = numpy_to_dtl_dtype(sendbuf.dtype());
            dtl_dtype recv_dtype = numpy_to_dtl_dtype(recvbuf.dtype());

            // Extract sendcounts and displs as arrays of dtl_size_t
            py::array_t<int64_t> sc = py::array_t<int64_t>::ensure(sendcounts_arr);
            py::array_t<int64_t> dp = py::array_t<int64_t>::ensure(displs_arr);

            auto sc_buf = sc.unchecked<1>();
            auto dp_buf = dp.unchecked<1>();
            std::vector<dtl_size_t> sendcounts(sc_buf.shape(0));
            std::vector<dtl_size_t> displs(dp_buf.shape(0));
            for (py::ssize_t i = 0; i < sc_buf.shape(0); ++i) {
                sendcounts[i] = static_cast<dtl_size_t>(sc_buf(i));
            }
            for (py::ssize_t i = 0; i < dp_buf.shape(0); ++i) {
                displs[i] = static_cast<dtl_size_t>(dp_buf(i));
            }

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_scatterv(
                    ctx,
                    sendbuf.data(),
                    sendcounts.data(),
                    displs.data(),
                    send_dtype,
                    recvbuf.mutable_data(),
                    static_cast<dtl_size_t>(recvbuf.size()),
                    recv_dtype,
                    static_cast<dtl_rank_t>(root)
                );
            }
            check_status(status);

            return recvbuf;
        },
        py::arg("ctx"),
        py::arg("sendbuf"),
        py::arg("sendcounts"),
        py::arg("displs"),
        py::arg("recvbuf"),
        py::arg("root") = 0,
        R"doc(
Variable-count scatter from root.

Root distributes different amounts of data to each rank.

Args:
    ctx: Execution context
    sendbuf: NumPy array with data to scatter (significant on root)
    sendcounts: Array of send counts (one per rank, significant on root)
    displs: Array of displacements in sendbuf (one per rank, significant on root)
    recvbuf: Pre-allocated receive buffer
    root: Root rank that scatters (default: 0)

Returns:
    The receive buffer with scattered data
)doc");

    // Allgather: gather and distribute to all
    m.def("allgather",
        [](py::object ctx_obj, py::object data_obj) {
            dtl_context_t ctx = get_native_context(ctx_obj);
            dtl_rank_t size = dtl_context_size(ctx);

            // Convert input to contiguous numpy array
            py::module_ np = py::module_::import("numpy");
            py::array data = py::array::ensure(np.attr("ascontiguousarray")(data_obj));

            dtl_dtype dtype = numpy_to_dtl_dtype(data.dtype());

            // Result array: size * data.size() elements
            py::list shape_list;
            shape_list.append(size);
            for (py::ssize_t i = 0; i < data.ndim(); ++i) {
                shape_list.append(data.shape(i));
            }
            py::array result = np.attr("empty")(py::tuple(shape_list), "dtype"_a=data.attr("dtype"));

            dtl_status status;
            {
                py::gil_scoped_release release;
                status = dtl_allgather(
                    ctx,
                    data.data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype,
                    result.mutable_data(),
                    static_cast<dtl_size_t>(data.size()),
                    dtype
                );
            }
            check_status(status);

            return result;
        },
        py::arg("ctx"),
        py::arg("data"),
        R"doc(
Gather data from all ranks to all ranks.

Each rank sends its data and receives data from all ranks.

Args:
    ctx: Execution context
    data: NumPy array to send

Returns:
    Array with shape (size, *data.shape) containing all ranks' data
)doc");
}
