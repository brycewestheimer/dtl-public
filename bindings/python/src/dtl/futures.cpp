// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file futures.cpp
 * @brief DTL Python bindings - Futures and asynchronous completion
 * @since 0.1.0
 *
 * Provides Python bindings for futures-based asynchronous programming,
 * including future creation, completion waiting, value transfer, and
 * combinators (when_all, when_any).
 *
 * WARNING: The futures API is experimental. The progress engine has known
 * stability issues (see KNOWN_ISSUES.md).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"
#include <dtl/bindings/c/dtl_futures.h>

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

}  // namespace

// ============================================================================
// Future Wrapper Class
// ============================================================================

/**
 * @brief Python wrapper for dtl_future_t
 *
 * Represents an asynchronous value that may not yet be available.
 */
class PyFuture {
public:
    /**
     * @brief Create an incomplete future
     */
    PyFuture() {
        dtl_status status = dtl_future_create(&fut_);
        check_status(status);
    }

    /**
     * @brief Construct from an existing native handle (takes ownership)
     */
    explicit PyFuture(dtl_future_t fut) : fut_(fut) {}

    ~PyFuture() {
        if (fut_) {
            dtl_future_destroy(fut_);
        }
    }

    // Disable copy
    PyFuture(const PyFuture&) = delete;
    PyFuture& operator=(const PyFuture&) = delete;

    // Enable move
    PyFuture(PyFuture&& other) noexcept : fut_(other.fut_) {
        other.fut_ = nullptr;
    }

    PyFuture& operator=(PyFuture&& other) noexcept {
        if (this != &other) {
            if (fut_) {
                dtl_future_destroy(fut_);
            }
            fut_ = other.fut_;
            other.fut_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Block until the future is complete
     */
    void wait() {
        if (!fut_) {
            throw std::runtime_error("Future has been destroyed");
        }
        py::gil_scoped_release release;
        dtl_status status = dtl_future_wait(fut_);
        check_status(status);
    }

    /**
     * @brief Non-blocking test for completion
     */
    bool test() {
        if (!fut_) {
            throw std::runtime_error("Future has been destroyed");
        }
        int completed = 0;
        dtl_status status = dtl_future_test(fut_, &completed);
        check_status(status);
        return completed != 0;
    }

    /**
     * @brief Get the result value as bytes
     */
    py::bytes get(dtl_size_t size) {
        if (!fut_) {
            throw std::runtime_error("Future has been destroyed");
        }
        std::vector<char> buffer(size);
        dtl_status status = dtl_future_get(fut_, buffer.data(), size);
        check_status(status);
        return py::bytes(buffer.data(), size);
    }

    /**
     * @brief Set the result value and mark complete
     */
    void set(py::bytes data) {
        if (!fut_) {
            throw std::runtime_error("Future has been destroyed");
        }
        char* buf = nullptr;
        py::ssize_t len = 0;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buf, &len) != 0) {
            throw std::runtime_error("Failed to extract bytes data");
        }
        dtl_status status = dtl_future_set(fut_, buf, static_cast<dtl_size_t>(len));
        check_status(status);
    }

    /**
     * @brief Explicitly destroy the future
     */
    void destroy() {
        if (fut_) {
            dtl_future_destroy(fut_);
            fut_ = nullptr;
        }
    }

    /**
     * @brief Get the native handle (for internal use)
     */
    std::uintptr_t native() const {
        return reinterpret_cast<std::uintptr_t>(fut_);
    }

    /**
     * @brief Get the raw native handle
     */
    dtl_future_t native_handle() const { return fut_; }

private:
    dtl_future_t fut_ = nullptr;
};

// ============================================================================
// Module Binding
// ============================================================================

void init_futures(py::module_& m) {
    auto futures = m.def_submodule("futures",
        "Experimental: Futures and asynchronous completion.\n\n"
        "WARNING: The futures API is experimental. The progress engine has\n"
        "known stability issues.");

    // Future class
    py::class_<PyFuture>(futures, "Future",
        R"doc(
Represents an asynchronous value that may not yet be available.

WARNING: Experimental API. The progress engine has known stability issues.

Futures can be waited on (blocking), tested (non-blocking), and composed
using when_all/when_any combinators.

Example:
    >>> fut = dtl.futures.Future()
    >>> fut.set(b"hello")
    >>> assert fut.test()
    >>> result = fut.get(5)
)doc")
        .def(py::init<>(),
             "Create an incomplete future")
        .def("wait", &PyFuture::wait,
             R"doc(
Block until the future is complete.

If the future is already complete, returns immediately.
Releases the GIL while waiting.
)doc")
        .def("test", &PyFuture::test,
             R"doc(
Non-blocking test for future completion.

Returns:
    True if the future is complete, False if still pending
)doc")
        .def("get", &PyFuture::get,
             py::arg("size"),
             R"doc(
Get the result value from a completed future.

The future must be complete before calling this method.

Args:
    size: Size of the expected result in bytes

Returns:
    Result value as bytes
)doc")
        .def("set", &PyFuture::set,
             py::arg("data"),
             R"doc(
Set the result value and mark the future as complete.

Stores a copy of the value and wakes any threads blocked in wait().

Args:
    data: Value to store as bytes
)doc")
        .def("destroy", &PyFuture::destroy,
             "Explicitly destroy the future and release resources")
        .def("native", &PyFuture::native,
             "Get native handle (for internal use)")
        .def("__repr__", [](const PyFuture& fut) {
            return "<Future handle=" +
                   std::to_string(fut.native()) + ">";
        });

    // when_all combinator
    futures.def("when_all",
        [](std::vector<PyFuture*> futs) -> PyFuture* {
            if (futs.empty()) {
                throw std::runtime_error("when_all requires at least one future");
            }

            // Extract native handles
            std::vector<dtl_future_t> native_futs;
            native_futs.reserve(futs.size());
            for (auto* f : futs) {
                if (!f || !f->native_handle()) {
                    throw std::runtime_error("Invalid or destroyed future in when_all");
                }
                native_futs.push_back(f->native_handle());
            }

            dtl_future_t result = nullptr;
            dtl_status status = dtl_when_all(
                native_futs.data(),
                static_cast<dtl_size_t>(native_futs.size()),
                &result
            );
            check_status(status);

            return new PyFuture(result);
        },
        py::arg("futures"),
        py::return_value_policy::take_ownership,
        R"doc(
Create a future that completes when all input futures complete.

The returned future carries no value (zero-size completion signal).

Args:
    futures: List of Future objects to wait on

Returns:
    A new Future that completes when all inputs are done

Warning:
    Input futures must remain valid until the returned future completes.

Example:
    >>> f1 = dtl.futures.Future()
    >>> f2 = dtl.futures.Future()
    >>> all_done = dtl.futures.when_all([f1, f2])
    >>> f1.set(b"a")
    >>> f2.set(b"b")
    >>> all_done.wait()
)doc");

    // when_any combinator
    futures.def("when_any",
        [](std::vector<PyFuture*> futs) -> py::tuple {
            if (futs.empty()) {
                throw std::runtime_error("when_any requires at least one future");
            }

            // Extract native handles
            std::vector<dtl_future_t> native_futs;
            native_futs.reserve(futs.size());
            for (auto* f : futs) {
                if (!f || !f->native_handle()) {
                    throw std::runtime_error("Invalid or destroyed future in when_any");
                }
                native_futs.push_back(f->native_handle());
            }

            dtl_future_t result = nullptr;
            dtl_size_t completed_index = 0;
            dtl_status status = dtl_when_any(
                native_futs.data(),
                static_cast<dtl_size_t>(native_futs.size()),
                &result,
                &completed_index
            );
            check_status(status);

            auto* py_result = new PyFuture(result);
            return py::make_tuple(
                py::cast(py_result, py::return_value_policy::take_ownership),
                completed_index
            );
        },
        py::arg("futures"),
        R"doc(
Create a future that completes when any input future completes.

Returns a tuple of (future, index) where index is the position of
the first completed future in the input list. The index is only
valid after the returned future has completed.

Args:
    futures: List of Future objects to monitor

Returns:
    Tuple of (Future, int): the combined future and the index of the
    first completed input future

Warning:
    Input futures must remain valid until the returned future completes.

Example:
    >>> f1 = dtl.futures.Future()
    >>> f2 = dtl.futures.Future()
    >>> any_done, idx = dtl.futures.when_any([f1, f2])
    >>> f2.set(b"first!")
    >>> any_done.wait()
    >>> print(f"Future {idx} completed first")
)doc");
}
