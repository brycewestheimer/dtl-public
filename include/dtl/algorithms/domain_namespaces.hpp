// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file domain_namespaces.hpp
/// @brief Canonical algorithm domain namespaces (Phase 02)
/// @details Exposes explicit local/collective/distributed semantic entry points.

#pragma once

#include <utility>

namespace dtl::algorithms {

namespace local {
template <typename Container, typename T, typename BinaryOp>
auto reduce(const Container& container, T init, BinaryOp op) {
	return ::dtl::local_reduce(container, init, std::move(op));
}

template <typename Container, typename T>
auto count(const Container& container, const T& value) {
	return ::dtl::local_count(container, value);
}

template <typename Container, typename Predicate>
auto count_if(const Container& container, Predicate pred) {
	return ::dtl::local_count_if(container, std::move(pred));
}

template <typename Container, typename T>
auto find(Container& container, const T& value) {
	return ::dtl::local_find(container, value);
}

template <typename Container, typename Predicate>
auto find_if(Container& container, Predicate pred) {
	return ::dtl::local_find_if(container, std::move(pred));
}

template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
void inclusive_scan(const Container& input, OutputContainer& output, T init, BinaryOp binary_op) {
	::dtl::local_inclusive_scan(input, output, init, std::move(binary_op));
}

template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
void exclusive_scan(const Container& input, OutputContainer& output, T init, BinaryOp binary_op) {
	::dtl::local_exclusive_scan(input, output, init, std::move(binary_op));
}
}

namespace collective {
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp, typename Comm>
auto reduce(ExecutionPolicy&& policy,
			const Container& container,
			T init,
			BinaryOp op,
			Comm& comm) {
	return ::dtl::global_reduce(std::forward<ExecutionPolicy>(policy), container,
								init, std::move(op), comm);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
auto count(ExecutionPolicy&& policy,
		   const Container& container,
		   const T& value,
		   Comm& comm) {
	return ::dtl::global_count(std::forward<ExecutionPolicy>(policy), container, value, comm);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
auto find(ExecutionPolicy&& policy,
		  const Container& container,
		  const T& value,
		  Comm& comm) {
	return ::dtl::global_find(std::forward<ExecutionPolicy>(policy), container, value, comm);
}

template <typename ExecutionPolicy, typename Container, typename OutputContainer,
		  typename T, typename BinaryOp, typename Comm>
auto inclusive_scan(ExecutionPolicy&& policy,
					const Container& input,
					OutputContainer& output,
					T init,
					BinaryOp binary_op,
					Comm& comm) {
	return ::dtl::global_inclusive_scan(std::forward<ExecutionPolicy>(policy), input,
										output, init, std::move(binary_op), comm);
}

template <typename ExecutionPolicy, typename Container, typename OutputContainer,
		  typename T, typename BinaryOp, typename Comm>
auto exclusive_scan(ExecutionPolicy&& policy,
					const Container& input,
					OutputContainer& output,
					T init,
					BinaryOp binary_op,
					Comm& comm) {
	return ::dtl::global_exclusive_scan(std::forward<ExecutionPolicy>(policy), input,
										output, init, std::move(binary_op), comm);
}
}

namespace distributed {
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
auto reduce(ExecutionPolicy&& policy,
			const Container& container,
			T init,
			BinaryOp op) {
	return ::dtl::distributed_reduce(std::forward<ExecutionPolicy>(policy), container,
									 init, std::move(op));
}
}

}  // namespace dtl::algorithms
