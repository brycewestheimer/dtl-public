// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file affinity.hpp
/// @brief CPU affinity control
/// @details Provides functions for getting/setting CPU affinity for threads.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/topology/hardware.hpp>

#include <cstdint>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif

namespace dtl::topology {

// ============================================================================
// CPU Set
// ============================================================================

/// @brief Set of CPUs (bitmask wrapper)
class cpu_set {
public:
    /// @brief Maximum supported CPUs
    static constexpr size_type max_cpus = 1024;

    /// @brief Default constructor (empty set)
    cpu_set() : bits_(max_cpus / 64, 0) {}

    /// @brief Construct from a single CPU
    /// @param cpu CPU ID to include
    explicit cpu_set(std::uint32_t cpu)
        : bits_(max_cpus / 64, 0) {
        add(cpu);
    }

    /// @brief Construct from a range of CPUs
    /// @param first First CPU ID
    /// @param last Last CPU ID (inclusive)
    cpu_set(std::uint32_t first, std::uint32_t last)
        : bits_(max_cpus / 64, 0) {
        for (std::uint32_t i = first; i <= last && i < max_cpus; ++i) {
            add(i);
        }
    }

    /// @brief Construct from a vector of CPU IDs
    explicit cpu_set(const std::vector<std::uint32_t>& cpus)
        : bits_(max_cpus / 64, 0) {
        for (auto cpu : cpus) {
            add(cpu);
        }
    }

    // ========================================================================
    // Set Operations
    // ========================================================================

    /// @brief Add a CPU to the set
    void add(std::uint32_t cpu) {
        if (cpu < max_cpus) {
            bits_[cpu / 64] |= (1ULL << (cpu % 64));
        }
    }

    /// @brief Remove a CPU from the set
    void remove(std::uint32_t cpu) {
        if (cpu < max_cpus) {
            bits_[cpu / 64] &= ~(1ULL << (cpu % 64));
        }
    }

    /// @brief Clear all CPUs
    void clear() {
        std::fill(bits_.begin(), bits_.end(), 0);
    }

    /// @brief Check if CPU is in set
    [[nodiscard]] bool contains(std::uint32_t cpu) const noexcept {
        if (cpu >= max_cpus) return false;
        return (bits_[cpu / 64] & (1ULL << (cpu % 64))) != 0;
    }

    /// @brief Check if set is empty
    [[nodiscard]] bool empty() const noexcept {
        for (auto word : bits_) {
            if (word != 0) return false;
        }
        return true;
    }

    /// @brief Count CPUs in set
    [[nodiscard]] size_type count() const noexcept {
        size_type n = 0;
        for (auto word : bits_) {
            // Popcount
            while (word) {
                n += word & 1;
                word >>= 1;
            }
        }
        return n;
    }

    /// @brief Get first CPU in set
    /// @return First CPU ID, or max_cpus if empty
    [[nodiscard]] std::uint32_t first() const noexcept {
        for (size_type word_idx = 0; word_idx < bits_.size(); ++word_idx) {
            if (bits_[word_idx] != 0) {
                // Find first set bit
                std::uint64_t word = bits_[word_idx];
                std::uint32_t bit = 0;
                while (!(word & 1)) {
                    word >>= 1;
                    ++bit;
                }
                return static_cast<std::uint32_t>(word_idx * 64 + bit);
            }
        }
        return max_cpus;
    }

    /// @brief Get all CPUs as vector
    [[nodiscard]] std::vector<std::uint32_t> to_vector() const {
        std::vector<std::uint32_t> result;
        for (std::uint32_t i = 0; i < max_cpus; ++i) {
            if (contains(i)) {
                result.push_back(i);
            }
        }
        return result;
    }

    // ========================================================================
    // Set Algebra
    // ========================================================================

    /// @brief Union with another set
    cpu_set operator|(const cpu_set& other) const {
        cpu_set result;
        for (size_type i = 0; i < bits_.size(); ++i) {
            result.bits_[i] = bits_[i] | other.bits_[i];
        }
        return result;
    }

    /// @brief Intersection with another set
    cpu_set operator&(const cpu_set& other) const {
        cpu_set result;
        for (size_type i = 0; i < bits_.size(); ++i) {
            result.bits_[i] = bits_[i] & other.bits_[i];
        }
        return result;
    }

    /// @brief Difference (this - other)
    cpu_set operator-(const cpu_set& other) const {
        cpu_set result;
        for (size_type i = 0; i < bits_.size(); ++i) {
            result.bits_[i] = bits_[i] & ~other.bits_[i];
        }
        return result;
    }

    /// @brief Equality comparison
    bool operator==(const cpu_set& other) const noexcept {
        return bits_ == other.bits_;
    }

    bool operator!=(const cpu_set& other) const noexcept {
        return bits_ != other.bits_;
    }

#if defined(__linux__)
    /// @brief Convert to native cpu_set_t
    cpu_set_t to_native() const {
        cpu_set_t native;
        CPU_ZERO(&native);
        for (std::uint32_t i = 0; i < max_cpus && i < CPU_SETSIZE; ++i) {
            if (contains(i)) {
                CPU_SET(i, &native);
            }
        }
        return native;
    }

    /// @brief Create from native cpu_set_t
    static cpu_set from_native(const cpu_set_t& native) {
        cpu_set result;
        for (size_type i = 0;
             i < max_cpus && i < static_cast<size_type>(CPU_SETSIZE);
             ++i) {
            if (CPU_ISSET(i, &native)) {
                result.add(static_cast<std::uint32_t>(i));
            }
        }
        return result;
    }
#endif

private:
    std::vector<std::uint64_t> bits_;
};

// ============================================================================
// Affinity Functions
// ============================================================================

/// @brief Get current thread's CPU affinity
/// @return Current affinity set, or empty on error
[[nodiscard]] inline result<cpu_set> get_affinity() {
#if defined(__linux__)
    cpu_set_t native;
    CPU_ZERO(&native);

    if (sched_getaffinity(0, sizeof(native), &native) != 0) {
        return status(status_code::backend_error, no_rank, "Failed to get CPU affinity");
    }

    return cpu_set::from_native(native);
#else
    // Unsupported platform - return all CPUs
    cpu_set all;
    auto count = cpu_count();
    for (std::uint32_t i = 0; i < count; ++i) {
        all.add(i);
    }
    return all;
#endif
}

/// @brief Set current thread's CPU affinity
/// @param cpus CPU set to bind to
/// @return Success or error
inline result<void> set_affinity(const cpu_set& cpus) {
#if defined(__linux__)
    if (cpus.empty()) {
        return status(status_code::invalid_argument, no_rank, "Cannot set empty affinity");
    }

    cpu_set_t native = cpus.to_native();

    if (sched_setaffinity(0, sizeof(native), &native) != 0) {
        return status(status_code::backend_error, no_rank, "Failed to set CPU affinity");
    }

    return {};
#else
    (void)cpus;
    return status(status_code::not_supported, no_rank, "CPU affinity not supported");
#endif
}

/// @brief Bind current thread to a NUMA node
/// @param node NUMA node ID
/// @return Success or error
inline result<void> bind_to_numa_node(std::uint32_t node) {
    auto cpus_vec = cpus_on_numa_node(node);
    if (cpus_vec.empty()) {
        return status(status_code::invalid_argument, no_rank, "Invalid NUMA node");
    }

    cpu_set cpus(cpus_vec);
    return set_affinity(cpus);
}

/// @brief Bind current thread to a single CPU
/// @param cpu CPU ID
/// @return Success or error
inline result<void> bind_to_cpu(std::uint32_t cpu) {
    cpu_set cpus(cpu);
    return set_affinity(cpus);
}

// ============================================================================
// Scoped Affinity
// ============================================================================

/// @brief RAII guard that restores affinity on destruction
class scoped_affinity {
public:
    /// @brief Construct and set new affinity
    /// @param cpus New affinity to set
    explicit scoped_affinity(const cpu_set& cpus) {
        auto orig = get_affinity();
        if (orig) {
            original_ = std::move(*orig);
            saved_ = true;
        }
        set_affinity(cpus);  // Ignore errors
    }

    /// @brief Construct and bind to a single CPU
    explicit scoped_affinity(std::uint32_t cpu)
        : scoped_affinity(cpu_set(cpu)) {}

    /// @brief Destructor - restores original affinity
    ~scoped_affinity() {
        if (saved_) {
            set_affinity(original_);  // Ignore errors
        }
    }

    // Non-copyable
    scoped_affinity(const scoped_affinity&) = delete;
    scoped_affinity& operator=(const scoped_affinity&) = delete;

    // Movable
    scoped_affinity(scoped_affinity&& other) noexcept
        : original_(std::move(other.original_))
        , saved_(other.saved_) {
        other.saved_ = false;
    }

    scoped_affinity& operator=(scoped_affinity&& other) noexcept {
        if (this != &other) {
            if (saved_) {
                set_affinity(original_);
            }
            original_ = std::move(other.original_);
            saved_ = other.saved_;
            other.saved_ = false;
        }
        return *this;
    }

    /// @brief Get the saved original affinity
    [[nodiscard]] const cpu_set& original() const noexcept {
        return original_;
    }

private:
    cpu_set original_;
    bool saved_ = false;
};

}  // namespace dtl::topology
