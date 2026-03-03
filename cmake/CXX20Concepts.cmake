# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# C++20 concepts feature detection
# =============================================================================
# Verifies that the compiler supports C++20 concepts as required by DTL.
# Sets DTL_HAS_CONCEPTS cache variable on success.
# =============================================================================

include(CheckCXXSourceCompiles)

# Store original flags
set(_ORIGINAL_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")

# Ensure C++20 mode for the check
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++20")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /std:c++20")
endif()

# Test code that requires full C++20 concepts support
set(_DTL_CONCEPT_TEST_CODE "
#include <concepts>
#include <type_traits>

// Test __cpp_concepts feature macro
#if !defined(__cpp_concepts) || __cpp_concepts < 201907L
#error \"C++20 concepts not supported\"
#endif

// Test basic concept definition
template <typename T>
concept Integral = std::is_integral_v<T>;

// Test concept with requires expression
template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

// Test concept conjunction
template <typename T>
concept IntegralAddable = Integral<T> && Addable<T>;

// Test constrained template parameter
template <Integral T>
constexpr T add_one(T value) {
    return value + 1;
}

// Test requires clause
template <typename T>
    requires Addable<T>
constexpr T add_values(T a, T b) {
    return a + b;
}

// Test nested requires
template <typename T>
concept HasSize = requires(T t) {
    { t.size() } -> std::convertible_to<std::size_t>;
    requires std::is_class_v<T>;
};

int main() {
    static_assert(Integral<int>);
    static_assert(!Integral<float>);
    static_assert(Addable<int>);
    static_assert(IntegralAddable<int>);

    constexpr auto result = add_one(41);
    static_assert(result == 42);

    constexpr auto sum = add_values(1, 2);
    static_assert(sum == 3);

    return 0;
}
")

# Perform the check
check_cxx_source_compiles("${_DTL_CONCEPT_TEST_CODE}" DTL_CONCEPTS_COMPILE_CHECK)

# Restore original flags
set(CMAKE_REQUIRED_FLAGS "${_ORIGINAL_CMAKE_REQUIRED_FLAGS}")

# Set cache variable and report result
if(DTL_CONCEPTS_COMPILE_CHECK)
    set(DTL_HAS_CONCEPTS TRUE CACHE BOOL "Compiler supports C++20 concepts" FORCE)
    message(STATUS "C++20 concepts support: YES")
else()
    set(DTL_HAS_CONCEPTS FALSE CACHE BOOL "Compiler supports C++20 concepts" FORCE)

    # Provide helpful error message
    message(FATAL_ERROR
        "DTL requires a compiler with full C++20 concepts support.\n"
        "Please ensure:\n"
        "  - GCC 10+ (recommended: GCC 11+)\n"
        "  - Clang 12+ (recommended: Clang 14+)\n"
        "  - MSVC 19.29+ (Visual Studio 2019 16.10+)\n"
        "And that C++20 is enabled (-std=c++20 or /std:c++20).\n"
        "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

# Additional feature detection for optional C++20/23 features
# =============================================================================

# Check for std::expected (C++23, but available in some implementations)
set(_DTL_EXPECTED_TEST_CODE "
#if __has_include(<expected>)
#include <expected>
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
int main() {
    std::expected<int, int> e = 42;
    return e.value() == 42 ? 0 : 1;
}
#else
#error \"std::expected not available\"
#endif
#else
#error \"<expected> header not found\"
#endif
")

check_cxx_source_compiles("${_DTL_EXPECTED_TEST_CODE}" DTL_HAS_STD_EXPECTED)
if(DTL_HAS_STD_EXPECTED)
    message(STATUS "std::expected support: YES")
else()
    message(STATUS "std::expected support: NO (using dtl::result)")
endif()

# Check for std::ranges
set(_DTL_RANGES_TEST_CODE "
#include <ranges>
#include <vector>
int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto even = v | std::views::filter([](int n) { return n % 2 == 0; });
    return 0;
}
")

check_cxx_source_compiles("${_DTL_RANGES_TEST_CODE}" DTL_HAS_RANGES)
if(DTL_HAS_RANGES)
    message(STATUS "std::ranges support: YES")
else()
    message(STATUS "std::ranges support: NO")
endif()

# Check for std::span
set(_DTL_SPAN_TEST_CODE "
#include <span>
#include <vector>
int main() {
    std::vector<int> v = {1, 2, 3};
    std::span<int> s(v);
    return s.size() == 3 ? 0 : 1;
}
")

check_cxx_source_compiles("${_DTL_SPAN_TEST_CODE}" DTL_HAS_STD_SPAN)
if(DTL_HAS_STD_SPAN)
    message(STATUS "std::span support: YES")
else()
    message(STATUS "std::span support: NO (using dtl::span)")
endif()

# Clean up temporary variables
unset(_DTL_CONCEPT_TEST_CODE)
unset(_DTL_EXPECTED_TEST_CODE)
unset(_DTL_RANGES_TEST_CODE)
unset(_DTL_SPAN_TEST_CODE)
unset(_ORIGINAL_CMAKE_REQUIRED_FLAGS)
