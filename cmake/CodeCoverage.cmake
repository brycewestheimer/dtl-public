# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Code Coverage Configuration
# =============================================================================
# Provides lcov/genhtml integration for C++ code coverage reports.
#
# Usage:
#   cmake -DDTL_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
#   make
#   make coverage        # Run tests and generate HTML report
#   make coverage-clean  # Reset coverage counters
#
# Output: ${CMAKE_BINARY_DIR}/coverage_report/index.html
# =============================================================================

# Only enable coverage if explicitly requested and in Debug mode
if(NOT DTL_ENABLE_COVERAGE)
    return()
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Code coverage requires CMAKE_BUILD_TYPE=Debug. Coverage disabled.")
    return()
endif()

# Find required tools
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)

if(NOT LCOV_PATH)
    message(WARNING "lcov not found. Coverage targets will not be available.")
    message(WARNING "Install with: sudo apt install lcov")
    return()
endif()

if(NOT GENHTML_PATH)
    message(WARNING "genhtml not found. Coverage targets will not be available.")
    message(WARNING "Install with: sudo apt install lcov")
    return()
endif()

message(STATUS "Code coverage enabled")
message(STATUS "  lcov: ${LCOV_PATH}")
message(STATUS "  genhtml: ${GENHTML_PATH}")

# -----------------------------------------------------------------------------
# Function: dtl_add_coverage_flags
# -----------------------------------------------------------------------------
# Adds coverage compiler and linker flags to a target.
#
# Usage:
#   dtl_add_coverage_flags(my_target)
# -----------------------------------------------------------------------------
function(dtl_add_coverage_flags target)
    target_compile_options(${target} PRIVATE
        --coverage
        -fprofile-arcs
        -ftest-coverage
        -O0  # Disable optimization for accurate coverage
    )
    target_link_options(${target} PRIVATE
        --coverage
        -fprofile-arcs
        -ftest-coverage
    )
endfunction()

# -----------------------------------------------------------------------------
# Function: dtl_setup_coverage_target
# -----------------------------------------------------------------------------
# Creates 'coverage' and 'coverage-clean' targets.
#
# Usage:
#   dtl_setup_coverage_target(
#       TARGET my_test_runner
#       OUTPUT_DIR coverage_report
#       EXCLUDE_PATTERNS "/usr/*" "*/tests/*"
#   )
# -----------------------------------------------------------------------------
function(dtl_setup_coverage_target)
    cmake_parse_arguments(
        COV
        ""
        "TARGET;OUTPUT_DIR"
        "EXCLUDE_PATTERNS"
        ${ARGN}
    )

    # Default values
    if(NOT COV_TARGET)
        set(COV_TARGET "dtl_unit_tests")
    endif()

    if(NOT COV_OUTPUT_DIR)
        set(COV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/coverage_report")
    endif()

    if(NOT COV_EXCLUDE_PATTERNS)
        set(COV_EXCLUDE_PATTERNS
            "/usr/*"
            "*/tests/*"
            "*/test/*"
            "*/_deps/*"
            "*/googletest/*"
            "*/googlemock/*"
            "*/benchmark/*"
        )
    endif()

    # Coverage data files
    set(COVERAGE_INFO "${CMAKE_BINARY_DIR}/coverage.info")
    set(COVERAGE_INFO_CLEAN "${CMAKE_BINARY_DIR}/coverage_clean.info")

    # Build exclude arguments for lcov
    set(LCOV_EXCLUDES "")
    foreach(pattern ${COV_EXCLUDE_PATTERNS})
        list(APPEND LCOV_EXCLUDES "--exclude" "${pattern}")
    endforeach()

    # -----------------------------------------------------------------------------
    # Target: coverage
    # -----------------------------------------------------------------------------
    # Runs tests and generates HTML coverage report.
    # -----------------------------------------------------------------------------
    add_custom_target(coverage
        COMMENT "Generating code coverage report..."

        # Step 1: Reset coverage counters (baseline)
        COMMAND ${LCOV_PATH}
            --directory ${CMAKE_BINARY_DIR}
            --zerocounters
            --quiet

        # Step 2: Capture baseline (before tests)
        COMMAND ${LCOV_PATH}
            --directory ${CMAKE_BINARY_DIR}
            --capture
            --initial
            --output-file ${COVERAGE_INFO}.base
            --quiet
            --ignore-errors mismatch

        # Step 3: Run tests
        COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}

        # Step 4: Capture coverage data (after tests)
        COMMAND ${LCOV_PATH}
            --directory ${CMAKE_BINARY_DIR}
            --capture
            --output-file ${COVERAGE_INFO}.test
            --quiet
            --ignore-errors mismatch

        # Step 5: Combine baseline and test coverage
        COMMAND ${LCOV_PATH}
            --add-tracefile ${COVERAGE_INFO}.base
            --add-tracefile ${COVERAGE_INFO}.test
            --output-file ${COVERAGE_INFO}.combined
            --quiet
            --ignore-errors mismatch

        # Step 6: Filter out excluded patterns
        COMMAND ${LCOV_PATH}
            --remove ${COVERAGE_INFO}.combined
            ${LCOV_EXCLUDES}
            --output-file ${COVERAGE_INFO}
            --quiet
            --ignore-errors mismatch

        # Step 7: Generate HTML report
        COMMAND ${GENHTML_PATH}
            ${COVERAGE_INFO}
            --output-directory ${COV_OUTPUT_DIR}
            --title "DTL Code Coverage"
            --legend
            --show-details
            --quiet

        # Step 8: Cleanup intermediate files
        COMMAND ${CMAKE_COMMAND} -E remove
            ${COVERAGE_INFO}.base
            ${COVERAGE_INFO}.test
            ${COVERAGE_INFO}.combined

        # Step 9: Print summary
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated: ${COV_OUTPUT_DIR}/index.html"
        COMMAND ${CMAKE_COMMAND} -E echo ""

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${COV_TARGET}
        VERBATIM
    )

    # -----------------------------------------------------------------------------
    # Target: coverage-clean
    # -----------------------------------------------------------------------------
    # Resets all coverage counters and removes generated files.
    # -----------------------------------------------------------------------------
    add_custom_target(coverage-clean
        COMMENT "Cleaning coverage data..."

        # Reset counters
        COMMAND ${LCOV_PATH}
            --directory ${CMAKE_BINARY_DIR}
            --zerocounters
            --quiet

        # Remove coverage report directory
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${COV_OUTPUT_DIR}

        # Remove coverage data files
        COMMAND ${CMAKE_COMMAND} -E remove -f
            ${COVERAGE_INFO}
            ${COVERAGE_INFO}.base
            ${COVERAGE_INFO}.test
            ${COVERAGE_INFO}.combined

        # Remove .gcda files
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -type f -delete 2>/dev/null || true

        COMMAND ${CMAKE_COMMAND} -E echo "Coverage data cleaned."

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        VERBATIM
    )

    # -----------------------------------------------------------------------------
    # Target: coverage-summary
    # -----------------------------------------------------------------------------
    # Shows coverage summary without generating HTML.
    # -----------------------------------------------------------------------------
    add_custom_target(coverage-summary
        COMMENT "Showing coverage summary..."

        COMMAND ${LCOV_PATH}
            --directory ${CMAKE_BINARY_DIR}
            --capture
            --output-file ${COVERAGE_INFO}.summary
            --quiet
            --ignore-errors mismatch

        COMMAND ${LCOV_PATH}
            --remove ${COVERAGE_INFO}.summary
            ${LCOV_EXCLUDES}
            --output-file ${COVERAGE_INFO}.filtered
            --quiet
            --ignore-errors mismatch

        COMMAND ${LCOV_PATH}
            --summary ${COVERAGE_INFO}.filtered

        COMMAND ${CMAKE_COMMAND} -E remove
            ${COVERAGE_INFO}.summary
            ${COVERAGE_INFO}.filtered

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        VERBATIM
    )

endfunction()

# -----------------------------------------------------------------------------
# Auto-setup if tests are being built
# -----------------------------------------------------------------------------
# Call dtl_setup_coverage_target() in your main CMakeLists.txt after defining
# your test target, or it will be called automatically with defaults.
# -----------------------------------------------------------------------------
