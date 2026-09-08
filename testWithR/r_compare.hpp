/**
 * @file r_compare.hpp
 * @brief Assertion helpers for comparing statcpp results against R reference values.
 *
 * The reference values in `r_reference_*.hpp` are emitted by
 * `generate_r_reference.R` as C++17 hexadecimal floating-point literals, so they
 * are bit-identical to the values R produced. The remaining difference is the
 * algorithmic difference between statcpp and R, which is what these helpers
 * measure.
 *
 * Comparison uses a combined absolute and relative tolerance:
 *
 *     |actual - expected| <= atol + rtol * |expected|
 *
 * A pure relative tolerance is not usable here. For example
 * `pnorm(37.10, log.p = TRUE)` is about -1.4e-301, so an absolute difference of
 * 3e-308 (entirely negligible) shows up as a relative error of 2.3e-07. The
 * absolute term absorbs those cases.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace statcpp_test {

/// @brief Positive infinity, used by generated reference headers.
inline constexpr double kInf = std::numeric_limits<double>::infinity();

/// @brief Quiet NaN, used by generated reference headers.
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief Compare one value against an R reference value.
 *
 * NaN matches NaN, and infinities match infinities of the same sign, so that
 * degenerate results (for example the infinite degrees of freedom reported by a
 * z-test) can be checked without special casing at every call site.
 *
 * @param actual Value produced by statcpp
 * @param expected Reference value produced by R
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @return Assertion result carrying a diagnostic message on failure
 */
[[nodiscard]] inline ::testing::AssertionResult RClose(double actual, double expected, double rtol,
                                                       double atol)
{
    const bool actual_nan = std::isnan(actual);
    const bool expected_nan = std::isnan(expected);
    if (actual_nan || expected_nan) {
        if (actual_nan && expected_nan) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
               << "NaN mismatch: statcpp=" << actual << " R=" << expected;
    }

    const bool actual_inf = std::isinf(actual);
    const bool expected_inf = std::isinf(expected);
    if (actual_inf || expected_inf) {
        if (actual_inf && expected_inf && ((actual > 0.0) == (expected > 0.0))) {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure()
               << "infinity mismatch: statcpp=" << actual << " R=" << expected;
    }

    const double diff = std::abs(actual - expected);
    const double limit = atol + rtol * std::abs(expected);
    if (diff <= limit) return ::testing::AssertionSuccess();

    const double relative = (expected == 0.0) ? diff : diff / std::abs(expected);
    return ::testing::AssertionFailure()
           << "\n    statcpp = " << ::testing::PrintToString(actual)
           << "\n    R       = " << ::testing::PrintToString(expected)
           << "\n    diff    = " << diff << " (relative " << relative << ")"
           << "\n    allowed = " << limit << " (rtol " << rtol << ", atol " << atol << ")";
}

/**
 * @brief Compare a sequence of values against R reference values element by element.
 *
 * @param actual Values produced by statcpp
 * @param expected Reference values produced by R
 * @param count Number of reference values
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @return Assertion result carrying a diagnostic message on failure
 */
[[nodiscard]] inline ::testing::AssertionResult RCloseRange(const std::vector<double>& actual,
                                                            const double* expected,
                                                            std::size_t count, double rtol,
                                                            double atol)
{
    if (actual.size() != count) {
        return ::testing::AssertionFailure()
               << "size mismatch: statcpp=" << actual.size() << " R=" << count;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto result = RClose(actual[i], expected[i], rtol, atol);
        if (!result) return ::testing::AssertionFailure() << "at index " << i << ":" << result.message();
    }
    return ::testing::AssertionSuccess();
}

/**
 * @brief Rebuild a vector of groups from the flattened form used by the generated headers.
 *
 * Group data is emitted as one flat array plus an array of group sizes, because
 * a jagged array cannot be expressed as a single `constexpr` initialiser.
 *
 * @param flat Concatenated group values
 * @param sizes Number of values in each group
 * @param n_groups Number of groups
 * @return Groups in the nested form the statcpp API expects
 */
[[nodiscard]] inline std::vector<std::vector<double>> MakeGroups(const double* flat,
                                                                 const std::size_t* sizes,
                                                                 std::size_t n_groups)
{
    std::vector<std::vector<double>> groups(n_groups);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n_groups; ++i) {
        groups[i].assign(flat + offset, flat + offset + sizes[i]);
        offset += sizes[i];
    }
    return groups;
}

/**
 * @brief Rebuild a row-major matrix from the flattened form used by the generated headers.
 *
 * @param flat Matrix values in row-major order
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix in the nested form the statcpp API expects
 */
[[nodiscard]] inline std::vector<std::vector<double>> MakeMatrix(const double* flat, std::size_t rows,
                                                                 std::size_t cols)
{
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            matrix[r][c] = flat[r * cols + c];
        }
    }
    return matrix;
}

}  // namespace statcpp_test

/**
 * @brief Assert that a statcpp value matches the reference field of a generated case struct.
 *
 * Uses the per-case tolerances stored in the struct so that a case which needs a
 * looser bound records the reason next to the value rather than at the call site.
 */
#define EXPECT_R_FIELD(actual, Case, Field)                                                     \
    EXPECT_TRUE(::statcpp_test::RClose((actual), Case::Field, Case::kRtol, Case::kAtol))        \
        << "  reference: " #Case "::" #Field

/// @brief Assert that a statcpp sequence matches a reference array of a generated case struct.
#define EXPECT_R_RANGE(actual, Case, Field)                                                     \
    EXPECT_TRUE(::statcpp_test::RCloseRange((actual), Case::Field, std::size(Case::Field),      \
                                            Case::kRtol, Case::kAtol))                          \
        << "  reference: " #Case "::" #Field

/**
 * @brief Assert a p-value, or any other quantity obtained through a distribution function.
 *
 * statcpp implements its own incomplete beta and incomplete gamma functions, so
 * values that pass through them agree with R to roughly 1e-9 relative rather
 * than to the 1e-12 that plain arithmetic reaches. `kPRtol` records that.
 */
#define EXPECT_R_PVALUE(actual, Case, Field)                                                    \
    EXPECT_TRUE(::statcpp_test::RClose((actual), Case::Field, Case::kPRtol, Case::kAtol))       \
        << "  reference: " #Case "::" #Field

/// @brief Assert a sequence of p-values against a reference array.
#define EXPECT_R_PVALUE_RANGE(actual, Case, Field)                                              \
    EXPECT_TRUE(::statcpp_test::RCloseRange((actual), Case::Field, std::size(Case::Field),      \
                                            Case::kPRtol, Case::kAtol))                         \
        << "  reference: " #Case "::" #Field
