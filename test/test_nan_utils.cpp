#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "statcpp/nan_utils.hpp"

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

/// @brief Projection that reads the first member of a pair
struct First {
    double operator()(const std::pair<double, int>& p) const { return p.first; }
};

/// @brief The message of the std::invalid_argument thrown by fn, or "" if none is thrown
template <typename Fn>
std::string invalid_argument_message(Fn fn)
{
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// ============================================================================
// is_nan_value Tests
// ============================================================================

/**
 * @brief Tests NaN detection on floating-point values
 * @test Verifies that NaN is detected for double and float, and that finite and infinite values are not NaN
 */
TEST(NanUtilsTest, IsNanValueFloatingPoint) {
    EXPECT_TRUE(statcpp::detail::is_nan_value(kNaN));
    EXPECT_TRUE(statcpp::detail::is_nan_value(std::numeric_limits<float>::quiet_NaN()));
    EXPECT_FALSE(statcpp::detail::is_nan_value(1.5));
    EXPECT_FALSE(statcpp::detail::is_nan_value(std::numeric_limits<double>::infinity()));
}

/**
 * @brief Tests NaN detection on non-floating-point values
 * @test Verifies that integer values are never NaN, so generic templates can use the same check
 */
TEST(NanUtilsTest, IsNanValueIntegral) {
    EXPECT_FALSE(statcpp::detail::is_nan_value(0));
    EXPECT_FALSE(statcpp::detail::is_nan_value(42L));
}

// ============================================================================
// has_nan Tests
// ============================================================================

/**
 * @brief Tests NaN detection in a range for every NaN position
 * @test Verifies that NaN is found first, in the middle, last and when every element is NaN
 */
TEST(NanUtilsTest, HasNanPositions) {
    const std::vector<std::vector<double>> with_nan = {
        {kNaN, 1.0, 2.0}, {1.0, kNaN, 2.0}, {1.0, 2.0, kNaN}, {kNaN, kNaN}};
    for (const auto& v : with_nan) {
        EXPECT_TRUE(statcpp::detail::has_nan(v.begin(), v.end()));
    }
}

/**
 * @brief Tests NaN detection on ranges without NaN
 * @test Verifies that clean, empty and integer ranges report no NaN
 */
TEST(NanUtilsTest, HasNanNone) {
    const std::vector<double> clean = {1.0, 2.0, std::numeric_limits<double>::infinity()};
    const std::vector<double> empty;
    const std::vector<int> integers = {1, 2, 3};
    EXPECT_FALSE(statcpp::detail::has_nan(clean.begin(), clean.end()));
    EXPECT_FALSE(statcpp::detail::has_nan(empty.begin(), empty.end()));
    EXPECT_FALSE(statcpp::detail::has_nan(integers.begin(), integers.end()));
}

/**
 * @brief Tests NaN detection after projection
 * @test Verifies that the projected value, not the element, is tested
 */
TEST(NanUtilsTest, HasNanProjection) {
    const std::vector<std::pair<double, int>> clean = {{1.0, 0}, {2.0, 0}};
    const std::vector<std::pair<double, int>> with_nan = {{1.0, 0}, {kNaN, 0}};
    EXPECT_FALSE(statcpp::detail::has_nan(clean.begin(), clean.end(), First{}));
    EXPECT_TRUE(statcpp::detail::has_nan(with_nan.begin(), with_nan.end(), First{}));
}

// ============================================================================
// drop_nan Tests
// ============================================================================

/**
 * @brief Tests NaN removal
 * @test Verifies that NaN is removed from every position and that the order of the rest is kept
 */
TEST(NanUtilsTest, DropNanKeepsOrder) {
    const std::vector<double> data = {kNaN, 3.0, kNaN, 1.0, 2.0, kNaN};
    const auto result = statcpp::detail::drop_nan(data.begin(), data.end());
    EXPECT_EQ(result, (std::vector<double>{3.0, 1.0, 2.0}));
}

/**
 * @brief Tests NaN removal on edge cases
 * @test Verifies that empty and all-NaN ranges give an empty result and integer ranges are unchanged
 */
TEST(NanUtilsTest, DropNanEdgeCases) {
    const std::vector<double> empty;
    const std::vector<double> all_nan = {kNaN, kNaN};
    const std::vector<int> integers = {3, 1, 2};
    EXPECT_TRUE(statcpp::detail::drop_nan(empty.begin(), empty.end()).empty());
    EXPECT_TRUE(statcpp::detail::drop_nan(all_nan.begin(), all_nan.end()).empty());
    EXPECT_EQ(statcpp::detail::drop_nan(integers.begin(), integers.end()), integers);
}

/**
 * @brief Tests NaN removal after projection
 * @test Verifies that the projected values that are not NaN are returned
 */
TEST(NanUtilsTest, DropNanProjection) {
    const std::vector<std::pair<double, int>> data = {{1.0, 0}, {kNaN, 0}, {3.0, 0}};
    const auto result = statcpp::detail::drop_nan(data.begin(), data.end(), First{});
    EXPECT_EQ(result, (std::vector<double>{1.0, 3.0}));
}

// ============================================================================
// drop_nan_pairs Tests
// ============================================================================

/**
 * @brief Tests removal of incomplete pairs
 * @test Verifies that a pair is dropped when either side is NaN, as R's complete.cases does
 */
TEST(NanUtilsTest, DropNanPairs) {
    const std::vector<double> x = {1.0, kNaN, 3.0, 4.0, kNaN};
    const std::vector<double> y = {10.0, 20.0, kNaN, 40.0, kNaN};
    const auto [xs, ys] = statcpp::detail::drop_nan_pairs(x.begin(), x.end(), y.begin());
    EXPECT_EQ(xs, (std::vector<double>{1.0, 4.0}));
    EXPECT_EQ(ys, (std::vector<double>{10.0, 40.0}));
}

/**
 * @brief Tests removal of incomplete pairs on an empty range
 * @test Verifies that empty input gives two empty vectors
 */
TEST(NanUtilsTest, DropNanPairsEmpty) {
    const std::vector<double> x;
    const std::vector<double> y;
    const auto [xs, ys] = statcpp::detail::drop_nan_pairs(x.begin(), x.end(), y.begin());
    EXPECT_TRUE(xs.empty());
    EXPECT_TRUE(ys.empty());
}

// ============================================================================
// nan_last_less / nan_last_greater Tests
// ============================================================================

/**
 * @brief Tests the strict weak ordering requirements with NaN
 * @test Verifies irreflexivity, asymmetry and that NaN values are equivalent to each other
 */
TEST(NanUtilsTest, NanLastLessIsStrictWeakOrdering) {
    const statcpp::detail::nan_last_less less;
    const std::vector<double> values = {kNaN, -1.0, 0.0, 2.5, std::numeric_limits<double>::infinity()};
    for (double a : values) {
        EXPECT_FALSE(less(a, a));  // irreflexive
        for (double b : values) {
            EXPECT_FALSE(less(a, b) && less(b, a));  // asymmetric
            for (double c : values) {
                if (less(a, b) && less(b, c)) {
                    EXPECT_TRUE(less(a, c));  // transitive
                }
            }
        }
    }
    EXPECT_FALSE(less(kNaN, kNaN));
    EXPECT_TRUE(less(std::numeric_limits<double>::infinity(), kNaN));
    EXPECT_FALSE(less(kNaN, -1.0));
}

/**
 * @brief Tests ascending sort with NaN
 * @test Verifies that non-NaN values are sorted ascending and every NaN is placed last
 */
TEST(NanUtilsTest, NanLastLessSort) {
    std::vector<double> data = {3.0, kNaN, 1.0, kNaN, 2.0};
    std::sort(data.begin(), data.end(), statcpp::detail::nan_last_less{});
    EXPECT_DOUBLE_EQ(data[0], 1.0);
    EXPECT_DOUBLE_EQ(data[1], 2.0);
    EXPECT_DOUBLE_EQ(data[2], 3.0);
    EXPECT_TRUE(std::isnan(data[3]));
    EXPECT_TRUE(std::isnan(data[4]));
}

/**
 * @brief Tests descending sort with NaN
 * @test Verifies that non-NaN values are sorted descending and NaN is still placed last
 */
TEST(NanUtilsTest, NanLastGreaterSort) {
    std::vector<double> data = {kNaN, 1.0, 3.0, kNaN, 2.0};
    std::sort(data.begin(), data.end(), statcpp::detail::nan_last_greater{});
    EXPECT_DOUBLE_EQ(data[0], 3.0);
    EXPECT_DOUBLE_EQ(data[1], 2.0);
    EXPECT_DOUBLE_EQ(data[2], 1.0);
    EXPECT_TRUE(std::isnan(data[3]));
    EXPECT_TRUE(std::isnan(data[4]));
}

/**
 * @brief Tests that a stable sort keeps NaN in their original order
 * @test Verifies the index order R's order(na.last = TRUE) produces for c(3, NA, 1, NA, 2)
 */
TEST(NanUtilsTest, NanLastStableIndexOrder) {
    const std::vector<double> data = {3.0, kNaN, 1.0, kNaN, 2.0};
    std::vector<std::size_t> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);
    const statcpp::detail::nan_last_less less;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t i, std::size_t j) { return less(data[i], data[j]); });
    // R: order(c(3, NA, 1, NA, 2)) == c(3, 5, 1, 2, 4) (1-based)
    EXPECT_EQ(idx, (std::vector<std::size_t>{2, 4, 0, 1, 3}));
}

// ============================================================================
// require_no_nan / require_param_not_nan Tests
// ============================================================================

/**
 * @brief Tests rejection of data containing NaN
 * @test Verifies the uniform message and that clean data is accepted
 */
TEST(NanUtilsTest, RequireNoNan) {
    const std::vector<double> with_nan = {1.0, kNaN};
    const std::vector<double> clean = {1.0, 2.0};
    EXPECT_EQ(invalid_argument_message([&] {
                  statcpp::detail::require_no_nan(with_nan.begin(), with_nan.end(), "value_counts");
              }),
              "statcpp::value_counts: data contains NaN");
    EXPECT_NO_THROW(statcpp::detail::require_no_nan(clean.begin(), clean.end(), "value_counts"));
}

/**
 * @brief Tests rejection of a NaN parameter
 * @test Verifies the uniform message and that finite and infinite parameters are accepted
 */
TEST(NanUtilsTest, RequireParamNotNan) {
    EXPECT_EQ(invalid_argument_message([] { statcpp::detail::require_param_not_nan(kNaN, "ci_mean", "confidence"); }),
              "statcpp::ci_mean: confidence is NaN");
    EXPECT_NO_THROW(statcpp::detail::require_param_not_nan(0.95, "ci_mean", "confidence"));
    EXPECT_NO_THROW(
        statcpp::detail::require_param_not_nan(std::numeric_limits<double>::infinity(), "ci_mean", "confidence"));
}

// ============================================================================
// map_non_nan Tests
// ============================================================================

/**
 * @brief Tests that map_non_nan transforms only the non-NaN values
 * @test Verifies the transform sees the non-NaN values in order and NaN positions stay NaN
 */
TEST(NanUtilsTest, MapNonNan) {
    const std::vector<double> values = {0.01, kNaN, 0.04, kNaN};
    std::size_t seen = 0;
    const auto result = statcpp::detail::map_non_nan(values, [&](const std::vector<double>& v) {
        seen = v.size();
        std::vector<double> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            out[i] = v[i] * static_cast<double>(v.size());
        }
        return out;
    });
    EXPECT_EQ(seen, 2u);
    ASSERT_EQ(result.size(), 4u);
    EXPECT_DOUBLE_EQ(result[0], 0.02);
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_DOUBLE_EQ(result[2], 0.08);
    EXPECT_TRUE(std::isnan(result[3]));
}

/**
 * @brief Tests map_non_nan when every value is NaN
 * @test Verifies the transform is not called and the result is all NaN
 */
TEST(NanUtilsTest, MapNonNanAllNaN) {
    bool called = false;
    const auto result = statcpp::detail::map_non_nan({kNaN, kNaN}, [&](const std::vector<double>& v) {
        called = true;
        return v;
    });
    EXPECT_FALSE(called);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
}

/**
 * @brief Tests the matrix NaN detection helpers
 * @test Verifies has_nan_matrix and has_nan_rows find NaN in X or y
 */
TEST(NanUtilsTest, MatrixDetection) {
    const std::vector<std::vector<double>> clean = {{1.0, 2.0}, {3.0, 4.0}};
    const std::vector<std::vector<double>> dirty = {{1.0, 2.0}, {3.0, kNaN}};
    EXPECT_FALSE(statcpp::detail::has_nan_matrix(clean));
    EXPECT_TRUE(statcpp::detail::has_nan_matrix(dirty));
    EXPECT_FALSE(statcpp::detail::has_nan_rows(clean, {1.0, 2.0}));
    EXPECT_TRUE(statcpp::detail::has_nan_rows(clean, {1.0, kNaN}));
    EXPECT_TRUE(statcpp::detail::has_nan_rows(dirty, {1.0, 2.0}));
    EXPECT_NO_THROW(statcpp::detail::require_no_nan_matrix(clean, "f"));
    EXPECT_THROW(statcpp::detail::require_no_nan_matrix(dirty, "f"), std::invalid_argument);
}

/**
 * @brief Tests drop_nan_rows
 * @test Verifies an observation is dropped when its response or any predictor is NaN, keeping the order
 */
TEST(NanUtilsTest, DropNanRows) {
    const std::vector<std::vector<double>> X = {{1.0, 2.0}, {kNaN, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
    const std::vector<double> y = {10.0, 20.0, kNaN, 40.0};
    const auto [Xc, yc] = statcpp::detail::drop_nan_rows(X, y);
    ASSERT_EQ(Xc.size(), 2u);
    EXPECT_EQ(Xc[0], (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(Xc[1], (std::vector<double>{7.0, 8.0}));
    EXPECT_EQ(yc, (std::vector<double>{10.0, 40.0}));

    const auto rows = statcpp::detail::drop_nan_rows(X);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[1], (std::vector<double>{5.0, 6.0}));
}
