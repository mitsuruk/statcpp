#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "statcpp/time_series.hpp"

// ============================================================================
// Autocorrelation Tests
// ============================================================================

/**
 * @brief Tests autocorrelation at lag 0
 * @test Verifies that autocorrelation at lag 0 is always exactly 1.0
 */
TEST(AutocorrelationTest, Lag0) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(statcpp::autocorrelation(data.begin(), data.end(), 0), 1.0);
}

/**
 * @brief Tests autocorrelation at lag 1 for trending data
 * @test Verifies that trending data exhibits strong positive autocorrelation at lag 1
 */
TEST(AutocorrelationTest, Lag1Trend) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double acf1 = statcpp::autocorrelation(data.begin(), data.end(), 1);
    // Strong positive autocorrelation for trending data
    EXPECT_GE(acf1, 0.7);
}

/**
 * @brief Tests autocorrelation function for multiple lags
 * @test Verifies that ACF returns correct number of values with lag 0 equal to 1
 */
TEST(AutocorrelationTest, ACFFunction) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto result = statcpp::acf(data.begin(), data.end(), 5);

    EXPECT_EQ(result.size(), 6);  // lag 0 to 5
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // lag 0 always 1
}

/**
 * @brief Tests error handling for empty data in autocorrelation
 * @test Verifies that autocorrelation throws exception for empty input
 */
TEST(AutocorrelationTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::autocorrelation(data.begin(), data.end(), 0), std::invalid_argument);
}

// ============================================================================
// PACF Tests
// ============================================================================

/**
 * @brief Tests basic partial autocorrelation function computation
 * @test Verifies that PACF returns correct number of values with lag 0 equal to 1
 */
TEST(PACFTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto result = statcpp::pacf(data.begin(), data.end(), 3);

    EXPECT_EQ(result.size(), 4);  // lag 0 to 3
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // PACF(0) always 1
}

// ============================================================================
// Forecast Error Metrics Tests
// ============================================================================

/**
 * @brief Tests Mean Absolute Error calculation
 * @test Verifies that MAE is correctly computed as average of absolute prediction errors
 */
TEST(MAETest, Basic) {
    std::vector<double> actual = {1, 2, 3, 4, 5};
    std::vector<double> predicted = {1, 2, 3, 4, 6};

    double result = statcpp::mae(actual.begin(), actual.end(), predicted.begin());
    EXPECT_DOUBLE_EQ(result, 0.2);  // |5-6| / 5 = 0.2
}

/**
 * @brief Tests Mean Squared Error calculation
 * @test Verifies that MSE is correctly computed as average of squared prediction errors
 */
TEST(MSETest, Basic) {
    std::vector<double> actual = {1, 2, 3, 4, 5};
    std::vector<double> predicted = {1, 2, 3, 4, 6};

    double result = statcpp::mse(actual.begin(), actual.end(), predicted.begin());
    EXPECT_DOUBLE_EQ(result, 0.2);  // (5-6)^2 / 5 = 0.2
}

/**
 * @brief Tests Root Mean Squared Error calculation
 * @test Verifies that RMSE is correctly computed as square root of MSE
 */
TEST(RMSETest, Basic) {
    std::vector<double> actual = {1, 2, 3, 4, 5};
    std::vector<double> predicted = {1, 2, 3, 4, 6};

    double result = statcpp::rmse(actual.begin(), actual.end(), predicted.begin());
    EXPECT_NEAR(result, std::sqrt(0.2), 1e-10);
}

/**
 * @brief Tests Mean Absolute Percentage Error calculation
 * @test Verifies that MAPE is correctly computed as average of percentage errors
 */
TEST(MAPETest, Basic) {
    std::vector<double> actual = {100, 100, 100, 100, 100};
    std::vector<double> predicted = {90, 95, 100, 105, 110};

    double result = statcpp::mape(actual.begin(), actual.end(), predicted.begin());
    EXPECT_NEAR(result, 6.0, 1e-10);  // Average of 10%, 5%, 0%, 5%, 10% = 6%
}

// ============================================================================
// Moving Average Tests
// ============================================================================

/**
 * @brief Tests simple moving average calculation
 * @test Verifies that moving average produces correct window-based averages with expected output size
 */
TEST(MovingAverageTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = statcpp::moving_average(data.begin(), data.end(), 3);

    EXPECT_EQ(result.size(), 3);  // 5 - 3 + 1 = 3
    EXPECT_DOUBLE_EQ(result[0], 2.0);  // (1+2+3)/3
    EXPECT_DOUBLE_EQ(result[1], 3.0);  // (2+3+4)/3
    EXPECT_DOUBLE_EQ(result[2], 4.0);  // (3+4+5)/3
}

/**
 * @brief Tests exponential moving average calculation
 * @test Verifies that exponential moving average produces output with correct size and initial value
 */
TEST(ExponentialMovingAverageTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = statcpp::exponential_moving_average(data.begin(), data.end(), 0.5);

    EXPECT_EQ(result.size(), 5);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
}

// ============================================================================
// Differencing Tests
// ============================================================================

/**
 * @brief Tests first-order differencing
 * @test Verifies that first-order differencing correctly computes consecutive differences
 */
TEST(DiffTest, FirstOrder) {
    std::vector<double> data = {1, 2, 4, 7, 11};
    auto result = statcpp::diff(data.begin(), data.end(), 1);

    EXPECT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
    EXPECT_DOUBLE_EQ(result[3], 4.0);
}

/**
 * @brief Tests second-order differencing
 * @test Verifies that second-order differencing correctly applies differencing twice
 */
TEST(DiffTest, SecondOrder) {
    std::vector<double> data = {1, 2, 4, 7, 11};
    auto result = statcpp::diff(data.begin(), data.end(), 2);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
}

/**
 * @brief Tests seasonal differencing
 * @test Verifies that seasonal differencing correctly computes differences at seasonal lag
 */
TEST(SeasonalDiffTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 2, 4, 6, 8};
    auto result = statcpp::seasonal_diff(data.begin(), data.end(), 4);

    EXPECT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // 2 - 1
    EXPECT_DOUBLE_EQ(result[1], 2.0);  // 4 - 2
}

// ============================================================================
// Lag Tests
// ============================================================================

/**
 * @brief Tests lag operation on time series
 * @test Verifies that lag operation correctly shifts series and returns appropriate size
 */
TEST(LagTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = statcpp::lag(data.begin(), data.end(), 2);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

// ============================================================================
// NaN Handling (v0.5.0, docs/NAN_POLICY.md)
// ============================================================================

/**
 * @brief Tests that the autocorrelation functions reject NaN
 * @test Verifies R's acf behaviour (na.action = na.fail raises an error), including lag 0
 */
TEST(TimeSeriesNanTest, AutocorrelationThrows) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> data = {1.0, 2.0, nan, 4.0, 5.0};
    EXPECT_THROW(statcpp::autocorrelation(data.begin(), data.end(), 0), std::invalid_argument);
    EXPECT_THROW(statcpp::autocorrelation(data.begin(), data.end(), 1), std::invalid_argument);
    EXPECT_THROW(statcpp::acf(data.begin(), data.end(), 2), std::invalid_argument);
    EXPECT_THROW(statcpp::pacf(data.begin(), data.end(), 2), std::invalid_argument);
}

/**
 * @brief Tests that moving_average gives NaN only for the windows that contain NaN
 * @test Expected values: stats::filter(x, rep(1 / w, w), sides = 1) in R 4.4.2, without the w - 1 leading NA
 */
TEST(TimeSeriesNanTest, MovingAverageNaNOnlyInAffectedWindows) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> data = {1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0};
    auto result = statcpp::moving_average(data.begin(), data.end(), 3);
    ASSERT_EQ(result.size(), 5u);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_TRUE(std::isnan(result[2]));
    EXPECT_DOUBLE_EQ(result[3], 5.0);
    EXPECT_DOUBLE_EQ(result[4], 6.0);

    std::vector<double> first_nan = {nan, 2.0, 3.0, 4.0, 5.0};
    auto r2 = statcpp::moving_average(first_nan.begin(), first_nan.end(), 2);
    ASSERT_EQ(r2.size(), 4u);
    EXPECT_TRUE(std::isnan(r2[0]));
    EXPECT_DOUBLE_EQ(r2[1], 2.5);
    EXPECT_DOUBLE_EQ(r2[2], 3.5);
    EXPECT_DOUBLE_EQ(r2[3], 4.5);
}

/**
 * @brief Tests that moving_average handles Inf per window
 * @test Expected values: stats::filter(x, rep(1 / 2, 2), sides = 1) in R 4.4.2, without the leading NA
 *       (Inf - Inf in the running sum used to make every later window NaN)
 */
TEST(TimeSeriesNanTest, MovingAverageInfPerWindow) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> a = {1.0, inf, 2.0, 3.0, 4.0};
    auto ra = statcpp::moving_average(a.begin(), a.end(), 2);
    ASSERT_EQ(ra.size(), 4u);
    EXPECT_EQ(ra[0], inf);
    EXPECT_EQ(ra[1], inf);
    EXPECT_DOUBLE_EQ(ra[2], 2.5);
    EXPECT_DOUBLE_EQ(ra[3], 3.5);

    std::vector<double> b = {inf, 1.0, -inf, 3.0};
    auto rb = statcpp::moving_average(b.begin(), b.end(), 2);
    ASSERT_EQ(rb.size(), 3u);
    EXPECT_EQ(rb[0], inf);
    EXPECT_EQ(rb[1], -inf);
    EXPECT_EQ(rb[2], -inf);
}
