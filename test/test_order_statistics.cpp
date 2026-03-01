#include <gtest/gtest.h>
#include "statcpp/order_statistics.hpp"
#include <vector>

// ============================================================================
// Minimum Tests
// ============================================================================

/**
 * @brief Test minimum value calculation
 * @test Verify minimum function returns smallest value in dataset
 */
TEST(MinimumTest, BasicMinimum) {
    std::vector<int> data = {5, 2, 8, 1, 9};
    EXPECT_EQ(statcpp::minimum(data.begin(), data.end()), 1);
}

/**
 * @brief Test minimum when all values are identical
 * @test Verify minimum function returns the repeated value when all elements are the same
 */
TEST(MinimumTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5};
    EXPECT_EQ(statcpp::minimum(data.begin(), data.end()), 5);
}

/**
 * @brief Test minimum with single element
 * @test Verify minimum function returns the element itself for single-element dataset
 */
TEST(MinimumTest, SingleElement) {
    std::vector<int> data = {42};
    EXPECT_EQ(statcpp::minimum(data.begin(), data.end()), 42);
}

/**
 * @brief Test minimum with empty range
 * @test Verify minimum function throws exception for empty input
 */
TEST(MinimumTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::minimum(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test minimum calculation with projection function
 * @test Verify minimum function works with custom projection to extract values
 */
TEST(MinimumTest, Projection) {
    struct Item { int value; };
    std::vector<Item> data = {{5}, {2}, {8}};
    auto result = statcpp::minimum(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_EQ(result, 2);
}

// ============================================================================
// Maximum Tests
// ============================================================================

/**
 * @brief Test maximum value calculation
 * @test Verify maximum function returns largest value in dataset
 */
TEST(MaximumTest, BasicMaximum) {
    std::vector<int> data = {5, 2, 8, 1, 9};
    EXPECT_EQ(statcpp::maximum(data.begin(), data.end()), 9);
}

/**
 * @brief Test maximum when all values are identical
 * @test Verify maximum function returns the repeated value when all elements are the same
 */
TEST(MaximumTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5};
    EXPECT_EQ(statcpp::maximum(data.begin(), data.end()), 5);
}

/**
 * @brief Test maximum with single element
 * @test Verify maximum function returns the element itself for single-element dataset
 */
TEST(MaximumTest, SingleElement) {
    std::vector<int> data = {42};
    EXPECT_EQ(statcpp::maximum(data.begin(), data.end()), 42);
}

/**
 * @brief Test maximum with empty range
 * @test Verify maximum function throws exception for empty input
 */
TEST(MaximumTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::maximum(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test maximum calculation with projection function
 * @test Verify maximum function works with custom projection to extract values
 */
TEST(MaximumTest, Projection) {
    struct Item { int value; };
    std::vector<Item> data = {{5}, {2}, {8}};
    auto result = statcpp::maximum(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_EQ(result, 8);
}

// ============================================================================
// Quartiles Tests
// ============================================================================

/**
 * @brief Test quartiles calculation with ten elements
 * @test Verify quartiles function computes Q1, Q2 (median), and Q3 correctly
 */
TEST(QuartilesTest, TenElements) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    auto result = statcpp::quartiles(data.begin(), data.end());
    // Q1: index = 0.25 * 9 = 2.25 -> 3 * 0.75 + 4 * 0.25 = 3.25
    // Q2: index = 0.50 * 9 = 4.5 -> 5 * 0.5 + 6 * 0.5 = 5.5
    // Q3: index = 0.75 * 9 = 6.75 -> 7 * 0.25 + 8 * 0.75 = 7.75
    EXPECT_NEAR(result.q1, 3.25, 1e-10);
    EXPECT_NEAR(result.q2, 5.5, 1e-10);
    EXPECT_NEAR(result.q3, 7.75, 1e-10);
}

/**
 * @brief Test quartiles calculation with four elements
 * @test Verify quartiles function handles small datasets correctly
 */
TEST(QuartilesTest, FourElements) {
    std::vector<double> data = {1, 2, 3, 4};  // sorted
    auto result = statcpp::quartiles(data.begin(), data.end());
    // Q1: index = 0.25 * 3 = 0.75 -> 1 * 0.25 + 2 * 0.75 = 1.75
    // Q2: index = 0.50 * 3 = 1.5 -> 2 * 0.5 + 3 * 0.5 = 2.5
    // Q3: index = 0.75 * 3 = 2.25 -> 3 * 0.75 + 4 * 0.25 = 3.25
    EXPECT_NEAR(result.q1, 1.75, 1e-10);
    EXPECT_NEAR(result.q2, 2.5, 1e-10);
    EXPECT_NEAR(result.q3, 3.25, 1e-10);
}

/**
 * @brief Test quartiles calculation with single element
 * @test Verify quartiles function returns same value for all quartiles with single element
 */
TEST(QuartilesTest, SingleElement) {
    std::vector<double> data = {5};
    auto result = statcpp::quartiles(data.begin(), data.end());
    EXPECT_DOUBLE_EQ(result.q1, 5.0);
    EXPECT_DOUBLE_EQ(result.q2, 5.0);
    EXPECT_DOUBLE_EQ(result.q3, 5.0);
}

/**
 * @brief Test quartiles calculation with empty range
 * @test Verify quartiles function throws exception for empty input
 */
TEST(QuartilesTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::quartiles(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Percentile Tests
// ============================================================================

/**
 * @brief Test percentile calculation for median (50th percentile)
 * @test Verify percentile function computes 50th percentile correctly
 */
TEST(PercentileTest, Median) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    EXPECT_NEAR(statcpp::percentile(data.begin(), data.end(), 0.5), 5.5, 1e-10);
}

/**
 * @brief Test percentile calculation for minimum (0th percentile)
 * @test Verify percentile function returns minimum for p=0.0
 */
TEST(PercentileTest, Minimum) {
    std::vector<double> data = {1, 2, 3, 4, 5};  // sorted
    EXPECT_DOUBLE_EQ(statcpp::percentile(data.begin(), data.end(), 0.0), 1.0);
}

/**
 * @brief Test percentile calculation for maximum (100th percentile)
 * @test Verify percentile function returns maximum for p=1.0
 */
TEST(PercentileTest, Maximum) {
    std::vector<double> data = {1, 2, 3, 4, 5};  // sorted
    EXPECT_DOUBLE_EQ(statcpp::percentile(data.begin(), data.end(), 1.0), 5.0);
}

/**
 * @brief Test percentile calculation for 90th percentile
 * @test Verify percentile function computes arbitrary percentiles correctly
 */
TEST(PercentileTest, NinetiethPercentile) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    // index = 0.9 * 9 = 8.1 -> 9 * 0.9 + 10 * 0.1 = 9.1
    EXPECT_NEAR(statcpp::percentile(data.begin(), data.end(), 0.9), 9.1, 1e-10);
}

/**
 * @brief Test percentile calculation with invalid percentile values
 * @test Verify percentile function throws exception for p outside [0, 1]
 */
TEST(PercentileTest, InvalidPercentile) {
    std::vector<double> data = {1, 2, 3};
    EXPECT_THROW(statcpp::percentile(data.begin(), data.end(), -0.1), std::invalid_argument);
    EXPECT_THROW(statcpp::percentile(data.begin(), data.end(), 1.1), std::invalid_argument);
}

/**
 * @brief Test percentile calculation with empty range
 * @test Verify percentile function throws exception for empty input
 */
TEST(PercentileTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::percentile(data.begin(), data.end(), 0.5), std::invalid_argument);
}

// ============================================================================
// Five Number Summary Tests
// ============================================================================

/**
 * @brief Test five number summary calculation with ten elements
 * @test Verify five_number_summary function computes min, Q1, median, Q3, max correctly
 */
TEST(FiveNumberSummaryTest, TenElements) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    auto result = statcpp::five_number_summary(data.begin(), data.end());
    EXPECT_DOUBLE_EQ(result.min, 1.0);
    EXPECT_NEAR(result.q1, 3.25, 1e-10);
    EXPECT_NEAR(result.median, 5.5, 1e-10);
    EXPECT_NEAR(result.q3, 7.75, 1e-10);
    EXPECT_DOUBLE_EQ(result.max, 10.0);
}

/**
 * @brief Test five number summary calculation with single element
 * @test Verify five_number_summary function returns same value for all statistics with single element
 */
TEST(FiveNumberSummaryTest, SingleElement) {
    std::vector<double> data = {42};
    auto result = statcpp::five_number_summary(data.begin(), data.end());
    EXPECT_DOUBLE_EQ(result.min, 42.0);
    EXPECT_DOUBLE_EQ(result.q1, 42.0);
    EXPECT_DOUBLE_EQ(result.median, 42.0);
    EXPECT_DOUBLE_EQ(result.q3, 42.0);
    EXPECT_DOUBLE_EQ(result.max, 42.0);
}

/**
 * @brief Test five number summary calculation with empty range
 * @test Verify five_number_summary function throws exception for empty input
 */
TEST(FiveNumberSummaryTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::five_number_summary(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Projection Tests
// ============================================================================

/**
 * @brief Test quartiles calculation with projection function
 * @test Verify quartiles function works with custom projection
 */
TEST(OrderStatisticsProjectionTest, QuartilesWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
    auto result = statcpp::quartiles(data.begin(), data.end(),
                                     [](const Item& i) { return i.value; });
    EXPECT_NEAR(result.q1, 3.25, 1e-10);
    EXPECT_NEAR(result.q2, 5.5, 1e-10);
    EXPECT_NEAR(result.q3, 7.75, 1e-10);
}

/**
 * @brief Test percentile calculation with projection function
 * @test Verify percentile function works with custom projection
 */
TEST(OrderStatisticsProjectionTest, PercentileWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}};
    auto result = statcpp::percentile(data.begin(), data.end(), 0.5,
                                      [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 3.0);
}

/**
 * @brief Test five number summary calculation with projection function
 * @test Verify five_number_summary function works with custom projection
 */
TEST(OrderStatisticsProjectionTest, FiveNumberSummaryWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}};
    auto result = statcpp::five_number_summary(data.begin(), data.end(),
                                               [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result.min, 1.0);
    EXPECT_DOUBLE_EQ(result.max, 5.0);
    EXPECT_DOUBLE_EQ(result.median, 3.0);
}

// ============================================================================
// Weighted Median Tests
// ============================================================================

/**
 * @brief Test weighted median with odd number of uniform weights
 * @test Verify weighted_median returns the middle value when all weights are equal
 */
TEST(WeightedMedianTest, UniformWeights) {
    std::vector<double> data    = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_median(data.begin(), data.end(), weights.begin()), 3.0);
}

/**
 * @brief Test weighted median where a single large weight dominates
 * @test Verify weighted_median returns the element with the dominant weight
 */
TEST(WeightedMedianTest, DominantWeight) {
    std::vector<double> data    = {1.0, 2.0, 3.0};
    std::vector<double> weights = {1.0, 10.0, 1.0};
    // Sum = 12, half = 6; after element 0 cumulative = 1, after element 1 cumulative = 11 >= 6
    EXPECT_DOUBLE_EQ(statcpp::weighted_median(data.begin(), data.end(), weights.begin()), 2.0);
}

/**
 * @brief Test weighted median when cumulative weight lands exactly on half (floating-point)
 * @test Verify the half-weight boundary case averages the two straddling values.
 *       This exercises the tolerance-based comparison that replaced exact == .
 */
TEST(WeightedMedianTest, ExactHalfWeightBoundary) {
    // weights {1, 1, 1, 1}: total = 4, half = 2.0
    // After element[1] cumulative = 2.0 exactly; should average elements[1] and [2]
    std::vector<double> data    = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> weights = {1.0,  1.0,  1.0,  1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_median(data.begin(), data.end(), weights.begin()), 25.0);
}

/**
 * @brief Test weighted median throws for empty range
 * @test Verify weighted_median throws std::invalid_argument for empty input
 */
TEST(WeightedMedianTest, EmptyRange) {
    std::vector<double> data;
    std::vector<double> weights;
    EXPECT_THROW(statcpp::weighted_median(data.begin(), data.end(), weights.begin()),
                 std::invalid_argument);
}

/**
 * @brief Test weighted median throws for negative weight
 * @test Verify weighted_median throws std::invalid_argument for negative weight
 */
TEST(WeightedMedianTest, NegativeWeight) {
    std::vector<double> data    = {1.0, 2.0, 3.0};
    std::vector<double> weights = {1.0, -1.0, 1.0};
    EXPECT_THROW(statcpp::weighted_median(data.begin(), data.end(), weights.begin()),
                 std::invalid_argument);
}

/**
 * @brief Test weighted median throws when all weights are zero
 * @test Verify weighted_median throws std::invalid_argument when sum of weights is zero
 */
TEST(WeightedMedianTest, AllZeroWeights) {
    std::vector<double> data    = {1.0, 2.0, 3.0};
    std::vector<double> weights = {0.0, 0.0, 0.0};
    EXPECT_THROW(statcpp::weighted_median(data.begin(), data.end(), weights.begin()),
                 std::invalid_argument);
}

// ============================================================================
// Weighted Percentile Tests
// ============================================================================

/**
 * @brief Test weighted percentile with uniform weights
 * @test Verify weighted_percentile returns the median for p=0.5 with equal weights
 */
TEST(WeightedPercentileTest, UniformWeightsMedian) {
    std::vector<double> data    = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 0.5), 3.0);
}

/**
 * @brief Test weighted percentile at p=0.0 boundary
 * @test Verify weighted_percentile returns the minimum value for p=0
 */
TEST(WeightedPercentileTest, PZero) {
    std::vector<double> data    = {10.0, 20.0, 30.0};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 0.0), 10.0);
}

/**
 * @brief Test weighted percentile at p=1.0 boundary
 * @test Verify weighted_percentile returns the maximum value for p=1
 */
TEST(WeightedPercentileTest, POne) {
    std::vector<double> data    = {10.0, 20.0, 30.0};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 1.0), 30.0);
}

/**
 * @brief Test weighted percentile with tolerance-based comparison
 * @test Verify that cumulative weight near target triggers averaging (not exact ==)
 */
TEST(WeightedPercentileTest, ToleranceComparison) {
    // weights {1, 1, 1, 1}: total = 4, target for p=0.5 = 2.0
    // After element[1] cumulative = 2.0; should average elements[1] and [2]
    std::vector<double> data    = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> weights = {1.0,  1.0,  1.0,  1.0};
    EXPECT_DOUBLE_EQ(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 0.5), 25.0);
}

/**
 * @brief Test weighted percentile with NaN p throws exception
 * @test Verify that NaN p is rejected by the range check
 */
TEST(WeightedPercentileTest, NaNPThrows) {
    std::vector<double> data    = {1.0, 2.0, 3.0};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    EXPECT_THROW(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(),
                 std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

/**
 * @brief Test weighted percentile with empty range throws exception
 * @test Verify weighted_percentile throws for empty input
 */
TEST(WeightedPercentileTest, EmptyRange) {
    std::vector<double> data;
    std::vector<double> weights;
    EXPECT_THROW(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 0.5),
                 std::invalid_argument);
}

/**
 * @brief Test weighted percentile with negative weight throws exception
 * @test Verify weighted_percentile throws for negative weight
 */
TEST(WeightedPercentileTest, NegativeWeight) {
    std::vector<double> data    = {1.0, 2.0, 3.0};
    std::vector<double> weights = {1.0, -1.0, 1.0};
    EXPECT_THROW(statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 0.5),
                 std::invalid_argument);
}

/**
 * @brief Test weighted percentile with projection
 * @test Verify weighted_percentile with projection function works correctly
 */
TEST(WeightedPercentileTest, WithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{10.0}, {20.0}, {30.0}};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    double result = statcpp::weighted_percentile(data.begin(), data.end(), weights.begin(), 1.0,
                                                  [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 30.0);
}
