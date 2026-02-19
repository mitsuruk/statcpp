#include <gtest/gtest.h>
#include "statcpp/dispersion_spread.hpp"
#include <vector>
#include <cmath>

// ============================================================================
// Range Tests
// ============================================================================

/**
 * @brief Test range calculation with basic dataset
 * @test Verify range function returns difference between max and min values
 */
TEST(RangeTest, BasicRange) {
    std::vector<int> data = {1, 5, 3, 9, 2};
    EXPECT_DOUBLE_EQ(statcpp::range(data.begin(), data.end()), 8.0);  // 9 - 1
}

/**
 * @brief Test range calculation when all values are identical
 * @test Verify range function returns 0 when all elements are the same
 */
TEST(RangeTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5};
    EXPECT_DOUBLE_EQ(statcpp::range(data.begin(), data.end()), 0.0);
}

/**
 * @brief Test range calculation with empty range
 * @test Verify range function throws exception for empty input
 */
TEST(RangeTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::range(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test range calculation with projection function
 * @test Verify range function works with custom projection to extract values
 */
TEST(RangeTest, Projection) {
    struct Item { double value; };
    std::vector<Item> data = {{1.0}, {5.0}, {3.0}};
    auto result = statcpp::range(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 4.0);
}

// ============================================================================
// var() with ddof Tests (NumPy-style)
// ============================================================================

/**
 * @brief Test variance with ddof=0 equals population variance
 * @test Verify var function with ddof=0 matches population_variance
 */
TEST(VarTest, Ddof0EqualsPopulationVariance) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // ddof=0 should equal population variance
    EXPECT_DOUBLE_EQ(statcpp::var(data.begin(), data.end(), 0),
                     statcpp::population_variance(data.begin(), data.end()));
}

/**
 * @brief Test variance with ddof=1 equals sample variance
 * @test Verify var function with ddof=1 matches sample_variance
 */
TEST(VarTest, Ddof1EqualsSampleVariance) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // ddof=1 should equal sample variance
    EXPECT_DOUBLE_EQ(statcpp::var(data.begin(), data.end(), 1),
                     statcpp::sample_variance(data.begin(), data.end()));
}

/**
 * @brief Test variance default ddof parameter
 * @test Verify var function defaults to ddof=0 (population variance)
 */
TEST(VarTest, DefaultDdofIsZero) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // Default ddof should be 0 (population variance)
    EXPECT_DOUBLE_EQ(statcpp::var(data.begin(), data.end()),
                     statcpp::var(data.begin(), data.end(), 0));
}

/**
 * @brief Test variance with invalid ddof values
 * @test Verify var function throws exception for ddof > 1
 */
TEST(VarTest, DdofMustBe0Or1) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    // ddof > 1 should throw
    EXPECT_THROW(statcpp::var(data.begin(), data.end(), 2), std::invalid_argument);
    EXPECT_THROW(statcpp::var(data.begin(), data.end(), 3), std::invalid_argument);
}

/**
 * @brief Test variance with ddof=1 requires at least 2 elements
 * @test Verify var function throws exception for single element with ddof=1
 */
TEST(VarTest, Ddof1NeedsAtLeast2Elements) {
    std::vector<double> data1 = {5.0};
    // ddof=1 with n=1 should throw
    EXPECT_THROW(statcpp::var(data1.begin(), data1.end(), 1), std::invalid_argument);
    // ddof=0 with n=1 should work
    EXPECT_NO_THROW(statcpp::var(data1.begin(), data1.end(), 0));
}

/**
 * @brief Test variance calculation with empty range
 * @test Verify var function throws exception for empty input
 */
TEST(VarTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::var(data.begin(), data.end(), 0), std::invalid_argument);
}

/**
 * @brief Test variance calculation with projection function
 * @test Verify var function works with custom projection for both ddof values
 */
TEST(VarTest, WithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}};
    auto proj = [](const Item& i) { return i.value; };

    EXPECT_DOUBLE_EQ(statcpp::var(data.begin(), data.end(), proj, 0), 4.0);
    EXPECT_NEAR(statcpp::var(data.begin(), data.end(), proj, 1), 32.0 / 7.0, 1e-10);
}

// ============================================================================
// stdev() with ddof Tests (NumPy-style)
// ============================================================================

/**
 * @brief Test standard deviation with ddof=0 equals population stddev
 * @test Verify stdev function with ddof=0 matches population_stddev
 */
TEST(StdevTest, Ddof0EqualsPopulationStddev) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::stdev(data.begin(), data.end(), 0),
                     statcpp::population_stddev(data.begin(), data.end()));
}

/**
 * @brief Test standard deviation with ddof=1 equals sample stddev
 * @test Verify stdev function with ddof=1 matches sample_stddev
 */
TEST(StdevTest, Ddof1EqualsSampleStddev) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::stdev(data.begin(), data.end(), 1),
                     statcpp::sample_stddev(data.begin(), data.end()));
}

/**
 * @brief Test standard deviation default ddof parameter
 * @test Verify stdev function defaults to ddof=0 (population stddev)
 */
TEST(StdevTest, DefaultDdofIsZero) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::stdev(data.begin(), data.end()),
                     statcpp::stdev(data.begin(), data.end(), 0));
}

/**
 * @brief Test standard deviation is square root of variance
 * @test Verify stdev function returns sqrt of var for all supported ddof values
 */
TEST(StdevTest, IsSqrtOfVar) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // Only ddof=0 and ddof=1 are supported
    for (std::size_t ddof = 0; ddof <= 1; ++ddof) {
        EXPECT_DOUBLE_EQ(statcpp::stdev(data.begin(), data.end(), ddof),
                         std::sqrt(statcpp::var(data.begin(), data.end(), ddof)));
    }
}

/**
 * @brief Test standard deviation calculation with projection function
 * @test Verify stdev function works with custom projection
 */
TEST(StdevTest, WithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}};
    auto proj = [](const Item& i) { return i.value; };

    EXPECT_DOUBLE_EQ(statcpp::stdev(data.begin(), data.end(), proj, 0), 2.0);
}

// ============================================================================
// Population Variance Tests
// ============================================================================

/**
 * @brief Test population variance calculation
 * @test Verify population_variance function computes correct variance using N denominator
 */
TEST(PopulationVarianceTest, BasicVariance) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // mean = 5, population variance = 4
    EXPECT_DOUBLE_EQ(statcpp::population_variance(data.begin(), data.end()), 4.0);
}

/**
 * @brief Test population variance when all values are identical
 * @test Verify population_variance function returns 0 when all elements are the same
 */
TEST(PopulationVarianceTest, AllSame) {
    std::vector<double> data = {5, 5, 5, 5};
    EXPECT_DOUBLE_EQ(statcpp::population_variance(data.begin(), data.end()), 0.0);
}

/**
 * @brief Test population variance with empty range
 * @test Verify population_variance function throws exception for empty input
 */
TEST(PopulationVarianceTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::population_variance(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test population variance with precomputed mean
 * @test Verify population_variance function accepts precomputed mean value
 */
TEST(PopulationVarianceTest, PrecomputedMean) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    double mean = 5.0;
    EXPECT_DOUBLE_EQ(statcpp::population_variance(data.begin(), data.end(), mean), 4.0);
}

// ============================================================================
// Sample Variance Tests
// ============================================================================

/**
 * @brief Test sample variance calculation
 * @test Verify sample_variance function computes correct variance using N-1 denominator
 */
TEST(SampleVarianceTest, BasicVariance) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // mean = 5, sample variance = 32/7 ≈ 4.571
    double expected = 32.0 / 7.0;
    EXPECT_NEAR(statcpp::sample_variance(data.begin(), data.end()), expected, 1e-10);
}

/**
 * @brief Test sample variance with two elements
 * @test Verify sample_variance function handles minimum viable sample size
 */
TEST(SampleVarianceTest, TwoElements) {
    std::vector<double> data = {1, 3};
    // mean = 2, sample variance = ((1-2)^2 + (3-2)^2) / 1 = 2
    EXPECT_DOUBLE_EQ(statcpp::sample_variance(data.begin(), data.end()), 2.0);
}

/**
 * @brief Test sample variance with single element
 * @test Verify sample_variance function throws exception for single element
 */
TEST(SampleVarianceTest, OneElement) {
    std::vector<double> data = {5};
    EXPECT_THROW(statcpp::sample_variance(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test sample variance with empty range
 * @test Verify sample_variance function throws exception for empty input
 */
TEST(SampleVarianceTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::sample_variance(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Variance Alias Tests
// ============================================================================

/**
 * @brief Test variance alias function
 * @test Verify variance function is an alias for sample_variance
 */
TEST(VarianceTest, IsSampleVariance) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::variance(data.begin(), data.end()),
                     statcpp::sample_variance(data.begin(), data.end()));
}

// ============================================================================
// Population Standard Deviation Tests
// ============================================================================

/**
 * @brief Test population standard deviation calculation
 * @test Verify population_stddev function computes sqrt of population variance
 */
TEST(PopulationStddevTest, BasicStddev) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    // population variance = 4, stddev = 2
    EXPECT_DOUBLE_EQ(statcpp::population_stddev(data.begin(), data.end()), 2.0);
}

/**
 * @brief Test population standard deviation when all values are identical
 * @test Verify population_stddev function returns 0 when all elements are the same
 */
TEST(PopulationStddevTest, AllSame) {
    std::vector<double> data = {5, 5, 5, 5};
    EXPECT_DOUBLE_EQ(statcpp::population_stddev(data.begin(), data.end()), 0.0);
}

// ============================================================================
// Sample Standard Deviation Tests
// ============================================================================

/**
 * @brief Test sample standard deviation calculation
 * @test Verify sample_stddev function computes sqrt of sample variance
 */
TEST(SampleStddevTest, BasicStddev) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    double expected = std::sqrt(32.0 / 7.0);
    EXPECT_NEAR(statcpp::sample_stddev(data.begin(), data.end()), expected, 1e-10);
}

// ============================================================================
// Stddev Alias Tests
// ============================================================================

/**
 * @brief Test stddev alias function
 * @test Verify stddev function is an alias for sample_stddev
 */
TEST(StddevTest, IsSampleStddev) {
    std::vector<double> data = {2, 4, 4, 4, 5, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::stddev(data.begin(), data.end()),
                     statcpp::sample_stddev(data.begin(), data.end()));
}

// ============================================================================
// Coefficient of Variation Tests
// ============================================================================

/**
 * @brief Test coefficient of variation calculation
 * @test Verify coefficient_of_variation function computes ratio of stddev to mean
 */
TEST(CoefficientOfVariationTest, BasicCV) {
    std::vector<double> data = {10, 20, 30, 40, 50};
    // mean = 30, sample stddev = sqrt(250) ≈ 15.81
    // CV = 15.81 / 30 ≈ 0.527
    double mean = 30.0;
    double stddev = std::sqrt(250.0);
    double expected = stddev / mean;
    EXPECT_NEAR(statcpp::coefficient_of_variation(data.begin(), data.end()), expected, 1e-10);
}

/**
 * @brief Test coefficient of variation with zero mean
 * @test Verify coefficient_of_variation function throws exception when mean is zero
 */
TEST(CoefficientOfVariationTest, ZeroMean) {
    std::vector<double> data = {-1, 0, 1};
    EXPECT_THROW(statcpp::coefficient_of_variation(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test coefficient of variation with single element
 * @test Verify coefficient_of_variation function throws exception for single element
 */
TEST(CoefficientOfVariationTest, OneElement) {
    std::vector<double> data = {5};
    EXPECT_THROW(statcpp::coefficient_of_variation(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// IQR Tests
// ============================================================================

/**
 * @brief Test interquartile range calculation
 * @test Verify iqr function computes difference between Q3 and Q1
 */
TEST(IQRTest, BasicIQR) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    // Q1 (p=0.25): index = 0.25 * 9 = 2.25 -> 3 * 0.75 + 4 * 0.25 = 3.25
    // Q3 (p=0.75): index = 0.75 * 9 = 6.75 -> 7 * 0.25 + 8 * 0.75 = 7.75
    // IQR = 7.75 - 3.25 = 4.5
    EXPECT_NEAR(statcpp::iqr(data.begin(), data.end()), 4.5, 1e-10);
}

/**
 * @brief Test interquartile range with empty range
 * @test Verify iqr function throws exception for empty input
 */
TEST(IQRTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::iqr(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Mean Absolute Deviation Tests
// ============================================================================

/**
 * @brief Test mean absolute deviation calculation
 * @test Verify mean_absolute_deviation function computes average of absolute deviations from mean
 */
TEST(MeanAbsoluteDeviationTest, BasicMAD) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    // mean = 3
    // MAD = (|1-3| + |2-3| + |3-3| + |4-3| + |5-3|) / 5 = (2+1+0+1+2)/5 = 1.2
    EXPECT_DOUBLE_EQ(statcpp::mean_absolute_deviation(data.begin(), data.end()), 1.2);
}

/**
 * @brief Test mean absolute deviation when all values are identical
 * @test Verify mean_absolute_deviation function returns 0 when all elements are the same
 */
TEST(MeanAbsoluteDeviationTest, AllSame) {
    std::vector<double> data = {5, 5, 5, 5};
    EXPECT_DOUBLE_EQ(statcpp::mean_absolute_deviation(data.begin(), data.end()), 0.0);
}

/**
 * @brief Test mean absolute deviation with empty range
 * @test Verify mean_absolute_deviation function throws exception for empty input
 */
TEST(MeanAbsoluteDeviationTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::mean_absolute_deviation(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test mean absolute deviation with precomputed mean
 * @test Verify mean_absolute_deviation function accepts precomputed mean value
 */
TEST(MeanAbsoluteDeviationTest, PrecomputedMean) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    double mean = 3.0;
    EXPECT_DOUBLE_EQ(statcpp::mean_absolute_deviation(data.begin(), data.end(), mean), 1.2);
}

// ============================================================================
// Projection Tests
// ============================================================================

/**
 * @brief Test variance calculation with projection function
 * @test Verify population_variance function works with custom projection
 */
TEST(DispersionProjectionTest, VarianceWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}};
    auto result = statcpp::population_variance(data.begin(), data.end(),
                                               [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 4.0);
}

/**
 * @brief Test standard deviation calculation with projection function
 * @test Verify population_stddev function works with custom projection
 */
TEST(DispersionProjectionTest, StddevWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}};
    auto result = statcpp::population_stddev(data.begin(), data.end(),
                                             [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 2.0);
}
