#include <gtest/gtest.h>
#include "statcpp/correlation_covariance.hpp"
#include <vector>
#include <cmath>

// ============================================================================
// Population Covariance Tests
// ============================================================================

/**
 * @brief Tests population covariance calculation for positive linear relationship
 * @test Verifies that perfect positive linear relationship produces correct positive covariance value
 */
TEST(PopulationCovarianceTest, PositiveCovariance) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    // Perfect positive linear relationship
    // mean_x = 3, mean_y = 6
    // cov = ((1-3)(2-6) + (2-3)(4-6) + ... + (5-3)(10-6)) / 5
    //     = (8 + 2 + 0 + 2 + 8) / 5 = 4
    EXPECT_DOUBLE_EQ(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end()), 4.0);
}

/**
 * @brief Tests population covariance calculation for negative linear relationship
 * @test Verifies that perfect negative linear relationship produces correct negative covariance value
 */
TEST(PopulationCovarianceTest, NegativeCovariance) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {10, 8, 6, 4, 2};
    // Perfect negative linear relationship
    EXPECT_DOUBLE_EQ(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end()), -4.0);
}

/**
 * @brief Tests population covariance calculation with constant variable
 * @test Verifies that covariance with constant variable equals zero
 */
TEST(PopulationCovarianceTest, ZeroCovariance) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {5, 5, 5, 5, 5};  // Constant
    EXPECT_DOUBLE_EQ(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end()), 0.0);
}

/**
 * @brief Tests population covariance calculation with empty data range
 * @test Verifies that empty ranges throw std::invalid_argument exception
 */
TEST(PopulationCovarianceTest, EmptyRange) {
    std::vector<double> x;
    std::vector<double> y;
    EXPECT_THROW(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests population covariance calculation with unequal length ranges
 * @test Verifies that unequal length ranges throw std::invalid_argument exception
 */
TEST(PopulationCovarianceTest, UnequalLength) {
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {1, 2, 3, 4};
    EXPECT_THROW(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests population covariance calculation with precomputed means
 * @test Verifies that using precomputed means produces correct covariance value
 */
TEST(PopulationCovarianceTest, PrecomputedMean) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    EXPECT_DOUBLE_EQ(statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end(), 3.0, 6.0), 4.0);
}

// ============================================================================
// Sample Covariance Tests
// ============================================================================

/**
 * @brief Tests sample covariance calculation for positive linear relationship
 * @test Verifies that perfect positive linear relationship produces correct sample covariance value
 */
TEST(SampleCovarianceTest, PositiveCovariance) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    // sample cov = 20 / 4 = 5
    EXPECT_DOUBLE_EQ(statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end()), 5.0);
}

/**
 * @brief Tests sample covariance calculation with single element
 * @test Verifies that single element throws std::invalid_argument exception
 */
TEST(SampleCovarianceTest, OneElement) {
    std::vector<double> x = {1};
    std::vector<double> y = {2};
    EXPECT_THROW(statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests sample covariance calculation with two elements
 * @test Verifies that two elements produce correct sample covariance value
 */
TEST(SampleCovarianceTest, TwoElements) {
    std::vector<double> x = {1, 2};
    std::vector<double> y = {3, 5};
    // mean_x = 1.5, mean_y = 4
    // sample cov = ((1-1.5)(3-4) + (2-1.5)(5-4)) / 1 = (0.5 + 0.5) / 1 = 1
    EXPECT_DOUBLE_EQ(statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end()), 1.0);
}

// ============================================================================
// Covariance Alias Tests
// ============================================================================

/**
 * @brief Tests that covariance alias function matches sample covariance
 * @test Verifies that covariance() is an alias for sample_covariance()
 */
TEST(CovarianceAliasTest, IsSampleCovariance) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    EXPECT_DOUBLE_EQ(statcpp::covariance(x.begin(), x.end(), y.begin(), y.end()),
                     statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end()));
}

// ============================================================================
// Pearson Correlation Tests
// ============================================================================

/**
 * @brief Tests Pearson correlation calculation for perfect positive correlation
 * @test Verifies that perfect positive linear relationship produces correlation of 1.0
 */
TEST(PearsonCorrelationTest, PerfectPositive) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    EXPECT_NEAR(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()), 1.0, 1e-10);
}

/**
 * @brief Tests Pearson correlation calculation for perfect negative correlation
 * @test Verifies that perfect negative linear relationship produces correlation of -1.0
 */
TEST(PearsonCorrelationTest, PerfectNegative) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {10, 8, 6, 4, 2};
    EXPECT_NEAR(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()), -1.0, 1e-10);
}

/**
 * @brief Tests Pearson correlation calculation for low correlation
 * @test Verifies that data with low correlation produces small absolute correlation value
 */
TEST(PearsonCorrelationTest, NoCorrelation) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {1, -1, 1, -1, 1};
    // Not perfectly uncorrelated, but low correlation
    double r = statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end());
    EXPECT_LT(std::abs(r), 0.5);
}

/**
 * @brief Tests Pearson correlation calculation with empty data range
 * @test Verifies that empty ranges throw std::invalid_argument exception
 */
TEST(PearsonCorrelationTest, EmptyRange) {
    std::vector<double> x;
    std::vector<double> y;
    EXPECT_THROW(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests Pearson correlation calculation with single element
 * @test Verifies that single element throws std::invalid_argument exception
 */
TEST(PearsonCorrelationTest, OneElement) {
    std::vector<double> x = {1};
    std::vector<double> y = {2};
    EXPECT_THROW(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests Pearson correlation calculation with zero variance data
 * @test Verifies that data with zero variance throws std::invalid_argument exception
 */
TEST(PearsonCorrelationTest, ZeroVariance) {
    std::vector<double> x = {5, 5, 5, 5};
    std::vector<double> y = {1, 2, 3, 4};
    EXPECT_THROW(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests Pearson correlation calculation with unequal length ranges
 * @test Verifies that unequal length ranges throw std::invalid_argument exception
 */
TEST(PearsonCorrelationTest, UnequalLength) {
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {1, 2, 3, 4};
    EXPECT_THROW(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests Pearson correlation calculation with precomputed means
 * @test Verifies that using precomputed means produces correct correlation value
 */
TEST(PearsonCorrelationTest, PrecomputedMean) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    EXPECT_NEAR(statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end(), 3.0, 6.0),
                1.0, 1e-10);
}

// ============================================================================
// Spearman Correlation Tests
// ============================================================================

/**
 * @brief Tests Spearman correlation calculation for perfect positive correlation
 * @test Verifies that same rank order produces Spearman correlation of 1.0
 */
TEST(SpearmanCorrelationTest, PerfectPositive) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {10, 20, 30, 40, 50};  // Same rank order
    EXPECT_NEAR(statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end()), 1.0, 1e-10);
}

/**
 * @brief Tests Spearman correlation calculation for perfect negative correlation
 * @test Verifies that reverse rank order produces Spearman correlation of -1.0
 */
TEST(SpearmanCorrelationTest, PerfectNegative) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {50, 40, 30, 20, 10};  // Reverse rank order
    EXPECT_NEAR(statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end()), -1.0, 1e-10);
}

/**
 * @brief Tests Spearman correlation calculation for monotonic non-linear relationship
 * @test Verifies that monotonic relationship produces Spearman correlation of 1.0 even if non-linear
 */
TEST(SpearmanCorrelationTest, MonotonicButNotLinear) {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {1, 4, 9, 16, 25};  // y = x^2, monotonic
    // Spearman should be 1 because rank order is preserved
    EXPECT_NEAR(statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end()), 1.0, 1e-10);
}

/**
 * @brief Tests Spearman correlation calculation with tied ranks
 * @test Verifies that tied values are handled correctly using average ranks
 */
TEST(SpearmanCorrelationTest, TiedRanks) {
    std::vector<double> x = {1, 2, 2, 4, 5};  // Tied values
    std::vector<double> y = {1, 3, 3, 4, 5};  // Tied values
    // Should handle ties using average ranks
    double rho = statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end());
    EXPECT_GT(rho, 0.9);  // Should be strongly positive
}

/**
 * @brief Tests Spearman correlation calculation with empty data range
 * @test Verifies that empty ranges throw std::invalid_argument exception
 */
TEST(SpearmanCorrelationTest, EmptyRange) {
    std::vector<double> x;
    std::vector<double> y;
    EXPECT_THROW(statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests Spearman correlation calculation with single element
 * @test Verifies that single element throws std::invalid_argument exception
 */
TEST(SpearmanCorrelationTest, OneElement) {
    std::vector<double> x = {1};
    std::vector<double> y = {2};
    EXPECT_THROW(statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

// ============================================================================
// Projection Tests
// ============================================================================

/**
 * @brief Tests covariance calculation with projection function
 * @test Verifies that population covariance works correctly with custom projection function
 */
TEST(CorrelationProjectionTest, CovarianceWithProjection) {
    struct Item { double value; };
    std::vector<Item> x = {{1}, {2}, {3}, {4}, {5}};
    std::vector<Item> y = {{2}, {4}, {6}, {8}, {10}};
    auto proj = [](const Item& i) { return i.value; };
    auto result = statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end(), proj, proj);
    // This computes cov(x.value, y.value) which should be 4.0
    // Note: This uses same projection for both, so it's testing the interface
    EXPECT_DOUBLE_EQ(result, 4.0);
}

/**
 * @brief Tests Pearson correlation calculation with projection function
 * @test Verifies that Pearson correlation works correctly with custom projection function
 */
TEST(CorrelationProjectionTest, PearsonWithProjection) {
    struct Item { double value; };
    std::vector<Item> x = {{1}, {2}, {3}, {4}, {5}};
    std::vector<Item> y = {{2}, {4}, {6}, {8}, {10}};
    auto proj = [](const Item& i) { return i.value; };
    auto result = statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end(), proj, proj);
    EXPECT_NEAR(result, 1.0, 1e-10);
}

/**
 * @brief Tests Spearman correlation calculation with projection function
 * @test Verifies that Spearman correlation works correctly with custom projection function
 */
TEST(CorrelationProjectionTest, SpearmanWithProjection) {
    struct Item { double value; };
    std::vector<Item> x = {{1}, {2}, {3}, {4}, {5}};
    std::vector<Item> y = {{10}, {20}, {30}, {40}, {50}};
    auto proj = [](const Item& i) { return i.value; };
    auto result = statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end(), proj, proj);
    EXPECT_NEAR(result, 1.0, 1e-10);
}

// ============================================================================
// Kendall's Tau Tests
// ============================================================================

/**
 * @brief Tests Kendall's tau for perfect positive concordance
 * @test Verifies tau = 1.0 when all pairs are concordant
 */
TEST(KendallTauTest, PerfectPositive) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end()), 1.0, 1e-10);
}

/**
 * @brief Tests Kendall's tau for perfect negative concordance
 * @test Verifies tau = -1.0 when all pairs are discordant
 */
TEST(KendallTauTest, PerfectNegative) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_NEAR(statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end()), -1.0, 1e-10);
}

/**
 * @brief Tests Kendall's tau with integer ties in x
 * @test Verifies tau-b handles ties correctly when x has repeated values
 */
TEST(KendallTauTest, TiedValues) {
    // x = {1, 1, 2, 3}, y = {1, 2, 3, 4}
    // Pairs (x_diff, y_diff): (0,1) tie_x, (1,2) conc, (2,3) conc,
    //                          (1,1) conc, (2,2) conc, (1,1) conc
    // tau-b = (C - D) / sqrt((n0 - T_x)(n0 - T_y))
    std::vector<double> x = {1.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    double tau = statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end());
    EXPECT_GE(tau, 0.0);
    EXPECT_LE(tau, 1.0);
}

/**
 * @brief Tests Kendall's tau with all values tied in x
 * @test Verifies tau = 0.0 when all x values are identical (degenerate case)
 */
TEST(KendallTauTest, AllTiedInX) {
    std::vector<double> x = {2.0, 2.0, 2.0};
    std::vector<double> y = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end()), 0.0);
}

/**
 * @brief Tests Kendall's tau throws for empty range
 * @test Verifies std::invalid_argument is thrown for empty input
 */
TEST(KendallTauTest, EmptyRange) {
    std::vector<double> x, y;
    EXPECT_THROW(statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

// ============================================================================
// Weighted Covariance Tests
// ============================================================================

/**
 * @brief Tests weighted covariance with uniform weights
 * @test With all weights = 1 the result should equal the sample covariance
 */
TEST(WeightedCovarianceTest, UniformWeightsMatchSampleCovariance) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> w = {1.0, 1.0, 1.0, 1.0, 1.0};
    double wc = statcpp::weighted_covariance(x.begin(), x.end(), y.begin(), y.end(), w.begin());
    double sc = statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end());
    EXPECT_NEAR(wc, sc, 1e-10);
}

/**
 * @brief Tests weighted covariance throws for negative weight
 * @test Verifies std::invalid_argument is thrown when a weight is negative
 */
TEST(WeightedCovarianceTest, NegativeWeight) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> w = {1.0, -1.0, 1.0};
    EXPECT_THROW(statcpp::weighted_covariance(x.begin(), x.end(), y.begin(), y.end(), w.begin()),
                 std::invalid_argument);
}
