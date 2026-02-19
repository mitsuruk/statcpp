#include <gtest/gtest.h>
#include "statcpp/shape_of_distribution.hpp"
#include <vector>
#include <cmath>

// ============================================================================
// Population Skewness Tests
// ============================================================================

/**
 * @brief Tests population skewness calculation for symmetric distribution
 * @test Verifies that a symmetric distribution has skewness approximately equal to zero
 */
TEST(PopulationSkewnessTest, SymmetricDistribution) {
    // Symmetric distribution should have skewness ≈ 0
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_NEAR(statcpp::population_skewness(data.begin(), data.end()), 0.0, 1e-10);
}

/**
 * @brief Tests population skewness calculation for right-skewed distribution
 * @test Verifies that a right-skewed distribution produces positive skewness value
 */
TEST(PopulationSkewnessTest, PositiveSkew) {
    // Right-skewed distribution
    std::vector<double> data = {1, 1, 1, 2, 2, 3, 5, 10};
    double skew = statcpp::population_skewness(data.begin(), data.end());
    EXPECT_GT(skew, 0.0);  // Should be positive
}

/**
 * @brief Tests population skewness calculation for left-skewed distribution
 * @test Verifies that a left-skewed distribution produces negative skewness value
 */
TEST(PopulationSkewnessTest, NegativeSkew) {
    // Left-skewed distribution
    std::vector<double> data = {1, 6, 8, 9, 9, 9, 10, 10};
    double skew = statcpp::population_skewness(data.begin(), data.end());
    EXPECT_LT(skew, 0.0);  // Should be negative
}

/**
 * @brief Tests population skewness calculation with empty data range
 * @test Verifies that an empty range throws std::invalid_argument exception
 */
TEST(PopulationSkewnessTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::population_skewness(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests population skewness calculation with zero variance data
 * @test Verifies that data with zero variance throws std::invalid_argument exception
 */
TEST(PopulationSkewnessTest, ZeroVariance) {
    std::vector<double> data = {5, 5, 5, 5};
    EXPECT_THROW(statcpp::population_skewness(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests population skewness calculation with precomputed mean
 * @test Verifies that using a precomputed mean produces correct skewness value
 */
TEST(PopulationSkewnessTest, PrecomputedMean) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double mean = 5.0;
    EXPECT_NEAR(statcpp::population_skewness(data.begin(), data.end(), mean), 0.0, 1e-10);
}

// ============================================================================
// Sample Skewness Tests
// ============================================================================

/**
 * @brief Tests sample skewness calculation for symmetric distribution
 * @test Verifies that a symmetric distribution has sample skewness approximately equal to zero
 */
TEST(SampleSkewnessTest, SymmetricDistribution) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_NEAR(statcpp::sample_skewness(data.begin(), data.end()), 0.0, 1e-10);
}

/**
 * @brief Tests sample skewness calculation with insufficient elements
 * @test Verifies that data with less than 3 elements throws std::invalid_argument exception
 */
TEST(SampleSkewnessTest, MinimumElements) {
    std::vector<double> data = {1, 2};
    EXPECT_THROW(statcpp::sample_skewness(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests sample skewness calculation with minimum required elements
 * @test Verifies that data with exactly 3 elements can be processed without throwing
 */
TEST(SampleSkewnessTest, ThreeElements) {
    std::vector<double> data = {1, 2, 3};
    EXPECT_NO_THROW(statcpp::sample_skewness(data.begin(), data.end()));
}

// ============================================================================
// Skewness Alias Tests
// ============================================================================

/**
 * @brief Tests that skewness alias function matches sample skewness
 * @test Verifies that skewness() is an alias for sample_skewness()
 */
TEST(SkewnessAliasTest, IsSampleSkewness) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_DOUBLE_EQ(statcpp::skewness(data.begin(), data.end()),
                     statcpp::sample_skewness(data.begin(), data.end()));
}

// ============================================================================
// Population Kurtosis Tests
// ============================================================================

/**
 * @brief Tests population kurtosis calculation for normal-like distribution
 * @test Verifies that a platykurtic distribution has negative excess kurtosis
 */
TEST(PopulationKurtosisTest, NormalLikeDistribution) {
    // For a perfectly normal distribution, excess kurtosis = 0
    // This data is approximately normal
    std::vector<double> data = {-2, -1, -1, 0, 0, 0, 0, 1, 1, 2};
    double kurt = statcpp::population_kurtosis(data.begin(), data.end());
    // This distribution is platykurtic (flatter than normal)
    EXPECT_NEAR(kurt, -0.5, 0.1);
}

/**
 * @brief Tests population kurtosis calculation for uniform distribution
 * @test Verifies that a uniform distribution has negative excess kurtosis (platykurtic)
 */
TEST(PopulationKurtosisTest, UniformDistribution) {
    // Uniform distribution has negative excess kurtosis
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double kurt = statcpp::population_kurtosis(data.begin(), data.end());
    EXPECT_LT(kurt, 0.0);  // Should be negative (platykurtic)
}

/**
 * @brief Tests population kurtosis calculation for heavy-tailed distribution
 * @test Verifies that a heavy-tailed distribution has positive excess kurtosis (leptokurtic)
 */
TEST(PopulationKurtosisTest, HeavyTails) {
    // Heavy-tailed distribution has positive excess kurtosis
    std::vector<double> data = {0, 0, 0, 0, 0, 0, 0, 0, 10, -10};
    double kurt = statcpp::population_kurtosis(data.begin(), data.end());
    EXPECT_GT(kurt, 0.0);  // Should be positive (leptokurtic)
}

/**
 * @brief Tests population kurtosis calculation with empty data range
 * @test Verifies that an empty range throws std::invalid_argument exception
 */
TEST(PopulationKurtosisTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::population_kurtosis(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests population kurtosis calculation with zero variance data
 * @test Verifies that data with zero variance throws std::invalid_argument exception
 */
TEST(PopulationKurtosisTest, ZeroVariance) {
    std::vector<double> data = {5, 5, 5, 5};
    EXPECT_THROW(statcpp::population_kurtosis(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests population kurtosis calculation with precomputed mean
 * @test Verifies that using a precomputed mean produces the same result as without
 */
TEST(PopulationKurtosisTest, PrecomputedMean) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double mean = 5.5;
    double kurt1 = statcpp::population_kurtosis(data.begin(), data.end());
    double kurt2 = statcpp::population_kurtosis(data.begin(), data.end(), mean);
    EXPECT_NEAR(kurt1, kurt2, 1e-10);
}

// ============================================================================
// Sample Kurtosis Tests
// ============================================================================

/**
 * @brief Tests sample kurtosis calculation with insufficient elements
 * @test Verifies that data with less than 4 elements throws std::invalid_argument exception
 */
TEST(SampleKurtosisTest, MinimumElements) {
    std::vector<double> data = {1, 2, 3};
    EXPECT_THROW(statcpp::sample_kurtosis(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Tests sample kurtosis calculation with minimum required elements
 * @test Verifies that data with exactly 4 elements can be processed without throwing
 */
TEST(SampleKurtosisTest, FourElements) {
    std::vector<double> data = {1, 2, 3, 4};
    EXPECT_NO_THROW(statcpp::sample_kurtosis(data.begin(), data.end()));
}

/**
 * @brief Tests bias correction in sample kurtosis calculation
 * @test Verifies that sample kurtosis differs from population kurtosis due to bias correction
 */
TEST(SampleKurtosisTest, BiasCorrection) {
    // Sample kurtosis should be different from population kurtosis
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double pop_kurt = statcpp::population_kurtosis(data.begin(), data.end());
    double sample_kurt = statcpp::sample_kurtosis(data.begin(), data.end());
    // They should be related but different due to bias correction
    EXPECT_NE(pop_kurt, sample_kurt);
}

// ============================================================================
// Kurtosis Alias Tests
// ============================================================================

/**
 * @brief Tests that kurtosis alias function matches sample kurtosis
 * @test Verifies that kurtosis() is an alias for sample_kurtosis()
 */
TEST(KurtosisAliasTest, IsSampleKurtosis) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_DOUBLE_EQ(statcpp::kurtosis(data.begin(), data.end()),
                     statcpp::sample_kurtosis(data.begin(), data.end()));
}

// ============================================================================
// Projection Tests
// ============================================================================

/**
 * @brief Tests skewness calculation with projection function
 * @test Verifies that population skewness works correctly with custom projection function
 */
TEST(ShapeProjectionTest, SkewnessWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
    auto result = statcpp::population_skewness(data.begin(), data.end(),
                                               [](const Item& i) { return i.value; });
    EXPECT_NEAR(result, 0.0, 1e-10);
}

/**
 * @brief Tests kurtosis calculation with projection function
 * @test Verifies that population kurtosis works correctly with custom projection function
 */
TEST(ShapeProjectionTest, KurtosisWithProjection) {
    struct Item { double value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
    auto result = statcpp::population_kurtosis(data.begin(), data.end(),
                                               [](const Item& i) { return i.value; });
    EXPECT_LT(result, 0.0);  // Uniform-like, should be platykurtic
}
