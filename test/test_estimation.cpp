#include <gtest/gtest.h>
#include "statcpp/estimation.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// Standard Error Tests
// ============================================================================

/**
 * @brief Test standard error computation with basic data.
 * @test Verifies standard error calculation using sample standard deviation.
 */
TEST(StandardErrorTest, BasicComputation) {
    std::vector<double> data = {2.0, 4.0, 6.0, 8.0, 10.0};
    // stddev = sqrt(10) ≈ 3.162, se = 3.162 / sqrt(5) ≈ 1.414
    double se = statcpp::standard_error(data.begin(), data.end());
    EXPECT_NEAR(se, std::sqrt(10.0) / std::sqrt(5.0), 1e-10);
}

/**
 * @brief Test standard error with precomputed standard deviation.
 * @test Verifies standard error calculation using provided standard deviation.
 */
TEST(StandardErrorTest, WithPrecomputedStddev) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double precomputed_sd = 2.0;
    double se = statcpp::standard_error(data.begin(), data.end(), precomputed_sd);
    EXPECT_NEAR(se, 2.0 / std::sqrt(5.0), 1e-10);
}

/**
 * @brief Test standard error with insufficient data.
 * @test Verifies that exception is thrown when data has fewer than 2 elements.
 */
TEST(StandardErrorTest, TooFewElements) {
    std::vector<double> data = {1.0};
    EXPECT_THROW(statcpp::standard_error(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Confidence Interval for Mean (t-based) Tests
// ============================================================================

/**
 * @brief Test confidence interval for mean with basic data.
 * @test Verifies CI calculation and that interval bounds the point estimate.
 */
TEST(CIMeanTest, BasicComputation) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    auto ci = statcpp::ci_mean(data.begin(), data.end(), 0.95);

    EXPECT_NEAR(ci.point_estimate, 14.0, 1e-10);
    EXPECT_EQ(ci.confidence_level, 0.95);
    EXPECT_LT(ci.lower, ci.point_estimate);
    EXPECT_GT(ci.upper, ci.point_estimate);
}

/**
 * @brief Test that CI width decreases with larger sample size.
 * @test Verifies that larger samples produce narrower confidence intervals.
 */
TEST(CIMeanTest, NarrowerAtHigherN) {
    std::vector<double> small_data = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> large_data = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};

    auto ci_small = statcpp::ci_mean(small_data.begin(), small_data.end(), 0.95);
    auto ci_large = statcpp::ci_mean(large_data.begin(), large_data.end(), 0.95);

    double width_small = ci_small.upper - ci_small.lower;
    double width_large = ci_large.upper - ci_large.lower;

    // Larger sample should have narrower CI
    EXPECT_LT(width_large, width_small);
}

/**
 * @brief Test that CI width increases with higher confidence level.
 * @test Verifies that higher confidence levels produce wider intervals.
 */
TEST(CIMeanTest, WiderAtHigherConfidence) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};

    auto ci_90 = statcpp::ci_mean(data.begin(), data.end(), 0.90);
    auto ci_95 = statcpp::ci_mean(data.begin(), data.end(), 0.95);
    auto ci_99 = statcpp::ci_mean(data.begin(), data.end(), 0.99);

    double width_90 = ci_90.upper - ci_90.lower;
    double width_95 = ci_95.upper - ci_95.lower;
    double width_99 = ci_99.upper - ci_99.lower;

    EXPECT_LT(width_90, width_95);
    EXPECT_LT(width_95, width_99);
}

/**
 * @brief Test CI for mean with invalid confidence levels.
 * @test Verifies that exceptions are thrown for invalid confidence levels.
 */
TEST(CIMeanTest, InvalidConfidence) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::ci_mean(data.begin(), data.end(), 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::ci_mean(data.begin(), data.end(), 1.0), std::invalid_argument);
    EXPECT_THROW(statcpp::ci_mean(data.begin(), data.end(), -0.5), std::invalid_argument);
}

// ============================================================================
// Confidence Interval for Mean (z-based) Tests
// ============================================================================

/**
 * @brief Test z-based confidence interval for mean with known sigma.
 * @test Verifies CI calculation using z-distribution with known population variance.
 */
TEST(CIMeanZTest, BasicComputation) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    double sigma = 3.0;
    auto ci = statcpp::ci_mean_z(data.begin(), data.end(), sigma, 0.95);

    EXPECT_NEAR(ci.point_estimate, 14.0, 1e-10);
    // For 95% CI with z = 1.96, margin = 1.96 * 3 / sqrt(5) ≈ 2.63
    double expected_margin = 1.96 * sigma / std::sqrt(5.0);
    EXPECT_NEAR(ci.lower, 14.0 - expected_margin, 0.1);
    EXPECT_NEAR(ci.upper, 14.0 + expected_margin, 0.1);
}

/**
 * @brief Test z-based CI for mean with invalid sigma.
 * @test Verifies that exceptions are thrown for non-positive sigma values.
 */
TEST(CIMeanZTest, InvalidSigma) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::ci_mean_z(data.begin(), data.end(), 0.0, 0.95), std::invalid_argument);
    EXPECT_THROW(statcpp::ci_mean_z(data.begin(), data.end(), -1.0, 0.95), std::invalid_argument);
}

// ============================================================================
// Confidence Interval for Proportion Tests
// ============================================================================

/**
 * @brief Test Wald method confidence interval for proportion.
 * @test Verifies Wald CI calculation and that interval is within [0, 1].
 */
TEST(CIProportionTest, WaldMethod) {
    auto ci = statcpp::ci_proportion(60, 100, 0.95);

    EXPECT_NEAR(ci.point_estimate, 0.6, 1e-10);
    EXPECT_LT(ci.lower, 0.6);
    EXPECT_GT(ci.upper, 0.6);
    EXPECT_GE(ci.lower, 0.0);
    EXPECT_LE(ci.upper, 1.0);
}

/**
 * @brief Test Wilson score method confidence interval for proportion.
 * @test Verifies Wilson CI calculation and that interval is strictly within (0, 1).
 */
TEST(CIProportionTest, WilsonMethod) {
    auto ci = statcpp::ci_proportion_wilson(60, 100, 0.95);

    EXPECT_NEAR(ci.point_estimate, 0.6, 1e-10);
    EXPECT_LT(ci.lower, 0.6);
    EXPECT_GT(ci.upper, 0.6);
    EXPECT_GT(ci.lower, 0.0);
    EXPECT_LT(ci.upper, 1.0);
}

/**
 * @brief Test Wilson method performance with extreme proportions.
 * @test Verifies that Wilson method provides better bounds for extreme proportions.
 */
TEST(CIProportionTest, WilsonBetterNearBoundary) {
    // Wilson method should be more accurate for extreme proportions
    auto ci_wilson = statcpp::ci_proportion_wilson(5, 100, 0.95);

    // Wilson should give a lower bound > 0 more reliably for small proportions
    EXPECT_GT(ci_wilson.lower, 0.0);
    EXPECT_LT(ci_wilson.upper, 1.0);
}

/**
 * @brief Test CI for proportion with invalid input.
 * @test Verifies that exceptions are thrown for invalid parameters.
 */
TEST(CIProportionTest, InvalidInput) {
    EXPECT_THROW(statcpp::ci_proportion(0, 0, 0.95), std::invalid_argument);
    EXPECT_THROW(statcpp::ci_proportion(10, 5, 0.95), std::invalid_argument);  // successes > trials
}

// ============================================================================
// Confidence Interval for Variance Tests
// ============================================================================

/**
 * @brief Test confidence interval for variance with basic data.
 * @test Verifies CI calculation for variance and that interval bounds the point estimate.
 */
TEST(CIVarianceTest, BasicComputation) {
    std::vector<double> data = {2.0, 4.0, 6.0, 8.0, 10.0};
    auto ci = statcpp::ci_variance(data.begin(), data.end(), 0.95);

    double sample_var = statcpp::sample_variance(data.begin(), data.end());
    EXPECT_NEAR(ci.point_estimate, sample_var, 1e-10);
    EXPECT_LT(ci.lower, ci.point_estimate);
    EXPECT_GT(ci.upper, ci.point_estimate);
}

/**
 * @brief Test CI for variance with insufficient data.
 * @test Verifies that exception is thrown when data has fewer than 2 elements.
 */
TEST(CIVarianceTest, TooFewElements) {
    std::vector<double> data = {1.0};
    EXPECT_THROW(statcpp::ci_variance(data.begin(), data.end(), 0.95), std::invalid_argument);
}

// ============================================================================
// Confidence Interval for Difference of Means Tests
// ============================================================================

/**
 * @brief Test CI for difference of means using pooled variance.
 * @test Verifies CI calculation for mean difference with equal variance assumption.
 */
TEST(CIMeanDiffTest, PooledVariance) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {8.0, 10.0, 12.0, 14.0, 16.0};

    auto ci = statcpp::ci_mean_diff(data1.begin(), data1.end(),
                                     data2.begin(), data2.end(), 0.95);

    double diff = statcpp::mean(data1.begin(), data1.end()) -
                  statcpp::mean(data2.begin(), data2.end());
    EXPECT_NEAR(ci.point_estimate, diff, 1e-10);
    EXPECT_NEAR(ci.point_estimate, 2.0, 1e-10);  // 14 - 12 = 2
}

/**
 * @brief Test CI for difference of means using Welch method.
 * @test Verifies CI calculation for mean difference with unequal variances.
 */
TEST(CIMeanDiffTest, WelchMethod) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {5.0, 15.0, 8.0, 12.0, 20.0};  // Different variance

    auto ci = statcpp::ci_mean_diff_welch(data1.begin(), data1.end(),
                                           data2.begin(), data2.end(), 0.95);

    double diff = statcpp::mean(data1.begin(), data1.end()) -
                  statcpp::mean(data2.begin(), data2.end());
    EXPECT_NEAR(ci.point_estimate, diff, 1e-10);
}

/**
 * @brief Test CI for mean difference with insufficient data.
 * @test Verifies that exception is thrown when either sample has too few elements.
 */
TEST(CIMeanDiffTest, TooFewElements) {
    std::vector<double> data1 = {1.0};
    std::vector<double> data2 = {1.0, 2.0, 3.0};

    EXPECT_THROW(statcpp::ci_mean_diff(data1.begin(), data1.end(),
                                        data2.begin(), data2.end(), 0.95),
                 std::invalid_argument);
}

// ============================================================================
// Sample Size for MOE Tests (Proportion)
// ============================================================================

/**
 * @brief Test sample size calculation for election poll with ±3% margin.
 * @test Verifies sample size for standard election poll (95% confidence, ±3% MOE).
 */
TEST(SampleSizeForMOEProportionTest, StandardElectionPoll) {
    // Election exit poll: ±3% precision, 95% confidence level
    std::size_t n = statcpp::sample_size_for_moe_proportion(0.03, 0.95, 0.5);
    EXPECT_EQ(n, 1068);  // (1.96/0.03)² × 0.25 ≈ 1067.11 → 1068
}

/**
 * @brief Test sample size calculation for tight race with ±2% margin.
 * @test Verifies sample size for precise election poll (95% confidence, ±2% MOE).
 */
TEST(SampleSizeForMOEProportionTest, TightRace) {
    // Close election: ±2% precision, 95% confidence level
    std::size_t n = statcpp::sample_size_for_moe_proportion(0.02, 0.95, 0.5);
    EXPECT_EQ(n, 2401);  // (1.96/0.02)² × 0.25 ≈ 2401
}

/**
 * @brief Test sample size calculation for local election with ±5% margin.
 * @test Verifies sample size for local election poll (95% confidence, ±5% MOE).
 */
TEST(SampleSizeForMOEProportionTest, LocalElection) {
    // Local election: ±5% precision, 95% confidence level
    std::size_t n = statcpp::sample_size_for_moe_proportion(0.05, 0.95, 0.5);
    EXPECT_EQ(n, 385);  // (1.96/0.05)² × 0.25 ≈ 384.16 → 385
}

/**
 * @brief Test sample size calculation with prior proportion estimate.
 * @test Verifies sample size using prior estimate p=0.4 instead of worst-case 0.5.
 */
TEST(SampleSizeForMOEProportionTest, WithPriorEstimate) {
    // When prior estimate is 40%
    std::size_t n = statcpp::sample_size_for_moe_proportion(0.03, 0.95, 0.40);
    // (1.96/0.03)² × 0.4 × 0.6 ≈ 1024.43 → 1025
    EXPECT_EQ(n, 1025);
}

/**
 * @brief Test sample size calculation with 99% confidence level.
 * @test Verifies sample size for higher confidence level (99% vs 95%).
 */
TEST(SampleSizeForMOEProportionTest, HighConfidenceLevel) {
    // 99% confidence level
    std::size_t n = statcpp::sample_size_for_moe_proportion(0.03, 0.99, 0.5);
    // (2.576/0.03)² × 0.25 ≈ 1843.27 → 1844
    EXPECT_EQ(n, 1844);
}

/**
 * @brief Test sample size calculation with invalid margin of error.
 * @test Verifies that exceptions are thrown for invalid MOE values.
 */
TEST(SampleSizeForMOEProportionTest, InvalidMOE) {
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.0, 0.95, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(-0.1, 0.95, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(1.0, 0.95, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(1.5, 0.95, 0.5),
                 std::invalid_argument);
}

/**
 * @brief Test sample size calculation with invalid confidence level.
 * @test Verifies that exceptions are thrown for invalid confidence levels.
 */
TEST(SampleSizeForMOEProportionTest, InvalidConfidence) {
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 0.0, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 1.0, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, -0.1, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 1.5, 0.5),
                 std::invalid_argument);
}

/**
 * @brief Test sample size calculation with invalid proportion estimate.
 * @test Verifies that exceptions are thrown for invalid p_estimate values.
 */
TEST(SampleSizeForMOEProportionTest, InvalidPEstimate) {
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 0.95, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 0.95, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 0.95, -0.1),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_proportion(0.03, 0.95, 1.5),
                 std::invalid_argument);
}

// ============================================================================
// Sample Size for MOE Tests (Mean)
// ============================================================================

/**
 * @brief Test sample size calculation for mean with basic parameters.
 * @test Verifies sample size for ±2 precision with σ=10 at 95% confidence.
 */
TEST(SampleSizeForMOEMeanTest, BasicComputation) {
    // Estimate from population with σ=10, ±2 precision, 95% confidence level
    std::size_t n = statcpp::sample_size_for_moe_mean(2.0, 10.0, 0.95);
    // (1.96 × 10 / 2)² = 96.04 → 97
    EXPECT_EQ(n, 97);
}

/**
 * @brief Test sample size calculation for high precision requirement.
 * @test Verifies sample size for ±1 precision with σ=15 at 95% confidence.
 */
TEST(SampleSizeForMOEMeanTest, HighPrecision) {
    // σ=15, margin of error ±1
    std::size_t n = statcpp::sample_size_for_moe_mean(1.0, 15.0, 0.95);
    // (1.96 × 15)² = 864.36 → 865
    EXPECT_EQ(n, 865);
}

/**
 * @brief Test sample size calculation for mean with 99% confidence.
 * @test Verifies sample size for higher confidence level (99% vs 95%).
 */
TEST(SampleSizeForMOEMeanTest, HighConfidenceLevel) {
    // 99% confidence level
    std::size_t n = statcpp::sample_size_for_moe_mean(2.0, 10.0, 0.99);
    // (2.576 × 10 / 2)² ≈ 165.89 → 166
    EXPECT_EQ(n, 166);
}

/**
 * @brief Test sample size calculation for mean with invalid MOE.
 * @test Verifies that exceptions are thrown for non-positive MOE values.
 */
TEST(SampleSizeForMOEMeanTest, InvalidMOE) {
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(0.0, 10.0, 0.95),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(-1.0, 10.0, 0.95),
                 std::invalid_argument);
}

/**
 * @brief Test sample size calculation for mean with invalid sigma.
 * @test Verifies that exceptions are thrown for non-positive sigma values.
 */
TEST(SampleSizeForMOEMeanTest, InvalidSigma) {
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(2.0, 0.0, 0.95),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(2.0, -5.0, 0.95),
                 std::invalid_argument);
}

/**
 * @brief Test sample size calculation for mean with invalid confidence level.
 * @test Verifies that exceptions are thrown for invalid confidence levels.
 */
TEST(SampleSizeForMOEMeanTest, InvalidConfidence) {
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(2.0, 10.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::sample_size_for_moe_mean(2.0, 10.0, 1.0),
                 std::invalid_argument);
}
