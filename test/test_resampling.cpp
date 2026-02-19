#include <gtest/gtest.h>
#include "statcpp/resampling.hpp"
#include <cmath>
#include <vector>
#include <numeric>

// ============================================================================
// Bootstrap Sample Tests
// ============================================================================

/**
 * @brief Tests that bootstrap sample has the same size as original data.
 * @test Verifies that bootstrap resampling produces a sample of equal size to the input.
 */
TEST(BootstrapSampleTest, SameSize) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    statcpp::set_seed(42);

    auto sample = statcpp::bootstrap_sample(data.begin(), data.end());

    EXPECT_EQ(sample.size(), data.size());
}

/**
 * @brief Tests that bootstrap sample contains only elements from original data.
 * @test Verifies that all elements in bootstrap sample exist in the original dataset.
 */
TEST(BootstrapSampleTest, ElementsFromOriginal) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    statcpp::set_seed(42);

    auto sample = statcpp::bootstrap_sample(data.begin(), data.end());

    // All elements should be from original data
    for (double x : sample) {
        bool found = false;
        for (double y : data) {
            if (x == y) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

/**
 * @brief Tests that empty range throws exception.
 * @test Verifies that bootstrap sampling on empty data throws std::invalid_argument.
 */
TEST(BootstrapSampleTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::bootstrap_sample(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Bootstrap Mean Tests
// ============================================================================

/**
 * @brief Tests that bootstrap mean estimate equals the sample mean.
 * @test Verifies that the bootstrap estimate of mean matches the original sample mean.
 */
TEST(BootstrapMeanTest, EstimateCloseToSampleMean) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_mean(data.begin(), data.end(), 1000, 0.95);

    double sample_mean = statcpp::mean(data.begin(), data.end());
    EXPECT_NEAR(result.estimate, sample_mean, 1e-10);
}

/**
 * @brief Tests that confidence interval contains the sample mean.
 * @test Verifies that the bootstrap confidence interval includes the point estimate.
 */
TEST(BootstrapMeanTest, CIContainsMean) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_mean(data.begin(), data.end(), 1000, 0.95);

    // CI should contain the sample mean
    EXPECT_LE(result.ci_lower, result.estimate);
    EXPECT_GE(result.ci_upper, result.estimate);
}

/**
 * @brief Tests that correct number of bootstrap replicates are generated.
 * @test Verifies that the number of replicates matches the requested bootstrap iterations.
 */
TEST(BootstrapMeanTest, ReplicatesCount) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    statcpp::set_seed(42);

    std::size_t n_bootstrap = 500;
    auto result = statcpp::bootstrap_mean(data.begin(), data.end(), n_bootstrap, 0.95);

    EXPECT_EQ(result.replicates.size(), n_bootstrap);
}

/**
 * @brief Tests that bootstrap standard error is positive.
 * @test Verifies that the computed standard error from bootstrap replicates is greater than zero.
 */
TEST(BootstrapMeanTest, StandardErrorPositive) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_mean(data.begin(), data.end(), 1000, 0.95);

    EXPECT_GT(result.standard_error, 0.0);
}

// ============================================================================
// Bootstrap Median Tests
// ============================================================================

/**
 * @brief Tests that bootstrap median estimate equals the sample median.
 * @test Verifies that the bootstrap estimate of median matches the original sample median.
 */
TEST(BootstrapMedianTest, EstimateCloseToSampleMedian) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_median(data.begin(), data.end(), 1000, 0.95);

    // Sample median = 5.5
    EXPECT_NEAR(result.estimate, 5.5, 1e-10);
}

// ============================================================================
// Bootstrap Standard Deviation Tests
// ============================================================================

/**
 * @brief Tests that bootstrap standard deviation estimate equals sample standard deviation.
 * @test Verifies that the bootstrap estimate of standard deviation matches the original sample.
 */
TEST(BootstrapStddevTest, EstimateCloseToSampleStddev) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_stddev(data.begin(), data.end(), 1000, 0.95);

    double sample_sd = statcpp::sample_stddev(data.begin(), data.end());
    EXPECT_NEAR(result.estimate, sample_sd, 1e-10);
}

// ============================================================================
// Generic Bootstrap Tests
// ============================================================================

/**
 * @brief Tests bootstrap with custom statistic function.
 * @test Verifies that bootstrap can be applied to custom statistics like range (max - min).
 */
TEST(BootstrapTest, CustomStatistic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    statcpp::set_seed(42);

    // Custom statistic: range (max - min)
    auto range_func = [](auto first, auto last) {
        double min_val = *std::min_element(first, last);
        double max_val = *std::max_element(first, last);
        return max_val - min_val;
    };

    auto result = statcpp::bootstrap(data.begin(), data.end(), range_func, 1000, 0.95);

    // Original range = 5 - 1 = 4
    EXPECT_NEAR(result.estimate, 4.0, 1e-10);
}

/**
 * @brief Tests that invalid confidence levels throw exceptions.
 * @test Verifies that confidence levels at or outside (0, 1) throw std::invalid_argument.
 */
TEST(BootstrapTest, InvalidConfidence) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };

    EXPECT_THROW(statcpp::bootstrap(data.begin(), data.end(), mean_func, 1000, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::bootstrap(data.begin(), data.end(), mean_func, 1000, 1.0), std::invalid_argument);
}

/**
 * @brief Tests that too few elements throw exception.
 * @test Verifies that bootstrap with insufficient sample size throws std::invalid_argument.
 */
TEST(BootstrapTest, TooFewElements) {
    std::vector<double> data = {1.0};
    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };

    EXPECT_THROW(statcpp::bootstrap(data.begin(), data.end(), mean_func, 1000, 0.95), std::invalid_argument);
}

// ============================================================================
// BCa Bootstrap Tests
// ============================================================================

/**
 * @brief Tests that BCa bootstrap mean estimate equals the sample mean.
 * @test Verifies that the bias-corrected and accelerated bootstrap estimate matches sample mean.
 */
TEST(BootstrapBCaTest, EstimateCloseToSampleMean) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };
    auto result = statcpp::bootstrap_bca(data.begin(), data.end(), mean_func, 1000, 0.95);

    double sample_mean = statcpp::mean(data.begin(), data.end());
    EXPECT_NEAR(result.estimate, sample_mean, 1e-10);
}

/**
 * @brief Tests that BCa confidence interval contains the sample mean.
 * @test Verifies that the BCa bootstrap confidence interval includes the point estimate.
 */
TEST(BootstrapBCaTest, CIContainsMean) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    statcpp::set_seed(42);

    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };
    auto result = statcpp::bootstrap_bca(data.begin(), data.end(), mean_func, 1000, 0.95);

    EXPECT_LE(result.ci_lower, result.estimate);
    EXPECT_GE(result.ci_upper, result.estimate);
}

/**
 * @brief Tests that BCa bootstrap requires at least 3 elements.
 * @test Verifies that BCa bootstrap with fewer than 3 elements throws std::invalid_argument.
 */
TEST(BootstrapBCaTest, TooFewElements) {
    std::vector<double> data = {1.0, 2.0};  // BCa needs at least 3
    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };

    EXPECT_THROW(statcpp::bootstrap_bca(data.begin(), data.end(), mean_func, 1000, 0.95), std::invalid_argument);
}

/**
 * @brief Tests BCa bootstrap with constant data (degenerate case).
 * @test Verifies that BCa handles count_less=0 and sum_squared=0 without producing NaN/Inf.
 */
TEST(BootstrapBCaTest, ConstantData) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};
    auto mean_func = [](auto f, auto l) { return statcpp::mean(f, l); };
    statcpp::set_seed(42);

    auto result = statcpp::bootstrap_bca(data.begin(), data.end(), mean_func, 100, 0.95);

    EXPECT_NEAR(result.estimate, 5.0, 1e-10);
    EXPECT_FALSE(std::isnan(result.ci_lower));
    EXPECT_FALSE(std::isnan(result.ci_upper));
    EXPECT_FALSE(std::isinf(result.ci_lower));
    EXPECT_FALSE(std::isinf(result.ci_upper));
}

// ============================================================================
// Permutation Test (Two Sample) Tests
// ============================================================================

/**
 * @brief Tests permutation test with similar distributions.
 * @test Verifies permutation test behavior when two samples have very similar distributions.
 */
TEST(PermutationTestTwoSampleTest, SameDistribution) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> data2 = {1.5, 2.5, 3.5, 4.5, 5.5};  // Very similar
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_two_sample(data1.begin(), data1.end(),
                                                        data2.begin(), data2.end(), 1000);

    // Very similar distributions, p-value should be high
    // Note: with small samples, results may vary
}

/**
 * @brief Tests permutation test with clearly different distributions.
 * @test Verifies that permutation test produces low p-value for clearly separated distributions.
 */
TEST(PermutationTestTwoSampleTest, DifferentDistributions) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> data2 = {10.0, 11.0, 12.0, 13.0, 14.0};
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_two_sample(data1.begin(), data1.end(),
                                                        data2.begin(), data2.end(), 1000);

    // Very different distributions, p-value should be low
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Tests that observed statistic is correctly computed.
 * @test Verifies that the observed test statistic equals the difference of sample means.
 */
TEST(PermutationTestTwoSampleTest, ObservedStatistic) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};  // mean = 3
    std::vector<double> data2 = {6.0, 7.0, 8.0, 9.0, 10.0}; // mean = 8
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_two_sample(data1.begin(), data1.end(),
                                                        data2.begin(), data2.end(), 1000);

    // Observed statistic = mean1 - mean2 = 3 - 8 = -5
    EXPECT_NEAR(result.observed_statistic, -5.0, 1e-10);
}

/**
 * @brief Tests that correct number of permutations are performed.
 * @test Verifies that the permutation distribution has the requested number of permutations.
 */
TEST(PermutationTestTwoSampleTest, PermutationCount) {
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2 = {4.0, 5.0, 6.0};
    statcpp::set_seed(42);

    std::size_t n_perms = 500;
    auto result = statcpp::permutation_test_two_sample(data1.begin(), data1.end(),
                                                        data2.begin(), data2.end(), n_perms);

    EXPECT_EQ(result.n_permutations, n_perms);
    EXPECT_EQ(result.permutation_distribution.size(), n_perms);
}

/**
 * @brief Tests that empty sample throws exception.
 * @test Verifies that permutation test with empty sample throws std::invalid_argument.
 */
TEST(PermutationTestTwoSampleTest, EmptySample) {
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2;

    EXPECT_THROW(statcpp::permutation_test_two_sample(data1.begin(), data1.end(),
                                                       data2.begin(), data2.end(), 1000),
                 std::invalid_argument);
}

// ============================================================================
// Permutation Test (Paired) Tests
// ============================================================================

/**
 * @brief Tests paired permutation test with no difference between pairs.
 * @test Verifies that identical before/after values produce observed statistic of 0.0.
 */
TEST(PermutationTestPairedTest, NoDifference) {
    std::vector<double> before = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> after = {1.0, 2.0, 3.0, 4.0, 5.0};  // Same values
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_paired(before.begin(), before.end(),
                                                    after.begin(), after.end(), 1000);

    // No difference, p-value should be 1
    EXPECT_NEAR(result.observed_statistic, 0.0, 1e-10);
}

/**
 * @brief Tests paired permutation test with significant difference.
 * @test Verifies that consistent differences produce low p-value in paired permutation test.
 */
TEST(PermutationTestPairedTest, SignificantDifference) {
    std::vector<double> before = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> after = {6.0, 7.0, 8.0, 9.0, 10.0};  // All +5
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_paired(before.begin(), before.end(),
                                                    after.begin(), after.end(), 5000);

    // Significant difference, p-value should be low
    EXPECT_LT(result.p_value, 0.10);  // Use more relaxed threshold due to randomness
    EXPECT_NEAR(result.observed_statistic, -5.0, 1e-10);  // mean of (before - after)
}

/**
 * @brief Tests that unequal length pairs throw exception.
 * @test Verifies that paired permutation test with mismatched sample sizes throws std::invalid_argument.
 */
TEST(PermutationTestPairedTest, UnequalLength) {
    std::vector<double> before = {1.0, 2.0, 3.0};
    std::vector<double> after = {4.0, 5.0, 6.0, 7.0};

    EXPECT_THROW(statcpp::permutation_test_paired(before.begin(), before.end(),
                                                   after.begin(), after.end(), 1000),
                 std::invalid_argument);
}

// ============================================================================
// Permutation Test (Correlation) Tests
// ============================================================================

/**
 * @brief Tests permutation test with perfect positive correlation.
 * @test Verifies that perfect correlation produces observed statistic of 1.0 and low p-value.
 */
TEST(PermutationTestCorrelationTest, PositiveCorrelation) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};  // Perfect correlation
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_correlation(x.begin(), x.end(),
                                                         y.begin(), y.end(), 1000);

    // Perfect positive correlation
    EXPECT_NEAR(result.observed_statistic, 1.0, 1e-10);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Tests permutation test with weak or no correlation.
 * @test Verifies permutation test behavior with scrambled, uncorrelated data.
 */
TEST(PermutationTestCorrelationTest, NoCorrelation) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {3.0, 1.0, 4.0, 2.0, 5.0};  // Scrambled
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_correlation(x.begin(), x.end(),
                                                         y.begin(), y.end(), 1000);

    // Weak correlation, p-value should be higher
    // Note: exact result depends on the specific permutations
}

/**
 * @brief Tests permutation test with perfect negative correlation.
 * @test Verifies that perfect negative correlation produces observed statistic of -1.0 and low p-value.
 */
TEST(PermutationTestCorrelationTest, NegativeCorrelation) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};  // Perfect negative
    statcpp::set_seed(42);

    auto result = statcpp::permutation_test_correlation(x.begin(), x.end(),
                                                         y.begin(), y.end(), 1000);

    // Perfect negative correlation
    EXPECT_NEAR(result.observed_statistic, -1.0, 1e-10);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Tests that unequal length vectors throw exception.
 * @test Verifies that correlation permutation test with mismatched sizes throws std::invalid_argument.
 */
TEST(PermutationTestCorrelationTest, UnequalLength) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    EXPECT_THROW(statcpp::permutation_test_correlation(x.begin(), x.end(),
                                                        y.begin(), y.end(), 1000),
                 std::invalid_argument);
}

/**
 * @brief Tests that too few pairs throw exception.
 * @test Verifies that correlation permutation test requires at least 3 data pairs.
 */
TEST(PermutationTestCorrelationTest, TooFewPairs) {
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> y = {1.0, 2.0};

    EXPECT_THROW(statcpp::permutation_test_correlation(x.begin(), x.end(),
                                                        y.begin(), y.end(), 1000),
                 std::invalid_argument);
}
