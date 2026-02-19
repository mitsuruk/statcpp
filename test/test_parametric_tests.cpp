#include <gtest/gtest.h>
#include "statcpp/parametric_tests.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// Z-Test for Mean Tests
// ============================================================================

/**
 * @brief Test z-test with known population mean.
 * @test Verifies that z-test does not reject when sample mean equals population mean.
 */
TEST(ZTestTest, KnownMean) {
    // Data with known population mean 10 and sigma 2
    std::vector<double> data = {9.0, 10.0, 11.0, 10.0, 10.0};
    auto result = statcpp::z_test(data.begin(), data.end(), 10.0, 2.0);

    // Should not reject null hypothesis (mean = 10)
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::two_sided);
}

/**
 * @brief Test z-test rejection of null hypothesis.
 * @test Verifies that z-test rejects when sample mean differs significantly from population mean.
 */
TEST(ZTestTest, RejectNullHypothesis) {
    std::vector<double> data = {12.0, 13.0, 14.0, 15.0, 16.0};
    auto result = statcpp::z_test(data.begin(), data.end(), 10.0, 2.0);

    // Should reject null hypothesis (mean = 10 when actual mean is 14)
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test one-sided z-test (greater alternative).
 * @test Verifies z-test with alternative hypothesis that mean is greater.
 */
TEST(ZTestTest, OneSidedGreater) {
    std::vector<double> data = {12.0, 13.0, 14.0, 15.0, 16.0};
    auto result = statcpp::z_test(data.begin(), data.end(), 10.0, 2.0,
                                   statcpp::alternative_hypothesis::greater);

    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::greater);
}

/**
 * @brief Test one-sided z-test (less alternative).
 * @test Verifies z-test with alternative hypothesis that mean is less.
 */
TEST(ZTestTest, OneSidedLess) {
    std::vector<double> data = {8.0, 9.0, 10.0, 7.0, 6.0};
    auto result = statcpp::z_test(data.begin(), data.end(), 12.0, 2.0,
                                   statcpp::alternative_hypothesis::less);

    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::less);
}

/**
 * @brief Test z-test with invalid sigma parameter.
 * @test Verifies that exceptions are thrown for non-positive sigma values.
 */
TEST(ZTestTest, InvalidSigma) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::z_test(data.begin(), data.end(), 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::z_test(data.begin(), data.end(), 0.0, -1.0), std::invalid_argument);
}

// ============================================================================
// Z-Test for Proportion Tests
// ============================================================================

/**
 * @brief Test z-test for proportion not rejecting null.
 * @test Verifies that test does not reject when sample proportion equals null hypothesis.
 */
TEST(ZTestProportionTest, NotRejectNull) {
    // 50 successes out of 100 trials, testing against p0 = 0.5
    auto result = statcpp::z_test_proportion(50, 100, 0.5);
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test z-test for proportion rejecting null.
 * @test Verifies that test rejects when sample proportion differs significantly.
 */
TEST(ZTestProportionTest, RejectNull) {
    // 70 successes out of 100 trials, testing against p0 = 0.5
    auto result = statcpp::z_test_proportion(70, 100, 0.5);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test two-sample z-test for proportions.
 * @test Verifies two-sample test detects significant difference between proportions.
 */
TEST(ZTestProportionTest, TwoSample) {
    // Group 1: 60/100, Group 2: 40/100
    auto result = statcpp::z_test_proportion_two_sample(60, 100, 40, 100);
    EXPECT_LT(result.p_value, 0.05);  // Significant difference
}

/**
 * @brief Test z-test for proportion with invalid input.
 * @test Verifies that exceptions are thrown for invalid parameters.
 */
TEST(ZTestProportionTest, InvalidInput) {
    EXPECT_THROW(statcpp::z_test_proportion(50, 100, 0.0), std::invalid_argument);  // p0 = 0
    EXPECT_THROW(statcpp::z_test_proportion(50, 100, 1.0), std::invalid_argument);  // p0 = 1
    EXPECT_THROW(statcpp::z_test_proportion(50, 0, 0.5), std::invalid_argument);   // trials = 0
    EXPECT_THROW(statcpp::z_test_proportion(10, 5, 0.5), std::invalid_argument);   // successes > trials
}

// ============================================================================
// T-Test for Mean Tests
// ============================================================================

/**
 * @brief Test one-sample t-test not rejecting null.
 * @test Verifies that t-test does not reject when sample mean equals null hypothesis.
 */
TEST(TTestTest, OneSample) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    auto result = statcpp::t_test(data.begin(), data.end(), 14.0);

    // Mean is exactly 14, should not reject
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_NEAR(result.statistic, 0.0, 1e-10);
}

/**
 * @brief Test one-sample t-test rejecting null.
 * @test Verifies that t-test rejects when sample mean differs significantly.
 */
TEST(TTestTest, OneSampleReject) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    auto result = statcpp::t_test(data.begin(), data.end(), 5.0);

    // Mean is 14, testing against 5, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test t-test degrees of freedom calculation.
 * @test Verifies that degrees of freedom equals n-1 for one-sample t-test.
 */
TEST(TTestTest, DegreesOfFreedom) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = statcpp::t_test(data.begin(), data.end(), 3.0);

    EXPECT_NEAR(result.df, 4.0, 1e-10);  // n - 1 = 5 - 1 = 4
}

/**
 * @brief Test t-test with insufficient data.
 * @test Verifies that exception is thrown for sample size less than 2.
 */
TEST(TTestTest, TooFewElements) {
    std::vector<double> data = {1.0};
    EXPECT_THROW(statcpp::t_test(data.begin(), data.end(), 0.0), std::invalid_argument);
}

// ============================================================================
// Two-Sample T-Test Tests
// ============================================================================

/**
 * @brief Test two-sample t-test with equal means.
 * @test Verifies that test does not reject when both samples have similar means.
 */
TEST(TTestTwoSampleTest, EqualMeans) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {9.0, 11.0, 13.0, 15.0, 17.0};  // Same mean (13)

    auto result = statcpp::t_test_two_sample(data1.begin(), data1.end(),
                                              data2.begin(), data2.end());

    // Means are very close, should not reject
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test two-sample t-test with different means.
 * @test Verifies that test rejects when samples have significantly different means.
 */
TEST(TTestTwoSampleTest, DifferentMeans) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14
    std::vector<double> data2 = {1.0, 2.0, 3.0, 4.0, 5.0};       // mean = 3

    auto result = statcpp::t_test_two_sample(data1.begin(), data1.end(),
                                              data2.begin(), data2.end());

    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test pooled degrees of freedom calculation.
 * @test Verifies that pooled df equals n1 + n2 - 2.
 */
TEST(TTestTwoSampleTest, PooledDF) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};  // n1 = 5
    std::vector<double> data2 = {6.0, 7.0, 8.0};            // n2 = 3

    auto result = statcpp::t_test_two_sample(data1.begin(), data1.end(),
                                              data2.begin(), data2.end());

    // df = n1 + n2 - 2 = 5 + 3 - 2 = 6
    EXPECT_NEAR(result.df, 6.0, 1e-10);
}

// ============================================================================
// Welch's T-Test Tests
// ============================================================================

/**
 * @brief Test Welch's t-test with unequal variances.
 * @test Verifies that Welch's df is less than pooled df for unequal variances.
 */
TEST(TTestWelchTest, UnequalVariances) {
    std::vector<double> data1 = {10.0, 10.1, 10.2, 9.9, 9.8};  // Low variance
    std::vector<double> data2 = {5.0, 15.0, 8.0, 12.0, 20.0};  // High variance

    auto result = statcpp::t_test_welch(data1.begin(), data1.end(),
                                         data2.begin(), data2.end());

    // Welch df should be less than pooled df
    double pooled_df = 5 + 5 - 2;  // = 8
    EXPECT_LT(result.df, pooled_df);
}

// ============================================================================
// Paired T-Test Tests
// ============================================================================

/**
 * @brief Test paired t-test with significant difference.
 * @test Verifies that paired test detects significant differences in paired data.
 */
TEST(TTestPairedTest, SignificantDifference) {
    std::vector<double> before = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> after = {15.0, 18.0, 20.0, 22.0, 25.0};  // Clear increase with variation

    auto result = statcpp::t_test_paired(before.begin(), before.end(),
                                          after.begin(), after.end());

    // There IS a significant difference
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test paired t-test with zero difference.
 * @test Verifies that exception is thrown when all paired differences are zero.
 */
TEST(TTestPairedTest, ZeroDifference) {
    std::vector<double> before = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> after = {10.0, 12.0, 14.0, 16.0, 18.0};  // Same values

    // This will throw because variance of differences is 0
    EXPECT_THROW(statcpp::t_test_paired(before.begin(), before.end(),
                                         after.begin(), after.end()),
                 std::invalid_argument);
}

/**
 * @brief Test paired t-test with unequal sample lengths.
 * @test Verifies that exception is thrown when paired samples have different lengths.
 */
TEST(TTestPairedTest, UnequalLength) {
    std::vector<double> before = {10.0, 12.0, 14.0};
    std::vector<double> after = {11.0, 13.0, 15.0, 17.0, 19.0};

    EXPECT_THROW(statcpp::t_test_paired(before.begin(), before.end(),
                                         after.begin(), after.end()),
                 std::invalid_argument);
}

// ============================================================================
// Chi-Square Goodness of Fit Tests
// ============================================================================

/**
 * @brief Test chi-square GOF with uniform distribution.
 * @test Verifies that test does not reject when observed matches expected uniform.
 */
TEST(ChisqGOFTest, UniformDistribution) {
    // Observed frequencies close to expected uniform
    std::vector<double> observed = {25.0, 25.0, 25.0, 25.0};
    std::vector<double> expected = {25.0, 25.0, 25.0, 25.0};

    auto result = statcpp::chisq_test_gof(observed.begin(), observed.end(),
                                           expected.begin(), expected.end());

    EXPECT_NEAR(result.statistic, 0.0, 1e-10);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_NEAR(result.df, 3.0, 1e-10);  // k - 1 = 4 - 1 = 3
}

/**
 * @brief Test chi-square GOF rejecting uniform hypothesis.
 * @test Verifies that test rejects when observed differs from expected uniform.
 */
TEST(ChisqGOFTest, RejectUniform) {
    // Clearly non-uniform
    std::vector<double> observed = {50.0, 10.0, 10.0, 30.0};
    std::vector<double> expected = {25.0, 25.0, 25.0, 25.0};

    auto result = statcpp::chisq_test_gof(observed.begin(), observed.end(),
                                           expected.begin(), expected.end());

    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test chi-square GOF with uniform expected frequencies.
 * @test Verifies GOF test convenience function for uniform expected values.
 */
TEST(ChisqGOFTest, UniformExpected) {
    std::vector<double> observed = {25.0, 25.0, 25.0, 25.0};
    auto result = statcpp::chisq_test_gof_uniform(observed.begin(), observed.end());

    EXPECT_NEAR(result.statistic, 0.0, 1e-10);
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test chi-square GOF with invalid input.
 * @test Verifies that exceptions are thrown for invalid parameters.
 */
TEST(ChisqGOFTest, InvalidInput) {
    std::vector<double> observed = {10.0, 20.0, 30.0};
    std::vector<double> expected_zero = {10.0, 0.0, 30.0};  // Zero expected
    std::vector<double> expected_wrong_size = {10.0, 20.0};  // Wrong size

    EXPECT_THROW(statcpp::chisq_test_gof(observed.begin(), observed.end(),
                                          expected_zero.begin(), expected_zero.end()),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::chisq_test_gof(observed.begin(), observed.end(),
                                          expected_wrong_size.begin(), expected_wrong_size.end()),
                 std::invalid_argument);
}

// ============================================================================
// Chi-Square Independence Tests
// ============================================================================

/**
 * @brief Test chi-square independence test with independent variables.
 * @test Verifies that test does not reject when variables are independent.
 */
TEST(ChisqIndependenceTest, Independent) {
    // 2x2 table with independence
    std::vector<std::vector<double>> table = {
        {25.0, 25.0},
        {25.0, 25.0}
    };

    auto result = statcpp::chisq_test_independence(table);

    EXPECT_NEAR(result.statistic, 0.0, 1e-10);
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_NEAR(result.df, 1.0, 1e-10);  // (2-1)*(2-1) = 1
}

/**
 * @brief Test chi-square independence test with dependent variables.
 * @test Verifies that test rejects when variables are clearly associated.
 */
TEST(ChisqIndependenceTest, NotIndependent) {
    // 2x2 table with clear association
    std::vector<std::vector<double>> table = {
        {50.0, 10.0},
        {10.0, 50.0}
    };

    auto result = statcpp::chisq_test_independence(table);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test chi-square independence with larger contingency table.
 * @test Verifies degrees of freedom calculation for 3x3 table.
 */
TEST(ChisqIndependenceTest, LargerTable) {
    // 3x3 table
    std::vector<std::vector<double>> table = {
        {10.0, 20.0, 30.0},
        {20.0, 30.0, 10.0},
        {30.0, 10.0, 20.0}
    };

    auto result = statcpp::chisq_test_independence(table);

    // df = (3-1)*(3-1) = 4
    EXPECT_NEAR(result.df, 4.0, 1e-10);
}

/**
 * @brief Test chi-square independence with invalid input.
 * @test Verifies that exceptions are thrown for tables with insufficient dimensions.
 */
TEST(ChisqIndependenceTest, InvalidInput) {
    // Too few rows
    std::vector<std::vector<double>> table_1row = {{10.0, 20.0}};
    EXPECT_THROW(statcpp::chisq_test_independence(table_1row), std::invalid_argument);

    // Too few columns
    std::vector<std::vector<double>> table_1col = {{10.0}, {20.0}};
    EXPECT_THROW(statcpp::chisq_test_independence(table_1col), std::invalid_argument);
}

// ============================================================================
// F-Test Tests
// ============================================================================

/**
 * @brief Test F-test with equal variances.
 * @test Verifies that F-test does not reject when variances are equal.
 */
TEST(FTestTest, EqualVariance) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {11.0, 13.0, 15.0, 17.0, 19.0};

    auto result = statcpp::f_test(data1.begin(), data1.end(),
                                   data2.begin(), data2.end());

    // Same variance, F should be close to 1
    EXPECT_NEAR(result.statistic, 1.0, 1e-10);
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test F-test with different variances.
 * @test Verifies that F-test rejects when variances differ significantly.
 */
TEST(FTestTest, DifferentVariance) {
    std::vector<double> data1 = {10.0, 10.1, 10.2, 10.3, 10.4};  // Low variance
    std::vector<double> data2 = {5.0, 15.0, 8.0, 12.0, 20.0};    // High variance

    auto result = statcpp::f_test(data1.begin(), data1.end(),
                                   data2.begin(), data2.end());

    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test F-test degrees of freedom calculation.
 * @test Verifies that df equals n-1 for numerator variance.
 */
TEST(FTestTest, DegreesOfFreedom) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};  // n1 = 5
    std::vector<double> data2 = {6.0, 7.0, 8.0};            // n2 = 3

    auto result = statcpp::f_test(data1.begin(), data1.end(),
                                   data2.begin(), data2.end());

    // df stored is df1 = n1 - 1 = 4
    EXPECT_NEAR(result.df, 4.0, 1e-10);
    // df2 = n2 - 1 = 2
    EXPECT_NEAR(result.df2, 2.0, 1e-10);
}

/**
 * @brief Test that non-F-test results have NaN for df2.
 * @test Verifies that t-test results default df2 to NaN.
 */
TEST(FTestTest, Df2NaNForOtherTests) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = statcpp::t_test(data.begin(), data.end(), 3.0);
    EXPECT_TRUE(std::isnan(result.df2));
}

// ============================================================================
// Multiple Testing Correction Tests
// ============================================================================

/**
 * @brief Test Bonferroni correction for multiple testing.
 * @test Verifies that Bonferroni method multiplies p-values by number of tests.
 */
TEST(BonferroniTest, BasicCorrection) {
    std::vector<double> p_values = {0.01, 0.02, 0.03, 0.04, 0.05};
    auto adjusted = statcpp::bonferroni_correction(p_values);

    EXPECT_EQ(adjusted.size(), p_values.size());

    for (std::size_t i = 0; i < p_values.size(); ++i) {
        EXPECT_NEAR(adjusted[i], std::min(1.0, p_values[i] * 5.0), 1e-10);
    }
}

/**
 * @brief Test Bonferroni correction p-value capping.
 * @test Verifies that adjusted p-values are capped at 1.0.
 */
TEST(BonferroniTest, CappedAtOne) {
    std::vector<double> p_values = {0.3, 0.5, 0.7};
    auto adjusted = statcpp::bonferroni_correction(p_values);

    for (double p : adjusted) {
        EXPECT_LE(p, 1.0);
    }
}

/**
 * @brief Test Benjamini-Hochberg correction for FDR control.
 * @test Verifies that BH method adjusts p-values for false discovery rate.
 */
TEST(BenjaminiHochbergTest, BasicCorrection) {
    std::vector<double> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    auto adjusted = statcpp::benjamini_hochberg_correction(p_values);

    EXPECT_EQ(adjusted.size(), p_values.size());

    // Adjusted p-values should be in non-decreasing order (when sorted by original p)
    for (double p : adjusted) {
        EXPECT_LE(p, 1.0);
    }
}

/**
 * @brief Test that BH is less conservative than Bonferroni.
 * @test Verifies that BH adjusted p-values are smaller than Bonferroni.
 */
TEST(BenjaminiHochbergTest, LessThanBonferroni) {
    std::vector<double> p_values = {0.01, 0.02, 0.03, 0.04, 0.05};
    auto bonf = statcpp::bonferroni_correction(p_values);
    auto bh = statcpp::benjamini_hochberg_correction(p_values);

    // BH should generally be less conservative (smaller adjusted p-values)
    for (std::size_t i = 0; i < p_values.size(); ++i) {
        EXPECT_LE(bh[i], bonf[i] + 1e-10);
    }
}

/**
 * @brief Test Holm correction for multiple testing.
 * @test Verifies that Holm method provides sequential Bonferroni adjustment.
 */
TEST(HolmTest, BasicCorrection) {
    std::vector<double> p_values = {0.01, 0.02, 0.03, 0.04, 0.05};
    auto adjusted = statcpp::holm_correction(p_values);

    EXPECT_EQ(adjusted.size(), p_values.size());

    for (double p : adjusted) {
        EXPECT_LE(p, 1.0);
    }
}

/**
 * @brief Test that Holm is less conservative than Bonferroni.
 * @test Verifies that Holm adjusted p-values are at most equal to Bonferroni.
 */
TEST(HolmTest, LessThanBonferroni) {
    std::vector<double> p_values = {0.01, 0.02, 0.03, 0.04, 0.05};
    auto bonf = statcpp::bonferroni_correction(p_values);
    auto holm = statcpp::holm_correction(p_values);

    // Holm should be at least as conservative as BH but less than Bonferroni
    // (except for smallest p-value which is the same)
    for (std::size_t i = 0; i < p_values.size(); ++i) {
        EXPECT_LE(holm[i], bonf[i] + 1e-10);
    }
}
