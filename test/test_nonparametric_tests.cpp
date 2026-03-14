#include <gtest/gtest.h>
#include "statcpp/nonparametric_tests.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// Rank Computation Helper Tests
// ============================================================================

/**
 * @brief Test rank computation without ties.
 * @test Verifies that ranks are computed correctly when all values are unique.
 */
TEST(ComputeRanksTest, NoTies) {
    std::vector<double> data = {3.0, 1.0, 4.0, 1.5, 2.0};
    auto ranks = statcpp::compute_ranks_with_ties(data.begin(), data.end());

    // Expected ranks: 1.0->1, 1.5->2, 2.0->3, 3.0->4, 4.0->5
    // Original order: 3.0(4), 1.0(1), 4.0(5), 1.5(2), 2.0(3)
    EXPECT_NEAR(ranks[0], 4.0, 1e-10);  // 3.0
    EXPECT_NEAR(ranks[1], 1.0, 1e-10);  // 1.0
    EXPECT_NEAR(ranks[2], 5.0, 1e-10);  // 4.0
    EXPECT_NEAR(ranks[3], 2.0, 1e-10);  // 1.5
    EXPECT_NEAR(ranks[4], 3.0, 1e-10);  // 2.0
}

/**
 * @brief Test rank computation with tied values.
 * @test Verifies that average ranks are assigned to tied values.
 */
TEST(ComputeRanksTest, WithTies) {
    std::vector<double> data = {3.0, 1.0, 3.0, 1.0, 2.0};
    auto ranks = statcpp::compute_ranks_with_ties(data.begin(), data.end());

    // Values sorted: 1.0, 1.0, 2.0, 3.0, 3.0
    // Ranks: 1.0->1.5 (avg of 1,2), 2.0->3, 3.0->4.5 (avg of 4,5)
    EXPECT_NEAR(ranks[0], 4.5, 1e-10);  // 3.0
    EXPECT_NEAR(ranks[1], 1.5, 1e-10);  // 1.0
    EXPECT_NEAR(ranks[2], 4.5, 1e-10);  // 3.0
    EXPECT_NEAR(ranks[3], 1.5, 1e-10);  // 1.0
    EXPECT_NEAR(ranks[4], 3.0, 1e-10);  // 2.0
}

// ============================================================================
// Shapiro-Wilk Test Tests
// ============================================================================

/**
 * @brief Test Shapiro-Wilk test with normal data.
 * @test Verifies that W statistic is close to 1 for normally distributed data.
 */
TEST(ShapiroWilkTest, NormalData) {
    // Data from a normal distribution
    std::vector<double> normal_data = {
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.2, -0.3, 0.8
    };

    auto result = statcpp::shapiro_wilk_test(normal_data.begin(), normal_data.end());

    // W statistic should be in [0, 1]
    EXPECT_GT(result.statistic, 0.0);
    EXPECT_LE(result.statistic, 1.0);
    // For roughly normal data, W should be close to 1
    EXPECT_GT(result.statistic, 0.8);
    // p-value should be valid
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

/**
 * @brief Test Shapiro-Wilk test with non-normal data.
 * @test Verifies that W statistic is lower for skewed/non-normal data.
 */
TEST(ShapiroWilkTest, NonNormalData) {
    // Highly skewed data
    std::vector<double> skewed_data = {
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.0, 20.0, 50.0
    };

    auto result = statcpp::shapiro_wilk_test(skewed_data.begin(), skewed_data.end());

    // W statistic should be in [0, 1]
    EXPECT_GT(result.statistic, 0.0);
    EXPECT_LE(result.statistic, 1.0);
    // For non-normal data, W should be lower
    EXPECT_LT(result.statistic, 0.95);
    // p-value should be valid
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

/**
 * @brief Test Shapiro-Wilk test with insufficient data.
 * @test Verifies that exception is thrown for sample size less than 3.
 */
TEST(ShapiroWilkTest, TooFewElements) {
    std::vector<double> data = {1.0, 2.0};
    EXPECT_THROW(statcpp::shapiro_wilk_test(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Kolmogorov-Smirnov Test Tests
// ============================================================================

// Suppress deprecation warnings for legacy ks_test_normal tests
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/**
 * @brief Test Kolmogorov-Smirnov test for normality (deprecated alias).
 * @test Verifies that KS test D statistic is computed for normal data.
 */
TEST(KSTestNormalTest, NormalData) {
    std::vector<double> normal_data = {
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.2, -0.3, 0.8
    };

    auto result = statcpp::ks_test_normal(normal_data.begin(), normal_data.end());

    EXPECT_GT(result.statistic, 0.0);  // D > 0
    EXPECT_LE(result.statistic, 1.0);  // D <= 1
}

/**
 * @brief Test KS test with insufficient data (deprecated alias).
 * @test Verifies that exception is thrown for empty sample.
 */
TEST(KSTestNormalTest, TooFewElements) {
    std::vector<double> data = {1.0};
    EXPECT_THROW(statcpp::ks_test_normal(data.begin(), data.end()), std::invalid_argument);
}

#pragma GCC diagnostic pop

// ============================================================================
// Lilliefors Test (new name) Tests
// ============================================================================

/**
 * @brief Test Lilliefors test (renamed from ks_test_normal) with normal data.
 * @test Verifies that lilliefors_test works correctly with the new function name.
 */
TEST(LillieforsTestTest, NormalData) {
    std::vector<double> normal_data = {
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.2, -0.3, 0.8
    };

    auto result = statcpp::lilliefors_test(normal_data.begin(), normal_data.end());

    EXPECT_GT(result.statistic, 0.0);
    EXPECT_LE(result.statistic, 1.0);
}

/**
 * @brief Test that deprecated ks_test_normal still works and produces identical results.
 * @test Verifies backward compatibility of the deprecated alias.
 */
TEST(LillieforsTestTest, DeprecatedAliasProducesSameResult) {
    std::vector<double> data = {
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.2, -0.3, 0.8
    };

    auto result_new = statcpp::lilliefors_test(data.begin(), data.end());

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto result_old = statcpp::ks_test_normal(data.begin(), data.end());
    #pragma GCC diagnostic pop

    EXPECT_DOUBLE_EQ(result_new.statistic, result_old.statistic);
    EXPECT_DOUBLE_EQ(result_new.p_value, result_old.p_value);
}

/**
 * @brief Test Lilliefors test with insufficient data.
 * @test Verifies that exception is thrown for insufficient sample.
 */
TEST(LillieforsTestTest, TooFewElements) {
    std::vector<double> data = {1.0};
    EXPECT_THROW(statcpp::lilliefors_test(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Levene's Test Tests
// ============================================================================

/**
 * @brief Test Levene's test with equal variances.
 * @test Verifies that Levene's test does not reject when variances are equal.
 */
TEST(LeveneTest, EqualVariances) {
    std::vector<std::vector<double>> groups = {
        {10.0, 12.0, 14.0, 16.0, 18.0},
        {11.0, 13.0, 15.0, 17.0, 19.0},
        {12.0, 14.0, 16.0, 18.0, 20.0}
    };

    auto result = statcpp::levene_test(groups);

    // Equal variances, should not reject
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test Levene's test with unequal variances.
 * @test Verifies that Levene's test rejects when variances differ significantly.
 */
TEST(LeveneTest, UnequalVariances) {
    std::vector<std::vector<double>> groups = {
        {10.0, 10.1, 10.2, 10.3, 10.4},  // Low variance
        {5.0, 15.0, 8.0, 12.0, 20.0}      // High variance
    };

    auto result = statcpp::levene_test(groups);

    // Unequal variances, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Levene's test degrees of freedom calculation.
 * @test Verifies that df equals k-1 where k is the number of groups.
 */
TEST(LeveneTest, DegreesOfFreedom) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},  // n1 = 5
        {6.0, 7.0, 8.0},            // n2 = 3
        {9.0, 10.0, 11.0, 12.0}     // n3 = 4
    };

    auto result = statcpp::levene_test(groups);

    // df1 = k - 1 = 3 - 1 = 2
    EXPECT_NEAR(result.df, 2.0, 1e-10);
}

/**
 * @brief Test Levene's test with insufficient groups.
 * @test Verifies that exception is thrown when fewer than 2 groups are provided.
 */
TEST(LeveneTest, TooFewGroups) {
    std::vector<std::vector<double>> groups = {{1.0, 2.0, 3.0}};
    EXPECT_THROW(statcpp::levene_test(groups), std::invalid_argument);
}

// ============================================================================
// Bartlett's Test Tests
// ============================================================================

/**
 * @brief Test Bartlett's test with equal variances.
 * @test Verifies that Bartlett's test does not reject when variances are equal.
 */
TEST(BartlettTest, EqualVariances) {
    std::vector<std::vector<double>> groups = {
        {10.0, 12.0, 14.0, 16.0, 18.0},
        {11.0, 13.0, 15.0, 17.0, 19.0}
    };

    auto result = statcpp::bartlett_test(groups);

    // Equal variances, should not reject
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test Bartlett's test with unequal variances.
 * @test Verifies that Bartlett's test rejects when variances differ significantly.
 */
TEST(BartlettTest, UnequalVariances) {
    std::vector<std::vector<double>> groups = {
        {10.0, 10.1, 10.2, 10.3, 10.4},  // Low variance
        {5.0, 15.0, 8.0, 12.0, 20.0}      // High variance
    };

    auto result = statcpp::bartlett_test(groups);

    // Unequal variances, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Bartlett's test with insufficient groups.
 * @test Verifies that exception is thrown when fewer than 2 groups are provided.
 */
TEST(BartlettTest, TooFewGroups) {
    std::vector<std::vector<double>> groups = {{1.0, 2.0, 3.0}};
    EXPECT_THROW(statcpp::bartlett_test(groups), std::invalid_argument);
}

// ============================================================================
// Wilcoxon Signed-Rank Test Tests
// ============================================================================

/**
 * @brief Test Wilcoxon signed-rank test with symmetric data.
 * @test Verifies that test does not reject when data is symmetric around zero.
 */
TEST(WilcoxonSignedRankTest, SymmetricAroundZero) {
    std::vector<double> data = {-2.0, -1.0, 0.5, 1.0, 2.0};
    auto result = statcpp::wilcoxon_signed_rank_test(data.begin(), data.end(), 0.0);

    // Symmetric around 0, should not reject
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test Wilcoxon signed-rank test with non-symmetric data.
 * @test Verifies that test rejects when data is not symmetric around zero.
 */
TEST(WilcoxonSignedRankTest, NotSymmetricAroundZero) {
    std::vector<double> data = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto result = statcpp::wilcoxon_signed_rank_test(data.begin(), data.end(), 0.0);

    // All positive, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Wilcoxon signed-rank test with custom location parameter.
 * @test Verifies test with non-zero hypothesized median.
 */
TEST(WilcoxonSignedRankTest, CustomMu0) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    auto result = statcpp::wilcoxon_signed_rank_test(data.begin(), data.end(), 14.0);

    // Symmetric around 14, should not reject
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test one-sided Wilcoxon signed-rank test (greater alternative).
 * @test Verifies test with alternative hypothesis that median is greater.
 */
TEST(WilcoxonSignedRankTest, OneSidedGreater) {
    std::vector<double> data = {5.0, 6.0, 7.0, 8.0, 9.0};
    auto result = statcpp::wilcoxon_signed_rank_test(data.begin(), data.end(), 0.0,
                                                      statcpp::alternative_hypothesis::greater);

    EXPECT_LT(result.p_value, 0.05);
    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::greater);
}

// ============================================================================
// Mann-Whitney U Test Tests
// ============================================================================

/**
 * @brief Test Mann-Whitney U test with similar distributions.
 * @test Verifies that p-value is valid for slightly shifted distributions.
 */
TEST(MannWhitneyUTest, SimilarDistributions) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> data2 = {1.5, 2.5, 3.5, 4.5, 5.5};  // Shifted by 0.5

    auto result = statcpp::mann_whitney_u_test(data1.begin(), data1.end(),
                                                data2.begin(), data2.end());

    // p-value should be valid
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

/**
 * @brief Test Mann-Whitney U test with clearly different distributions.
 * @test Verifies that test rejects when distributions differ significantly.
 */
TEST(MannWhitneyUTest, ClearlyDifferent) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> data2 = {10.0, 11.0, 12.0, 13.0, 14.0};

    auto result = statcpp::mann_whitney_u_test(data1.begin(), data1.end(),
                                                data2.begin(), data2.end());

    // Clearly different, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test one-sided Mann-Whitney U test (greater alternative).
 * @test Verifies test with alternative hypothesis that first sample is greater.
 */
TEST(MannWhitneyUTest, OneSidedGreater) {
    std::vector<double> data1 = {10.0, 11.0, 12.0, 13.0, 14.0};
    std::vector<double> data2 = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = statcpp::mann_whitney_u_test(data1.begin(), data1.end(),
                                                data2.begin(), data2.end(),
                                                statcpp::alternative_hypothesis::greater);

    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Mann-Whitney U test with insufficient data.
 * @test Verifies that exception is thrown when either sample is too small.
 */
TEST(MannWhitneyUTest, TooFewElements) {
    std::vector<double> data1 = {1.0};
    std::vector<double> data2 = {2.0, 3.0, 4.0};

    EXPECT_THROW(statcpp::mann_whitney_u_test(data1.begin(), data1.end(),
                                               data2.begin(), data2.end()),
                 std::invalid_argument);
}

// ============================================================================
// Kruskal-Wallis Test Tests
// ============================================================================

/**
 * @brief Test Kruskal-Wallis test with similar groups.
 * @test Verifies that H statistic and p-value are valid for similar groups.
 */
TEST(KruskalWallisTest, SimilarGroups) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.5, 2.5, 3.5, 4.5, 5.5},
        {2.0, 3.0, 4.0, 5.0, 6.0}
    };

    auto result = statcpp::kruskal_wallis_test(groups);

    // p-value should be valid
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    // H statistic should be non-negative
    EXPECT_GE(result.statistic, 0.0);
}

/**
 * @brief Test Kruskal-Wallis test with clearly different groups.
 * @test Verifies that test rejects when groups differ significantly.
 */
TEST(KruskalWallisTest, ClearlyDifferent) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {20.0, 21.0, 22.0, 23.0, 24.0}
    };

    auto result = statcpp::kruskal_wallis_test(groups);

    // Clearly different, should reject
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Kruskal-Wallis degrees of freedom calculation.
 * @test Verifies that df equals k-1 where k is the number of groups.
 */
TEST(KruskalWallisTest, DegreesOfFreedom) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0}
    };

    auto result = statcpp::kruskal_wallis_test(groups);

    // df = k - 1 = 4 - 1 = 3
    EXPECT_NEAR(result.df, 3.0, 1e-10);
}

/**
 * @brief Test Kruskal-Wallis with insufficient groups.
 * @test Verifies that exception is thrown when fewer than 2 groups are provided.
 */
TEST(KruskalWallisTest, TooFewGroups) {
    std::vector<std::vector<double>> groups = {{1.0, 2.0, 3.0}};
    EXPECT_THROW(statcpp::kruskal_wallis_test(groups), std::invalid_argument);
}

/**
 * @brief Test Kruskal-Wallis with empty group.
 * @test Verifies that exception is thrown when a group is empty.
 */
TEST(KruskalWallisTest, EmptyGroup) {
    std::vector<std::vector<double>> groups = {{1.0, 2.0}, {}};
    EXPECT_THROW(statcpp::kruskal_wallis_test(groups), std::invalid_argument);
}

// ============================================================================
// Tie Correction Tests
// ============================================================================

/**
 * @brief Test Wilcoxon signed-rank test with tied differences.
 * @test Verifies that the tie correction produces valid results when differences are tied.
 */
TEST(WilcoxonSignedRankTest, WithTiedDifferences) {
    // Data where differences from mu0=0 produce ties: |5|,|5|,|5|,|10|,|10|
    std::vector<double> data = {5.0, 5.0, -5.0, 10.0, -10.0};
    auto result = statcpp::wilcoxon_signed_rank_test(data.begin(), data.end(), 0.0);

    // Symmetric around 0, should not reject
    EXPECT_GT(result.p_value, 0.05);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

/**
 * @brief Test Mann-Whitney U test with tied values between groups.
 * @test Verifies that the tie correction produces valid results when values are shared across groups.
 */
TEST(MannWhitneyUTest, WithTiedValues) {
    // Groups share some identical values (ties across groups)
    std::vector<double> data1 = {1.0, 2.0, 3.0, 3.0, 4.0};
    std::vector<double> data2 = {3.0, 3.0, 5.0, 6.0, 7.0};

    auto result = statcpp::mann_whitney_u_test(data1.begin(), data1.end(),
                                                data2.begin(), data2.end());

    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    // Group 2 is generally larger, should show some significance
    EXPECT_LT(result.p_value, 0.15);
}

/**
 * @brief Test Kruskal-Wallis test with tied values across groups.
 * @test Verifies that the tie correction factor is applied to the H statistic.
 */
TEST(KruskalWallisTest, WithTiedValues) {
    // Groups with ties across them
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 2.0, 3.0, 3.0},
        {3.0, 3.0, 4.0, 5.0, 5.0},
        {5.0, 5.0, 6.0, 7.0, 7.0}
    };

    auto result = statcpp::kruskal_wallis_test(groups);

    EXPECT_GE(result.statistic, 0.0);
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    // Groups are clearly different, should reject
    EXPECT_LT(result.p_value, 0.05);
}

// ============================================================================
// Fisher's Exact Test Tests
// ============================================================================

/**
 * @brief Test Fisher's exact test with no association.
 * @test Verifies that odds ratio equals 1 and test does not reject for independent data.
 */
TEST(FisherExactTest, NoAssociation) {
    // 2x2 table with no association
    // a=25, b=25, c=25, d=25
    auto result = statcpp::fisher_exact_test(25, 25, 25, 25);

    // Odds ratio should be 1
    EXPECT_NEAR(result.statistic, 1.0, 1e-10);
    EXPECT_GT(result.p_value, 0.05);
}

/**
 * @brief Test Fisher's exact test with strong association.
 * @test Verifies that test rejects and odds ratio is greater than 1 for associated data.
 */
TEST(FisherExactTest, StrongAssociation) {
    // 2x2 table with strong association
    // a=30, b=5, c=5, d=30
    auto result = statcpp::fisher_exact_test(30, 5, 5, 30);

    // Strong association, should reject
    EXPECT_LT(result.p_value, 0.05);
    EXPECT_GT(result.statistic, 1.0);  // Odds ratio > 1
}

/**
 * @brief Test one-sided Fisher's exact test (greater alternative).
 * @test Verifies test with alternative hypothesis of positive association.
 */
TEST(FisherExactTest, OneSidedGreater) {
    auto result = statcpp::fisher_exact_test(30, 5, 5, 30,
                                              statcpp::alternative_hypothesis::greater);

    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::greater);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test one-sided Fisher's exact test (less alternative).
 * @test Verifies test with alternative hypothesis of negative association.
 */
TEST(FisherExactTest, OneSidedLess) {
    // Reverse the association
    auto result = statcpp::fisher_exact_test(5, 30, 30, 5,
                                              statcpp::alternative_hypothesis::less);

    EXPECT_EQ(result.alternative, statcpp::alternative_hypothesis::less);
    EXPECT_LT(result.p_value, 0.05);
}

/**
 * @brief Test Fisher's exact test with very small sample.
 * @test Verifies that test handles small samples and computes infinite odds ratio correctly.
 */
TEST(FisherExactTest, SmallSample) {
    // Very small sample - Fisher's exact test is especially useful here
    auto result = statcpp::fisher_exact_test(3, 0, 0, 3);

    // Perfect association - odds ratio should be infinity (b=0 or c=0)
    EXPECT_EQ(result.statistic, std::numeric_limits<double>::infinity());
    // p-value should be small for this extreme case
    EXPECT_LT(result.p_value, 0.15);
}
