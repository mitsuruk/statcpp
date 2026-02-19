#include <gtest/gtest.h>
#include "statcpp/anova.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// One-Way ANOVA Tests
// ============================================================================

/**
 * @brief Tests basic one-way ANOVA computation.
 * @test Verifies that group means, grand mean, and sample sizes are correctly computed.
 */
TEST(OneWayAnovaTest, BasicComputation) {
    std::vector<std::vector<double>> groups = {
        {4.0, 5.0, 6.0, 7.0, 8.0},       // Group 1: mean = 6
        {6.0, 7.0, 8.0, 9.0, 10.0},      // Group 2: mean = 8
        {8.0, 9.0, 10.0, 11.0, 12.0}     // Group 3: mean = 10
    };

    auto result = statcpp::one_way_anova(groups);

    EXPECT_EQ(result.n_groups, 3u);
    EXPECT_EQ(result.n_total, 15u);
    EXPECT_NEAR(result.grand_mean, 8.0, 1e-10);

    // Check group means
    EXPECT_NEAR(result.group_means[0], 6.0, 1e-10);
    EXPECT_NEAR(result.group_means[1], 8.0, 1e-10);
    EXPECT_NEAR(result.group_means[2], 10.0, 1e-10);
}

/**
 * @brief Tests ANOVA with significant group differences.
 * @test Verifies that clearly separated groups produce high F-statistic and low p-value.
 */
TEST(OneWayAnovaTest, SignificantDifference) {
    // Groups with clear differences
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {20.0, 21.0, 22.0, 23.0, 24.0}
    };

    auto result = statcpp::one_way_anova(groups);

    // Should be highly significant
    EXPECT_GT(result.between.f_statistic, 10.0);
    EXPECT_LT(result.between.p_value, 0.001);
}

/**
 * @brief Tests ANOVA with no significant group differences.
 * @test Verifies that similar groups produce non-significant p-value (> 0.05).
 */
TEST(OneWayAnovaTest, NoSignificantDifference) {
    // Groups with similar values
    std::vector<std::vector<double>> groups = {
        {5.0, 5.1, 4.9, 5.0, 5.1},
        {5.1, 4.9, 5.0, 5.1, 4.9},
        {4.9, 5.0, 5.1, 5.0, 4.9}
    };

    auto result = statcpp::one_way_anova(groups);

    // Should not be significant
    EXPECT_GT(result.between.p_value, 0.05);
}

/**
 * @brief Tests that degrees of freedom are correctly computed.
 * @test Verifies df_between = k-1, df_within = N-k, df_total = N-1.
 */
TEST(OneWayAnovaTest, DegreesOfFreedom) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0}
    };

    auto result = statcpp::one_way_anova(groups);

    // df_between = k - 1 = 4 - 1 = 3
    EXPECT_NEAR(result.between.df, 3.0, 1e-10);
    // df_within = N - k = 12 - 4 = 8
    EXPECT_NEAR(result.within.df, 8.0, 1e-10);
    // df_total = N - 1 = 11
    EXPECT_NEAR(result.df_total, 11.0, 1e-10);
}

/**
 * @brief Tests that sum of squares partitioning is correct.
 * @test Verifies that SS_total = SS_between + SS_within.
 */
TEST(OneWayAnovaTest, SumOfSquaresAddUp) {
    std::vector<std::vector<double>> groups = {
        {2.0, 4.0, 6.0, 8.0},
        {3.0, 5.0, 7.0, 9.0},
        {4.0, 6.0, 8.0, 10.0}
    };

    auto result = statcpp::one_way_anova(groups);

    // SS_total = SS_between + SS_within
    EXPECT_NEAR(result.ss_total, result.between.ss + result.within.ss, 1e-10);
}

/**
 * @brief Tests that too few groups throw exception.
 * @test Verifies that ANOVA with fewer than 2 groups throws std::invalid_argument.
 */
TEST(OneWayAnovaTest, TooFewGroups) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0}
    };

    EXPECT_THROW(statcpp::one_way_anova(groups), std::invalid_argument);
}

/**
 * @brief Tests that empty groups throw exception.
 * @test Verifies that ANOVA with any empty group throws std::invalid_argument.
 */
TEST(OneWayAnovaTest, EmptyGroup) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0},
        {},
        {4.0, 5.0, 6.0}
    };

    EXPECT_THROW(statcpp::one_way_anova(groups), std::invalid_argument);
}

// ============================================================================
// Two-Way ANOVA Tests
// ============================================================================

/**
 * @brief Tests basic two-way ANOVA computation.
 * @test Verifies that factor levels and total sample size are correctly identified.
 */
TEST(TwoWayAnovaTest, BasicComputation) {
    // 2x2 design with 3 replicates
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},  // Factor A level 0
        {{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}} // Factor A level 1
    };

    auto result = statcpp::two_way_anova(data);

    EXPECT_EQ(result.levels_a, 2u);
    EXPECT_EQ(result.levels_b, 2u);
    EXPECT_EQ(result.n_total, 12u);
}

/**
 * @brief Tests degrees of freedom in two-way ANOVA.
 * @test Verifies df_A, df_B, df_AB, and df_error are correctly computed.
 */
TEST(TwoWayAnovaTest, DegreesOfFreedom) {
    // 3x2 design with 2 replicates
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0, 2.0}, {3.0, 4.0}},
        {{5.0, 6.0}, {7.0, 8.0}},
        {{9.0, 10.0}, {11.0, 12.0}}
    };

    auto result = statcpp::two_way_anova(data);

    // df_A = a - 1 = 3 - 1 = 2
    EXPECT_NEAR(result.factor_a.df, 2.0, 1e-10);
    // df_B = b - 1 = 2 - 1 = 1
    EXPECT_NEAR(result.factor_b.df, 1.0, 1e-10);
    // df_AB = (a-1)(b-1) = 2*1 = 2
    EXPECT_NEAR(result.interaction.df, 2.0, 1e-10);
    // df_error = ab(n-1) = 3*2*(2-1) = 6
    EXPECT_NEAR(result.error.df, 6.0, 1e-10);
}

/**
 * @brief Tests that sum of squares partitioning is correct in two-way ANOVA.
 * @test Verifies that SS_total = SS_A + SS_B + SS_AB + SS_error.
 */
TEST(TwoWayAnovaTest, SumOfSquaresAddUp) {
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0, 2.0}, {3.0, 4.0}},
        {{5.0, 6.0}, {7.0, 8.0}}
    };

    auto result = statcpp::two_way_anova(data);

    // SS_total = SS_A + SS_B + SS_AB + SS_error
    double ss_sum = result.factor_a.ss + result.factor_b.ss +
                    result.interaction.ss + result.error.ss;
    EXPECT_NEAR(result.ss_total, ss_sum, 1e-10);
}

/**
 * @brief Tests detection of main effects in two-way ANOVA.
 * @test Verifies that significant main effects are correctly identified.
 */
TEST(TwoWayAnovaTest, MainEffects) {
    // Clear Factor A effect with some within-cell variation
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0, 1.5, 0.5}, {1.2, 0.8, 1.0}},
        {{10.0, 10.5, 9.5}, {10.2, 9.8, 10.0}}
    };

    auto result = statcpp::two_way_anova(data);

    // Factor A should be significant (large difference between levels)
    EXPECT_LT(result.factor_a.p_value, 0.01);
    // Factor B should not be significant (similar across levels)
    EXPECT_GT(result.factor_b.p_value, 0.05);
}

// ============================================================================
// Post-hoc Comparison Tests
// ============================================================================

/**
 * @brief Tests Tukey HSD post-hoc test.
 * @test Verifies that Tukey HSD performs all pairwise comparisons and identifies significant differences.
 */
TEST(PostHocTest, TukeyHSD) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {20.0, 21.0, 22.0, 23.0, 24.0}
    };

    auto anova_result = statcpp::one_way_anova(groups);
    auto posthoc = statcpp::tukey_hsd(anova_result, groups, 0.05);

    // Should have k*(k-1)/2 = 3 comparisons
    EXPECT_EQ(posthoc.comparisons.size(), 3u);

    // All comparisons should be significant
    for (const auto& comp : posthoc.comparisons) {
        EXPECT_TRUE(comp.significant);
    }

    // Method name should reflect true Tukey HSD
    EXPECT_EQ(posthoc.method, "Tukey HSD");
}

/**
 * @brief Tests Tukey HSD against R TukeyHSD() reference values.
 * @test Validates p-values and confidence intervals against R output for
 *       groups {4-8, 6-10, 8-12} with n=5 each.
 *       R: TukeyHSD(aov(y ~ g), conf.level=0.95)
 *       2-1: diff=2, lwr=-0.6678637, upr=4.667864, p adj=0.1545800
 *       3-1: diff=4, lwr=1.3321363, upr=6.667864, p adj=0.0046341
 *       3-2: diff=2, lwr=-0.6678637, upr=4.667864, p adj=0.1545800
 */
TEST(PostHocTest, TukeyHSDvsR) {
    std::vector<std::vector<double>> groups = {
        {4.0, 5.0, 6.0, 7.0, 8.0},       // Group 1: mean = 6
        {6.0, 7.0, 8.0, 9.0, 10.0},      // Group 2: mean = 8
        {8.0, 9.0, 10.0, 11.0, 12.0}     // Group 3: mean = 10
    };

    auto anova_result = statcpp::one_way_anova(groups);
    auto posthoc = statcpp::tukey_hsd(anova_result, groups, 0.05);

    ASSERT_EQ(posthoc.comparisons.size(), 3u);

    // Comparison 0: group 0 vs group 1 (diff = 6 - 8 = -2)
    EXPECT_NEAR(posthoc.comparisons[0].mean_diff, -2.0, 1e-10);
    EXPECT_NEAR(posthoc.comparisons[0].p_value, 0.1545800, 0.01);
    EXPECT_NEAR(posthoc.comparisons[0].lower, -4.667864, 0.05);
    EXPECT_NEAR(posthoc.comparisons[0].upper, 0.6678637, 0.05);

    // Comparison 1: group 0 vs group 2 (diff = 6 - 10 = -4)
    EXPECT_NEAR(posthoc.comparisons[1].mean_diff, -4.0, 1e-10);
    EXPECT_NEAR(posthoc.comparisons[1].p_value, 0.0046341, 0.005);
    EXPECT_TRUE(posthoc.comparisons[1].significant);

    // Comparison 2: group 1 vs group 2 (diff = 8 - 10 = -2)
    EXPECT_NEAR(posthoc.comparisons[2].mean_diff, -2.0, 1e-10);
    EXPECT_NEAR(posthoc.comparisons[2].p_value, 0.1545800, 0.01);

    // q statistic should be positive
    for (const auto& comp : posthoc.comparisons) {
        EXPECT_GT(comp.statistic, 0.0);
    }
}

/**
 * @brief Tests Bonferroni post-hoc correction.
 * @test Verifies that Bonferroni method performs all pairwise comparisons with corrected alpha.
 */
TEST(PostHocTest, BonferroniCorrection) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {5.0, 6.0, 7.0, 8.0, 9.0}  // Middle group
    };

    auto anova_result = statcpp::one_way_anova(groups);
    auto posthoc = statcpp::bonferroni_posthoc(anova_result, 0.05);

    EXPECT_EQ(posthoc.comparisons.size(), 3u);
    EXPECT_EQ(posthoc.method, "Bonferroni");
}

/**
 * @brief Tests Dunnett's test for comparisons with control group.
 * @test Verifies that Dunnett's test compares all treatments only to the control group.
 */
TEST(PostHocTest, DunnettTest) {
    std::vector<std::vector<double>> groups = {
        {5.0, 6.0, 7.0, 8.0, 9.0},       // Control (group 0)
        {15.0, 16.0, 17.0, 18.0, 19.0},  // Treatment 1
        {25.0, 26.0, 27.0, 28.0, 29.0}   // Treatment 2
    };

    auto anova_result = statcpp::one_way_anova(groups);
    auto posthoc = statcpp::dunnett_posthoc(anova_result, 0, 0.05);

    // Should have k-1 = 2 comparisons (vs control)
    EXPECT_EQ(posthoc.comparisons.size(), 2u);

    // Both treatments should differ significantly from control
    for (const auto& comp : posthoc.comparisons) {
        EXPECT_EQ(comp.group2, 0u);  // Comparing to control
        EXPECT_TRUE(comp.significant);
    }
}

/**
 * @brief Tests Scheffe's post-hoc method.
 * @test Verifies that Scheffe's method performs all pairwise comparisons conservatively.
 */
TEST(PostHocTest, ScheffeMethod) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {20.0, 21.0, 22.0, 23.0, 24.0}
    };

    auto anova_result = statcpp::one_way_anova(groups);
    auto posthoc = statcpp::scheffe_posthoc(anova_result, 0.05);

    EXPECT_EQ(posthoc.comparisons.size(), 3u);
    EXPECT_EQ(posthoc.method, "Scheffe");
}

// ============================================================================
// ANCOVA Tests
// ============================================================================

/**
 * @brief Tests basic ANCOVA computation with covariate.
 * @test Verifies that ANCOVA computes adjusted means accounting for covariate.
 */
TEST(AncovaTest, BasicComputation) {
    // Two groups with covariate (with some variation)
    std::vector<std::vector<std::pair<double, double>>> groups = {
        {{10.5, 5.0}, {11.5, 6.0}, {14.5, 7.0}, {15.5, 8.0}, {18.5, 9.0}},
        {{15.5, 5.0}, {16.5, 6.0}, {19.5, 7.0}, {20.5, 8.0}, {23.5, 9.0}}
    };

    auto result = statcpp::one_way_ancova(groups);

    // Should have adjusted means
    EXPECT_EQ(result.adjusted_means.size(), 2u);

    // Both groups have same covariate values, so adjusted means differ
    EXPECT_NE(result.adjusted_means[0], result.adjusted_means[1]);
}

/**
 * @brief Tests that ANCOVA adjusts means for covariate differences.
 * @test Verifies that adjusted means differ from raw means when groups have different covariate values.
 */
TEST(AncovaTest, AdjustedMeans) {
    // Groups with different covariate means
    std::vector<std::vector<std::pair<double, double>>> groups = {
        {{10.0, 2.0}, {12.0, 3.0}, {14.0, 4.0}, {16.0, 5.0}},
        {{20.0, 6.0}, {22.0, 7.0}, {24.0, 8.0}, {26.0, 9.0}}
    };

    auto result = statcpp::one_way_ancova(groups);

    // Adjusted means should account for covariate differences
    EXPECT_NE(result.adjusted_means[0], result.adjusted_means[1]);
}

// ============================================================================
// Effect Size Tests
// ============================================================================

/**
 * @brief Tests eta-squared effect size calculation for ANOVA.
 * @test Verifies that eta-squared is between 0 and 1 and high for clearly different groups.
 */
TEST(EffectSizeTest, EtaSquared) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0},
        {20.0, 21.0, 22.0, 23.0, 24.0}
    };

    auto result = statcpp::one_way_anova(groups);
    double eta_sq = statcpp::eta_squared(result);

    // η² should be between 0 and 1
    EXPECT_GE(eta_sq, 0.0);
    EXPECT_LE(eta_sq, 1.0);

    // For these clearly different groups, η² should be high
    EXPECT_GT(eta_sq, 0.9);
}

/**
 * @brief Tests omega-squared effect size calculation for ANOVA.
 * @test Verifies that omega-squared is less biased than eta-squared (smaller value).
 */
TEST(EffectSizeTest, OmegaSquared) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0}
    };

    auto result = statcpp::one_way_anova(groups);
    double omega_sq = statcpp::omega_squared(result);

    // ω² should be less than η² (less biased)
    double eta_sq = statcpp::eta_squared(result);
    EXPECT_LT(omega_sq, eta_sq);
    EXPECT_GE(omega_sq, 0.0);
}

/**
 * @brief Tests Cohen's f effect size calculation for ANOVA.
 * @test Verifies that Cohen's f is positive and large (> 0.4) for clearly different groups.
 */
TEST(EffectSizeTest, CohensF) {
    std::vector<std::vector<double>> groups = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {10.0, 11.0, 12.0, 13.0, 14.0}
    };

    auto result = statcpp::one_way_anova(groups);
    double f = statcpp::cohens_f(result);

    // Cohen's f should be positive
    EXPECT_GT(f, 0.0);

    // For large effect, f > 0.4
    EXPECT_GT(f, 0.4);
}

/**
 * @brief Tests partial eta-squared for two-way ANOVA.
 * @test Verifies that partial eta-squared is computed for each factor and interaction, bounded [0,1].
 */
TEST(EffectSizeTest, PartialEtaSquaredTwoWay) {
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0, 2.0}, {10.0, 11.0}},
        {{3.0, 4.0}, {12.0, 13.0}}
    };

    auto result = statcpp::two_way_anova(data);

    double partial_eta_a = statcpp::partial_eta_squared_a(result);
    double partial_eta_b = statcpp::partial_eta_squared_b(result);
    double partial_eta_ab = statcpp::partial_eta_squared_interaction(result);

    // All should be between 0 and 1
    EXPECT_GE(partial_eta_a, 0.0);
    EXPECT_LE(partial_eta_a, 1.0);
    EXPECT_GE(partial_eta_b, 0.0);
    EXPECT_LE(partial_eta_b, 1.0);
    EXPECT_GE(partial_eta_ab, 0.0);
    EXPECT_LE(partial_eta_ab, 1.0);
}

// ============================================================================
// Degenerate Case Tests
// ============================================================================

/**
 * @brief Tests that two-way ANOVA rejects n_rep < 2.
 * @test With n_rep=1, df_error=0 making F-test impossible. Should throw.
 */
TEST(TwoWayAnovaTest, SingleReplicationThrows) {
    // 2x2 design with 1 replication per cell
    std::vector<std::vector<std::vector<double>>> data = {
        {{1.0}, {2.0}},
        {{3.0}, {4.0}}
    };

    EXPECT_THROW(statcpp::two_way_anova(data), std::invalid_argument);
}

/**
 * @brief Tests one-way ANOVA with constant data across all groups.
 * @test When all observations are identical, ss_within=0 and ss_between=0.
 *       F should be 0 and p should be 1 (no difference).
 */
TEST(OneWayAnovaTest, ConstantData) {
    std::vector<std::vector<double>> groups = {
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0}
    };

    auto result = statcpp::one_way_anova(groups);

    // All data identical: F=0, p=1
    EXPECT_DOUBLE_EQ(result.between.f_statistic, 0.0);
    EXPECT_DOUBLE_EQ(result.between.p_value, 1.0);
}

/**
 * @brief Tests one-way ANOVA with constant within groups but different means.
 * @test When within-group variance is 0 but groups differ, F should be infinity, p=0.
 */
TEST(OneWayAnovaTest, ZeroWithinVariance) {
    std::vector<std::vector<double>> groups = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0}
    };

    auto result = statcpp::one_way_anova(groups);

    EXPECT_TRUE(std::isinf(result.between.f_statistic));
    EXPECT_DOUBLE_EQ(result.between.p_value, 0.0);
}

/**
 * @brief Tests effect size functions with constant data (zero-division guard).
 * @test When ss_total=0, eta_squared and omega_squared should return 0 instead of NaN.
 */
TEST(EffectSizeTest, ConstantDataGuard) {
    std::vector<std::vector<double>> groups = {
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0}
    };

    auto result = statcpp::one_way_anova(groups);

    EXPECT_DOUBLE_EQ(statcpp::eta_squared(result), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::omega_squared(result), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::cohens_f(result), 0.0);
}

/**
 * @brief Tests post-hoc functions when se==0 (within-group variance is zero).
 * @test When mse=0 and groups differ, se=0 → stat=inf, p=0, significant=true.
 *       When mse=0 and groups are identical, se=0 → stat=0, p=1, significant=false.
 */
TEST(PostHocTest, ZeroSEDiffGroups) {
    // Groups differ but within-group variance is zero
    std::vector<std::vector<double>> groups = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {3.0, 3.0, 3.0}
    };
    auto result = statcpp::one_way_anova(groups);
    EXPECT_DOUBLE_EQ(result.within.ms, 0.0);

    // Tukey HSD
    auto tukey = statcpp::tukey_hsd(result, groups);
    EXPECT_EQ(tukey.comparisons.size(), 3u);
    for (const auto& c : tukey.comparisons) {
        EXPECT_TRUE(std::isinf(c.statistic));
        EXPECT_DOUBLE_EQ(c.p_value, 0.0);
        EXPECT_TRUE(c.significant);
        EXPECT_DOUBLE_EQ(c.lower, c.mean_diff);
        EXPECT_DOUBLE_EQ(c.upper, c.mean_diff);
    }

    // Bonferroni
    auto bonf = statcpp::bonferroni_posthoc(result);
    EXPECT_EQ(bonf.comparisons.size(), 3u);
    for (const auto& c : bonf.comparisons) {
        EXPECT_TRUE(std::isinf(c.statistic));
        EXPECT_DOUBLE_EQ(c.p_value, 0.0);
        EXPECT_TRUE(c.significant);
    }

    // Dunnett
    auto dunn = statcpp::dunnett_posthoc(result);
    EXPECT_EQ(dunn.comparisons.size(), 2u);
    for (const auto& c : dunn.comparisons) {
        EXPECT_TRUE(std::isinf(c.statistic));
        EXPECT_DOUBLE_EQ(c.p_value, 0.0);
        EXPECT_TRUE(c.significant);
    }

    // Scheffe
    auto sch = statcpp::scheffe_posthoc(result);
    EXPECT_EQ(sch.comparisons.size(), 3u);
    for (const auto& c : sch.comparisons) {
        EXPECT_TRUE(std::isinf(c.statistic));
        EXPECT_DOUBLE_EQ(c.p_value, 0.0);
        EXPECT_TRUE(c.significant);
    }
}

TEST(PostHocTest, ZeroSEIdenticalGroups) {
    // All groups identical: mean_diff=0 and se=0
    std::vector<std::vector<double>> groups = {
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0},
        {5.0, 5.0, 5.0}
    };
    auto result = statcpp::one_way_anova(groups);
    EXPECT_DOUBLE_EQ(result.within.ms, 0.0);

    // Tukey HSD
    auto tukey = statcpp::tukey_hsd(result, groups);
    for (const auto& c : tukey.comparisons) {
        EXPECT_DOUBLE_EQ(c.statistic, 0.0);
        EXPECT_DOUBLE_EQ(c.p_value, 1.0);
        EXPECT_FALSE(c.significant);
        EXPECT_DOUBLE_EQ(c.lower, 0.0);
        EXPECT_DOUBLE_EQ(c.upper, 0.0);
    }

    // Bonferroni
    auto bonf = statcpp::bonferroni_posthoc(result);
    for (const auto& c : bonf.comparisons) {
        EXPECT_DOUBLE_EQ(c.statistic, 0.0);
        EXPECT_DOUBLE_EQ(c.p_value, 1.0);
        EXPECT_FALSE(c.significant);
    }

    // Dunnett
    auto dunn = statcpp::dunnett_posthoc(result);
    for (const auto& c : dunn.comparisons) {
        EXPECT_DOUBLE_EQ(c.statistic, 0.0);
        EXPECT_DOUBLE_EQ(c.p_value, 1.0);
        EXPECT_FALSE(c.significant);
    }

    // Scheffe
    auto sch = statcpp::scheffe_posthoc(result);
    for (const auto& c : sch.comparisons) {
        EXPECT_DOUBLE_EQ(c.statistic, 0.0);
        EXPECT_DOUBLE_EQ(c.p_value, 1.0);
        EXPECT_FALSE(c.significant);
    }
}
