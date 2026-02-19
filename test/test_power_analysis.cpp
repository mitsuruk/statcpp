/**
 * @file test_power_analysis.cpp
 * @brief Unit tests for power analysis functions
 */

#include <gtest/gtest.h>
#include "statcpp/power_analysis.hpp"
#include <cmath>
#include <limits>

// ============================================================
// Test fixture
// ============================================================

class PowerAnalysisTest : public ::testing::Test {
protected:
    // Tolerance
    static constexpr double TOLERANCE = 1e-3;
    static constexpr double LOOSE_TOLERANCE = 1e-2;
};

// ============================================================
// One-sample t-test: Power calculation tests
// ============================================================

/**
 * @brief Tests power calculation for one-sample t-test with small effect size.
 * @test Verifies that power is approximately 0.80 for d=0.2 with n=200.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_SmallEffect) {
    // Small effect size (d=0.2) with n=200
    double power = statcpp::power_t_test_one_sample(0.2, 200);

    // Theoretical value is approximately 0.80
    EXPECT_GT(power, 0.75);
    EXPECT_LT(power, 0.85);
}

/**
 * @brief Tests power calculation for one-sample t-test with medium effect size.
 * @test Verifies that power is approximately 0.93 for d=0.5 with n=50.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_MediumEffect) {
    // Medium effect size (d=0.5) with n=50
    double power = statcpp::power_t_test_one_sample(0.5, 50);

    // Theoretical value is approximately 0.93
    EXPECT_GT(power, 0.90);
    EXPECT_LT(power, 0.96);
}

/**
 * @brief Tests power calculation for one-sample t-test with large effect size.
 * @test Verifies that power is approximately 0.992 for d=0.8 with n=30.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_LargeEffect) {
    // Large effect size (d=0.8) with n=30
    double power = statcpp::power_t_test_one_sample(0.8, 30);

    // Theoretical value is approximately 0.992
    EXPECT_GT(power, 0.98);
    EXPECT_LT(power, 1.00);
}

/**
 * @brief Tests that power increases with sample size.
 * @test Verifies that larger sample sizes produce higher statistical power.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_IncreasesWithN) {
    // Power increases with sample size
    double power_30 = statcpp::power_t_test_one_sample(0.5, 30);
    double power_50 = statcpp::power_t_test_one_sample(0.5, 50);
    double power_100 = statcpp::power_t_test_one_sample(0.5, 100);

    EXPECT_LT(power_30, power_50);
    EXPECT_LT(power_50, power_100);
}

/**
 * @brief Tests that power increases with effect size.
 * @test Verifies that larger effect sizes produce higher statistical power.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_IncreasesWithEffectSize) {
    // Power increases with effect size
    double power_02 = statcpp::power_t_test_one_sample(0.2, 50);
    double power_05 = statcpp::power_t_test_one_sample(0.5, 50);
    double power_08 = statcpp::power_t_test_one_sample(0.8, 50);

    EXPECT_LT(power_02, power_05);
    EXPECT_LT(power_05, power_08);
}

/**
 * @brief Tests that one-sided tests have higher power than two-sided tests.
 * @test Verifies that one-tailed test produces greater power than two-tailed test.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_OneSided) {
    // One-sided test has higher power than two-sided test
    double power_two = statcpp::power_t_test_one_sample(0.5, 50, 0.05, "two.sided");
    double power_one = statcpp::power_t_test_one_sample(0.5, 50, 0.05, "greater");

    EXPECT_GT(power_one, power_two);
}

/**
 * @brief Tests the effect of alpha level on power.
 * @test Verifies that smaller alpha values decrease statistical power.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_AlphaEffect) {
    // Smaller alpha results in lower power
    double power_10 = statcpp::power_t_test_one_sample(0.5, 50, 0.10);
    double power_05 = statcpp::power_t_test_one_sample(0.5, 50, 0.05);
    double power_01 = statcpp::power_t_test_one_sample(0.5, 50, 0.01);

    EXPECT_GT(power_10, power_05);
    EXPECT_GT(power_05, power_01);
}

/**
 * @brief Tests that power is bounded between 0 and 1.
 * @test Verifies that computed power values are within the valid [0, 1] range.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_BoundedBetweenZeroAndOne) {
    // Power is between 0 and 1
    double power = statcpp::power_t_test_one_sample(0.3, 100);

    EXPECT_GE(power, 0.0);
    EXPECT_LE(power, 1.0);
}

/**
 * @brief Tests that power approaches 1.0 with very large sample size.
 * @test Verifies that power exceeds 0.99 for very large n.
 */
TEST_F(PowerAnalysisTest, PowerOneSample_VeryLargeN) {
    // Power approaches 1 with very large n
    double power = statcpp::power_t_test_one_sample(0.5, 10000);

    EXPECT_GT(power, 0.99);
}

// ============================================================
// One-sample t-test: Sample size calculation tests
// ============================================================

/**
 * @brief Tests sample size calculation for standard case.
 * @test Verifies that required n is approximately 34 for d=0.5, power=0.8, alpha=0.05.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_StandardCase) {
    // Standard case with d=0.5, power=0.8, alpha=0.05
    std::size_t n = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05);

    // Theoretical value is approximately 34
    EXPECT_GT(n, 30);
    EXPECT_LT(n, 40);
}

/**
 * @brief Tests that small effect sizes require larger sample sizes.
 * @test Verifies that n is approximately 200 for small effect size d=0.2.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_SmallEffect) {
    // Small effect size requires more samples
    std::size_t n = statcpp::sample_size_t_test_one_sample(0.2, 0.80, 0.05);

    // Theoretical value is approximately 200
    EXPECT_GT(n, 180);
    EXPECT_LT(n, 220);
}

/**
 * @brief Tests that large effect sizes require smaller sample sizes.
 * @test Verifies that n is approximately 15 for large effect size d=0.8.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_LargeEffect) {
    // Large effect size requires fewer samples
    std::size_t n = statcpp::sample_size_t_test_one_sample(0.8, 0.80, 0.05);

    // Theoretical value is approximately 15
    EXPECT_GT(n, 12);
    EXPECT_LT(n, 20);
}

/**
 * @brief Tests that higher power requirements increase sample size.
 * @test Verifies that required n increases with higher target power levels.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_HighPower) {
    // Higher power requirement needs more samples
    std::size_t n_80 = statcpp::sample_size_t_test_one_sample(0.5, 0.80);
    std::size_t n_90 = statcpp::sample_size_t_test_one_sample(0.5, 0.90);
    std::size_t n_95 = statcpp::sample_size_t_test_one_sample(0.5, 0.95);

    EXPECT_LT(n_80, n_90);
    EXPECT_LT(n_90, n_95);
}

/**
 * @brief Tests that one-sided tests require smaller sample sizes.
 * @test Verifies that one-tailed test requires fewer observations than two-tailed test.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_OneSided) {
    // One-sided test requires fewer samples
    std::size_t n_two = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "two.sided");
    std::size_t n_one = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "greater");

    EXPECT_LT(n_one, n_two);
}

/**
 * @brief Tests consistency between sample size and power calculations.
 * @test Verifies that computed sample size achieves target power when used in power calculation.
 */
TEST_F(PowerAnalysisTest, SampleSizeOneSample_ConsistencyCheck) {
    // Verify that computed sample size achieves target power
    double effect_size = 0.5;
    double target_power = 0.80;
    double alpha = 0.05;

    std::size_t n = statcpp::sample_size_t_test_one_sample(effect_size, target_power, alpha);
    double achieved_power = statcpp::power_t_test_one_sample(effect_size, n, alpha);

    // Computed power should be at least the target value
    EXPECT_GE(achieved_power, target_power - LOOSE_TOLERANCE);
}

// ============================================================
// Two-sample t-test: Power calculation tests
// ============================================================

/**
 * @brief Tests power calculation for two-sample t-test with equal sample sizes.
 * @test Verifies that power is approximately 0.80 for d=0.5 with n1=n2=64.
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_EqualN) {
    // Power with equal sample sizes
    double power = statcpp::power_t_test_two_sample(0.5, 64, 64);

    // Theoretical value is approximately 0.80
    EXPECT_GT(power, 0.75);
    EXPECT_LT(power, 0.85);
}

/**
 * @brief Tests power calculation with unequal sample sizes.
 * @test Verifies that power calculation works with unequal group sizes (less efficient).
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_UnequalN) {
    // Power with unequal sample sizes
    double power = statcpp::power_t_test_two_sample(0.5, 50, 100);

    // Unequal allocation is less efficient
    EXPECT_GT(power, 0.0);
    EXPECT_LT(power, 1.0);
}

/**
 * @brief Tests that two-sample power increases with sample size.
 * @test Verifies that larger equal sample sizes produce higher statistical power.
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_IncreasesWithN) {
    // Power increases with sample size
    double power_30 = statcpp::power_t_test_two_sample(0.5, 30, 30);
    double power_50 = statcpp::power_t_test_two_sample(0.5, 50, 50);
    double power_100 = statcpp::power_t_test_two_sample(0.5, 100, 100);

    EXPECT_LT(power_30, power_50);
    EXPECT_LT(power_50, power_100);
}

/**
 * @brief Tests that two-sample power increases with effect size.
 * @test Verifies that larger effect sizes produce higher statistical power in two-sample tests.
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_IncreasesWithEffectSize) {
    // Power increases with effect size
    double power_02 = statcpp::power_t_test_two_sample(0.2, 50, 50);
    double power_05 = statcpp::power_t_test_two_sample(0.5, 50, 50);
    double power_08 = statcpp::power_t_test_two_sample(0.8, 50, 50);

    EXPECT_LT(power_02, power_05);
    EXPECT_LT(power_05, power_08);
}

/**
 * @brief Tests that one-sided two-sample tests have higher power.
 * @test Verifies that one-tailed two-sample test has greater power than two-tailed.
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_OneSided) {
    // One-sided test has higher power than two-sided test
    double power_two = statcpp::power_t_test_two_sample(0.5, 50, 50, 0.05, "two.sided");
    double power_one = statcpp::power_t_test_two_sample(0.5, 50, 50, 0.05, "greater");

    EXPECT_GT(power_one, power_two);
}

/**
 * @brief Tests the effect of alpha on two-sample power.
 * @test Verifies that smaller alpha values decrease power in two-sample tests.
 */
TEST_F(PowerAnalysisTest, PowerTwoSample_AlphaEffect) {
    // Smaller alpha results in lower power
    double power_10 = statcpp::power_t_test_two_sample(0.5, 50, 50, 0.10);
    double power_05 = statcpp::power_t_test_two_sample(0.5, 50, 50, 0.05);
    double power_01 = statcpp::power_t_test_two_sample(0.5, 50, 50, 0.01);

    EXPECT_GT(power_10, power_05);
    EXPECT_GT(power_05, power_01);
}

// ============================================================
// Two-sample t-test: Sample size calculation tests
// ============================================================

/**
 * @brief Tests two-sample size calculation for standard case.
 * @test Verifies that required n per group is approximately 64 for d=0.5, power=0.8.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_StandardCase) {
    // Standard case with d=0.5, power=0.8, alpha=0.05
    std::size_t n = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05);

    // Theoretical value is approximately 64 (per group)
    EXPECT_GT(n, 55);
    EXPECT_LT(n, 75);
}

/**
 * @brief Tests that small effects require larger two-sample sizes.
 * @test Verifies that n per group is approximately 400 for small effect d=0.2.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_SmallEffect) {
    // Small effect size requires more samples
    std::size_t n = statcpp::sample_size_t_test_two_sample(0.2, 0.80, 0.05);

    // Theoretical value is approximately 400 (per group)
    EXPECT_GT(n, 350);
    EXPECT_LT(n, 450);
}

/**
 * @brief Tests that large effects require smaller two-sample sizes.
 * @test Verifies that n per group is approximately 26 for large effect d=0.8.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_LargeEffect) {
    // Large effect size requires fewer samples
    std::size_t n = statcpp::sample_size_t_test_two_sample(0.8, 0.80, 0.05);

    // Theoretical value is approximately 26 (per group)
    EXPECT_GT(n, 20);
    EXPECT_LT(n, 35);
}

/**
 * @brief Tests that higher power increases required two-sample size.
 * @test Verifies that n per group increases with higher target power.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_HighPower) {
    // Higher power requirement needs more samples
    std::size_t n_80 = statcpp::sample_size_t_test_two_sample(0.5, 0.80);
    std::size_t n_90 = statcpp::sample_size_t_test_two_sample(0.5, 0.90);
    std::size_t n_95 = statcpp::sample_size_t_test_two_sample(0.5, 0.95);

    EXPECT_LT(n_80, n_90);
    EXPECT_LT(n_90, n_95);
}

/**
 * @brief Tests that one-sided tests require smaller two-sample sizes.
 * @test Verifies that one-tailed test requires fewer observations per group.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_OneSided) {
    // One-sided test requires fewer samples
    std::size_t n_two = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 1.0, "two.sided");
    std::size_t n_one = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 1.0, "greater");

    EXPECT_LT(n_one, n_two);
}

/**
 * @brief Tests that unequal allocation increases total sample size.
 * @test Verifies that unequal group sizes require more total observations than equal allocation.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_UnequalRatio) {
    // Unequal allocation increases total sample size
    std::size_t n_equal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 1.0);
    std::size_t n_unequal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 2.0);

    // When ratio is 2, n_unequal is n1, and n2 = 2*n1
    // Total is n1 + 2*n1 = 3*n1 vs 2*n_equal
    // Unequal allocation results in larger total
    EXPECT_GT(n_unequal * 3, n_equal * 2);
}

/**
 * @brief Tests consistency between two-sample size and power calculations.
 * @test Verifies that computed sample size achieves target power in two-sample test.
 */
TEST_F(PowerAnalysisTest, SampleSizeTwoSample_ConsistencyCheck) {
    // Verify that computed sample size achieves target power
    double effect_size = 0.5;
    double target_power = 0.80;
    double alpha = 0.05;

    std::size_t n = statcpp::sample_size_t_test_two_sample(effect_size, target_power, alpha);
    double achieved_power = statcpp::power_t_test_two_sample(effect_size, n, n, alpha);

    // Computed power should be at least the target value
    EXPECT_GE(achieved_power, target_power - LOOSE_TOLERANCE);
}

// ============================================================
// Proportion test: Power calculation tests
// ============================================================

/**
 * @brief Tests power calculation for proportion test with standard parameters.
 * @test Verifies power is between 0.5-0.85 for p1=0.5, p2=0.6, n=200.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_StandardCase) {
    // Standard case with p1=0.5, p2=0.6, n=200
    double power = statcpp::power_prop_test(0.5, 0.6, 200);

    // Power is approximately 0.5-0.8
    EXPECT_GT(power, 0.50);
    EXPECT_LT(power, 0.85);
}

/**
 * @brief Tests that small proportion differences have lower power.
 * @test Verifies that smaller difference in proportions produces lower power.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_SmallDifference) {
    // Small difference results in lower power
    double power_small = statcpp::power_prop_test(0.5, 0.52, 200);
    double power_large = statcpp::power_prop_test(0.5, 0.60, 200);

    EXPECT_LT(power_small, power_large);
}

/**
 * @brief Tests that proportion test power increases with sample size.
 * @test Verifies that larger samples produce higher power in proportion tests.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_IncreasesWithN) {
    // Power increases with sample size
    double power_100 = statcpp::power_prop_test(0.5, 0.6, 100);
    double power_200 = statcpp::power_prop_test(0.5, 0.6, 200);
    double power_500 = statcpp::power_prop_test(0.5, 0.6, 500);

    EXPECT_LT(power_100, power_200);
    EXPECT_LT(power_200, power_500);
}

/**
 * @brief Tests that one-sided proportion tests have higher power.
 * @test Verifies that one-tailed proportion test has greater power than two-tailed.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_OneSided) {
    // One-sided test has higher power than two-sided test
    // Test "greater" with p1 = 0.6, p2 = 0.5 (p1 > p2)
    double power_two = statcpp::power_prop_test(0.6, 0.5, 200, 0.05, "two.sided");
    double power_one = statcpp::power_prop_test(0.6, 0.5, 200, 0.05, "greater");

    EXPECT_GT(power_one, power_two);
}

/**
 * @brief Tests the effect of alpha on proportion test power.
 * @test Verifies that smaller alpha values decrease power in proportion tests.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_AlphaEffect) {
    // Smaller alpha results in lower power
    double power_10 = statcpp::power_prop_test(0.5, 0.6, 200, 0.10);
    double power_05 = statcpp::power_prop_test(0.5, 0.6, 200, 0.05);
    double power_01 = statcpp::power_prop_test(0.5, 0.6, 200, 0.01);

    EXPECT_GT(power_10, power_05);
    EXPECT_GT(power_05, power_01);
}

/**
 * @brief Tests that proportion test power is symmetric.
 * @test Verifies that swapping p1 and p2 produces equal power in two-sided tests.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_SymmetricDifference) {
    // Swapping p1 and p2 yields the same power (for two-sided test)
    double power1 = statcpp::power_prop_test(0.4, 0.5, 200);
    double power2 = statcpp::power_prop_test(0.5, 0.4, 200);

    EXPECT_NEAR(power1, power2, TOLERANCE);
}

/**
 * @brief Tests power calculation with extreme proportions near 0 or 1.
 * @test Verifies that power can be computed for proportions close to boundaries.
 */
TEST_F(PowerAnalysisTest, PowerPropTest_ExtremeProportion) {
    // Calculation is possible even with extreme proportions (near 0 or 1)
    double power_low = statcpp::power_prop_test(0.01, 0.05, 500);
    double power_high = statcpp::power_prop_test(0.95, 0.99, 500);

    EXPECT_GT(power_low, 0.0);
    EXPECT_LT(power_low, 1.0);
    EXPECT_GT(power_high, 0.0);
    EXPECT_LT(power_high, 1.0);
}

// ============================================================
// Proportion test: Sample size calculation tests
// ============================================================

/**
 * @brief Tests sample size calculation for proportion test with standard parameters.
 * @test Verifies that required n is approximately 385 per group for p1=0.5, p2=0.6.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_StandardCase) {
    // Standard case with p1=0.5, p2=0.6, power=0.8, alpha=0.05
    std::size_t n = statcpp::sample_size_prop_test(0.5, 0.6, 0.80, 0.05);

    // Theoretical value is approximately 385 (per group)
    EXPECT_GT(n, 350);
    EXPECT_LT(n, 420);
}

/**
 * @brief Tests that small proportion differences require larger samples.
 * @test Verifies that smaller differences in proportions require more observations.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_SmallDifference) {
    // Small difference requires more samples
    std::size_t n_small = statcpp::sample_size_prop_test(0.5, 0.52, 0.80, 0.05);
    std::size_t n_large = statcpp::sample_size_prop_test(0.5, 0.60, 0.80, 0.05);

    EXPECT_GT(n_small, n_large);
}

/**
 * @brief Tests that large proportion differences require smaller samples.
 * @test Verifies that n is approximately 93 per group for large difference (0.5 to 0.7).
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_LargeDifference) {
    // Large difference requires fewer samples
    std::size_t n = statcpp::sample_size_prop_test(0.5, 0.7, 0.80, 0.05);

    // Theoretical value is approximately 93 (per group)
    EXPECT_GT(n, 85);
    EXPECT_LT(n, 105);
}

/**
 * @brief Tests that higher power increases required sample size for proportions.
 * @test Verifies that n per group increases with higher target power in proportion tests.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_HighPower) {
    // Higher power requirement needs more samples
    std::size_t n_80 = statcpp::sample_size_prop_test(0.5, 0.6, 0.80);
    std::size_t n_90 = statcpp::sample_size_prop_test(0.5, 0.6, 0.90);
    std::size_t n_95 = statcpp::sample_size_prop_test(0.5, 0.6, 0.95);

    EXPECT_LT(n_80, n_90);
    EXPECT_LT(n_90, n_95);
}

/**
 * @brief Tests that one-sided proportion tests require smaller samples.
 * @test Verifies that one-tailed test requires fewer observations than two-tailed.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_OneSided) {
    // One-sided test requires fewer samples
    // Test "greater" with p1 = 0.6, p2 = 0.5 (p1 > p2)
    std::size_t n_two = statcpp::sample_size_prop_test(0.6, 0.5, 0.80, 0.05, "two.sided");
    std::size_t n_one = statcpp::sample_size_prop_test(0.6, 0.5, 0.80, 0.05, "greater");

    EXPECT_LT(n_one, n_two);
}

/**
 * @brief Tests baseline proportion effect on required sample size.
 * @test Verifies that mid-range baseline proportions (around 0.5) require largest samples.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_BaselineEffect) {
    // Baseline rate near 0.5 requires the most samples
    std::size_t n_low = statcpp::sample_size_prop_test(0.1, 0.2, 0.80, 0.05);
    std::size_t n_mid = statcpp::sample_size_prop_test(0.4, 0.5, 0.80, 0.05);
    std::size_t n_high = statcpp::sample_size_prop_test(0.8, 0.9, 0.80, 0.05);

    // Mid-range baseline rate requires the largest sample size
    EXPECT_GT(n_mid, n_low);
    EXPECT_GT(n_mid, n_high);
}

/**
 * @brief Tests consistency between proportion sample size and power calculations.
 * @test Verifies that computed sample size achieves target power in proportion test.
 */
TEST_F(PowerAnalysisTest, SampleSizePropTest_ConsistencyCheck) {
    // Verify that computed sample size achieves target power
    double p1 = 0.5;
    double p2 = 0.6;
    double target_power = 0.80;
    double alpha = 0.05;

    std::size_t n = statcpp::sample_size_prop_test(p1, p2, target_power, alpha);
    double achieved_power = statcpp::power_prop_test(p1, p2, n, alpha);

    // Computed power should be at least the target value
    EXPECT_GE(achieved_power, target_power - LOOSE_TOLERANCE);
}

// ============================================================
// Edge case tests
// ============================================================

/**
 * @brief Tests that zero effect size produces power equal to alpha.
 * @test Verifies that power equals alpha when there is no effect (null hypothesis is true).
 */
TEST_F(PowerAnalysisTest, EdgeCase_ZeroEffectSize) {
    // When effect size is 0, power equals alpha
    double power = statcpp::power_t_test_one_sample(0.0, 100, 0.05);

    EXPECT_NEAR(power, 0.05, LOOSE_TOLERANCE);
}

/**
 * @brief Tests power calculation with very small alpha level.
 * @test Verifies that power can be computed even with very stringent alpha (e.g., 0.001).
 */
TEST_F(PowerAnalysisTest, EdgeCase_VerySmallAlpha) {
    // Calculation is possible even with very small alpha
    double power = statcpp::power_t_test_one_sample(0.5, 100, 0.001);

    EXPECT_GT(power, 0.0);
    EXPECT_LT(power, 1.0);
}

/**
 * @brief Tests sample size calculation with very high target power.
 * @test Verifies that sample size can be computed for power=0.99 within reasonable limits.
 */
TEST_F(PowerAnalysisTest, EdgeCase_VeryHighPower) {
    // Sample size can be computed even with very high power requirement
    std::size_t n = statcpp::sample_size_t_test_one_sample(0.5, 0.99, 0.05);

    EXPECT_GT(n, 0);
    EXPECT_LT(n, 10000);  // Realistic range
}

/**
 * @brief Tests sample size calculation with proportions near boundaries.
 * @test Verifies that sample size can be computed for extreme proportions (near 0 or 1).
 */
TEST_F(PowerAnalysisTest, EdgeCase_ProportionNearBoundary) {
    // When proportion is near 0 or 1
    std::size_t n1 = statcpp::sample_size_prop_test(0.01, 0.05, 0.80, 0.05);
    std::size_t n2 = statcpp::sample_size_prop_test(0.95, 0.99, 0.80, 0.05);

    EXPECT_GT(n1, 0);
    EXPECT_GT(n2, 0);
}

/**
 * @brief Tests power calculation with small sample size.
 * @test Verifies that power can be computed even with small n (e.g., n=10).
 */
TEST_F(PowerAnalysisTest, EdgeCase_SmallSampleSize) {
    // Calculation is possible even with small sample size
    double power = statcpp::power_t_test_one_sample(0.5, 10, 0.05);

    EXPECT_GT(power, 0.0);
    EXPECT_LT(power, 1.0);
}

// ============================================================
// Theoretical relationship tests
// ============================================================

/**
 * @brief Tests quadratic relationship between effect size and sample size.
 * @test Verifies that halving effect size approximately quadruples required sample size.
 */
TEST_F(PowerAnalysisTest, Theory_EffectSizeQuadraticRelation) {
    // When effect size is halved, sample size approximately quadruples
    std::size_t n1 = statcpp::sample_size_t_test_one_sample(0.4, 0.80);
    std::size_t n2 = statcpp::sample_size_t_test_one_sample(0.2, 0.80);

    double ratio = static_cast<double>(n2) / n1;
    EXPECT_GT(ratio, 3.5);
    EXPECT_LT(ratio, 4.5);
}

/**
 * @brief Tests that two-sample tests require more observations than one-sample.
 * @test Verifies that each group in two-sample test needs more observations than one-sample test.
 */
TEST_F(PowerAnalysisTest, Theory_TwoSampleNeedsMoreThanOneSample) {
    // Two-sample test requires more samples than one-sample test (comparing per group)
    std::size_t n_one = statcpp::sample_size_t_test_one_sample(0.5, 0.80);
    std::size_t n_two = statcpp::sample_size_t_test_two_sample(0.5, 0.80);

    // Two-sample test requires approximately twice as many samples per group
    EXPECT_GT(n_two, n_one);
}

/**
 * @brief Tests the tradeoff between Type I and Type II error rates.
 * @test Verifies that decreasing alpha requires larger sample size to maintain power.
 */
TEST_F(PowerAnalysisTest, Theory_AlphaBetaTradeoff) {
    // Tradeoff between alpha and beta (1-power)
    // Decreasing alpha requires more samples to achieve the same power
    std::size_t n_alpha05 = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05);
    std::size_t n_alpha01 = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.01);

    EXPECT_LT(n_alpha05, n_alpha01);
}

// ============================================================
// Practical scenario tests
// ============================================================

/**
 * @brief Tests typical clinical trial scenario.
 * @test Verifies sample size and power for typical clinical trial (d=0.5, power=0.9).
 */
TEST_F(PowerAnalysisTest, Practical_ClinicalTrial) {
    // Typical clinical trial scenario
    // Effect size: medium (d=0.5)
    // Power: 90%
    // Significance level: 0.05

    std::size_t n = statcpp::sample_size_t_test_two_sample(0.5, 0.90, 0.05);

    // Verify that 90% power is actually achieved
    double power = statcpp::power_t_test_two_sample(0.5, n, n, 0.05);

    EXPECT_GE(power, 0.89);
    EXPECT_LE(power, 0.92);
}

/**
 * @brief Tests typical A/B testing scenario.
 * @test Verifies sample size and power for typical A/B test (5% to 6% conversion).
 */
TEST_F(PowerAnalysisTest, Practical_ABTesting) {
    // Typical A/B testing scenario
    // Conversion rate: 5% -> 6%
    // Power: 80%

    std::size_t n = statcpp::sample_size_prop_test(0.05, 0.06, 0.80, 0.05);

    // Verify that 80% power is actually achieved
    double power = statcpp::power_prop_test(0.05, 0.06, n, 0.05);

    EXPECT_GE(power, 0.79);
    EXPECT_LE(power, 0.82);
}

/**
 * @brief Tests typical psychology study scenario.
 * @test Verifies that required sample size is in realistic range for psychology (d=0.4).
 */
TEST_F(PowerAnalysisTest, Practical_PsychologyStudy) {
    // Typical psychology study scenario
    // Effect size: somewhat small (d=0.4)
    // Power: 80%

    std::size_t n = statcpp::sample_size_t_test_two_sample(0.4, 0.80, 0.05);

    // Verify that sample size is within a realistic range
    EXPECT_GT(n, 50);
    EXPECT_LT(n, 200);
}

// ============================================================
// Main function
// ============================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
