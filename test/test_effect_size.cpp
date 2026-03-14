#include <gtest/gtest.h>
#include "statcpp/effect_size.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// Cohen's d (One Sample) Tests
// ============================================================================

/**
 * @brief Tests Cohen's d calculation for one sample with known population standard deviation.
 * @test Verifies that Cohen's d is correctly computed when sigma is provided, expecting d = 2.0 for the test data.
 */
TEST(CohensDTest, OneSampleWithKnownSigma) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14
    double mu0 = 10.0;
    double sigma = 2.0;

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0, sigma);

    // d = (14 - 10) / 2 = 2.0
    EXPECT_NEAR(d, 2.0, 1e-10);
}

/**
 * @brief Tests Cohen's d calculation for one sample using sample standard deviation.
 * @test Verifies that Cohen's d is correctly computed using sample standard deviation when sigma is not provided.
 */
TEST(CohensDTest, OneSampleWithSampleStddev) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14, sd = sqrt(10)
    double mu0 = 10.0;

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0);

    // d = (14 - 10) / sqrt(10) ≈ 1.265
    double expected = 4.0 / std::sqrt(10.0);
    EXPECT_NEAR(d, expected, 1e-10);
}

/**
 * @brief Tests Cohen's d when there is zero effect (mean equals comparison value).
 * @test Verifies that Cohen's d equals 0.0 when the sample mean equals the comparison mean.
 */
TEST(CohensDTest, ZeroEffect) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14
    double mu0 = 14.0;  // Same as mean

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0);

    EXPECT_NEAR(d, 0.0, 1e-10);
}

/**
 * @brief Tests Cohen's d with negative effect size.
 * @test Verifies that Cohen's d is negative when the sample mean is less than the comparison mean.
 */
TEST(CohensDTest, NegativeEffect) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14
    double mu0 = 20.0;  // Greater than mean

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0);

    EXPECT_LT(d, 0.0);
}

/**
 * @brief Tests that invalid sigma values throw exceptions.
 * @test Verifies that providing zero or negative sigma throws std::invalid_argument.
 */
TEST(CohensDTest, InvalidSigma) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::cohens_d(data.begin(), data.end(), 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::cohens_d(data.begin(), data.end(), 0.0, -1.0), std::invalid_argument);
}

// ============================================================================
// Cohen's d (Two Sample) Tests
// ============================================================================

/**
 * @brief Tests basic two-sample Cohen's d calculation.
 * @test Verifies that Cohen's d is correctly computed for two independent samples using pooled standard deviation.
 */
TEST(CohensDTwoSampleTest, BasicComputation) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};  // mean = 14
    std::vector<double> data2 = {5.0, 7.0, 9.0, 11.0, 13.0};     // mean = 9

    double d = statcpp::cohens_d_two_sample(data1.begin(), data1.end(),
                                             data2.begin(), data2.end());

    // Difference = 5, pooled sd should be sqrt(10) for both groups
    double expected = 5.0 / std::sqrt(10.0);
    EXPECT_NEAR(d, expected, 1e-10);
}

/**
 * @brief Tests two-sample Cohen's d with minimal difference between groups.
 * @test Verifies that Cohen's d reflects small effect size when groups have minimal mean difference.
 */
TEST(CohensDTwoSampleTest, NoDifference) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {11.0, 13.0, 15.0, 17.0, 19.0};  // Same variance, +1 shift

    double d = statcpp::cohens_d_two_sample(data1.begin(), data1.end(),
                                             data2.begin(), data2.end());

    // Very small effect (just 1 unit difference with sd = sqrt(10))
    EXPECT_NEAR(std::abs(d), 1.0 / std::sqrt(10.0), 1e-10);
}

/**
 * @brief Tests that too few elements throws an exception.
 * @test Verifies that insufficient sample size throws std::invalid_argument.
 */
TEST(CohensDTwoSampleTest, TooFewElements) {
    std::vector<double> data1 = {1.0};
    std::vector<double> data2 = {1.0, 2.0, 3.0};

    EXPECT_THROW(statcpp::cohens_d_two_sample(data1.begin(), data1.end(),
                                               data2.begin(), data2.end()),
                 std::invalid_argument);
}

// ============================================================================
// Hedges' g Tests
// ============================================================================

/**
 * @brief Tests that Hedges' g is smaller than Cohen's d due to bias correction.
 * @test Verifies that the bias-corrected Hedges' g is smaller in absolute value than Cohen's d.
 */
TEST(HedgesGTest, SmallerThanCohensD) {
    std::vector<double> data = {10.0, 12.0, 14.0, 16.0, 18.0};
    double mu0 = 10.0;

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0);
    double g = statcpp::hedges_g(data.begin(), data.end(), mu0);

    // Hedges' g should be smaller (bias-corrected)
    EXPECT_LT(std::abs(g), std::abs(d));
}

/**
 * @brief Tests that Hedges' g converges to Cohen's d with large sample size.
 * @test Verifies that for large n, Hedges' g approaches Cohen's d (ratio close to 1.0).
 */
TEST(HedgesGTest, ConvergesToCohensDForLargeN) {
    // Generate larger sample
    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<double>(i));
    }
    double mu0 = 0.0;

    double d = statcpp::cohens_d(data.begin(), data.end(), mu0);
    double g = statcpp::hedges_g(data.begin(), data.end(), mu0);

    // For large n, g should be close to d
    EXPECT_NEAR(g / d, 1.0, 0.05);
}

/**
 * @brief Tests basic two-sample Hedges' g calculation.
 * @test Verifies that two-sample Hedges' g is smaller than Cohen's d due to bias correction.
 */
TEST(HedgesGTwoSampleTest, BasicComputation) {
    std::vector<double> data1 = {10.0, 12.0, 14.0, 16.0, 18.0};
    std::vector<double> data2 = {5.0, 7.0, 9.0, 11.0, 13.0};

    double d = statcpp::cohens_d_two_sample(data1.begin(), data1.end(),
                                             data2.begin(), data2.end());
    double g = statcpp::hedges_g_two_sample(data1.begin(), data1.end(),
                                             data2.begin(), data2.end());

    EXPECT_LT(std::abs(g), std::abs(d));
}

// ============================================================================
// Glass's Delta Tests
// ============================================================================

/**
 * @brief Tests basic Glass's delta calculation.
 * @test Verifies that Glass's delta is correctly computed using control group standard deviation.
 */
TEST(GlassDeltaTest, BasicComputation) {
    std::vector<double> control = {10.0, 12.0, 14.0, 16.0, 18.0};     // mean = 14, sd = sqrt(10)
    std::vector<double> treatment = {15.0, 17.0, 19.0, 21.0, 23.0};   // mean = 19

    double delta = statcpp::glass_delta(control.begin(), control.end(),
                                         treatment.begin(), treatment.end());

    // delta = (19 - 14) / sqrt(10)
    double expected = 5.0 / std::sqrt(10.0);
    EXPECT_NEAR(delta, expected, 1e-10);
}

/**
 * @brief Tests that Glass's delta uses only control group standard deviation.
 * @test Verifies that Glass's delta uses control SD even when treatment group has different variance.
 */
TEST(GlassDeltaTest, UsesControlSD) {
    std::vector<double> control = {10.0, 10.1, 10.2, 10.3, 10.4};     // Low variance
    std::vector<double> treatment = {5.0, 15.0, 8.0, 12.0, 20.0};     // High variance, mean = 12

    double delta = statcpp::glass_delta(control.begin(), control.end(),
                                         treatment.begin(), treatment.end());

    // Uses control SD (very small), so delta should be large
    double control_mean = statcpp::mean(control.begin(), control.end());
    double treatment_mean = statcpp::mean(treatment.begin(), treatment.end());
    double control_sd = statcpp::sample_stddev(control.begin(), control.end());

    double expected = (treatment_mean - control_mean) / control_sd;
    EXPECT_NEAR(delta, expected, 1e-10);
}

// ============================================================================
// Effect Size Conversions Tests
// ============================================================================

/**
 * @brief Tests conversion from t-statistic to correlation coefficient.
 * @test Verifies that t-to-r conversion formula produces correct correlation values.
 */
TEST(TToRTest, BasicConversion) {
    double t = 2.0;
    double df = 20.0;

    double r = statcpp::t_to_r(t, df);

    // r = t / sqrt(t^2 + df) = 2 / sqrt(4 + 20) = 2 / sqrt(24)
    double expected = 2.0 / std::sqrt(24.0);
    EXPECT_NEAR(r, expected, 1e-10);
}

/**
 * @brief Tests conversion from Cohen's d to correlation coefficient.
 * @test Verifies that d-to-r conversion formula produces correct correlation values.
 */
TEST(DToRTest, BasicConversion) {
    double d = 0.5;
    double r = statcpp::d_to_r(d);

    // r = d / sqrt(d^2 + 4) = 0.5 / sqrt(0.25 + 4) = 0.5 / sqrt(4.25)
    double expected = 0.5 / std::sqrt(4.25);
    EXPECT_NEAR(r, expected, 1e-10);
}

/**
 * @brief Tests conversion from correlation coefficient to Cohen's d.
 * @test Verifies that r-to-d conversion formula produces correct effect size values.
 */
TEST(RToDTest, BasicConversion) {
    double r = 0.3;
    double d = statcpp::r_to_d(r);

    // d = 2r / sqrt(1 - r^2) = 0.6 / sqrt(0.91)
    double expected = 0.6 / std::sqrt(0.91);
    EXPECT_NEAR(d, expected, 1e-10);
}

/**
 * @brief Tests that r-to-d is the inverse of d-to-r conversion.
 * @test Verifies that converting d→r→d returns the original Cohen's d value.
 */
TEST(RToDTest, InverseOfDToR) {
    double original_d = 0.8;
    double r = statcpp::d_to_r(original_d);
    double converted_d = statcpp::r_to_d(r);

    EXPECT_NEAR(converted_d, original_d, 1e-10);
}

/**
 * @brief Tests that invalid correlation values throw exceptions.
 * @test Verifies that r = ±1.0 throws std::invalid_argument in r-to-d conversion.
 */
TEST(RToDTest, InvalidR) {
    EXPECT_THROW(statcpp::r_to_d(1.0), std::invalid_argument);
    EXPECT_THROW(statcpp::r_to_d(-1.0), std::invalid_argument);
}

// ============================================================================
// Eta-Squared Tests
// ============================================================================

/**
 * @brief Tests basic eta-squared calculation.
 * @test Verifies that eta-squared is correctly computed from sum of squares.
 */
TEST(EtaSquaredTest, BasicComputation) {
    double ss_effect = 100.0;
    double ss_total = 500.0;

    double eta2 = statcpp::eta_squared(ss_effect, ss_total);

    EXPECT_NEAR(eta2, 0.2, 1e-10);
}

/**
 * @brief Tests that invalid sum of squares values throw exceptions.
 * @test Verifies that zero or negative total sum of squares throws std::invalid_argument.
 */
TEST(EtaSquaredTest, InvalidInput) {
    EXPECT_THROW(statcpp::eta_squared(100.0, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::eta_squared(100.0, -100.0), std::invalid_argument);
}

/**
 * @brief Tests basic partial eta-squared calculation.
 * @test Verifies that partial eta-squared is correctly computed from F-statistic and degrees of freedom.
 */
TEST(PartialEtaSquaredTest, BasicComputation) {
    double f = 5.0;
    double df1 = 2.0;
    double df2 = 50.0;

    double partial_eta2 = statcpp::partial_eta_squared(f, df1, df2);

    // partial_eta2 = (f * df1) / (f * df1 + df2) = 10 / 60
    EXPECT_NEAR(partial_eta2, 10.0 / 60.0, 1e-10);
}

// ============================================================================
// Omega-Squared Tests
// ============================================================================

/**
 * @brief Tests basic omega-squared calculation.
 * @test Verifies that omega-squared is correctly computed as an unbiased effect size estimate.
 */
TEST(OmegaSquaredTest, BasicComputation) {
    double ss_effect = 100.0;
    double ss_total = 500.0;
    double ms_error = 10.0;
    double df_effect = 2.0;

    double omega2 = statcpp::omega_squared(ss_effect, ss_total, ms_error, df_effect);

    // omega2 = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    //        = (100 - 20) / (500 + 10) = 80 / 510
    EXPECT_NEAR(omega2, 80.0 / 510.0, 1e-10);
}

// ============================================================================
// Cohen's h Tests
// ============================================================================

/**
 * @brief Tests basic Cohen's h calculation for proportions.
 * @test Verifies that Cohen's h is correctly computed for two proportions using arcsine transformation.
 */
TEST(CohensHTest, BasicComputation) {
    double p1 = 0.6;
    double p2 = 0.4;

    double h = statcpp::cohens_h(p1, p2);

    // h = 2 * (asin(sqrt(0.6)) - asin(sqrt(0.4)))
    double expected = 2.0 * (std::asin(std::sqrt(0.6)) - std::asin(std::sqrt(0.4)));
    EXPECT_NEAR(h, expected, 1e-10);
}

/**
 * @brief Tests Cohen's h with zero effect (equal proportions).
 * @test Verifies that Cohen's h equals 0.0 when both proportions are equal.
 */
TEST(CohensHTest, ZeroEffect) {
    double h = statcpp::cohens_h(0.5, 0.5);
    EXPECT_NEAR(h, 0.0, 1e-10);
}

/**
 * @brief Tests that invalid proportions throw exceptions.
 * @test Verifies that proportions outside [0, 1] throw std::invalid_argument.
 */
TEST(CohensHTest, InvalidProportions) {
    EXPECT_THROW(statcpp::cohens_h(-0.1, 0.5), std::invalid_argument);
    EXPECT_THROW(statcpp::cohens_h(0.5, 1.1), std::invalid_argument);
}

// ============================================================================
// Odds Ratio and Risk Ratio Tests
// ============================================================================

/**
 * @brief Tests basic odds ratio calculation from 2x2 contingency table.
 * @test Verifies that odds ratio is correctly computed as (a*d)/(b*c).
 */
TEST(OddsRatioTest, BasicComputation) {
    // 2x2 table: a=30, b=10, c=20, d=40
    double or_val = statcpp::odds_ratio(30.0, 10.0, 20.0, 40.0);

    // OR = (30 * 40) / (10 * 20) = 1200 / 200 = 6.0
    EXPECT_NEAR(or_val, 6.0, 1e-10);
}

/**
 * @brief Tests odds ratio with no association.
 * @test Verifies that odds ratio equals 1.0 when all cells are equal (no association).
 */
TEST(OddsRatioTest, NoAssociation) {
    double or_val = statcpp::odds_ratio(25.0, 25.0, 25.0, 25.0);
    EXPECT_NEAR(or_val, 1.0, 1e-10);
}

/**
 * @brief Tests that zero cells in contingency table throw exceptions.
 * @test Verifies that zero values in any cell throw std::invalid_argument.
 */
TEST(OddsRatioTest, ZeroCell) {
    EXPECT_THROW(statcpp::odds_ratio(30.0, 0.0, 20.0, 40.0), std::invalid_argument);
    EXPECT_THROW(statcpp::odds_ratio(30.0, 10.0, 0.0, 40.0), std::invalid_argument);
}

/**
 * @brief Tests basic risk ratio calculation.
 * @test Verifies that risk ratio is correctly computed as (a/(a+b))/(c/(c+d)).
 */
TEST(RiskRatioTest, BasicComputation) {
    // 2x2 table: a=30, b=70, c=20, d=80
    double rr = statcpp::risk_ratio(30.0, 70.0, 20.0, 80.0);

    // Risk1 = 30/100 = 0.3, Risk2 = 20/100 = 0.2
    // RR = 0.3 / 0.2 = 1.5
    EXPECT_NEAR(rr, 1.5, 1e-10);
}

/**
 * @brief Tests that zero risk in denominator throws exception.
 * @test Verifies that zero risk in control group throws std::invalid_argument.
 */
TEST(RiskRatioTest, ZeroRisk) {
    EXPECT_THROW(statcpp::risk_ratio(30.0, 70.0, 0.0, 100.0), std::invalid_argument);
}

// ============================================================================
// Effect Size Interpretation Tests
// ============================================================================

/**
 * @brief Tests interpretation of negligible Cohen's d values.
 * @test Verifies that Cohen's d values < 0.2 are classified as negligible effect size.
 */
TEST(InterpretCohensDTest, Negligible) {
    EXPECT_EQ(statcpp::interpret_cohens_d(0.1), statcpp::effect_size_magnitude::negligible);
    EXPECT_EQ(statcpp::interpret_cohens_d(-0.1), statcpp::effect_size_magnitude::negligible);
}

/**
 * @brief Tests interpretation of small Cohen's d values.
 * @test Verifies that Cohen's d ≈ 0.3 is classified as small effect size.
 */
TEST(InterpretCohensDTest, Small) {
    EXPECT_EQ(statcpp::interpret_cohens_d(0.3), statcpp::effect_size_magnitude::small);
    EXPECT_EQ(statcpp::interpret_cohens_d(-0.3), statcpp::effect_size_magnitude::small);
}

/**
 * @brief Tests interpretation of medium Cohen's d values.
 * @test Verifies that Cohen's d ≈ 0.6 is classified as medium effect size.
 */
TEST(InterpretCohensDTest, Medium) {
    EXPECT_EQ(statcpp::interpret_cohens_d(0.6), statcpp::effect_size_magnitude::medium);
    EXPECT_EQ(statcpp::interpret_cohens_d(-0.6), statcpp::effect_size_magnitude::medium);
}

/**
 * @brief Tests interpretation of large Cohen's d values.
 * @test Verifies that Cohen's d ≈ 1.0 is classified as large effect size.
 */
TEST(InterpretCohensDTest, Large) {
    EXPECT_EQ(statcpp::interpret_cohens_d(1.0), statcpp::effect_size_magnitude::large);
    EXPECT_EQ(statcpp::interpret_cohens_d(-1.0), statcpp::effect_size_magnitude::large);
}

/**
 * @brief Tests interpretation of negligible correlation values.
 * @test Verifies that correlations < 0.1 are classified as negligible effect size.
 */
TEST(InterpretCorrelationTest, Negligible) {
    EXPECT_EQ(statcpp::interpret_correlation(0.05), statcpp::effect_size_magnitude::negligible);
}

/**
 * @brief Tests interpretation of small correlation values.
 * @test Verifies that correlation ≈ 0.2 is classified as small effect size.
 */
TEST(InterpretCorrelationTest, Small) {
    EXPECT_EQ(statcpp::interpret_correlation(0.2), statcpp::effect_size_magnitude::small);
}

/**
 * @brief Tests interpretation of medium correlation values.
 * @test Verifies that correlation ≈ 0.4 is classified as medium effect size.
 */
TEST(InterpretCorrelationTest, Medium) {
    EXPECT_EQ(statcpp::interpret_correlation(0.4), statcpp::effect_size_magnitude::medium);
}

/**
 * @brief Tests interpretation of large correlation values.
 * @test Verifies that correlation ≈ 0.6 is classified as large effect size.
 */
TEST(InterpretCorrelationTest, Large) {
    EXPECT_EQ(statcpp::interpret_correlation(0.6), statcpp::effect_size_magnitude::large);
}

/**
 * @brief Tests interpretation of negligible eta-squared values.
 * @test Verifies that eta-squared < 0.01 is classified as negligible effect size.
 */
TEST(InterpretEtaSquaredTest, Negligible) {
    EXPECT_EQ(statcpp::interpret_eta_squared(0.005), statcpp::effect_size_magnitude::negligible);
}

/**
 * @brief Tests interpretation of small eta-squared values.
 * @test Verifies that eta-squared ≈ 0.03 is classified as small effect size.
 */
TEST(InterpretEtaSquaredTest, Small) {
    EXPECT_EQ(statcpp::interpret_eta_squared(0.03), statcpp::effect_size_magnitude::small);
}

/**
 * @brief Tests interpretation of medium eta-squared values.
 * @test Verifies that eta-squared ≈ 0.10 is classified as medium effect size.
 */
TEST(InterpretEtaSquaredTest, Medium) {
    EXPECT_EQ(statcpp::interpret_eta_squared(0.10), statcpp::effect_size_magnitude::medium);
}

/**
 * @brief Tests interpretation of large eta-squared values.
 * @test Verifies that eta-squared ≈ 0.20 is classified as large effect size.
 */
TEST(InterpretEtaSquaredTest, Large) {
    EXPECT_EQ(statcpp::interpret_eta_squared(0.20), statcpp::effect_size_magnitude::large);
}
