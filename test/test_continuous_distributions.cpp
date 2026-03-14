#include <gtest/gtest.h>
#include "statcpp/continuous_distributions.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>

// ============================================================================
// Uniform Distribution Tests
// ============================================================================

/**
 * @brief Test uniform PDF with standard uniform distribution (0, 1).
 * @test Verifies PDF values at boundary and outside the range.
 */
TEST(UniformPdfTest, StandardUniform) {
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(0.5), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(0.0), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(1.0), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(-0.1), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(1.1), 0.0);
}

/**
 * @brief Test uniform PDF with custom range.
 * @test Verifies PDF value for a custom range [0, 10].
 */
TEST(UniformPdfTest, CustomRange) {
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(5.0, 0.0, 10.0), 0.1);
    EXPECT_DOUBLE_EQ(statcpp::uniform_pdf(-1.0, 0.0, 10.0), 0.0);
}

/**
 * @brief Test uniform CDF with standard uniform distribution.
 * @test Verifies CDF values at key points (0.0, 0.5, 1.0).
 */
TEST(UniformCdfTest, StandardUniform) {
    EXPECT_DOUBLE_EQ(statcpp::uniform_cdf(0.0), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_cdf(0.5), 0.5);
    EXPECT_DOUBLE_EQ(statcpp::uniform_cdf(1.0), 1.0);
}

/**
 * @brief Test uniform quantile function with standard uniform distribution.
 * @test Verifies quantile values at key probabilities.
 */
TEST(UniformQuantileTest, StandardUniform) {
    EXPECT_DOUBLE_EQ(statcpp::uniform_quantile(0.0), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::uniform_quantile(0.5), 0.5);
    EXPECT_DOUBLE_EQ(statcpp::uniform_quantile(1.0), 1.0);
}

/**
 * @brief Test inverse relationship between uniform CDF and quantile.
 * @test Verifies that quantile(cdf(x)) = x.
 */
TEST(UniformQuantileTest, InverseProperty) {
    double x = 0.3;
    EXPECT_NEAR(statcpp::uniform_quantile(statcpp::uniform_cdf(x)), x, 1e-10);
}

/**
 * @brief Test uniform random number generator.
 * @test Verifies that generated random numbers are within the expected range.
 */
TEST(UniformRandTest, InRange) {
    statcpp::set_seed(42);
    for (int i = 0; i < 100; ++i) {
        double r = statcpp::uniform_rand(0.0, 1.0);
        EXPECT_GE(r, 0.0);
        EXPECT_LE(r, 1.0);
    }
}

// ============================================================================
// Normal Distribution Tests
// ============================================================================

/**
 * @brief Test normal PDF with standard normal distribution.
 * @test Verifies PDF value at mean and symmetry property.
 */
TEST(NormalPdfTest, StandardNormal) {
    // PDF at mean should be 1/sqrt(2*pi)
    EXPECT_NEAR(statcpp::normal_pdf(0.0), 1.0 / statcpp::sqrt_2_pi, 1e-10);
    // Symmetry
    EXPECT_NEAR(statcpp::normal_pdf(1.0), statcpp::normal_pdf(-1.0), 1e-10);
}

/**
 * @brief Test normal PDF with custom parameters.
 * @test Verifies PDF value at mean with custom mu and sigma.
 */
TEST(NormalPdfTest, CustomParameters) {
    // PDF at mean should be 1/(sigma * sqrt(2*pi))
    double mu = 5.0, sigma = 2.0;
    EXPECT_NEAR(statcpp::normal_pdf(mu, mu, sigma), 1.0 / (sigma * statcpp::sqrt_2_pi), 1e-10);
}

/**
 * @brief Test normal CDF with standard normal distribution.
 * @test Verifies CDF values at key points (0.0, 1.96, -1.96).
 */
TEST(NormalCdfTest, StandardNormal) {
    EXPECT_NEAR(statcpp::normal_cdf(0.0), 0.5, 1e-8);
    EXPECT_NEAR(statcpp::normal_cdf(1.96), 0.975, 1e-2);
    EXPECT_NEAR(statcpp::normal_cdf(-1.96), 0.025, 1e-2);
}

/**
 * @brief Test normal quantile function with standard normal distribution.
 * @test Verifies quantile values at key probabilities.
 */
TEST(NormalQuantileTest, StandardNormal) {
    EXPECT_NEAR(statcpp::normal_quantile(0.5), 0.0, 1e-10);
    EXPECT_NEAR(statcpp::normal_quantile(0.975), 1.96, 1e-2);
}

/**
 * @brief Test inverse relationship between normal CDF and quantile.
 * @test Verifies that quantile(cdf(x)) = x.
 */
TEST(NormalQuantileTest, InverseProperty) {
    double x = 1.5;
    EXPECT_NEAR(statcpp::normal_quantile(statcpp::normal_cdf(x)), x, 1e-6);
}

/**
 * @brief Test normal random number generator.
 * @test Verifies that sample mean converges to expected mean.
 */
TEST(NormalRandTest, MeanAndVariance) {
    statcpp::set_seed(42);
    std::vector<double> samples(10000);
    for (auto& s : samples) {
        s = statcpp::normal_rand(0.0, 1.0);
    }
    double mean_val = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    EXPECT_NEAR(mean_val, 0.0, 0.1);
}

// ============================================================================
// Exponential Distribution Tests
// ============================================================================

/**
 * @brief Test exponential PDF with basic values.
 * @test Verifies PDF values at x=0, x=1, and negative x.
 */
TEST(ExponentialPdfTest, BasicValues) {
    EXPECT_NEAR(statcpp::exponential_pdf(0.0, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::exponential_pdf(1.0, 1.0), std::exp(-1.0), 1e-10);
    EXPECT_DOUBLE_EQ(statcpp::exponential_pdf(-1.0, 1.0), 0.0);
}

/**
 * @brief Test exponential CDF with basic values.
 * @test Verifies CDF values at x=0 and x=1.
 */
TEST(ExponentialCdfTest, BasicValues) {
    EXPECT_DOUBLE_EQ(statcpp::exponential_cdf(0.0, 1.0), 0.0);
    EXPECT_NEAR(statcpp::exponential_cdf(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-10);
}

/**
 * @brief Test inverse relationship between exponential CDF and quantile.
 * @test Verifies that quantile(cdf(x)) = x.
 */
TEST(ExponentialQuantileTest, InverseProperty) {
    double x = 2.0;
    double lambda = 1.5;
    EXPECT_NEAR(statcpp::exponential_quantile(statcpp::exponential_cdf(x, lambda), lambda), x, 1e-10);
}

// ============================================================================
// Gamma Distribution Tests
// ============================================================================

/**
 * @brief Test gamma PDF special case: Gamma(1, rate) = Exponential(rate).
 * @test Verifies that gamma PDF matches exponential PDF when shape=1.
 */
TEST(GammaPdfTest, ExponentialSpecialCase) {
    // Gamma(1, rate) = Exponential(rate)
    double rate = 2.0;
    double x = 1.5;
    EXPECT_NEAR(statcpp::gamma_pdf(x, 1.0, rate), statcpp::exponential_pdf(x, rate), 1e-10);
}

/**
 * @brief Test gamma CDF special case: Gamma(1, rate) = Exponential(rate).
 * @test Verifies that gamma CDF matches exponential CDF when shape=1.
 */
TEST(GammaCdfTest, ExponentialSpecialCase) {
    double rate = 2.0;
    double x = 1.5;
    EXPECT_NEAR(statcpp::gamma_cdf(x, 1.0, rate), statcpp::exponential_cdf(x, rate), 1e-10);
}

/**
 * @brief Test inverse relationship between gamma CDF and quantile.
 * @test Verifies that cdf(quantile(p)) = p.
 */
TEST(GammaQuantileTest, InverseProperty) {
    double shape = 3.0, rate = 2.0;
    double p = 0.7;
    double x = statcpp::gamma_quantile(p, shape, rate);
    EXPECT_NEAR(statcpp::gamma_cdf(x, shape, rate), p, 1e-8);
}

// ============================================================================
// Beta Distribution Tests
// ============================================================================

/**
 * @brief Test beta PDF special case: Beta(1, 1) = Uniform(0, 1).
 * @test Verifies that beta PDF equals 1.0 when alpha=beta=1.
 */
TEST(BetaPdfTest, UniformSpecialCase) {
    // Beta(1, 1) = Uniform(0, 1)
    EXPECT_NEAR(statcpp::beta_pdf(0.5, 1.0, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::beta_pdf(0.3, 1.0, 1.0), 1.0, 1e-10);
}

/**
 * @brief Test beta CDF special case: Beta(1, 1) = Uniform(0, 1).
 * @test Verifies that beta CDF equals x when alpha=beta=1.
 */
TEST(BetaCdfTest, UniformSpecialCase) {
    // Beta(1, 1) CDF = x
    EXPECT_NEAR(statcpp::beta_cdf(0.5, 1.0, 1.0), 0.5, 1e-10);
    EXPECT_NEAR(statcpp::beta_cdf(0.3, 1.0, 1.0), 0.3, 1e-10);
}

/**
 * @brief Test inverse relationship between beta CDF and quantile.
 * @test Verifies that cdf(quantile(p)) = p.
 */
TEST(BetaQuantileTest, InverseProperty) {
    double alpha = 2.0, beta_param = 3.0;
    double p = 0.6;
    double x = statcpp::beta_quantile(p, alpha, beta_param);
    EXPECT_NEAR(statcpp::beta_cdf(x, alpha, beta_param), p, 1e-8);
}

/**
 * @brief Test beta random number generator.
 * @test Verifies that generated random numbers are within (0, 1).
 */
TEST(BetaRandTest, InRange) {
    statcpp::set_seed(42);
    for (int i = 0; i < 100; ++i) {
        double r = statcpp::beta_rand(2.0, 3.0);
        EXPECT_GT(r, 0.0);
        EXPECT_LT(r, 1.0);
    }
}

// ============================================================================
// Chi-Square Distribution Tests
// ============================================================================

/**
 * @brief Test chi-square PDF relation to gamma distribution.
 * @test Verifies that chi-square(df) = Gamma(df/2, 1/2).
 */
TEST(ChisqPdfTest, GammaRelation) {
    // Chi-square(df) = Gamma(df/2, 1/2)
    double df = 5.0;
    double x = 3.0;
    EXPECT_NEAR(statcpp::chisq_pdf(x, df), statcpp::gamma_pdf(x, df/2.0, 0.5), 1e-10);
}

/**
 * @brief Test chi-square CDF with known critical values.
 * @test Verifies CDF values against known chi-square critical values.
 */
TEST(ChisqCdfTest, KnownValues) {
    // Chi-square critical values
    EXPECT_NEAR(statcpp::chisq_cdf(3.841, 1.0), 0.95, 1e-2);
    EXPECT_NEAR(statcpp::chisq_cdf(5.991, 2.0), 0.95, 1e-2);
}

/**
 * @brief Test inverse relationship between chi-square CDF and quantile.
 * @test Verifies that cdf(quantile(p)) = p.
 */
TEST(ChisqQuantileTest, InverseProperty) {
    double df = 10.0;
    double p = 0.95;
    double x = statcpp::chisq_quantile(p, df);
    EXPECT_NEAR(statcpp::chisq_cdf(x, df), p, 1e-8);
}

// ============================================================================
// Student's t-Distribution Tests
// ============================================================================

/**
 * @brief Test t-distribution PDF symmetry.
 * @test Verifies that PDF is symmetric around zero.
 */
TEST(TPdfTest, Symmetry) {
    double df = 5.0;
    EXPECT_NEAR(statcpp::t_pdf(1.0, df), statcpp::t_pdf(-1.0, df), 1e-10);
}

/**
 * @brief Test t-distribution convergence to normal as df increases.
 * @test Verifies that t-distribution approaches normal distribution for large df.
 */
TEST(TPdfTest, NormalLimit) {
    // As df -> infinity, t approaches normal
    double x = 1.0;
    double t_pdf_large_df = statcpp::t_pdf(x, 1000.0);
    double normal_pdf_val = statcpp::normal_pdf(x);
    EXPECT_NEAR(t_pdf_large_df, normal_pdf_val, 1e-3);
}

/**
 * @brief Test t-distribution CDF symmetry.
 * @test Verifies that CDF(x) + CDF(-x) = 1.
 */
TEST(TCdfTest, Symmetry) {
    double df = 5.0;
    EXPECT_NEAR(statcpp::t_cdf(1.0, df) + statcpp::t_cdf(-1.0, df), 1.0, 1e-10);
}

/**
 * @brief Test t-distribution CDF with known critical values.
 * @test Verifies CDF values against known t-distribution critical values.
 */
TEST(TCdfTest, KnownValues) {
    // t critical values (two-tailed 95%)
    EXPECT_NEAR(statcpp::t_cdf(2.776, 4.0), 0.975, 1e-2);
    EXPECT_NEAR(statcpp::t_cdf(2.228, 10.0), 0.975, 1e-2);
}

/**
 * @brief Test inverse relationship between t-distribution CDF and quantile.
 * @test Verifies that cdf(quantile(p)) = p.
 */
TEST(TQuantileTest, InverseProperty) {
    double df = 10.0;
    double p = 0.975;
    double x = statcpp::t_quantile(p, df);
    EXPECT_NEAR(statcpp::t_cdf(x, df), p, 1e-8);
}

// ============================================================================
// F-Distribution Tests
// ============================================================================

/**
 * @brief Test F-distribution PDF support on positive values only.
 * @test Verifies that PDF is zero for non-positive values.
 */
TEST(FPdfTest, PositiveOnly) {
    EXPECT_DOUBLE_EQ(statcpp::f_pdf(0.0, 5.0, 10.0), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::f_pdf(-1.0, 5.0, 10.0), 0.0);
    EXPECT_GT(statcpp::f_pdf(1.0, 5.0, 10.0), 0.0);
}

/**
 * @brief Test F-distribution CDF with known critical values.
 * @test Verifies CDF values against known F-distribution critical values.
 */
TEST(FCdfTest, KnownValues) {
    // F critical values
    EXPECT_NEAR(statcpp::f_cdf(4.26, 3.0, 10.0), 0.95, 2e-2);
    EXPECT_NEAR(statcpp::f_cdf(3.89, 2.0, 20.0), 0.95, 2e-2);
}

/**
 * @brief Test inverse relationship between F-distribution CDF and quantile.
 * @test Verifies that cdf(quantile(p)) = p.
 */
TEST(FQuantileTest, InverseProperty) {
    double df1 = 5.0, df2 = 10.0;
    double p = 0.95;
    double x = statcpp::f_quantile(p, df1, df2);
    EXPECT_NEAR(statcpp::f_cdf(x, df1, df2), p, 1e-6);
}

/**
 * @brief Test F-distribution quantile boundary values.
 * @test Verifies quantile values at p=0 and p=1.
 */
TEST(FQuantileTest, BoundaryValues) {
    EXPECT_DOUBLE_EQ(statcpp::f_quantile(0.0, 5.0, 10.0), 0.0);
    EXPECT_EQ(statcpp::f_quantile(1.0, 5.0, 10.0), std::numeric_limits<double>::infinity());
}

// ============================================================================
// Weibull Distribution Tests
// ============================================================================

/**
 * @brief Test Weibull PDF at x=0 boundary for shape < 1.
 * @test Verifies that PDF returns infinity when shape < 1 and x = 0.
 */
TEST(WeibullPdfTest, BoundaryXZeroShapeLessThan1) {
    // shape < 1: PDF(0) = +infinity
    double result = statcpp::weibull_pdf(0.0, 0.5, 1.0);
    EXPECT_EQ(result, std::numeric_limits<double>::infinity());
}

/**
 * @brief Test Weibull PDF at x=0 boundary for shape = 1.
 * @test Verifies that PDF returns 1/scale when shape = 1 and x = 0 (exponential case).
 */
TEST(WeibullPdfTest, BoundaryXZeroShapeEqual1) {
    // shape = 1: PDF(0) = 1/scale (exponential distribution)
    double scale = 2.0;
    double result = statcpp::weibull_pdf(0.0, 1.0, scale);
    EXPECT_DOUBLE_EQ(result, 1.0 / scale);
}

/**
 * @brief Test Weibull PDF at x=0 boundary for shape > 1.
 * @test Verifies that PDF returns 0 when shape > 1 and x = 0.
 */
TEST(WeibullPdfTest, BoundaryXZeroShapeGreaterThan1) {
    // shape > 1: PDF(0) = 0
    double result = statcpp::weibull_pdf(0.0, 2.0, 1.0);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

/**
 * @brief Test Weibull PDF at positive x values.
 * @test Verifies that PDF returns positive values for positive x.
 */
TEST(WeibullPdfTest, PositiveX) {
    double result = statcpp::weibull_pdf(1.0, 2.0, 1.0);
    EXPECT_GT(result, 0.0);
}

// ============================================================================
// Invalid Input Tests
// ============================================================================

/**
 * @brief Test exception handling for invalid distribution parameters.
 * @test Verifies that appropriate exceptions are thrown for invalid parameters.
 */
TEST(DistributionExceptionTest, InvalidParameters) {
    // Uniform: a >= b
    EXPECT_THROW(statcpp::uniform_pdf(0.5, 1.0, 0.0), std::invalid_argument);

    // Normal: sigma <= 0
    EXPECT_THROW(statcpp::normal_pdf(0.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::normal_pdf(0.0, 0.0, -1.0), std::invalid_argument);

    // Exponential: lambda <= 0
    EXPECT_THROW(statcpp::exponential_pdf(1.0, 0.0), std::invalid_argument);

    // Gamma: shape <= 0 or rate <= 0
    EXPECT_THROW(statcpp::gamma_pdf(1.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(statcpp::gamma_pdf(1.0, 1.0, 0.0), std::invalid_argument);

    // Beta: alpha <= 0 or beta <= 0
    EXPECT_THROW(statcpp::beta_pdf(0.5, 0.0, 1.0), std::invalid_argument);

    // Chi-square: df <= 0
    EXPECT_THROW(statcpp::chisq_pdf(1.0, 0.0), std::invalid_argument);

    // t: df <= 0
    EXPECT_THROW(statcpp::t_pdf(0.0, 0.0), std::invalid_argument);

    // F: df1 <= 0 or df2 <= 0
    EXPECT_THROW(statcpp::f_pdf(1.0, 0.0, 10.0), std::invalid_argument);
    EXPECT_THROW(statcpp::f_pdf(1.0, 10.0, 0.0), std::invalid_argument);
}

// ============================================================================
// Studentized Range Distribution Tests
// ============================================================================

/**
 * @brief Test studentized range CDF against R ptukey values.
 * @test Validates CDF output against R's ptukey() for various (q, k, df) combinations.
 */
TEST(StudentizedRangeCdfTest, AgainstR) {
    // R: ptukey(3.0, 2, 10)
    EXPECT_NEAR(statcpp::studentized_range_cdf(3.0, 2, 10), 0.940109675612279, 1e-4);
    // R: ptukey(3.0, 3, 10)
    EXPECT_NEAR(statcpp::studentized_range_cdf(3.0, 3, 10), 0.865016584810091, 1e-4);
    // R: ptukey(3.53, 3, 12)
    EXPECT_NEAR(statcpp::studentized_range_cdf(3.53, 3, 12), 0.932524585775835, 1e-4);
    // R: ptukey(4.20, 4, 20)
    EXPECT_NEAR(statcpp::studentized_range_cdf(4.20, 4, 20), 0.964911490330727, 1e-4);
    // R: ptukey(3.77, 3, 30)
    EXPECT_NEAR(statcpp::studentized_range_cdf(3.77, 3, 30), 0.968081290679369, 1e-4);
    // R: ptukey(5.0, 5, 60)
    EXPECT_NEAR(statcpp::studentized_range_cdf(5.0, 5, 60), 0.993166417135187, 1e-4);
}

/**
 * @brief Test studentized range CDF special case k=2.
 * @test For k=2, CDF should equal 2*t_cdf(q/sqrt(2), df) - 1.
 */
TEST(StudentizedRangeCdfTest, K2SpecialCase) {
    double q = 3.0;
    double df = 10.0;
    double expected = 2.0 * statcpp::t_cdf(q / std::sqrt(2.0), df) - 1.0;
    EXPECT_NEAR(statcpp::studentized_range_cdf(q, 2, df), expected, 1e-10);
}

/**
 * @brief Test studentized range CDF with large df (asymptotic branch).
 * @test Validates that the asymptotic branch (df > 25000) gives correct results.
 */
TEST(StudentizedRangeCdfTest, LargeDf) {
    // R: ptukey(4.0, 3, 100000)
    EXPECT_NEAR(statcpp::studentized_range_cdf(4.0, 3, 100000), 0.987012338626815, 1e-4);
}

/**
 * @brief Test studentized range CDF boundary values.
 * @test q <= 0 should return 0, very large q should return ~1.
 */
TEST(StudentizedRangeCdfTest, BoundaryValues) {
    EXPECT_DOUBLE_EQ(statcpp::studentized_range_cdf(0.0, 3, 10), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::studentized_range_cdf(-1.0, 3, 10), 0.0);
    EXPECT_NEAR(statcpp::studentized_range_cdf(20.0, 3, 10), 1.0, 1e-6);
}

/**
 * @brief Test studentized range CDF parameter validation.
 * @test k < 2 or df <= 0 should throw std::invalid_argument.
 */
TEST(StudentizedRangeCdfTest, InvalidParameters) {
    EXPECT_THROW(statcpp::studentized_range_cdf(3.0, 1, 10), std::invalid_argument);
    EXPECT_THROW(statcpp::studentized_range_cdf(3.0, 3, 0), std::invalid_argument);
    EXPECT_THROW(statcpp::studentized_range_cdf(3.0, 3, -1), std::invalid_argument);
}

/**
 * @brief Test studentized range quantile against R qtukey values.
 * @test Validates quantile output against R's qtukey() for various (p, k, df) combinations.
 */
TEST(StudentizedRangeQuantileTest, AgainstR) {
    // R: qtukey(0.95, 3, 12)
    EXPECT_NEAR(statcpp::studentized_range_quantile(0.95, 3, 12), 3.772928959408349, 0.01);
    // R: qtukey(0.95, 4, 20)
    EXPECT_NEAR(statcpp::studentized_range_quantile(0.95, 4, 20), 3.958293461450393, 0.01);
    // R: qtukey(0.95, 5, 30)
    EXPECT_NEAR(statcpp::studentized_range_quantile(0.95, 5, 30), 4.102079019621741, 0.01);
    // R: qtukey(0.99, 3, 10)
    EXPECT_NEAR(statcpp::studentized_range_quantile(0.99, 3, 10), 5.270161537033280, 0.01);
}

/**
 * @brief Test CDF(quantile(p)) = p inverse relationship.
 * @test Validates that applying CDF to a quantile result returns the original probability.
 */
TEST(StudentizedRangeQuantileTest, InverseRelationship) {
    double p = 0.95;
    double k = 4;
    double df = 20;
    double q = statcpp::studentized_range_quantile(p, k, df);
    double p_back = statcpp::studentized_range_cdf(q, k, df);
    EXPECT_NEAR(p_back, p, 1e-6);
}
