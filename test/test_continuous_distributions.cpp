#include <gtest/gtest.h>
#include "continuous_distributions.hpp"
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
