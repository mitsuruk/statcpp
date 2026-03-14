#include <gtest/gtest.h>
#include "statcpp/discrete_distributions.hpp"
#include <cmath>
#include <numeric>
#include <vector>

// ============================================================================
// Binomial Coefficient Tests
// ============================================================================

/**
 * @brief Test binomial coefficient with known values.
 * @test Verifies binomial coefficient calculation for known combinations.
 */
TEST(BinomialCoefTest, KnownValues) {
    EXPECT_NEAR(statcpp::binomial_coef(5, 0), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::binomial_coef(5, 5), 1.0, 1e-10);
    EXPECT_NEAR(statcpp::binomial_coef(5, 2), 10.0, 1e-10);
    EXPECT_NEAR(statcpp::binomial_coef(10, 3), 120.0, 1e-10);
}

/**
 * @brief Test binomial coefficient with out-of-range values.
 * @test Verifies that binomial coefficient returns 0 when k > n.
 */
TEST(BinomialCoefTest, OutOfRange) {
    EXPECT_DOUBLE_EQ(statcpp::binomial_coef(5, 6), 0.0);
}

// ============================================================================
// Binomial Distribution Tests
// ============================================================================

/**
 * @brief Test binomial PMF with basic values.
 * @test Verifies PMF calculation and that sum of all PMF values equals 1.
 */
TEST(BinomialPmfTest, BasicValues) {
    // Fair coin, 10 flips
    std::uint64_t n = 10;
    double p = 0.5;

    // P(X=5) should be maximum for symmetric binomial
    EXPECT_NEAR(statcpp::binomial_pmf(5, n, p), 0.2461, 1e-3);
    // Sum of all PMF should be 1
    double sum = 0.0;
    for (std::uint64_t k = 0; k <= n; ++k) {
        sum += statcpp::binomial_pmf(k, n, p);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

/**
 * @brief Test binomial PMF edge cases.
 * @test Verifies PMF behavior at boundary values (p=0, p=1, k>n).
 */
TEST(BinomialPmfTest, EdgeCases) {
    EXPECT_DOUBLE_EQ(statcpp::binomial_pmf(0, 10, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::binomial_pmf(10, 10, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::binomial_pmf(5, 10, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::binomial_pmf(11, 10, 0.5), 0.0);
}

/**
 * @brief Test binomial CDF with basic values.
 * @test Verifies CDF calculation and that CDF at maximum equals 1.
 */
TEST(BinomialCdfTest, BasicValues) {
    std::uint64_t n = 10;
    double p = 0.5;

    EXPECT_NEAR(statcpp::binomial_cdf(4, n, p), 0.377, 1e-2);
    EXPECT_DOUBLE_EQ(statcpp::binomial_cdf(10, n, p), 1.0);
}

/**
 * @brief Test inverse relationship between binomial CDF and quantile.
 * @test Verifies that cdf(quantile(p)) >= p and cdf(quantile(p)-1) < p.
 */
TEST(BinomialQuantileTest, InverseProperty) {
    std::uint64_t n = 20;
    double p = 0.3;
    double prob = 0.7;

    std::uint64_t k = statcpp::binomial_quantile(prob, n, p);
    EXPECT_GE(statcpp::binomial_cdf(k, n, p), prob);
    if (k > 0) {
        EXPECT_LT(statcpp::binomial_cdf(k - 1, n, p), prob);
    }
}

/**
 * @brief Test binomial random number generator.
 * @test Verifies that generated random numbers are within valid range [0, n].
 */
TEST(BinomialRandTest, InRange) {
    statcpp::set_seed(42);
    std::uint64_t n = 10;
    double p = 0.3;
    for (int i = 0; i < 100; ++i) {
        std::uint64_t r = statcpp::binomial_rand(n, p);
        EXPECT_LE(r, n);
    }
}

// ============================================================================
// Poisson Distribution Tests
// ============================================================================

/**
 * @brief Test Poisson PMF with basic values.
 * @test Verifies PMF calculation and that sum of PMF values converges to 1.
 */
TEST(PoissonPmfTest, BasicValues) {
    double lambda = 3.0;

    // Sum of PMF should be approximately 1
    double sum = 0.0;
    for (std::uint64_t k = 0; k <= 30; ++k) {
        sum += statcpp::poisson_pmf(k, lambda);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);

    // Known value
    EXPECT_NEAR(statcpp::poisson_pmf(3, lambda), std::exp(-3.0) * 27.0 / 6.0, 1e-10);
}

/**
 * @brief Test Poisson PMF with lambda=0.
 * @test Verifies PMF behavior when lambda=0 (all mass at k=0).
 */
TEST(PoissonPmfTest, ZeroLambda) {
    EXPECT_DOUBLE_EQ(statcpp::poisson_pmf(0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::poisson_pmf(1, 0.0), 0.0);
}

/**
 * @brief Test Poisson CDF with basic values.
 * @test Verifies that CDF is monotonically increasing.
 */
TEST(PoissonCdfTest, BasicValues) {
    double lambda = 5.0;

    // CDF should be monotonically increasing
    double prev = 0.0;
    for (std::uint64_t k = 0; k <= 20; ++k) {
        double curr = statcpp::poisson_cdf(k, lambda);
        EXPECT_GE(curr, prev);
        prev = curr;
    }
}

/**
 * @brief Test inverse relationship between Poisson CDF and quantile.
 * @test Verifies that cdf(quantile(p)) >= p and cdf(quantile(p)-1) < p.
 */
TEST(PoissonQuantileTest, InverseProperty) {
    double lambda = 4.0;
    double p = 0.8;

    std::uint64_t k = statcpp::poisson_quantile(p, lambda);
    EXPECT_GE(statcpp::poisson_cdf(k, lambda), p);
    if (k > 0) {
        EXPECT_LT(statcpp::poisson_cdf(k - 1, lambda), p);
    }
}

// ============================================================================
// Geometric Distribution Tests
// ============================================================================

/**
 * @brief Test geometric PMF with basic values.
 * @test Verifies PMF calculation and that sum converges to 1.
 */
TEST(GeometricPmfTest, BasicValues) {
    double p = 0.3;

    // P(X=0) = p
    EXPECT_NEAR(statcpp::geometric_pmf(0, p), p, 1e-10);
    // P(X=1) = (1-p)*p
    EXPECT_NEAR(statcpp::geometric_pmf(1, p), (1.0 - p) * p, 1e-10);

    // Sum should converge to 1
    double sum = 0.0;
    for (std::uint64_t k = 0; k <= 50; ++k) {
        sum += statcpp::geometric_pmf(k, p);
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

/**
 * @brief Test geometric CDF formula.
 * @test Verifies CDF using closed-form formula: CDF = 1 - (1-p)^(k+1).
 */
TEST(GeometricCdfTest, Formula) {
    double p = 0.4;
    std::uint64_t k = 3;

    // CDF = 1 - (1-p)^(k+1)
    double expected = 1.0 - std::pow(1.0 - p, k + 1);
    EXPECT_NEAR(statcpp::geometric_cdf(k, p), expected, 1e-10);
}

/**
 * @brief Test inverse relationship between geometric CDF and quantile.
 * @test Verifies that cdf(quantile(p)) >= p.
 */
TEST(GeometricQuantileTest, InverseProperty) {
    double p = 0.25;
    double prob = 0.9;

    std::uint64_t k = statcpp::geometric_quantile(prob, p);
    EXPECT_GE(statcpp::geometric_cdf(k, p), prob);
}

// ============================================================================
// Negative Binomial Distribution Tests
// ============================================================================

/**
 * @brief Test negative binomial PMF special case: NB(r=1) = Geometric.
 * @test Verifies that negative binomial reduces to geometric when r=1.
 */
TEST(NbinomPmfTest, GeometricSpecialCase) {
    // Negative binomial with r=1 is geometric
    double p = 0.3;
    for (std::uint64_t k = 0; k <= 10; ++k) {
        EXPECT_NEAR(statcpp::nbinom_pmf(k, 1.0, p), statcpp::geometric_pmf(k, p), 1e-10);
    }
}

/**
 * @brief Test negative binomial PMF sum.
 * @test Verifies that sum of PMF values converges to 1.
 */
TEST(NbinomPmfTest, SumToOne) {
    double r = 3.0, p = 0.4;
    double sum = 0.0;
    for (std::uint64_t k = 0; k <= 100; ++k) {
        sum += statcpp::nbinom_pmf(k, r, p);
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

/**
 * @brief Test negative binomial CDF special case: NB(r=1) = Geometric.
 * @test Verifies that negative binomial CDF reduces to geometric CDF when r=1.
 */
TEST(NbinomCdfTest, GeometricSpecialCase) {
    double p = 0.3;
    for (std::uint64_t k = 0; k <= 10; ++k) {
        EXPECT_NEAR(statcpp::nbinom_cdf(k, 1.0, p), statcpp::geometric_cdf(k, p), 1e-10);
    }
}

/**
 * @brief Test inverse relationship between negative binomial CDF and quantile.
 * @test Verifies that cdf(quantile(p)) >= p and cdf(quantile(p)-1) < p.
 */
TEST(NbinomQuantileTest, InverseProperty) {
    double r = 5.0, p = 0.4;
    double prob = 0.75;

    std::uint64_t k = statcpp::nbinom_quantile(prob, r, p);
    EXPECT_GE(statcpp::nbinom_cdf(k, r, p), prob);
    if (k > 0) {
        EXPECT_LT(statcpp::nbinom_cdf(k - 1, r, p), prob);
    }
}

// ============================================================================
// Hypergeometric Distribution Tests
// ============================================================================

/**
 * @brief Test hypergeometric PMF with basic values.
 * @test Verifies PMF calculation and that sum of all PMF values equals 1.
 */
TEST(HypergeomPmfTest, BasicValues) {
    // Urn with 20 balls, 7 white, draw 12
    std::uint64_t N = 20, K = 7, n = 12;

    // Sum of PMF should be 1
    double sum = 0.0;
    for (std::uint64_t k = 0; k <= std::min(n, K); ++k) {
        sum += statcpp::hypergeom_pmf(k, N, K, n);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

/**
 * @brief Test hypergeometric PMF boundary cases.
 * @test Verifies PMF behavior when all or no successes are in population.
 */
TEST(HypergeomPmfTest, BoundaryCase) {
    // All successes: N=K
    EXPECT_DOUBLE_EQ(statcpp::hypergeom_pmf(5, 10, 10, 5), 1.0);
    // No successes: K=0
    EXPECT_DOUBLE_EQ(statcpp::hypergeom_pmf(0, 10, 0, 5), 1.0);
    EXPECT_DOUBLE_EQ(statcpp::hypergeom_pmf(1, 10, 0, 5), 0.0);
}

/**
 * @brief Test hypergeometric CDF monotonicity.
 * @test Verifies that CDF is monotonically increasing and reaches 1.
 */
TEST(HypergeomCdfTest, Monotonicity) {
    std::uint64_t N = 50, K = 20, n = 15;

    double prev = 0.0;
    for (std::uint64_t k = 0; k <= std::min(n, K); ++k) {
        double curr = statcpp::hypergeom_cdf(k, N, K, n);
        EXPECT_GE(curr, prev);
        prev = curr;
    }
    EXPECT_DOUBLE_EQ(statcpp::hypergeom_cdf(std::min(n, K), N, K, n), 1.0);
}

/**
 * @brief Test inverse relationship between hypergeometric CDF and quantile.
 * @test Verifies that cdf(quantile(p)) >= p.
 */
TEST(HypergeomQuantileTest, InverseProperty) {
    std::uint64_t N = 30, K = 10, n = 8;
    double p = 0.6;

    std::uint64_t k = statcpp::hypergeom_quantile(p, N, K, n);
    EXPECT_GE(statcpp::hypergeom_cdf(k, N, K, n), p);
}

/**
 * @brief Test hypergeometric random number generator.
 * @test Verifies that generated random numbers are within valid range.
 */
TEST(HypergeomRandTest, InRange) {
    statcpp::set_seed(42);
    std::uint64_t N = 100, K = 30, n = 20;

    for (int i = 0; i < 100; ++i) {
        std::uint64_t r = statcpp::hypergeom_rand(N, K, n);
        EXPECT_LE(r, std::min(n, K));
        EXPECT_GE(r, (n > N - K) ? n - (N - K) : 0);
    }
}

// ============================================================================
// Invalid Input Tests
// ============================================================================

/**
 * @brief Test exception handling for invalid discrete distribution parameters.
 * @test Verifies that appropriate exceptions are thrown for invalid parameters.
 */
TEST(DiscreteDistributionExceptionTest, InvalidParameters) {
    // Binomial: p out of range
    EXPECT_THROW(statcpp::binomial_pmf(5, 10, -0.1), std::invalid_argument);
    EXPECT_THROW(statcpp::binomial_pmf(5, 10, 1.1), std::invalid_argument);

    // Poisson: negative lambda
    EXPECT_THROW(statcpp::poisson_pmf(5, -1.0), std::invalid_argument);

    // Geometric: p <= 0 or p > 1
    EXPECT_THROW(statcpp::geometric_pmf(5, 0.0), std::invalid_argument);
    EXPECT_THROW(statcpp::geometric_pmf(5, 1.1), std::invalid_argument);

    // Negative binomial: r <= 0 or p out of range
    EXPECT_THROW(statcpp::nbinom_pmf(5, 0.0, 0.5), std::invalid_argument);
    EXPECT_THROW(statcpp::nbinom_pmf(5, 3.0, 0.0), std::invalid_argument);

    // Hypergeometric: K > N or n > N
    EXPECT_THROW(statcpp::hypergeom_pmf(5, 10, 15, 5), std::invalid_argument);
    EXPECT_THROW(statcpp::hypergeom_pmf(5, 10, 5, 15), std::invalid_argument);
}
