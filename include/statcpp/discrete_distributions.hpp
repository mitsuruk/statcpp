/**
 * @file discrete_distributions.hpp
 * @brief Discrete probability distribution functions
 *
 * This file provides probability mass functions (PMF), cumulative distribution functions (CDF),
 * quantile functions, and random number generation for discrete probability distributions
 * (binomial, Poisson, geometric, etc.).
 */

#pragma once

#include "statcpp/special_functions.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Helper: Log factorial and binomial coefficient
// ============================================================================

/**
 * @brief Calculate log factorial
 *
 * @param n Non-negative integer
 * @return Value of log(n!)
 */
inline double log_factorial(std::uint64_t n)
{
    if (n <= 1) return 0.0;
    return lgamma(static_cast<double>(n + 1));
}

/**
 * @brief Calculate log binomial coefficient
 *
 * Calculates log(C(n, k)) = log(n! / (k! * (n-k)!))
 *
 * @param n Total number of elements
 * @param k Number of elements to choose
 * @return Value of log(C(n, k))
 */
inline double log_binomial_coef(std::uint64_t n, std::uint64_t k)
{
    if (k > n) return -std::numeric_limits<double>::infinity();
    if (k == 0 || k == n) return 0.0;
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k);
}

/**
 * @brief Calculate binomial coefficient
 *
 * Calculates C(n, k) = n! / (k! * (n-k)!)
 *
 * @param n Total number of elements
 * @param k Number of elements to choose
 * @return Value of C(n, k)
 */
inline double binomial_coef(std::uint64_t n, std::uint64_t k)
{
    if (k > n) return 0.0;
    return std::exp(log_binomial_coef(n, k));
}

// ============================================================================
// Binomial Distribution
// ============================================================================

/**
 * @brief Binomial distribution probability mass function (PMF)
 *
 * P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
 *
 * @param k Number of successes
 * @param n Number of trials
 * @param p Probability of success in each trial
 * @return Probability P(X = k)
 * @throw std::invalid_argument If p is outside [0, 1]
 */
inline double binomial_pmf(std::uint64_t k, std::uint64_t n, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::binomial_pmf: p must be in [0, 1]");
    }
    if (k > n) return 0.0;

    if (p == 0.0) return (k == 0) ? 1.0 : 0.0;
    if (p == 1.0) return (k == n) ? 1.0 : 0.0;

    double log_pmf = log_binomial_coef(n, k) + k * std::log(p) + (n - k) * std::log(1.0 - p);
    return std::exp(log_pmf);
}

/**
 * @brief Binomial distribution cumulative distribution function (CDF)
 *
 * Calculates P(X <= k).
 *
 * @param k Upper bound of number of successes
 * @param n Number of trials
 * @param p Probability of success in each trial
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If p is outside [0, 1]
 */
inline double binomial_cdf(std::uint64_t k, std::uint64_t n, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::binomial_cdf: p must be in [0, 1]");
    }
    if (k >= n) return 1.0;

    // Use regularized incomplete beta function
    // P(X <= k) = I_{1-p}(n-k, k+1)
    return betainc(static_cast<double>(n - k), static_cast<double>(k + 1), 1.0 - p);
}

/**
 * @brief Binomial distribution quantile function
 *
 * Returns the smallest k such that P(X <= k) >= prob.
 *
 * @param prob Probability value
 * @param n Number of trials
 * @param p Probability of success in each trial
 * @return Quantile
 * @throw std::invalid_argument If prob or p is in an invalid range
 */
inline std::uint64_t binomial_quantile(double prob, std::uint64_t n, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::binomial_quantile: p must be in [0, 1]");
    }
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("statcpp::binomial_quantile: prob must be in [0, 1]");
    }
    if (prob == 0.0) return 0;
    if (prob == 1.0) return n;

    // Binary search
    std::uint64_t lo = 0;
    std::uint64_t hi = n;

    while (lo < hi) {
        std::uint64_t mid = lo + (hi - lo) / 2;
        if (binomial_cdf(mid, n, p) < prob) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    return lo;
}

/**
 * @brief Binomial distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param n Number of trials
 * @param p Probability of success in each trial
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If p is outside [0, 1]
 */
template <typename Engine = default_random_engine>
std::uint64_t binomial_rand(std::uint64_t n, double p, Engine& engine)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::binomial_rand: p must be in [0, 1]");
    }
    std::binomial_distribution<std::uint64_t> dist(n, p);
    return dist(engine);
}

/**
 * @brief Binomial distribution random number generation (using default engine)
 *
 * @param n Number of trials
 * @param p Probability of success in each trial
 * @return Generated random number
 */
inline std::uint64_t binomial_rand(std::uint64_t n, double p)
{
    return binomial_rand(n, p, get_random_engine());
}

// ============================================================================
// Poisson Distribution
// ============================================================================

/**
 * @brief Poisson distribution probability mass function (PMF)
 *
 * P(X = k) = (lambda^k * e^(-lambda)) / k!
 *
 * @param k Number of event occurrences
 * @param lambda Mean rate (lambda > 0)
 * @return Probability P(X = k)
 * @throw std::invalid_argument If lambda is negative
 */
inline double poisson_pmf(std::uint64_t k, double lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::poisson_pmf: lambda must be non-negative");
    }
    if (lambda == 0.0) return (k == 0) ? 1.0 : 0.0;

    double log_pmf = k * std::log(lambda) - lambda - log_factorial(k);
    return std::exp(log_pmf);
}

/**
 * @brief Poisson distribution cumulative distribution function (CDF)
 *
 * Calculates P(X <= k).
 *
 * @param k Upper bound of number of event occurrences
 * @param lambda Mean rate
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If lambda is negative
 */
inline double poisson_cdf(std::uint64_t k, double lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::poisson_cdf: lambda must be non-negative");
    }
    if (lambda == 0.0) return 1.0;

    // P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda) (upper regularized incomplete gamma)
    return gammainc_upper(static_cast<double>(k + 1), lambda);
}

/**
 * @brief Poisson distribution quantile function
 *
 * @param p Probability value
 * @param lambda Mean rate
 * @return Quantile
 * @throw std::invalid_argument If lambda is negative or p is in an invalid range
 */
inline std::uint64_t poisson_quantile(double p, double lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::poisson_quantile: lambda must be non-negative");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::poisson_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0;
    if (lambda == 0.0) return 0;

    // Start with Gaussian approximation
    double z = norm_quantile(p);
    double guess = lambda + z * std::sqrt(lambda);
    std::uint64_t k = static_cast<std::uint64_t>(std::max(0.0, guess));

    // Adjust up or down
    while (k > 0 && poisson_cdf(k - 1, lambda) >= p) {
        --k;
    }
    while (poisson_cdf(k, lambda) < p) {
        ++k;
    }

    return k;
}

/**
 * @brief Poisson distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param lambda Mean rate
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If lambda is negative
 */
template <typename Engine = default_random_engine>
std::uint64_t poisson_rand(double lambda, Engine& engine)
{
    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::poisson_rand: lambda must be non-negative");
    }
    std::poisson_distribution<std::uint64_t> dist(lambda);
    return dist(engine);
}

/**
 * @brief Poisson distribution random number generation (using default engine)
 *
 * @param lambda Mean rate
 * @return Generated random number
 */
inline std::uint64_t poisson_rand(double lambda)
{
    return poisson_rand(lambda, get_random_engine());
}

// ============================================================================
// Geometric Distribution
// ============================================================================

/**
 * @brief Geometric distribution probability mass function (PMF)
 *
 * P(X = k) = (1-p)^k * p
 * X = Number of failures before first success (support: k = 0, 1, 2, ...)
 *
 * @param k Number of failures
 * @param p Probability of success in each trial
 * @return Probability P(X = k)
 * @throw std::invalid_argument If p is outside (0, 1]
 */
inline double geometric_pmf(std::uint64_t k, double p)
{
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::geometric_pmf: p must be in (0, 1]");
    }
    if (p == 1.0) return (k == 0) ? 1.0 : 0.0;

    return std::pow(1.0 - p, static_cast<double>(k)) * p;
}

/**
 * @brief Geometric distribution cumulative distribution function (CDF)
 *
 * P(X <= k) = 1 - (1-p)^(k+1)
 *
 * @param k Upper bound of number of failures
 * @param p Probability of success in each trial
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If p is outside (0, 1]
 */
inline double geometric_cdf(std::uint64_t k, double p)
{
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::geometric_cdf: p must be in (0, 1]");
    }
    if (p == 1.0) return 1.0;

    return 1.0 - std::pow(1.0 - p, static_cast<double>(k + 1));
}

/**
 * @brief Geometric distribution quantile function (inverse CDF)
 *
 * @param prob Cumulative probability
 * @param p Probability of success in each trial
 * @return Smallest k such that P(X <= k) >= prob
 * @throw std::invalid_argument If prob is outside [0, 1] or p is outside (0, 1]
 */
inline std::uint64_t geometric_quantile(double prob, double p)
{
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::geometric_quantile: p must be in (0, 1]");
    }
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("statcpp::geometric_quantile: prob must be in [0, 1]");
    }
    if (prob == 0.0) return 0;
    if (p == 1.0) return 0;

    // Q(prob) = ceil(log(1 - prob) / log(1 - p)) - 1
    double k_real = std::ceil(std::log(1.0 - prob) / std::log(1.0 - p)) - 1.0;
    return static_cast<std::uint64_t>(std::max(0.0, k_real));
}

/**
 * @brief Geometric distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param p Probability of success in each trial
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If p is outside (0, 1]
 */
template <typename Engine = default_random_engine>
std::uint64_t geometric_rand(double p, Engine& engine)
{
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::geometric_rand: p must be in (0, 1]");
    }
    std::geometric_distribution<std::uint64_t> dist(p);
    return dist(engine);
}

/**
 * @brief Geometric distribution random number generation (using default engine)
 *
 * @param p Probability of success in each trial
 * @return Generated random number
 */
inline std::uint64_t geometric_rand(double p)
{
    return geometric_rand(p, get_random_engine());
}

// ============================================================================
// Hypergeometric Distribution
// ============================================================================

/**
 * @brief Hypergeometric distribution probability mass function (PMF)
 *
 * P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
 *
 * @param k Number of success draws
 * @param N Population size
 * @param K Number of success states in population
 * @param n Number of draws
 * @return Probability P(X = k)
 * @throw std::invalid_argument If parameters are invalid (K > N or n > N)
 */
inline double hypergeom_pmf(std::uint64_t k, std::uint64_t N, std::uint64_t K, std::uint64_t n)
{
    if (K > N) {
        throw std::invalid_argument("statcpp::hypergeom_pmf: K must be <= N");
    }
    if (n > N) {
        throw std::invalid_argument("statcpp::hypergeom_pmf: n must be <= N");
    }

    // k must be in valid range
    std::uint64_t k_min = (n > N - K) ? n - (N - K) : 0;
    std::uint64_t k_max = std::min(n, K);

    if (k < k_min || k > k_max) return 0.0;

    double log_pmf = log_binomial_coef(K, k) + log_binomial_coef(N - K, n - k) - log_binomial_coef(N, n);
    return std::exp(log_pmf);
}

/**
 * @brief Hypergeometric distribution cumulative distribution function (CDF)
 *
 * Calculates P(X <= k).
 *
 * @param k Upper bound of success draws
 * @param N Population size
 * @param K Number of success states in population
 * @param n Number of draws
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If parameters are invalid
 */
inline double hypergeom_cdf(std::uint64_t k, std::uint64_t N, std::uint64_t K, std::uint64_t n)
{
    if (K > N) {
        throw std::invalid_argument("statcpp::hypergeom_cdf: K must be <= N");
    }
    if (n > N) {
        throw std::invalid_argument("statcpp::hypergeom_cdf: n must be <= N");
    }

    std::uint64_t k_min = (n > N - K) ? n - (N - K) : 0;
    std::uint64_t k_max = std::min(n, K);

    if (k >= k_max) return 1.0;

    double sum = 0.0;
    for (std::uint64_t i = k_min; i <= k; ++i) {
        sum += hypergeom_pmf(i, N, K, n);
    }
    return std::min(1.0, sum);
}

/**
 * @brief Hypergeometric distribution quantile function (inverse CDF)
 *
 * @param p Cumulative probability
 * @param N Population size
 * @param K Number of success states in population
 * @param n Number of draws
 * @return Smallest k such that P(X <= k) >= p
 * @throw std::invalid_argument If parameters are outside valid range
 */
inline std::uint64_t hypergeom_quantile(double p, std::uint64_t N, std::uint64_t K, std::uint64_t n)
{
    if (K > N) {
        throw std::invalid_argument("statcpp::hypergeom_quantile: K must be <= N");
    }
    if (n > N) {
        throw std::invalid_argument("statcpp::hypergeom_quantile: n must be <= N");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::hypergeom_quantile: p must be in [0, 1]");
    }

    std::uint64_t k_min = (n > N - K) ? n - (N - K) : 0;
    std::uint64_t k_max = std::min(n, K);

    if (p == 0.0) return k_min;
    if (p == 1.0) return k_max;

    double cum = 0.0;
    for (std::uint64_t k = k_min; k <= k_max; ++k) {
        cum += hypergeom_pmf(k, N, K, n);
        if (cum >= p) return k;
    }

    return k_max;
}

/**
 * @brief Hypergeometric distribution random number generation
 *
 * Uses direct sampling method (suitable for moderate n).
 *
 * @tparam Engine Random engine type
 * @param N Population size
 * @param K Number of success states in population
 * @param n Number of draws
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If parameters are invalid
 */
template <typename Engine = default_random_engine>
std::uint64_t hypergeom_rand(std::uint64_t N, std::uint64_t K, std::uint64_t n, Engine& engine)
{
    if (K > N) {
        throw std::invalid_argument("statcpp::hypergeom_rand: K must be <= N");
    }
    if (n > N) {
        throw std::invalid_argument("statcpp::hypergeom_rand: n must be <= N");
    }

    // Direct sampling (for moderate n)
    std::uint64_t successes = 0;
    std::uint64_t population = N;
    std::uint64_t success_states = K;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (std::uint64_t i = 0; i < n; ++i) {
        double p = static_cast<double>(success_states) / static_cast<double>(population);
        if (uniform(engine) < p) {
            ++successes;
            --success_states;
        }
        --population;
    }

    return successes;
}

/**
 * @brief Hypergeometric distribution random number generation (using default engine)
 *
 * @param N Population size
 * @param K Number of success states in population
 * @param n Number of draws
 * @return Generated random number
 */
inline std::uint64_t hypergeom_rand(std::uint64_t N, std::uint64_t K, std::uint64_t n)
{
    return hypergeom_rand(N, K, n, get_random_engine());
}

// ============================================================================
// Negative Binomial Distribution
// ============================================================================

/**
 * @brief Negative binomial distribution probability mass function (PMF)
 *
 * P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 * X = Number of failures before r successes (support: k = 0, 1, 2, ...)
 *
 * @param k Number of failures
 * @param r Number of successes required (dispersion parameter, > 0, can be non-integer)
 * @param p Probability of success
 * @return Probability P(X = k)
 * @throw std::invalid_argument If r <= 0 or p is outside (0, 1]
 *
 * @note About parameterization:
 *       There are multiple parameterizations for the negative binomial distribution.
 *       This implementation uses the "number of failures" form (X = number of failures
 *       before r successes).
 *
 *       - R's dnbinom(x, size, prob): size=r, prob=p (same, counts failures)
 *       - Python scipy.stats.nbinom(k, n, p): n=r, p=p (same, counts failures)
 *       - Some textbooks: "number of trials until r successes" = X + r
 *
 *       There is also a "mean-dispersion" parameterization:
 *       - Mean μ = r(1-p)/p
 *       - Variance σ² = r(1-p)/p² = μ + μ²/r
 *       This form is commonly used for modeling overdispersed data (GLM, etc.).
 */
inline double nbinom_pmf(std::uint64_t k, double r, double p)
{
    if (r <= 0.0) {
        throw std::invalid_argument("statcpp::nbinom_pmf: r must be positive");
    }
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::nbinom_pmf: p must be in (0, 1]");
    }
    if (p == 1.0) return (k == 0) ? 1.0 : 0.0;

    double log_pmf = lgamma(k + r) - log_factorial(k) - lgamma(r)
                   + r * std::log(p) + k * std::log(1.0 - p);
    return std::exp(log_pmf);
}

/**
 * @brief Negative binomial distribution cumulative distribution function (CDF)
 *
 * P(X <= k) = I_p(r, k+1) (using incomplete beta function)
 *
 * @param k Upper bound of failures
 * @param r Number of successes required
 * @param p Probability of success
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If r <= 0 or p is outside (0, 1]
 */
inline double nbinom_cdf(std::uint64_t k, double r, double p)
{
    if (r <= 0.0) {
        throw std::invalid_argument("statcpp::nbinom_cdf: r must be positive");
    }
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::nbinom_cdf: p must be in (0, 1]");
    }
    if (p == 1.0) return 1.0;

    return betainc(r, static_cast<double>(k + 1), p);
}

/**
 * @brief Negative binomial distribution quantile function (inverse CDF)
 *
 * @param prob Cumulative probability
 * @param r Number of successes required
 * @param p Probability of success
 * @return Smallest k such that P(X <= k) >= prob
 * @throw std::invalid_argument If parameters are outside valid range
 */
inline std::uint64_t nbinom_quantile(double prob, double r, double p)
{
    if (r <= 0.0) {
        throw std::invalid_argument("statcpp::nbinom_quantile: r must be positive");
    }
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::nbinom_quantile: p must be in (0, 1]");
    }
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("statcpp::nbinom_quantile: prob must be in [0, 1]");
    }
    if (prob == 0.0) return 0;
    if (p == 1.0) return 0;

    // Start with Gaussian approximation
    double mean_val = r * (1.0 - p) / p;
    double var_val = r * (1.0 - p) / (p * p);
    double z = norm_quantile(prob);
    double guess = mean_val + z * std::sqrt(var_val);
    std::uint64_t k = static_cast<std::uint64_t>(std::max(0.0, guess));

    // Adjust
    while (k > 0 && nbinom_cdf(k - 1, r, p) >= prob) {
        --k;
    }
    while (nbinom_cdf(k, r, p) < prob) {
        ++k;
    }

    return k;
}

/**
 * @brief Negative binomial distribution random number generation
 *
 * Generated as a Poisson-Gamma mixture.
 *
 * @tparam Engine Random engine type
 * @param r Number of successes required
 * @param p Probability of success
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If parameters are outside valid range
 */
template <typename Engine = default_random_engine>
std::uint64_t nbinom_rand(double r, double p, Engine& engine)
{
    if (r <= 0.0) {
        throw std::invalid_argument("statcpp::nbinom_rand: r must be positive");
    }
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::nbinom_rand: p must be in (0, 1]");
    }

    // Negative binomial as Poisson-Gamma mixture
    // X ~ NB(r, p) can be generated as Poisson(Y) where Y ~ Gamma(r, p/(1-p))
    std::gamma_distribution<double> gamma_dist(r, (1.0 - p) / p);
    double y = gamma_dist(engine);
    std::poisson_distribution<std::uint64_t> poisson_dist(y);
    return poisson_dist(engine);
}

/**
 * @brief Negative binomial distribution random number generation (using default engine)
 *
 * @param r Number of successes required
 * @param p Probability of success
 * @return Generated random number
 */
inline std::uint64_t nbinom_rand(double r, double p)
{
    return nbinom_rand(r, p, get_random_engine());
}

// ============================================================================
// Bernoulli Distribution
// ============================================================================

/**
 * @brief Bernoulli distribution probability mass function (PMF)
 *
 * P(X = k) = p^k * (1-p)^(1-k) for k ∈ {0, 1}
 *
 * @param k Outcome (0 or 1)
 * @param p Probability of success
 * @return Probability P(X = k)
 * @throw std::invalid_argument If p is outside [0, 1]
 */
inline double bernoulli_pmf(std::uint64_t k, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_pmf: p must be in [0, 1]");
    }
    if (k == 0) {
        return 1.0 - p;
    } else if (k == 1) {
        return p;
    } else {
        return 0.0;
    }
}

/**
 * @brief Bernoulli distribution cumulative distribution function (CDF)
 *
 * @param k Upper bound
 * @param p Probability of success
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If p is outside [0, 1]
 */
inline double bernoulli_cdf(std::uint64_t k, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_cdf: p must be in [0, 1]");
    }
    if (k == 0) {
        return 1.0 - p;
    } else {
        return 1.0;
    }
}

/**
 * @brief Bernoulli distribution quantile function
 *
 * @param prob Cumulative probability
 * @param p Probability of success
 * @return Smallest k such that P(X <= k) >= prob
 */
inline std::uint64_t bernoulli_quantile(double prob, double p)
{
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_quantile: prob must be in [0, 1]");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_quantile: p must be in [0, 1]");
    }
    if (prob <= 1.0 - p) return 0;
    return 1;
}

/**
 * @brief Bernoulli distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param p Probability of success
 * @param engine Random engine
 * @return Generated random number (0 or 1)
 * @throw std::invalid_argument If p is outside [0, 1]
 */
template <typename Engine = default_random_engine>
std::uint64_t bernoulli_rand(double p, Engine& engine)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_rand: p must be in [0, 1]");
    }
    std::bernoulli_distribution dist(p);
    return dist(engine) ? 1 : 0;
}

/**
 * @brief Bernoulli distribution random number generation (using default engine)
 *
 * @param p Probability of success
 * @return Generated random number (0 or 1)
 */
inline std::uint64_t bernoulli_rand(double p)
{
    return bernoulli_rand(p, get_random_engine());
}

// ============================================================================
// Discrete Uniform Distribution
// ============================================================================

/**
 * @brief Discrete uniform distribution probability mass function (PMF)
 *
 * P(X = k) = 1 / (b - a + 1) for a <= k <= b
 *
 * @param k Value
 * @param a Lower bound (inclusive)
 * @param b Upper bound (inclusive)
 * @return Probability P(X = k)
 * @throw std::invalid_argument If a > b
 */
inline double discrete_uniform_pmf(std::int64_t k, std::int64_t a, std::int64_t b)
{
    if (a > b) {
        throw std::invalid_argument("statcpp::discrete_uniform_pmf: a must be <= b");
    }
    if (k < a || k > b) {
        return 0.0;
    }
    return 1.0 / static_cast<double>(b - a + 1);
}

/**
 * @brief Discrete uniform distribution cumulative distribution function (CDF)
 *
 * @param k Upper bound
 * @param a Lower bound (inclusive)
 * @param b Upper bound (inclusive)
 * @return Cumulative probability P(X <= k)
 * @throw std::invalid_argument If a > b
 */
inline double discrete_uniform_cdf(std::int64_t k, std::int64_t a, std::int64_t b)
{
    if (a > b) {
        throw std::invalid_argument("statcpp::discrete_uniform_cdf: a must be <= b");
    }
    if (k < a) {
        return 0.0;
    }
    if (k >= b) {
        return 1.0;
    }
    return static_cast<double>(k - a + 1) / static_cast<double>(b - a + 1);
}

/**
 * @brief Discrete uniform distribution quantile function
 *
 * @param p Cumulative probability
 * @param a Lower bound (inclusive)
 * @param b Upper bound (inclusive)
 * @return Smallest k such that P(X <= k) >= p
 * @throw std::invalid_argument If a > b or p is outside [0, 1]
 */
inline std::int64_t discrete_uniform_quantile(double p, std::int64_t a, std::int64_t b)
{
    if (a > b) {
        throw std::invalid_argument("statcpp::discrete_uniform_quantile: a must be <= b");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::discrete_uniform_quantile: p must be in [0, 1]");
    }
    std::int64_t range = b - a + 1;
    std::int64_t k = a + static_cast<std::int64_t>(std::floor(p * static_cast<double>(range)));
    if (k > b) k = b;
    return k;
}

/**
 * @brief Discrete uniform distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param a Lower bound (inclusive)
 * @param b Upper bound (inclusive)
 * @param engine Random engine
 * @return Generated random number
 * @throw std::invalid_argument If a > b
 */
template <typename Engine = default_random_engine>
std::int64_t discrete_uniform_rand(std::int64_t a, std::int64_t b, Engine& engine)
{
    if (a > b) {
        throw std::invalid_argument("statcpp::discrete_uniform_rand: a must be <= b");
    }
    std::uniform_int_distribution<std::int64_t> dist(a, b);
    return dist(engine);
}

/**
 * @brief Discrete uniform distribution random number generation (using default engine)
 *
 * @param a Lower bound (inclusive)
 * @param b Upper bound (inclusive)
 * @return Generated random number
 */
inline std::int64_t discrete_uniform_rand(std::int64_t a, std::int64_t b)
{
    return discrete_uniform_rand(a, b, get_random_engine());
}

} // namespace statcpp