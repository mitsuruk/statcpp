/**
 * @file continuous_distributions.hpp
 * @brief Continuous distribution functions
 *
 * Provides PDF, CDF, quantile, and random number generation functions for
 * uniform, normal, exponential, gamma, beta, chi-square, t, F,
 * log-normal, and Weibull distributions.
 */

#pragma once

#include "special_functions.hpp"
#include "random_engine.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Uniform Distribution
// ============================================================================

/**
 * @brief Uniform distribution probability density function (PDF)
 *
 * f(x) = 1 / (b - a) for a <= x <= b
 *
 * @param x Random variable value
 * @param a Lower bound (default: 0.0)
 * @param b Upper bound (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If a >= b
 */
inline double uniform_pdf(double x, double a = 0.0, double b = 1.0)
{
    if (a >= b) {
        throw std::invalid_argument("statcpp::uniform_pdf: a must be less than b");
    }
    if (x < a || x > b) {
        return 0.0;
    }
    return 1.0 / (b - a);
}

/**
 * @brief Uniform distribution cumulative distribution function (CDF)
 *
 * F(x) = (x - a) / (b - a)
 *
 * @param x Random variable value
 * @param a Lower bound (default: 0.0)
 * @param b Upper bound (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If a >= b
 */
inline double uniform_cdf(double x, double a = 0.0, double b = 1.0)
{
    if (a >= b) {
        throw std::invalid_argument("statcpp::uniform_cdf: a must be less than b");
    }
    if (x < a) return 0.0;
    if (x > b) return 1.0;
    return (x - a) / (b - a);
}

/**
 * @brief Uniform distribution quantile function (inverse CDF)
 *
 * Q(p) = a + p * (b - a)
 *
 * @param p Probability (0 <= p <= 1)
 * @param a Lower bound (default: 0.0)
 * @param b Upper bound (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If a >= b or p is outside [0, 1]
 */
inline double uniform_quantile(double p, double a = 0.0, double b = 1.0)
{
    if (a >= b) {
        throw std::invalid_argument("statcpp::uniform_quantile: a must be less than b");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::uniform_quantile: p must be in [0, 1]");
    }
    return a + p * (b - a);
}

/**
 * @brief Uniform distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param a Lower bound
 * @param b Upper bound
 * @param engine Random engine
 * @return Random number following uniform distribution
 * @throws std::invalid_argument If a >= b
 */
template <typename Engine = default_random_engine>
double uniform_rand(double a, double b, Engine& engine)
{
    if (a >= b) {
        throw std::invalid_argument("statcpp::uniform_rand: a must be less than b");
    }
    std::uniform_real_distribution<double> dist(a, b);
    return dist(engine);
}

/**
 * @brief Uniform distribution random number generation (using default engine)
 *
 * @param a Lower bound (default: 0.0)
 * @param b Upper bound (default: 1.0)
 * @return Random number following uniform distribution
 */
inline double uniform_rand(double a = 0.0, double b = 1.0)
{
    return uniform_rand(a, b, get_random_engine());
}

// ============================================================================
// Normal Distribution
// ============================================================================

/**
 * @brief Normal distribution probability density function (PDF)
 *
 * f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(-(x-mu)^2/(2*sigma^2))
 *
 * @param x Random variable value
 * @param mu Mean (default: 0.0)
 * @param sigma Standard deviation (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If sigma <= 0
 */
inline double normal_pdf(double x, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::normal_pdf: sigma must be positive");
    }
    double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * sqrt_2_pi);
}

/**
 * @brief Normal distribution cumulative distribution function (CDF)
 *
 * F(x) = Phi((x - mu) / sigma)
 *
 * @param x Random variable value
 * @param mu Mean (default: 0.0)
 * @param sigma Standard deviation (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If sigma <= 0
 */
inline double normal_cdf(double x, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::normal_cdf: sigma must be positive");
    }
    return norm_cdf((x - mu) / sigma);
}

/**
 * @brief Normal distribution quantile function (inverse CDF, percent point function)
 *
 * Returns the quantile (percentile value) corresponding to probability p.
 * Q(p) = mu + sigma * Phi^(-1)(p)
 *
 * where Phi^(-1)(p) is the inverse cumulative distribution function of the
 * standard normal distribution (probit function).
 * Example: Q(0.975) is approximately mu + 1.96*sigma (upper bound of 95% confidence interval)
 *
 * @param p Probability (0 < p < 1)
 * @param mu Mean (default: 0.0)
 * @param sigma Standard deviation (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If sigma <= 0 or p is outside (0, 1)
 *
 * @note Boundary value handling: returns -infinity for p = 0, +infinity for p = 1.
 */
inline double normal_quantile(double p, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::normal_quantile: sigma must be positive");
    }
    if (p <= 0.0 || p >= 1.0) {
        if (p == 0.0) return -std::numeric_limits<double>::infinity();
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        throw std::invalid_argument("statcpp::normal_quantile: p must be in (0, 1)");
    }
    return mu + sigma * norm_quantile(p);
}

/**
 * @brief Normal distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param mu Mean
 * @param sigma Standard deviation
 * @param engine Random engine
 * @return Random number following normal distribution
 * @throws std::invalid_argument If sigma <= 0
 */
template <typename Engine = default_random_engine>
double normal_rand(double mu, double sigma, Engine& engine)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::normal_rand: sigma must be positive");
    }
    std::normal_distribution<double> dist(mu, sigma);
    return dist(engine);
}

/**
 * @brief Normal distribution random number generation (using default engine)
 *
 * @param mu Mean (default: 0.0)
 * @param sigma Standard deviation (default: 1.0)
 * @return Random number following normal distribution
 */
inline double normal_rand(double mu = 0.0, double sigma = 1.0)
{
    return normal_rand(mu, sigma, get_random_engine());
}

// ============================================================================
// Exponential Distribution
// ============================================================================

/**
 * @brief Exponential distribution probability density function (PDF)
 *
 * f(x) = lambda * exp(-lambda*x) for x >= 0
 *
 * @param x Random variable value
 * @param lambda Rate parameter (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If lambda <= 0
 */
inline double exponential_pdf(double x, double lambda = 1.0)
{
    if (lambda <= 0.0) {
        throw std::invalid_argument("statcpp::exponential_pdf: lambda must be positive");
    }
    if (x < 0.0) return 0.0;
    return lambda * std::exp(-lambda * x);
}

/**
 * @brief Exponential distribution cumulative distribution function (CDF)
 *
 * F(x) = 1 - exp(-lambda*x)
 *
 * @param x Random variable value
 * @param lambda Rate parameter (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If lambda <= 0
 */
inline double exponential_cdf(double x, double lambda = 1.0)
{
    if (lambda <= 0.0) {
        throw std::invalid_argument("statcpp::exponential_cdf: lambda must be positive");
    }
    if (x < 0.0) return 0.0;
    return 1.0 - std::exp(-lambda * x);
}

/**
 * @brief Exponential distribution quantile function
 *
 * Q(p) = -ln(1-p) / lambda
 *
 * @param p Probability (0 <= p < 1)
 * @param lambda Rate parameter (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If lambda <= 0 or p is outside [0, 1)
 */
inline double exponential_quantile(double p, double lambda = 1.0)
{
    if (lambda <= 0.0) {
        throw std::invalid_argument("statcpp::exponential_quantile: lambda must be positive");
    }
    if (p < 0.0 || p >= 1.0) {
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        throw std::invalid_argument("statcpp::exponential_quantile: p must be in [0, 1)");
    }
    return -std::log(1.0 - p) / lambda;
}

/**
 * @brief Exponential distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param lambda Rate parameter
 * @param engine Random engine
 * @return Random number following exponential distribution
 * @throws std::invalid_argument If lambda <= 0
 */
template <typename Engine = default_random_engine>
double exponential_rand(double lambda, Engine& engine)
{
    if (lambda <= 0.0) {
        throw std::invalid_argument("statcpp::exponential_rand: lambda must be positive");
    }
    std::exponential_distribution<double> dist(lambda);
    return dist(engine);
}

/**
 * @brief Exponential distribution random number generation (using default engine)
 *
 * @param lambda Rate parameter (default: 1.0)
 * @return Random number following exponential distribution
 */
inline double exponential_rand(double lambda = 1.0)
{
    return exponential_rand(lambda, get_random_engine());
}

// ============================================================================
// Gamma Distribution
// ============================================================================

/**
 * @brief Gamma distribution probability density function (PDF)
 *
 * f(x) = (beta^alpha / Gamma(alpha)) * x^(alpha-1) * exp(-beta*x) for x > 0
 * Parameters: shape = alpha (k), rate = beta (1/theta)
 *
 * @param x Random variable value
 * @param shape Shape parameter alpha
 * @param rate Rate parameter beta (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If shape <= 0 or rate <= 0
 */
inline double gamma_pdf(double x, double shape, double rate = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_pdf: shape must be positive");
    }
    if (rate <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_pdf: rate must be positive");
    }
    if (x <= 0.0) return 0.0;

    return std::exp(shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x - lgamma(shape));
}

/**
 * @brief Gamma distribution cumulative distribution function (CDF)
 *
 * F(x) = P(alpha, beta*x) (regularized lower incomplete gamma function)
 *
 * @param x Random variable value
 * @param shape Shape parameter alpha
 * @param rate Rate parameter beta (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If shape <= 0 or rate <= 0
 */
inline double gamma_cdf(double x, double shape, double rate = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_cdf: shape must be positive");
    }
    if (rate <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_cdf: rate must be positive");
    }
    if (x <= 0.0) return 0.0;

    return gammainc_lower(shape, rate * x);
}

/**
 * @brief Gamma distribution quantile function
 *
 * @param p Probability (0 <= p <= 1)
 * @param shape Shape parameter alpha
 * @param rate Rate parameter beta (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If shape <= 0, rate <= 0, or p is outside [0, 1]
 */
inline double gamma_quantile(double p, double shape, double rate = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_quantile: shape must be positive");
    }
    if (rate <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_quantile: rate must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::gamma_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    return gammainc_lower_inv(shape, p) / rate;
}

/**
 * @brief Gamma distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param shape Shape parameter alpha
 * @param rate Rate parameter beta
 * @param engine Random engine
 * @return Random number following gamma distribution
 * @throws std::invalid_argument If shape <= 0 or rate <= 0
 */
template <typename Engine = default_random_engine>
double gamma_rand(double shape, double rate, Engine& engine)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_rand: shape must be positive");
    }
    if (rate <= 0.0) {
        throw std::invalid_argument("statcpp::gamma_rand: rate must be positive");
    }
    // std::gamma_distribution uses shape and scale (1/rate)
    std::gamma_distribution<double> dist(shape, 1.0 / rate);
    return dist(engine);
}

/**
 * @brief Gamma distribution random number generation (using default engine)
 *
 * @param shape Shape parameter alpha
 * @param rate Rate parameter beta (default: 1.0)
 * @return Random number following gamma distribution
 */
inline double gamma_rand(double shape, double rate = 1.0)
{
    return gamma_rand(shape, rate, get_random_engine());
}

// ============================================================================
// Beta Distribution
// ============================================================================

/**
 * @brief Beta distribution probability density function (PDF)
 *
 * f(x) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta) for 0 < x < 1
 *
 * @param x Random variable value
 * @param alpha Shape parameter alpha
 * @param beta_param Shape parameter beta
 * @return Probability density
 * @throws std::invalid_argument If alpha <= 0 or beta_param <= 0
 */
inline double beta_pdf(double x, double alpha, double beta_param)
{
    if (alpha <= 0.0) {
        throw std::invalid_argument("statcpp::beta_pdf: alpha must be positive");
    }
    if (beta_param <= 0.0) {
        throw std::invalid_argument("statcpp::beta_pdf: beta must be positive");
    }
    if (x <= 0.0 || x >= 1.0) return 0.0;

    return std::exp((alpha - 1.0) * std::log(x) + (beta_param - 1.0) * std::log(1.0 - x) - lbeta(alpha, beta_param));
}

/**
 * @brief Beta distribution cumulative distribution function (CDF)
 *
 * F(x) = I_x(alpha, beta) (regularized incomplete beta function)
 *
 * @param x Random variable value
 * @param alpha Shape parameter alpha
 * @param beta_param Shape parameter beta
 * @return Cumulative probability
 * @throws std::invalid_argument If alpha <= 0 or beta_param <= 0
 */
inline double beta_cdf(double x, double alpha, double beta_param)
{
    if (alpha <= 0.0) {
        throw std::invalid_argument("statcpp::beta_cdf: alpha must be positive");
    }
    if (beta_param <= 0.0) {
        throw std::invalid_argument("statcpp::beta_cdf: beta must be positive");
    }
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    return betainc(alpha, beta_param, x);
}

/**
 * @brief Beta distribution quantile function
 *
 * @param p Probability (0 <= p <= 1)
 * @param alpha Shape parameter alpha
 * @param beta_param Shape parameter beta
 * @return Quantile
 * @throws std::invalid_argument If alpha <= 0, beta_param <= 0, or p is outside [0, 1]
 */
inline double beta_quantile(double p, double alpha, double beta_param)
{
    if (alpha <= 0.0) {
        throw std::invalid_argument("statcpp::beta_quantile: alpha must be positive");
    }
    if (beta_param <= 0.0) {
        throw std::invalid_argument("statcpp::beta_quantile: beta must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::beta_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;

    return betaincinv(alpha, beta_param, p);
}

/**
 * @brief Beta distribution random number generation (using gamma variates)
 *
 * @tparam Engine Random engine type
 * @param alpha Shape parameter alpha
 * @param beta_param Shape parameter beta
 * @param engine Random engine
 * @return Random number following beta distribution
 * @throws std::invalid_argument If alpha <= 0 or beta_param <= 0
 */
template <typename Engine = default_random_engine>
double beta_rand(double alpha, double beta_param, Engine& engine)
{
    if (alpha <= 0.0) {
        throw std::invalid_argument("statcpp::beta_rand: alpha must be positive");
    }
    if (beta_param <= 0.0) {
        throw std::invalid_argument("statcpp::beta_rand: beta must be positive");
    }

    std::gamma_distribution<double> dist_a(alpha, 1.0);
    std::gamma_distribution<double> dist_b(beta_param, 1.0);

    double x = dist_a(engine);
    double y = dist_b(engine);
    return x / (x + y);
}

/**
 * @brief Beta distribution random number generation (using default engine)
 *
 * @param alpha Shape parameter alpha
 * @param beta_param Shape parameter beta
 * @return Random number following beta distribution
 */
inline double beta_rand(double alpha, double beta_param)
{
    return beta_rand(alpha, beta_param, get_random_engine());
}

// ============================================================================
// Chi-Square Distribution
// ============================================================================

/**
 * @brief Chi-square distribution probability density function (PDF)
 *
 * Special case of gamma distribution (shape = df/2, rate = 1/2)
 *
 * @param x Random variable value
 * @param df Degrees of freedom
 * @return Probability density
 * @throws std::invalid_argument If df <= 0
 */
inline double chisq_pdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_pdf: df must be positive");
    }
    return gamma_pdf(x, df / 2.0, 0.5);
}

/**
 * @brief Chi-square distribution cumulative distribution function (CDF)
 *
 * Chi-square distribution is a special case of gamma distribution: chi^2(df) = Gamma(df/2, 1/2)
 * This implementation uses the regularized incomplete gamma function:
 * F(x; df) = gamma(df/2, x/2) / Gamma(df/2) = P(df/2, x/2)
 *
 * @param x Random variable value
 * @param df Degrees of freedom
 * @return Cumulative probability
 * @throws std::invalid_argument If df <= 0
 */
inline double chisq_cdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_cdf: df must be positive");
    }
    return gamma_cdf(x, df / 2.0, 0.5);
}

/**
 * @brief Chi-square distribution quantile function
 *
 * @param p Probability (0 <= p <= 1)
 * @param df Degrees of freedom
 * @return Quantile
 * @throws std::invalid_argument If df <= 0 or p is outside [0, 1]
 */
inline double chisq_quantile(double p, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_quantile: df must be positive");
    }
    return gamma_quantile(p, df / 2.0, 0.5);
}

/**
 * @brief Chi-square distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param df Degrees of freedom
 * @param engine Random engine
 * @return Random number following chi-square distribution
 * @throws std::invalid_argument If df <= 0
 */
template <typename Engine = default_random_engine>
double chisq_rand(double df, Engine& engine)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_rand: df must be positive");
    }
    return gamma_rand(df / 2.0, 0.5, engine);
}

/**
 * @brief Chi-square distribution random number generation (using default engine)
 *
 * @param df Degrees of freedom
 * @return Random number following chi-square distribution
 */
inline double chisq_rand(double df)
{
    return chisq_rand(df, get_random_engine());
}

// ============================================================================
// Student's t-Distribution
// ============================================================================

/**
 * @brief t-distribution probability density function (PDF)
 *
 * f(x) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + x^2/nu)^(-(nu+1)/2)
 *
 * @param x Random variable value
 * @param df Degrees of freedom
 * @return Probability density
 * @throws std::invalid_argument If df <= 0
 */
inline double t_pdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::t_pdf: df must be positive");
    }

    double coef = std::exp(lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0)) / std::sqrt(df * pi);
    return coef * std::pow(1.0 + x * x / df, -(df + 1.0) / 2.0);
}

/**
 * @brief t-distribution cumulative distribution function (CDF)
 *
 * Computed using the regularized incomplete beta function.
 * The CDF of t-distribution is expressed as:
 *
 * F(x; nu) = 1 - 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2)  (for x >= 0)
 * F(x; nu) = 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2)      (for x < 0)
 *
 * where I_z(a,b) is the regularized incomplete beta function and nu is degrees of freedom.
 * This transformation enables efficient computation of t-distribution CDF via beta distribution CDF.
 *
 * @param x Random variable value
 * @param df Degrees of freedom nu
 * @return Cumulative probability
 * @throws std::invalid_argument If df <= 0
 */
inline double t_cdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::t_cdf: df must be positive");
    }

    double t2 = x * x;
    double p = betainc(df / 2.0, 0.5, df / (df + t2));

    if (x >= 0.0) {
        return 1.0 - 0.5 * p;
    } else {
        return 0.5 * p;
    }
}

/**
 * @brief t-distribution quantile function (Newton-Raphson method)
 *
 * @param p Probability (0 < p < 1)
 * @param df Degrees of freedom
 * @return Quantile
 * @throws std::invalid_argument If df <= 0 or p is outside (0, 1)
 */
inline double t_quantile(double p, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::t_quantile: df must be positive");
    }
    if (p <= 0.0 || p >= 1.0) {
        if (p == 0.0) return -std::numeric_limits<double>::infinity();
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        throw std::invalid_argument("statcpp::t_quantile: p must be in (0, 1)");
    }

    // Use normal quantile as initial guess for large df
    double x = norm_quantile(p);

    // For small df, adjust initial guess
    if (df < 4.0) {
        x *= std::sqrt(df / (df - 2.0 + 0.1));
    }

    const double eps = 1e-10;
    const int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        double f = t_cdf(x, df) - p;
        if (std::abs(f) < eps) {
            return x;
        }
        double fprime = t_pdf(x, df);
        if (fprime == 0.0) break;

        double x_new = x - f / fprime;

        if (std::abs(x_new - x) < eps * (1.0 + std::abs(x))) {
            return x_new;
        }

        x = x_new;
    }

    return x;
}

/**
 * @brief t-distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param df Degrees of freedom
 * @param engine Random engine
 * @return Random number following t-distribution
 * @throws std::invalid_argument If df <= 0
 */
template <typename Engine = default_random_engine>
double t_rand(double df, Engine& engine)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::t_rand: df must be positive");
    }
    std::student_t_distribution<double> dist(df);
    return dist(engine);
}

/**
 * @brief t-distribution random number generation (using default engine)
 *
 * @param df Degrees of freedom
 * @return Random number following t-distribution
 */
inline double t_rand(double df)
{
    return t_rand(df, get_random_engine());
}

// ============================================================================
// F-Distribution
// ============================================================================

/**
 * @brief F-distribution probability density function (PDF)
 *
 * f(x) = sqrt((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2))
 *
 * @param x Random variable value
 * @param df1 First degrees of freedom
 * @param df2 Second degrees of freedom
 * @return Probability density
 * @throws std::invalid_argument If df1 <= 0 or df2 <= 0
 */
inline double f_pdf(double x, double df1, double df2)
{
    if (df1 <= 0.0) {
        throw std::invalid_argument("statcpp::f_pdf: df1 must be positive");
    }
    if (df2 <= 0.0) {
        throw std::invalid_argument("statcpp::f_pdf: df2 must be positive");
    }
    if (x <= 0.0) return 0.0;

    double log_pdf = (df1 / 2.0) * std::log(df1) + (df2 / 2.0) * std::log(df2)
                   + (df1 / 2.0 - 1.0) * std::log(x)
                   - ((df1 + df2) / 2.0) * std::log(df1 * x + df2)
                   - lbeta(df1 / 2.0, df2 / 2.0);

    return std::exp(log_pdf);
}

/**
 * @brief F-distribution cumulative distribution function (CDF)
 *
 * Computed using the incomplete beta function.
 * F(x) = I_{d1*x/(d1*x + d2)}(d1/2, d2/2)
 *
 * @param x Random variable value
 * @param df1 First degrees of freedom
 * @param df2 Second degrees of freedom
 * @return Cumulative probability
 * @throws std::invalid_argument If df1 <= 0 or df2 <= 0
 */
inline double f_cdf(double x, double df1, double df2)
{
    if (df1 <= 0.0) {
        throw std::invalid_argument("statcpp::f_cdf: df1 must be positive");
    }
    if (df2 <= 0.0) {
        throw std::invalid_argument("statcpp::f_cdf: df2 must be positive");
    }
    if (x <= 0.0) return 0.0;

    double z = df1 * x / (df1 * x + df2);
    return betainc(df1 / 2.0, df2 / 2.0, z);
}

/**
 * @brief F-distribution quantile function (Newton-Raphson method)
 *
 * @param p Probability (0 <= p <= 1)
 * @param df1 First degrees of freedom
 * @param df2 Second degrees of freedom
 * @return Quantile
 * @throws std::invalid_argument If df1 <= 0, df2 <= 0, or p is outside [0, 1]
 */
inline double f_quantile(double p, double df1, double df2)
{
    if (df1 <= 0.0) {
        throw std::invalid_argument("statcpp::f_quantile: df1 must be positive");
    }
    if (df2 <= 0.0) {
        throw std::invalid_argument("statcpp::f_quantile: df2 must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::f_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    // Initial guess using beta quantile
    double z = betaincinv(df1 / 2.0, df2 / 2.0, p);
    double x = df2 * z / (df1 * (1.0 - z));

    const double eps = 1e-10;
    const int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        double f = f_cdf(x, df1, df2) - p;
        if (std::abs(f) < eps) {
            return x;
        }
        double fprime = f_pdf(x, df1, df2);
        if (fprime == 0.0) break;

        double x_new = x - f / fprime;
        if (x_new <= 0.0) {
            x_new = x / 2.0;
        }

        if (std::abs(x_new - x) < eps * x) {
            return x_new;
        }

        x = x_new;
    }

    return x;
}

/**
 * @brief F-distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param df1 First degrees of freedom
 * @param df2 Second degrees of freedom
 * @param engine Random engine
 * @return Random number following F-distribution
 * @throws std::invalid_argument If df1 <= 0 or df2 <= 0
 */
template <typename Engine = default_random_engine>
double f_rand(double df1, double df2, Engine& engine)
{
    if (df1 <= 0.0) {
        throw std::invalid_argument("statcpp::f_rand: df1 must be positive");
    }
    if (df2 <= 0.0) {
        throw std::invalid_argument("statcpp::f_rand: df2 must be positive");
    }
    std::fisher_f_distribution<double> dist(df1, df2);
    return dist(engine);
}

/**
 * @brief F-distribution random number generation (using default engine)
 *
 * @param df1 First degrees of freedom
 * @param df2 Second degrees of freedom
 * @return Random number following F-distribution
 */
inline double f_rand(double df1, double df2)
{
    return f_rand(df1, df2, get_random_engine());
}

// ============================================================================
// Log-normal Distribution
// ============================================================================

/**
 * @brief Log-normal distribution probability density function (PDF)
 *
 * f(x) = (1 / (x * sigma * sqrt(2*pi))) * exp(-(ln(x) - mu)^2 / (2*sigma^2))
 *
 * @param x Random variable value
 * @param mu Log-mean (default: 0.0)
 * @param sigma Log-standard deviation (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If sigma <= 0
 */
inline double lognormal_pdf(double x, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::lognormal_pdf: sigma must be positive");
    }
    if (x <= 0.0) {
        return 0.0;
    }
    double log_x = std::log(x);
    double z = (log_x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (x * sigma * sqrt_2_pi);
}

/**
 * @brief Log-normal distribution cumulative distribution function (CDF)
 *
 * F(x) = Phi((ln(x) - mu) / sigma)
 *
 * @param x Random variable value
 * @param mu Log-mean (default: 0.0)
 * @param sigma Log-standard deviation (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If sigma <= 0
 */
inline double lognormal_cdf(double x, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::lognormal_cdf: sigma must be positive");
    }
    if (x <= 0.0) {
        return 0.0;
    }
    double log_x = std::log(x);
    return norm_cdf((log_x - mu) / sigma);
}

/**
 * @brief Log-normal distribution quantile function
 *
 * Q(p) = exp(mu + sigma * Phi^(-1)(p))
 *
 * @param p Probability (0 < p < 1)
 * @param mu Log-mean (default: 0.0)
 * @param sigma Log-standard deviation (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If sigma <= 0 or p is outside (0, 1)
 */
inline double lognormal_quantile(double p, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::lognormal_quantile: sigma must be positive");
    }
    if (p <= 0.0 || p >= 1.0) {
        if (p == 0.0) return 0.0;
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        throw std::invalid_argument("statcpp::lognormal_quantile: p must be in (0, 1)");
    }
    return std::exp(mu + sigma * norm_quantile(p));
}

/**
 * @brief Log-normal distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param mu Log-mean
 * @param sigma Log-standard deviation
 * @param engine Random engine
 * @return Random number following log-normal distribution
 * @throws std::invalid_argument If sigma <= 0
 */
template <typename Engine = default_random_engine>
double lognormal_rand(double mu, double sigma, Engine& engine)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::lognormal_rand: sigma must be positive");
    }
    std::lognormal_distribution<double> dist(mu, sigma);
    return dist(engine);
}

/**
 * @brief Log-normal distribution random number generation (using default engine)
 *
 * @param mu Log-mean (default: 0.0)
 * @param sigma Log-standard deviation (default: 1.0)
 * @return Random number following log-normal distribution
 */
inline double lognormal_rand(double mu = 0.0, double sigma = 1.0)
{
    return lognormal_rand(mu, sigma, get_random_engine());
}

// ============================================================================
// Weibull Distribution
// ============================================================================

/**
 * @brief Weibull distribution probability density function (PDF)
 *
 * f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
 *
 * @param x Random variable value
 * @param shape Shape parameter k
 * @param scale Scale parameter lambda (default: 1.0)
 * @return Probability density
 * @throws std::invalid_argument If shape <= 0 or scale <= 0
 */
inline double weibull_pdf(double x, double shape, double scale = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_pdf: shape must be positive");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_pdf: scale must be positive");
    }
    if (x < 0.0) {
        return 0.0;
    }
    if (x == 0.0) {
        return (shape == 1.0) ? (1.0 / scale) : 0.0;
    }
    double z = x / scale;
    return (shape / scale) * std::pow(z, shape - 1.0) * std::exp(-std::pow(z, shape));
}

/**
 * @brief Weibull distribution cumulative distribution function (CDF)
 *
 * F(x) = 1 - exp(-(x/lambda)^k)
 *
 * @param x Random variable value
 * @param shape Shape parameter k
 * @param scale Scale parameter lambda (default: 1.0)
 * @return Cumulative probability
 * @throws std::invalid_argument If shape <= 0 or scale <= 0
 */
inline double weibull_cdf(double x, double shape, double scale = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_cdf: shape must be positive");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_cdf: scale must be positive");
    }
    if (x <= 0.0) {
        return 0.0;
    }
    double z = x / scale;
    return 1.0 - std::exp(-std::pow(z, shape));
}

/**
 * @brief Weibull distribution quantile function
 *
 * Q(p) = lambda * (-ln(1 - p))^(1/k)
 *
 * @param p Probability (0 <= p <= 1)
 * @param shape Shape parameter k
 * @param scale Scale parameter lambda (default: 1.0)
 * @return Quantile
 * @throws std::invalid_argument If shape <= 0, scale <= 0, or p is outside [0, 1]
 */
inline double weibull_quantile(double p, double shape, double scale = 1.0)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_quantile: shape must be positive");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_quantile: scale must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::weibull_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    return scale * std::pow(-std::log(1.0 - p), 1.0 / shape);
}

/**
 * @brief Weibull distribution random number generation
 *
 * @tparam Engine Random engine type
 * @param shape Shape parameter k
 * @param scale Scale parameter lambda
 * @param engine Random engine
 * @return Random number following Weibull distribution
 * @throws std::invalid_argument If shape <= 0 or scale <= 0
 */
template <typename Engine = default_random_engine>
double weibull_rand(double shape, double scale, Engine& engine)
{
    if (shape <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_rand: shape must be positive");
    }
    if (scale <= 0.0) {
        throw std::invalid_argument("statcpp::weibull_rand: scale must be positive");
    }
    std::weibull_distribution<double> dist(shape, scale);
    return dist(engine);
}

/**
 * @brief Weibull distribution random number generation (using default engine)
 *
 * @param shape Shape parameter k
 * @param scale Scale parameter lambda (default: 1.0)
 * @return Random number following Weibull distribution
 */
inline double weibull_rand(double shape, double scale = 1.0)
{
    return weibull_rand(shape, scale, get_random_engine());
}

} // namespace statcpp
