/**
 * @file special_functions.hpp
 * @brief Special mathematical functions implementation
 *
 * Provides special mathematical functions required for statistical calculations
 * including gamma, beta, and error functions.
 */

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Constants
// ============================================================================

/**
 * @brief Pi constant
 */
inline constexpr double pi = 3.14159265358979323846;

/**
 * @brief Square root of 2
 */
inline constexpr double sqrt_2 = 1.41421356237309504880;

/**
 * @brief Square root of 2*pi
 */
inline constexpr double sqrt_2_pi = 2.50662827463100050242;

/**
 * @brief Natural logarithm of sqrt(2*pi)
 */
inline constexpr double log_sqrt_2_pi = 0.91893853320467274178;

// ============================================================================
// Gamma Function / Log-Gamma Function
// ============================================================================

/**
 * @brief Internal log-gamma function implementation
 *
 * Calculates log(Gamma(x)) using Lanczos approximation.
 *
 * @param x Argument (must be positive or non-integer if negative)
 * @return log(Gamma(x))
 * @throws std::domain_error If x is a non-positive integer
 *
 * @note Reference: Numerical Recipes, Press et al.
 * @note Uses Lanczos coefficients for g=7
 */
inline double lgamma_impl(double x)
{
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("statcpp::lgamma: non-positive integer argument");
    }

    // Lanczos coefficients for g=7
    static const double c[9] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    if (x < 0.5) {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        return std::log(pi / std::sin(pi * x)) - lgamma_impl(1.0 - x);
    }

    x -= 1.0;
    double a = c[0];
    for (int i = 1; i < 9; ++i) {
        a += c[i] / (x + static_cast<double>(i));
    }

    double t = x + 7.5;
    return log_sqrt_2_pi + (x + 0.5) * std::log(t) - t + std::log(a);
}

/**
 * @brief Log-gamma function
 *
 * Computes the natural logarithm of the gamma function.
 *
 * @param x Argument
 * @return log(Gamma(x))
 * @throws std::domain_error If x is a non-positive integer
 */
inline double lgamma(double x)
{
    return lgamma_impl(x);
}

/**
 * @brief Gamma function
 *
 * Computes the gamma function Gamma(x).
 *
 * @param x Argument
 * @return Gamma(x)
 * @throws std::domain_error If x is a non-positive integer
 *
 * @note For small positive integers, computes factorial directly.
 */
inline double tgamma(double x)
{
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("statcpp::tgamma: non-positive integer argument");
    }

    // For small positive integers, use factorial
    if (x > 0.0 && x <= 20.0 && x == std::floor(x)) {
        double result = 1.0;
        for (int i = 2; i < static_cast<int>(x); ++i) {
            result *= i;
        }
        return result;
    }

    return std::exp(lgamma_impl(x));
}

// ============================================================================
// Beta Function / Incomplete Beta Function
// ============================================================================

/**
 * @brief Beta function
 *
 * Computes the beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).
 *
 * @param a First parameter (must be positive)
 * @param b Second parameter (must be positive)
 * @return B(a, b)
 * @throws std::domain_error If a or b is non-positive
 */
inline double beta(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::beta: parameters must be positive");
    }
    return std::exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

/**
 * @brief Log-beta function
 *
 * Computes the natural logarithm of the beta function.
 *
 * @param a First parameter (must be positive)
 * @param b Second parameter (must be positive)
 * @return log(B(a, b))
 * @throws std::domain_error If a or b is non-positive
 */
inline double lbeta(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::lbeta: parameters must be positive");
    }
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/**
 * @brief Internal regularized incomplete beta function
 *
 * Computes the regularized incomplete beta function I_x(a, b) using continued fraction expansion.
 *
 * @param a First parameter
 * @param b Second parameter
 * @param x Upper limit of integration (must be in [0, 1])
 * @param recursion_depth Recursion depth for tracking
 * @return I_x(a, b)
 * @throws std::runtime_error If recursion depth is exceeded
 *
 * @note Reference: Numerical Recipes, Press et al.
 * @note Uses Lentz's algorithm
 */
inline double betainc_impl(double a, double b, double x, int recursion_depth)
{
    // Recursion depth check (prevent infinite recursion)
    if (recursion_depth > 1) {
        throw std::runtime_error("statcpp::betainc: maximum recursion depth exceeded");
    }

    // Use symmetry relation for better convergence
    // I_x(a, b) = 1 - I_{1-x}(b, a)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - betainc_impl(b, a, 1.0 - x, recursion_depth + 1);
    }

    // Continued fraction (Lentz's algorithm)
    const double eps = std::numeric_limits<double>::epsilon();
    const double tiny = std::numeric_limits<double>::min();
    const int max_iter = 200;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < tiny) d = tiny;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= max_iter; ++m) {
        int m2 = 2 * m;

        // Even step
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double del = d * c;
        h *= del;

        if (std::abs(del - 1.0) < eps) {
            break;
        }
    }

    double front = std::exp(a * std::log(x) + b * std::log(1.0 - x) - lbeta(a, b)) / a;
    return front * h;
}

/**
 * @brief Regularized incomplete beta function
 *
 * Computes the regularized incomplete beta function I_x(a, b).
 *
 * @param a First parameter (must be positive)
 * @param b Second parameter (must be positive)
 * @param x Upper limit of integration (must be in [0, 1])
 * @return I_x(a, b)
 * @throws std::domain_error If parameters are invalid
 *
 * @note Used for computing CDFs of beta, F, and t distributions.
 */
inline double betainc(double a, double b, double x)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::betainc: parameters must be positive");
    }
    if (x < 0.0 || x > 1.0) {
        throw std::domain_error("statcpp::betainc: x must be in [0, 1]");
    }
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    return betainc_impl(a, b, x, 0);
}

/**
 * @brief Inverse regularized incomplete beta function
 *
 * Computes x such that I_x(a, b) = p (quantile function).
 *
 * @param a First parameter (must be positive)
 * @param b Second parameter (must be positive)
 * @param p Probability value (must be in [0, 1])
 * @return x such that I_x(a, b) = p
 * @throws std::domain_error If parameters are invalid
 *
 * @note Uses Newton-Raphson iteration
 */
inline double betaincinv(double a, double b, double p)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::betaincinv: parameters must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::domain_error("statcpp::betaincinv: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;

    const double eps = 1e-10;
    const int max_iter = 100;

    // Initial guess using approximation
    double x = a / (a + b);
    if (a < 1.0 || b < 1.0) {
        x = 0.5;
    }

    // Newton-Raphson with bisection fallback
    double lo = 0.0, hi = 1.0;

    for (int i = 0; i < max_iter; ++i) {
        double f = betainc(a, b, x) - p;

        if (std::abs(f) < eps) {
            return x;
        }

        // Update bisection bounds
        if (f < 0.0) {
            lo = x;
        } else {
            hi = x;
        }

        // Derivative: d/dx I_x(a,b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
        double df = std::exp((a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x) - lbeta(a, b));

        double dx = f / df;
        double x_new = x - dx;

        // Use bisection if Newton step goes out of bounds
        if (x_new <= lo || x_new >= hi) {
            x_new = (lo + hi) / 2.0;
        }

        if (std::abs(x_new - x) < eps * x) {
            return x_new;
        }

        x = x_new;
    }

    return x;
}

// ============================================================================
// Error Function (erf / erfc)
// ============================================================================

/**
 * @brief Error function
 *
 * Computes the error function erf(x).
 *
 * @param x Argument
 * @return erf(x)
 *
 * @note Reference: Abramowitz and Stegun, 7.1.26
 * @note Uses Horner's method approximation
 */
inline double erf(double x)
{
    // Constants
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x >= 0) ? 1 : -1;
    x = std::abs(x);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return sign * y;
}

/**
 * @brief Complementary error function
 *
 * Computes the complementary error function erfc(x) = 1 - erf(x).
 *
 * @param x Argument
 * @return erfc(x)
 */
inline double erfc(double x)
{
    return 1.0 - erf(x);
}

// ============================================================================
// Normal Distribution CDF and Quantile (Phi and Phi^{-1})
// ============================================================================

/**
 * @brief Standard normal CDF
 *
 * Computes the cumulative distribution function Phi(x) of the standard normal distribution.
 *
 * @param x Argument
 * @return Phi(x) = P(Z <= x) where Z ~ N(0, 1)
 */
inline double norm_cdf(double x)
{
    return 0.5 * (1.0 + erf(x / sqrt_2));
}

/**
 * @brief Standard normal quantile function
 *
 * Computes the quantile function Phi^{-1}(p) of the standard normal distribution.
 *
 * @param p Probability (must be in (0, 1))
 * @return x such that Phi(x) = p
 *
 * @note Reference: https://home.online.no/~pjacklam/notes/invnorm/
 * @note Rational approximation by Acklam's algorithm
 */
inline double norm_quantile(double p)
{
    if (p <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    // Coefficients for rational approximation
    static const double a[6] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    };
    static const double b[5] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[6] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    };
    static const double d[4] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    };

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r;

    if (p < p_low) {
        // Rational approximation for lower region
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        // Rational approximation for central region
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        // Rational approximation for upper region
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// ============================================================================
// Lower Incomplete Gamma Function (for chi-square, gamma distributions)
// ============================================================================

/**
 * @brief Lower regularized incomplete gamma function
 *
 * Computes the lower regularized incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a).
 *
 * @param a Shape parameter (must be positive)
 * @param x Upper limit of integration (must be non-negative)
 * @return P(a, x)
 * @throws std::domain_error If parameters are invalid
 *
 * @note Uses series expansion for x < a + 1, continued fraction for x >= a + 1
 */
inline double gammainc_lower(double a, double x)
{
    if (a <= 0.0) {
        throw std::domain_error("statcpp::gammainc_lower: a must be positive");
    }
    if (x < 0.0) {
        throw std::domain_error("statcpp::gammainc_lower: x must be non-negative");
    }
    if (x == 0.0) return 0.0;

    const double eps = std::numeric_limits<double>::epsilon();
    const int max_iter = 200;

    // Use series expansion for x < a + 1
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n <= max_iter; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < eps * std::abs(sum)) {
                break;
            }
        }
        return sum * std::exp(-x + a * std::log(x) - lgamma(a));
    }

    // Use continued fraction for x >= a + 1 (Lentz's algorithm)
    const double tiny = std::numeric_limits<double>::min();

    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;

    for (int i = 1; i <= max_iter; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < eps) {
            break;
        }
    }

    // Q(a,x) = 1 - P(a,x), where we computed Q via continued fraction
    double q = std::exp(-x + a * std::log(x) - lgamma(a)) * h;
    return 1.0 - q;
}

/**
 * @brief Upper regularized incomplete gamma function
 *
 * Computes the upper regularized incomplete gamma function Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x).
 *
 * @param a Shape parameter (must be positive)
 * @param x Lower limit of integration (must be non-negative)
 * @return Q(a, x)
 * @throws std::domain_error If parameters are invalid
 */
inline double gammainc_upper(double a, double x)
{
    return 1.0 - gammainc_lower(a, x);
}

/**
 * @brief Inverse lower regularized incomplete gamma function
 *
 * Computes x such that P(a, x) = p.
 *
 * @param a Shape parameter (must be positive)
 * @param p Probability value (must be in [0, 1])
 * @return x such that P(a, x) = p
 * @throws std::domain_error If parameters are invalid
 *
 * @note Uses Newton-Raphson iteration
 * @note Uses Wilson-Hilferty approximation for initial guess when a > 1
 */
inline double gammainc_lower_inv(double a, double p)
{
    if (a <= 0.0) {
        throw std::domain_error("statcpp::gammainc_lower_inv: a must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::domain_error("statcpp::gammainc_lower_inv: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    const double eps = 1e-10;
    const int max_iter = 100;

    // Initial guess
    double x;
    if (a > 1.0) {
        // Use Wilson-Hilferty approximation
        double t = norm_quantile(p);
        double v = t * std::sqrt(1.0 / (9.0 * a)) + 1.0 - 1.0 / (9.0 * a);
        x = a * v * v * v;
        if (x <= 0.0) x = 0.5;
    } else {
        x = std::pow(p * tgamma(a + 1.0), 1.0 / a);
    }

    // Newton-Raphson
    for (int i = 0; i < max_iter; ++i) {
        double f = gammainc_lower(a, x) - p;
        if (std::abs(f) < eps) {
            return x;
        }

        // Derivative: d/dx P(a,x) = x^(a-1) * e^(-x) / Gamma(a)
        double df = std::exp((a - 1.0) * std::log(x) - x - lgamma(a));
        if (df == 0.0) {
            // Fallback to bisection
            break;
        }

        double x_new = x - f / df;
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

} // namespace statcpp
