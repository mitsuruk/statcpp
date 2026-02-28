/**
 * @file special_functions.hpp
 * @brief 特殊関数の実装 (Special mathematical functions implementation)
 *
 * ガンマ関数、ベータ関数、誤差関数など統計計算に必要な特殊関数を提供します。
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
 * @brief 円周率 π (Pi constant)
 */
inline constexpr double pi = 3.14159265358979323846;

/**
 * @brief √2 の値 (Square root of 2)
 */
inline constexpr double sqrt_2 = 1.41421356237309504880;

/**
 * @brief √(2π) の値 (Square root of 2π)
 */
inline constexpr double sqrt_2_pi = 2.50662827463100050242;

/**
 * @brief log(√(2π)) の値 (Natural logarithm of √(2π))
 */
inline constexpr double log_sqrt_2_pi = 0.91893853320467274178;

// ============================================================================
// Gamma Function / Log-Gamma Function
// ============================================================================

/**
 * @brief 対数ガンマ関数の内部実装 (Internal log-gamma function implementation)
 *
 * Lanczos approximation を使用して log(Γ(x)) を計算します。
 * Calculates log(Γ(x)) using Lanczos approximation.
 *
 * @param x 引数 (argument, must be positive or non-integer if negative)
 * @return log(Γ(x))
 * @throws std::domain_error x が非正の整数の場合 (if x is a non-positive integer)
 *
 * @note Reference: Numerical Recipes, Press et al.
 * @note g=7 の Lanczos 係数を使用 (Uses Lanczos coefficients for g=7)
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
 * @brief 対数ガンマ関数 (Log-gamma function)
 *
 * ガンマ関数の自然対数を計算します。
 * Computes the natural logarithm of the gamma function.
 *
 * @param x 引数 (argument)
 * @return log(Γ(x))
 * @throws std::domain_error x が非正の整数の場合 (if x is a non-positive integer)
 */
inline double lgamma(double x)
{
    return lgamma_impl(x);
}

/**
 * @brief ガンマ関数 (Gamma function)
 *
 * ガンマ関数 Γ(x) を計算します。
 * Computes the gamma function Γ(x).
 *
 * @param x 引数 (argument)
 * @return Γ(x)
 * @throws std::domain_error x が非正の整数の場合 (if x is a non-positive integer)
 *
 * @note 小さい正の整数については階乗を直接計算します。
 * For small positive integers, computes factorial directly.
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
 * @brief ベータ関数 (Beta function)
 *
 * ベータ関数 B(a, b) = Γ(a) * Γ(b) / Γ(a + b) を計算します。
 * Computes the beta function B(a, b) = Γ(a) * Γ(b) / Γ(a + b).
 *
 * @param a 第1パラメータ (first parameter, must be positive)
 * @param b 第2パラメータ (second parameter, must be positive)
 * @return B(a, b)
 * @throws std::domain_error a または b が非正の場合 (if a or b is non-positive)
 */
inline double beta(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::beta: parameters must be positive");
    }
    return std::exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

/**
 * @brief 対数ベータ関数 (Log-beta function)
 *
 * ベータ関数の自然対数を計算します。
 * Computes the natural logarithm of the beta function.
 *
 * @param a 第1パラメータ (first parameter, must be positive)
 * @param b 第2パラメータ (second parameter, must be positive)
 * @return log(B(a, b))
 * @throws std::domain_error a または b が非正の場合 (if a or b is non-positive)
 */
inline double lbeta(double a, double b)
{
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("statcpp::lbeta: parameters must be positive");
    }
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/**
 * @brief 正則化不完全ベータ関数の内部実装 (Internal regularized incomplete beta function)
 *
 * 連分数展開を使用して正則化不完全ベータ関数 I_x(a, b) を計算します。
 * Computes the regularized incomplete beta function I_x(a, b) using continued fraction expansion.
 *
 * @param a 第1パラメータ (first parameter)
 * @param b 第2パラメータ (second parameter)
 * @param x 積分上限 (upper limit of integration, must be in [0, 1])
 * @param recursion_depth 再帰深さ (recursion depth for tracking)
 * @return I_x(a, b)
 * @throws std::runtime_error 再帰深さが上限を超えた場合 (if recursion depth is exceeded)
 *
 * @note Reference: Numerical Recipes, Press et al.
 * @note Lentz's algorithm を使用 (Uses Lentz's algorithm)
 */
inline double betainc_impl(double a, double b, double x, int recursion_depth)
{
    // 再帰深さチェック（無限再帰防止）
    // 対称変換 I_x(a,b) = 1 - I_{1-x}(b,a) による再帰呼び出しは最大1回だけ発生する。
    // depth=1 で渡された (b, a, 1-x) に対して再び対称変換が必要になることは
    // 数学的にあり得ないため、depth > 1 は実際には到達不可能だが、
    // 安全ガードとして保持する。
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
 * @brief 正則化不完全ベータ関数 (Regularized incomplete beta function)
 *
 * 正則化不完全ベータ関数 I_x(a, b) を計算します。
 * Computes the regularized incomplete beta function I_x(a, b).
 *
 * @param a 第1パラメータ (first parameter, must be positive)
 * @param b 第2パラメータ (second parameter, must be positive)
 * @param x 積分上限 (upper limit of integration, must be in [0, 1])
 * @return I_x(a, b)
 * @throws std::domain_error パラメータが無効な場合 (if parameters are invalid)
 *
 * @note ベータ分布、F分布、t分布の累積分布関数の計算に使用されます。
 * Used for computing CDFs of beta, F, and t distributions.
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
 * @brief 正則化不完全ベータ関数の逆関数 (Inverse regularized incomplete beta function)
 *
 * I_x(a, b) = p を満たす x を計算します（分位点関数）。
 * Computes x such that I_x(a, b) = p (quantile function).
 *
 * @param a 第1パラメータ (first parameter, must be positive)
 * @param b 第2パラメータ (second parameter, must be positive)
 * @param p 確率値 (probability value, must be in [0, 1])
 * @return x such that I_x(a, b) = p
 * @throws std::domain_error パラメータが無効な場合 (if parameters are invalid)
 *
 * @note Newton-Raphson 法を使用 (Uses Newton-Raphson iteration)
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
 * @brief 誤差関数 (Error function)
 *
 * 誤差関数 erf(x) を計算します。
 * Computes the error function erf(x).
 *
 * @param x 引数 (argument)
 * @return erf(x)
 *
 * @note Reference: Abramowitz and Stegun, 7.1.26
 * @note Horner 法による多項式近似を使用 (Uses Horner's method approximation)
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
 * @brief 相補誤差関数 (Complementary error function)
 *
 * 相補誤差関数 erfc(x) = 1 - erf(x) を計算します。
 * Computes the complementary error function erfc(x) = 1 - erf(x).
 *
 * @param x 引数 (argument)
 * @return erfc(x)
 */
inline double erfc(double x)
{
    return 1.0 - erf(x);
}

// ============================================================================
// Normal Distribution CDF and Quantile (Φ and Φ⁻¹)
// ============================================================================

/**
 * @brief 標準正規分布の累積分布関数 (Standard normal CDF)
 *
 * 標準正規分布の累積分布関数 Φ(x) を計算します。
 * Computes the cumulative distribution function Φ(x) of the standard normal distribution.
 *
 * @param x 引数 (argument)
 * @return Φ(x) = P(Z ≤ x) where Z ~ N(0, 1)
 */
inline double norm_cdf(double x)
{
    return 0.5 * (1.0 + erf(x / sqrt_2));
}

/**
 * @brief 標準正規分布の逆累積分布関数 (Standard normal quantile function)
 *
 * 標準正規分布の分位点関数 Φ⁻¹(p) を計算します。
 * Computes the quantile function Φ⁻¹(p) of the standard normal distribution.
 *
 * @param p 確率値 (probability, must be in (0, 1))
 * @return x such that Φ(x) = p
 *
 * @note Reference: https://home.online.no/~pjacklam/notes/invnorm/
 * @note Acklam のアルゴリズムによる有理関数近似 (Rational approximation by Acklam's algorithm)
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
 * @brief 下側正則化不完全ガンマ関数 (Lower regularized incomplete gamma function)
 *
 * 下側正則化不完全ガンマ関数 P(a, x) = γ(a, x) / Γ(a) を計算します。
 * Computes the lower regularized incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
 *
 * @param a 形状パラメータ (shape parameter, must be positive)
 * @param x 積分上限 (upper limit of integration, must be non-negative)
 * @return P(a, x)
 * @throws std::domain_error パラメータが無効な場合 (if parameters are invalid)
 *
 * @note x < a + 1 の場合は級数展開、x ≥ a + 1 の場合は連分数展開を使用
 * Uses series expansion for x < a + 1, continued fraction for x ≥ a + 1
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
 * @brief 上側正則化不完全ガンマ関数 (Upper regularized incomplete gamma function)
 *
 * 上側正則化不完全ガンマ関数 Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x) を計算します。
 * Computes the upper regularized incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x).
 *
 * @param a 形状パラメータ (shape parameter, must be positive)
 * @param x 積分下限 (lower limit of integration, must be non-negative)
 * @return Q(a, x)
 * @throws std::domain_error パラメータが無効な場合 (if parameters are invalid)
 */
inline double gammainc_upper(double a, double x)
{
    return 1.0 - gammainc_lower(a, x);
}

/**
 * @brief 下側正則化不完全ガンマ関数の逆関数 (Inverse lower regularized incomplete gamma function)
 *
 * P(a, x) = p を満たす x を計算します。
 * Computes x such that P(a, x) = p.
 *
 * @param a 形状パラメータ (shape parameter, must be positive)
 * @param p 確率値 (probability value, must be in [0, 1])
 * @return x such that P(a, x) = p
 * @throws std::domain_error パラメータが無効な場合 (if parameters are invalid)
 *
 * @note Newton-Raphson 法を使用 (Uses Newton-Raphson iteration)
 * @note a > 1 の場合は Wilson-Hilferty 近似で初期値を設定
 * Uses Wilson-Hilferty approximation for initial guess when a > 1
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
