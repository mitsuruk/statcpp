/**
 * @file continuous_distributions.hpp
 * @brief 連続分布関数
 *
 * 一様分布、正規分布、指数分布、ガンマ分布、ベータ分布、カイ二乗分布、t分布、F分布、
 * 対数正規分布、ワイブル分布のPDF、CDF、分位点、乱数生成関数を提供します。
 */

#pragma once

#include "statcpp/special_functions.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Uniform Distribution
// ============================================================================

/**
 * @brief 一様分布の確率密度関数 (PDF)
 *
 * f(x) = 1 / (b - a) for a <= x <= b
 *
 * @param x 確率変数の値
 * @param a 下限（デフォルト: 0.0）
 * @param b 上限（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument a >= b の場合
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
 * @brief 一様分布の累積分布関数 (CDF)
 *
 * F(x) = (x - a) / (b - a)
 *
 * @param x 確率変数の値
 * @param a 下限（デフォルト: 0.0）
 * @param b 上限（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument a >= b の場合
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
 * @brief 一様分布の分位点関数 (逆CDF)
 *
 * Q(p) = a + p * (b - a)
 *
 * @param p 確率 (0 <= p <= 1)
 * @param a 下限（デフォルト: 0.0）
 * @param b 上限（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument a >= b または p が [0, 1] の範囲外の場合
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
 * @brief 一様分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param a 下限
 * @param b 上限
 * @param engine 乱数エンジン
 * @return 一様分布に従う乱数
 * @throws std::invalid_argument a >= b の場合
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
 * @brief 一様分布の乱数生成（デフォルトエンジン使用）
 *
 * @param a 下限（デフォルト: 0.0）
 * @param b 上限（デフォルト: 1.0）
 * @return 一様分布に従う乱数
 */
inline double uniform_rand(double a = 0.0, double b = 1.0)
{
    return uniform_rand(a, b, get_random_engine());
}

// ============================================================================
// Normal Distribution
// ============================================================================

/**
 * @brief 正規分布の確率密度関数 (PDF)
 *
 * f(x) = (1 / (σ√(2π))) * exp(-(x-μ)²/(2σ²))
 *
 * @param x 確率変数の値
 * @param mu 平均（デフォルト: 0.0）
 * @param sigma 標準偏差（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument sigma <= 0 の場合
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
 * @brief 正規分布の累積分布関数 (CDF)
 *
 * F(x) = Φ((x - μ) / σ)
 *
 * @param x 確率変数の値
 * @param mu 平均（デフォルト: 0.0）
 * @param sigma 標準偏差（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument sigma <= 0 の場合
 */
inline double normal_cdf(double x, double mu = 0.0, double sigma = 1.0)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::normal_cdf: sigma must be positive");
    }
    return norm_cdf((x - mu) / sigma);
}

/**
 * @brief 正規分布の分位点関数（逆CDF、パーセント点関数）
 *
 * 確率 p に対応する分位点（パーセンタイル値）を返します。
 * Q(p) = μ + σ * Φ⁻¹(p)
 *
 * ここで Φ⁻¹(p) は標準正規分布の逆累積分布関数（probit関数）です。
 * 例: Q(0.975) ≈ μ + 1.96σ（95%信頼区間の上限）
 *
 * @param p 確率 (0 < p < 1)
 * @param mu 平均（デフォルト: 0.0）
 * @param sigma 標準偏差（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument sigma <= 0 または p が (0, 1) の範囲外の場合
 *
 * @note 境界値の扱い: p = 0 の場合は -∞、p = 1 の場合は +∞ を返します。
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
 * @brief 正規分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param mu 平均
 * @param sigma 標準偏差
 * @param engine 乱数エンジン
 * @return 正規分布に従う乱数
 * @throws std::invalid_argument sigma <= 0 の場合
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
 * @brief 正規分布の乱数生成（デフォルトエンジン使用）
 *
 * @param mu 平均（デフォルト: 0.0）
 * @param sigma 標準偏差（デフォルト: 1.0）
 * @return 正規分布に従う乱数
 */
inline double normal_rand(double mu = 0.0, double sigma = 1.0)
{
    return normal_rand(mu, sigma, get_random_engine());
}

// ============================================================================
// Exponential Distribution
// ============================================================================

/**
 * @brief 指数分布の確率密度関数 (PDF)
 *
 * f(x) = λ * exp(-λx) for x >= 0
 *
 * @param x 確率変数の値
 * @param lambda 率パラメータ（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument lambda <= 0 の場合
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
 * @brief 指数分布の累積分布関数 (CDF)
 *
 * F(x) = 1 - exp(-λx)
 *
 * @param x 確率変数の値
 * @param lambda 率パラメータ（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument lambda <= 0 の場合
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
 * @brief 指数分布の分位点関数
 *
 * Q(p) = -ln(1-p) / λ
 *
 * @param p 確率 (0 <= p < 1)
 * @param lambda 率パラメータ（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument lambda <= 0 または p が [0, 1) の範囲外の場合
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
 * @brief 指数分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param lambda 率パラメータ
 * @param engine 乱数エンジン
 * @return 指数分布に従う乱数
 * @throws std::invalid_argument lambda <= 0 の場合
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
 * @brief 指数分布の乱数生成（デフォルトエンジン使用）
 *
 * @param lambda 率パラメータ（デフォルト: 1.0）
 * @return 指数分布に従う乱数
 */
inline double exponential_rand(double lambda = 1.0)
{
    return exponential_rand(lambda, get_random_engine());
}

// ============================================================================
// Gamma Distribution
// ============================================================================

/**
 * @brief ガンマ分布の確率密度関数 (PDF)
 *
 * f(x) = (β^α / Γ(α)) * x^(α-1) * exp(-βx) for x > 0
 * パラメータ: shape = α (k), rate = β (1/θ)
 *
 * @param x 確率変数の値
 * @param shape 形状パラメータ α
 * @param rate 率パラメータ β（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument shape <= 0 または rate <= 0 の場合
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
 * @brief ガンマ分布の累積分布関数 (CDF)
 *
 * F(x) = P(α, βx) （正則化下側不完全ガンマ関数）
 *
 * @param x 確率変数の値
 * @param shape 形状パラメータ α
 * @param rate 率パラメータ β（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument shape <= 0 または rate <= 0 の場合
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
 * @brief ガンマ分布の分位点関数
 *
 * @param p 確率 (0 <= p <= 1)
 * @param shape 形状パラメータ α
 * @param rate 率パラメータ β（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument shape <= 0, rate <= 0, または p が [0, 1] の範囲外の場合
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
 * @brief ガンマ分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param shape 形状パラメータ α
 * @param rate 率パラメータ β
 * @param engine 乱数エンジン
 * @return ガンマ分布に従う乱数
 * @throws std::invalid_argument shape <= 0 または rate <= 0 の場合
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
 * @brief ガンマ分布の乱数生成（デフォルトエンジン使用）
 *
 * @param shape 形状パラメータ α
 * @param rate 率パラメータ β（デフォルト: 1.0）
 * @return ガンマ分布に従う乱数
 */
inline double gamma_rand(double shape, double rate = 1.0)
{
    return gamma_rand(shape, rate, get_random_engine());
}

// ============================================================================
// Beta Distribution
// ============================================================================

/**
 * @brief ベータ分布の確率密度関数 (PDF)
 *
 * f(x) = x^(α-1) * (1-x)^(β-1) / B(α, β) for 0 < x < 1
 *
 * @param x 確率変数の値
 * @param alpha 形状パラメータ α
 * @param beta_param 形状パラメータ β
 * @return 確率密度
 * @throws std::invalid_argument alpha <= 0 または beta_param <= 0 の場合
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
 * @brief ベータ分布の累積分布関数 (CDF)
 *
 * F(x) = I_x(α, β) （正則化不完全ベータ関数）
 *
 * @param x 確率変数の値
 * @param alpha 形状パラメータ α
 * @param beta_param 形状パラメータ β
 * @return 累積確率
 * @throws std::invalid_argument alpha <= 0 または beta_param <= 0 の場合
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
 * @brief ベータ分布の分位点関数
 *
 * @param p 確率 (0 <= p <= 1)
 * @param alpha 形状パラメータ α
 * @param beta_param 形状パラメータ β
 * @return 分位点
 * @throws std::invalid_argument alpha <= 0, beta_param <= 0, または p が [0, 1] の範囲外の場合
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
 * @brief ベータ分布の乱数生成（ガンマ変量を使用）
 *
 * @tparam Engine 乱数エンジン型
 * @param alpha 形状パラメータ α
 * @param beta_param 形状パラメータ β
 * @param engine 乱数エンジン
 * @return ベータ分布に従う乱数
 * @throws std::invalid_argument alpha <= 0 または beta_param <= 0 の場合
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
 * @brief ベータ分布の乱数生成（デフォルトエンジン使用）
 *
 * @param alpha 形状パラメータ α
 * @param beta_param 形状パラメータ β
 * @return ベータ分布に従う乱数
 */
inline double beta_rand(double alpha, double beta_param)
{
    return beta_rand(alpha, beta_param, get_random_engine());
}

// ============================================================================
// Chi-Square Distribution
// ============================================================================

/**
 * @brief χ²分布の確率密度関数 (PDF)
 *
 * ガンマ分布の特殊ケース (shape = df/2, rate = 1/2)
 *
 * @param x 確率変数の値
 * @param df 自由度
 * @return 確率密度
 * @throws std::invalid_argument df <= 0 の場合
 */
inline double chisq_pdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_pdf: df must be positive");
    }
    return gamma_pdf(x, df / 2.0, 0.5);
}

/**
 * @brief χ²分布の累積分布関数 (CDF)
 *
 * χ²分布はガンマ分布の特殊ケースです：χ²(df) = Gamma(df/2, 1/2)
 * この実装では正則化不完全ガンマ関数を使用して計算します：
 * F(x; df) = γ(df/2, x/2) / Γ(df/2) = P(df/2, x/2)
 *
 * @param x 確率変数の値
 * @param df 自由度
 * @return 累積確率
 * @throws std::invalid_argument df <= 0 の場合
 */
inline double chisq_cdf(double x, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_cdf: df must be positive");
    }
    return gamma_cdf(x, df / 2.0, 0.5);
}

/**
 * @brief χ²分布の分位点関数
 *
 * @param p 確率 (0 <= p <= 1)
 * @param df 自由度
 * @return 分位点
 * @throws std::invalid_argument df <= 0 または p が [0, 1] の範囲外の場合
 */
inline double chisq_quantile(double p, double df)
{
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::chisq_quantile: df must be positive");
    }
    return gamma_quantile(p, df / 2.0, 0.5);
}

/**
 * @brief χ²分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param df 自由度
 * @param engine 乱数エンジン
 * @return χ²分布に従う乱数
 * @throws std::invalid_argument df <= 0 の場合
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
 * @brief χ²分布の乱数生成（デフォルトエンジン使用）
 *
 * @param df 自由度
 * @return χ²分布に従う乱数
 */
inline double chisq_rand(double df)
{
    return chisq_rand(df, get_random_engine());
}

// ============================================================================
// Student's t-Distribution
// ============================================================================

/**
 * @brief t分布の確率密度関数 (PDF)
 *
 * f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
 *
 * @param x 確率変数の値
 * @param df 自由度
 * @return 確率密度
 * @throws std::invalid_argument df <= 0 の場合
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
 * @brief t分布の累積分布関数 (CDF)
 *
 * 正則化不完全ベータ関数を使用して計算します。
 * t分布のCDFは以下の関係式で表されます：
 *
 * F(x; ν) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2)  (x ≥ 0 の場合)
 * F(x; ν) = 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2)      (x < 0 の場合)
 *
 * ここで I_z(a,b) は正則化不完全ベータ関数、ν は自由度です。
 * この変換によりt分布のCDFをベータ分布のCDFで効率的に計算できます。
 *
 * @param x 確率変数の値
 * @param df 自由度 ν
 * @return 累積確率
 * @throws std::invalid_argument df <= 0 の場合
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
 * @brief t分布の分位点関数（Newton-Raphson法）
 *
 * @param p 確率 (0 < p < 1)
 * @param df 自由度
 * @return 分位点
 * @throws std::invalid_argument df <= 0 または p が (0, 1) の範囲外の場合
 *
 * @note Newton-Raphson反復法を使用（許容誤差 eps = 1e-10、最大反復回数 50）。
 *       最大反復回数に達しても収束しない場合、その時点での最良の近似値を返します
 *       （例外は送出されません）。内部点（0 < p < 1）では戻り値は有限値です。
 *       境界値 p = 0 または p = 1 では、分布の定義に従い +/-infinity を
 *       返す場合があります。実用上、一般的な入力範囲では常に収束します。
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

    // For small df (but df > 2), adjust initial guess using variance scaling
    if (df > 2.0 && df < 4.0) {
        x *= std::sqrt(df / (df - 2.0));
    }

    const double eps = 1e-10;
    const int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        if (!std::isfinite(x)) {
            x = norm_quantile(p);
        }
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

    if (!std::isfinite(x)) {
        x = norm_quantile(p);
    }
    return x;
}

/**
 * @brief t分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param df 自由度
 * @param engine 乱数エンジン
 * @return t分布に従う乱数
 * @throws std::invalid_argument df <= 0 の場合
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
 * @brief t分布の乱数生成（デフォルトエンジン使用）
 *
 * @param df 自由度
 * @return t分布に従う乱数
 */
inline double t_rand(double df)
{
    return t_rand(df, get_random_engine());
}

// ============================================================================
// F-Distribution
// ============================================================================

/**
 * @brief F分布の確率密度関数 (PDF)
 *
 * f(x) = sqrt((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2))
 *
 * @param x 確率変数の値
 * @param df1 第1自由度
 * @param df2 第2自由度
 * @return 確率密度
 * @throws std::invalid_argument df1 <= 0 または df2 <= 0 の場合
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
 * @brief F分布の累積分布関数 (CDF)
 *
 * 不完全ベータ関数を使用して計算します。
 * F(x) = I_{d1*x/(d1*x + d2)}(d1/2, d2/2)
 *
 * @param x 確率変数の値
 * @param df1 第1自由度
 * @param df2 第2自由度
 * @return 累積確率
 * @throws std::invalid_argument df1 <= 0 または df2 <= 0 の場合
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
 * @brief F分布の分位点関数（Newton-Raphson法）
 *
 * @param p 確率 (0 <= p <= 1)
 * @param df1 第1自由度
 * @param df2 第2自由度
 * @return 分位点
 * @throws std::invalid_argument df1 <= 0, df2 <= 0, または p が [0, 1] の範囲外の場合
 *
 * @note Newton-Raphson反復法を使用（許容誤差 eps = 1e-10、最大反復回数 50）。
 *       最大反復回数に達しても収束しない場合、その時点での最良の近似値を返します
 *       （例外は送出されません）。内部点（0 < p < 1）では戻り値は有限値です。
 *       境界値 p = 0 または p = 1 では、分布の定義に従い +/-infinity を
 *       返す場合があります。実用上、一般的な入力範囲では常に収束します。
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
    if (!std::isfinite(x) || x <= 0.0) {
        x = 1.0;
    }

    const double eps = 1e-10;
    const int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        if (!std::isfinite(x) || x <= 0.0) {
            x = 1.0;
        }
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

    if (!std::isfinite(x) || x <= 0.0) {
        x = 1.0;
    }
    return x;
}

/**
 * @brief F分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param df1 第1自由度
 * @param df2 第2自由度
 * @param engine 乱数エンジン
 * @return F分布に従う乱数
 * @throws std::invalid_argument df1 <= 0 または df2 <= 0 の場合
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
 * @brief F分布の乱数生成（デフォルトエンジン使用）
 *
 * @param df1 第1自由度
 * @param df2 第2自由度
 * @return F分布に従う乱数
 */
inline double f_rand(double df1, double df2)
{
    return f_rand(df1, df2, get_random_engine());
}

// ============================================================================
// Log-normal Distribution
// ============================================================================

/**
 * @brief 対数正規分布の確率密度関数 (PDF)
 *
 * f(x) = (1 / (x σ √(2π))) * exp(-(ln(x) - μ)² / (2σ²))
 *
 * @param x 確率変数の値
 * @param mu 対数平均（デフォルト: 0.0）
 * @param sigma 対数標準偏差（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument sigma <= 0 の場合
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
 * @brief 対数正規分布の累積分布関数 (CDF)
 *
 * F(x) = Φ((ln(x) - μ) / σ)
 *
 * @param x 確率変数の値
 * @param mu 対数平均（デフォルト: 0.0）
 * @param sigma 対数標準偏差（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument sigma <= 0 の場合
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
 * @brief 対数正規分布の分位点関数
 *
 * Q(p) = exp(μ + σ * Φ⁻¹(p))
 *
 * @param p 確率 (0 < p < 1)
 * @param mu 対数平均（デフォルト: 0.0）
 * @param sigma 対数標準偏差（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument sigma <= 0 または p が (0, 1) の範囲外の場合
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
 * @brief 対数正規分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param mu 対数平均
 * @param sigma 対数標準偏差
 * @param engine 乱数エンジン
 * @return 対数正規分布に従う乱数
 * @throws std::invalid_argument sigma <= 0 の場合
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
 * @brief 対数正規分布の乱数生成（デフォルトエンジン使用）
 *
 * @param mu 対数平均（デフォルト: 0.0）
 * @param sigma 対数標準偏差（デフォルト: 1.0）
 * @return 対数正規分布に従う乱数
 */
inline double lognormal_rand(double mu = 0.0, double sigma = 1.0)
{
    return lognormal_rand(mu, sigma, get_random_engine());
}

// ============================================================================
// Weibull Distribution
// ============================================================================

/**
 * @brief ワイブル分布の確率密度関数 (PDF)
 *
 * f(x) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)
 *
 * @param x 確率変数の値
 * @param shape 形状パラメータ k
 * @param scale 尺度パラメータ λ（デフォルト: 1.0）
 * @return 確率密度
 * @throws std::invalid_argument shape <= 0 または scale <= 0 の場合
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
        if (shape < 1.0) {
            return std::numeric_limits<double>::infinity();
        } else if (shape == 1.0) {
            return 1.0 / scale;
        } else {
            return 0.0;
        }
    }
    double z = x / scale;
    return (shape / scale) * std::pow(z, shape - 1.0) * std::exp(-std::pow(z, shape));
}

/**
 * @brief ワイブル分布の累積分布関数 (CDF)
 *
 * F(x) = 1 - exp(-(x/λ)^k)
 *
 * @param x 確率変数の値
 * @param shape 形状パラメータ k
 * @param scale 尺度パラメータ λ（デフォルト: 1.0）
 * @return 累積確率
 * @throws std::invalid_argument shape <= 0 または scale <= 0 の場合
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
 * @brief ワイブル分布の分位点関数
 *
 * Q(p) = λ * (-ln(1 - p))^(1/k)
 *
 * @param p 確率 (0 <= p <= 1)
 * @param shape 形状パラメータ k
 * @param scale 尺度パラメータ λ（デフォルト: 1.0）
 * @return 分位点
 * @throws std::invalid_argument shape <= 0, scale <= 0, または p が [0, 1] の範囲外の場合
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
 * @brief ワイブル分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param shape 形状パラメータ k
 * @param scale 尺度パラメータ λ
 * @param engine 乱数エンジン
 * @return ワイブル分布に従う乱数
 * @throws std::invalid_argument shape <= 0 または scale <= 0 の場合
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
 * @brief ワイブル分布の乱数生成（デフォルトエンジン使用）
 *
 * @param shape 形状パラメータ k
 * @param scale 尺度パラメータ λ（デフォルト: 1.0）
 * @return ワイブル分布に従う乱数
 */
inline double weibull_rand(double shape, double scale = 1.0)
{
    return weibull_rand(shape, scale, get_random_engine());
}

// ============================================================================
// Studentized Range Distribution（スチューデント化された範囲分布）
// ============================================================================

/**
 * @brief スチューデント化された範囲分布のCDF
 *
 * k群、自由度dfのスチューデント化された範囲分布においてP(Q <= q)を計算します。
 *
 * Copenhaver & Holland (1988) アルゴリズムによるガウス・ルジャンドル求積法を使用。
 * Rの ptukey() と同等の実装です。
 *
 * @param q 分位数値
 * @param k 群の数（2以上）
 * @param df 自由度（正の値）
 * @return 確率 P(Q <= q)
 *
 * @throws std::invalid_argument k < 2 または df <= 0 の場合
 */
inline double studentized_range_cdf(double q, double k, double df)
{
    if (k < 2.0) {
        throw std::invalid_argument("statcpp::studentized_range_cdf: k must be >= 2");
    }
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::studentized_range_cdf: df must be positive");
    }
    if (q <= 0.0) return 0.0;

    // k=2 特殊ケース: t分布による正確な計算
    if (k == 2.0) {
        double result = 2.0 * t_cdf(q / sqrt_2, df) - 1.0;
        return std::max(0.0, std::min(1.0, result));
    }

    // 12点ガウス・ルジャンドル半ノード・半重み（内部積分用）
    static constexpr int ihalf = 6;
    static constexpr double xleg[ihalf] = {
        0.981560634246719250690549090149,
        0.904117256370474856678465866119,
        0.769902674194304687036893833213,
        0.587317954286617447296702418941,
        0.367831498998180193752691536644,
        0.125233408511468915472441369464
    };
    static constexpr double aleg[ihalf] = {
        0.047175336386511827194615961485,
        0.106939325995318430960254718194,
        0.160078328543346226334652529543,
        0.203167426723065921749064455810,
        0.233492536538354808760849898925,
        0.249147045813402785000562436043
    };

    // 内部関数: 範囲確率（Hartley形式）
    // Rの wprob() の忠実な移植
    auto wprob = [&](double w, double cc) -> double {
        static constexpr double C1 = -30.0;
        static constexpr double C2 = -50.0;
        static constexpr double C3 = 60.0;
        static constexpr double bb = 8.0;
        static constexpr double wlar = 3.0;
        static constexpr double wincr1 = 2.0;
        static constexpr double wincr2 = 3.0;
        static constexpr double M_1_SQRT_2PI = 0.398942280401432677939946059934;

        double qsqz = w * 0.5;
        if (qsqz >= bb) return 1.0;

        // 第1項: [2*Phi(w/2) - 1]^cc
        double pr_w = 2.0 * norm_cdf(qsqz) - 1.0;
        if (pr_w >= std::exp(C2 / cc))
            pr_w = std::pow(pr_w, cc);
        else
            pr_w = 0.0;

        // 第2積分の区間分割
        double wincr = (w > wlar) ? wincr1 : wincr2;

        // w/2 から 8 までガウス・ルジャンドル求積法で積分
        double blb = qsqz;
        double binc = (bb - qsqz) / wincr;
        double bub = blb + binc;
        double einsum = 0.0;
        double cc1 = cc - 1.0;

        for (int wi = 0; wi < static_cast<int>(wincr); ++wi) {
            double elsum = 0.0;
            double a = 0.5 * (bub + blb);
            double b = 0.5 * (bub - blb);

            for (int jj = 1; jj <= 12; ++jj) {
                double xx;
                int j;
                if (ihalf < jj) {
                    j = 12 - jj;
                    xx = xleg[j];
                } else {
                    j = jj - 1;
                    xx = -xleg[j];
                }
                double ac = a + b * xx;

                double qexpo = ac * ac;
                if (qexpo > C3) break;

                double pplus = 2.0 * norm_cdf(ac);
                double pminus = 2.0 * norm_cdf(ac - w);
                double rinsum = (pplus * 0.5) - (pminus * 0.5);

                if (rinsum >= std::exp(C1 / cc1)) {
                    rinsum = aleg[jj <= ihalf ? jj - 1 : 12 - jj]
                             * std::exp(-0.5 * qexpo) * std::pow(rinsum, cc1);
                    elsum += rinsum;
                }
            }
            elsum *= (2.0 * b) * cc * M_1_SQRT_2PI;
            einsum += elsum;
            blb = bub;
            bub += binc;
        }

        pr_w += einsum;
        if (pr_w <= std::exp(C1)) return 0.0;

        pr_w = std::pow(pr_w, 1.0);  // rr=1（Tukey HSD用）
        return std::max(0.0, std::min(1.0, pr_w));
    };

    // 漸近ケース: df > 25000
    if (df > 25000.0) {
        return wprob(q, k);
    }

    // 一般ケース: カイ二乗密度上の積分
    // 16点ガウス・ルジャンドル半ノード・半重み（外部積分用）
    static constexpr int ihalfq = 8;
    static constexpr double xlegq[ihalfq] = {
        0.989400934991649932596154244360,
        0.944575023073232576077988415535,
        0.865631202387831743880467897712,
        0.755404408355003033895101194847,
        0.617876244402643748446671764049,
        0.458016777657227386342419442984,
        0.281603550779258913230460501460,
        0.095012509837637440185319335425
    };
    static constexpr double alegq[ihalfq] = {
        0.027152459411754094851780572456,
        0.062253523938647892862843836994,
        0.095158511682492784809925107602,
        0.124628971255533872052476282192,
        0.149595988816576732081501730547,
        0.169156519395002538189312079030,
        0.182603415044923588866763667969,
        0.189450610455068496285396723208
    };

    // カイ二乗密度の定数を事前計算（R ptukey.c に準拠）
    double f2 = df * 0.5;
    double f2lf = (f2 * std::log(df)) - (df * std::log(2.0)) - lgamma(f2);
    double f21 = f2 - 1.0;
    double ff4 = df * 0.25;

    // 適応的区間幅
    double ulen;
    if (df <= 100.0) ulen = 1.0;
    else if (df <= 800.0) ulen = 0.5;
    else if (df <= 5000.0) ulen = 0.25;
    else ulen = 0.125;

    f2lf += std::log(ulen);

    double ans = 0.0;

    for (int i = 1; i <= 50; ++i) {
        double otsum = 0.0;
        double twa1 = (2.0 * i - 1.0) * ulen;

        for (int jj = 1; jj <= 16; ++jj) {
            double t1;
            int j;
            double t_point;

            if (ihalfq < jj) {
                j = jj - ihalfq - 1;
                t_point = twa1 + xlegq[j] * ulen;
                t1 = f2lf + f21 * std::log(t_point) - t_point * ff4;
            } else {
                j = jj - 1;
                t_point = twa1 - xlegq[j] * ulen;
                t1 = f2lf + f21 * std::log(t_point) - t_point * ff4;
            }

            if (t_point <= 0.0) continue;
            if (t1 >= -30.0) {
                double qsqz;
                if (ihalfq < jj) {
                    qsqz = q * std::sqrt((xlegq[j] * ulen + twa1) * 0.5);
                } else {
                    qsqz = q * std::sqrt((-(xlegq[j] * ulen) + twa1) * 0.5);
                }

                double wprb = wprob(qsqz, k);
                otsum += (wprb * alegq[j]) * std::exp(t1);
            }
        }

        if (static_cast<double>(i) * ulen >= 1.0 && otsum <= 1e-14) {
            break;
        }

        ans += otsum;
    }

    return std::max(0.0, std::min(1.0, ans));
}

/**
 * @brief スチューデント化された範囲分布の分位関数
 *
 * k群、自由度dfのスチューデント化された範囲分布においてP(Q <= q) = pとなるqを計算します。
 *
 * 数値微分（中心差分）によるニュートン・ラフソン反復法を使用します。
 *
 * @param p 確率（[0, 1]の範囲）
 * @param k 群の数（2以上）
 * @param df 自由度（正の値）
 * @return 分位数値
 *
 * @throws std::invalid_argument k < 2、df <= 0、またはpが[0, 1]の範囲外の場合
 *
 * @note Newton-Raphson反復法を使用（許容誤差 eps = 1e-10、最大反復回数 50）。
 *       最大反復回数に達しても収束しない場合、その時点での最良の近似値を返します
 *       （例外は送出されません）。内部点（0 < p < 1）では戻り値は有限値です。
 *       境界値 p = 0 または p = 1 では、分布の定義に従い +/-infinity を
 *       返す場合があります。実用上、一般的な入力範囲では常に収束します。
 */
inline double studentized_range_quantile(double p, double k, double df)
{
    if (k < 2.0) {
        throw std::invalid_argument("statcpp::studentized_range_quantile: k must be >= 2");
    }
    if (df <= 0.0) {
        throw std::invalid_argument("statcpp::studentized_range_quantile: df must be positive");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::studentized_range_quantile: p must be in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    // 初期推定値: 正規分位数に基づくヒューリスティック
    double q0 = norm_quantile(1.0 - (1.0 - p) * 0.5);
    double x = q0 * (1.0 + 0.2 * (k - 2.0) / std::sqrt(df));
    if (!std::isfinite(x) || x < 1.0) x = 1.0;

    // ニュートン・ラフソン反復法
    const double eps = 1e-10;
    const int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        if (!std::isfinite(x) || x <= 0.0) {
            x = 1.0;
        }
        double f = studentized_range_cdf(x, k, df) - p;
        if (std::abs(f) < eps) return x;

        // 数値微分（中心差分）
        double dx = std::max(1e-6, x * 1e-6);
        double fprime = (studentized_range_cdf(x + dx, k, df)
                       - studentized_range_cdf(x - dx, k, df)) / (2.0 * dx);
        if (fprime <= 0.0) break;

        double x_new = x - f / fprime;
        if (x_new <= 0.0) x_new = x * 0.5;

        if (std::abs(x_new - x) < eps * (1.0 + std::abs(x))) return x_new;
        x = x_new;
    }

    if (!std::isfinite(x) || x <= 0.0) {
        x = 1.0;
    }
    return x;
}

} // namespace statcpp
