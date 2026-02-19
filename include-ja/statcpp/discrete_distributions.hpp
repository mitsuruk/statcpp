/**
 * @file discrete_distributions.hpp
 * @brief 離散確率分布関数
 *
 * このファイルは離散確率分布（二項分布、ポアソン分布、幾何分布など）の
 * 確率質量関数（PMF）、累積分布関数（CDF）、分位関数、乱数生成を提供します。
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
 * @brief 対数階乗の計算
 *
 * @param n 非負整数
 * @return log(n!)の値
 */
inline double log_factorial(std::uint64_t n)
{
    if (n <= 1) return 0.0;
    return lgamma(static_cast<double>(n + 1));
}

/**
 * @brief 対数二項係数の計算
 *
 * log(C(n, k)) = log(n! / (k! * (n-k)!)) を計算します。
 *
 * @param n 全体の要素数
 * @param k 選択する要素数
 * @return log(C(n, k))の値
 */
inline double log_binomial_coef(std::uint64_t n, std::uint64_t k)
{
    if (k > n) return -std::numeric_limits<double>::infinity();
    if (k == 0 || k == n) return 0.0;
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k);
}

/**
 * @brief 二項係数の計算
 *
 * C(n, k) = n! / (k! * (n-k)!) を計算します。
 *
 * @param n 全体の要素数
 * @param k 選択する要素数
 * @return C(n, k)の値
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
 * @brief 二項分布の確率質量関数（PMF）
 *
 * P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
 *
 * @param k 成功回数
 * @param n 試行回数
 * @param p 各試行の成功確率
 * @return 確率 P(X = k)
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief 二項分布の累積分布関数（CDF）
 *
 * P(X <= k) を計算します。
 *
 * @param k 成功回数の上限
 * @param n 試行回数
 * @param p 各試行の成功確率
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief 二項分布の分位関数
 *
 * P(X <= k) >= prob となる最小のkを返します。
 *
 * @param prob 確率値
 * @param n 試行回数
 * @param p 各試行の成功確率
 * @return 分位点
 * @throw std::invalid_argument probまたはpが不正な範囲の場合
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
 * @brief 二項分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param n 試行回数
 * @param p 各試行の成功確率
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief 二項分布の乱数生成（デフォルトエンジン使用）
 *
 * @param n 試行回数
 * @param p 各試行の成功確率
 * @return 生成された乱数
 */
inline std::uint64_t binomial_rand(std::uint64_t n, double p)
{
    return binomial_rand(n, p, get_random_engine());
}

// ============================================================================
// Poisson Distribution
// ============================================================================

/**
 * @brief ポアソン分布の確率質量関数（PMF）
 *
 * P(X = k) = (λ^k * e^(-λ)) / k!
 *
 * @param k 事象の発生回数
 * @param lambda 平均発生率（λ > 0）
 * @return 確率 P(X = k)
 * @throw std::invalid_argument lambdaが負の場合
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
 * @brief ポアソン分布の累積分布関数（CDF）
 *
 * P(X <= k) を計算します。
 *
 * @param k 事象の発生回数の上限
 * @param lambda 平均発生率
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument lambdaが負の場合
 */
inline double poisson_cdf(std::uint64_t k, double lambda)
{
    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::poisson_cdf: lambda must be non-negative");
    }
    if (lambda == 0.0) return 1.0;

    // P(X <= k) = Q(k+1, λ) = 1 - P(k+1, λ) (upper regularized incomplete gamma)
    return gammainc_upper(static_cast<double>(k + 1), lambda);
}

/**
 * @brief ポアソン分布の分位関数
 *
 * @param p 確率値
 * @param lambda 平均発生率
 * @return 分位点
 * @throw std::invalid_argument lambdaが負、またはpが不正な範囲の場合
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
 * @brief ポアソン分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param lambda 平均発生率
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument lambdaが負の場合
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
 * @brief ポアソン分布の乱数生成（デフォルトエンジン使用）
 *
 * @param lambda 平均発生率
 * @return 生成された乱数
 */
inline std::uint64_t poisson_rand(double lambda)
{
    return poisson_rand(lambda, get_random_engine());
}

// ============================================================================
// Geometric Distribution
// ============================================================================

/**
 * @brief 幾何分布の確率質量関数（PMF）
 *
 * P(X = k) = (1-p)^k * p
 * X = 最初の成功までの失敗回数（サポート: k = 0, 1, 2, ...）
 *
 * @param k 失敗回数
 * @param p 各試行の成功確率
 * @return 確率 P(X = k)
 * @throw std::invalid_argument pが(0, 1]の範囲外の場合
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
 * @brief 幾何分布の累積分布関数（CDF）
 *
 * P(X <= k) = 1 - (1-p)^(k+1)
 *
 * @param k 失敗回数の上限
 * @param p 各試行の成功確率
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument pが(0, 1]の範囲外の場合
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
 * @brief 幾何分布の分位関数
 *
 * @param prob 確率値
 * @param p 各試行の成功確率
 * @return 分位点
 * @throw std::invalid_argument pまたはprobが不正な範囲の場合
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
 * @brief 幾何分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param p 各試行の成功確率
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument pが(0, 1]の範囲外の場合
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
 * @brief 幾何分布の乱数生成（デフォルトエンジン使用）
 *
 * @param p 各試行の成功確率
 * @return 生成された乱数
 */
inline std::uint64_t geometric_rand(double p)
{
    return geometric_rand(p, get_random_engine());
}

// ============================================================================
// Negative Binomial Distribution
// ============================================================================

/**
 * @brief 負の二項分布の確率質量関数（PMF）
 *
 * P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 * X = r回成功するまでの失敗回数（サポート: k = 0, 1, 2, ...）
 *
 * @param k 失敗回数
 * @param r 成功回数（分散パラメータ、> 0、非整数も可）
 * @param p 各試行の成功確率
 * @return 確率 P(X = k)
 * @throw std::invalid_argument rが非正、またはpが(0, 1]の範囲外の場合
 *
 * @note パラメータ化の違いについて：
 *       負の二項分布には複数のパラメータ化が存在します。この実装は「失敗回数」を
 *       モデル化する形式（r回成功するまでの失敗回数 X）を使用しています。
 *
 *       - R の dnbinom(x, size, prob): size=r, prob=p で同じ（失敗回数）
 *       - Python scipy.stats.nbinom(k, n, p): n=r, p=p で同じ（失敗回数）
 *       - 一部の教科書: 「r回成功するまでの試行回数」= X + r
 *
 *       また、平均と分散でパラメータ化する「mean-dispersion」形式もあります：
 *       - 平均 μ = r(1-p)/p
 *       - 分散 σ² = r(1-p)/p² = μ + μ²/r
 *       この形式は過分散データのモデリング（GLM等）でよく使われます。
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
 * @brief 負の二項分布の累積分布関数（CDF）
 *
 * P(X <= k) = I_p(r, k+1) (incomplete beta関数を使用)
 *
 * @param k 失敗回数の上限
 * @param r 成功回数
 * @param p 各試行の成功確率
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument rが非正、またはpが(0, 1]の範囲外の場合
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
 * @brief 負の二項分布の分位関数
 *
 * @param prob 確率値
 * @param r 成功回数
 * @param p 各試行の成功確率
 * @return 分位点
 * @throw std::invalid_argument パラメータが不正な範囲の場合
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
 * @brief 負の二項分布の乱数生成
 *
 * Poisson-Gamma混合として生成します。
 *
 * @tparam Engine 乱数エンジン型
 * @param r 成功回数
 * @param p 各試行の成功確率
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument パラメータが不正な範囲の場合
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
 * @brief 負の二項分布の乱数生成（デフォルトエンジン使用）
 *
 * @param r 成功回数
 * @param p 各試行の成功確率
 * @return 生成された乱数
 */
inline std::uint64_t nbinom_rand(double r, double p)
{
    return nbinom_rand(r, p, get_random_engine());
}

// ============================================================================
// Hypergeometric Distribution
// ============================================================================

/**
 * @brief 超幾何分布の確率質量関数（PMF）
 *
 * P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
 *
 * @param k 抽出した成功の数
 * @param N 母集団サイズ
 * @param K 成功状態の数
 * @param n 抽出数
 * @return 確率 P(X = k)
 * @throw std::invalid_argument パラメータが不正な場合（K > N または n > N）
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
 * @brief 超幾何分布の累積分布関数（CDF）
 *
 * P(X <= k) を計算します。
 *
 * @param k 抽出した成功数の上限
 * @param N 母集団サイズ
 * @param K 成功状態の数
 * @param n 抽出数
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument パラメータが不正な場合
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
 * @brief 超幾何分布の分位関数
 *
 * @param p 確率値
 * @param N 母集団サイズ
 * @param K 成功状態の数
 * @param n 抽出数
 * @return 分位点
 * @throw std::invalid_argument パラメータが不正な範囲の場合
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
 * @brief 超幾何分布の乱数生成
 *
 * 直接サンプリング法を使用します（中程度のnに適しています）。
 *
 * @tparam Engine 乱数エンジン型
 * @param N 母集団サイズ
 * @param K 成功状態の数
 * @param n 抽出数
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument パラメータが不正な場合
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
 * @brief 超幾何分布の乱数生成（デフォルトエンジン使用）
 *
 * @param N 母集団サイズ
 * @param K 成功状態の数
 * @param n 抽出数
 * @return 生成された乱数
 */
inline std::uint64_t hypergeom_rand(std::uint64_t N, std::uint64_t K, std::uint64_t n)
{
    return hypergeom_rand(N, K, n, get_random_engine());
}

// ============================================================================
// Bernoulli Distribution
// ============================================================================

/**
 * @brief ベルヌーイ分布の確率質量関数（PMF）
 *
 * P(X = k) = p^k * (1-p)^(1-k) for k ∈ {0, 1}
 *
 * @param k 結果（0または1）
 * @param p 成功確率
 * @return 確率 P(X = k)
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief ベルヌーイ分布の累積分布関数（CDF）
 *
 * @param k 結果の上限
 * @param p 成功確率
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief ベルヌーイ分布の分位関数
 *
 * @param prob 確率値
 * @param p 成功確率
 * @return 分位点（0または1）
 * @throw std::invalid_argument probまたはpが不正な範囲の場合
 */
inline std::uint64_t bernoulli_quantile(double prob, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_quantile: p must be in [0, 1]");
    }
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("statcpp::bernoulli_quantile: prob must be in [0, 1]");
    }
    return (prob <= 1.0 - p) ? 0 : 1;
}

/**
 * @brief ベルヌーイ分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param p 成功確率
 * @param engine 乱数エンジン
 * @return 生成された乱数（0または1）
 * @throw std::invalid_argument pが[0, 1]の範囲外の場合
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
 * @brief ベルヌーイ分布の乱数生成（デフォルトエンジン使用）
 *
 * @param p 成功確率
 * @return 生成された乱数（0または1）
 */
inline std::uint64_t bernoulli_rand(double p)
{
    return bernoulli_rand(p, get_random_engine());
}

// ============================================================================
// Discrete Uniform Distribution
// ============================================================================

/**
 * @brief 離散一様分布の確率質量関数（PMF）
 *
 * P(X = k) = 1 / (b - a + 1) for a <= k <= b
 *
 * @param k 値
 * @param a 範囲の下限
 * @param b 範囲の上限
 * @return 確率 P(X = k)
 * @throw std::invalid_argument a > bの場合
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
 * @brief 離散一様分布の累積分布関数（CDF）
 *
 * @param k 値の上限
 * @param a 範囲の下限
 * @param b 範囲の上限
 * @return 累積確率 P(X <= k)
 * @throw std::invalid_argument a > bの場合
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
 * @brief 離散一様分布の分位関数
 *
 * @param p 確率値
 * @param a 範囲の下限
 * @param b 範囲の上限
 * @return 分位点
 * @throw std::invalid_argument a > bまたはpが不正な範囲の場合
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
 * @brief 離散一様分布の乱数生成
 *
 * @tparam Engine 乱数エンジン型
 * @param a 範囲の下限
 * @param b 範囲の上限
 * @param engine 乱数エンジン
 * @return 生成された乱数
 * @throw std::invalid_argument a > bの場合
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
 * @brief 離散一様分布の乱数生成（デフォルトエンジン使用）
 *
 * @param a 範囲の下限
 * @param b 範囲の上限
 * @return 生成された乱数
 */
inline std::int64_t discrete_uniform_rand(std::int64_t a, std::int64_t b)
{
    return discrete_uniform_rand(a, b, get_random_engine());
}

} // namespace statcpp
