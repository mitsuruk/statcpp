/**
 * @file estimation.hpp
 * @brief 推定と信頼区間の関数
 *
 * 標準誤差の計算、信頼区間の推定、および誤差マージンとサンプルサイズの計算を提供します。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Standard Error
// ============================================================================

/**
 * @brief 平均の標準誤差を計算
 *
 * 標準誤差: SE = s / √n
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 標準誤差
 * @throws std::invalid_argument 要素数が2未満の場合
 */
template <typename Iterator>
double standard_error(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::standard_error: need at least 2 elements");
    }
    double s = statcpp::sample_stddev(first, last);
    return s / std::sqrt(static_cast<double>(n));
}

/**
 * @brief 平均の標準誤差を計算（射影版）
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 標準誤差
 * @throws std::invalid_argument 要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double standard_error(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::standard_error: need at least 2 elements");
    }
    double s = statcpp::sample_stddev(first, last, proj);
    return s / std::sqrt(static_cast<double>(n));
}

/**
 * @brief 平均の標準誤差を計算（事前計算済み標準偏差を使用）
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_stddev 事前計算済み標準偏差
 * @return 標準誤差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double standard_error(Iterator first, Iterator last, double precomputed_stddev)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::standard_error: empty range");
    }
    return precomputed_stddev / std::sqrt(static_cast<double>(n));
}

// ============================================================================
// Confidence Interval Result Structure
// ============================================================================

/**
 * @brief 信頼区間の結果を格納する構造体
 */
struct confidence_interval {
    double lower;              ///< 信頼区間下限
    double upper;              ///< 信頼区間上限
    double point_estimate;     ///< 点推定値
    double confidence_level;   ///< 信頼水準
};

// ============================================================================
// Confidence Interval for Mean (using t-distribution)
// ============================================================================

/**
 * @brief 平均の信頼区間を計算（t分布ベース）
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または要素数が2未満の場合
 */
template <typename Iterator>
confidence_interval ci_mean(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_mean: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double se = statcpp::standard_error(first, last);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

/**
 * @brief 平均の信頼区間を計算（射影版）
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param confidence 信頼水準
 * @param proj 射影関数
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
confidence_interval ci_mean(Iterator first, Iterator last, double confidence, Projection proj)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_mean: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last, proj);
    double se = statcpp::standard_error(first, last, proj);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

// ============================================================================
// Confidence Interval for Mean (using z-distribution, known variance)
// ============================================================================

/**
 * @brief 平均の信頼区間を計算（z分布ベース、既知の分散）
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param sigma 既知の母標準偏差
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、sigmaが正でない、または空の範囲の場合
 */
template <typename Iterator>
confidence_interval ci_mean_z(Iterator first, Iterator last, double sigma, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_z: confidence must be in (0, 1)");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::ci_mean_z: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::ci_mean_z: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    double se = sigma / std::sqrt(static_cast<double>(n));

    double alpha = 1.0 - confidence;
    double z_crit = norm_quantile(1.0 - alpha / 2.0);

    double margin = z_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

// ============================================================================
// Confidence Interval for Proportion
// ============================================================================

/**
 * @brief 比率の信頼区間を計算（Wald法）
 *
 * @param successes 成功回数
 * @param trials 試行回数
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、試行回数が0、または成功回数が試行回数を超える場合
 */
inline confidence_interval ci_proportion(std::size_t successes, std::size_t trials, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion: confidence must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::ci_proportion: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::ci_proportion: successes cannot exceed trials");
    }

    double p_hat = static_cast<double>(successes) / static_cast<double>(trials);
    double se = std::sqrt(p_hat * (1.0 - p_hat) / static_cast<double>(trials));

    double alpha = 1.0 - confidence;
    double z_crit = norm_quantile(1.0 - alpha / 2.0);

    double margin = z_crit * se;
    double lower = std::max(0.0, p_hat - margin);
    double upper = std::min(1.0, p_hat + margin);

    return {lower, upper, p_hat, confidence};
}

/**
 * @brief 比率の信頼区間を計算（Wilson法、推奨）
 *
 * Wilson法はWald法よりも優れた性質を持ち、特に小サンプルや極端な比率で推奨されます。
 *
 * @param successes 成功回数
 * @param trials 試行回数
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、試行回数が0、または成功回数が試行回数を超える場合
 */
inline confidence_interval ci_proportion_wilson(std::size_t successes, std::size_t trials, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: confidence must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: successes cannot exceed trials");
    }

    double n = static_cast<double>(trials);
    double p_hat = static_cast<double>(successes) / n;

    double alpha = 1.0 - confidence;
    double z = norm_quantile(1.0 - alpha / 2.0);
    double z2 = z * z;

    double denom = 1.0 + z2 / n;
    double center = (p_hat + z2 / (2.0 * n)) / denom;
    double margin = z * std::sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom;

    return {center - margin, center + margin, p_hat, confidence};
}

// ============================================================================
// Confidence Interval for Variance (Chi-square based)
// ============================================================================

/**
 * @brief 分散の信頼区間を計算（χ²分布ベース）
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または要素数が2未満の場合
 */
template <typename Iterator>
confidence_interval ci_variance(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_variance: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_variance: need at least 2 elements");
    }

    double var = statcpp::sample_variance(first, last);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double chi2_lower = chisq_quantile(alpha / 2.0, df);
    double chi2_upper = chisq_quantile(1.0 - alpha / 2.0, df);

    double lower = df * var / chi2_upper;
    double upper = df * var / chi2_lower;

    return {lower, upper, var, confidence};
}

// ============================================================================
// Confidence Interval for Difference of Means
// ============================================================================

/**
 * @brief 2標本平均の差の信頼区間を計算（独立標本、プール分散）
 *
 * 等分散を仮定した2標本平均の差の信頼区間を計算します。
 *
 * @tparam Iterator1 第1イテレータ型
 * @tparam Iterator2 第2イテレータ型
 * @param first1 第1データ範囲の開始イテレータ
 * @param last1 第1データ範囲の終了イテレータ
 * @param first2 第2データ範囲の開始イテレータ
 * @param last2 第2データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または各標本が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff(Iterator1 first1, Iterator1 last1,
                                  Iterator2 first2, Iterator2 last2,
                                  double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    // Pooled variance
    double df = static_cast<double>(n1 + n2 - 2);
    double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df;
    double se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

/**
 * @brief 2標本平均の差の信頼区間を計算（Welch法、等分散を仮定しない）
 *
 * 等分散を仮定しない2標本平均の差の信頼区間を計算します（Welch-Satterthwaite法）。
 *
 * @tparam Iterator1 第1イテレータ型
 * @tparam Iterator2 第2イテレータ型
 * @param first1 第1データ範囲の開始イテレータ
 * @param last1 第1データ範囲の終了イテレータ
 * @param first2 第2データ範囲の開始イテレータ
 * @param last2 第2データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、各標本が2未満、または分散が両方とも0の場合
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff_welch(Iterator1 first1, Iterator1 last1,
                                        Iterator2 first2, Iterator2 last2,
                                        double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    double se1 = var1 / n1;
    double se2 = var2 / n2;
    double se = std::sqrt(se1 + se2);

    // Welch–Satterthwaite approximation for degrees of freedom
    double num = (se1 + se2) * (se1 + se2);
    double denom = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1);

    // 分母が0になる場合の保護（両方の分散が0の場合）
    if (denom == 0.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: cannot compute degrees of freedom with zero variances");
    }

    double df = num / denom;

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

// ============================================================================
// Margin of Error
// ============================================================================

/**
 * @brief 平均の誤差マージンを計算
 *
 * MoE = t_{α/2, df} × SE
 *
 * @tparam Iterator イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 誤差マージン
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または要素数が2未満の場合
 */
template <typename Iterator>
double margin_of_error_mean(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: need at least 2 elements");
    }

    double se = statcpp::standard_error(first, last);
    double df = static_cast<double>(n - 1);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    return t_crit * se;
}

/**
 * @brief 平均の誤差マージンを計算（射影版）
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param confidence 信頼水準
 * @param proj 射影関数
 * @return 誤差マージン
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double margin_of_error_mean(Iterator first, Iterator last, double confidence, Projection proj)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: need at least 2 elements");
    }

    double se = statcpp::standard_error(first, last, proj);
    double df = static_cast<double>(n - 1);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    return t_crit * se;
}

/**
 * @brief 比率の誤差マージンを計算
 *
 * MoE = z_{α/2} × √(p(1-p)/n)
 *
 * @param successes 成功回数
 * @param n サンプルサイズ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 誤差マージン
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、nが0、または成功回数がnを超える場合
 */
inline double margin_of_error_proportion(std::size_t successes, std::size_t n, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: confidence must be in (0, 1)");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: n must be positive");
    }
    if (successes > n) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: successes cannot exceed n");
    }

    double p = static_cast<double>(successes) / static_cast<double>(n);
    double se = std::sqrt(p * (1.0 - p) / static_cast<double>(n));

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    return z_crit * se;
}

/**
 * @brief 最悪ケースの比率誤差マージンを計算
 *
 * p=0.5 で最大となる: MoE = z_{α/2} × 0.5/√n
 *
 * @param n サンプルサイズ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 誤差マージン
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、またはnが0の場合
 */
inline double margin_of_error_proportion_worst_case(std::size_t n, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion_worst_case: confidence must be in (0, 1)");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion_worst_case: n must be positive");
    }

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    return z_crit * 0.5 / std::sqrt(static_cast<double>(n));
}

// ============================================================================
// Sample Size Calculation (Margin-of-Error based)
// ============================================================================

/**
 * @brief 比率推定のためのサンプルサイズを計算
 *
 * 指定した誤差マージンを達成するために必要なサンプルサイズを計算します。
 * n = (z_{α/2} / MoE)² × p(1-p)
 *
 * @param margin_of_error 目標とする誤差マージン
 * @param confidence_level 信頼水準（デフォルト: 0.95）
 * @param p_estimate 事前の比率推定値（デフォルト: 0.5で最も保守的な見積もり）
 * @return 必要なサンプルサイズ
 * @throws std::invalid_argument パラメータが有効な範囲外の場合
 */
inline std::size_t sample_size_for_moe_proportion(double margin_of_error,
                                                   double confidence_level = 0.95,
                                                   double p_estimate = 0.5)
{
    if (margin_of_error <= 0.0 || margin_of_error >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: margin_of_error must be in (0, 1)");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: confidence_level must be in (0, 1)");
    }
    if (p_estimate <= 0.0 || p_estimate >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: p_estimate must be in (0, 1)");
    }

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    // n = (z / MoE)² × p(1-p)
    double n_exact = std::pow(z / margin_of_error, 2.0) * p_estimate * (1.0 - p_estimate);

    // 切り上げて整数に変換
    return static_cast<std::size_t>(std::ceil(n_exact));
}

/**
 * @brief 平均推定のためのサンプルサイズを計算（母標準偏差が既知の場合）
 *
 * 指定した誤差マージンを達成するために必要なサンプルサイズを計算します。
 * n = (z_{α/2} × σ / MoE)²
 *
 * @param margin_of_error 目標とする誤差マージン
 * @param sigma 母標準偏差（既知または推定値）
 * @param confidence_level 信頼水準（デフォルト: 0.95）
 * @return 必要なサンプルサイズ
 * @throws std::invalid_argument パラメータが有効な範囲外の場合
 */
inline std::size_t sample_size_for_moe_mean(double margin_of_error,
                                             double sigma,
                                             double confidence_level = 0.95)
{
    if (margin_of_error <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: margin_of_error must be positive");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: sigma must be positive");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: confidence_level must be in (0, 1)");
    }

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    // n = (z × σ / MoE)²
    double n_exact = std::pow(z * sigma / margin_of_error, 2.0);

    // 切り上げて整数に変換
    return static_cast<std::size_t>(std::ceil(n_exact));
}

// ============================================================================
// Two-Sample Mean Difference Confidence Interval
// ============================================================================

/**
 * @brief 2標本平均差の信頼区間を計算（等分散を仮定）
 *
 * @tparam Iterator1 第1イテレータ型
 * @tparam Iterator2 第2イテレータ型
 * @param first1 第1データ範囲の開始イテレータ
 * @param last1 第1データ範囲の終了イテレータ
 * @param first2 第2データ範囲の開始イテレータ
 * @param last2 第2データ範囲の終了イテレータ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument 信頼水準が (0, 1) の範囲外、または各標本が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff_pooled(Iterator1 first1, Iterator1 last1,
                                        Iterator2 first2, Iterator2 last2,
                                        double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_pooled: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff_pooled: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    // Pooled variance
    double pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    double se = std::sqrt(pooled_var * (1.0 / n1 + 1.0 / n2));

    double df = static_cast<double>(n1 + n2 - 2);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

// ============================================================================
// Two-Sample Proportion Difference Confidence Interval
// ============================================================================

/**
 * @brief 2標本比率差の信頼区間を計算
 *
 * @param successes1 第1標本の成功回数
 * @param n1 第1標本のサイズ
 * @param successes2 第2標本の成功回数
 * @param n2 第2標本のサイズ
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @return 信頼区間
 * @throws std::invalid_argument パラメータが有効な範囲外の場合
 */
inline confidence_interval ci_proportion_diff(std::size_t successes1, std::size_t n1,
                                               std::size_t successes2, std::size_t n2,
                                               double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: confidence must be in (0, 1)");
    }
    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: sample sizes must be positive");
    }
    if (successes1 > n1 || successes2 > n2) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: successes cannot exceed sample size");
    }

    double p1 = static_cast<double>(successes1) / static_cast<double>(n1);
    double p2 = static_cast<double>(successes2) / static_cast<double>(n2);

    double diff = p1 - p2;

    // Standard error for difference of proportions
    double se1 = p1 * (1.0 - p1) / static_cast<double>(n1);
    double se2 = p2 * (1.0 - p2) / static_cast<double>(n2);
    double se = std::sqrt(se1 + se2);

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    double margin = z_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

} // namespace statcpp
