/**
 * @file parametric_tests.hpp
 * @brief パラメトリック検定関数
 *
 * t検定、z検定、F検定、カイ二乗検定などのパラメトリック統計検定を提供します。
 * 多重検定補正（Bonferroni、Benjamini-Hochberg、Holm）も含みます。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Test Result Structure
// ============================================================================

/**
 * @brief 対立仮説の種類を表す列挙型
 *
 * 統計検定における対立仮説の方向を指定します。
 */
enum class alternative_hypothesis {
    two_sided,  ///< 両側検定
    less,       ///< 片側検定（小さい方向）
    greater     ///< 片側検定（大きい方向）
};

/**
 * @brief 統計検定の結果を格納する構造体
 *
 * 検定統計量、p値、自由度、対立仮説の情報を保持します。
 */
struct test_result {
    double statistic;                ///< 検定統計量
    double p_value;                  ///< p値
    double df;                       ///< 自由度
    alternative_hypothesis alternative;  ///< 対立仮説の種類
    double df2 = std::numeric_limits<double>::quiet_NaN();  ///< 第2自由度（F検定で使用）
};

// ============================================================================
// Z-Test for Mean (known variance)
// ============================================================================

/**
 * @brief 1標本z検定（既知の分散）
 *
 * 母分散が既知の場合に、標本平均が特定の値と等しいかどうかを検定します。
 *
 * @tparam Iterator 入力イテレータ型
 * @param first 標本データの開始イテレータ
 * @param last 標本データの終了イテレータ
 * @param mu0 帰無仮説の母平均
 * @param sigma 既知の母標準偏差
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（z統計量、p値、自由度=無限大）
 * @throws std::invalid_argument sigmaが正でない場合、または空の範囲の場合
 */
template <typename Iterator>
test_result z_test(Iterator first, Iterator last, double mu0, double sigma,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::z_test: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::z_test: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    double se = sigma / std::sqrt(static_cast<double>(n));
    double z = (mean_val - mu0) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

// ============================================================================
// Z-Test for Proportion
// ============================================================================

/**
 * @brief 1標本比率z検定
 *
 * 標本比率が特定の母比率と等しいかどうかを検定します。
 *
 * @param successes 成功回数
 * @param trials 試行回数
 * @param p0 帰無仮説の母比率
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（z統計量、p値、自由度=無限大）
 * @throws std::invalid_argument p0が(0,1)の範囲外、trialsが0、またはsuccessesがtrialsを超える場合
 */
inline test_result z_test_proportion(std::size_t successes, std::size_t trials, double p0,
                                     alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (p0 <= 0.0 || p0 >= 1.0) {
        throw std::invalid_argument("statcpp::z_test_proportion: p0 must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::z_test_proportion: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::z_test_proportion: successes cannot exceed trials");
    }

    double n = static_cast<double>(trials);
    double p_hat = static_cast<double>(successes) / n;
    double se = std::sqrt(p0 * (1.0 - p0) / n);
    double z = (p_hat - p0) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

/**
 * @brief 2標本比率z検定
 *
 * 2つの標本の比率が等しいかどうかを検定します（プール比率を使用）。
 *
 * @param successes1 第1標本の成功回数
 * @param trials1 第1標本の試行回数
 * @param successes2 第2標本の成功回数
 * @param trials2 第2標本の試行回数
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（z統計量、p値、自由度=無限大）
 * @throws std::invalid_argument trialsが0、またはsuccessesがtrialsを超える場合
 */
inline test_result z_test_proportion_two_sample(std::size_t successes1, std::size_t trials1,
                                                 std::size_t successes2, std::size_t trials2,
                                                 alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (trials1 == 0 || trials2 == 0) {
        throw std::invalid_argument("statcpp::z_test_proportion_two_sample: trials must be positive");
    }
    if (successes1 > trials1 || successes2 > trials2) {
        throw std::invalid_argument("statcpp::z_test_proportion_two_sample: successes cannot exceed trials");
    }

    double n1 = static_cast<double>(trials1);
    double n2 = static_cast<double>(trials2);
    double p1 = static_cast<double>(successes1) / n1;
    double p2 = static_cast<double>(successes2) / n2;

    // Pooled proportion
    double p_pooled = static_cast<double>(successes1 + successes2) / (n1 + n2);
    double se = std::sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2));

    double z = (p1 - p2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

// ============================================================================
// T-Test for Mean
// ============================================================================

/**
 * @brief 1標本t検定
 *
 * 標本平均が特定の値と等しいかどうかを検定します（母分散未知）。
 *
 * @tparam Iterator 入力イテレータ型
 * @param first 標本データの開始イテレータ
 * @param last 標本データの終了イテレータ
 * @param mu0 帰無仮説の母平均
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（t統計量、p値、自由度）
 * @throws std::invalid_argument 要素数が2未満、または分散が0の場合
 */
template <typename Iterator>
test_result t_test(Iterator first, Iterator last, double mu0,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::t_test: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::t_test: zero variance");
    }

    double se = s / std::sqrt(static_cast<double>(n));
    double t = (mean_val - mu0) / se;
    double df = static_cast<double>(n - 1);

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief 2標本t検定（独立標本、プール分散）
 *
 * 2つの独立した標本の平均が等しいかどうかを検定します。
 * 等分散を仮定し、プール分散を使用します。
 *
 * @tparam Iterator1 第1標本の入力イテレータ型
 * @tparam Iterator2 第2標本の入力イテレータ型
 * @param first1 第1標本の開始イテレータ
 * @param last1 第1標本の終了イテレータ
 * @param first2 第2標本の開始イテレータ
 * @param last2 第2標本の終了イテレータ
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（t統計量、p値、自由度）
 * @throws std::invalid_argument いずれかの標本の要素数が2未満、または分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_two_sample(Iterator1 first1, Iterator1 last1,
                               Iterator2 first2, Iterator2 last2,
                               alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::t_test_two_sample: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    // Pooled variance
    double df = static_cast<double>(n1 + n2 - 2);
    double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df;
    double se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));

    if (se == 0.0) {
        throw std::invalid_argument("statcpp::t_test_two_sample: zero variance");
    }

    double t = (mean1 - mean2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief 2標本t検定（Welch法）
 *
 * 2つの独立した標本の平均が等しいかどうかを検定します。
 * 等分散を仮定せず、Welch-Satterthwaite近似で自由度を計算します。
 *
 * @tparam Iterator1 第1標本の入力イテレータ型
 * @tparam Iterator2 第2標本の入力イテレータ型
 * @param first1 第1標本の開始イテレータ
 * @param last1 第1標本の終了イテレータ
 * @param first2 第2標本の開始イテレータ
 * @param last2 第2標本の終了イテレータ
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（t統計量、p値、Welch近似自由度）
 * @throws std::invalid_argument いずれかの標本の要素数が2未満、または分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_welch(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::t_test_welch: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double se1 = var1 / n1;
    double se2 = var2 / n2;
    double se = std::sqrt(se1 + se2);

    if (se == 0.0) {
        throw std::invalid_argument("statcpp::t_test_welch: zero variance");
    }

    // Welch–Satterthwaite approximation
    double num = (se1 + se2) * (se1 + se2);
    double denom = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1);

    // 分母が0になる場合の保護（両方の分散が0の場合）
    if (denom == 0.0) {
        throw std::invalid_argument("statcpp::t_test_welch: cannot compute degrees of freedom with zero variances");
    }

    double df = num / denom;

    double t = (mean1 - mean2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief 対応のあるt検定
 *
 * 対応のある2つの標本の差の平均が0と等しいかどうかを検定します。
 *
 * @tparam Iterator1 第1標本の入力イテレータ型
 * @tparam Iterator2 第2標本の入力イテレータ型
 * @param first1 第1標本の開始イテレータ
 * @param last1 第1標本の終了イテレータ
 * @param first2 第2標本の開始イテレータ
 * @param last2 第2標本の終了イテレータ
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（t統計量、p値、自由度）
 * @throws std::invalid_argument 標本の長さが異なる、または2未満の場合
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_paired(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::t_test_paired: samples must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::t_test_paired: need at least 2 pairs");
    }

    // Compute differences
    std::vector<double> diffs;
    diffs.reserve(n1);

    auto it1 = first1;
    auto it2 = first2;
    while (it1 != last1) {
        diffs.push_back(static_cast<double>(*it1) - static_cast<double>(*it2));
        ++it1;
        ++it2;
    }

    // Apply one-sample t-test on differences
    return t_test(diffs.begin(), diffs.end(), 0.0, alt);
}

// ============================================================================
// Chi-Square Test for Goodness of Fit
// ============================================================================

/**
 * @brief カイ二乗適合度検定
 *
 * 観測度数が期待度数と一致するかどうかを検定します。
 *
 * @tparam Iterator1 観測度数の入力イテレータ型
 * @tparam Iterator2 期待度数の入力イテレータ型
 * @param observed_first 観測度数の開始イテレータ
 * @param observed_last 観測度数の終了イテレータ
 * @param expected_first 期待度数の開始イテレータ
 * @param expected_last 期待度数の終了イテレータ
 * @return test_result 検定結果（カイ二乗統計量、p値、自由度）
 * @throws std::invalid_argument 観測と期待の長さが異なる、カテゴリ数が2未満、または期待度数が0以下の場合
 */
template <typename Iterator1, typename Iterator2>
test_result chisq_test_gof(Iterator1 observed_first, Iterator1 observed_last,
                           Iterator2 expected_first, Iterator2 expected_last)
{
    auto n_obs = statcpp::count(observed_first, observed_last);
    auto n_exp = statcpp::count(expected_first, expected_last);

    if (n_obs != n_exp) {
        throw std::invalid_argument("statcpp::chisq_test_gof: observed and expected must have same length");
    }
    if (n_obs < 2) {
        throw std::invalid_argument("statcpp::chisq_test_gof: need at least 2 categories");
    }

    double chi2 = 0.0;
    auto it_obs = observed_first;
    auto it_exp = expected_first;

    while (it_obs != observed_last) {
        double o = static_cast<double>(*it_obs);
        double e = static_cast<double>(*it_exp);

        if (e <= 0.0) {
            throw std::invalid_argument("statcpp::chisq_test_gof: expected values must be positive");
        }

        chi2 += (o - e) * (o - e) / e;

        ++it_obs;
        ++it_exp;
    }

    double df = static_cast<double>(n_obs - 1);
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

/**
 * @brief カイ二乗適合度検定（均等期待度数）
 *
 * 観測度数が均等分布に従うかどうかを検定します。
 *
 * @tparam Iterator 観測度数の入力イテレータ型
 * @param observed_first 観測度数の開始イテレータ
 * @param observed_last 観測度数の終了イテレータ
 * @return test_result 検定結果（カイ二乗統計量、p値、自由度）
 * @throws std::invalid_argument カテゴリ数が2未満の場合
 */
template <typename Iterator>
test_result chisq_test_gof_uniform(Iterator observed_first, Iterator observed_last)
{
    auto n = statcpp::count(observed_first, observed_last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::chisq_test_gof_uniform: need at least 2 categories");
    }

    double total = 0.0;
    for (auto it = observed_first; it != observed_last; ++it) {
        total += static_cast<double>(*it);
    }

    double expected = total / static_cast<double>(n);

    double chi2 = 0.0;
    for (auto it = observed_first; it != observed_last; ++it) {
        double o = static_cast<double>(*it);
        chi2 += (o - expected) * (o - expected) / expected;
    }

    double df = static_cast<double>(n - 1);
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// Chi-Square Test for Independence
// ============================================================================

/**
 * @brief カイ二乗独立性検定
 *
 * 分割表に基づいて2つの変数の独立性を検定します。
 *
 * @param contingency_table 分割表（行優先の2次元配列）
 * @return test_result 検定結果（カイ二乗統計量、p値、自由度）
 * @throws std::invalid_argument 行数または列数が2未満、列数が不均一、負の値がある、またはテーブルが空の場合
 */
inline test_result chisq_test_independence(const std::vector<std::vector<double>>& contingency_table)
{
    std::size_t rows = contingency_table.size();
    if (rows < 2) {
        throw std::invalid_argument("statcpp::chisq_test_independence: need at least 2 rows");
    }

    std::size_t cols = contingency_table[0].size();
    if (cols < 2) {
        throw std::invalid_argument("statcpp::chisq_test_independence: need at least 2 columns");
    }

    // Check all rows have same number of columns
    for (const auto& row : contingency_table) {
        if (row.size() != cols) {
            throw std::invalid_argument("statcpp::chisq_test_independence: inconsistent column count");
        }
    }

    // Compute row and column totals
    std::vector<double> row_totals(rows, 0.0);
    std::vector<double> col_totals(cols, 0.0);
    double grand_total = 0.0;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double val = contingency_table[i][j];
            if (val < 0.0) {
                throw std::invalid_argument("statcpp::chisq_test_independence: negative cell value");
            }
            row_totals[i] += val;
            col_totals[j] += val;
            grand_total += val;
        }
    }

    if (grand_total == 0.0) {
        throw std::invalid_argument("statcpp::chisq_test_independence: empty table");
    }

    // Compute chi-square statistic
    double chi2 = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double expected = row_totals[i] * col_totals[j] / grand_total;
            if (expected > 0.0) {
                double observed = contingency_table[i][j];
                chi2 += (observed - expected) * (observed - expected) / expected;
            }
        }
    }

    double df = static_cast<double>((rows - 1) * (cols - 1));
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// F-Test for Variance Ratio
// ============================================================================

/**
 * @brief F検定（分散比較）
 *
 * 2つの標本の分散が等しいかどうかを検定します。
 *
 * @tparam Iterator1 第1標本の入力イテレータ型
 * @tparam Iterator2 第2標本の入力イテレータ型
 * @param first1 第1標本の開始イテレータ
 * @param last1 第1標本の終了イテレータ
 * @param first2 第2標本の開始イテレータ
 * @param last2 第2標本の終了イテレータ
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result 検定結果（F統計量、p値、df=df1、df2=df2）
 * @throws std::invalid_argument いずれかの標本の要素数が2未満、または第2標本の分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
test_result f_test(Iterator1 first1, Iterator1 last1,
                   Iterator2 first2, Iterator2 last2,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::f_test: need at least 2 elements in each sample");
    }

    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    if (var2 == 0.0) {
        throw std::invalid_argument("statcpp::f_test: second sample has zero variance");
    }

    double f = var1 / var2;
    double df1 = static_cast<double>(n1 - 1);
    double df2 = static_cast<double>(n2 - 1);

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = f_cdf(f, df1, df2);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - f_cdf(f, df1, df2);
            break;
        case alternative_hypothesis::two_sided:
        default:
            {
                double p1 = f_cdf(f, df1, df2);
                double p2 = 1.0 - p1;
                p_value = 2.0 * std::min(p1, p2);
            }
            break;
    }

    return {f, p_value, df1, alt, df2};
}

// ============================================================================
// Multiple Testing Correction
// ============================================================================

/**
 * @brief Bonferroni補正
 *
 * 多重検定のp値をBonferroni法で補正します。
 * 各p値にテスト数を乗じ、1を超える場合は1に切り詰めます。
 *
 * @param p_values 元のp値のベクトル
 * @return std::vector<double> 補正後のp値のベクトル
 */
inline std::vector<double> bonferroni_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    std::vector<double> adjusted(n);

    for (std::size_t i = 0; i < n; ++i) {
        adjusted[i] = std::min(1.0, p_values[i] * static_cast<double>(n));
    }

    return adjusted;
}

/**
 * @brief Benjamini-Hochberg補正（FDR制御）
 *
 * 多重検定のp値をBenjamini-Hochberg法で補正します。
 * 偽発見率（FDR）を制御するための段階的な補正を行います。
 *
 * @param p_values 元のp値のベクトル
 * @return std::vector<double> 補正後のp値のベクトル
 */
inline std::vector<double> benjamini_hochberg_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    if (n == 0) return {};

    // Create index-pvalue pairs and sort by p-value
    std::vector<std::pair<std::size_t, double>> indexed(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed[i] = {i, p_values[i]};
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<double> adjusted(n);

    // Compute adjusted p-values
    double prev_adj = 1.0;
    for (std::size_t i = n; i > 0; --i) {
        std::size_t idx = indexed[i - 1].first;
        double p = indexed[i - 1].second;
        double adj = p * static_cast<double>(n) / static_cast<double>(i);
        adj = std::min(adj, prev_adj);
        adj = std::min(adj, 1.0);
        adjusted[idx] = adj;
        prev_adj = adj;
    }

    return adjusted;
}

/**
 * @brief Holm補正（段階的Bonferroni法）
 *
 * 多重検定のp値をHolm法で補正します。
 * Bonferroni補正の段階的バージョンで、より検出力が高いです。
 *
 * @param p_values 元のp値のベクトル
 * @return std::vector<double> 補正後のp値のベクトル
 */
inline std::vector<double> holm_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    if (n == 0) return {};

    // Create index-pvalue pairs and sort by p-value
    std::vector<std::pair<std::size_t, double>> indexed(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed[i] = {i, p_values[i]};
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<double> adjusted(n);

    double max_adj = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = indexed[i].first;
        double p = indexed[i].second;
        double adj = p * static_cast<double>(n - i);
        adj = std::max(adj, max_adj);
        adj = std::min(adj, 1.0);
        adjusted[idx] = adj;
        max_adj = adj;
    }

    return adjusted;
}

} // namespace statcpp
