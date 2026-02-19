/**
 * @file resampling.hpp
 * @brief リサンプリング手法
 *
 * ブートストラップ、ジャックナイフ、置換検定などのリサンプリング手法を提供します。
 * 統計量の標準誤差推定、信頼区間計算、仮説検定に使用できます。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Bootstrap Result Structure
// ============================================================================

/**
 * @brief ブートストラップ推定の結果を格納する構造体
 *
 * ブートストラップ法による統計量の推定結果、標準誤差、信頼区間、
 * バイアス、および全ての複製統計量を保持します。
 */
struct bootstrap_result {
    double estimate;                 ///< 元データから計算された統計量の推定値
    double standard_error;           ///< ブートストラップ標準誤差
    double ci_lower;                 ///< 信頼区間の下限
    double ci_upper;                 ///< 信頼区間の上限
    double bias;                     ///< バイアス（複製平均 - 推定値）
    std::vector<double> replicates;  ///< 全てのブートストラップ複製統計量
};

// ============================================================================
// Bootstrap Sampling
// ============================================================================

/**
 * @brief ブートストラップサンプルを1つ生成する
 *
 * 与えられたデータから復元抽出により、元のサンプルサイズと同じ大きさの
 * ブートストラップサンプルを生成します。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param engine 乱数エンジンへの参照
 * @return 復元抽出されたブートストラップサンプル
 * @throws std::invalid_argument 入力範囲が空の場合
 */
template <typename Iterator, typename Engine = default_random_engine>
std::vector<typename std::iterator_traits<Iterator>::value_type>
bootstrap_sample(Iterator first, Iterator last, Engine& engine)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::bootstrap_sample: empty range");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);
    std::vector<value_type> sample(n);

    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (std::size_t i = 0; i < n; ++i) {
        sample[i] = original[dist(engine)];
    }

    return sample;
}

/**
 * @brief ブートストラップサンプルを1つ生成する（デフォルト乱数エンジン使用）
 *
 * グローバルなデフォルト乱数エンジンを使用してブートストラップサンプルを生成します。
 *
 * @tparam Iterator 入力イテレータの型
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @return 復元抽出されたブートストラップサンプル
 * @throws std::invalid_argument 入力範囲が空の場合
 */
template <typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type>
bootstrap_sample(Iterator first, Iterator last)
{
    return bootstrap_sample(first, last, get_random_engine());
}

// ============================================================================
// Bootstrap Estimation
// ============================================================================

/**
 * @brief 汎用ブートストラップ推定を行う
 *
 * 任意の統計関数に対してブートストラップ法を適用し、統計量の推定値、
 * 標準誤差、パーセンタイル信頼区間、バイアスを計算します。
 *
 * パーセンタイル信頼区間の計算:
 * - 下限インデックス: floor(α/2 * B)
 * - 上限インデックス: floor((1 - α/2) * B) - 1
 * - ここで α = 1 - confidence, B = n_bootstrap
 *
 * @note パーセンタイル法は最も単純な信頼区間計算法です。より正確な
 *       信頼区間が必要な場合は bootstrap_bca() を使用してください。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Statistic 統計関数の型（イテレータペアを受け取りdoubleを返す関数）
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param stat_func 統計量を計算する関数オブジェクト
 * @param n_bootstrap ブートストラップ反復回数（デフォルト: 1000）
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @param engine 乱数エンジンへの参照
 * @return ブートストラップ推定結果を格納したbootstrap_result構造体
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 * @throws std::invalid_argument 入力要素数が2未満の場合
 */
template <typename Iterator, typename Statistic, typename Engine = default_random_engine>
bootstrap_result bootstrap(Iterator first, Iterator last, Statistic stat_func,
                           std::size_t n_bootstrap = 1000, double confidence = 0.95,
                           Engine& engine = get_random_engine())
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::bootstrap: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::bootstrap: need at least 2 elements");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);

    // Original statistic
    double theta_hat = stat_func(original.begin(), original.end());

    // Bootstrap replicates
    std::vector<double> replicates(n_bootstrap);

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        auto sample = bootstrap_sample(original.begin(), original.end(), engine);
        replicates[b] = stat_func(sample.begin(), sample.end());
    }

    // Sort replicates for percentile CI
    std::vector<double> sorted_replicates = replicates;
    std::sort(sorted_replicates.begin(), sorted_replicates.end());

    // Standard error
    double mean_rep = std::accumulate(replicates.begin(), replicates.end(), 0.0) / n_bootstrap;
    double se = 0.0;
    for (double rep : replicates) {
        se += (rep - mean_rep) * (rep - mean_rep);
    }
    se = std::sqrt(se / (n_bootstrap - 1));

    // Bias
    double bias = mean_rep - theta_hat;

    // Percentile CI (0-indexed)
    // lower_idx corresponds to the (α/2)-th percentile
    // upper_idx corresponds to the (1 - α/2)-th percentile
    double alpha = 1.0 - confidence;
    std::size_t lower_idx = static_cast<std::size_t>(std::floor(alpha / 2.0 * static_cast<double>(n_bootstrap)));
    std::size_t upper_idx = static_cast<std::size_t>(std::floor((1.0 - alpha / 2.0) * static_cast<double>(n_bootstrap)));

    // Boundary check
    if (lower_idx >= n_bootstrap) lower_idx = n_bootstrap - 1;
    if (upper_idx >= n_bootstrap) upper_idx = n_bootstrap - 1;

    double ci_lower = sorted_replicates[lower_idx];
    double ci_upper = sorted_replicates[upper_idx];

    return {theta_hat, se, ci_lower, ci_upper, bias, std::move(replicates)};
}

/**
 * @brief 平均のブートストラップ推定を行う
 *
 * サンプル平均に対してブートストラップ法を適用し、標準誤差と信頼区間を計算します。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param n_bootstrap ブートストラップ反復回数（デフォルト: 1000）
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @param engine 乱数エンジンへの参照
 * @return ブートストラップ推定結果を格納したbootstrap_result構造体
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 * @throws std::invalid_argument 入力要素数が2未満の場合
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_mean(Iterator first, Iterator last,
                                std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) { return statcpp::mean(f, l); };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

/**
 * @brief 中央値のブートストラップ推定を行う
 *
 * サンプル中央値に対してブートストラップ法を適用し、標準誤差と信頼区間を計算します。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param n_bootstrap ブートストラップ反復回数（デフォルト: 1000）
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @param engine 乱数エンジンへの参照
 * @return ブートストラップ推定結果を格納したbootstrap_result構造体
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 * @throws std::invalid_argument 入力要素数が2未満の場合
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_median(Iterator first, Iterator last,
                                  std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                  Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) {
        std::vector<typename std::iterator_traits<decltype(f)>::value_type> sorted(f, l);
        std::sort(sorted.begin(), sorted.end());
        return statcpp::median(sorted.begin(), sorted.end());
    };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

/**
 * @brief 標準偏差のブートストラップ推定を行う
 *
 * サンプル標準偏差に対してブートストラップ法を適用し、標準誤差と信頼区間を計算します。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param n_bootstrap ブートストラップ反復回数（デフォルト: 1000）
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @param engine 乱数エンジンへの参照
 * @return ブートストラップ推定結果を格納したbootstrap_result構造体
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 * @throws std::invalid_argument 入力要素数が2未満の場合
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_stddev(Iterator first, Iterator last,
                                  std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                  Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) { return statcpp::sample_stddev(f, l); };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

// ============================================================================
// BCa Bootstrap (Bias-corrected and accelerated)
// ============================================================================

/**
 * @brief BCa（バイアス補正・加速）ブートストラップ信頼区間を計算する
 *
 * 標準的なパーセンタイル法よりも正確な信頼区間を提供するBCa法を実装しています。
 * ジャックナイフ法を使用して加速係数を推定し、バイアス補正を行います。
 *
 * @tparam Iterator 入力イテレータの型
 * @tparam Statistic 統計関数の型（イテレータペアを受け取りdoubleを返す関数）
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first 入力範囲の先頭イテレータ
 * @param last 入力範囲の終端イテレータ
 * @param stat_func 統計量を計算する関数オブジェクト
 * @param n_bootstrap ブートストラップ反復回数（デフォルト: 1000）
 * @param confidence 信頼水準（デフォルト: 0.95）
 * @param engine 乱数エンジンへの参照
 * @return ブートストラップ推定結果を格納したbootstrap_result構造体
 * @throws std::invalid_argument confidenceが(0, 1)の範囲外の場合
 * @throws std::invalid_argument 入力要素数が3未満の場合
 */
template <typename Iterator, typename Statistic, typename Engine = default_random_engine>
bootstrap_result bootstrap_bca(Iterator first, Iterator last, Statistic stat_func,
                               std::size_t n_bootstrap = 1000, double confidence = 0.95,
                               Engine& engine = get_random_engine())
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::bootstrap_bca: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 3) {
        throw std::invalid_argument("statcpp::bootstrap_bca: need at least 3 elements");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);

    // Original statistic
    double theta_hat = stat_func(original.begin(), original.end());

    // Bootstrap replicates
    std::vector<double> replicates(n_bootstrap);

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        auto sample = bootstrap_sample(original.begin(), original.end(), engine);
        replicates[b] = stat_func(sample.begin(), sample.end());
    }

    // Bias correction factor z0
    std::size_t count_less = 0;
    for (double rep : replicates) {
        if (rep < theta_hat) ++count_less;
    }
    // Clip to [1, B-1] to avoid norm_quantile(0) = -inf or norm_quantile(1) = +inf
    if (count_less == 0) count_less = 1;
    if (count_less >= n_bootstrap) count_less = n_bootstrap - 1;
    double z0 = norm_quantile(static_cast<double>(count_less) / n_bootstrap);

    // Acceleration factor using jackknife
    std::vector<double> jackknife_estimates(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<value_type> jackknife_sample;
        jackknife_sample.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i) {
                jackknife_sample.push_back(original[j]);
            }
        }
        jackknife_estimates[i] = stat_func(jackknife_sample.begin(), jackknife_sample.end());
    }

    double jack_mean = std::accumulate(jackknife_estimates.begin(), jackknife_estimates.end(), 0.0) / n;
    double sum_cubed = 0.0;
    double sum_squared = 0.0;
    for (double j : jackknife_estimates) {
        double d = jack_mean - j;
        sum_squared += d * d;
        sum_cubed += d * d * d;
    }

    double a = (sum_squared == 0.0) ? 0.0 : sum_cubed / (6.0 * std::pow(sum_squared, 1.5));

    // Adjusted percentiles
    double alpha = 1.0 - confidence;
    double z_alpha_lower = norm_quantile(alpha / 2.0);
    double z_alpha_upper = norm_quantile(1.0 - alpha / 2.0);

    double alpha1 = norm_cdf(z0 + (z0 + z_alpha_lower) / (1.0 - a * (z0 + z_alpha_lower)));
    double alpha2 = norm_cdf(z0 + (z0 + z_alpha_upper) / (1.0 - a * (z0 + z_alpha_upper)));

    // Sort replicates
    std::vector<double> sorted_replicates = replicates;
    std::sort(sorted_replicates.begin(), sorted_replicates.end());

    std::size_t lower_idx = static_cast<std::size_t>(alpha1 * n_bootstrap);
    std::size_t upper_idx = static_cast<std::size_t>(alpha2 * n_bootstrap);

    if (lower_idx >= n_bootstrap) lower_idx = n_bootstrap - 1;
    if (upper_idx >= n_bootstrap) upper_idx = n_bootstrap - 1;

    double ci_lower = sorted_replicates[lower_idx];
    double ci_upper = sorted_replicates[upper_idx];

    // Standard error
    double mean_rep = std::accumulate(replicates.begin(), replicates.end(), 0.0) / n_bootstrap;
    double se = 0.0;
    for (double rep : replicates) {
        se += (rep - mean_rep) * (rep - mean_rep);
    }
    se = std::sqrt(se / (n_bootstrap - 1));

    double bias = mean_rep - theta_hat;

    return {theta_hat, se, ci_lower, ci_upper, bias, std::move(replicates)};
}

// ============================================================================
// Permutation Test Result
// ============================================================================

/**
 * @brief 置換検定の結果を格納する構造体
 *
 * 置換検定による観測統計量、p値、置換回数、および置換分布を保持します。
 */
struct permutation_result {
    double observed_statistic;                   ///< 観測された検定統計量
    double p_value;                              ///< 両側p値
    std::size_t n_permutations;                  ///< 実行された置換回数
    std::vector<double> permutation_distribution; ///< 置換統計量の分布
};

// ============================================================================
// Permutation Test (Two-Sample)
// ============================================================================

/**
 * @brief 2標本置換検定（平均差の検定）を行う
 *
 * 2つの独立したサンプル間の平均差について、置換検定を実行します。
 * 帰無仮説は「2群の平均が等しい」です。両側検定のp値を計算します。
 *
 * p値の計算には inclusive method を使用:
 * p = (count(|T*| >= |T_obs|) + 1) / (n_permutations + 1)
 * これにより観測データ自体も帰無分布の一部として含まれ、
 * p値が0になることを防ぎます（Phipson & Smyth, 2010）。
 *
 * @tparam Iterator1 第1サンプルのイテレータ型
 * @tparam Iterator2 第2サンプルのイテレータ型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first1 第1サンプルの先頭イテレータ
 * @param last1 第1サンプルの終端イテレータ
 * @param first2 第2サンプルの先頭イテレータ
 * @param last2 第2サンプルの終端イテレータ
 * @param n_permutations 置換回数（デフォルト: 10000）
 * @param engine 乱数エンジンへの参照
 * @return 置換検定結果を格納したpermutation_result構造体
 * @throws std::invalid_argument いずれかのサンプルが空の場合
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_two_sample(Iterator1 first1, Iterator1 last1,
                                                Iterator2 first2, Iterator2 last2,
                                                std::size_t n_permutations = 10000,
                                                Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::permutation_test_two_sample: empty sample");
    }

    // Combine samples
    std::vector<double> combined;
    combined.reserve(n1 + n2);
    for (auto it = first1; it != last1; ++it) {
        combined.push_back(static_cast<double>(*it));
    }
    for (auto it = first2; it != last2; ++it) {
        combined.push_back(static_cast<double>(*it));
    }

    // Observed statistic
    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double observed = mean1 - mean2;

    // Permutation distribution
    std::vector<double> perm_stats(n_permutations);

    for (std::size_t p = 0; p < n_permutations; ++p) {
        std::shuffle(combined.begin(), combined.end(), engine);

        double perm_mean1 = std::accumulate(combined.begin(), combined.begin() + n1, 0.0) / n1;
        double perm_mean2 = std::accumulate(combined.begin() + n1, combined.end(), 0.0) / n2;
        perm_stats[p] = perm_mean1 - perm_mean2;
    }

    // Two-sided p-value (inclusive method)
    // Include the observed statistic as part of the null distribution
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

// ============================================================================
// Permutation Test (Paired)
// ============================================================================

/**
 * @brief 対応のある置換検定を行う
 *
 * 対応のある2つのサンプル間の差について、置換検定を実行します。
 * 差の符号をランダムに反転させることで帰無仮説下の分布を生成します。
 * 帰無仮説は「差の平均が0である」です。両側検定のp値を計算します。
 *
 * p値の計算には inclusive method を使用:
 * p = (count(|T*| >= |T_obs|) + 1) / (n_permutations + 1)
 *
 * @tparam Iterator1 第1サンプルのイテレータ型
 * @tparam Iterator2 第2サンプルのイテレータ型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first1 第1サンプルの先頭イテレータ
 * @param last1 第1サンプルの終端イテレータ
 * @param first2 第2サンプルの先頭イテレータ
 * @param last2 第2サンプルの終端イテレータ
 * @param n_permutations 置換回数（デフォルト: 10000）
 * @param engine 乱数エンジンへの参照
 * @return 置換検定結果を格納したpermutation_result構造体
 * @throws std::invalid_argument 2つのサンプルの長さが異なる場合
 * @throws std::invalid_argument サンプルが空の場合
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_paired(Iterator1 first1, Iterator1 last1,
                                            Iterator2 first2, Iterator2 last2,
                                            std::size_t n_permutations = 10000,
                                            Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::permutation_test_paired: samples must have equal length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::permutation_test_paired: empty samples");
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

    // Observed statistic (mean of differences)
    double observed = statcpp::mean(diffs.begin(), diffs.end());

    // Permutation by randomly flipping signs
    std::vector<double> perm_stats(n_permutations);
    std::uniform_int_distribution<int> coin(0, 1);

    for (std::size_t p = 0; p < n_permutations; ++p) {
        double sum = 0.0;
        for (double d : diffs) {
            sum += (coin(engine) == 0) ? d : -d;
        }
        perm_stats[p] = sum / static_cast<double>(n1);
    }

    // Two-sided p-value (inclusive method)
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

// ============================================================================
// Permutation Test for Correlation
// ============================================================================

/**
 * @brief 相関係数の置換検定を行う
 *
 * 2つの変数間のピアソン相関係数について、置換検定を実行します。
 * 一方の変数をシャッフルすることで帰無仮説下の分布を生成します。
 * 帰無仮説は「2変数間に相関がない（相関係数が0）」です。
 * 両側検定のp値を計算します。
 *
 * @note p値の計算には inclusive method を使用しています：
 *       p = (観測値以上に極端な置換統計量の数 + 1) / (置換回数 + 1)
 *       この方法により、p値が0になることを防ぎ、より正確な推定が可能です
 *       (Phipson & Smyth, 2010)。
 *
 * @tparam Iterator1 第1変数のイテレータ型
 * @tparam Iterator2 第2変数のイテレータ型
 * @tparam Engine 乱数エンジンの型（デフォルト: default_random_engine）
 * @param first1 第1変数の先頭イテレータ
 * @param last1 第1変数の終端イテレータ
 * @param first2 第2変数の先頭イテレータ
 * @param last2 第2変数の終端イテレータ
 * @param n_permutations 置換回数（デフォルト: 10000）
 * @param engine 乱数エンジンへの参照
 * @return 置換検定結果を格納したpermutation_result構造体
 * @throws std::invalid_argument 2つの変数の長さが異なる場合
 * @throws std::invalid_argument データペア数が3未満の場合
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_correlation(Iterator1 first1, Iterator1 last1,
                                                 Iterator2 first2, Iterator2 last2,
                                                 std::size_t n_permutations = 10000,
                                                 Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::permutation_test_correlation: samples must have equal length");
    }
    if (n1 < 3) {
        throw std::invalid_argument("statcpp::permutation_test_correlation: need at least 3 pairs");
    }

    // Copy data
    std::vector<double> x, y;
    x.reserve(n1);
    y.reserve(n1);

    for (auto it = first1; it != last1; ++it) {
        x.push_back(static_cast<double>(*it));
    }
    for (auto it = first2; it != last2; ++it) {
        y.push_back(static_cast<double>(*it));
    }

    // Helper to compute correlation
    auto compute_corr = [n1](const std::vector<double>& a, const std::vector<double>& b) {
        double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / n1;
        double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / n1;

        double cov = 0.0, var_a = 0.0, var_b = 0.0;
        for (std::size_t i = 0; i < n1; ++i) {
            double da = a[i] - mean_a;
            double db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        return cov / std::sqrt(var_a * var_b);
    };

    // Observed correlation
    double observed = compute_corr(x, y);

    // Permutation distribution
    std::vector<double> perm_stats(n_permutations);
    std::vector<double> y_perm = y;

    for (std::size_t p = 0; p < n_permutations; ++p) {
        std::shuffle(y_perm.begin(), y_perm.end(), engine);
        perm_stats[p] = compute_corr(x, y_perm);
    }

    // Two-sided p-value (inclusive method)
    // Include the observed statistic as part of the null distribution
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

} // namespace statcpp
