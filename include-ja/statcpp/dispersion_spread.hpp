/**
 * @file dispersion_spread.hpp
 * @brief 散布度・分散の計算関数
 *
 * 分散、標準偏差、範囲、四分位範囲などのデータの散らばりを測定する関数を提供します。
 * イテレータベースのインターフェースで、様々なコンテナに対応しています。
 *
 * @note 数値安定性: 分散・標準偏差の計算には two-pass アルゴリズム（1パス目で平均を
 *       計算し、2パス目で平均からの偏差の二乗和を計算）を使用しています。一般的な
 *       データセットに対して数値的に安定です。全ての値が非常に大きく相対差が極めて
 *       小さいような極端なケース（例: 1e15付近の値で差が1e-2程度）では、IEEE 754
 *       倍精度の有効桁数（約15桁）の制約により精度が制限される場合があります。
 *       この制約は浮動小数点表現に固有のものであり、Welford法を含むすべての
 *       アルゴリズムに共通します。R も同じ two-pass 方式を使用しています。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"

namespace statcpp {

// ============================================================================
// Range
// ============================================================================

/**
 * @brief 範囲（最大値 - 最小値）
 *
 * 範囲内の最大値と最小値の差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 範囲（最大値 - 最小値）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double range(Iterator first, Iterator last)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::range: empty range");
    }
    auto [min_it, max_it] = std::minmax_element(first, last);
    return static_cast<double>(*max_it) - static_cast<double>(*min_it);
}

/**
 * @brief ラムダ式で射影した値の範囲
 *
 * 各要素に射影関数を適用した結果の範囲を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の範囲（最大値 - 最小値）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double range(Iterator first, Iterator last, Projection proj)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::range: empty range");
    }
    auto it = first;
    double min_val = static_cast<double>(std::invoke(proj, *it));
    double max_val = min_val;
    ++it;
    for (; it != last; ++it) {
        double val = static_cast<double>(std::invoke(proj, *it));
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val - min_val;
}

// ============================================================================
// Variance with ddof
// ============================================================================

/**
 * @brief 分散（ddof = Delta Degrees of Freedom）
 *
 * ddof = 0: 母分散（N で割る）
 * ddof = 1: 標本分散/不偏分散（N-1 で割る）
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param ddof 自由度の補正値（0または1）
 * @return 分散
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator>
double var(Iterator first, Iterator last, std::size_t ddof = 0)
{
    if (ddof > 1) {
        throw std::invalid_argument("statcpp::var: ddof must be 0 or 1");
    }
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::var: empty range");
    }
    if (ddof == 1 && n < 2) {
        throw std::invalid_argument("statcpp::var: need at least 2 elements for ddof=1");
    }
    double m = statcpp::mean(first, last);
    double sum_sq = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(n - ddof);
}

/**
 * @brief 事前計算済み平均を使う分散（ddof 付き）
 *
 * 平均値が事前に計算されている場合に使用します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @param ddof 自由度の補正値（0または1）
 * @return 分散
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator>
double var(Iterator first, Iterator last, double precomputed_mean, std::size_t ddof)
{
    if (ddof > 1) {
        throw std::invalid_argument("statcpp::var: ddof must be 0 or 1");
    }
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::var: empty range");
    }
    if (ddof == 1 && n < 2) {
        throw std::invalid_argument("statcpp::var: need at least 2 elements for ddof=1");
    }
    double sum_sq = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - precomputed_mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(n - ddof);
}

/**
 * @brief ラムダ式で射影した値の分散（ddof 付き）
 *
 * 各要素に射影関数を適用した結果の分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param ddof 自由度の補正値（0または1）
 * @return 射影後の分散
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double var(Iterator first, Iterator last, Projection proj, std::size_t ddof = 0)
{
    if (ddof > 1) {
        throw std::invalid_argument("statcpp::var: ddof must be 0 or 1");
    }
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::var: empty range");
    }
    if (ddof == 1 && n < 2) {
        throw std::invalid_argument("statcpp::var: need at least 2 elements for ddof=1");
    }
    double m = statcpp::mean(first, last, proj);
    double sum_sq = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(n - ddof);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の分散（ddof 付き）
 *
 * 射影関数と事前計算された平均値を使用して分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @param ddof 自由度の補正値（0または1）
 * @return 射影後の分散
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double var(Iterator first, Iterator last, Projection proj, double precomputed_mean, std::size_t ddof)
{
    if (ddof > 1) {
        throw std::invalid_argument("statcpp::var: ddof must be 0 or 1");
    }
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::var: empty range");
    }
    if (ddof == 1 && n < 2) {
        throw std::invalid_argument("statcpp::var: need at least 2 elements for ddof=1");
    }
    double sum_sq = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - precomputed_mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(n - ddof);
}

// ============================================================================
// Population Variance (ddof=0 のエイリアス)
// ============================================================================

/**
 * @brief 母分散
 *
 * 母集団の分散を計算します（N で割る）。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 母分散
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double population_variance(Iterator first, Iterator last)
{
    return var(first, last, 0);
}

/**
 * @brief 事前計算済み平均を使う母分散
 *
 * 平均値が事前に計算されている場合の母分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 母分散
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double population_variance(Iterator first, Iterator last, double precomputed_mean)
{
    return var(first, last, precomputed_mean, 0);
}

/**
 * @brief ラムダ式で射影した値の母分散
 *
 * 各要素に射影関数を適用した結果の母分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の母分散
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double population_variance(Iterator first, Iterator last, Projection proj)
{
    return var(first, last, proj, 0);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の母分散
 *
 * 射影関数と事前計算された平均値を使用して母分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の母分散
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double population_variance(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return var(first, last, proj, precomputed_mean, 0);
}

// ============================================================================
// Sample Variance (ddof=1 のエイリアス)
// ============================================================================

/**
 * @brief 標本分散（不偏分散）
 *
 * 標本の不偏分散を計算します（N-1 で割る）。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 標本分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double sample_variance(Iterator first, Iterator last)
{
    return var(first, last, 1);
}

/**
 * @brief 事前計算済み平均を使う標本分散
 *
 * 平均値が事前に計算されている場合の標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 標本分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double sample_variance(Iterator first, Iterator last, double precomputed_mean)
{
    return var(first, last, precomputed_mean, 1);
}

/**
 * @brief ラムダ式で射影した値の標本分散
 *
 * 各要素に射影関数を適用した結果の標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の標本分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double sample_variance(Iterator first, Iterator last, Projection proj)
{
    return var(first, last, proj, 1);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の標本分散
 *
 * 射影関数と事前計算された平均値を使用して標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の標本分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double sample_variance(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return var(first, last, proj, precomputed_mean, 1);
}

// ============================================================================
// Variance (= Sample Variance)
// ============================================================================

/**
 * @brief 分散（sample_variance のエイリアス）
 *
 * 標本分散を計算します。sample_variance と同等です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double variance(Iterator first, Iterator last)
{
    return sample_variance(first, last);
}

/**
 * @brief 事前計算済み平均を使う分散
 *
 * 平均値が事前に計算されている場合の標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double variance(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_variance(first, last, precomputed_mean);
}

/**
 * @brief ラムダ式で射影した値の分散
 *
 * 各要素に射影関数を適用した結果の標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double variance(Iterator first, Iterator last, Projection proj)
{
    return sample_variance(first, last, proj);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の分散
 *
 * 射影関数と事前計算された平均値を使用して標本分散を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の分散
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double variance(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_variance(first, last, proj, precomputed_mean);
}

// ============================================================================
// Standard Deviation with ddof (NumPy-style)
// ============================================================================

/**
 * @brief 標準偏差（ddof = Delta Degrees of Freedom）
 *
 * ddof = 0: 母標準偏差（N で割った分散の平方根）
 * ddof = 1: 標本標準偏差（N-1 で割った分散の平方根）
 * NumPy の np.std(a, ddof=...) と同様の動作をします。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param ddof 自由度の補正値（0または1）
 * @return 標準偏差
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator>
double stdev(Iterator first, Iterator last, std::size_t ddof = 0)
{
    return std::sqrt(var(first, last, ddof));
}

/**
 * @brief 事前計算済み平均を使う標準偏差（ddof 付き）
 *
 * 平均値が事前に計算されている場合に使用します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @param ddof 自由度の補正値（0または1）
 * @return 標準偏差
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator>
double stdev(Iterator first, Iterator last, double precomputed_mean, std::size_t ddof)
{
    return std::sqrt(var(first, last, precomputed_mean, ddof));
}

/**
 * @brief ラムダ式で射影した値の標準偏差（ddof 付き）
 *
 * 各要素に射影関数を適用した結果の標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param ddof 自由度の補正値（0または1）
 * @return 射影後の標準偏差
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double stdev(Iterator first, Iterator last, Projection proj, std::size_t ddof = 0)
{
    return std::sqrt(var(first, last, proj, ddof));
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の標準偏差（ddof 付き）
 *
 * 射影関数と事前計算された平均値を使用して標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @param ddof 自由度の補正値（0または1）
 * @return 射影後の標準偏差
 * @throws std::invalid_argument 空の範囲の場合、ddofが0または1以外の場合、ddof=1で要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double stdev(Iterator first, Iterator last, Projection proj, double precomputed_mean, std::size_t ddof)
{
    return std::sqrt(var(first, last, proj, precomputed_mean, ddof));
}

// ============================================================================
// Population Standard Deviation (ddof=0 のエイリアス)
// ============================================================================

/**
 * @brief 母標準偏差
 *
 * 母集団の標準偏差を計算します（N で割った分散の平方根）。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 母標準偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double population_stddev(Iterator first, Iterator last)
{
    return stdev(first, last, 0);
}

/**
 * @brief 事前計算済み平均を使う母標準偏差
 *
 * 平均値が事前に計算されている場合の母標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 母標準偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double population_stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return stdev(first, last, precomputed_mean, 0);
}

/**
 * @brief ラムダ式で射影した値の母標準偏差
 *
 * 各要素に射影関数を適用した結果の母標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の母標準偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double population_stddev(Iterator first, Iterator last, Projection proj)
{
    return stdev(first, last, proj, 0);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の母標準偏差
 *
 * 射影関数と事前計算された平均値を使用して母標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の母標準偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double population_stddev(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return stdev(first, last, proj, precomputed_mean, 0);
}

// ============================================================================
// Sample Standard Deviation (ddof=1 のエイリアス)
// ============================================================================

/**
 * @brief 標本標準偏差
 *
 * 標本の標準偏差を計算します（N-1 で割った分散の平方根）。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 標本標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double sample_stddev(Iterator first, Iterator last)
{
    return stdev(first, last, 1);
}

/**
 * @brief 事前計算済み平均を使う標本標準偏差
 *
 * 平均値が事前に計算されている場合の標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 標本標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double sample_stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return stdev(first, last, precomputed_mean, 1);
}

/**
 * @brief ラムダ式で射影した値の標本標準偏差
 *
 * 各要素に射影関数を適用した結果の標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の標本標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double sample_stddev(Iterator first, Iterator last, Projection proj)
{
    return stdev(first, last, proj, 1);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の標本標準偏差
 *
 * 射影関数と事前計算された平均値を使用して標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の標本標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double sample_stddev(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return stdev(first, last, proj, precomputed_mean, 1);
}

// ============================================================================
// Standard Deviation (= Sample Standard Deviation)
// ============================================================================

/**
 * @brief 標準偏差（sample_stddev のエイリアス）
 *
 * 標本標準偏差を計算します。sample_stddev と同等です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double stddev(Iterator first, Iterator last)
{
    return sample_stddev(first, last);
}

/**
 * @brief 事前計算済み平均を使う標準偏差
 *
 * 平均値が事前に計算されている場合の標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator>
double stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_stddev(first, last, precomputed_mean);
}

/**
 * @brief ラムダ式で射影した値の標準偏差
 *
 * 各要素に射影関数を適用した結果の標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double stddev(Iterator first, Iterator last, Projection proj)
{
    return sample_stddev(first, last, proj);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の標準偏差
 *
 * 射影関数と事前計算された平均値を使用して標本標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の標準偏差
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合
 */
template <typename Iterator, typename Projection>
double stddev(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_stddev(first, last, proj, precomputed_mean);
}

// ============================================================================
// Coefficient of Variation
// ============================================================================

/**
 * @brief 変動係数
 *
 * 標準偏差を平均で割った値を計算します（比率で返す。パーセント表示には x100 すること）。
 * データの相対的なばらつきを測定するのに有用です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 変動係数（比率）
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合、平均が0の場合
 */
template <typename Iterator>
double coefficient_of_variation(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: empty range");
    }
    if (n < 2) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: need at least 2 elements");
    }
    double m = statcpp::mean(first, last);
    if (m == 0.0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: mean is zero");
    }
    return sample_stddev(first, last, m) / std::abs(m);
}

/**
 * @brief 事前計算済み平均を使う変動係数
 *
 * 平均値が事前に計算されている場合の変動係数を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 変動係数（比率）
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合、平均が0の場合
 */
template <typename Iterator>
double coefficient_of_variation(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: empty range");
    }
    if (n < 2) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: need at least 2 elements");
    }
    if (precomputed_mean == 0.0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: mean is zero");
    }
    return sample_stddev(first, last, precomputed_mean) / std::abs(precomputed_mean);
}

/**
 * @brief ラムダ式で射影した値の変動係数
 *
 * 各要素に射影関数を適用した結果の変動係数を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の変動係数（比率）
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合、平均が0の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double coefficient_of_variation(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: empty range");
    }
    if (n < 2) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: need at least 2 elements");
    }
    double m = statcpp::mean(first, last, proj);
    if (m == 0.0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: mean is zero");
    }
    return sample_stddev(first, last, proj, m) / std::abs(m);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の変動係数
 *
 * 射影関数と事前計算された平均値を使用して変動係数を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の変動係数（比率）
 * @throws std::invalid_argument 空の範囲の場合、要素数が2未満の場合、平均が0の場合
 */
template <typename Iterator, typename Projection>
double coefficient_of_variation(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: empty range");
    }
    if (n < 2) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: need at least 2 elements");
    }
    if (precomputed_mean == 0.0) {
        throw std::invalid_argument("statcpp::coefficient_of_variation: mean is zero");
    }
    return sample_stddev(first, last, proj, precomputed_mean) / std::abs(precomputed_mean);
}

// ============================================================================
// Interquartile Range (IQR)
// ============================================================================

/**
 * @brief 四分位範囲（ソート済みの範囲を受け取る）
 *
 * 第3四分位数と第1四分位数の差を計算します。
 * データの中央50%の広がりを測定するのに有用です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 四分位範囲（Q3 - Q1）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double iqr(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::iqr: empty range");
    }
    double q1 = statcpp::interpolate_at(first, n, 0.25);
    double q3 = statcpp::interpolate_at(first, n, 0.75);
    return q3 - q1;
}

/**
 * @brief ラムダ式で射影した値の四分位範囲（射影結果がソート済みの順序で並んでいること）
 *
 * 各要素に射影関数を適用した結果の四分位範囲を計算します。
 * 射影結果がソート済みであることを前提とします。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の四分位範囲（Q3 - Q1）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double iqr(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::iqr: empty range");
    }
    double q1 = statcpp::interpolate_at(first, n, 0.25, proj);
    double q3 = statcpp::interpolate_at(first, n, 0.75, proj);
    return q3 - q1;
}

// ============================================================================
// Mean Absolute Deviation
// ============================================================================

/**
 * @brief 平均偏差
 *
 * 各要素と平均との差の絶対値の平均を計算します。
 * 分散よりも外れ値の影響を受けにくい散布度の指標です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 平均偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double mean_absolute_deviation(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean_absolute_deviation: empty range");
    }
    double m = statcpp::mean(first, last);
    double sum_abs = 0.0;
    for (auto it = first; it != last; ++it) {
        sum_abs += std::abs(static_cast<double>(*it) - m);
    }
    return sum_abs / static_cast<double>(n);
}

/**
 * @brief 事前計算済み平均を使う平均偏差
 *
 * 平均値が事前に計算されている場合の平均偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param precomputed_mean 事前計算された平均値
 * @return 平均偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double mean_absolute_deviation(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean_absolute_deviation: empty range");
    }
    double sum_abs = 0.0;
    for (auto it = first; it != last; ++it) {
        sum_abs += std::abs(static_cast<double>(*it) - precomputed_mean);
    }
    return sum_abs / static_cast<double>(n);
}

/**
 * @brief ラムダ式で射影した値の平均偏差
 *
 * 各要素に射影関数を適用した結果の平均偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の平均偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double mean_absolute_deviation(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean_absolute_deviation: empty range");
    }
    double m = statcpp::mean(first, last, proj);
    double sum_abs = 0.0;
    for (auto it = first; it != last; ++it) {
        sum_abs += std::abs(static_cast<double>(std::invoke(proj, *it)) - m);
    }
    return sum_abs / static_cast<double>(n);
}

/**
 * @brief 事前計算済み平均を使うラムダ式射影版の平均偏差
 *
 * 射影関数と事前計算された平均値を使用して平均偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算された平均値
 * @return 射影後の平均偏差
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double mean_absolute_deviation(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean_absolute_deviation: empty range");
    }
    double sum_abs = 0.0;
    for (auto it = first; it != last; ++it) {
        sum_abs += std::abs(static_cast<double>(std::invoke(proj, *it)) - precomputed_mean);
    }
    return sum_abs / static_cast<double>(n);
}

// ============================================================================
// Weighted Variance
// ============================================================================

/**
 * @brief 重み付き分散（frequency weights: 各データポイントの繰り返し数）
 *
 * 各要素に重みを適用した分散を計算します。
 * ビッセル補正を適用して不偏推定量を返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @return 重み付き分散
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、重みの合計が0の場合、有効サンプルサイズが不足している場合
 */
template <typename Iterator, typename WeightIterator>
double weighted_variance(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_variance: empty range");
    }

    // 重み付き平均を計算
    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_variance: negative weight");
        }

        sum_weighted += value * weight;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_variance: sum of weights is zero");
    }

    double mean = sum_weighted / sum_weights;

    // 重み付き分散を計算（ビッセル補正版）
    double sum_squared_dev = 0.0;
    double sum_weights_squared = 0.0;
    weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);
        double dev = value - mean;

        sum_squared_dev += weight * dev * dev;
        sum_weights_squared += weight * weight;
    }

    // ビッセル補正: V = (Σw * Σw(x-μ)²) / (Σw² - Σw²)
    // 簡略版: V = Σw(x-μ)² / (Σw - Σw²/Σw)
    double correction = sum_weights - (sum_weights_squared / sum_weights);

    if (correction <= 0.0) {
        throw std::invalid_argument("statcpp::weighted_variance: insufficient effective sample size");
    }

    return sum_squared_dev / correction;
}

/**
 * @brief 重み付き分散（射影版）
 *
 * 各要素に射影関数を適用した結果の重み付き分散を計算します。
 * ビッセル補正を適用して不偏推定量を返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @param proj 射影関数
 * @return 射影後の重み付き分散
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、重みの合計が0の場合、有効サンプルサイズが不足している場合
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_variance(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_variance: empty range");
    }

    // 重み付き平均を計算
    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_variance: negative weight");
        }

        sum_weighted += value * weight;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_variance: sum of weights is zero");
    }

    double mean = sum_weighted / sum_weights;

    // 重み付き分散を計算
    double sum_squared_dev = 0.0;
    double sum_weights_squared = 0.0;
    weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);
        double dev = value - mean;

        sum_squared_dev += weight * dev * dev;
        sum_weights_squared += weight * weight;
    }

    double correction = sum_weights - (sum_weights_squared / sum_weights);

    if (correction <= 0.0) {
        throw std::invalid_argument("statcpp::weighted_variance: insufficient effective sample size");
    }

    return sum_squared_dev / correction;
}

/**
 * @brief 重み付き標準偏差
 *
 * 重み付き分散の平方根を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @return 重み付き標準偏差
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、重みの合計が0の場合、有効サンプルサイズが不足している場合
 */
template <typename Iterator, typename WeightIterator>
double weighted_stddev(Iterator first, Iterator last, WeightIterator weight_first)
{
    return std::sqrt(weighted_variance(first, last, weight_first));
}

/**
 * @brief 重み付き標準偏差（射影版）
 *
 * 各要素に射影関数を適用した結果の重み付き標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @param proj 射影関数
 * @return 射影後の重み付き標準偏差
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、重みの合計が0の場合、有効サンプルサイズが不足している場合
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_stddev(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    return std::sqrt(weighted_variance(first, last, weight_first, proj));
}

// ============================================================================
// Geometric Standard Deviation
// ============================================================================

/**
 * @brief 幾何標準偏差 (Geometric Standard Deviation)
 *
 * 対数変換したデータの標準偏差の指数を計算します。
 * GSD = exp(sigma_ln) where sigma_ln is the standard deviation of ln(x)
 * 対数正規分布に従うデータの散布度を測定するのに有用です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 幾何標準偏差
 * @throws std::invalid_argument 空の範囲の場合、値が0以下の場合
 */
template <typename Iterator>
double geometric_stddev(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_stddev: empty range");
    }

    // 対数変換したデータの標準偏差を計算
    std::vector<double> log_values;
    log_values.reserve(n);

    for (auto it = first; it != last; ++it) {
        double value = static_cast<double>(*it);
        if (value <= 0.0) {
            throw std::invalid_argument("statcpp::geometric_stddev: all values must be positive");
        }
        log_values.push_back(std::log(value));
    }

    double log_stddev = statcpp::sample_stddev(log_values.begin(), log_values.end());
    return std::exp(log_stddev);
}

/**
 * @brief 幾何標準偏差（射影版）
 *
 * 各要素に射影関数を適用した結果の幾何標準偏差を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の幾何標準偏差
 * @throws std::invalid_argument 空の範囲の場合、値が0以下の場合
 */
template <typename Iterator, typename Projection>
double geometric_stddev(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_stddev: empty range");
    }

    // 対数変換したデータの標準偏差を計算
    std::vector<double> log_values;
    log_values.reserve(n);

    for (auto it = first; it != last; ++it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        if (value <= 0.0) {
            throw std::invalid_argument("statcpp::geometric_stddev: all values must be positive");
        }
        log_values.push_back(std::log(value));
    }

    double log_stddev = statcpp::sample_stddev(log_values.begin(), log_values.end());
    return std::exp(log_stddev);
}

} // namespace statcpp
