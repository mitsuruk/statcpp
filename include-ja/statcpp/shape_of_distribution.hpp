/**
 * @file shape_of_distribution.hpp
 * @brief 分布の形状測定（歪度と尖度）
 *
 * このファイルは分布の形状を表す統計量である歪度（Skewness）と
 * 尖度（Kurtosis）を計算する関数を提供します。
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>

#include "statcpp/basic_statistics.hpp"

namespace statcpp {

// ============================================================================
// Skewness (歪度)
// ============================================================================

/**
 * @brief 母歪度の計算（Fisher の定義）
 *
 * 母集団の歪度を計算します。分布の非対称性を示す指標です。
 *
 * 計算式: γ₁ = E[(X - μ)³] / σ³
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 母歪度の値（正の値は右に歪んだ分布、負の値は左に歪んだ分布）
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 *
 * @note 正規分布の歪度は0です
 */
template <typename Iterator>
double population_skewness(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_skewness: empty range");
    }

    double m = statcpp::mean(first, last);
    double sum_cubed = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - m;
        sum_sq += diff * diff;
        sum_cubed += diff * diff * diff;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_skewness: zero variance");
    }

    double stddev = std::sqrt(var);
    return (sum_cubed / static_cast<double>(n)) / (stddev * stddev * stddev);
}

/**
 * @brief 事前計算済み平均を使う母歪度の計算
 *
 * 既に平均値が計算済みの場合に使用します。計算効率が向上します。
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 母歪度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator>
double population_skewness(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_skewness: empty range");
    }

    double sum_cubed = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - precomputed_mean;
        sum_sq += diff * diff;
        sum_cubed += diff * diff * diff;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_skewness: zero variance");
    }

    double stddev = std::sqrt(var);
    return (sum_cubed / static_cast<double>(n)) / (stddev * stddev * stddev);
}

/**
 * @brief 射影版の母歪度の計算
 *
 * 射影関数を使用して、データから特定の値を抽出して歪度を計算します。
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 母歪度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double population_skewness(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_skewness: empty range");
    }

    double m = statcpp::mean(first, last, proj);
    double sum_cubed = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - m;
        sum_sq += diff * diff;
        sum_cubed += diff * diff * diff;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_skewness: zero variance");
    }

    double stddev = std::sqrt(var);
    return (sum_cubed / static_cast<double>(n)) / (stddev * stddev * stddev);
}

/**
 * @brief 事前計算済み平均を使う射影版の母歪度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 母歪度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator, typename Projection>
double population_skewness(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_skewness: empty range");
    }

    double sum_cubed = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - precomputed_mean;
        sum_sq += diff * diff;
        sum_cubed += diff * diff * diff;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_skewness: zero variance");
    }

    double stddev = std::sqrt(var);
    return (sum_cubed / static_cast<double>(n)) / (stddev * stddev * stddev);
}

/**
 * @brief 標本歪度の計算（バイアス補正版）
 *
 * 標本からの歪度推定値をバイアス補正して計算します。
 *
 * 計算式: G₁ = √(n(n-1)) / (n-2) × g₁
 * ここで g₁ は母歪度の推定量
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 標本歪度の値
 * @throw std::invalid_argument 要素数が3未満の場合
 */
template <typename Iterator>
double sample_skewness(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 3) {
        throw std::invalid_argument("statcpp::sample_skewness: need at least 3 elements");
    }

    double g1 = population_skewness(first, last);
    double correction = std::sqrt(static_cast<double>(n * (n - 1))) / static_cast<double>(n - 2);
    return correction * g1;
}

/**
 * @brief 事前計算済み平均を使う標本歪度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本歪度の値
 * @throw std::invalid_argument 要素数が3未満の場合
 */
template <typename Iterator>
double sample_skewness(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 3) {
        throw std::invalid_argument("statcpp::sample_skewness: need at least 3 elements");
    }

    double g1 = population_skewness(first, last, precomputed_mean);
    double correction = std::sqrt(static_cast<double>(n * (n - 1))) / static_cast<double>(n - 2);
    return correction * g1;
}

/**
 * @brief 射影版の標本歪度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 標本歪度の値
 * @throw std::invalid_argument 要素数が3未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double sample_skewness(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 3) {
        throw std::invalid_argument("statcpp::sample_skewness: need at least 3 elements");
    }

    double g1 = population_skewness(first, last, proj);
    double correction = std::sqrt(static_cast<double>(n * (n - 1))) / static_cast<double>(n - 2);
    return correction * g1;
}

/**
 * @brief 事前計算済み平均を使う射影版の標本歪度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本歪度の値
 * @throw std::invalid_argument 要素数が3未満の場合
 */
template <typename Iterator, typename Projection>
double sample_skewness(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 3) {
        throw std::invalid_argument("statcpp::sample_skewness: need at least 3 elements");
    }

    double g1 = population_skewness(first, last, proj, precomputed_mean);
    double correction = std::sqrt(static_cast<double>(n * (n - 1))) / static_cast<double>(n - 2);
    return correction * g1;
}

/**
 * @brief 歪度の計算（sample_skewness のエイリアス）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 標本歪度の値
 */
template <typename Iterator>
double skewness(Iterator first, Iterator last)
{
    return sample_skewness(first, last);
}

/**
 * @brief 歪度の計算（事前計算済み平均版）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本歪度の値
 */
template <typename Iterator>
double skewness(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_skewness(first, last, precomputed_mean);
}

/**
 * @brief 歪度の計算（射影版）
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 標本歪度の値
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double skewness(Iterator first, Iterator last, Projection proj)
{
    return sample_skewness(first, last, proj);
}

/**
 * @brief 歪度の計算（射影版、事前計算済み平均）
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本歪度の値
 */
template <typename Iterator, typename Projection>
double skewness(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_skewness(first, last, proj, precomputed_mean);
}

// ============================================================================
// Kurtosis (尖度)
// ============================================================================

/**
 * @brief 母尖度の計算（超過尖度: Excess Kurtosis）
 *
 * 母集団の尖度を計算します。分布の裾の重さを示す指標です。
 *
 * 計算式: γ₂ = E[(X - μ)⁴] / σ⁴ - 3
 * 正規分布の尖度が0になるよう -3 します（超過尖度）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 母尖度の値（正の値は裾が重い、負の値は裾が軽い）
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 *
 * @note 正規分布の尖度（超過尖度）は0です
 */
template <typename Iterator>
double population_kurtosis(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_kurtosis: empty range");
    }

    double m = statcpp::mean(first, last);
    double sum_fourth = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - m;
        double diff_sq = diff * diff;
        sum_sq += diff_sq;
        sum_fourth += diff_sq * diff_sq;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_kurtosis: zero variance");
    }

    double fourth_moment = sum_fourth / static_cast<double>(n);
    return (fourth_moment / (var * var)) - 3.0;
}

/**
 * @brief 事前計算済み平均を使う母尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 母尖度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator>
double population_kurtosis(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_kurtosis: empty range");
    }

    double sum_fourth = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - precomputed_mean;
        double diff_sq = diff * diff;
        sum_sq += diff_sq;
        sum_fourth += diff_sq * diff_sq;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_kurtosis: zero variance");
    }

    double fourth_moment = sum_fourth / static_cast<double>(n);
    return (fourth_moment / (var * var)) - 3.0;
}

/**
 * @brief 射影版の母尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 母尖度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double population_kurtosis(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_kurtosis: empty range");
    }

    double m = statcpp::mean(first, last, proj);
    double sum_fourth = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - m;
        double diff_sq = diff * diff;
        sum_sq += diff_sq;
        sum_fourth += diff_sq * diff_sq;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_kurtosis: zero variance");
    }

    double fourth_moment = sum_fourth / static_cast<double>(n);
    return (fourth_moment / (var * var)) - 3.0;
}

/**
 * @brief 事前計算済み平均を使う射影版の母尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 母尖度の値
 * @throw std::invalid_argument 範囲が空の場合、または分散が0の場合
 */
template <typename Iterator, typename Projection>
double population_kurtosis(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::population_kurtosis: empty range");
    }

    double sum_fourth = 0.0;
    double sum_sq = 0.0;

    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(std::invoke(proj, *it)) - precomputed_mean;
        double diff_sq = diff * diff;
        sum_sq += diff_sq;
        sum_fourth += diff_sq * diff_sq;
    }

    double var = sum_sq / static_cast<double>(n);
    if (var == 0.0) {
        throw std::invalid_argument("statcpp::population_kurtosis: zero variance");
    }

    double fourth_moment = sum_fourth / static_cast<double>(n);
    return (fourth_moment / (var * var)) - 3.0;
}

/**
 * @brief 標本尖度の計算（バイアス補正版）
 *
 * 標本からの尖度推定値をバイアス補正して計算します。
 *
 * 計算式: G₂ = ((n+1) × g₂ + 6) × (n-1) / ((n-2)(n-3))
 * ここで g₂ は母尖度の推定量（超過尖度）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 標本尖度の値
 * @throw std::invalid_argument 要素数が4未満の場合
 */
template <typename Iterator>
double sample_kurtosis(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 4) {
        throw std::invalid_argument("statcpp::sample_kurtosis: need at least 4 elements");
    }

    double g2 = population_kurtosis(first, last);
    double nd = static_cast<double>(n);
    double correction = ((nd + 1.0) * g2 + 6.0) * (nd - 1.0) / ((nd - 2.0) * (nd - 3.0));
    return correction;
}

/**
 * @brief 事前計算済み平均を使う標本尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本尖度の値
 * @throw std::invalid_argument 要素数が4未満の場合
 */
template <typename Iterator>
double sample_kurtosis(Iterator first, Iterator last, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 4) {
        throw std::invalid_argument("statcpp::sample_kurtosis: need at least 4 elements");
    }

    double g2 = population_kurtosis(first, last, precomputed_mean);
    double nd = static_cast<double>(n);
    double correction = ((nd + 1.0) * g2 + 6.0) * (nd - 1.0) / ((nd - 2.0) * (nd - 3.0));
    return correction;
}

/**
 * @brief 射影版の標本尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 標本尖度の値
 * @throw std::invalid_argument 要素数が4未満の場合
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double sample_kurtosis(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 4) {
        throw std::invalid_argument("statcpp::sample_kurtosis: need at least 4 elements");
    }

    double g2 = population_kurtosis(first, last, proj);
    double nd = static_cast<double>(n);
    double correction = ((nd + 1.0) * g2 + 6.0) * (nd - 1.0) / ((nd - 2.0) * (nd - 3.0));
    return correction;
}

/**
 * @brief 事前計算済み平均を使う射影版の標本尖度の計算
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本尖度の値
 * @throw std::invalid_argument 要素数が4未満の場合
 */
template <typename Iterator, typename Projection>
double sample_kurtosis(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 4) {
        throw std::invalid_argument("statcpp::sample_kurtosis: need at least 4 elements");
    }

    double g2 = population_kurtosis(first, last, proj, precomputed_mean);
    double nd = static_cast<double>(n);
    double correction = ((nd + 1.0) * g2 + 6.0) * (nd - 1.0) / ((nd - 2.0) * (nd - 3.0));
    return correction;
}

/**
 * @brief 尖度の計算（sample_kurtosis のエイリアス）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @return 標本尖度の値
 */
template <typename Iterator>
double kurtosis(Iterator first, Iterator last)
{
    return sample_kurtosis(first, last);
}

/**
 * @brief 尖度の計算（事前計算済み平均版）
 *
 * @tparam Iterator 入力イテレータ型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本尖度の値
 */
template <typename Iterator>
double kurtosis(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_kurtosis(first, last, precomputed_mean);
}

/**
 * @brief 尖度の計算（射影版）
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @return 標本尖度の値
 */
template <typename Iterator, typename Projection,
          typename = std::enable_if_t<
              std::is_invocable_v<Projection,
                  typename std::iterator_traits<Iterator>::value_type>>>
double kurtosis(Iterator first, Iterator last, Projection proj)
{
    return sample_kurtosis(first, last, proj);
}

/**
 * @brief 尖度の計算（射影版、事前計算済み平均）
 *
 * @tparam Iterator 入力イテレータ型
 * @tparam Projection 射影関数型
 * @param first データ範囲の開始イテレータ
 * @param last データ範囲の終了イテレータ
 * @param proj 射影関数
 * @param precomputed_mean 事前計算済みの平均値
 * @return 標本尖度の値
 */
template <typename Iterator, typename Projection>
double kurtosis(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_kurtosis(first, last, proj, precomputed_mean);
}

} // namespace statcpp
