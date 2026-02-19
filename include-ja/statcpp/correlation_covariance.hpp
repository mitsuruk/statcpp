/**
 * @file correlation_covariance.hpp
 * @brief 相関・共分散の計算関数
 *
 * 共分散、ピアソン相関係数、スピアマン順位相関係数、ケンドールの順位相関係数など、
 * 変数間の関連性を測定する関数を提供します。
 * イテレータベースのインターフェースで、様々なコンテナに対応しています。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "statcpp/basic_statistics.hpp"

namespace statcpp {

// ============================================================================
// Covariance (共分散)
// ============================================================================

/**
 * @brief 母共分散
 *
 * 2つの変数の母共分散を計算します。
 * Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)] = (1/n) Σ(xᵢ - x̄)(yᵢ - ȳ)
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return 母共分散
 * @throws std::invalid_argument 空の範囲または長さが異なる場合
 */
template <typename Iterator1, typename Iterator2>
double population_covariance(Iterator1 first1, Iterator1 last1,
                             Iterator2 first2, Iterator2 last2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::population_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::population_covariance: ranges must have equal length");
    }

    double mean_x = statcpp::mean(first1, last1);
    double mean_y = statcpp::mean(first2, last2);

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(*it1) - mean_x) * (static_cast<double>(*it2) - mean_y);
    }

    return sum / static_cast<double>(n1);
}

/**
 * @brief 事前計算済み平均を使う母共分散
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param mean_x 1番目のデータの平均
 * @param mean_y 2番目のデータの平均
 * @return 母共分散
 * @throws std::invalid_argument 空の範囲または長さが異なる場合
 */
template <typename Iterator1, typename Iterator2>
double population_covariance(Iterator1 first1, Iterator1 last1,
                             Iterator2 first2, Iterator2 last2,
                             double mean_x, double mean_y)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::population_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::population_covariance: ranges must have equal length");
    }

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(*it1) - mean_x) * (static_cast<double>(*it2) - mean_y);
    }

    return sum / static_cast<double>(n1);
}

/**
 * @brief 射影版の母共分散
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return 母共分散
 * @throws std::invalid_argument 空の範囲または長さが異なる場合
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double population_covariance(Iterator1 first1, Iterator1 last1,
                             Iterator2 first2, Iterator2 last2,
                             Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::population_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::population_covariance: ranges must have equal length");
    }

    double mean_x = statcpp::mean(first1, last1, proj1);
    double mean_y = statcpp::mean(first2, last2, proj2);

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(std::invoke(proj1, *it1)) - mean_x)
             * (static_cast<double>(std::invoke(proj2, *it2)) - mean_y);
    }

    return sum / static_cast<double>(n1);
}

/**
 * @brief 標本共分散（不偏共分散）
 *
 * 2つの変数の標本共分散（不偏推定量）を計算します。
 * s_xy = (1/(n-1)) Σ(xᵢ - x̄)(yᵢ - ȳ)
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return 標本共分散
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
double sample_covariance(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::sample_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::sample_covariance: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::sample_covariance: need at least 2 elements");
    }

    double mean_x = statcpp::mean(first1, last1);
    double mean_y = statcpp::mean(first2, last2);

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(*it1) - mean_x) * (static_cast<double>(*it2) - mean_y);
    }

    return sum / static_cast<double>(n1 - 1);
}

/**
 * @brief 事前計算済み平均を使う標本共分散
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param mean_x 1番目のデータの平均
 * @param mean_y 2番目のデータの平均
 * @return 標本共分散
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
double sample_covariance(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2,
                         double mean_x, double mean_y)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::sample_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::sample_covariance: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::sample_covariance: need at least 2 elements");
    }

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(*it1) - mean_x) * (static_cast<double>(*it2) - mean_y);
    }

    return sum / static_cast<double>(n1 - 1);
}

/**
 * @brief 射影版の標本共分散
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return 標本共分散
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double sample_covariance(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2,
                         Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::sample_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::sample_covariance: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::sample_covariance: need at least 2 elements");
    }

    double mean_x = statcpp::mean(first1, last1, proj1);
    double mean_y = statcpp::mean(first2, last2, proj2);

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        sum += (static_cast<double>(std::invoke(proj1, *it1)) - mean_x)
             * (static_cast<double>(std::invoke(proj2, *it2)) - mean_y);
    }

    return sum / static_cast<double>(n1 - 1);
}

/**
 * @brief 共分散（sample_covariance のエイリアス）
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return 標本共分散
 */
template <typename Iterator1, typename Iterator2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2)
{
    return sample_covariance(first1, last1, first2, last2);
}

/**
 * @brief 共分散（事前計算済み平均版、sample_covariance のエイリアス）
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param mean_x 1番目のデータの平均
 * @param mean_y 2番目のデータの平均
 * @return 標本共分散
 */
template <typename Iterator1, typename Iterator2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  double mean_x, double mean_y)
{
    return sample_covariance(first1, last1, first2, last2, mean_x, mean_y);
}

/**
 * @brief 共分散（射影版、sample_covariance のエイリアス）
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return 標本共分散
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  Projection1 proj1, Projection2 proj2)
{
    return sample_covariance(first1, last1, first2, last2, proj1, proj2);
}

// ============================================================================
// Pearson Correlation Coefficient (ピアソン相関係数)
// ============================================================================

/**
 * @brief ピアソン相関係数
 *
 * 2つの変数のピアソン積率相関係数を計算します。
 * r = Cov(X, Y) / (σₓ × σᵧ)
 * 値は -1 から 1 の範囲で、1 は完全な正の相関、-1 は完全な負の相関を示します。
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return ピアソン相関係数（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、要素数が2未満の場合、
 *         または片方の分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
double pearson_correlation(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::pearson_correlation: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::pearson_correlation: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::pearson_correlation: need at least 2 elements");
    }

    double mean_x = statcpp::mean(first1, last1);
    double mean_y = statcpp::mean(first2, last2);

    double sum_xy = 0.0;
    double sum_x_sq = 0.0;
    double sum_y_sq = 0.0;

    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        double dx = static_cast<double>(*it1) - mean_x;
        double dy = static_cast<double>(*it2) - mean_y;
        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }

    if (sum_x_sq == 0.0 || sum_y_sq == 0.0) {
        throw std::invalid_argument("statcpp::pearson_correlation: zero variance in one or both variables");
    }

    return sum_xy / std::sqrt(sum_x_sq * sum_y_sq);
}

/**
 * @brief 事前計算済み平均を使うピアソン相関係数
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param mean_x 1番目のデータの平均
 * @param mean_y 2番目のデータの平均
 * @return ピアソン相関係数（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、要素数が2未満の場合、
 *         または片方の分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
double pearson_correlation(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           double mean_x, double mean_y)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::pearson_correlation: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::pearson_correlation: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::pearson_correlation: need at least 2 elements");
    }

    double sum_xy = 0.0;
    double sum_x_sq = 0.0;
    double sum_y_sq = 0.0;

    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        double dx = static_cast<double>(*it1) - mean_x;
        double dy = static_cast<double>(*it2) - mean_y;
        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }

    if (sum_x_sq == 0.0 || sum_y_sq == 0.0) {
        throw std::invalid_argument("statcpp::pearson_correlation: zero variance in one or both variables");
    }

    return sum_xy / std::sqrt(sum_x_sq * sum_y_sq);
}

/**
 * @brief 射影版のピアソン相関係数
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return ピアソン相関係数（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、要素数が2未満の場合、
 *         または片方の分散が0の場合
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double pearson_correlation(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::pearson_correlation: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::pearson_correlation: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::pearson_correlation: need at least 2 elements");
    }

    double mean_x = statcpp::mean(first1, last1, proj1);
    double mean_y = statcpp::mean(first2, last2, proj2);

    double sum_xy = 0.0;
    double sum_x_sq = 0.0;
    double sum_y_sq = 0.0;

    auto it1 = first1;
    auto it2 = first2;
    for (; it1 != last1; ++it1, ++it2) {
        double dx = static_cast<double>(std::invoke(proj1, *it1)) - mean_x;
        double dy = static_cast<double>(std::invoke(proj2, *it2)) - mean_y;
        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }

    if (sum_x_sq == 0.0 || sum_y_sq == 0.0) {
        throw std::invalid_argument("statcpp::pearson_correlation: zero variance in one or both variables");
    }

    return sum_xy / std::sqrt(sum_x_sq * sum_y_sq);
}

// ============================================================================
// Spearman's Rank Correlation Coefficient (スピアマン順位相関係数)
// ============================================================================

namespace detail {

/**
 * @brief 順位を計算するヘルパー関数
 *
 * 同順位は平均順位を使用します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 順位のベクトル
 */
template <typename Iterator>
std::vector<double> compute_ranks(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    // インデックスと値のペアを作成
    std::vector<std::pair<std::size_t, double>> indexed_values;
    indexed_values.reserve(n);
    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        indexed_values.emplace_back(idx, static_cast<double>(*it));
    }

    // 値でソート
    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // 順位を計算（同順位は平均順位）
    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        // 同じ値を持つ要素の範囲を見つける
        while (j < n && indexed_values[j].second == indexed_values[i].second) {
            ++j;
        }
        // 平均順位を計算（1始まり）
        double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            ranks[indexed_values[k].first] = avg_rank;
        }
        i = j;
    }

    return ranks;
}

/**
 * @brief 射影版の順位計算
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 順位のベクトル
 */
template <typename Iterator, typename Projection>
std::vector<double> compute_ranks(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    std::vector<std::pair<std::size_t, double>> indexed_values;
    indexed_values.reserve(n);
    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        indexed_values.emplace_back(idx, static_cast<double>(std::invoke(proj, *it)));
    }

    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j < n && indexed_values[j].second == indexed_values[i].second) {
            ++j;
        }
        double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            ranks[indexed_values[k].first] = avg_rank;
        }
        i = j;
    }

    return ranks;
}

} // namespace detail

/**
 * @brief スピアマン順位相関係数
 *
 * 2つの変数のスピアマン順位相関係数を計算します。
 * ρ = Pearson(rank(X), rank(Y))
 * ピアソン相関係数を順位データに適用したものです。
 * 単調な関係を検出でき、外れ値に対してロバストです。
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return スピアマン順位相関係数（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
double spearman_correlation(Iterator1 first1, Iterator1 last1,
                            Iterator2 first2, Iterator2 last2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::spearman_correlation: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::spearman_correlation: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::spearman_correlation: need at least 2 elements");
    }

    auto ranks_x = detail::compute_ranks(first1, last1);
    auto ranks_y = detail::compute_ranks(first2, last2);

    return pearson_correlation(ranks_x.begin(), ranks_x.end(),
                               ranks_y.begin(), ranks_y.end());
}

/**
 * @brief 射影版のスピアマン順位相関係数
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return スピアマン順位相関係数（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double spearman_correlation(Iterator1 first1, Iterator1 last1,
                            Iterator2 first2, Iterator2 last2,
                            Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::spearman_correlation: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::spearman_correlation: ranges must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::spearman_correlation: need at least 2 elements");
    }

    auto ranks_x = detail::compute_ranks(first1, last1, proj1);
    auto ranks_y = detail::compute_ranks(first2, last2, proj2);

    return pearson_correlation(ranks_x.begin(), ranks_x.end(),
                               ranks_y.begin(), ranks_y.end());
}

// ============================================================================
// Kendall's Tau (Rank Correlation Coefficient)
// ============================================================================

/**
 * @brief ケンドールの順位相関係数（τ_b）
 *
 * 2つの変数のケンドールの順位相関係数（tau-b）を計算します。
 * 同順位を考慮した版です。
 * τ_b = (concordant - discordant) / sqrt((n0 - tie_x) * (n0 - tie_y))
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @return ケンドールのτ_b（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
double kendall_tau(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::kendall_tau: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::kendall_tau: ranges must have equal length");
    }

    auto n = n1;
    if (n < 2) {
        throw std::invalid_argument("statcpp::kendall_tau: need at least 2 elements");
    }

    // ペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);

    auto it1 = first1;
    auto it2 = first2;
    for (std::size_t i = 0; i < n; ++i, ++it1, ++it2) {
        pairs.push_back({static_cast<double>(*it1), static_cast<double>(*it2)});
    }

    // 一致・不一致・同順位のペア数をカウント
    std::size_t concordant = 0;
    std::size_t discordant = 0;
    std::size_t tie_x = 0;
    std::size_t tie_y = 0;
    std::size_t tie_xy = 0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double x_diff = pairs[i].first - pairs[j].first;
            double y_diff = pairs[i].second - pairs[j].second;

            if (x_diff == 0.0 && y_diff == 0.0) {
                tie_xy++;
            } else if (x_diff == 0.0) {
                tie_x++;
            } else if (y_diff == 0.0) {
                tie_y++;
            } else if ((x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0)) {
                concordant++;
            } else {
                discordant++;
            }
        }
    }

    // Kendall's tau-b
    std::size_t n0 = n * (n - 1) / 2;
    std::size_t tie_count_x = tie_x + tie_xy;
    std::size_t tie_count_y = tie_y + tie_xy;

    if (n0 == tie_count_x || n0 == tie_count_y) {
        // すべて同順位の場合
        return 0.0;
    }

    double numerator = static_cast<double>(concordant) - static_cast<double>(discordant);
    double denominator = std::sqrt(static_cast<double>(n0 - tie_count_x) * static_cast<double>(n0 - tie_count_y));

    return numerator / denominator;
}

/**
 * @brief 射影版のケンドールの順位相関係数
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return ケンドールのτ_b（-1 から 1）
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、または要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double kendall_tau(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2,
                   Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::kendall_tau: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::kendall_tau: ranges must have equal length");
    }

    auto n = n1;
    if (n < 2) {
        throw std::invalid_argument("statcpp::kendall_tau: need at least 2 elements");
    }

    // 射影値のペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);

    auto it1 = first1;
    auto it2 = first2;
    for (std::size_t i = 0; i < n; ++i, ++it1, ++it2) {
        pairs.push_back({static_cast<double>(std::invoke(proj1, *it1)),
                         static_cast<double>(std::invoke(proj2, *it2))});
    }

    // 一致・不一致・同順位のペア数をカウント
    std::size_t concordant = 0;
    std::size_t discordant = 0;
    std::size_t tie_x = 0;
    std::size_t tie_y = 0;
    std::size_t tie_xy = 0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double x_diff = pairs[i].first - pairs[j].first;
            double y_diff = pairs[i].second - pairs[j].second;

            if (x_diff == 0.0 && y_diff == 0.0) {
                tie_xy++;
            } else if (x_diff == 0.0) {
                tie_x++;
            } else if (y_diff == 0.0) {
                tie_y++;
            } else if ((x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0)) {
                concordant++;
            } else {
                discordant++;
            }
        }
    }

    std::size_t n0 = n * (n - 1) / 2;
    std::size_t tie_count_x = tie_x + tie_xy;
    std::size_t tie_count_y = tie_y + tie_xy;

    if (n0 == tie_count_x || n0 == tie_count_y) {
        return 0.0;
    }

    double numerator = static_cast<double>(concordant) - static_cast<double>(discordant);
    double denominator = std::sqrt(static_cast<double>(n0 - tie_count_x) * static_cast<double>(n0 - tie_count_y));

    return numerator / denominator;
}

// ============================================================================
// Weighted Covariance
// ============================================================================

/**
 * @brief 重み付き共分散
 *
 * 重みを考慮した共分散を計算します。Bessel補正を適用しています。
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @return 重み付き共分散
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、負の重み、
 *         または重みの合計が0の場合
 */
template <typename Iterator1, typename Iterator2, typename WeightIterator>
double weighted_covariance(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           WeightIterator weight_first)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::weighted_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::weighted_covariance: ranges must have the same size");
    }

    // 重み付き平均を計算
    double sum_weights = 0.0;
    double sum_wx = 0.0;
    double sum_wy = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    auto weight_it = weight_first;

    for (; it1 != last1; ++it1, ++it2, ++weight_it) {
        double x = static_cast<double>(*it1);
        double y = static_cast<double>(*it2);
        double w = static_cast<double>(*weight_it);

        if (w < 0.0) {
            throw std::invalid_argument("statcpp::weighted_covariance: negative weight");
        }

        sum_weights += w;
        sum_wx += w * x;
        sum_wy += w * y;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_covariance: sum of weights is zero");
    }

    double mean_x = sum_wx / sum_weights;
    double mean_y = sum_wy / sum_weights;

    // 重み付き共分散を計算
    double sum_w_sq = 0.0;
    double cov = 0.0;
    it1 = first1;
    it2 = first2;
    weight_it = weight_first;

    for (; it1 != last1; ++it1, ++it2, ++weight_it) {
        double x = static_cast<double>(*it1);
        double y = static_cast<double>(*it2);
        double w = static_cast<double>(*weight_it);

        cov += w * (x - mean_x) * (y - mean_y);
        sum_w_sq += w * w;
    }

    // Bessel補正
    double correction = sum_weights / (sum_weights * sum_weights - sum_w_sq);
    return cov * correction;
}

/**
 * @brief 重み付き共分散（射影版）
 *
 * @tparam Iterator1 1番目のデータのイテレータ型
 * @tparam Iterator2 2番目のデータのイテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @tparam Projection1 1番目のデータの射影関数型
 * @tparam Projection2 2番目のデータの射影関数型
 * @param first1 1番目のデータの開始イテレータ
 * @param last1 1番目のデータの終了イテレータ
 * @param first2 2番目のデータの開始イテレータ
 * @param last2 2番目のデータの終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @param proj1 1番目のデータの射影関数
 * @param proj2 2番目のデータの射影関数
 * @return 重み付き共分散
 * @throws std::invalid_argument 空の範囲、長さが異なる場合、負の重み、
 *         または重みの合計が0の場合
 */
template <typename Iterator1, typename Iterator2, typename WeightIterator,
          typename Projection1, typename Projection2>
double weighted_covariance(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           WeightIterator weight_first,
                           Projection1 proj1, Projection2 proj2)
{
    auto n1 = static_cast<std::size_t>(std::distance(first1, last1));
    auto n2 = static_cast<std::size_t>(std::distance(first2, last2));

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::weighted_covariance: empty range");
    }
    if (n1 != n2) {
        throw std::invalid_argument("statcpp::weighted_covariance: ranges must have the same size");
    }

    // 重み付き平均を計算
    double sum_weights = 0.0;
    double sum_wx = 0.0;
    double sum_wy = 0.0;
    auto it1 = first1;
    auto it2 = first2;
    auto weight_it = weight_first;

    for (; it1 != last1; ++it1, ++it2, ++weight_it) {
        double x = static_cast<double>(std::invoke(proj1, *it1));
        double y = static_cast<double>(std::invoke(proj2, *it2));
        double w = static_cast<double>(*weight_it);

        if (w < 0.0) {
            throw std::invalid_argument("statcpp::weighted_covariance: negative weight");
        }

        sum_weights += w;
        sum_wx += w * x;
        sum_wy += w * y;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_covariance: sum of weights is zero");
    }

    double mean_x = sum_wx / sum_weights;
    double mean_y = sum_wy / sum_weights;

    // 重み付き共分散を計算
    double sum_w_sq = 0.0;
    double cov = 0.0;
    it1 = first1;
    it2 = first2;
    weight_it = weight_first;

    for (; it1 != last1; ++it1, ++it2, ++weight_it) {
        double x = static_cast<double>(std::invoke(proj1, *it1));
        double y = static_cast<double>(std::invoke(proj2, *it2));
        double w = static_cast<double>(*weight_it);

        cov += w * (x - mean_x) * (y - mean_y);
        sum_w_sq += w * w;
    }

    // Bessel補正
    double correction = sum_weights / (sum_weights * sum_weights - sum_w_sq);
    return cov * correction;
}

} // namespace statcpp
