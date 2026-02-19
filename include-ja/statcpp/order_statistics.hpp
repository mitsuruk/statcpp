/**
 * @file order_statistics.hpp
 * @brief 順序統計量の実装 (Order statistics implementation)
 *
 * 最小値、最大値、四分位数、パーセンタイル、重み付き中央値などを提供します。
 * Provides minimum, maximum, quartiles, percentiles, weighted median, and more.
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

namespace statcpp {

// ============================================================================
// 戻り値の構造体 (Result Structures)
// ============================================================================

/**
 * @brief 四分位数の結果 (Quartile result)
 */
struct quartile_result {
    double q1;  ///< 第1四分位数 (first quartile, 25th percentile)
    double q2;  ///< 第2四分位数（中央値）(second quartile, median, 50th percentile)
    double q3;  ///< 第3四分位数 (third quartile, 75th percentile)
};

/**
 * @brief 五数要約の結果 (Five-number summary result)
 */
struct five_number_summary_result {
    double min;     ///< 最小値 (minimum)
    double q1;      ///< 第1四分位数 (first quartile)
    double median;  ///< 中央値 (median)
    double q3;      ///< 第3四分位数 (third quartile)
    double max;     ///< 最大値 (maximum)
};

// ============================================================================
// 線形補間ヘルパー（R type=7 / Excel QUARTILE.INC 相当）
// Linear Interpolation Helper (R type=7 / Excel QUARTILE.INC equivalent)
// ============================================================================

/**
 * @brief 線形補間による位置の計算 (Linear interpolation at position)
 *
 * 位置 p × (n-1) を整数部 lo と小数部 frac に分解し、線形補間します。
 * Decomposes position p × (n-1) into integer part lo and fractional part frac,
 * then performs linear interpolation.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param n 要素数 (number of elements)
 * @param p 位置（0.0〜1.0）(position, 0.0 to 1.0)
 * @return 補間された値 (interpolated value)
 *
 * @note ソート済み範囲を前提とします。(Assumes sorted range)
 */
template <typename Iterator>
double interpolate_at(Iterator first, std::size_t n, double p)
{
    double index = p * static_cast<double>(n - 1);
    auto lo = static_cast<std::size_t>(std::floor(index));
    double frac = index - static_cast<double>(lo);
    if (lo + 1 >= n) {
        return static_cast<double>(*(first + lo));
    }
    return static_cast<double>(*(first + lo)) * (1.0 - frac)
         + static_cast<double>(*(first + lo + 1)) * frac;
}

/**
 * @brief 線形補間による位置の計算（射影版）(Linear interpolation with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param n 要素数 (number of elements)
 * @param p 位置（0.0〜1.0）(position, 0.0 to 1.0)
 * @param proj 射影関数 (projection function)
 * @return 補間された値 (interpolated value)
 */
template <typename Iterator, typename Projection>
double interpolate_at(Iterator first, std::size_t n, double p, Projection proj)
{
    double index = p * static_cast<double>(n - 1);
    auto lo = static_cast<std::size_t>(std::floor(index));
    double frac = index - static_cast<double>(lo);
    if (lo + 1 >= n) {
        return static_cast<double>(std::invoke(proj, *(first + lo)));
    }
    return static_cast<double>(std::invoke(proj, *(first + lo))) * (1.0 - frac)
         + static_cast<double>(std::invoke(proj, *(first + lo + 1))) * frac;
}

// ============================================================================
// Minimum
// ============================================================================

/**
 * @brief 最小値を返す (Return minimum value)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @return 最小値 (minimum value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator>
auto minimum(Iterator first, Iterator last)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::minimum: empty range");
    }
    return *std::min_element(first, last);
}

/**
 * @brief 最小値を返す（射影版）(Return minimum value with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param proj 射影関数 (projection function)
 * @return 最小値 (minimum value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator, typename Projection>
auto minimum(Iterator first, Iterator last, Projection proj)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::minimum: empty range");
    }
    auto min_it = first;
    auto min_val = std::invoke(proj, *first);
    for (auto it = std::next(first); it != last; ++it) {
        auto val = std::invoke(proj, *it);
        if (val < min_val) {
            min_val = val;
            min_it = it;
        }
    }
    return min_val;
}

// ============================================================================
// Maximum
// ============================================================================

/**
 * @brief 最大値を返す (Return maximum value)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @return 最大値 (maximum value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator>
auto maximum(Iterator first, Iterator last)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::maximum: empty range");
    }
    return *std::max_element(first, last);
}

/**
 * @brief 最大値を返す（射影版）(Return maximum value with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param proj 射影関数 (projection function)
 * @return 最大値 (maximum value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator, typename Projection>
auto maximum(Iterator first, Iterator last, Projection proj)
{
    if (first == last) {
        throw std::invalid_argument("statcpp::maximum: empty range");
    }
    auto max_val = std::invoke(proj, *first);
    for (auto it = std::next(first); it != last; ++it) {
        auto val = std::invoke(proj, *it);
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}

// ============================================================================
// Quartiles (Q1, Q2, Q3)
// ============================================================================

/**
 * @brief 四分位数を返す (Return quartiles)
 *
 * ソート済み範囲から四分位数（Q1, Q2, Q3）を計算します。
 * Calculates quartiles (Q1, Q2, Q3) from sorted range.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @return 四分位数 (quartiles)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note R の type=7（デフォルト）または Excel の QUARTILE.INC と同等です。
 * Equivalent to R's type=7 (default) or Excel's QUARTILE.INC.
 */
template <typename Iterator>
quartile_result quartiles(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::quartiles: empty range");
    }
    return {
        interpolate_at(first, n, 0.25),
        interpolate_at(first, n, 0.50),
        interpolate_at(first, n, 0.75)
    };
}

/**
 * @brief 四分位数を返す（射影版）(Return quartiles with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @param proj 射影関数 (projection function)
 * @return 四分位数 (quartiles)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator, typename Projection>
quartile_result quartiles(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::quartiles: empty range");
    }
    return {
        interpolate_at(first, n, 0.25, proj),
        interpolate_at(first, n, 0.50, proj),
        interpolate_at(first, n, 0.75, proj)
    };
}

// ============================================================================
// Percentile
// ============================================================================

/**
 * @brief パーセンタイルを返す (Return percentile)
 *
 * ソート済み範囲から指定したパーセンタイルを計算します。
 * Calculates specified percentile from sorted range.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @param p パーセンタイル（0.0〜1.0の割合で指定、例: 90パーセンタイル → p = 0.9）
 *          (percentile as proportion 0.0-1.0, e.g., 90th percentile → p = 0.9)
 * @return パーセンタイル値 (percentile value)
 * @throws std::invalid_argument 範囲が空、または p が範囲外の場合
 *         (if range is empty or p is out of range)
 */
template <typename Iterator>
double percentile(Iterator first, Iterator last, double p)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::percentile: empty range");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::percentile: p must be in [0, 1]");
    }
    return interpolate_at(first, n, p);
}

/**
 * @brief パーセンタイルを返す（射影版）(Return percentile with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @param p パーセンタイル (percentile as proportion 0.0-1.0)
 * @param proj 射影関数 (projection function)
 * @return パーセンタイル値 (percentile value)
 * @throws std::invalid_argument 範囲が空、または p が範囲外の場合
 *         (if range is empty or p is out of range)
 */
template <typename Iterator, typename Projection>
double percentile(Iterator first, Iterator last, double p, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::percentile: empty range");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::percentile: p must be in [0, 1]");
    }
    return interpolate_at(first, n, p, proj);
}

// ============================================================================
// Five-Number Summary
// ============================================================================

/**
 * @brief 五数要約を返す (Return five-number summary)
 *
 * ソート済み範囲から五数要約（最小値、Q1、中央値、Q3、最大値）を計算します。
 * Calculates five-number summary (min, Q1, median, Q3, max) from sorted range.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @return 五数要約 (five-number summary)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note 箱ひげ図の描画に必要な基本統計量を提供します。
 * Provides basic statistics needed for box plot visualization.
 */
template <typename Iterator>
five_number_summary_result five_number_summary(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::five_number_summary: empty range");
    }
    return {
        static_cast<double>(*first),
        interpolate_at(first, n, 0.25),
        interpolate_at(first, n, 0.50),
        interpolate_at(first, n, 0.75),
        static_cast<double>(*(first + (n - 1)))
    };
}

/**
 * @brief 五数要約を返す（射影版）(Return five-number summary with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of sorted range)
 * @param last 範囲の終了 (end of sorted range)
 * @param proj 射影関数 (projection function)
 * @return 五数要約 (five-number summary)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 */
template <typename Iterator, typename Projection>
five_number_summary_result five_number_summary(Iterator first, Iterator last, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::five_number_summary: empty range");
    }
    return {
        static_cast<double>(std::invoke(proj, *first)),
        interpolate_at(first, n, 0.25, proj),
        interpolate_at(first, n, 0.50, proj),
        interpolate_at(first, n, 0.75, proj),
        static_cast<double>(std::invoke(proj, *(first + (n - 1))))
    };
}

// ============================================================================
// Weighted Median
// ============================================================================

/**
 * @brief 重み付き中央値 (Weighted median)
 *
 * 重み付き中央値を計算します。
 * Calculates the weighted median.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam WeightIterator 重みイテレータ型 (weight iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param weight_first 重みの開始 (beginning of weights)
 * @return 重み付き中央値 (weighted median)
 * @throws std::invalid_argument 範囲が空、重みが負、または重みの総和が0の場合
 *         (if range is empty, weight is negative, or sum of weights is zero)
 */
template <typename Iterator, typename WeightIterator>
double weighted_median(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_median: empty range");
    }

    // 値と重みのペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);
    auto weight_it = weight_first;
    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);
        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_median: negative weight");
        }
        pairs.emplace_back(value, weight);
    }

    // 値でソート
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 重みの総和
    double total_weight = 0.0;
    for (const auto& p : pairs) {
        total_weight += p.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_median: sum of weights is zero");
    }

    // 累積重みを計算して中央値を見つける
    double cumulative = 0.0;
    double half_weight = total_weight / 2.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= half_weight) {
            // 累積重みがちょうど半分の場合は次の値との平均を取る
            if (cumulative == half_weight && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

/**
 * @brief 重み付き中央値（射影版）(Weighted median with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam WeightIterator 重みイテレータ型 (weight iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param weight_first 重みの開始 (beginning of weights)
 * @param proj 射影関数 (projection function)
 * @return 重み付き中央値 (weighted median)
 * @throws std::invalid_argument 範囲が空、重みが負、または重みの総和が0の場合
 *         (if range is empty, weight is negative, or sum of weights is zero)
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_median(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_median: empty range");
    }

    // 値と重みのペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);
    auto weight_it = weight_first;
    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);
        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_median: negative weight");
        }
        pairs.emplace_back(value, weight);
    }

    // 値でソート
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 重みの総和
    double total_weight = 0.0;
    for (const auto& p : pairs) {
        total_weight += p.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_median: sum of weights is zero");
    }

    // 累積重みを計算して中央値を見つける
    double cumulative = 0.0;
    double half_weight = total_weight / 2.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= half_weight) {
            // 累積重みがちょうど半分の場合は次の値との平均を取る
            if (cumulative == half_weight && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

// ============================================================================
// Weighted Percentile
// ============================================================================

/**
 * @brief 重み付きパーセンタイル (Weighted percentile)
 *
 * 重み付きパーセンタイルを計算します。
 * Calculates the weighted percentile.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam WeightIterator 重みイテレータ型 (weight iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param weight_first 重みの開始 (beginning of weights)
 * @param p パーセンタイル（0.0〜1.0）(percentile as proportion 0.0-1.0)
 * @return 重み付きパーセンタイル値 (weighted percentile value)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
template <typename Iterator, typename WeightIterator>
double weighted_percentile(Iterator first, Iterator last, WeightIterator weight_first, double p)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: p must be in [0, 1]");
    }

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_percentile: empty range");
    }

    // 値と重みのペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);
    auto weight_it = weight_first;
    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);
        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_percentile: negative weight");
        }
        pairs.emplace_back(value, weight);
    }

    // 値でソート
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 重みの総和
    double total_weight = 0.0;
    for (const auto& pair : pairs) {
        total_weight += pair.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: sum of weights is zero");
    }

    // 累積重みを計算して目標パーセンタイルを見つける
    // p=0.5の場合、総重みの50%の位置を見つける
    double target = p * total_weight;
    double cumulative = 0.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= target) {
            // 累積重みがちょうど目標値の場合は次の値との平均を取る
            if (cumulative == target && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

/**
 * @brief 重み付きパーセンタイル（射影版）(Weighted percentile with projection)
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @tparam WeightIterator 重みイテレータ型 (weight iterator type)
 * @tparam Projection 射影関数型 (projection function type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param weight_first 重みの開始 (beginning of weights)
 * @param p パーセンタイル（0.0〜1.0）(percentile as proportion 0.0-1.0)
 * @param proj 射影関数 (projection function)
 * @return 重み付きパーセンタイル値 (weighted percentile value)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_percentile(Iterator first, Iterator last, WeightIterator weight_first, double p, Projection proj)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: p must be in [0, 1]");
    }

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_percentile: empty range");
    }

    // 値と重みのペアを作成
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);
    auto weight_it = weight_first;
    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);
        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_percentile: negative weight");
        }
        pairs.emplace_back(value, weight);
    }

    // 値でソート
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 重みの総和
    double total_weight = 0.0;
    for (const auto& pair : pairs) {
        total_weight += pair.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: sum of weights is zero");
    }

    // 累積重みを計算して目標パーセンタイルを見つける
    // p=0.5の場合、総重みの50%の位置を見つける
    double target = p * total_weight;
    double cumulative = 0.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= target) {
            // 累積重みがちょうど目標値の場合は次の値との平均を取る
            if (cumulative == target && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

} // namespace statcpp
