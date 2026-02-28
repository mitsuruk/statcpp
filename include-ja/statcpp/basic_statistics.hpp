/**
 * @file basic_statistics.hpp
 * @brief 基本統計量の計算関数
 *
 * 平均、中央値、最頻値などの基本的な記述統計量を計算する関数を提供します。
 * イテレータベースのインターフェースで、様々なコンテナに対応しています。
 *
 * @note NaN の扱い: statcpp の関数は IEEE 754 の NaN 伝播セマンティクスに
 *       従います。入力データに NaN が含まれる場合、結果は通常 NaN になります。
 *       計算前に NaN を除外するには、data_wrangling.hpp の remove_na() を
 *       使用してください。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace statcpp {

// ============================================================================
// Sum
// ============================================================================

/**
 * @brief 合計
 *
 * 範囲内のすべての要素の合計を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 合計値
 */
template <typename Iterator>
auto sum(Iterator first, Iterator last)
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    return std::accumulate(first, last, value_type{});
}

/**
 * @brief ラムダ式で射影した値の合計
 *
 * 各要素に射影関数を適用した結果の合計を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の合計値
 */
template <typename Iterator, typename Projection>
auto sum(Iterator first, Iterator last, Projection proj)
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;
    result_type total{};
    for (auto it = first; it != last; ++it) {
        total += std::invoke(proj, *it);
    }
    return total;
}

// ============================================================================
// Count
// ============================================================================

/**
 * @brief データ数
 *
 * 範囲内の要素数を返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 要素数
 */
template <typename Iterator>
std::size_t count(Iterator first, Iterator last)
{
    return static_cast<std::size_t>(std::distance(first, last));
}

// ============================================================================
// Mean (Arithmetic Mean)
// ============================================================================

/**
 * @brief 算術平均
 *
 * 範囲内の要素の算術平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 算術平均
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double mean(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean: empty range");
    }
    return static_cast<double>(statcpp::sum(first, last)) / static_cast<double>(n);
}

/**
 * @brief ラムダ式で射影した値の算術平均
 *
 * 各要素に射影関数を適用した結果の算術平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の算術平均
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double mean(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::mean: empty range");
    }
    return static_cast<double>(statcpp::sum(first, last, proj)) / static_cast<double>(n);
}

// ============================================================================
// Median
// ============================================================================

/**
 * @brief 中央値（ソート済みの範囲を受け取る）
 *
 * ソート済みの範囲の中央値を計算します。
 * 要素数が偶数の場合は中央2値の平均を返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 中央値
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
double median(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::median: empty range");
    }

    auto mid = n / 2;
    if (n % 2 == 0) {
        return (static_cast<double>(*(first + (mid - 1))) + static_cast<double>(*(first + mid))) / 2.0;
    }
    return static_cast<double>(*(first + mid));
}

/**
 * @brief ラムダ式で射影した値の中央値（射影結果がソート済みの順序で並んでいること）
 *
 * 各要素に射影関数を適用した結果の中央値を計算します。
 * 射影結果がソート済みであることを前提とします。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の中央値
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
double median(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::median: empty range");
    }

    auto mid = n / 2;
    if (n % 2 == 0) {
        return (static_cast<double>(std::invoke(proj, *(first + (mid - 1))))
              + static_cast<double>(std::invoke(proj, *(first + mid)))) / 2.0;
    }
    return static_cast<double>(std::invoke(proj, *(first + mid)));
}

// ============================================================================
// Mode
// ============================================================================

/**
 * @brief 最頻値（複数ある場合は最小の値を返す：決定論的動作を保証）
 *
 * 範囲内で最も頻度の高い値を返します。
 * 最頻値が複数ある場合は、最小の値を返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 最頻値
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
auto mode(Iterator first, Iterator last)
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    if (first == last) {
        throw std::invalid_argument("statcpp::mode: empty range");
    }

    std::map<value_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[*it];
    }

    auto best = freq.begin();
    for (auto it = freq.begin(); it != freq.end(); ++it) {
        if (it->second > best->second) {
            best = it;
        }
    }
    return best->first;
}

/**
 * @brief ラムダ式で射影した値の最頻値
 *
 * 各要素に射影関数を適用した結果の最頻値を返します。
 * 最頻値が複数ある場合は、最小の値を返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の最頻値
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
auto mode(Iterator first, Iterator last, Projection proj)
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;
    if (first == last) {
        throw std::invalid_argument("statcpp::mode: empty range");
    }

    std::map<result_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[std::invoke(proj, *it)];
    }

    auto best = freq.begin();
    for (auto it = freq.begin(); it != freq.end(); ++it) {
        if (it->second > best->second) {
            best = it;
        }
    }
    return best->first;
}

// ============================================================================
// Modes (複数の最頻値)
// ============================================================================

/**
 * @brief 最頻値をすべて返す（昇順でソートされたvectorを返す）
 *
 * 範囲内で最も頻度の高い値をすべて返します。
 * 複数の最頻値がある場合、すべて昇順のvectorで返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 最頻値のvector（昇順）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
auto modes(Iterator first, Iterator last)
    -> std::vector<typename std::iterator_traits<Iterator>::value_type>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    if (first == last) {
        throw std::invalid_argument("statcpp::modes: empty range");
    }

    std::map<value_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[*it];
    }

    // 最大頻度を求める
    std::size_t max_freq = 0;
    for (const auto& pair : freq) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
        }
    }

    // 最大頻度を持つすべての値を収集（mapなので昇順）
    std::vector<value_type> result;
    for (const auto& pair : freq) {
        if (pair.second == max_freq) {
            result.push_back(pair.first);
        }
    }

    return result;
}

/**
 * @brief ラムダ式で射影した値の最頻値をすべて返す
 *
 * 各要素に射影関数を適用した結果の最頻値をすべて返します。
 * 複数の最頻値がある場合、すべて昇順のvectorで返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の最頻値のvector（昇順）
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
auto modes(Iterator first, Iterator last, Projection proj)
    -> std::vector<std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;
    if (first == last) {
        throw std::invalid_argument("statcpp::modes: empty range");
    }

    std::map<result_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[std::invoke(proj, *it)];
    }

    // 最大頻度を求める
    std::size_t max_freq = 0;
    for (const auto& pair : freq) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
        }
    }

    // 最大頻度を持つすべての値を収集（mapなので昇順）
    std::vector<result_type> result;
    for (const auto& pair : freq) {
        if (pair.second == max_freq) {
            result.push_back(pair.first);
        }
    }

    return result;
}

// ============================================================================
// Geometric Mean
// ============================================================================

/**
 * @brief 幾何平均
 *
 * 範囲内の要素の幾何平均を計算します。
 * すべての値は正である必要があります。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 幾何平均
 * @throws std::invalid_argument 空の範囲の場合、または値が0以下の場合
 */
template <typename Iterator>
double geometric_mean(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_mean: empty range");
    }
    double log_sum = 0.0;
    for (auto it = first; it != last; ++it) {
        if (static_cast<double>(*it) <= 0.0) {
            throw std::invalid_argument("statcpp::geometric_mean: all values must be positive");
        }
        log_sum += std::log(static_cast<double>(*it));
    }
    return std::exp(log_sum / static_cast<double>(n));
}

/**
 * @brief ラムダ式で射影した値の幾何平均
 *
 * 各要素に射影関数を適用した結果の幾何平均を計算します。
 * すべての値は正である必要があります。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の幾何平均
 * @throws std::invalid_argument 空の範囲の場合、または値が0以下の場合
 */
template <typename Iterator, typename Projection>
double geometric_mean(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_mean: empty range");
    }
    double log_sum = 0.0;
    for (auto it = first; it != last; ++it) {
        double val = static_cast<double>(std::invoke(proj, *it));
        if (val <= 0.0) {
            throw std::invalid_argument("statcpp::geometric_mean: all values must be positive");
        }
        log_sum += std::log(val);
    }
    return std::exp(log_sum / static_cast<double>(n));
}

// ============================================================================
// Harmonic Mean
// ============================================================================

/**
 * @brief 調和平均
 *
 * 範囲内の要素の調和平均を計算します。
 * すべての値は0でない必要があります。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 調和平均
 * @throws std::invalid_argument 空の範囲の場合、または値が0の場合
 */
template <typename Iterator>
double harmonic_mean(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::harmonic_mean: empty range");
    }
    double reciprocal_sum = 0.0;
    for (auto it = first; it != last; ++it) {
        double val = static_cast<double>(*it);
        // 厳密なゼロだけでなく、1/val が無限大になるほど小さい値も拒否する。
        // std::numeric_limits<double>::min() は最小の正規化 double（約 2.2e-308）なので、
        // 非正規化数やゼロを検出しつつ、通常の小さな値は通過させる。
        if (std::abs(val) < std::numeric_limits<double>::min()) {
            throw std::invalid_argument("statcpp::harmonic_mean: zero or near-zero value encountered");
        }
        reciprocal_sum += 1.0 / val;
    }
    return static_cast<double>(n) / reciprocal_sum;
}

/**
 * @brief ラムダ式で射影した値の調和平均
 *
 * 各要素に射影関数を適用した結果の調和平均を計算します。
 * すべての値は0でない必要があります。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 射影後の調和平均
 * @throws std::invalid_argument 空の範囲の場合、または値が0の場合
 */
template <typename Iterator, typename Projection>
double harmonic_mean(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::harmonic_mean: empty range");
    }
    double reciprocal_sum = 0.0;
    for (auto it = first; it != last; ++it) {
        double val = static_cast<double>(std::invoke(proj, *it));
        if (std::abs(val) < std::numeric_limits<double>::min()) {
            throw std::invalid_argument("statcpp::harmonic_mean: zero or near-zero value encountered");
        }
        reciprocal_sum += 1.0 / val;
    }
    return static_cast<double>(n) / reciprocal_sum;
}

// ============================================================================
// Trimmed Mean
// ============================================================================

/**
 * @brief トリム平均（ソート済みの範囲を受け取る。proportion: 片側の除外割合, 0.0〜0.5未満）
 *
 * 両端から一定割合のデータを除外した平均を計算します。
 * 外れ値の影響を軽減するのに有用です。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proportion 片側の除外割合（0.0〜0.5未満）
 * @return トリム平均
 * @throws std::invalid_argument 空の範囲の場合、proportionが範囲外の場合、またはすべての要素が除外される場合
 */
template <typename Iterator>
double trimmed_mean(Iterator first, Iterator last, double proportion)
{
    if (proportion < 0.0 || proportion >= 0.5) {
        throw std::invalid_argument("statcpp::trimmed_mean: proportion must be in [0.0, 0.5)");
    }
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::trimmed_mean: empty range");
    }

    auto trim_count = static_cast<std::size_t>(static_cast<double>(n) * proportion);
    if (n - 2 * trim_count == 0) {
        throw std::invalid_argument("statcpp::trimmed_mean: all elements trimmed");
    }

    double total = 0.0;
    for (std::size_t i = trim_count; i < n - trim_count; ++i) {
        total += static_cast<double>(*(first + i));
    }
    return total / static_cast<double>(n - 2 * trim_count);
}

/**
 * @brief ラムダ式で射影した値のトリム平均（射影結果がソート済みの順序で並んでいること）
 *
 * 各要素に射影関数を適用した結果のトリム平均を計算します。
 * 射影結果がソート済みであることを前提とします。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proportion 片側の除外割合（0.0〜0.5未満）
 * @param proj 射影関数
 * @return 射影後のトリム平均
 * @throws std::invalid_argument 空の範囲の場合、proportionが範囲外の場合、またはすべての要素が除外される場合
 */
template <typename Iterator, typename Projection>
double trimmed_mean(Iterator first, Iterator last, double proportion, Projection proj)
{
    if (proportion < 0.0 || proportion >= 0.5) {
        throw std::invalid_argument("statcpp::trimmed_mean: proportion must be in [0.0, 0.5)");
    }
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::trimmed_mean: empty range");
    }

    auto trim_count = static_cast<std::size_t>(static_cast<double>(n) * proportion);
    if (n - 2 * trim_count == 0) {
        throw std::invalid_argument("statcpp::trimmed_mean: all elements trimmed");
    }

    double total = 0.0;
    for (std::size_t i = trim_count; i < n - trim_count; ++i) {
        total += static_cast<double>(std::invoke(proj, *(first + i)));
    }
    return total / static_cast<double>(n - 2 * trim_count);
}

// ============================================================================
// Weighted Mean
// ============================================================================

/**
 * @brief 重み付き平均
 *
 * 各要素に重みを適用した平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @return 重み付き平均
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、または重みの合計が0の場合
 */
template <typename Iterator, typename WeightIterator>
double weighted_mean(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_mean: empty range");
    }

    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_mean: negative weight");
        }

        sum_weighted += value * weight;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_mean: sum of weights is zero");
    }

    return sum_weighted / sum_weights;
}

/**
 * @brief 重み付き平均（射影版）
 *
 * 各要素に射影関数を適用した結果の重み付き平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @param proj 射影関数
 * @return 射影後の重み付き平均
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、または重みの合計が0の場合
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_mean(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_mean: empty range");
    }

    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_mean: negative weight");
        }

        sum_weighted += value * weight;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_mean: sum of weights is zero");
    }

    return sum_weighted / sum_weights;
}

// ============================================================================
// Logarithmic Mean
// ============================================================================

/**
 * @brief 対数平均 (Logarithmic Mean)
 *
 * 2つの正の値の対数平均を計算します。
 * LM(a, b) = (b - a) / (ln(b) - ln(a)) for a ≠ b
 * LM(a, a) = a
 *
 * @tparam T1 第1引数の型
 * @tparam T2 第2引数の型
 * @param a 第1の値
 * @param b 第2の値
 * @return 対数平均
 * @throws std::invalid_argument 引数が正でない場合
 */
template <typename T1, typename T2>
double logarithmic_mean(T1 a, T2 b)
{
    double x = static_cast<double>(a);
    double y = static_cast<double>(b);

    if (x <= 0.0 || y <= 0.0) {
        throw std::invalid_argument("statcpp::logarithmic_mean: arguments must be positive");
    }

    // 値がほぼ等しい場合は x をそのまま返す。
    // 絶対差分ではなく相対差分で判定することで、x = 1e15, y = 1e15 + 1e5 のような
    // 大きな値どうしで閾値を誤って超えてしまう問題を防ぐ。
    if (std::abs(x - y) <= 1e-10 * std::max(x, y)) {
        return x;
    }

    // 通常の対数平均: (y - x) / (ln(y) - ln(x))
    return (y - x) / (std::log(y) - std::log(x));
}

// ============================================================================
// Weighted Harmonic Mean
// ============================================================================

/**
 * @brief 重み付き調和平均
 *
 * 各要素に重みを適用した調和平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @return 重み付き調和平均
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、値が0の場合、または重みの合計が0の場合
 */
template <typename Iterator, typename WeightIterator>
double weighted_harmonic_mean(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_harmonic_mean: empty range");
    }

    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(*it);
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_harmonic_mean: negative weight");
        }

        if (value == 0.0) {
            throw std::invalid_argument("statcpp::weighted_harmonic_mean: zero value");
        }

        sum_weighted += weight / value;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_harmonic_mean: sum of weights is zero");
    }

    return sum_weights / sum_weighted;
}

/**
 * @brief 重み付き調和平均（射影版）
 *
 * 各要素に射影関数を適用した結果の重み付き調和平均を計算します。
 *
 * @tparam Iterator イテレータ型
 * @tparam WeightIterator 重みのイテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param weight_first 重みの開始イテレータ
 * @param proj 射影関数
 * @return 射影後の重み付き調和平均
 * @throws std::invalid_argument 空の範囲の場合、負の重みがある場合、値が0の場合、または重みの合計が0の場合
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_harmonic_mean(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_harmonic_mean: empty range");
    }

    double sum_weighted = 0.0;
    double sum_weights = 0.0;
    auto weight_it = weight_first;

    for (auto it = first; it != last; ++it, ++weight_it) {
        double value = static_cast<double>(std::invoke(proj, *it));
        double weight = static_cast<double>(*weight_it);

        if (weight < 0.0) {
            throw std::invalid_argument("statcpp::weighted_harmonic_mean: negative weight");
        }

        if (value == 0.0) {
            throw std::invalid_argument("statcpp::weighted_harmonic_mean: zero value");
        }

        sum_weighted += weight / value;
        sum_weights += weight;
    }

    if (sum_weights == 0.0) {
        throw std::invalid_argument("statcpp::weighted_harmonic_mean: sum of weights is zero");
    }

    return sum_weights / sum_weighted;
}

// ============================================================================
// Argmin / Argmax
// ============================================================================

/**
 * @brief 最小値のインデックスを返す
 *
 * 範囲内で最小値を持つ要素のインデックスを返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 最小値のインデックス
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
std::size_t argmin(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::argmin: empty range");
    }

    auto min_it = std::min_element(first, last);
    return static_cast<std::size_t>(std::distance(first, min_it));
}

/**
 * @brief 最小値のインデックスを返す（射影版）
 *
 * 射影関数を適用した結果が最小となる要素のインデックスを返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 最小値のインデックス
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
std::size_t argmin(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::argmin: empty range");
    }

    auto min_it = std::min_element(first, last,
        [&proj](const auto& a, const auto& b) {
            return std::invoke(proj, a) < std::invoke(proj, b);
        });

    return static_cast<std::size_t>(std::distance(first, min_it));
}

/**
 * @brief 最大値のインデックスを返す
 *
 * 範囲内で最大値を持つ要素のインデックスを返します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 最大値のインデックス
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
std::size_t argmax(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::argmax: empty range");
    }

    auto max_it = std::max_element(first, last);
    return static_cast<std::size_t>(std::distance(first, max_it));
}

/**
 * @brief 最大値のインデックスを返す（射影版）
 *
 * 射影関数を適用した結果が最大となる要素のインデックスを返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 最大値のインデックス
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator, typename Projection>
std::size_t argmax(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::argmax: empty range");
    }

    auto max_it = std::max_element(first, last,
        [&proj](const auto& a, const auto& b) {
            return std::invoke(proj, a) < std::invoke(proj, b);
        });

    return static_cast<std::size_t>(std::distance(first, max_it));
}

} // namespace statcpp
