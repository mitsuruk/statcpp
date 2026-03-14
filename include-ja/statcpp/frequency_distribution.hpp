/**
 * @file frequency_distribution.hpp
 * @brief 度数分布表の作成と度数関連の計算
 *
 * 度数表、相対度数、累積度数などの度数分布に関する関数を提供します。
 * カテゴリデータや離散データの分析に有用です。
 */

#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// 戻り値の構造体
// ============================================================================

/**
 * @brief 度数表のエントリ
 *
 * 各値の度数、相対度数、累積度数、累積相対度数を保持します。
 *
 * @tparam T 値の型
 */
template <typename T>
struct frequency_entry {
    T value;                                ///< 値
    std::size_t count;                      ///< 度数
    double relative_frequency;              ///< 相対度数
    std::size_t cumulative_count;           ///< 累積度数
    double cumulative_relative_frequency;   ///< 累積相対度数
};

/**
 * @brief 度数表の結果
 *
 * 度数表全体の情報を保持します。
 *
 * @tparam T 値の型
 */
template <typename T>
struct frequency_table_result {
    std::vector<frequency_entry<T>> entries;  ///< 度数表のエントリ
    std::size_t total_count;                   ///< 総度数
};

// ============================================================================
// Frequency Table (度数表)
// ============================================================================

/**
 * @brief 度数表を作成（ソート済みのキーで返す）
 *
 * 各値の度数、相対度数、累積度数、累積相対度数を計算します。
 * 結果はキーの昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 度数表の結果
 */
template <typename Iterator>
auto frequency_table(Iterator first, Iterator last)
    -> frequency_table_result<typename std::iterator_traits<Iterator>::value_type>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    if (first == last) {
        return {{}, 0};
    }

    // 度数をカウント（std::map でソート順を保持）
    std::map<value_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[*it];
    }

    std::size_t total = static_cast<std::size_t>(std::distance(first, last));
    double total_d = static_cast<double>(total);

    // 結果を構築
    std::vector<frequency_entry<value_type>> entries;
    entries.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        entries.push_back({
            pair.first,
            pair.second,
            static_cast<double>(pair.second) / total_d,
            cumulative,
            static_cast<double>(cumulative) / total_d
        });
    }

    return {std::move(entries), total};
}

/**
 * @brief 射影版の度数表
 *
 * 各要素に射影関数を適用した結果の度数表を作成します。
 * 結果はキーの昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 度数表の結果
 */
template <typename Iterator, typename Projection>
auto frequency_table(Iterator first, Iterator last, Projection proj)
    -> frequency_table_result<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    if (first == last) {
        return {{}, 0};
    }

    std::map<result_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[std::invoke(proj, *it)];
    }

    std::size_t total = static_cast<std::size_t>(std::distance(first, last));
    double total_d = static_cast<double>(total);

    std::vector<frequency_entry<result_type>> entries;
    entries.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        entries.push_back({
            pair.first,
            pair.second,
            static_cast<double>(pair.second) / total_d,
            cumulative,
            static_cast<double>(cumulative) / total_d
        });
    }

    return {std::move(entries), total};
}

// ============================================================================
// Frequency Count (度数)
// ============================================================================

/**
 * @brief 各値の度数を返す（unordered_map 版、高速）
 *
 * 各値の出現回数をカウントします。
 * ソート順は保証されませんが、高速に動作します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 値と度数のマップ
 */
template <typename Iterator>
auto frequency_count(Iterator first, Iterator last)
    -> std::unordered_map<typename std::iterator_traits<Iterator>::value_type, std::size_t>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    std::unordered_map<value_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[*it];
    }
    return freq;
}

/**
 * @brief 射影版の度数
 *
 * 各要素に射影関数を適用した結果の度数を返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 値と度数のマップ
 */
template <typename Iterator, typename Projection>
auto frequency_count(Iterator first, Iterator last, Projection proj)
    -> std::unordered_map<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, std::size_t>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    std::unordered_map<result_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[std::invoke(proj, *it)];
    }
    return freq;
}

// ============================================================================
// Relative Frequency (相対度数)
// ============================================================================

/**
 * @brief 各値の相対度数を返す
 *
 * 各値の度数を全体の要素数で割った相対度数を計算します。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return 値と相対度数のマップ
 */
template <typename Iterator>
auto relative_frequency(Iterator first, Iterator last)
    -> std::unordered_map<typename std::iterator_traits<Iterator>::value_type, double>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto freq = frequency_count(first, last);
    double total = static_cast<double>(n);

    std::unordered_map<value_type, double> result;
    result.reserve(freq.size());
    for (const auto& pair : freq) {
        result[pair.first] = static_cast<double>(pair.second) / total;
    }
    return result;
}

/**
 * @brief 射影版の相対度数
 *
 * 各要素に射影関数を適用した結果の相対度数を返します。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return 値と相対度数のマップ
 */
template <typename Iterator, typename Projection>
auto relative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::unordered_map<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, double>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto freq = frequency_count(first, last, proj);
    double total = static_cast<double>(n);

    std::unordered_map<result_type, double> result;
    result.reserve(freq.size());
    for (const auto& pair : freq) {
        result[pair.first] = static_cast<double>(pair.second) / total;
    }
    return result;
}

// ============================================================================
// Cumulative Frequency (累積度数)
// ============================================================================

/**
 * @brief 累積度数を返す（ソート済みの値に対して）
 *
 * 各値までの累積度数を計算します。
 * 結果は値の昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return (値, 累積度数) のペアのベクトル
 */
template <typename Iterator>
auto cumulative_frequency(Iterator first, Iterator last)
    -> std::vector<std::pair<typename std::iterator_traits<Iterator>::value_type, std::size_t>>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    if (first == last) {
        return {};
    }

    // ソートされた順序で度数を取得
    std::map<value_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[*it];
    }

    std::vector<std::pair<value_type, std::size_t>> result;
    result.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        result.emplace_back(pair.first, cumulative);
    }

    return result;
}

/**
 * @brief 射影版の累積度数
 *
 * 各要素に射影関数を適用した結果の累積度数を返します。
 * 結果は値の昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return (値, 累積度数) のペアのベクトル
 */
template <typename Iterator, typename Projection>
auto cumulative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::vector<std::pair<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, std::size_t>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    if (first == last) {
        return {};
    }

    std::map<result_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[std::invoke(proj, *it)];
    }

    std::vector<std::pair<result_type, std::size_t>> result;
    result.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        result.emplace_back(pair.first, cumulative);
    }

    return result;
}

// ============================================================================
// Cumulative Relative Frequency (累積相対度数)
// ============================================================================

/**
 * @brief 累積相対度数を返す（ソート済みの値に対して）
 *
 * 各値までの累積相対度数を計算します。
 * 結果は値の昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @return (値, 累積相対度数) のペアのベクトル
 */
template <typename Iterator>
auto cumulative_relative_frequency(Iterator first, Iterator last)
    -> std::vector<std::pair<typename std::iterator_traits<Iterator>::value_type, double>>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto cum_freq = cumulative_frequency(first, last);
    double total = static_cast<double>(n);

    std::vector<std::pair<value_type, double>> result;
    result.reserve(cum_freq.size());

    for (const auto& pair : cum_freq) {
        result.emplace_back(pair.first, static_cast<double>(pair.second) / total);
    }

    return result;
}

/**
 * @brief 射影版の累積相対度数
 *
 * 各要素に射影関数を適用した結果の累積相対度数を返します。
 * 結果は値の昇順でソートされます。
 *
 * @tparam Iterator イテレータ型
 * @tparam Projection 射影関数型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param proj 射影関数
 * @return (値, 累積相対度数) のペアのベクトル
 */
template <typename Iterator, typename Projection>
auto cumulative_relative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::vector<std::pair<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, double>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto cum_freq = cumulative_frequency(first, last, proj);
    double total = static_cast<double>(n);

    std::vector<std::pair<result_type, double>> result;
    result.reserve(cum_freq.size());

    for (const auto& pair : cum_freq) {
        result.emplace_back(pair.first, static_cast<double>(pair.second) / total);
    }

    return result;
}

} // namespace statcpp
