/**
 * @file data_wrangling.hpp
 * @brief データラングリング（データ整形・変換）機能
 *
 * 欠損値処理、フィルタリング、変換、グループ化、集約、サンプリング、
 * ローリング集計、カテゴリカルエンコーディングなどの機能を提供します。
 */

#ifndef STATCPP_DATA_WRANGLING_HPP
#define STATCPP_DATA_WRANGLING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/random_engine.hpp"

namespace statcpp {

// ============================================================================
// Missing Data Handling
// ============================================================================

/**
 * @brief NAを表す定数（NaN）
 */
inline constexpr double NA = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief 値がNAかどうかを判定
 * @param x 判定する値
 * @return NAの場合true、それ以外false
 */
inline bool is_na(double x) {
    return std::isnan(x);
}

/**
 * @brief NAを含む行を削除
 * @tparam T データ型
 * @param data 2次元データ
 * @return NAを含まない行のみのデータ
 */
template <typename T>
std::vector<std::vector<T>> dropna(const std::vector<std::vector<T>>& data)
{
    std::vector<std::vector<T>> result;
    result.reserve(data.size());

    for (const auto& row : data) {
        bool has_na = false;
        for (const auto& val : row) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(static_cast<double>(val))) {
                    has_na = true;
                    break;
                }
            }
        }
        if (!has_na) {
            result.push_back(row);
        }
    }
    return result;
}

/**
 * @brief 1次元ベクトルからNAを削除
 * @tparam T データ型
 * @param data 1次元データ
 * @return NAを含まないデータ
 */
template <typename T>
std::vector<T> dropna(const std::vector<T>& data)
{
    std::vector<T> result;
    result.reserve(data.size());

    for (const auto& val : data) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isnan(static_cast<double>(val))) {
                result.push_back(val);
            }
        } else {
            result.push_back(val);
        }
    }
    return result;
}

/**
 * @brief NAを指定値で埋める
 * @tparam T データ型
 * @param data データベクトル
 * @param fill_value 埋める値
 * @return NAが埋められたデータ
 */
template <typename T>
std::vector<T> fillna(const std::vector<T>& data, T fill_value)
{
    std::vector<T> result = data;
    for (auto& val : result) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(static_cast<double>(val))) {
                val = fill_value;
            }
        }
    }
    return result;
}

/**
 * @brief NAを平均値で埋める
 * @param data データベクトル
 * @return NAが平均値で埋められたデータ
 */
inline std::vector<double> fillna_mean(const std::vector<double>& data)
{
    std::vector<double> non_na;
    non_na.reserve(data.size());

    for (double val : data) {
        if (!std::isnan(val)) {
            non_na.push_back(val);
        }
    }

    if (non_na.empty()) {
        return data;  // すべてNAの場合はそのまま返す
    }

    double m = mean(non_na.begin(), non_na.end());
    return fillna(data, m);
}

/**
 * @brief NAを中央値で埋める
 * @param data データベクトル
 * @return NAが中央値で埋められたデータ
 */
inline std::vector<double> fillna_median(const std::vector<double>& data)
{
    std::vector<double> non_na;
    non_na.reserve(data.size());

    for (double val : data) {
        if (!std::isnan(val)) {
            non_na.push_back(val);
        }
    }

    if (non_na.empty()) {
        return data;
    }

    double med = median(non_na.begin(), non_na.end());
    return fillna(data, med);
}

/**
 * @brief NAを前方向の値で埋める (forward fill)
 * @param data データベクトル
 * @return NAが前方向の値で埋められたデータ
 */
inline std::vector<double> fillna_ffill(const std::vector<double>& data)
{
    std::vector<double> result = data;
    double last_valid = NA;

    for (auto& val : result) {
        if (!std::isnan(val)) {
            last_valid = val;
        } else if (!std::isnan(last_valid)) {
            val = last_valid;
        }
    }
    return result;
}

/**
 * @brief NAを後方向の値で埋める (backward fill)
 * @param data データベクトル
 * @return NAが後方向の値で埋められたデータ
 */
inline std::vector<double> fillna_bfill(const std::vector<double>& data)
{
    std::vector<double> result = data;
    double next_valid = NA;

    for (auto it = result.rbegin(); it != result.rend(); ++it) {
        if (!std::isnan(*it)) {
            next_valid = *it;
        } else if (!std::isnan(next_valid)) {
            *it = next_valid;
        }
    }
    return result;
}

/**
 * @brief NAを線形補間で埋める
 * @param data データベクトル
 * @return NAが線形補間で埋められたデータ
 */
inline std::vector<double> fillna_interpolate(const std::vector<double>& data)
{
    std::vector<double> result = data;
    std::size_t n = result.size();

    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(result[i])) {
            // 前後の有効な値を探す
            std::size_t prev_idx = i;
            std::size_t next_idx = i;

            // 前方向
            while (prev_idx > 0 && std::isnan(result[prev_idx])) {
                --prev_idx;
            }
            if (std::isnan(result[prev_idx])) {
                continue;  // 前方に有効な値がない
            }

            // 後方向
            while (next_idx < n - 1 && std::isnan(result[next_idx])) {
                ++next_idx;
            }
            if (std::isnan(result[next_idx])) {
                continue;  // 後方に有効な値がない
            }

            // 線形補間
            double prev_val = result[prev_idx];
            double next_val = result[next_idx];
            double ratio = static_cast<double>(i - prev_idx) / static_cast<double>(next_idx - prev_idx);
            result[i] = prev_val + ratio * (next_val - prev_val);
        }
    }
    return result;
}

// ============================================================================
// Filtering
// ============================================================================

/**
 * @brief 条件に合致する要素をフィルタリング
 * @tparam T データ型
 * @tparam Predicate 述語型
 * @param data データベクトル
 * @param pred 述語関数
 * @return 条件を満たす要素のみのベクトル
 */
template <typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& data, Predicate pred)
{
    std::vector<T> result;
    result.reserve(data.size());

    for (const auto& val : data) {
        if (pred(val)) {
            result.push_back(val);
        }
    }
    return result;
}

/**
 * @brief 条件に合致する行をフィルタリング（2次元）
 * @tparam T データ型
 * @tparam Predicate 述語型
 * @param data 2次元データ
 * @param pred 述語関数
 * @return 条件を満たす行のみのデータ
 */
template <typename T, typename Predicate>
std::vector<std::vector<T>> filter_rows(const std::vector<std::vector<T>>& data, Predicate pred)
{
    std::vector<std::vector<T>> result;
    result.reserve(data.size());

    for (const auto& row : data) {
        if (pred(row)) {
            result.push_back(row);
        }
    }
    return result;
}

/**
 * @brief 範囲内の値をフィルタリング
 * @tparam T データ型
 * @param data データベクトル
 * @param min_val 最小値
 * @param max_val 最大値
 * @return 範囲内の値のみのベクトル
 */
template <typename T>
std::vector<T> filter_range(const std::vector<T>& data, T min_val, T max_val)
{
    return filter(data, [min_val, max_val](const T& val) {
        return val >= min_val && val <= max_val;
    });
}

// ============================================================================
// Transformations
// ============================================================================

/**
 * @brief 対数変換（自然対数）
 * @param data データベクトル
 * @return 対数変換されたデータ（0以下の値はNA）
 */
inline std::vector<double> log_transform(const std::vector<double>& data)
{
    std::vector<double> result;
    result.reserve(data.size());

    for (double val : data) {
        if (val <= 0.0) {
            result.push_back(NA);
        } else {
            result.push_back(std::log(val));
        }
    }
    return result;
}

/**
 * @brief 対数変換（log1p: log(1 + x)）
 * @param data データベクトル
 * @return log1p変換されたデータ（-1未満の値はNA）
 */
inline std::vector<double> log1p_transform(const std::vector<double>& data)
{
    std::vector<double> result;
    result.reserve(data.size());

    for (double val : data) {
        if (val < -1.0) {
            result.push_back(NA);
        } else {
            result.push_back(std::log1p(val));
        }
    }
    return result;
}

/**
 * @brief 平方根変換
 * @param data データベクトル
 * @return 平方根変換されたデータ（負の値はNA）
 */
inline std::vector<double> sqrt_transform(const std::vector<double>& data)
{
    std::vector<double> result;
    result.reserve(data.size());

    for (double val : data) {
        if (val < 0.0) {
            result.push_back(NA);
        } else {
            result.push_back(std::sqrt(val));
        }
    }
    return result;
}

/**
 * @brief Box-Cox変換
 *
 * lambda = 0 の場合は対数変換
 *
 * @param data データベクトル
 * @param lambda 変換パラメータ
 * @return Box-Cox変換されたデータ（0以下の値はNA）
 */
inline std::vector<double> boxcox_transform(const std::vector<double>& data, double lambda)
{
    std::vector<double> result;
    result.reserve(data.size());

    for (double val : data) {
        if (val <= 0.0) {
            result.push_back(NA);
        } else if (std::abs(lambda) < 1e-10) {
            result.push_back(std::log(val));
        } else {
            result.push_back((std::pow(val, lambda) - 1.0) / lambda);
        }
    }
    return result;
}

/**
 * @brief 順位変換
 *
 * タイの場合は平均順位を使用
 *
 * @param data データベクトル
 * @return 順位のベクトル
 */
inline std::vector<double> rank_transform(const std::vector<double>& data)
{
    std::size_t n = data.size();
    if (n == 0) {
        return {};
    }

    // インデックスと値のペアを作成
    std::vector<std::pair<std::size_t, double>> indexed;
    indexed.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed.emplace_back(i, data[i]);
    }

    // 値でソート
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // 順位を割り当て（タイの場合は平均順位）
    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        // 同じ値を持つ要素を見つける
        while (j < n && indexed[j].second == indexed[i].second) {
            ++j;
        }
        // 平均順位を計算
        double avg_rank = (static_cast<double>(i) + static_cast<double>(j) - 1.0) / 2.0 + 1.0;
        for (std::size_t k = i; k < j; ++k) {
            ranks[indexed[k].first] = avg_rank;
        }
        i = j;
    }
    return ranks;
}

// ============================================================================
// Group-by and Aggregation
// ============================================================================

/**
 * @brief グループ化結果
 * @tparam K キーの型
 * @tparam V 値の型
 */
template <typename K, typename V>
struct group_result {
    std::map<K, std::vector<V>> groups;  ///< グループごとの値
};

/**
 * @brief グループごとの集約結果
 * @tparam K キーの型
 */
template <typename K>
struct aggregation_result {
    std::vector<K> keys;            ///< キーのベクトル
    std::vector<double> values;     ///< 集約された値のベクトル
};

/**
 * @brief グループ化
 * @tparam K キーの型
 * @tparam V 値の型
 * @param keys キーのベクトル
 * @param values 値のベクトル
 * @return グループ化結果
 */
template <typename K, typename V>
group_result<K, V> group_by(const std::vector<K>& keys, const std::vector<V>& values)
{
    if (keys.size() != values.size()) {
        throw std::invalid_argument("statcpp::group_by: keys and values must have same size");
    }

    group_result<K, V> result;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        result.groups[keys[i]].push_back(values[i]);
    }
    return result;
}

/**
 * @brief グループごとの平均
 * @tparam K キーの型
 * @param keys キーのベクトル
 * @param values 値のベクトル
 * @return グループごとの平均値
 */
template <typename K>
aggregation_result<K> group_mean(const std::vector<K>& keys, const std::vector<double>& values)
{
    auto groups = group_by(keys, values);
    aggregation_result<K> result;

    for (const auto& pair : groups.groups) {
        result.keys.push_back(pair.first);
        result.values.push_back(mean(pair.second.begin(), pair.second.end()));
    }
    return result;
}

/**
 * @brief グループごとの合計
 * @tparam K キーの型
 * @param keys キーのベクトル
 * @param values 値のベクトル
 * @return グループごとの合計値
 */
template <typename K>
aggregation_result<K> group_sum(const std::vector<K>& keys, const std::vector<double>& values)
{
    auto groups = group_by(keys, values);
    aggregation_result<K> result;

    for (const auto& pair : groups.groups) {
        result.keys.push_back(pair.first);
        result.values.push_back(sum(pair.second.begin(), pair.second.end()));
    }
    return result;
}

/**
 * @brief グループごとのカウント
 * @tparam K キーの型
 * @param keys キーのベクトル
 * @param values 値のベクトル
 * @return グループごとの要素数
 */
template <typename K>
aggregation_result<K> group_count(const std::vector<K>& keys, const std::vector<double>& values)
{
    auto groups = group_by(keys, values);
    aggregation_result<K> result;

    for (const auto& pair : groups.groups) {
        result.keys.push_back(pair.first);
        result.values.push_back(static_cast<double>(pair.second.size()));
    }
    return result;
}

// ============================================================================
// Sorting
// ============================================================================

/**
 * @brief ソートされたベクトルを返す（昇順）
 * @tparam T データ型
 * @param data データベクトル
 * @param ascending trueで昇順、falseで降順
 * @return ソートされたベクトル
 */
template <typename T>
std::vector<T> sort_values(const std::vector<T>& data, bool ascending = true)
{
    std::vector<T> result = data;
    if (ascending) {
        std::sort(result.begin(), result.end());
    } else {
        std::sort(result.begin(), result.end(), std::greater<T>());
    }
    return result;
}

/**
 * @brief ソートされた順序のインデックスを返す
 * @tparam T データ型
 * @param data データベクトル
 * @param ascending trueで昇順、falseで降順
 * @return ソート順のインデックス
 */
template <typename T>
std::vector<std::size_t> argsort(const std::vector<T>& data, bool ascending = true)
{
    std::vector<std::size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (ascending) {
        std::sort(indices.begin(), indices.end(),
                  [&data](std::size_t i, std::size_t j) { return data[i] < data[j]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                  [&data](std::size_t i, std::size_t j) { return data[i] > data[j]; });
    }
    return indices;
}

// ============================================================================
// Sampling
// ============================================================================

/**
 * @brief ランダムサンプリング（復元抽出）
 * @tparam T データ型
 * @param data データベクトル
 * @param n サンプル数
 * @return サンプリングされたデータ
 */
template <typename T>
std::vector<T> sample_with_replacement(const std::vector<T>& data, std::size_t n)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::sample_with_replacement: empty data");
    }

    std::vector<T> result;
    result.reserve(n);

    auto& rng = get_random_engine();
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);

    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(data[dist(rng)]);
    }
    return result;
}

/**
 * @brief ランダムサンプリング（非復元抽出）
 * @tparam T データ型
 * @param data データベクトル
 * @param n サンプル数
 * @return サンプリングされたデータ
 */
template <typename T>
std::vector<T> sample_without_replacement(const std::vector<T>& data, std::size_t n)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::sample_without_replacement: empty data");
    }
    if (n > data.size()) {
        throw std::invalid_argument("statcpp::sample_without_replacement: n > data.size()");
    }

    std::vector<T> pool = data;
    auto& rng = get_random_engine();

    // Fisher-Yates shuffle の最初の n 回だけ実行
    for (std::size_t i = 0; i < n; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, pool.size() - 1);
        std::swap(pool[i], pool[dist(rng)]);
    }

    return std::vector<T>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(n));
}

/**
 * @brief 層化サンプリング
 * @tparam K 層の型
 * @tparam V データ型
 * @param strata 層のベクトル
 * @param data データベクトル
 * @param sample_ratio サンプリング比率
 * @return 層化サンプリングされたデータ
 */
template <typename K, typename V>
std::vector<V> stratified_sample(const std::vector<K>& strata,
                                  const std::vector<V>& data,
                                  double sample_ratio)
{
    if (strata.size() != data.size()) {
        throw std::invalid_argument("statcpp::stratified_sample: strata and data must have same size");
    }
    if (sample_ratio <= 0.0 || sample_ratio > 1.0) {
        throw std::invalid_argument("statcpp::stratified_sample: sample_ratio must be in (0, 1]");
    }

    // 層ごとにグループ化
    auto groups = group_by(strata, data);

    std::vector<V> result;
    auto& rng = get_random_engine();

    for (auto& pair : groups.groups) {
        std::size_t n = static_cast<std::size_t>(std::ceil(pair.second.size() * sample_ratio));
        n = std::min(n, pair.second.size());

        // シャッフルして最初のn個を取得
        std::shuffle(pair.second.begin(), pair.second.end(), rng);
        for (std::size_t i = 0; i < n; ++i) {
            result.push_back(pair.second[i]);
        }
    }
    return result;
}

// ============================================================================
// Duplicate Handling
// ============================================================================

/**
 * @brief 重複を削除
 * @tparam T データ型
 * @param data データベクトル
 * @return 重複が削除されたデータ
 */
template <typename T>
std::vector<T> drop_duplicates(const std::vector<T>& data)
{
    std::vector<T> result;
    std::unordered_set<T> seen;

    for (const auto& val : data) {
        if (seen.find(val) == seen.end()) {
            result.push_back(val);
            seen.insert(val);
        }
    }
    return result;
}

/**
 * @brief 重複をカウント
 * @tparam T データ型
 * @param data データベクトル
 * @return 値とその出現回数のマップ
 */
template <typename T>
std::map<T, std::size_t> value_counts(const std::vector<T>& data)
{
    std::map<T, std::size_t> counts;
    for (const auto& val : data) {
        ++counts[val];
    }
    return counts;
}

/**
 * @brief 重複している値を取得
 * @tparam T データ型
 * @param data データベクトル
 * @return 重複している値のベクトル
 */
template <typename T>
std::vector<T> get_duplicates(const std::vector<T>& data)
{
    std::unordered_map<T, std::size_t> counts;
    for (const auto& val : data) {
        ++counts[val];
    }

    std::vector<T> result;
    std::unordered_set<T> added;
    for (const auto& pair : counts) {
        if (pair.second > 1 && added.find(pair.first) == added.end()) {
            result.push_back(pair.first);
            added.insert(pair.first);
        }
    }
    return result;
}

// ============================================================================
// Rolling Aggregations
// ============================================================================

/**
 * @brief 移動平均
 * @param data データベクトル
 * @param window ウィンドウサイズ
 * @return 移動平均のベクトル
 */
inline std::vector<double> rolling_mean(const std::vector<double>& data, std::size_t window)
{
    if (window == 0 || window > data.size()) {
        throw std::invalid_argument("statcpp::rolling_mean: invalid window size");
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    double sum = 0.0;
    for (std::size_t i = 0; i < window; ++i) {
        sum += data[i];
    }
    result.push_back(sum / static_cast<double>(window));

    for (std::size_t i = window; i < data.size(); ++i) {
        sum += data[i] - data[i - window];
        result.push_back(sum / static_cast<double>(window));
    }
    return result;
}

/**
 * @brief 移動標準偏差
 * @param data データベクトル
 * @param window ウィンドウサイズ
 * @return 移動標準偏差のベクトル
 */
inline std::vector<double> rolling_std(const std::vector<double>& data, std::size_t window)
{
    if (window < 2 || window > data.size()) {
        throw std::invalid_argument("statcpp::rolling_std: invalid window size");
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (std::size_t i = 0; i <= data.size() - window; ++i) {
        auto start = data.begin() + static_cast<std::ptrdiff_t>(i);
        auto end = start + static_cast<std::ptrdiff_t>(window);
        double m = mean(start, end);
        double var = 0.0;
        for (auto it = start; it != end; ++it) {
            double diff = *it - m;
            var += diff * diff;
        }
        result.push_back(std::sqrt(var / static_cast<double>(window - 1)));
    }
    return result;
}

/**
 * @brief 移動最小値
 * @param data データベクトル
 * @param window ウィンドウサイズ
 * @return 移動最小値のベクトル
 */
inline std::vector<double> rolling_min(const std::vector<double>& data, std::size_t window)
{
    if (window == 0 || window > data.size()) {
        throw std::invalid_argument("statcpp::rolling_min: invalid window size");
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (std::size_t i = 0; i <= data.size() - window; ++i) {
        auto start = data.begin() + static_cast<std::ptrdiff_t>(i);
        auto end = start + static_cast<std::ptrdiff_t>(window);
        result.push_back(*std::min_element(start, end));
    }
    return result;
}

/**
 * @brief 移動最大値
 * @param data データベクトル
 * @param window ウィンドウサイズ
 * @return 移動最大値のベクトル
 */
inline std::vector<double> rolling_max(const std::vector<double>& data, std::size_t window)
{
    if (window == 0 || window > data.size()) {
        throw std::invalid_argument("statcpp::rolling_max: invalid window size");
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (std::size_t i = 0; i <= data.size() - window; ++i) {
        auto start = data.begin() + static_cast<std::ptrdiff_t>(i);
        auto end = start + static_cast<std::ptrdiff_t>(window);
        result.push_back(*std::max_element(start, end));
    }
    return result;
}

/**
 * @brief 移動合計
 * @param data データベクトル
 * @param window ウィンドウサイズ
 * @return 移動合計のベクトル
 */
inline std::vector<double> rolling_sum(const std::vector<double>& data, std::size_t window)
{
    if (window == 0 || window > data.size()) {
        throw std::invalid_argument("statcpp::rolling_sum: invalid window size");
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    double s = 0.0;
    for (std::size_t i = 0; i < window; ++i) {
        s += data[i];
    }
    result.push_back(s);

    for (std::size_t i = window; i < data.size(); ++i) {
        s += data[i] - data[i - window];
        result.push_back(s);
    }
    return result;
}

// ============================================================================
// Categorical Encoding
// ============================================================================

/**
 * @brief ラベルエンコーディング結果
 * @tparam T データ型
 */
template <typename T>
struct label_encoding_result {
    std::vector<std::size_t> encoded;   ///< エンコードされた値
    std::map<T, std::size_t> mapping;   ///< 元の値からエンコード値へのマッピング
    std::vector<T> classes;             ///< クラスのリスト
};

/**
 * @brief ラベルエンコーディング
 * @tparam T データ型
 * @param data データベクトル
 * @return ラベルエンコーディング結果
 */
template <typename T>
label_encoding_result<T> label_encode(const std::vector<T>& data)
{
    label_encoding_result<T> result;
    std::map<T, std::size_t> mapping;
    std::vector<T> classes;

    result.encoded.reserve(data.size());

    for (const auto& val : data) {
        auto it = mapping.find(val);
        if (it == mapping.end()) {
            std::size_t idx = classes.size();
            mapping[val] = idx;
            classes.push_back(val);
            result.encoded.push_back(idx);
        } else {
            result.encoded.push_back(it->second);
        }
    }

    result.mapping = std::move(mapping);
    result.classes = std::move(classes);
    return result;
}

/**
 * @brief ワンホットエンコーディング
 * @tparam T データ型
 * @param data データベクトル
 * @return ワンホットエンコードされた2次元ベクトル
 */
template <typename T>
std::vector<std::vector<double>> one_hot_encode(const std::vector<T>& data)
{
    auto label_result = label_encode(data);
    std::size_t n_classes = label_result.classes.size();
    std::size_t n = data.size();

    std::vector<std::vector<double>> result(n, std::vector<double>(n_classes, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        result[i][label_result.encoded[i]] = 1.0;
    }
    return result;
}

/**
 * @brief ビニング（等幅）
 * @param data データベクトル
 * @param n_bins ビン数
 * @return ビン番号のベクトル
 */
inline std::vector<std::size_t> bin_equal_width(const std::vector<double>& data, std::size_t n_bins)
{
    if (n_bins == 0) {
        throw std::invalid_argument("statcpp::bin_equal_width: n_bins must be > 0");
    }
    if (data.empty()) {
        return {};
    }

    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());

    if (min_val == max_val) {
        return std::vector<std::size_t>(data.size(), 0);
    }

    double bin_width = (max_val - min_val) / static_cast<double>(n_bins);

    std::vector<std::size_t> result;
    result.reserve(data.size());

    for (double val : data) {
        auto bin = static_cast<std::size_t>((val - min_val) / bin_width);
        if (bin >= n_bins) {
            bin = n_bins - 1;
        }
        result.push_back(bin);
    }
    return result;
}

/**
 * @brief ビニング（等頻度）
 * @param data データベクトル
 * @param n_bins ビン数
 * @return ビン番号のベクトル
 */
inline std::vector<std::size_t> bin_equal_freq(const std::vector<double>& data, std::size_t n_bins)
{
    if (n_bins == 0) {
        throw std::invalid_argument("statcpp::bin_equal_freq: n_bins must be > 0");
    }
    if (data.empty()) {
        return {};
    }

    // ソートされたインデックスを取得
    auto sorted_idx = argsort(data);
    std::size_t n = data.size();
    std::size_t bin_size = (n + n_bins - 1) / n_bins;  // 切り上げ

    std::vector<std::size_t> result(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t bin = i / bin_size;
        if (bin >= n_bins) {
            bin = n_bins - 1;
        }
        result[sorted_idx[i]] = bin;
    }
    return result;
}

// ============================================================================
// Data Validation
// ============================================================================

/**
 * @brief データ検証結果
 */
struct validation_result {
    bool is_valid = true;                           ///< データが有効かどうか
    std::size_t n_missing = 0;                      ///< 欠損値の数
    std::size_t n_infinite = 0;                     ///< 無限大の数
    std::size_t n_negative = 0;                     ///< 負の値の数
    std::vector<std::size_t> missing_indices;       ///< 欠損値のインデックス
    std::vector<std::size_t> infinite_indices;      ///< 無限大のインデックス
    std::vector<std::size_t> negative_indices;      ///< 負の値のインデックス
};

/**
 * @brief データ検証
 * @param data データベクトル
 * @param allow_missing 欠損値を許可するか
 * @param allow_infinite 無限大を許可するか
 * @param allow_negative 負の値を許可するか
 * @return 検証結果
 */
inline validation_result validate_data(const std::vector<double>& data,
                                       bool allow_missing = false,
                                       bool allow_infinite = false,
                                       bool allow_negative = true)
{
    validation_result result;

    for (std::size_t i = 0; i < data.size(); ++i) {
        double val = data[i];

        if (std::isnan(val)) {
            ++result.n_missing;
            result.missing_indices.push_back(i);
            if (!allow_missing) {
                result.is_valid = false;
            }
        } else if (std::isinf(val)) {
            ++result.n_infinite;
            result.infinite_indices.push_back(i);
            if (!allow_infinite) {
                result.is_valid = false;
            }
        } else if (val < 0.0) {
            ++result.n_negative;
            result.negative_indices.push_back(i);
            if (!allow_negative) {
                result.is_valid = false;
            }
        }
    }
    return result;
}

/**
 * @brief 範囲検証
 * @param data データベクトル
 * @param min_val 最小値
 * @param max_val 最大値
 * @return すべての値が範囲内の場合true
 */
inline bool validate_range(const std::vector<double>& data,
                          double min_val = -std::numeric_limits<double>::infinity(),
                          double max_val = std::numeric_limits<double>::infinity())
{
    for (double val : data) {
        if (std::isnan(val)) {
            continue;  // NAは範囲外とみなさない
        }
        if (val < min_val || val > max_val) {
            return false;
        }
    }
    return true;
}

}  // namespace statcpp

#endif  // STATCPP_DATA_WRANGLING_HPP
