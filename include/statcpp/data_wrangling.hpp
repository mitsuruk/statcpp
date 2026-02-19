/**
 * @file data_wrangling.hpp
 * @brief Data wrangling (data manipulation and transformation) functions
 *
 * Provides missing value handling, filtering, transformation, grouping, aggregation,
 * sampling, rolling aggregations, categorical encoding, and other functions.
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
 * @brief Constant representing NA (NaN)
 */
inline constexpr double NA = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief Check if a value is NA
 * @param x Value to check
 * @return true if NA, false otherwise
 */
inline bool is_na(double x) {
    return std::isnan(x);
}

/**
 * @brief Drop rows containing NA
 * @tparam T Data type
 * @param data 2-dimensional data
 * @return Data containing only rows without NA
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
 * @brief Drop NA from a 1-dimensional vector
 * @tparam T Data type
 * @param data 1-dimensional data
 * @return Data without NA
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
 * @brief Fill NA with a specified value
 * @tparam T Data type
 * @param data Data vector
 * @param fill_value Value to fill
 * @return Data with NA filled
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
 * @brief Fill NA with mean
 * @param data Data vector
 * @return Data with NA filled with mean
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
        return data;  // Return as-is if all values are NA
    }

    double m = mean(non_na.begin(), non_na.end());
    return fillna(data, m);
}

/**
 * @brief Fill NA with median
 * @param data Data vector
 * @return Data with NA filled with median
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
 * @brief Fill NA with forward fill
 * @param data Data vector
 * @return Data with NA filled with forward values
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
 * @brief Fill NA with backward fill
 * @param data Data vector
 * @return Data with NA filled with backward values
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
 * @brief Fill NA with linear interpolation
 * @param data Data vector
 * @return Data with NA filled with linear interpolation
 */
inline std::vector<double> fillna_interpolate(const std::vector<double>& data)
{
    std::vector<double> result = data;
    std::size_t n = result.size();

    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(result[i])) {
            // Find valid values before and after
            std::size_t prev_idx = i;
            std::size_t next_idx = i;

            // Forward direction
            while (prev_idx > 0 && std::isnan(result[prev_idx])) {
                --prev_idx;
            }
            if (std::isnan(result[prev_idx])) {
                continue;  // No valid value before
            }

            // Backward direction
            while (next_idx < n - 1 && std::isnan(result[next_idx])) {
                ++next_idx;
            }
            if (std::isnan(result[next_idx])) {
                continue;  // No valid value after
            }

            // Linear interpolation
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
 * @brief Filter elements that match a condition
 * @tparam T Data type
 * @tparam Predicate Predicate type
 * @param data Data vector
 * @param pred Predicate function
 * @return Vector of elements satisfying the condition
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
 * @brief Filter rows that match a condition (2-dimensional)
 * @tparam T Data type
 * @tparam Predicate Predicate type
 * @param data 2-dimensional data
 * @param pred Predicate function
 * @return Data containing only rows satisfying the condition
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
 * @brief Filter values within a range
 * @tparam T Data type
 * @param data Data vector
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return Vector of values within the range
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
 * @brief Logarithmic transformation (natural logarithm)
 * @param data Data vector
 * @return Log-transformed data (values <= 0 become NA)
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
 * @brief Logarithmic transformation (log1p: log(1 + x))
 * @param data Data vector
 * @return log1p-transformed data (values < -1 become NA)
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
 * @brief Square root transformation
 * @param data Data vector
 * @return Square root-transformed data (negative values become NA)
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
 * @brief Box-Cox transformation
 *
 * When lambda = 0, performs logarithmic transformation
 *
 * @param data Data vector
 * @param lambda Transformation parameter
 * @return Box-Cox transformed data (values <= 0 become NA)
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
 * @brief Rank transformation
 *
 * Uses average rank for ties
 *
 * @param data Data vector
 * @return Vector of ranks
 */
inline std::vector<double> rank_transform(const std::vector<double>& data)
{
    std::size_t n = data.size();
    if (n == 0) {
        return {};
    }

    // Create pairs of index and value
    std::vector<std::pair<std::size_t, double>> indexed;
    indexed.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed.emplace_back(i, data[i]);
    }

    // Sort by value
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Assign ranks (average rank for ties)
    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        // Find elements with the same value
        while (j < n && indexed[j].second == indexed[i].second) {
            ++j;
        }
        // Compute average rank
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
 * @brief Grouping result
 * @tparam K Key type
 * @tparam V Value type
 */
template <typename K, typename V>
struct group_result {
    std::map<K, std::vector<V>> groups;  ///< Values for each group
};

/**
 * @brief Aggregation result per group
 * @tparam K Key type
 */
template <typename K>
struct aggregation_result {
    std::vector<K> keys;            ///< Vector of keys
    std::vector<double> values;     ///< Vector of aggregated values
};

/**
 * @brief Group by
 * @tparam K Key type
 * @tparam V Value type
 * @param keys Vector of keys
 * @param values Vector of values
 * @return Grouping result
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
 * @brief Mean per group
 * @tparam K Key type
 * @param keys Vector of keys
 * @param values Vector of values
 * @return Mean value for each group
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
 * @brief Sum per group
 * @tparam K Key type
 * @param keys Vector of keys
 * @param values Vector of values
 * @return Sum for each group
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
 * @brief Count per group
 * @tparam K Key type
 * @param keys Vector of keys
 * @param values Vector of values
 * @return Element count for each group
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
 * @brief Return a sorted vector (ascending)
 * @tparam T Data type
 * @param data Data vector
 * @param ascending true for ascending, false for descending
 * @return Sorted vector
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
 * @brief Return indices in sorted order
 * @tparam T Data type
 * @param data Data vector
 * @param ascending true for ascending, false for descending
 * @return Indices in sorted order
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
 * @brief Random sampling (with replacement)
 * @tparam T Data type
 * @param data Data vector
 * @param n Sample size
 * @return Sampled data
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
 * @brief Random sampling (without replacement)
 * @tparam T Data type
 * @param data Data vector
 * @param n Sample size
 * @return Sampled data
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

    // Execute only the first n iterations of Fisher-Yates shuffle
    for (std::size_t i = 0; i < n; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, pool.size() - 1);
        std::swap(pool[i], pool[dist(rng)]);
    }

    return std::vector<T>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(n));
}

/**
 * @brief Stratified sampling
 * @tparam K Stratum type
 * @tparam V Data type
 * @param strata Vector of strata
 * @param data Data vector
 * @param sample_ratio Sampling ratio
 * @return Stratified sampled data
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

    // Group by stratum
    auto groups = group_by(strata, data);

    std::vector<V> result;
    auto& rng = get_random_engine();

    for (auto& pair : groups.groups) {
        std::size_t n = static_cast<std::size_t>(std::ceil(pair.second.size() * sample_ratio));
        n = std::min(n, pair.second.size());

        // Shuffle and take first n elements
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
 * @brief Drop duplicates
 * @tparam T Data type
 * @param data Data vector
 * @return Data with duplicates removed
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
 * @brief Count duplicates
 * @tparam T Data type
 * @param data Data vector
 * @return Map of values and their occurrence counts
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
 * @brief Get duplicate values
 * @tparam T Data type
 * @param data Data vector
 * @return Vector of duplicate values
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
 * @brief Moving average
 * @param data Data vector
 * @param window Window size
 * @return Vector of moving averages
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
 * @brief Moving standard deviation
 * @param data Data vector
 * @param window Window size
 * @return Vector of moving standard deviations
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
 * @brief Moving minimum
 * @param data Data vector
 * @param window Window size
 * @return Vector of moving minimums
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
 * @brief Moving maximum
 * @param data Data vector
 * @param window Window size
 * @return Vector of moving maximums
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
 * @brief Moving sum
 * @param data Data vector
 * @param window Window size
 * @return Vector of moving sums
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
 * @brief Label encoding result
 * @tparam T Data type
 */
template <typename T>
struct label_encoding_result {
    std::vector<std::size_t> encoded;   ///< Encoded values
    std::map<T, std::size_t> mapping;   ///< Mapping from original values to encoded values
    std::vector<T> classes;             ///< List of classes
};

/**
 * @brief Label encoding
 * @tparam T Data type
 * @param data Data vector
 * @return Label encoding result
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
 * @brief One-hot encoding
 * @tparam T Data type
 * @param data Data vector
 * @return One-hot encoded 2-dimensional vector
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
 * @brief Binning (equal width)
 * @param data Data vector
 * @param n_bins Number of bins
 * @return Vector of bin numbers
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
 * @brief Binning (equal frequency)
 * @param data Data vector
 * @param n_bins Number of bins
 * @return Vector of bin numbers
 */
inline std::vector<std::size_t> bin_equal_freq(const std::vector<double>& data, std::size_t n_bins)
{
    if (n_bins == 0) {
        throw std::invalid_argument("statcpp::bin_equal_freq: n_bins must be > 0");
    }
    if (data.empty()) {
        return {};
    }

    // Get sorted indices
    auto sorted_idx = argsort(data);
    std::size_t n = data.size();
    std::size_t bin_size = (n + n_bins - 1) / n_bins;  // Ceiling

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
 * @brief Data validation result
 */
struct validation_result {
    bool is_valid = true;                           ///< Whether data is valid
    std::size_t n_missing = 0;                      ///< Number of missing values
    std::size_t n_infinite = 0;                     ///< Number of infinite values
    std::size_t n_negative = 0;                     ///< Number of negative values
    std::vector<std::size_t> missing_indices;       ///< Indices of missing values
    std::vector<std::size_t> infinite_indices;      ///< Indices of infinite values
    std::vector<std::size_t> negative_indices;      ///< Indices of negative values
};

/**
 * @brief Data validation
 * @param data Data vector
 * @param allow_missing Allow missing values
 * @param allow_infinite Allow infinite values
 * @param allow_negative Allow negative values
 * @return Validation result
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
 * @brief Range validation
 * @param data Data vector
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return true if all values are within range
 */
inline bool validate_range(const std::vector<double>& data,
                          double min_val = -std::numeric_limits<double>::infinity(),
                          double max_val = std::numeric_limits<double>::infinity())
{
    for (double val : data) {
        if (std::isnan(val)) {
            continue;  // NA is not considered out of range
        }
        if (val < min_val || val > max_val) {
            return false;
        }
    }
    return true;
}

}  // namespace statcpp

#endif  // STATCPP_DATA_WRANGLING_HPP
