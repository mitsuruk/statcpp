/**
 * @file basic_statistics.hpp
 * @brief Basic statistical computation functions
 *
 * Provides functions to compute basic descriptive statistics such as
 * mean, median, mode, etc.
 * Uses iterator-based interface compatible with various containers.
 *
 * @note NaN handling: Functions in statcpp follow IEEE 754 NaN propagation
 *       semantics. If input data contains NaN values, the result will
 *       typically be NaN. To exclude NaN values before computation,
 *       use remove_na() from data_wrangling.hpp to filter input data.
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
 * @brief Sum
 *
 * Computes the sum of all elements in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Sum value
 */
template <typename Iterator>
auto sum(Iterator first, Iterator last)
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    return std::accumulate(first, last, value_type{});
}

/**
 * @brief Sum of projected values using a lambda expression
 *
 * Computes the sum of the results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Sum of projected values
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
 * @brief Data count
 *
 * Returns the number of elements in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Number of elements
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
 * @brief Arithmetic mean
 *
 * Computes the arithmetic mean of elements in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Arithmetic mean
 * @throws std::invalid_argument If range is empty
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
 * @brief Arithmetic mean of projected values using a lambda expression
 *
 * Computes the arithmetic mean of the results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Arithmetic mean of projected values
 * @throws std::invalid_argument If range is empty
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
 * @brief Median (accepts a sorted range)
 *
 * Computes the median of a sorted range.
 * Returns the average of the two middle values if the number of elements is even.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Median
 * @throws std::invalid_argument If range is empty
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
 * @brief Median of projected values using a lambda expression (projection results must be in sorted order)
 *
 * Computes the median of the results after applying a projection function to each element.
 * Assumes that projection results are sorted.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Median of projected values
 * @throws std::invalid_argument If range is empty
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
 * @brief Mode (returns the smallest value when there are multiple modes: guarantees deterministic behavior)
 *
 * Returns the most frequent value in the range.
 * When there are multiple modes, returns the smallest value.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Mode
 * @throws std::invalid_argument If range is empty
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
 * @brief Mode of projected values using a lambda expression
 *
 * Returns the most frequent value after applying a projection function to each element.
 * When there are multiple modes, returns the smallest value.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Mode of projected values
 * @throws std::invalid_argument If range is empty
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
// Modes (multiple modes)
// ============================================================================

/**
 * @brief Returns all modes (returns a vector sorted in ascending order)
 *
 * Returns all most frequent values in the range.
 * When there are multiple modes, returns all of them in ascending order as a vector.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Vector of modes (ascending order)
 * @throws std::invalid_argument If range is empty
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

    // Find maximum frequency
    std::size_t max_freq = 0;
    for (const auto& pair : freq) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
        }
    }

    // Collect all values with maximum frequency (ascending order due to map)
    std::vector<value_type> result;
    for (const auto& pair : freq) {
        if (pair.second == max_freq) {
            result.push_back(pair.first);
        }
    }

    return result;
}

/**
 * @brief Returns all modes of projected values using a lambda expression
 *
 * Returns all most frequent values after applying a projection function to each element.
 * When there are multiple modes, returns all of them in ascending order as a vector.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Vector of modes of projected values (ascending order)
 * @throws std::invalid_argument If range is empty
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

    // Find maximum frequency
    std::size_t max_freq = 0;
    for (const auto& pair : freq) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
        }
    }

    // Collect all values with maximum frequency (ascending order due to map)
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
 * @brief Geometric mean
 *
 * Computes the geometric mean of elements in the range.
 * All values must be positive.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Geometric mean
 * @throws std::invalid_argument If range is empty or if a value is zero or negative
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
 * @brief Geometric mean of projected values using a lambda expression
 *
 * Computes the geometric mean of the results after applying a projection function to each element.
 * All values must be positive.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Geometric mean of projected values
 * @throws std::invalid_argument If range is empty or if a value is zero or negative
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
 * @brief Harmonic mean
 *
 * Computes the harmonic mean of elements in the range.
 * All values must be non-zero.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Harmonic mean
 * @throws std::invalid_argument If range is empty or if a value is zero
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
        // Reject both exact zero and values so small that 1/val overflows to infinity.
        // std::numeric_limits<double>::min() is the smallest positive normal double (~2.2e-308),
        // so checking |val| < min() catches subnormals and zero without false positives on
        // legitimate small values that are still representable.
        if (std::abs(val) < std::numeric_limits<double>::min()) {
            throw std::invalid_argument("statcpp::harmonic_mean: zero or near-zero value encountered");
        }
        reciprocal_sum += 1.0 / val;
    }
    return static_cast<double>(n) / reciprocal_sum;
}

/**
 * @brief Harmonic mean of projected values using a lambda expression
 *
 * Computes the harmonic mean of the results after applying a projection function to each element.
 * All values must be non-zero.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Harmonic mean of projected values
 * @throws std::invalid_argument If range is empty or if a value is zero
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
 * @brief Trimmed mean (accepts a sorted range. proportion: exclusion ratio per side, 0.0 to less than 0.5)
 *
 * Computes the mean after excluding a certain proportion of data from both ends.
 * Useful for reducing the influence of outliers.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param proportion Exclusion ratio per side (0.0 to less than 0.5)
 * @return Trimmed mean
 * @throws std::invalid_argument If range is empty, proportion is out of range, or all elements are excluded
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
 * @brief Trimmed mean of projected values using a lambda expression (projection results must be in sorted order)
 *
 * Computes the trimmed mean of the results after applying a projection function to each element.
 * Assumes that projection results are sorted.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proportion Exclusion ratio per side (0.0 to less than 0.5)
 * @param proj Projection function
 * @return Trimmed mean of projected values
 * @throws std::invalid_argument If range is empty, proportion is out of range, or all elements are excluded
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
 * @brief Weighted mean
 *
 * Computes the mean with weights applied to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @return Weighted mean
 * @throws std::invalid_argument If range is empty, if negative weight exists, or if sum of weights is zero
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
 * @brief Weighted mean (projection version)
 *
 * Computes the weighted mean of the results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @param proj Projection function
 * @return Weighted mean of projected values
 * @throws std::invalid_argument If range is empty, if negative weight exists, or if sum of weights is zero
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
 * @brief Logarithmic Mean
 *
 * Computes the logarithmic mean of two positive values.
 * LM(a, b) = (b - a) / (ln(b) - ln(a)) for a != b
 * LM(a, a) = a
 *
 * @tparam T1 Type of first argument
 * @tparam T2 Type of second argument
 * @param a First value
 * @param b Second value
 * @return Logarithmic mean
 * @throws std::invalid_argument If arguments are not positive
 */
template <typename T1, typename T2>
double logarithmic_mean(T1 a, T2 b)
{
    double x = static_cast<double>(a);
    double y = static_cast<double>(b);

    if (x <= 0.0 || y <= 0.0) {
        throw std::invalid_argument("statcpp::logarithmic_mean: arguments must be positive");
    }

    // If values are approximately equal, return x directly.
    // Use relative difference to avoid misclassifying large values as distinct
    // when the relative gap is tiny (e.g., x = 1e15, y = 1e15 + 1e5).
    if (std::abs(x - y) <= 1e-10 * std::max(x, y)) {
        return x;
    }

    // Standard logarithmic mean: (y - x) / (ln(y) - ln(x))
    return (y - x) / (std::log(y) - std::log(x));
}

// ============================================================================
// Weighted Harmonic Mean
// ============================================================================

/**
 * @brief Weighted harmonic mean
 *
 * Computes the harmonic mean with weights applied to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @return Weighted harmonic mean
 * @throws std::invalid_argument If range is empty, if negative weight exists, if value is zero, or if sum of weights is zero
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
 * @brief Weighted harmonic mean (projection version)
 *
 * Computes the weighted harmonic mean of the results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @param proj Projection function
 * @return Weighted harmonic mean of projected values
 * @throws std::invalid_argument If range is empty, if negative weight exists, if value is zero, or if sum of weights is zero
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
 * @brief Returns the index of the minimum value
 *
 * Returns the index of the element with the minimum value in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Index of the minimum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Returns the index of the minimum value (projection version)
 *
 * Returns the index of the element whose projected value is minimum.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Index of the minimum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Returns the index of the maximum value
 *
 * Returns the index of the element with the maximum value in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Index of the maximum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Returns the index of the maximum value (projection version)
 *
 * Returns the index of the element whose projected value is maximum.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Index of the maximum value
 * @throws std::invalid_argument If range is empty
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
