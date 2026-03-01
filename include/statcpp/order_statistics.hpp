/**
 * @file order_statistics.hpp
 * @brief Order statistics implementation
 *
 * Provides minimum, maximum, quartiles, percentiles, weighted median, and more.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Result Structures
// ============================================================================

/**
 * @brief Quartile result
 */
struct quartile_result {
    double q1;  ///< First quartile (25th percentile)
    double q2;  ///< Second quartile (median, 50th percentile)
    double q3;  ///< Third quartile (75th percentile)
};

/**
 * @brief Five-number summary result
 */
struct five_number_summary_result {
    double min;     ///< Minimum
    double q1;      ///< First quartile
    double median;  ///< Median
    double q3;      ///< Third quartile
    double max;     ///< Maximum
};

// ============================================================================
// Linear Interpolation Helper (R type=7 / Excel QUARTILE.INC equivalent)
// ============================================================================

/**
 * @brief Linear interpolation at position
 *
 * Decomposes position p * (n-1) into integer part lo and fractional part frac,
 * then performs linear interpolation.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of sorted range
 * @param n Number of elements
 * @param p Position (0.0 to 1.0)
 * @return Interpolated value
 *
 * @note Assumes sorted range.
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
 * @brief Linear interpolation at position with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of sorted range
 * @param n Number of elements
 * @param p Position (0.0 to 1.0)
 * @param proj Projection function
 * @return Interpolated value
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
 * @brief Return minimum value
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return Minimum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Return minimum value with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param proj Projection function
 * @return Minimum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Return maximum value
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return Maximum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Return maximum value with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param proj Projection function
 * @return Maximum value
 * @throws std::invalid_argument If range is empty
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
 * @brief Return quartiles
 *
 * Calculates quartiles (Q1, Q2, Q3) from sorted range.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @return Quartiles
 * @throws std::invalid_argument If range is empty
 *
 * @note Equivalent to R's type=7 (default) or Excel's QUARTILE.INC.
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
 * @brief Return quartiles with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @param proj Projection function
 * @return Quartiles
 * @throws std::invalid_argument If range is empty
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
 * @brief Return percentile
 *
 * Calculates specified percentile from sorted range.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @param p Percentile as proportion 0.0-1.0 (e.g., 90th percentile -> p = 0.9)
 * @return Percentile value
 * @throws std::invalid_argument If range is empty or p is out of range
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
 * @brief Return percentile with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @param p Percentile as proportion 0.0-1.0
 * @param proj Projection function
 * @return Percentile value
 * @throws std::invalid_argument If range is empty or p is out of range
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
 * @brief Return five-number summary
 *
 * Calculates five-number summary (min, Q1, median, Q3, max) from sorted range.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @return Five-number summary
 * @throws std::invalid_argument If range is empty
 *
 * @note Provides basic statistics needed for box plot visualization.
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
 * @brief Return five-number summary with projection
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of sorted range
 * @param last End of sorted range
 * @param proj Projection function
 * @return Five-number summary
 * @throws std::invalid_argument If range is empty
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
 * @brief Weighted median
 *
 * Calculates the weighted median.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param weight_first Beginning of weights
 * @return Weighted median
 * @throws std::invalid_argument If range is empty, weight is negative, or sum of weights is zero
 */
template <typename Iterator, typename WeightIterator>
double weighted_median(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_median: empty range");
    }

    // Create value-weight pairs
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

    // Sort by value
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Sum of weights
    double total_weight = 0.0;
    for (const auto& p : pairs) {
        total_weight += p.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_median: sum of weights is zero");
    }

    // Calculate cumulative weight to find median.
    // Use a relative tolerance when testing whether cumulative == half_weight.
    // Plain == on a floating-point accumulator is unreliable because sequential
    // additions may produce a value within rounding error of half_weight without
    // being bit-identical.
    double cumulative = 0.0;
    double half_weight = total_weight / 2.0;
    const double tol = std::numeric_limits<double>::epsilon() * half_weight;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= half_weight) {
            // If cumulative weight is (approximately) exactly half, take average with next value
            if (std::abs(cumulative - half_weight) <= tol && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

/**
 * @brief Weighted median with projection
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param weight_first Beginning of weights
 * @param proj Projection function
 * @return Weighted median
 * @throws std::invalid_argument If range is empty, weight is negative, or sum of weights is zero
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_median(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_median: empty range");
    }

    // Create value-weight pairs
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

    // Sort by value
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Sum of weights
    double total_weight = 0.0;
    for (const auto& p : pairs) {
        total_weight += p.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_median: sum of weights is zero");
    }

    // Calculate cumulative weight to find median.
    // Use a relative tolerance when testing whether cumulative == half_weight.
    // Plain == on a floating-point accumulator is unreliable because sequential
    // additions may produce a value within rounding error of half_weight without
    // being bit-identical.
    double cumulative = 0.0;
    double half_weight = total_weight / 2.0;
    const double tol = std::numeric_limits<double>::epsilon() * half_weight;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= half_weight) {
            // If cumulative weight is (approximately) exactly half, take average with next value
            if (std::abs(cumulative - half_weight) <= tol && i + 1 < pairs.size()) {
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
 * @brief Weighted percentile
 *
 * Calculates the weighted percentile.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param weight_first Beginning of weights
 * @param p Percentile as proportion 0.0-1.0
 * @return Weighted percentile value
 * @throws std::invalid_argument If parameters are invalid
 */
template <typename Iterator, typename WeightIterator>
double weighted_percentile(Iterator first, Iterator last, WeightIterator weight_first, double p)
{
    if (!(0.0 <= p && p <= 1.0)) {
        throw std::invalid_argument("statcpp::weighted_percentile: p must be in [0, 1]");
    }

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_percentile: empty range");
    }

    // Create value-weight pairs
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

    // Sort by value
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Sum of weights
    double total_weight = 0.0;
    for (const auto& pair : pairs) {
        total_weight += pair.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: sum of weights is zero");
    }

    // Explicit endpoint handling to avoid relying on floating-point loop
    if (p <= 0.0) return pairs.front().first;
    if (p >= 1.0) return pairs.back().first;

    // Calculate cumulative weight to find target percentile
    // For p=0.5, find position at 50% of total weight
    double target = p * total_weight;
    const double tol = std::numeric_limits<double>::epsilon() * total_weight;
    double cumulative = 0.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= target) {
            // If cumulative weight is approximately at target, take average with next value
            if (std::abs(cumulative - target) <= tol && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

/**
 * @brief Weighted percentile with projection
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Beginning of range
 * @param last End of range
 * @param weight_first Beginning of weights
 * @param p Percentile as proportion 0.0-1.0
 * @param proj Projection function
 * @return Weighted percentile value
 * @throws std::invalid_argument If parameters are invalid
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_percentile(Iterator first, Iterator last, WeightIterator weight_first, double p, Projection proj)
{
    if (!(0.0 <= p && p <= 1.0)) {
        throw std::invalid_argument("statcpp::weighted_percentile: p must be in [0, 1]");
    }

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_percentile: empty range");
    }

    // Create value-weight pairs
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

    // Sort by value
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Sum of weights
    double total_weight = 0.0;
    for (const auto& pair : pairs) {
        total_weight += pair.second;
    }

    if (total_weight == 0.0) {
        throw std::invalid_argument("statcpp::weighted_percentile: sum of weights is zero");
    }

    // Explicit endpoint handling to avoid relying on floating-point loop
    if (p <= 0.0) return pairs.front().first;
    if (p >= 1.0) return pairs.back().first;

    // Calculate cumulative weight to find target percentile
    // For p=0.5, find position at 50% of total weight
    double target = p * total_weight;
    const double tol = std::numeric_limits<double>::epsilon() * total_weight;
    double cumulative = 0.0;

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        cumulative += pairs[i].second;
        if (cumulative >= target) {
            // If cumulative weight is approximately at target, take average with next value
            if (std::abs(cumulative - target) <= tol && i + 1 < pairs.size()) {
                return (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
            return pairs[i].first;
        }
    }

    return pairs.back().first;
}

} // namespace statcpp
