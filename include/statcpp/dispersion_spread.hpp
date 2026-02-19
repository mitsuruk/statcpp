/**
 * @file dispersion_spread.hpp
 * @brief Dispersion and variance calculation functions
 *
 * Provides functions for measuring data spread, including variance, standard deviation,
 * range, and interquartile range. Uses iterator-based interface compatible with various containers.
 *
 * @note Numerical stability: Variance and standard deviation functions use the two-pass
 *       algorithm (first pass computes the mean, second pass computes the sum of squared
 *       deviations from the mean). This is numerically stable for typical datasets.
 *       For extreme cases where all values are very large with tiny relative differences
 *       (e.g., values near 1e15 differing by 1e-2), precision may be limited by the
 *       ~15 significant digits of IEEE 754 double-precision arithmetic. This limitation
 *       is inherent to floating-point representation and applies to all algorithms
 *       including Welford's method. R uses the same two-pass approach.
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
 * @brief Range (maximum - minimum)
 *
 * Computes the difference between the maximum and minimum values in the range.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Range (maximum - minimum)
 * @throws std::invalid_argument If the range is empty
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
 * @brief Range of projected values using a lambda expression
 *
 * Computes the range of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Range of projected values (maximum - minimum)
 * @throws std::invalid_argument If the range is empty
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
 * @brief Variance (ddof = Delta Degrees of Freedom)
 *
 * ddof = 0: Population variance (divide by N)
 * ddof = 1: Sample variance / unbiased variance (divide by N-1)
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Variance
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
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
 * @brief Variance using precomputed mean (with ddof)
 *
 * Use when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Variance
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
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
 * @brief Variance of projected values using a lambda expression (with ddof)
 *
 * Computes the variance of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Variance of projected values
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
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
 * @brief Variance of projected values using precomputed mean (with ddof)
 *
 * Computes variance using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Variance of projected values
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
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
// Population Variance (alias for ddof=0)
// ============================================================================

/**
 * @brief Population variance
 *
 * Computes the population variance (divide by N).
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Population variance
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator>
double population_variance(Iterator first, Iterator last)
{
    return var(first, last, 0);
}

/**
 * @brief Population variance using precomputed mean
 *
 * Computes population variance when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Population variance
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator>
double population_variance(Iterator first, Iterator last, double precomputed_mean)
{
    return var(first, last, precomputed_mean, 0);
}

/**
 * @brief Population variance of projected values using a lambda expression
 *
 * Computes the population variance of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Population variance of projected values
 * @throws std::invalid_argument If the range is empty
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
 * @brief Population variance of projected values using precomputed mean
 *
 * Computes population variance using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Population variance of projected values
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator, typename Projection>
double population_variance(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return var(first, last, proj, precomputed_mean, 0);
}

// ============================================================================
// Sample Variance (alias for ddof=1)
// ============================================================================

/**
 * @brief Sample variance (unbiased variance)
 *
 * Computes the unbiased sample variance (divide by N-1).
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Sample variance
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double sample_variance(Iterator first, Iterator last)
{
    return var(first, last, 1);
}

/**
 * @brief Sample variance using precomputed mean
 *
 * Computes sample variance when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Sample variance
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double sample_variance(Iterator first, Iterator last, double precomputed_mean)
{
    return var(first, last, precomputed_mean, 1);
}

/**
 * @brief Sample variance of projected values using a lambda expression
 *
 * Computes the sample variance of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Sample variance of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Sample variance of projected values using precomputed mean
 *
 * Computes sample variance using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample variance of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Variance (alias for sample_variance)
 *
 * Computes sample variance. Equivalent to sample_variance.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Variance
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double variance(Iterator first, Iterator last)
{
    return sample_variance(first, last);
}

/**
 * @brief Variance using precomputed mean
 *
 * Computes sample variance when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Variance
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double variance(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_variance(first, last, precomputed_mean);
}

/**
 * @brief Variance of projected values using a lambda expression
 *
 * Computes the sample variance of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Variance of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Variance of projected values using precomputed mean
 *
 * Computes sample variance using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Variance of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Standard deviation (ddof = Delta Degrees of Freedom)
 *
 * ddof = 0: Population standard deviation (square root of variance divided by N)
 * ddof = 1: Sample standard deviation (square root of variance divided by N-1)
 * Behaves like NumPy's np.std(a, ddof=...).
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Standard deviation
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
 */
template <typename Iterator>
double stdev(Iterator first, Iterator last, std::size_t ddof = 0)
{
    return std::sqrt(var(first, last, ddof));
}

/**
 * @brief Standard deviation using precomputed mean (with ddof)
 *
 * Use when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Standard deviation
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
 */
template <typename Iterator>
double stdev(Iterator first, Iterator last, double precomputed_mean, std::size_t ddof)
{
    return std::sqrt(var(first, last, precomputed_mean, ddof));
}

/**
 * @brief Standard deviation of projected values using a lambda expression (with ddof)
 *
 * Computes the standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Standard deviation of projected values
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
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
 * @brief Standard deviation of projected values using precomputed mean (with ddof)
 *
 * Computes standard deviation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @param ddof Degrees of freedom correction (0 or 1)
 * @return Standard deviation of projected values
 * @throws std::invalid_argument If range is empty, ddof is not 0 or 1, or ddof=1 with fewer than 2 elements
 */
template <typename Iterator, typename Projection>
double stdev(Iterator first, Iterator last, Projection proj, double precomputed_mean, std::size_t ddof)
{
    return std::sqrt(var(first, last, proj, precomputed_mean, ddof));
}

// ============================================================================
// Population Standard Deviation (alias for ddof=0)
// ============================================================================

/**
 * @brief Population standard deviation
 *
 * Computes the population standard deviation (square root of variance divided by N).
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Population standard deviation
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator>
double population_stddev(Iterator first, Iterator last)
{
    return stdev(first, last, 0);
}

/**
 * @brief Population standard deviation using precomputed mean
 *
 * Computes population standard deviation when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Population standard deviation
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator>
double population_stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return stdev(first, last, precomputed_mean, 0);
}

/**
 * @brief Population standard deviation of projected values using a lambda expression
 *
 * Computes the population standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Population standard deviation of projected values
 * @throws std::invalid_argument If the range is empty
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
 * @brief Population standard deviation of projected values using precomputed mean
 *
 * Computes population standard deviation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Population standard deviation of projected values
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator, typename Projection>
double population_stddev(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return stdev(first, last, proj, precomputed_mean, 0);
}

// ============================================================================
// Sample Standard Deviation (alias for ddof=1)
// ============================================================================

/**
 * @brief Sample standard deviation
 *
 * Computes the sample standard deviation (square root of variance divided by N-1).
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Sample standard deviation
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double sample_stddev(Iterator first, Iterator last)
{
    return stdev(first, last, 1);
}

/**
 * @brief Sample standard deviation using precomputed mean
 *
 * Computes sample standard deviation when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Sample standard deviation
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double sample_stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return stdev(first, last, precomputed_mean, 1);
}

/**
 * @brief Sample standard deviation of projected values using a lambda expression
 *
 * Computes the sample standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Sample standard deviation of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Sample standard deviation of projected values using precomputed mean
 *
 * Computes sample standard deviation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample standard deviation of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Standard deviation (alias for sample_stddev)
 *
 * Computes sample standard deviation. Equivalent to sample_stddev.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Standard deviation
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double stddev(Iterator first, Iterator last)
{
    return sample_stddev(first, last);
}

/**
 * @brief Standard deviation using precomputed mean
 *
 * Computes sample standard deviation when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Standard deviation
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
 */
template <typename Iterator>
double stddev(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_stddev(first, last, precomputed_mean);
}

/**
 * @brief Standard deviation of projected values using a lambda expression
 *
 * Computes the sample standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Standard deviation of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Standard deviation of projected values using precomputed mean
 *
 * Computes sample standard deviation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Standard deviation of projected values
 * @throws std::invalid_argument If the range is empty or has fewer than 2 elements
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
 * @brief Coefficient of variation
 *
 * Computes the standard deviation divided by the mean (returned as a ratio; multiply by 100 for percentage).
 * Useful for measuring relative variability of data.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Coefficient of variation (ratio)
 * @throws std::invalid_argument If range is empty, has fewer than 2 elements, or mean is zero
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
 * @brief Coefficient of variation using precomputed mean
 *
 * Computes coefficient of variation when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Coefficient of variation (ratio)
 * @throws std::invalid_argument If range is empty, has fewer than 2 elements, or mean is zero
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
 * @brief Coefficient of variation of projected values using a lambda expression
 *
 * Computes the coefficient of variation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Coefficient of variation of projected values (ratio)
 * @throws std::invalid_argument If range is empty, has fewer than 2 elements, or mean is zero
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
 * @brief Coefficient of variation of projected values using precomputed mean
 *
 * Computes coefficient of variation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Coefficient of variation of projected values (ratio)
 * @throws std::invalid_argument If range is empty, has fewer than 2 elements, or mean is zero
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
 * @brief Interquartile range (accepts a sorted range)
 *
 * Computes the difference between the third and first quartiles.
 * Useful for measuring the spread of the middle 50% of the data.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Interquartile range (Q3 - Q1)
 * @throws std::invalid_argument If the range is empty
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
 * @brief Interquartile range of projected values (projection results must be in sorted order)
 *
 * Computes the interquartile range of results after applying a projection function to each element.
 * Assumes projection results are in sorted order.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Interquartile range of projected values (Q3 - Q1)
 * @throws std::invalid_argument If the range is empty
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
 * @brief Mean absolute deviation
 *
 * Computes the mean of absolute deviations from the mean.
 * A measure of dispersion less sensitive to outliers than variance.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Mean absolute deviation
 * @throws std::invalid_argument If the range is empty
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
 * @brief Mean absolute deviation using precomputed mean
 *
 * Computes mean absolute deviation when the mean has been precomputed.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param precomputed_mean Precomputed mean value
 * @return Mean absolute deviation
 * @throws std::invalid_argument If the range is empty
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
 * @brief Mean absolute deviation of projected values using a lambda expression
 *
 * Computes the mean absolute deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Mean absolute deviation of projected values
 * @throws std::invalid_argument If the range is empty
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
 * @brief Mean absolute deviation of projected values using precomputed mean
 *
 * Computes mean absolute deviation using a projection function and precomputed mean.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Mean absolute deviation of projected values
 * @throws std::invalid_argument If the range is empty
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
 * @brief Weighted variance (frequency weights: repetition count for each data point)
 *
 * Computes variance with weights applied to each element.
 * Returns unbiased estimator with Bessel's correction.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @return Weighted variance
 * @throws std::invalid_argument If range is empty, negative weights exist, sum of weights is zero, or insufficient effective sample size
 */
template <typename Iterator, typename WeightIterator>
double weighted_variance(Iterator first, Iterator last, WeightIterator weight_first)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_variance: empty range");
    }

    // Compute weighted mean
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

    // Compute weighted variance (with Bessel's correction)
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

    // Bessel's correction: V = (Σw * Σw(x-μ)²) / (Σw² - Σw²)
    // Simplified: V = Σw(x-μ)² / (Σw - Σw²/Σw)
    double correction = sum_weights - (sum_weights_squared / sum_weights);

    if (correction <= 0.0) {
        throw std::invalid_argument("statcpp::weighted_variance: insufficient effective sample size");
    }

    return sum_squared_dev / correction;
}

/**
 * @brief Weighted variance (projection version)
 *
 * Computes weighted variance of results after applying a projection function to each element.
 * Returns unbiased estimator with Bessel's correction.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @param proj Projection function
 * @return Weighted variance of projected values
 * @throws std::invalid_argument If range is empty, negative weights exist, sum of weights is zero, or insufficient effective sample size
 */
template <typename Iterator, typename WeightIterator, typename Projection>
double weighted_variance(Iterator first, Iterator last, WeightIterator weight_first, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::weighted_variance: empty range");
    }

    // Compute weighted mean
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

    // Compute weighted variance
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
 * @brief Weighted standard deviation
 *
 * Computes the square root of weighted variance.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @return Weighted standard deviation
 * @throws std::invalid_argument If range is empty, negative weights exist, sum of weights is zero, or insufficient effective sample size
 */
template <typename Iterator, typename WeightIterator>
double weighted_stddev(Iterator first, Iterator last, WeightIterator weight_first)
{
    return std::sqrt(weighted_variance(first, last, weight_first));
}

/**
 * @brief Weighted standard deviation (projection version)
 *
 * Computes the weighted standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam WeightIterator Weight iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param weight_first Begin iterator for weights
 * @param proj Projection function
 * @return Weighted standard deviation of projected values
 * @throws std::invalid_argument If range is empty, negative weights exist, sum of weights is zero, or insufficient effective sample size
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
 * @brief Geometric standard deviation
 *
 * Computes the exponential of the standard deviation of log-transformed data.
 * GSD = exp(sigma_ln) where sigma_ln is the standard deviation of ln(x)
 * Useful for measuring dispersion of data following a log-normal distribution.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Geometric standard deviation
 * @throws std::invalid_argument If the range is empty or values are non-positive
 */
template <typename Iterator>
double geometric_stddev(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_stddev: empty range");
    }

    // Compute standard deviation of log-transformed data
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
 * @brief Geometric standard deviation (projection version)
 *
 * Computes the geometric standard deviation of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Geometric standard deviation of projected values
 * @throws std::invalid_argument If the range is empty or values are non-positive
 */
template <typename Iterator, typename Projection>
double geometric_stddev(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::geometric_stddev: empty range");
    }

    // Compute standard deviation of log-transformed data
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
