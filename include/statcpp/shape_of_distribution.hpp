/**
 * @file shape_of_distribution.hpp
 * @brief Distribution shape measurement (skewness and kurtosis)
 *
 * This file provides functions to calculate skewness and kurtosis,
 * which are statistics representing the shape of a distribution.
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
// Skewness
// ============================================================================

/**
 * @brief Calculate population skewness (Fisher's definition)
 *
 * Calculates the skewness of a population. An indicator of distribution asymmetry.
 *
 * Formula: g1 = E[(X - mu)^3] / sigma^3
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Population skewness value (positive = right-skewed, negative = left-skewed)
 * @throw std::invalid_argument If range is empty or variance is zero
 *
 * @note The skewness of a normal distribution is 0
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
 * @brief Calculate population skewness with precomputed mean
 *
 * Use when the mean has already been calculated. Improves computational efficiency.
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Population skewness value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate population skewness with projection
 *
 * Uses a projection function to extract specific values from the data to calculate skewness.
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Population skewness value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate population skewness with projection and precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Population skewness value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate sample skewness (bias-corrected version)
 *
 * Calculates the bias-corrected skewness estimate from a sample.
 *
 * Formula: G1 = sqrt(n(n-1)) / (n-2) * g1
 * where g1 is the population skewness estimator
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Sample skewness value
 * @throw std::invalid_argument If number of elements is less than 3
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
 * @brief Calculate sample skewness with precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Sample skewness value
 * @throw std::invalid_argument If number of elements is less than 3
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
 * @brief Calculate sample skewness with projection
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Sample skewness value
 * @throw std::invalid_argument If number of elements is less than 3
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
 * @brief Calculate sample skewness with projection and precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample skewness value
 * @throw std::invalid_argument If number of elements is less than 3
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
 * @brief Calculate skewness (alias for sample_skewness)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Sample skewness value
 */
template <typename Iterator>
double skewness(Iterator first, Iterator last)
{
    return sample_skewness(first, last);
}

/**
 * @brief Calculate skewness (precomputed mean version)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Sample skewness value
 */
template <typename Iterator>
double skewness(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_skewness(first, last, precomputed_mean);
}

/**
 * @brief Calculate skewness (projection version)
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Sample skewness value
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
 * @brief Calculate skewness (projection version, precomputed mean)
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample skewness value
 */
template <typename Iterator, typename Projection>
double skewness(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_skewness(first, last, proj, precomputed_mean);
}

// ============================================================================
// Kurtosis
// ============================================================================

/**
 * @brief Calculate population kurtosis (Excess Kurtosis)
 *
 * Calculates the kurtosis of a population. An indicator of tail heaviness.
 *
 * Formula: g2 = E[(X - mu)^4] / sigma^4 - 3
 * Subtracts 3 so that normal distribution kurtosis is 0 (excess kurtosis)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Population kurtosis value (positive = heavy tails, negative = light tails)
 * @throw std::invalid_argument If range is empty or variance is zero
 *
 * @note The kurtosis (excess kurtosis) of a normal distribution is 0
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
 * @brief Calculate population kurtosis with precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Population kurtosis value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate population kurtosis with projection
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Population kurtosis value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate population kurtosis with projection and precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Population kurtosis value
 * @throw std::invalid_argument If range is empty or variance is zero
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
 * @brief Calculate sample kurtosis (bias-corrected version)
 *
 * Calculates the bias-corrected kurtosis estimate from a sample.
 *
 * Formula: G2 = ((n+1) * g2 + 6) * (n-1) / ((n-2)(n-3))
 * where g2 is the population kurtosis estimator (excess kurtosis)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Sample kurtosis value
 * @throw std::invalid_argument If number of elements is less than 4
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
 * @brief Calculate sample kurtosis with precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Sample kurtosis value
 * @throw std::invalid_argument If number of elements is less than 4
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
 * @brief Calculate sample kurtosis with projection
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Sample kurtosis value
 * @throw std::invalid_argument If number of elements is less than 4
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
 * @brief Calculate sample kurtosis with projection and precomputed mean
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample kurtosis value
 * @throw std::invalid_argument If number of elements is less than 4
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
 * @brief Calculate kurtosis (alias for sample_kurtosis)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @return Sample kurtosis value
 */
template <typename Iterator>
double kurtosis(Iterator first, Iterator last)
{
    return sample_kurtosis(first, last);
}

/**
 * @brief Calculate kurtosis (precomputed mean version)
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param precomputed_mean Precomputed mean value
 * @return Sample kurtosis value
 */
template <typename Iterator>
double kurtosis(Iterator first, Iterator last, double precomputed_mean)
{
    return sample_kurtosis(first, last, precomputed_mean);
}

/**
 * @brief Calculate kurtosis (projection version)
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @return Sample kurtosis value
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
 * @brief Calculate kurtosis (projection version, precomputed mean)
 *
 * @tparam Iterator Input iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last End iterator of data range
 * @param proj Projection function
 * @param precomputed_mean Precomputed mean value
 * @return Sample kurtosis value
 */
template <typename Iterator, typename Projection>
double kurtosis(Iterator first, Iterator last, Projection proj, double precomputed_mean)
{
    return sample_kurtosis(first, last, proj, precomputed_mean);
}

} // namespace statcpp
