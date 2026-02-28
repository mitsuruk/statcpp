/**
 * @file correlation_covariance.hpp
 * @brief Correlation and covariance computation functions
 *
 * Provides functions to measure relationships between variables including
 * covariance, Pearson correlation coefficient, Spearman's rank correlation coefficient,
 * and Kendall's rank correlation coefficient.
 * Uses iterator-based interface compatible with various containers.
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
// Covariance
// ============================================================================

/**
 * @brief Population covariance
 *
 * Computes the population covariance of two variables.
 * Cov(X, Y) = E[(X - mu_x)(Y - mu_y)] = (1/n) * sum((x_i - x_bar)(y_i - y_bar))
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Population covariance
 * @throws std::invalid_argument If range is empty or lengths differ
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
 * @brief Population covariance with precomputed means
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param mean_x Mean of the first data
 * @param mean_y Mean of the second data
 * @return Population covariance
 * @throws std::invalid_argument If range is empty or lengths differ
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
 * @brief Population covariance (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Population covariance
 * @throws std::invalid_argument If range is empty or lengths differ
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
 * @brief Sample covariance (unbiased covariance)
 *
 * Computes the sample covariance (unbiased estimator) of two variables.
 * s_xy = (1/(n-1)) * sum((x_i - x_bar)(y_i - y_bar))
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Sample covariance
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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
 * @brief Sample covariance with precomputed means
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param mean_x Mean of the first data
 * @param mean_y Mean of the second data
 * @return Sample covariance
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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
 * @brief Sample covariance (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Sample covariance
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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
 * @brief Covariance (alias for sample_covariance)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Sample covariance
 */
template <typename Iterator1, typename Iterator2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2)
{
    return sample_covariance(first1, last1, first2, last2);
}

/**
 * @brief Covariance with precomputed means (alias for sample_covariance)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param mean_x Mean of the first data
 * @param mean_y Mean of the second data
 * @return Sample covariance
 */
template <typename Iterator1, typename Iterator2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  double mean_x, double mean_y)
{
    return sample_covariance(first1, last1, first2, last2, mean_x, mean_y);
}

/**
 * @brief Covariance (projection version, alias for sample_covariance)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Sample covariance
 */
template <typename Iterator1, typename Iterator2, typename Projection1, typename Projection2>
double covariance(Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  Projection1 proj1, Projection2 proj2)
{
    return sample_covariance(first1, last1, first2, last2, proj1, proj2);
}

// ============================================================================
// Pearson Correlation Coefficient
// ============================================================================

/**
 * @brief Pearson correlation coefficient
 *
 * Computes the Pearson product-moment correlation coefficient of two variables.
 * r = Cov(X, Y) / (sigma_x * sigma_y)
 * Values range from -1 to 1, where 1 indicates perfect positive correlation
 * and -1 indicates perfect negative correlation.
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Pearson correlation coefficient (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, number of elements is less than 2,
 *         or variance of either variable is zero
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
 * @brief Pearson correlation coefficient with precomputed means
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param mean_x Mean of the first data
 * @param mean_y Mean of the second data
 * @return Pearson correlation coefficient (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, number of elements is less than 2,
 *         or variance of either variable is zero
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
 * @brief Pearson correlation coefficient (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Pearson correlation coefficient (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, number of elements is less than 2,
 *         or variance of either variable is zero
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
// Spearman's Rank Correlation Coefficient
// ============================================================================

namespace detail {

/**
 * @brief Helper function to compute ranks
 *
 * Uses average rank for ties.
 *
 * @tparam Iterator Iterator type
 * @param first Begin iterator
 * @param last End iterator
 * @return Vector of ranks
 */
template <typename Iterator>
std::vector<double> compute_ranks(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    // Create pairs of index and value
    std::vector<std::pair<std::size_t, double>> indexed_values;
    indexed_values.reserve(n);
    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        indexed_values.emplace_back(idx, static_cast<double>(*it));
    }

    // Sort by value
    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Compute ranks (average rank for ties)
    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        // Find range of elements with the same value
        while (j < n && indexed_values[j].second == indexed_values[i].second) {
            ++j;
        }
        // Compute average rank (1-based)
        double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            ranks[indexed_values[k].first] = avg_rank;
        }
        i = j;
    }

    return ranks;
}

/**
 * @brief Rank computation (projection version)
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Begin iterator
 * @param last End iterator
 * @param proj Projection function
 * @return Vector of ranks
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
 * @brief Spearman's rank correlation coefficient
 *
 * Computes Spearman's rank correlation coefficient of two variables.
 * rho = Pearson(rank(X), rank(Y))
 * Applies Pearson correlation coefficient to ranked data.
 * Can detect monotonic relationships and is robust to outliers.
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Spearman's rank correlation coefficient (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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
 * @brief Spearman's rank correlation coefficient (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Spearman's rank correlation coefficient (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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
 * @brief Kendall's rank correlation coefficient (tau-b)
 *
 * Computes Kendall's rank correlation coefficient (tau-b) of two variables.
 * This version accounts for ties.
 * tau_b = (concordant - discordant) / sqrt((n0 - tie_x) * (n0 - tie_y))
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @return Kendall's tau-b (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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

    // Create pairs
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);

    auto it1 = first1;
    auto it2 = first2;
    for (std::size_t i = 0; i < n; ++i, ++it1, ++it2) {
        pairs.push_back({static_cast<double>(*it1), static_cast<double>(*it2)});
    }

    // Count concordant, discordant, and tied pairs.
    //
    // Tie detection uses exact equality (diff == 0.0).
    // This is correct when the input values are integers or bit-identical doubles
    // (the common case for statistical data).
    // If the inputs are the result of floating-point computations, two values that
    // are mathematically equal may differ by a rounding error and thus not be
    // detected as tied.  Callers should round or quantise such data before passing
    // it to this function.
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
        // All tied case
        return 0.0;
    }

    double numerator = static_cast<double>(concordant) - static_cast<double>(discordant);
    double denominator = std::sqrt(static_cast<double>(n0 - tie_count_x) * static_cast<double>(n0 - tie_count_y));

    return numerator / denominator;
}

/**
 * @brief Kendall's rank correlation coefficient (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Kendall's tau-b (-1 to 1)
 * @throws std::invalid_argument If range is empty, lengths differ, or number of elements is less than 2
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

    // Create pairs of projected values
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(n);

    auto it1 = first1;
    auto it2 = first2;
    for (std::size_t i = 0; i < n; ++i, ++it1, ++it2) {
        pairs.push_back({static_cast<double>(std::invoke(proj1, *it1)),
                         static_cast<double>(std::invoke(proj2, *it2))});
    }

    // Count concordant, discordant, and tied pairs.
    //
    // Tie detection uses exact equality (diff == 0.0).
    // This is correct when the projected values are integers or bit-identical doubles.
    // For inputs derived from floating-point computations, callers should round or
    // quantise the data before passing it to this function.
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
 * @brief Weighted covariance
 *
 * Computes covariance with weights applied. Applies Bessel correction.
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam WeightIterator Iterator type for weights
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param weight_first Begin iterator for weights
 * @return Weighted covariance
 * @throws std::invalid_argument If range is empty, lengths differ, negative weight exists,
 *         or sum of weights is zero
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

    // Compute weighted mean
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

    // Compute weighted covariance
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

    // Bessel correction for frequency weights.
    // The formula  W / (W^2 - sum(w_i^2))  yields the unbiased estimate when weights
    // represent repeat counts (frequency weights).  It reduces to n/(n-1) when all
    // weights equal 1.
    // For precision weights (inverse-variance), a different correction is required.
    double correction = sum_weights / (sum_weights * sum_weights - sum_w_sq);
    return cov * correction;
}

/**
 * @brief Weighted covariance (projection version)
 *
 * @tparam Iterator1 Iterator type for the first data
 * @tparam Iterator2 Iterator type for the second data
 * @tparam WeightIterator Iterator type for weights
 * @tparam Projection1 Projection function type for the first data
 * @tparam Projection2 Projection function type for the second data
 * @param first1 Begin iterator for the first data
 * @param last1 End iterator for the first data
 * @param first2 Begin iterator for the second data
 * @param last2 End iterator for the second data
 * @param weight_first Begin iterator for weights
 * @param proj1 Projection function for the first data
 * @param proj2 Projection function for the second data
 * @return Weighted covariance
 * @throws std::invalid_argument If range is empty, lengths differ, negative weight exists,
 *         or sum of weights is zero
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

    // Compute weighted mean
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

    // Compute weighted covariance
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

    // Bessel correction for frequency weights.
    // The formula  W / (W^2 - sum(w_i^2))  yields the unbiased estimate when weights
    // represent repeat counts (frequency weights).  It reduces to n/(n-1) when all
    // weights equal 1.
    // For precision weights (inverse-variance), a different correction is required.
    double correction = sum_weights / (sum_weights * sum_weights - sum_w_sq);
    return cov * correction;
}

} // namespace statcpp
