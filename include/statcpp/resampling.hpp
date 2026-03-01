/**
 * @file resampling.hpp
 * @brief Resampling methods
 *
 * Provides resampling methods including bootstrap, jackknife, and permutation tests.
 * Can be used for standard error estimation of statistics, confidence interval calculation,
 * and hypothesis testing.
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Bootstrap Result Structure
// ============================================================================

/**
 * @brief Structure to store bootstrap estimation results
 *
 * Holds the estimated statistic from bootstrap method, standard error, confidence interval,
 * bias, and all replicate statistics.
 */
struct bootstrap_result {
    double estimate;                 ///< Estimated statistic computed from original data
    double standard_error;           ///< Bootstrap standard error
    double ci_lower;                 ///< Lower bound of confidence interval
    double ci_upper;                 ///< Upper bound of confidence interval
    double bias;                     ///< Bias (replicate mean - estimate)
    std::vector<double> replicates;  ///< All bootstrap replicate statistics
};

// ============================================================================
// Bootstrap Sampling
// ============================================================================

/**
 * @brief Generate a single bootstrap sample
 *
 * Generates a bootstrap sample of the same size as the original sample
 * by sampling with replacement from the given data.
 *
 * @tparam Iterator Input iterator type
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param engine Reference to random engine
 * @return Bootstrap sample obtained by sampling with replacement
 * @throws std::invalid_argument If input range is empty
 */
template <typename Iterator, typename Engine = default_random_engine>
std::vector<typename std::iterator_traits<Iterator>::value_type>
bootstrap_sample(Iterator first, Iterator last, Engine& engine)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::bootstrap_sample: empty range");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);
    std::vector<value_type> sample(n);

    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (std::size_t i = 0; i < n; ++i) {
        sample[i] = original[dist(engine)];
    }

    return sample;
}

/**
 * @brief Generate a single bootstrap sample (using default random engine)
 *
 * Generates a bootstrap sample using the global default random engine.
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @return Bootstrap sample obtained by sampling with replacement
 * @throws std::invalid_argument If input range is empty
 */
template <typename Iterator>
std::vector<typename std::iterator_traits<Iterator>::value_type>
bootstrap_sample(Iterator first, Iterator last)
{
    return bootstrap_sample(first, last, get_random_engine());
}

// ============================================================================
// Bootstrap Estimation
// ============================================================================

/**
 * @brief Perform general bootstrap estimation
 *
 * Applies the bootstrap method to any statistical function and computes
 * the estimated statistic, standard error, percentile confidence interval, and bias.
 *
 * Percentile confidence interval calculation:
 * - Lower index: floor(alpha/2 * B)
 * - Upper index: floor((1 - alpha/2) * B) - 1
 * - Where alpha = 1 - confidence, B = n_bootstrap
 *
 * @note The percentile method is the simplest confidence interval calculation method.
 *       For more accurate confidence intervals, use bootstrap_bca().
 *
 * @tparam Iterator Input iterator type
 * @tparam Statistic Statistical function type (function taking iterator pair and returning double)
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param stat_func Function object that computes the statistic
 * @param n_bootstrap Number of bootstrap iterations (default: 1000)
 * @param confidence Confidence level (default: 0.95)
 * @param engine Reference to random engine
 * @return bootstrap_result structure containing bootstrap estimation results
 * @throws std::invalid_argument If confidence is outside (0, 1)
 * @throws std::invalid_argument If number of input elements is less than 2
 */
template <typename Iterator, typename Statistic, typename Engine = default_random_engine>
bootstrap_result bootstrap(Iterator first, Iterator last, Statistic stat_func,
                           std::size_t n_bootstrap = 1000, double confidence = 0.95,
                           Engine& engine = get_random_engine())
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::bootstrap: confidence must be in (0, 1)");
    }

    if (n_bootstrap < 2) {
        throw std::invalid_argument("statcpp::bootstrap: n_bootstrap must be at least 2");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::bootstrap: need at least 2 elements");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);

    // Original statistic
    double theta_hat = stat_func(original.begin(), original.end());

    // Bootstrap replicates
    std::vector<double> replicates(n_bootstrap);

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        auto sample = bootstrap_sample(original.begin(), original.end(), engine);
        replicates[b] = stat_func(sample.begin(), sample.end());
    }

    // Sort replicates for percentile CI
    std::vector<double> sorted_replicates = replicates;
    std::sort(sorted_replicates.begin(), sorted_replicates.end());

    // Standard error
    double mean_rep = std::accumulate(replicates.begin(), replicates.end(), 0.0) / n_bootstrap;
    double se = 0.0;
    for (double rep : replicates) {
        se += (rep - mean_rep) * (rep - mean_rep);
    }
    se = std::sqrt(se / (n_bootstrap - 1));

    // Bias
    double bias = mean_rep - theta_hat;

    // Percentile CI (0-indexed)
    // lower_idx corresponds to the (alpha/2)-th percentile
    // upper_idx corresponds to the (1 - alpha/2)-th percentile
    double alpha = 1.0 - confidence;
    std::size_t lower_idx = static_cast<std::size_t>(std::floor(alpha / 2.0 * static_cast<double>(n_bootstrap)));
    std::size_t upper_idx = static_cast<std::size_t>(std::floor((1.0 - alpha / 2.0) * static_cast<double>(n_bootstrap)));

    // Boundary check
    if (lower_idx >= n_bootstrap) lower_idx = n_bootstrap - 1;
    if (upper_idx >= n_bootstrap) upper_idx = n_bootstrap - 1;

    double ci_lower = sorted_replicates[lower_idx];
    double ci_upper = sorted_replicates[upper_idx];

    return {theta_hat, se, ci_lower, ci_upper, bias, std::move(replicates)};
}

/**
 * @brief Perform bootstrap estimation of the mean
 *
 * Applies the bootstrap method to the sample mean and computes standard error
 * and confidence interval.
 *
 * @tparam Iterator Input iterator type
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param n_bootstrap Number of bootstrap iterations (default: 1000)
 * @param confidence Confidence level (default: 0.95)
 * @param engine Reference to random engine
 * @return bootstrap_result structure containing bootstrap estimation results
 * @throws std::invalid_argument If confidence is outside (0, 1)
 * @throws std::invalid_argument If number of input elements is less than 2
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_mean(Iterator first, Iterator last,
                                std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) { return statcpp::mean(f, l); };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

/**
 * @brief Perform bootstrap estimation of the median
 *
 * Applies the bootstrap method to the sample median and computes standard error
 * and confidence interval.
 *
 * @tparam Iterator Input iterator type
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param n_bootstrap Number of bootstrap iterations (default: 1000)
 * @param confidence Confidence level (default: 0.95)
 * @param engine Reference to random engine
 * @return bootstrap_result structure containing bootstrap estimation results
 * @throws std::invalid_argument If confidence is outside (0, 1)
 * @throws std::invalid_argument If number of input elements is less than 2
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_median(Iterator first, Iterator last,
                                  std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                  Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) {
        std::vector<typename std::iterator_traits<decltype(f)>::value_type> sorted(f, l);
        std::sort(sorted.begin(), sorted.end());
        return statcpp::median(sorted.begin(), sorted.end());
    };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

/**
 * @brief Perform bootstrap estimation of the standard deviation
 *
 * Applies the bootstrap method to the sample standard deviation and computes
 * standard error and confidence interval.
 *
 * @tparam Iterator Input iterator type
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param n_bootstrap Number of bootstrap iterations (default: 1000)
 * @param confidence Confidence level (default: 0.95)
 * @param engine Reference to random engine
 * @return bootstrap_result structure containing bootstrap estimation results
 * @throws std::invalid_argument If confidence is outside (0, 1)
 * @throws std::invalid_argument If number of input elements is less than 2
 */
template <typename Iterator, typename Engine = default_random_engine>
bootstrap_result bootstrap_stddev(Iterator first, Iterator last,
                                  std::size_t n_bootstrap = 1000, double confidence = 0.95,
                                  Engine& engine = get_random_engine())
{
    auto stat_func = [](auto f, auto l) { return statcpp::sample_stddev(f, l); };
    return bootstrap(first, last, stat_func, n_bootstrap, confidence, engine);
}

// ============================================================================
// BCa Bootstrap (Bias-corrected and accelerated)
// ============================================================================

/**
 * @brief Compute BCa (bias-corrected and accelerated) bootstrap confidence interval
 *
 * Implements the BCa method which provides more accurate confidence intervals
 * than the standard percentile method. Uses jackknife to estimate the acceleration
 * factor and applies bias correction.
 *
 * @tparam Iterator Input iterator type
 * @tparam Statistic Statistical function type (function taking iterator pair and returning double)
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first Beginning iterator of input range
 * @param last End iterator of input range
 * @param stat_func Function object that computes the statistic
 * @param n_bootstrap Number of bootstrap iterations (default: 1000)
 * @param confidence Confidence level (default: 0.95)
 * @param engine Reference to random engine
 * @return bootstrap_result structure containing bootstrap estimation results
 * @throws std::invalid_argument If confidence is outside (0, 1)
 * @throws std::invalid_argument If number of input elements is less than 3
 */
template <typename Iterator, typename Statistic, typename Engine = default_random_engine>
bootstrap_result bootstrap_bca(Iterator first, Iterator last, Statistic stat_func,
                               std::size_t n_bootstrap = 1000, double confidence = 0.95,
                               Engine& engine = get_random_engine())
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::bootstrap_bca: confidence must be in (0, 1)");
    }

    if (n_bootstrap < 2) {
        throw std::invalid_argument("statcpp::bootstrap_bca: n_bootstrap must be at least 2");
    }

    auto n = statcpp::count(first, last);
    if (n < 3) {
        throw std::invalid_argument("statcpp::bootstrap_bca: need at least 3 elements");
    }

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    std::vector<value_type> original(first, last);

    // Original statistic
    double theta_hat = stat_func(original.begin(), original.end());

    // Bootstrap replicates
    std::vector<double> replicates(n_bootstrap);

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        auto sample = bootstrap_sample(original.begin(), original.end(), engine);
        replicates[b] = stat_func(sample.begin(), sample.end());
    }

    // Bias correction factor z0
    std::size_t count_less = 0;
    for (double rep : replicates) {
        if (rep < theta_hat) ++count_less;
    }
    // Clip to [1, B-1] to avoid norm_quantile(0) = -inf or norm_quantile(1) = +inf
    if (count_less == 0) count_less = 1;
    if (count_less >= n_bootstrap) count_less = n_bootstrap - 1;
    double z0 = norm_quantile(static_cast<double>(count_less) / n_bootstrap);

    // Acceleration factor using jackknife
    std::vector<double> jackknife_estimates(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<value_type> jackknife_sample;
        jackknife_sample.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i) {
                jackknife_sample.push_back(original[j]);
            }
        }
        jackknife_estimates[i] = stat_func(jackknife_sample.begin(), jackknife_sample.end());
    }

    double jack_mean = std::accumulate(jackknife_estimates.begin(), jackknife_estimates.end(), 0.0) / n;
    double sum_cubed = 0.0;
    double sum_squared = 0.0;
    for (double j : jackknife_estimates) {
        double d = jack_mean - j;
        sum_squared += d * d;
        sum_cubed += d * d * d;
    }

    double a = (sum_squared == 0.0) ? 0.0 : sum_cubed / (6.0 * std::pow(sum_squared, 1.5));

    // Adjusted percentiles
    double alpha = 1.0 - confidence;
    double z_alpha_lower = norm_quantile(alpha / 2.0);
    double z_alpha_upper = norm_quantile(1.0 - alpha / 2.0);

    double alpha1 = norm_cdf(z0 + (z0 + z_alpha_lower) / (1.0 - a * (z0 + z_alpha_lower)));
    double alpha2 = norm_cdf(z0 + (z0 + z_alpha_upper) / (1.0 - a * (z0 + z_alpha_upper)));

    // Sort replicates
    std::vector<double> sorted_replicates = replicates;
    std::sort(sorted_replicates.begin(), sorted_replicates.end());

    auto clamp_index = [&](double a_val) -> std::size_t {
        double idx = a_val * static_cast<double>(n_bootstrap);
        if (idx < 0.0) return 0;
        if (idx >= static_cast<double>(n_bootstrap)) return n_bootstrap - 1;
        return static_cast<std::size_t>(idx);
    };
    std::size_t lower_idx = clamp_index(alpha1);
    std::size_t upper_idx = clamp_index(alpha2);

    double ci_lower = sorted_replicates[lower_idx];
    double ci_upper = sorted_replicates[upper_idx];

    // Standard error
    double mean_rep = std::accumulate(replicates.begin(), replicates.end(), 0.0) / n_bootstrap;
    double se = 0.0;
    for (double rep : replicates) {
        se += (rep - mean_rep) * (rep - mean_rep);
    }
    se = std::sqrt(se / (n_bootstrap - 1));

    double bias = mean_rep - theta_hat;

    return {theta_hat, se, ci_lower, ci_upper, bias, std::move(replicates)};
}

// ============================================================================
// Permutation Test Result
// ============================================================================

/**
 * @brief Structure to store permutation test results
 *
 * Holds the observed statistic, p-value, number of permutations, and permutation
 * distribution from the permutation test.
 */
struct permutation_result {
    double observed_statistic;                   ///< Observed test statistic
    double p_value;                              ///< Two-sided p-value
    std::size_t n_permutations;                  ///< Number of permutations performed
    std::vector<double> permutation_distribution; ///< Distribution of permutation statistics
};

// ============================================================================
// Permutation Test (Two-Sample)
// ============================================================================

/**
 * @brief Perform two-sample permutation test (test of difference in means)
 *
 * Performs a permutation test for the difference in means between two independent samples.
 * The null hypothesis is "the two group means are equal". Computes two-sided p-value.
 *
 * P-value calculation uses the inclusive method:
 * p = (count(|T*| >= |T_obs|) + 1) / (n_permutations + 1)
 * This includes the observed data itself as part of the null distribution,
 * preventing p-values from being zero (Phipson & Smyth, 2010).
 *
 * @tparam Iterator1 Iterator type for first sample
 * @tparam Iterator2 Iterator type for second sample
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param n_permutations Number of permutations (default: 10000)
 * @param engine Reference to random engine
 * @return permutation_result structure containing permutation test results
 * @throws std::invalid_argument If either sample is empty
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_two_sample(Iterator1 first1, Iterator1 last1,
                                                Iterator2 first2, Iterator2 last2,
                                                std::size_t n_permutations = 10000,
                                                Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::permutation_test_two_sample: empty sample");
    }

    // Combine samples
    std::vector<double> combined;
    combined.reserve(n1 + n2);
    for (auto it = first1; it != last1; ++it) {
        combined.push_back(static_cast<double>(*it));
    }
    for (auto it = first2; it != last2; ++it) {
        combined.push_back(static_cast<double>(*it));
    }

    // Observed statistic
    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double observed = mean1 - mean2;

    // Permutation distribution
    std::vector<double> perm_stats(n_permutations);

    for (std::size_t p = 0; p < n_permutations; ++p) {
        std::shuffle(combined.begin(), combined.end(), engine);

        double perm_mean1 = std::accumulate(combined.begin(), combined.begin() + n1, 0.0) / n1;
        double perm_mean2 = std::accumulate(combined.begin() + n1, combined.end(), 0.0) / n2;
        perm_stats[p] = perm_mean1 - perm_mean2;
    }

    // Two-sided p-value (inclusive method)
    // Include the observed statistic as part of the null distribution
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

// ============================================================================
// Permutation Test (Paired)
// ============================================================================

/**
 * @brief Perform paired permutation test
 *
 * Performs a permutation test for differences between two paired samples.
 * Generates the null distribution by randomly flipping the signs of differences.
 * The null hypothesis is "the mean of differences is zero". Computes two-sided p-value.
 *
 * P-value calculation uses the inclusive method:
 * p = (count(|T*| >= |T_obs|) + 1) / (n_permutations + 1)
 *
 * @tparam Iterator1 Iterator type for first sample
 * @tparam Iterator2 Iterator type for second sample
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param n_permutations Number of permutations (default: 10000)
 * @param engine Reference to random engine
 * @return permutation_result structure containing permutation test results
 * @throws std::invalid_argument If the two samples have different lengths
 * @throws std::invalid_argument If samples are empty
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_paired(Iterator1 first1, Iterator1 last1,
                                            Iterator2 first2, Iterator2 last2,
                                            std::size_t n_permutations = 10000,
                                            Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::permutation_test_paired: samples must have equal length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::permutation_test_paired: empty samples");
    }

    // Compute differences
    std::vector<double> diffs;
    diffs.reserve(n1);

    auto it1 = first1;
    auto it2 = first2;
    while (it1 != last1) {
        diffs.push_back(static_cast<double>(*it1) - static_cast<double>(*it2));
        ++it1;
        ++it2;
    }

    // Observed statistic (mean of differences)
    double observed = statcpp::mean(diffs.begin(), diffs.end());

    // Permutation by randomly flipping signs
    std::vector<double> perm_stats(n_permutations);
    std::uniform_int_distribution<int> coin(0, 1);

    for (std::size_t p = 0; p < n_permutations; ++p) {
        double sum = 0.0;
        for (double d : diffs) {
            sum += (coin(engine) == 0) ? d : -d;
        }
        perm_stats[p] = sum / static_cast<double>(n1);
    }

    // Two-sided p-value (inclusive method)
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

// ============================================================================
// Permutation Test for Correlation
// ============================================================================

/**
 * @brief Perform permutation test for correlation
 *
 * Performs a permutation test for Pearson correlation coefficient between two variables.
 * Generates the null distribution by shuffling one variable.
 * The null hypothesis is "there is no correlation between the two variables (correlation is 0)".
 * Computes two-sided p-value.
 *
 * @note P-value calculation uses the inclusive method:
 *       p = (count of permutation statistics as extreme or more extreme than observed + 1) / (n_permutations + 1)
 *       This method prevents p-values from being zero and provides more accurate estimation
 *       (Phipson & Smyth, 2010).
 *
 * @tparam Iterator1 Iterator type for first variable
 * @tparam Iterator2 Iterator type for second variable
 * @tparam Engine Random engine type (default: default_random_engine)
 * @param first1 Beginning iterator of first variable
 * @param last1 End iterator of first variable
 * @param first2 Beginning iterator of second variable
 * @param last2 End iterator of second variable
 * @param n_permutations Number of permutations (default: 10000)
 * @param engine Reference to random engine
 * @return permutation_result structure containing permutation test results
 * @throws std::invalid_argument If the two variables have different lengths
 * @throws std::invalid_argument If number of data pairs is less than 3
 */
template <typename Iterator1, typename Iterator2, typename Engine = default_random_engine>
permutation_result permutation_test_correlation(Iterator1 first1, Iterator1 last1,
                                                 Iterator2 first2, Iterator2 last2,
                                                 std::size_t n_permutations = 10000,
                                                 Engine& engine = get_random_engine())
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::permutation_test_correlation: samples must have equal length");
    }
    if (n1 < 3) {
        throw std::invalid_argument("statcpp::permutation_test_correlation: need at least 3 pairs");
    }

    // Copy data
    std::vector<double> x, y;
    x.reserve(n1);
    y.reserve(n1);

    for (auto it = first1; it != last1; ++it) {
        x.push_back(static_cast<double>(*it));
    }
    for (auto it = first2; it != last2; ++it) {
        y.push_back(static_cast<double>(*it));
    }

    // Helper to compute correlation
    auto compute_corr = [n1](const std::vector<double>& a, const std::vector<double>& b) {
        double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / n1;
        double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / n1;

        double cov = 0.0, var_a = 0.0, var_b = 0.0;
        for (std::size_t i = 0; i < n1; ++i) {
            double da = a[i] - mean_a;
            double db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        return cov / std::sqrt(var_a * var_b);
    };

    // Observed correlation
    double observed = compute_corr(x, y);

    // Permutation distribution
    std::vector<double> perm_stats(n_permutations);
    std::vector<double> y_perm = y;

    for (std::size_t p = 0; p < n_permutations; ++p) {
        std::shuffle(y_perm.begin(), y_perm.end(), engine);
        perm_stats[p] = compute_corr(x, y_perm);
    }

    // Two-sided p-value (inclusive method)
    // Include the observed statistic as part of the null distribution
    double abs_observed = std::abs(observed);
    std::size_t count_extreme = 1;  // Count the observed statistic itself
    for (double stat : perm_stats) {
        if (std::abs(stat) >= abs_observed) {
            ++count_extreme;
        }
    }

    double p_value = static_cast<double>(count_extreme) / static_cast<double>(n_permutations + 1);

    return {observed, p_value, n_permutations, std::move(perm_stats)};
}

} // namespace statcpp
