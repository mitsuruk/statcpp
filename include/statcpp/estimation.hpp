/**
 * @file estimation.hpp
 * @brief Estimation and confidence interval functions
 *
 * Provides standard error calculations, confidence interval estimation,
 * margin of error and sample size calculations.
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Standard Error
// ============================================================================

/**
 * @brief Calculate standard error of the mean
 *
 * Standard error: SE = s / sqrt(n)
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @return Standard error
 * @throws std::invalid_argument If there are fewer than 2 elements
 */
template <typename Iterator>
double standard_error(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::standard_error: need at least 2 elements");
    }
    double s = statcpp::sample_stddev(first, last);
    return s / std::sqrt(static_cast<double>(n));
}

/**
 * @brief Calculate standard error of the mean (projection version)
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param proj Projection function
 * @return Standard error
 * @throws std::invalid_argument If there are fewer than 2 elements
 */
template <typename Iterator, typename Projection>
double standard_error(Iterator first, Iterator last, Projection proj)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::standard_error: need at least 2 elements");
    }
    double s = statcpp::sample_stddev(first, last, proj);
    return s / std::sqrt(static_cast<double>(n));
}

/**
 * @brief Calculate standard error of the mean (using precomputed standard deviation)
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param precomputed_stddev Precomputed standard deviation
 * @return Standard error
 * @throws std::invalid_argument If the range is empty
 */
template <typename Iterator>
double standard_error(Iterator first, Iterator last, double precomputed_stddev)
{
    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::standard_error: empty range");
    }
    return precomputed_stddev / std::sqrt(static_cast<double>(n));
}

// ============================================================================
// Confidence Interval Result Structure
// ============================================================================

/**
 * @brief Structure to store confidence interval results
 */
struct confidence_interval {
    double lower;              ///< Lower bound of confidence interval
    double upper;              ///< Upper bound of confidence interval
    double point_estimate;     ///< Point estimate
    double confidence_level;   ///< Confidence level
};

// ============================================================================
// Confidence Interval for Mean (using t-distribution)
// ============================================================================

/**
 * @brief Calculate confidence interval for mean (t-distribution based)
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1) or there are fewer than 2 elements
 */
template <typename Iterator>
confidence_interval ci_mean(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_mean: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double se = statcpp::standard_error(first, last);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

/**
 * @brief Calculate confidence interval for mean (projection version)
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param confidence Confidence level
 * @param proj Projection function
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1) or there are fewer than 2 elements
 */
template <typename Iterator, typename Projection>
confidence_interval ci_mean(Iterator first, Iterator last, double confidence, Projection proj)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_mean: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last, proj);
    double se = statcpp::standard_error(first, last, proj);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

// ============================================================================
// Confidence Interval for Mean (using z-distribution, known variance)
// ============================================================================

/**
 * @brief Calculate confidence interval for mean (z-distribution based, known variance)
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param sigma Known population standard deviation
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1), sigma is not positive, or the range is empty
 */
template <typename Iterator>
confidence_interval ci_mean_z(Iterator first, Iterator last, double sigma, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_z: confidence must be in (0, 1)");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::ci_mean_z: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::ci_mean_z: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    double se = sigma / std::sqrt(static_cast<double>(n));

    double alpha = 1.0 - confidence;
    double z_crit = norm_quantile(1.0 - alpha / 2.0);

    double margin = z_crit * se;

    return {mean_val - margin, mean_val + margin, mean_val, confidence};
}

// ============================================================================
// Confidence Interval for Proportion
// ============================================================================

/**
 * @brief Calculate confidence interval for proportion (Wald method)
 *
 * @param successes Number of successes
 * @param trials Number of trials
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1), trials is 0, or successes exceeds trials
 */
inline confidence_interval ci_proportion(std::size_t successes, std::size_t trials, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion: confidence must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::ci_proportion: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::ci_proportion: successes cannot exceed trials");
    }

    double p_hat = static_cast<double>(successes) / static_cast<double>(trials);
    double se = std::sqrt(p_hat * (1.0 - p_hat) / static_cast<double>(trials));

    double alpha = 1.0 - confidence;
    double z_crit = norm_quantile(1.0 - alpha / 2.0);

    double margin = z_crit * se;
    double lower = std::max(0.0, p_hat - margin);
    double upper = std::min(1.0, p_hat + margin);

    return {lower, upper, p_hat, confidence};
}

/**
 * @brief Calculate confidence interval for proportion (Wilson method, recommended)
 *
 * The Wilson method has better properties than the Wald method,
 * especially for small samples or extreme proportions.
 *
 * @param successes Number of successes
 * @param trials Number of trials
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1), trials is 0, or successes exceeds trials
 */
inline confidence_interval ci_proportion_wilson(std::size_t successes, std::size_t trials, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: confidence must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::ci_proportion_wilson: successes cannot exceed trials");
    }

    double n = static_cast<double>(trials);
    double p_hat = static_cast<double>(successes) / n;

    double alpha = 1.0 - confidence;
    double z = norm_quantile(1.0 - alpha / 2.0);
    double z2 = z * z;

    double denom = 1.0 + z2 / n;
    double center = (p_hat + z2 / (2.0 * n)) / denom;
    double margin = z * std::sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom;

    return {center - margin, center + margin, p_hat, confidence};
}

// ============================================================================
// Confidence Interval for Variance (Chi-square based)
// ============================================================================

/**
 * @brief Calculate confidence interval for variance (chi-square distribution based)
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1) or there are fewer than 2 elements
 */
template <typename Iterator>
confidence_interval ci_variance(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_variance: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::ci_variance: need at least 2 elements");
    }

    double var = statcpp::sample_variance(first, last);
    double df = static_cast<double>(n - 1);

    double alpha = 1.0 - confidence;
    double chi2_lower = chisq_quantile(alpha / 2.0, df);
    double chi2_upper = chisq_quantile(1.0 - alpha / 2.0, df);

    double lower = df * var / chi2_upper;
    double upper = df * var / chi2_lower;

    return {lower, upper, var, confidence};
}

// ============================================================================
// Confidence Interval for Difference of Means
// ============================================================================

/**
 * @brief Calculate confidence interval for difference of two-sample means (independent samples, pooled variance)
 *
 * Calculates the confidence interval for the difference of two-sample means assuming equal variances.
 *
 * @tparam Iterator1 First iterator type
 * @tparam Iterator2 Second iterator type
 * @param first1 Beginning iterator of first data range
 * @param last1 Ending iterator of first data range
 * @param first2 Beginning iterator of second data range
 * @param last2 Ending iterator of second data range
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1) or either sample has fewer than 2 elements
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff(Iterator1 first1, Iterator1 last1,
                                  Iterator2 first2, Iterator2 last2,
                                  double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    // Pooled variance
    double df = static_cast<double>(n1 + n2 - 2);
    double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df;
    double se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

/**
 * @brief Calculate confidence interval for difference of two-sample means (Welch method, not assuming equal variances)
 *
 * Calculates the confidence interval for the difference of two-sample means without assuming equal variances (Welch-Satterthwaite method).
 *
 * @tparam Iterator1 First iterator type
 * @tparam Iterator2 Second iterator type
 * @param first1 Beginning iterator of first data range
 * @param last1 Ending iterator of first data range
 * @param first2 Beginning iterator of second data range
 * @param last2 Ending iterator of second data range
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1), either sample has fewer than 2 elements, or both variances are zero
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff_welch(Iterator1 first1, Iterator1 last1,
                                        Iterator2 first2, Iterator2 last2,
                                        double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    double se1 = var1 / n1;
    double se2 = var2 / n2;
    double se = std::sqrt(se1 + se2);

    // Welch-Satterthwaite approximation for degrees of freedom
    double num = (se1 + se2) * (se1 + se2);
    double denom = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1);

    // Protection against zero denominator (when both variances are zero)
    if (denom == 0.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_welch: cannot compute degrees of freedom with zero variances");
    }

    double df = num / denom;

    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

// ============================================================================
// Margin of Error
// ============================================================================

/**
 * @brief Calculate margin of error for mean
 *
 * MoE = t_{alpha/2, df} * SE
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param confidence Confidence level (default: 0.95)
 * @return Margin of error
 * @throws std::invalid_argument If confidence level is outside (0, 1) or there are fewer than 2 elements
 */
template <typename Iterator>
double margin_of_error_mean(Iterator first, Iterator last, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: need at least 2 elements");
    }

    double se = statcpp::standard_error(first, last);
    double df = static_cast<double>(n - 1);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    return t_crit * se;
}

/**
 * @brief Calculate margin of error for mean (projection version)
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator of data range
 * @param last Ending iterator of data range
 * @param confidence Confidence level
 * @param proj Projection function
 * @return Margin of error
 * @throws std::invalid_argument If confidence level is outside (0, 1) or there are fewer than 2 elements
 */
template <typename Iterator, typename Projection>
double margin_of_error_mean(Iterator first, Iterator last, double confidence, Projection proj)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::margin_of_error_mean: need at least 2 elements");
    }

    double se = statcpp::standard_error(first, last, proj);
    double df = static_cast<double>(n - 1);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    return t_crit * se;
}

/**
 * @brief Calculate margin of error for proportion
 *
 * MoE = z_{alpha/2} * sqrt(p(1-p)/n)
 *
 * @param successes Number of successes
 * @param n Sample size
 * @param confidence Confidence level (default: 0.95)
 * @return Margin of error
 * @throws std::invalid_argument If confidence level is outside (0, 1), n is 0, or successes exceeds n
 */
inline double margin_of_error_proportion(std::size_t successes, std::size_t n, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: confidence must be in (0, 1)");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: n must be positive");
    }
    if (successes > n) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion: successes cannot exceed n");
    }

    double p = static_cast<double>(successes) / static_cast<double>(n);
    double se = std::sqrt(p * (1.0 - p) / static_cast<double>(n));

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    return z_crit * se;
}

/**
 * @brief Calculate worst-case margin of error for proportion
 *
 * Maximum at p=0.5: MoE = z_{alpha/2} * 0.5/sqrt(n)
 *
 * @param n Sample size
 * @param confidence Confidence level (default: 0.95)
 * @return Margin of error
 * @throws std::invalid_argument If confidence level is outside (0, 1) or n is 0
 */
inline double margin_of_error_proportion_worst_case(std::size_t n, double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion_worst_case: confidence must be in (0, 1)");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::margin_of_error_proportion_worst_case: n must be positive");
    }

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    return z_crit * 0.5 / std::sqrt(static_cast<double>(n));
}

// ============================================================================
// Sample Size Calculation (Margin-of-Error based)
// ============================================================================

/**
 * @brief Calculate sample size for proportion estimation
 *
 * Calculates the sample size required to achieve the specified margin of error.
 * n = (z_{alpha/2} / MoE)^2 * p(1-p)
 *
 * @param margin_of_error Target margin of error
 * @param confidence_level Confidence level (default: 0.95)
 * @param p_estimate Prior estimate of proportion (default: 0.5 for most conservative estimate)
 * @return Required sample size
 * @throws std::invalid_argument If parameters are outside valid range
 */
inline std::size_t sample_size_for_moe_proportion(double margin_of_error,
                                                   double confidence_level = 0.95,
                                                   double p_estimate = 0.5)
{
    if (margin_of_error <= 0.0 || margin_of_error >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: margin_of_error must be in (0, 1)");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: confidence_level must be in (0, 1)");
    }
    if (p_estimate <= 0.0 || p_estimate >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_proportion: p_estimate must be in (0, 1)");
    }

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    // n = (z / MoE)^2 * p(1-p)
    double n_exact = std::pow(z / margin_of_error, 2.0) * p_estimate * (1.0 - p_estimate);

    // Round up to integer
    return static_cast<std::size_t>(std::ceil(n_exact));
}

/**
 * @brief Calculate sample size for mean estimation (known population standard deviation)
 *
 * Calculates the sample size required to achieve the specified margin of error.
 * n = (z_{alpha/2} * sigma / MoE)^2
 *
 * @param margin_of_error Target margin of error
 * @param sigma Population standard deviation (known or estimated)
 * @param confidence_level Confidence level (default: 0.95)
 * @return Required sample size
 * @throws std::invalid_argument If parameters are outside valid range
 */
inline std::size_t sample_size_for_moe_mean(double margin_of_error,
                                             double sigma,
                                             double confidence_level = 0.95)
{
    if (margin_of_error <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: margin_of_error must be positive");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: sigma must be positive");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_for_moe_mean: confidence_level must be in (0, 1)");
    }

    double alpha = 1.0 - confidence_level;
    double z = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    // n = (z * sigma / MoE)^2
    double n_exact = std::pow(z * sigma / margin_of_error, 2.0);

    // Round up to integer
    return static_cast<std::size_t>(std::ceil(n_exact));
}

// ============================================================================
// Two-Sample Mean Difference Confidence Interval
// ============================================================================

/**
 * @brief Calculate confidence interval for two-sample mean difference (assuming equal variances)
 *
 * @tparam Iterator1 First iterator type
 * @tparam Iterator2 Second iterator type
 * @param first1 Beginning iterator of first data range
 * @param last1 Ending iterator of first data range
 * @param first2 Beginning iterator of second data range
 * @param last2 Ending iterator of second data range
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If confidence level is outside (0, 1) or either sample has fewer than 2 elements
 */
template <typename Iterator1, typename Iterator2>
confidence_interval ci_mean_diff_pooled(Iterator1 first1, Iterator1 last1,
                                        Iterator2 first2, Iterator2 last2,
                                        double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_mean_diff_pooled: confidence must be in (0, 1)");
    }

    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::ci_mean_diff_pooled: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double diff = mean1 - mean2;

    // Pooled variance
    double pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    double se = std::sqrt(pooled_var * (1.0 / n1 + 1.0 / n2));

    double df = static_cast<double>(n1 + n2 - 2);
    double alpha = 1.0 - confidence;
    double t_crit = t_quantile(1.0 - alpha / 2.0, df);

    double margin = t_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

// ============================================================================
// Two-Sample Proportion Difference Confidence Interval
// ============================================================================

/**
 * @brief Calculate confidence interval for two-sample proportion difference
 *
 * @param successes1 Number of successes in first sample
 * @param n1 First sample size
 * @param successes2 Number of successes in second sample
 * @param n2 Second sample size
 * @param confidence Confidence level (default: 0.95)
 * @return Confidence interval
 * @throws std::invalid_argument If parameters are outside valid range
 */
inline confidence_interval ci_proportion_diff(std::size_t successes1, std::size_t n1,
                                               std::size_t successes2, std::size_t n2,
                                               double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: confidence must be in (0, 1)");
    }
    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: sample sizes must be positive");
    }
    if (successes1 > n1 || successes2 > n2) {
        throw std::invalid_argument("statcpp::ci_proportion_diff: successes cannot exceed sample size");
    }

    double p1 = static_cast<double>(successes1) / static_cast<double>(n1);
    double p2 = static_cast<double>(successes2) / static_cast<double>(n2);

    double diff = p1 - p2;

    // Standard error for difference of proportions
    double se1 = p1 * (1.0 - p1) / static_cast<double>(n1);
    double se2 = p2 * (1.0 - p2) / static_cast<double>(n2);
    double se = std::sqrt(se1 + se2);

    double alpha = 1.0 - confidence;
    double z_crit = normal_quantile(1.0 - alpha / 2.0, 0.0, 1.0);

    double margin = z_crit * se;

    return {diff - margin, diff + margin, diff, confidence};
}

} // namespace statcpp
