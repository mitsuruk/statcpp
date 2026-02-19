/**
 * @file power_analysis.hpp
 * @brief Power analysis implementation
 *
 * Provides power calculation and sample size determination for t-tests, proportion tests, and more.
 *
 * @note Implementation note (about normal distribution approximation):
 *       This implementation uses normal distribution approximation for t-test power calculations.
 *       Strictly speaking, the noncentral t-distribution should be used, but
 *       the normal approximation provides sufficient accuracy under the following conditions:
 *       - Large sample sizes (n > 30 approximately)
 *       - Medium or larger effect sizes
 *
 *       For small samples or small effect sizes, this approximation may slightly overestimate
 *       the power. For more precise calculations, consider using specialized software
 *       such as R's pwr package or G*Power.
 */

#pragma once

#include "statcpp/special_functions.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/parametric_tests.hpp"  // for alternative_hypothesis enum

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace statcpp {

// ============================================================================
// Power Analysis Result Structure
// ============================================================================

/**
 * @brief Power analysis result
 */
struct power_result {
    double power;          ///< Statistical power (1-beta)
    double sample_size;    ///< Sample size
    double effect_size;    ///< Effect size
    double alpha;          ///< Significance level
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Internal helper functions
 */
namespace detail {

/**
 * @brief Calculate noncentrality parameter for t-distribution
 *
 * @param effect_size Effect size
 * @param n Sample size
 * @return Noncentrality parameter
 */
inline double noncentrality_parameter_t(double effect_size, double n) {
    return effect_size * std::sqrt(n);
}

/**
 * @brief Calculate noncentrality parameter for two-sample case
 *
 * @param effect_size Effect size
 * @param n1 Sample size of group 1
 * @param n2 Sample size of group 2
 * @return Noncentrality parameter
 */
inline double noncentrality_parameter_t_two_sample(double effect_size, double n1, double n2) {
    return effect_size * std::sqrt((n1 * n2) / (n1 + n2));
}

/**
 * @brief Calculate critical value for two-sided t-test
 *
 * @param alpha Significance level
 * @param df Degrees of freedom
 * @return Critical value
 */
inline double critical_t_two_sided(double alpha, double df) {
    return t_quantile(1.0 - alpha / 2.0, df);
}

/**
 * @brief Calculate critical value for one-sided t-test
 *
 * @param alpha Significance level
 * @param df Degrees of freedom
 * @return Critical value
 */
inline double critical_t_one_sided(double alpha, double df) {
    return t_quantile(1.0 - alpha, df);
}

/**
 * @brief Calculate critical value for two-sided normal test
 *
 * @param alpha Significance level
 * @return Critical value
 */
inline double critical_z_two_sided(double alpha) {
    return norm_quantile(1.0 - alpha / 2.0);
}

/**
 * @brief Calculate critical value for one-sided normal test
 *
 * @param alpha Significance level
 * @return Critical value
 */
inline double critical_z_one_sided(double alpha) {
    return norm_quantile(1.0 - alpha);
}

/**
 * @brief Convert alternative_hypothesis enum to string
 *
 * @param alt Alternative hypothesis enum value
 * @return Corresponding string ("two.sided", "greater", or "less")
 */
inline const char* alternative_to_string(alternative_hypothesis alt) {
    switch (alt) {
        case alternative_hypothesis::two_sided: return "two.sided";
        case alternative_hypothesis::greater:   return "greater";
        case alternative_hypothesis::less:      return "less";
        default:                                return "two.sided";
    }
}

} // namespace detail

// ============================================================================
// One-sample t-test power analysis
// ============================================================================

/**
 * @brief Calculate power for one-sample t-test
 *
 * @param effect_size Effect size (Cohen's d)
 * @param n Sample size
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis: "two.sided", "greater", "less" (default: "two.sided")
 * @return Statistical power (0.0 to 1.0)
 * @throws std::invalid_argument If parameters are invalid
 *
 * @note This function uses normal distribution approximation. Strictly speaking, the noncentral
 *       t-distribution should be used, but for large sample sizes (n > 30 approximately),
 *       sufficient accuracy is obtained. For small samples, power may be slightly overestimated.
 */
inline double power_t_test_one_sample(double effect_size, std::size_t n,
                                      double alpha = 0.05,
                                      const std::string& alternative = "two.sided")
{
    if (n == 0) {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: sample size must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: alpha must be in (0, 1)");
    }

    double ncp = detail::noncentrality_parameter_t(effect_size, static_cast<double>(n));

    if (alternative == "two.sided") {
        // Two-sided test: P(|T| > t_crit | ncp)
        // Simple approximation: approximate with normal distribution
        double z_crit = detail::critical_z_two_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief Calculate required sample size for one-sample t-test
 *
 * @param effect_size Effect size (Cohen's d)
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Required sample size
 * @throws std::invalid_argument If parameters are invalid
 */
inline std::size_t sample_size_t_test_one_sample(double effect_size, double power = 0.80,
                                                  double alpha = 0.05,
                                                  const std::string& alternative = "two.sided")
{
    if (effect_size == 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: effect size must be non-zero");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: alpha must be in (0, 1)");
    }

    // Initial estimate using normal approximation
    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    // Initial estimate: n = ((z_alpha + z_beta) / d)^2
    double n_approx = std::pow((z_alpha + z_beta) / std::abs(effect_size), 2.0);

    // Minimum sample size
    std::size_t n = std::max(2.0, std::ceil(n_approx));

    // Refine with iteration (maximum 100 iterations)
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_t_test_one_sample(effect_size, n, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n++;
    }

    return n;
}

// ============================================================================
// Two-sample t-test power analysis
// ============================================================================

/**
 * @brief Calculate power for two-sample t-test
 *
 * @param effect_size Effect size (Cohen's d)
 * @param n1 Sample size of group 1
 * @param n2 Sample size of group 2
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Statistical power (0.0 to 1.0)
 * @throws std::invalid_argument If parameters are invalid
 *
 * @note This function uses normal distribution approximation. Strictly speaking, the noncentral
 *       t-distribution should be used, but for large total sample sizes (n1 + n2 > 60 approximately),
 *       sufficient accuracy is obtained. For small samples, power may be slightly overestimated.
 */
inline double power_t_test_two_sample(double effect_size, std::size_t n1, std::size_t n2,
                                      double alpha = 0.05,
                                      const std::string& alternative = "two.sided")
{
    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: sample sizes must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: alpha must be in (0, 1)");
    }

    double ncp = detail::noncentrality_parameter_t_two_sample(effect_size,
                                                                static_cast<double>(n1),
                                                                static_cast<double>(n2));

    if (alternative == "two.sided") {
        double z_crit = detail::critical_z_two_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief Calculate required sample size for two-sample t-test (per group)
 *
 * @param effect_size Effect size (Cohen's d)
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param ratio Ratio n2/n1 (default: 1.0 = equal sizes)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Required sample size for group 1 (group 2 size is n1 * ratio)
 * @throws std::invalid_argument If parameters are invalid
 */
inline std::size_t sample_size_t_test_two_sample(double effect_size, double power = 0.80,
                                                  double alpha = 0.05, double ratio = 1.0,
                                                  const std::string& alternative = "two.sided")
{
    if (effect_size == 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: effect size must be non-zero");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: alpha must be in (0, 1)");
    }
    if (ratio <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: ratio must be positive");
    }

    // Normal approximation
    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    // Initial estimate: n1 = ((z_alpha + z_beta) / d)^2 * (1 + 1/r)
    double n1_approx = std::pow((z_alpha + z_beta) / std::abs(effect_size), 2.0) * (1.0 + 1.0 / ratio);

    std::size_t n1 = std::max(2.0, std::ceil(n1_approx));
    std::size_t n2 = std::max(2.0, std::ceil(n1 * ratio));

    // Refine with iteration
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_t_test_two_sample(effect_size, n1, n2, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n1++;
        n2 = std::max(2.0, std::ceil(n1 * ratio));
    }

    return n1;
}

// ============================================================================
// Proportion test power analysis
// ============================================================================

/**
 * @brief Calculate power for two-sample proportion test
 *
 * @param p1 Proportion of group 1
 * @param p2 Proportion of group 2
 * @param n Sample size per group
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Statistical power (0.0 to 1.0)
 * @throws std::invalid_argument If parameters are invalid
 */
inline double power_prop_test(double p1, double p2, std::size_t n,
                              double alpha = 0.05,
                              const std::string& alternative = "two.sided")
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::power_prop_test: proportions must be in [0, 1]");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::power_prop_test: sample size must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_prop_test: alpha must be in (0, 1)");
    }

    // Standard error under alternative hypothesis
    double se_alt = std::sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n);

    double diff = p1 - p2;

    if (alternative == "two.sided") {
        double z_crit = detail::critical_z_two_sided(alpha);
        double ncp = diff / se_alt;
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        double ncp = diff / se_alt;
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        double ncp = diff / se_alt;
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_prop_test: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief Calculate required sample size for two-sample proportion test
 *
 * @param p1 Proportion of group 1
 * @param p2 Proportion of group 2
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Required sample size per group
 * @throws std::invalid_argument If parameters are invalid
 */
inline std::size_t sample_size_prop_test(double p1, double p2, double power = 0.80,
                                         double alpha = 0.05,
                                         const std::string& alternative = "two.sided")
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: proportions must be in [0, 1]");
    }
    if (std::abs(p1 - p2) < 1e-10) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: proportions must be different");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: alpha must be in (0, 1)");
    }

    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    double p_pooled = (p1 + p2) / 2.0;
    double diff = std::abs(p1 - p2);

    // Initial estimate
    double n_approx = (std::pow(z_alpha * std::sqrt(2.0 * p_pooled * (1.0 - p_pooled)) +
                                z_beta * std::sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2)), 2.0)) /
                      std::pow(diff, 2.0);

    std::size_t n = std::max(2.0, std::ceil(n_approx));

    // Refine with iteration
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_prop_test(p1, p2, n, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n++;
    }

    return n;
}

// ============================================================================
// Power analysis wrapper functions (returning structs)
// ============================================================================

/**
 * @brief Power analysis for one-sample t-test (calculate power)
 *
 * @param effect_size Effect size
 * @param n Sample size
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Power analysis result
 */
inline power_result power_analysis_t_one_sample(double effect_size, std::size_t n,
                                                 double alpha = 0.05,
                                                 const std::string& alternative = "two.sided")
{
    power_result result;
    result.effect_size = effect_size;
    result.sample_size = static_cast<double>(n);
    result.alpha = alpha;
    result.power = power_t_test_one_sample(effect_size, n, alpha, alternative);
    return result;
}

/**
 * @brief Power analysis for one-sample t-test (calculate sample size)
 *
 * @param effect_size Effect size
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alternative Type of alternative hypothesis (default: "two.sided")
 * @return Power analysis result
 */
inline power_result power_analysis_t_one_sample_n(double effect_size, double power = 0.80,
                                                   double alpha = 0.05,
                                                   const std::string& alternative = "two.sided")
{
    power_result result;
    result.effect_size = effect_size;
    result.power = power;
    result.alpha = alpha;
    result.sample_size = static_cast<double>(sample_size_t_test_one_sample(effect_size, power, alpha, alternative));
    return result;
}

// ============================================================================
// Enum overloads (type-safe alternative to string-based API)
// ============================================================================

/**
 * @brief Calculate power for one-sample t-test (enum overload)
 *
 * @param effect_size Effect size (Cohen's d)
 * @param n Sample size
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Statistical power (0.0 to 1.0)
 */
inline double power_t_test_one_sample(double effect_size, std::size_t n,
                                      double alpha,
                                      alternative_hypothesis alt)
{
    return power_t_test_one_sample(effect_size, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Calculate required sample size for one-sample t-test (enum overload)
 *
 * @param effect_size Effect size (Cohen's d)
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Required sample size
 */
inline std::size_t sample_size_t_test_one_sample(double effect_size, double power,
                                                  double alpha,
                                                  alternative_hypothesis alt)
{
    return sample_size_t_test_one_sample(effect_size, power, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Calculate power for two-sample t-test (enum overload)
 *
 * @param effect_size Effect size (Cohen's d)
 * @param n1 Sample size of group 1
 * @param n2 Sample size of group 2
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Statistical power (0.0 to 1.0)
 */
inline double power_t_test_two_sample(double effect_size, std::size_t n1, std::size_t n2,
                                      double alpha,
                                      alternative_hypothesis alt)
{
    return power_t_test_two_sample(effect_size, n1, n2, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Calculate required sample size for two-sample t-test (enum overload)
 *
 * @param effect_size Effect size (Cohen's d)
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param ratio Ratio n2/n1 (default: 1.0)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Required sample size for group 1
 */
inline std::size_t sample_size_t_test_two_sample(double effect_size, double power,
                                                  double alpha, double ratio,
                                                  alternative_hypothesis alt)
{
    return sample_size_t_test_two_sample(effect_size, power, alpha, ratio, detail::alternative_to_string(alt));
}

/**
 * @brief Calculate power for two-sample proportion test (enum overload)
 *
 * @param p1 Proportion of group 1
 * @param p2 Proportion of group 2
 * @param n Sample size per group
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Statistical power (0.0 to 1.0)
 */
inline double power_prop_test(double p1, double p2, std::size_t n,
                              double alpha,
                              alternative_hypothesis alt)
{
    return power_prop_test(p1, p2, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Calculate required sample size for two-sample proportion test (enum overload)
 *
 * @param p1 Proportion of group 1
 * @param p2 Proportion of group 2
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Required sample size per group
 */
inline std::size_t sample_size_prop_test(double p1, double p2, double power,
                                         double alpha,
                                         alternative_hypothesis alt)
{
    return sample_size_prop_test(p1, p2, power, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Power analysis for one-sample t-test (enum overload, returning struct)
 *
 * @param effect_size Effect size
 * @param n Sample size
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Power analysis result
 */
inline power_result power_analysis_t_one_sample(double effect_size, std::size_t n,
                                                 double alpha,
                                                 alternative_hypothesis alt)
{
    return power_analysis_t_one_sample(effect_size, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief Power analysis for one-sample t-test, sample size (enum overload, returning struct)
 *
 * @param effect_size Effect size
 * @param power Target power (default: 0.80)
 * @param alpha Significance level (default: 0.05)
 * @param alt Type of alternative hypothesis (default: two_sided)
 * @return Power analysis result
 */
inline power_result power_analysis_t_one_sample_n(double effect_size, double power,
                                                   double alpha,
                                                   alternative_hypothesis alt)
{
    return power_analysis_t_one_sample_n(effect_size, power, alpha, detail::alternative_to_string(alt));
}

} // namespace statcpp
