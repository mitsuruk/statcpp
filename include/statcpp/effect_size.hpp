/**
 * @file effect_size.hpp
 * @brief Effect size calculation and interpretation
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

#include <cmath>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Cohen's d (Standardized Mean Difference)
// ============================================================================

/**
 * @brief Cohen's d (one-sample, known population standard deviation)
 *
 * Calculates the standardized mean difference. Used when the population
 * standard deviation is known.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param mu0 Population mean to compare against
 * @param sigma Population standard deviation
 * @return Cohen's d
 * @throws std::invalid_argument If the range is empty or sigma is not positive
 */
template <typename Iterator>
double cohens_d(Iterator first, Iterator last, double mu0, double sigma)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::cohens_d: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::cohens_d: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    return (mean_val - mu0) / sigma;
}

/**
 * @brief Cohen's d (one-sample, using sample standard deviation)
 *
 * Calculates the standardized mean difference using the sample standard deviation.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param mu0 Population mean to compare against
 * @return Cohen's d
 * @throws std::invalid_argument If there are fewer than 2 elements or variance is zero
 */
template <typename Iterator>
double cohens_d(Iterator first, Iterator last, double mu0)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::cohens_d: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::cohens_d: zero variance");
    }

    return (mean_val - mu0) / s;
}

/**
 * @brief Cohen's d (two-sample, pooled standard deviation)
 *
 * Calculates the standardized mean difference between two groups.
 * Uses pooled standard deviation.
 *
 * @tparam Iterator1 RandomAccessIterator type for first sample
 * @tparam Iterator2 RandomAccessIterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 Ending iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 Ending iterator of second sample
 * @return Cohen's d
 * @throws std::invalid_argument If either sample has fewer than 2 elements or pooled variance is zero
 */
template <typename Iterator1, typename Iterator2>
double cohens_d_two_sample(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::cohens_d_two_sample: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    // Pooled standard deviation
    double sp = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));

    if (sp == 0.0) {
        throw std::invalid_argument("statcpp::cohens_d_two_sample: zero pooled variance");
    }

    return (mean1 - mean2) / sp;
}

// ============================================================================
// Hedges' g (Bias-corrected Cohen's d)
// ============================================================================

/**
 * @brief Hedges' bias correction factor J
 *
 * Calculates the correction factor for small sample size bias.
 *
 * @param df Degrees of freedom
 * @return Bias correction factor
 */
inline double hedges_correction_factor(double df)
{
    // J ≈ 1 - 3 / (4 * df - 1)
    return 1.0 - 3.0 / (4.0 * df - 1.0);
}

/**
 * @brief Hedges' g (one-sample)
 *
 * Calculates bias-corrected Cohen's d.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param mu0 Population mean to compare against
 * @return Hedges' g
 * @throws std::invalid_argument If there are fewer than 2 elements
 */
template <typename Iterator>
double hedges_g(Iterator first, Iterator last, double mu0)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::hedges_g: need at least 2 elements");
    }

    double d = cohens_d(first, last, mu0);
    double df = static_cast<double>(n - 1);
    return d * hedges_correction_factor(df);
}

/**
 * @brief Hedges' g (two-sample)
 *
 * Calculates bias-corrected standardized mean difference between two groups.
 *
 * @tparam Iterator1 RandomAccessIterator type for first sample
 * @tparam Iterator2 RandomAccessIterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 Ending iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 Ending iterator of second sample
 * @return Hedges' g
 * @throws std::invalid_argument If either sample has fewer than 2 elements
 */
template <typename Iterator1, typename Iterator2>
double hedges_g_two_sample(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::hedges_g_two_sample: need at least 2 elements in each sample");
    }

    double d = cohens_d_two_sample(first1, last1, first2, last2);
    double df = static_cast<double>(n1 + n2 - 2);
    return d * hedges_correction_factor(df);
}

// ============================================================================
// Glass's Delta (using control group's SD)
// ============================================================================

/**
 * @brief Glass's Delta (using control group's standard deviation)
 *
 * Calculates the effect size using only the control group's standard deviation.
 * Useful when the variances of the experimental and control groups differ substantially.
 *
 * @tparam Iterator1 RandomAccessIterator type for control group
 * @tparam Iterator2 RandomAccessIterator type for treatment group
 * @param control_first Beginning iterator of control group
 * @param control_last Ending iterator of control group
 * @param treatment_first Beginning iterator of treatment group
 * @param treatment_last Ending iterator of treatment group
 * @return Glass's Delta
 * @throws std::invalid_argument If control group has fewer than 2 elements, treatment group is empty, or control group has zero variance
 */
template <typename Iterator1, typename Iterator2>
double glass_delta(Iterator1 control_first, Iterator1 control_last,
                   Iterator2 treatment_first, Iterator2 treatment_last)
{
    auto n1 = statcpp::count(control_first, control_last);
    auto n2 = statcpp::count(treatment_first, treatment_last);

    if (n1 < 2) {
        throw std::invalid_argument("statcpp::glass_delta: control group needs at least 2 elements");
    }
    if (n2 == 0) {
        throw std::invalid_argument("statcpp::glass_delta: treatment group is empty");
    }

    double mean1 = statcpp::mean(control_first, control_last);
    double mean2 = statcpp::mean(treatment_first, treatment_last);
    double s1 = statcpp::sample_stddev(control_first, control_last);

    if (s1 == 0.0) {
        throw std::invalid_argument("statcpp::glass_delta: control group has zero variance");
    }

    return (mean2 - mean1) / s1;
}

// ============================================================================
// Correlation-based Effect Size (r)
// ============================================================================

/**
 * @brief Convert t-value to correlation coefficient
 *
 * @param t t-statistic
 * @param df Degrees of freedom
 * @return Correlation coefficient
 */
inline double t_to_r(double t, double df)
{
    return t / std::sqrt(t * t + df);
}

/**
 * @brief Convert Cohen's d to correlation coefficient
 *
 * @param d Cohen's d
 * @return Correlation coefficient
 */
inline double d_to_r(double d)
{
    // r = d / sqrt(d^2 + 4)
    return d / std::sqrt(d * d + 4.0);
}

/**
 * @brief Convert correlation coefficient to Cohen's d
 *
 * @param r Correlation coefficient
 * @return Cohen's d
 * @throws std::invalid_argument If |r| >= 1
 */
inline double r_to_d(double r)
{
    // d = 2r / sqrt(1 - r^2)
    if (std::abs(r) >= 1.0) {
        throw std::invalid_argument("statcpp::r_to_d: |r| must be less than 1");
    }
    return 2.0 * r / std::sqrt(1.0 - r * r);
}

// ============================================================================
// Eta-squared and Partial Eta-squared
// ============================================================================

/**
 * @brief Calculate eta-squared from F-test
 *
 * Calculates the effect size from the sum of squares for effect and total.
 *
 * @param ss_effect Sum of squares for effect
 * @param ss_total Total sum of squares
 * @return Eta-squared
 * @throws std::invalid_argument If ss_total is not positive
 */
inline double eta_squared(double ss_effect, double ss_total)
{
    if (ss_total <= 0.0) {
        throw std::invalid_argument("statcpp::eta_squared: ss_total must be positive");
    }
    return ss_effect / ss_total;
}

/**
 * @brief Calculate partial eta-squared from F-test
 *
 * @param f F-statistic
 * @param df1 Numerator degrees of freedom
 * @param df2 Denominator degrees of freedom
 * @return Partial eta-squared
 */
inline double partial_eta_squared(double f, double df1, double df2)
{
    return (f * df1) / (f * df1 + df2);
}

// ============================================================================
// Omega-squared (less biased than eta-squared)
// ============================================================================

/**
 * @brief Omega-squared
 *
 * A less biased estimate of effect size than eta-squared.
 *
 * @param ss_effect Sum of squares for effect
 * @param ss_total Total sum of squares
 * @param ms_error Mean square error
 * @param df_effect Degrees of freedom for effect
 * @return Omega-squared
 * @throws std::invalid_argument If ss_total is not positive
 */
inline double omega_squared(double ss_effect, double ss_total, double ms_error, double df_effect)
{
    if (ss_total <= 0.0) {
        throw std::invalid_argument("statcpp::omega_squared: ss_total must be positive");
    }
    return (ss_effect - df_effect * ms_error) / (ss_total + ms_error);
}

// ============================================================================
// Cohen's h (Effect Size for Proportions)
// ============================================================================

/**
 * @brief Cohen's h (effect size for difference between two proportions)
 *
 * Calculates the effect size representing the difference between two proportions.
 *
 * @param p1 Proportion in first group
 * @param p2 Proportion in second group
 * @return Cohen's h
 * @throws std::invalid_argument If proportions are outside [0, 1] range
 */
inline double cohens_h(double p1, double p2)
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::cohens_h: proportions must be in [0, 1]");
    }

    // h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    return 2.0 * (std::asin(std::sqrt(p1)) - std::asin(std::sqrt(p2)));
}

// ============================================================================
// Odds Ratio and Risk Ratio
// ============================================================================

/**
 * @brief Odds ratio
 *
 * Calculates the odds ratio from a 2x2 contingency table.
 *
 * @param a Cell (1,1) frequency (exposed, disease present)
 * @param b Cell (1,2) frequency (exposed, disease absent)
 * @param c Cell (2,1) frequency (unexposed, disease present)
 * @param d Cell (2,2) frequency (unexposed, disease absent)
 * @return Odds ratio
 * @throws std::invalid_argument If b or c is zero
 */
inline double odds_ratio(double a, double b, double c, double d)
{
    // 2x2 table: a, b, c, d
    if (b == 0.0 || c == 0.0) {
        throw std::invalid_argument("statcpp::odds_ratio: cell b or c is zero");
    }
    return (a * d) / (b * c);
}

/**
 * @brief Relative risk (risk ratio)
 *
 * Calculates the relative risk from a 2x2 contingency table.
 *
 * @param a Cell (1,1) frequency (exposed, disease present)
 * @param b Cell (1,2) frequency (exposed, disease absent)
 * @param c Cell (2,1) frequency (unexposed, disease present)
 * @param d Cell (2,2) frequency (unexposed, disease absent)
 * @return Relative risk
 * @throws std::invalid_argument If row total is zero or risk in group 2 is zero
 */
inline double risk_ratio(double a, double b, double c, double d)
{
    // Risk in group 1: a / (a + b)
    // Risk in group 2: c / (c + d)
    if (a + b == 0.0 || c + d == 0.0) {
        throw std::invalid_argument("statcpp::risk_ratio: row total is zero");
    }
    double risk1 = a / (a + b);
    double risk2 = c / (c + d);

    if (risk2 == 0.0) {
        throw std::invalid_argument("statcpp::risk_ratio: risk in group 2 is zero");
    }

    return risk1 / risk2;
}

// ============================================================================
// Effect Size Interpretation
// ============================================================================

/**
 * @brief Enumeration for effect size magnitude
 */
enum class effect_size_magnitude {
    negligible,  ///< Negligible
    small,       ///< Small
    medium,      ///< Medium
    large        ///< Large
};

/**
 * @brief Interpret Cohen's d
 *
 * Determines the magnitude of effect size based on Cohen (1988) criteria.
 *
 * @param d Cohen's d
 * @return Effect size magnitude
 */
inline effect_size_magnitude interpret_cohens_d(double d)
{
    double abs_d = std::abs(d);
    if (abs_d < 0.2) return effect_size_magnitude::negligible;
    if (abs_d < 0.5) return effect_size_magnitude::small;
    if (abs_d < 0.8) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

/**
 * @brief Interpret correlation coefficient
 *
 * Determines the strength of correlation based on Cohen (1988) criteria.
 *
 * @param r Correlation coefficient
 * @return Effect size magnitude
 */
inline effect_size_magnitude interpret_correlation(double r)
{
    double abs_r = std::abs(r);
    if (abs_r < 0.1) return effect_size_magnitude::negligible;
    if (abs_r < 0.3) return effect_size_magnitude::small;
    if (abs_r < 0.5) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

/**
 * @brief Interpret eta-squared
 *
 * Determines the magnitude of effect size based on Cohen (1988) criteria.
 *
 * @param eta2 Eta-squared
 * @return Effect size magnitude
 */
inline effect_size_magnitude interpret_eta_squared(double eta2)
{
    if (eta2 < 0.01) return effect_size_magnitude::negligible;
    if (eta2 < 0.06) return effect_size_magnitude::small;
    if (eta2 < 0.14) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

} // namespace statcpp
