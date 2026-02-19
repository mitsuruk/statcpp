/**
 * @file parametric_tests.hpp
 * @brief Parametric test functions
 *
 * Provides parametric statistical tests including t-tests, z-tests, F-tests, and chi-square tests.
 * Also includes multiple testing corrections (Bonferroni, Benjamini-Hochberg, Holm).
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Test Result Structure
// ============================================================================

/**
 * @brief Enumeration representing the type of alternative hypothesis
 *
 * Specifies the direction of the alternative hypothesis in statistical tests.
 */
enum class alternative_hypothesis {
    two_sided,  ///< Two-sided test
    less,       ///< One-sided test (less than)
    greater     ///< One-sided test (greater than)
};

/**
 * @brief Structure to store statistical test results
 *
 * Holds the test statistic, p-value, degrees of freedom, and alternative hypothesis information.
 */
struct test_result {
    double statistic;                ///< Test statistic
    double p_value;                  ///< p-value
    double df;                       ///< Degrees of freedom
    alternative_hypothesis alternative;  ///< Type of alternative hypothesis
    double df2 = std::numeric_limits<double>::quiet_NaN();  ///< Second degrees of freedom (used by F-test)
};

// ============================================================================
// Z-Test for Mean (known variance)
// ============================================================================

/**
 * @brief One-sample z-test (known variance)
 *
 * Tests whether the sample mean equals a specific value when the population variance is known.
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of sample data
 * @param last End iterator of sample data
 * @param mu0 Population mean under null hypothesis
 * @param sigma Known population standard deviation
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (z-statistic, p-value, df=infinity)
 * @throws std::invalid_argument If sigma is not positive or range is empty
 */
template <typename Iterator>
test_result z_test(Iterator first, Iterator last, double mu0, double sigma,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::z_test: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::z_test: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    double se = sigma / std::sqrt(static_cast<double>(n));
    double z = (mean_val - mu0) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

// ============================================================================
// Z-Test for Proportion
// ============================================================================

/**
 * @brief One-sample proportion z-test
 *
 * Tests whether the sample proportion equals a specific population proportion.
 *
 * @param successes Number of successes
 * @param trials Number of trials
 * @param p0 Population proportion under null hypothesis
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (z-statistic, p-value, df=infinity)
 * @throws std::invalid_argument If p0 is outside (0,1), trials is 0, or successes exceeds trials
 */
inline test_result z_test_proportion(std::size_t successes, std::size_t trials, double p0,
                                     alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (p0 <= 0.0 || p0 >= 1.0) {
        throw std::invalid_argument("statcpp::z_test_proportion: p0 must be in (0, 1)");
    }
    if (trials == 0) {
        throw std::invalid_argument("statcpp::z_test_proportion: trials must be positive");
    }
    if (successes > trials) {
        throw std::invalid_argument("statcpp::z_test_proportion: successes cannot exceed trials");
    }

    double n = static_cast<double>(trials);
    double p_hat = static_cast<double>(successes) / n;
    double se = std::sqrt(p0 * (1.0 - p0) / n);
    double z = (p_hat - p0) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

/**
 * @brief Two-sample proportion z-test
 *
 * Tests whether two sample proportions are equal (using pooled proportion).
 *
 * @param successes1 Number of successes in first sample
 * @param trials1 Number of trials in first sample
 * @param successes2 Number of successes in second sample
 * @param trials2 Number of trials in second sample
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (z-statistic, p-value, df=infinity)
 * @throws std::invalid_argument If trials is 0 or successes exceeds trials
 */
inline test_result z_test_proportion_two_sample(std::size_t successes1, std::size_t trials1,
                                                 std::size_t successes2, std::size_t trials2,
                                                 alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    if (trials1 == 0 || trials2 == 0) {
        throw std::invalid_argument("statcpp::z_test_proportion_two_sample: trials must be positive");
    }
    if (successes1 > trials1 || successes2 > trials2) {
        throw std::invalid_argument("statcpp::z_test_proportion_two_sample: successes cannot exceed trials");
    }

    double n1 = static_cast<double>(trials1);
    double n2 = static_cast<double>(trials2);
    double p1 = static_cast<double>(successes1) / n1;
    double p2 = static_cast<double>(successes2) / n2;

    // Pooled proportion
    double p_pooled = static_cast<double>(successes1 + successes2) / (n1 + n2);
    double se = std::sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2));

    double z = (p1 - p2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - norm_cdf(std::abs(z)));
            break;
    }

    return {z, p_value, std::numeric_limits<double>::infinity(), alt};
}

// ============================================================================
// T-Test for Mean
// ============================================================================

/**
 * @brief One-sample t-test
 *
 * Tests whether the sample mean equals a specific value (unknown population variance).
 *
 * @tparam Iterator Input iterator type
 * @param first Beginning iterator of sample data
 * @param last End iterator of sample data
 * @param mu0 Population mean under null hypothesis
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (t-statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If number of elements is less than 2 or variance is zero
 */
template <typename Iterator>
test_result t_test(Iterator first, Iterator last, double mu0,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::t_test: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::t_test: zero variance");
    }

    double se = s / std::sqrt(static_cast<double>(n));
    double t = (mean_val - mu0) / se;
    double df = static_cast<double>(n - 1);

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief Two-sample t-test (independent samples, pooled variance)
 *
 * Tests whether the means of two independent samples are equal.
 * Assumes equal variance and uses pooled variance.
 *
 * @tparam Iterator1 Input iterator type for first sample
 * @tparam Iterator2 Input iterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (t-statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If either sample has less than 2 elements or variance is zero
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_two_sample(Iterator1 first1, Iterator1 last1,
                               Iterator2 first2, Iterator2 last2,
                               alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::t_test_two_sample: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    // Pooled variance
    double df = static_cast<double>(n1 + n2 - 2);
    double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df;
    double se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));

    if (se == 0.0) {
        throw std::invalid_argument("statcpp::t_test_two_sample: zero variance");
    }

    double t = (mean1 - mean2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief Two-sample t-test (Welch's method)
 *
 * Tests whether the means of two independent samples are equal.
 * Does not assume equal variance; uses Welch-Satterthwaite approximation for degrees of freedom.
 *
 * @tparam Iterator1 Input iterator type for first sample
 * @tparam Iterator2 Input iterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (t-statistic, p-value, Welch approximation degrees of freedom)
 * @throws std::invalid_argument If either sample has less than 2 elements or variance is zero
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_welch(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::t_test_welch: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    double se1 = var1 / n1;
    double se2 = var2 / n2;
    double se = std::sqrt(se1 + se2);

    if (se == 0.0) {
        throw std::invalid_argument("statcpp::t_test_welch: zero variance");
    }

    // Welch-Satterthwaite approximation
    double num = (se1 + se2) * (se1 + se2);
    double denom = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1);

    // Protection when denominator is zero (when both variances are zero)
    if (denom == 0.0) {
        throw std::invalid_argument("statcpp::t_test_welch: cannot compute degrees of freedom with zero variances");
    }

    double df = num / denom;

    double t = (mean1 - mean2) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = t_cdf(t, df);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - t_cdf(t, df);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * (1.0 - t_cdf(std::abs(t), df));
            break;
    }

    return {t, p_value, df, alt};
}

/**
 * @brief Paired t-test
 *
 * Tests whether the mean of differences between two paired samples equals zero.
 *
 * @tparam Iterator1 Input iterator type for first sample
 * @tparam Iterator2 Input iterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (t-statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If sample lengths differ or are less than 2
 */
template <typename Iterator1, typename Iterator2>
test_result t_test_paired(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2,
                           alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::t_test_paired: samples must have equal length");
    }
    if (n1 < 2) {
        throw std::invalid_argument("statcpp::t_test_paired: need at least 2 pairs");
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

    // Apply one-sample t-test on differences
    return t_test(diffs.begin(), diffs.end(), 0.0, alt);
}

// ============================================================================
// Chi-Square Test for Goodness of Fit
// ============================================================================

/**
 * @brief Chi-square goodness of fit test
 *
 * Tests whether observed frequencies match expected frequencies.
 *
 * @tparam Iterator1 Input iterator type for observed frequencies
 * @tparam Iterator2 Input iterator type for expected frequencies
 * @param observed_first Beginning iterator of observed frequencies
 * @param observed_last End iterator of observed frequencies
 * @param expected_first Beginning iterator of expected frequencies
 * @param expected_last End iterator of expected frequencies
 * @return test_result Test result (chi-square statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If observed and expected lengths differ, fewer than 2 categories, or expected frequency is non-positive
 */
template <typename Iterator1, typename Iterator2>
test_result chisq_test_gof(Iterator1 observed_first, Iterator1 observed_last,
                           Iterator2 expected_first, Iterator2 expected_last)
{
    auto n_obs = statcpp::count(observed_first, observed_last);
    auto n_exp = statcpp::count(expected_first, expected_last);

    if (n_obs != n_exp) {
        throw std::invalid_argument("statcpp::chisq_test_gof: observed and expected must have same length");
    }
    if (n_obs < 2) {
        throw std::invalid_argument("statcpp::chisq_test_gof: need at least 2 categories");
    }

    double chi2 = 0.0;
    auto it_obs = observed_first;
    auto it_exp = expected_first;

    while (it_obs != observed_last) {
        double o = static_cast<double>(*it_obs);
        double e = static_cast<double>(*it_exp);

        if (e <= 0.0) {
            throw std::invalid_argument("statcpp::chisq_test_gof: expected values must be positive");
        }

        chi2 += (o - e) * (o - e) / e;

        ++it_obs;
        ++it_exp;
    }

    double df = static_cast<double>(n_obs - 1);
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

/**
 * @brief Chi-square goodness of fit test (uniform expected frequencies)
 *
 * Tests whether observed frequencies follow a uniform distribution.
 *
 * @tparam Iterator Input iterator type for observed frequencies
 * @param observed_first Beginning iterator of observed frequencies
 * @param observed_last End iterator of observed frequencies
 * @return test_result Test result (chi-square statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If fewer than 2 categories
 */
template <typename Iterator>
test_result chisq_test_gof_uniform(Iterator observed_first, Iterator observed_last)
{
    auto n = statcpp::count(observed_first, observed_last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::chisq_test_gof_uniform: need at least 2 categories");
    }

    double total = 0.0;
    for (auto it = observed_first; it != observed_last; ++it) {
        total += static_cast<double>(*it);
    }

    double expected = total / static_cast<double>(n);

    double chi2 = 0.0;
    for (auto it = observed_first; it != observed_last; ++it) {
        double o = static_cast<double>(*it);
        chi2 += (o - expected) * (o - expected) / expected;
    }

    double df = static_cast<double>(n - 1);
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// Chi-Square Test for Independence
// ============================================================================

/**
 * @brief Chi-square test for independence
 *
 * Tests independence of two variables based on a contingency table.
 *
 * @param contingency_table Contingency table (2D array in row-major order)
 * @return test_result Test result (chi-square statistic, p-value, degrees of freedom)
 * @throws std::invalid_argument If rows or columns are less than 2, column counts are inconsistent, negative values exist, or table is empty
 */
inline test_result chisq_test_independence(const std::vector<std::vector<double>>& contingency_table)
{
    std::size_t rows = contingency_table.size();
    if (rows < 2) {
        throw std::invalid_argument("statcpp::chisq_test_independence: need at least 2 rows");
    }

    std::size_t cols = contingency_table[0].size();
    if (cols < 2) {
        throw std::invalid_argument("statcpp::chisq_test_independence: need at least 2 columns");
    }

    // Check all rows have same number of columns
    for (const auto& row : contingency_table) {
        if (row.size() != cols) {
            throw std::invalid_argument("statcpp::chisq_test_independence: inconsistent column count");
        }
    }

    // Compute row and column totals
    std::vector<double> row_totals(rows, 0.0);
    std::vector<double> col_totals(cols, 0.0);
    double grand_total = 0.0;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double val = contingency_table[i][j];
            if (val < 0.0) {
                throw std::invalid_argument("statcpp::chisq_test_independence: negative cell value");
            }
            row_totals[i] += val;
            col_totals[j] += val;
            grand_total += val;
        }
    }

    if (grand_total == 0.0) {
        throw std::invalid_argument("statcpp::chisq_test_independence: empty table");
    }

    // Compute chi-square statistic
    double chi2 = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double expected = row_totals[i] * col_totals[j] / grand_total;
            if (expected > 0.0) {
                double observed = contingency_table[i][j];
                chi2 += (observed - expected) * (observed - expected) / expected;
            }
        }
    }

    double df = static_cast<double>((rows - 1) * (cols - 1));
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// F-Test for Variance Ratio
// ============================================================================

/**
 * @brief F-test (variance comparison)
 *
 * Tests whether the variances of two samples are equal.
 *
 * @tparam Iterator1 Input iterator type for first sample
 * @tparam Iterator2 Input iterator type for second sample
 * @param first1 Beginning iterator of first sample
 * @param last1 End iterator of first sample
 * @param first2 Beginning iterator of second sample
 * @param last2 End iterator of second sample
 * @param alt Type of alternative hypothesis (default: two-sided)
 * @return test_result Test result (F-statistic, p-value, df=df1, df2=df2)
 * @throws std::invalid_argument If either sample has less than 2 elements or second sample variance is zero
 */
template <typename Iterator1, typename Iterator2>
test_result f_test(Iterator1 first1, Iterator1 last1,
                   Iterator2 first2, Iterator2 last2,
                   alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::f_test: need at least 2 elements in each sample");
    }

    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    if (var2 == 0.0) {
        throw std::invalid_argument("statcpp::f_test: second sample has zero variance");
    }

    double f = var1 / var2;
    double df1 = static_cast<double>(n1 - 1);
    double df2 = static_cast<double>(n2 - 1);

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = f_cdf(f, df1, df2);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - f_cdf(f, df1, df2);
            break;
        case alternative_hypothesis::two_sided:
        default:
            {
                double p1 = f_cdf(f, df1, df2);
                double p2 = 1.0 - p1;
                p_value = 2.0 * std::min(p1, p2);
            }
            break;
    }

    return {f, p_value, df1, alt, df2};
}

// ============================================================================
// Multiple Testing Correction
// ============================================================================

/**
 * @brief Bonferroni correction
 *
 * Corrects p-values for multiple testing using the Bonferroni method.
 * Each p-value is multiplied by the number of tests, capped at 1.
 *
 * @param p_values Vector of original p-values
 * @return std::vector<double> Vector of corrected p-values
 */
inline std::vector<double> bonferroni_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    std::vector<double> adjusted(n);

    for (std::size_t i = 0; i < n; ++i) {
        adjusted[i] = std::min(1.0, p_values[i] * static_cast<double>(n));
    }

    return adjusted;
}

/**
 * @brief Benjamini-Hochberg correction (FDR control)
 *
 * Corrects p-values for multiple testing using the Benjamini-Hochberg method.
 * Performs step-up correction to control the false discovery rate (FDR).
 *
 * @param p_values Vector of original p-values
 * @return std::vector<double> Vector of corrected p-values
 */
inline std::vector<double> benjamini_hochberg_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    if (n == 0) return {};

    // Create index-pvalue pairs and sort by p-value
    std::vector<std::pair<std::size_t, double>> indexed(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed[i] = {i, p_values[i]};
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<double> adjusted(n);

    // Compute adjusted p-values
    double prev_adj = 1.0;
    for (std::size_t i = n; i > 0; --i) {
        std::size_t idx = indexed[i - 1].first;
        double p = indexed[i - 1].second;
        double adj = p * static_cast<double>(n) / static_cast<double>(i);
        adj = std::min(adj, prev_adj);
        adj = std::min(adj, 1.0);
        adjusted[idx] = adj;
        prev_adj = adj;
    }

    return adjusted;
}

/**
 * @brief Holm correction (step-down Bonferroni method)
 *
 * Corrects p-values for multiple testing using the Holm method.
 * A step-down version of Bonferroni correction with higher power.
 *
 * @param p_values Vector of original p-values
 * @return std::vector<double> Vector of corrected p-values
 */
inline std::vector<double> holm_correction(const std::vector<double>& p_values)
{
    std::size_t n = p_values.size();
    if (n == 0) return {};

    // Create index-pvalue pairs and sort by p-value
    std::vector<std::pair<std::size_t, double>> indexed(n);
    for (std::size_t i = 0; i < n; ++i) {
        indexed[i] = {i, p_values[i]};
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<double> adjusted(n);

    double max_adj = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = indexed[i].first;
        double p = indexed[i].second;
        double adj = p * static_cast<double>(n - i);
        adj = std::max(adj, max_adj);
        adj = std::min(adj, 1.0);
        adjusted[idx] = adj;
        max_adj = adj;
    }

    return adjusted;
}

} // namespace statcpp
