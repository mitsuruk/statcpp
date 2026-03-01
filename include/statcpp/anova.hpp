/**
 * @file anova.hpp
 * @brief Analysis of Variance (ANOVA)
 *
 * Provides ANOVA-related functions including one-way ANOVA, two-way ANOVA,
 * analysis of covariance, and post-hoc tests.
 */

#pragma once

#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// ANOVA Result Structures
// ============================================================================

/**
 * @brief Structure representing a row in the ANOVA table
 *
 * Holds statistics corresponding to each source of variation (between groups, within groups, etc.)
 * in the ANOVA table.
 */
struct anova_row {
    std::string source;     ///< Name of the source of variation
    double ss;              ///< Sum of Squares
    double df;              ///< Degrees of freedom
    double ms;              ///< Mean Square
    double f_statistic;     ///< F statistic
    double p_value;         ///< p-value
};

/**
 * @brief Structure storing one-way ANOVA results
 *
 * Holds all results from one-way ANOVA (between/within variation, statistics, group information).
 */
struct one_way_anova_result {
    anova_row between;      ///< Between-group variation
    anova_row within;       ///< Within-group variation (residual)
    double ss_total;        ///< Total sum of squares
    double df_total;        ///< Total degrees of freedom
    std::size_t n_groups;   ///< Number of groups
    std::size_t n_total;    ///< Total number of observations
    double grand_mean;      ///< Grand mean
    std::vector<double> group_means;       ///< Mean of each group
    std::vector<std::size_t> group_sizes;  ///< Size of each group
};

/**
 * @brief Structure storing two-way ANOVA results
 *
 * Holds all results from two-way ANOVA (factor A effect, factor B effect, interaction, error, etc.).
 */
struct two_way_anova_result {
    anova_row factor_a;         ///< Effect of factor A
    anova_row factor_b;         ///< Effect of factor B
    anova_row interaction;      ///< Interaction effect
    anova_row error;            ///< Error (residual)
    double ss_total;            ///< Total sum of squares
    double df_total;            ///< Total degrees of freedom
    std::size_t levels_a;       ///< Number of levels for factor A
    std::size_t levels_b;       ///< Number of levels for factor B
    std::size_t n_total;        ///< Total number of observations
    double grand_mean;          ///< Grand mean
};

/**
 * @brief Structure storing individual pairwise comparison result in post-hoc tests
 *
 * Holds statistics and decision results for a post-hoc comparison between two groups.
 */
struct posthoc_comparison {
    std::size_t group1;     ///< Index of group 1
    std::size_t group2;     ///< Index of group 2
    double mean_diff;       ///< Difference in means
    double se;              ///< Standard error
    double statistic;       ///< Test statistic
    double p_value;         ///< p-value
    double lower;           ///< Confidence interval lower bound
    double upper;           ///< Confidence interval upper bound
    bool significant;       ///< Whether significant
};

/**
 * @brief Structure storing post-hoc comparison results
 *
 * Holds the method name, all comparison results, and parameters used in the post-hoc test.
 */
struct posthoc_result {
    std::string method;                          ///< Method name
    std::vector<posthoc_comparison> comparisons; ///< All comparisons
    double alpha;                                ///< Significance level
    double mse;                                  ///< Mean square error
    double df_error;                             ///< Error degrees of freedom
};

// ============================================================================
// One-Way ANOVA
// ============================================================================

/**
 * @brief Perform one-way analysis of variance
 *
 * Tests whether there are significant differences in means among multiple groups.
 * Uses F-test to compare between-group and within-group variation.
 *
 * @param groups Vector of vectors containing data for each group
 * @return one_way_anova_result Results of the analysis of variance
 * @throws std::invalid_argument If number of groups is less than 2
 * @throws std::invalid_argument If an empty group exists
 * @throws std::invalid_argument If total number of observations is less than or equal to number of groups
 */
inline one_way_anova_result one_way_anova(const std::vector<std::vector<double>>& groups)
{
    std::size_t k = groups.size();  // Number of groups
    if (k < 2) {
        throw std::invalid_argument("statcpp::one_way_anova: need at least 2 groups");
    }

    // Calculate size and mean for each group
    std::vector<std::size_t> group_sizes(k);
    std::vector<double> group_means(k);
    std::size_t n_total = 0;
    double grand_sum = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        if (groups[i].empty()) {
            throw std::invalid_argument("statcpp::one_way_anova: empty group detected");
        }
        group_sizes[i] = groups[i].size();
        n_total += group_sizes[i];
        double group_sum = std::accumulate(groups[i].begin(), groups[i].end(), 0.0);
        group_means[i] = group_sum / static_cast<double>(group_sizes[i]);
        grand_sum += group_sum;
    }

    if (n_total <= k) {
        throw std::invalid_argument("statcpp::one_way_anova: need more observations than groups");
    }

    double grand_mean = grand_sum / static_cast<double>(n_total);

    // Calculate sum of squares
    double ss_between = 0.0;  // Between-group sum of squares
    double ss_within = 0.0;   // Within-group sum of squares

    for (std::size_t i = 0; i < k; ++i) {
        double diff = group_means[i] - grand_mean;
        ss_between += static_cast<double>(group_sizes[i]) * diff * diff;

        for (double x : groups[i]) {
            double diff_within = x - group_means[i];
            ss_within += diff_within * diff_within;
        }
    }

    double ss_total = ss_between + ss_within;

    // Degrees of freedom
    double df_between = static_cast<double>(k - 1);
    double df_within = static_cast<double>(n_total - k);
    double df_total = static_cast<double>(n_total - 1);

    // Mean squares
    double ms_between = ss_between / df_between;
    double ms_within = ss_within / df_within;

    // F statistic and p-value (handle degenerate case ms_within == 0)
    double f_statistic;
    double p_value;
    if (ms_within == 0.0) {
        f_statistic = (ms_between == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        p_value = (ms_between == 0.0) ? 1.0 : 0.0;
    } else {
        f_statistic = ms_between / ms_within;
        p_value = 1.0 - f_cdf(f_statistic, df_between, df_within);
    }

    anova_row between{"Between Groups", ss_between, df_between, ms_between, f_statistic, p_value};
    anova_row within{"Within Groups", ss_within, df_within, ms_within, 0.0, 0.0};

    return {between, within, ss_total, df_total, k, n_total, grand_mean, group_means, group_sizes};
}

// ============================================================================
// Two-Way ANOVA
// ============================================================================

/**
 * @brief Perform two-way analysis of variance (with replication)
 *
 * Tests the effects of two factors (A, B) and their interaction on the dependent variable.
 * Assumes equal cell sizes.
 *
 * @param data 3-dimensional data array. data[i][j] is the vector of observations for factor A=i, factor B=j
 * @return two_way_anova_result Results of the analysis of variance
 * @throws std::invalid_argument If number of levels for factor A is less than 2
 * @throws std::invalid_argument If number of levels for factor B is less than 2
 * @throws std::invalid_argument If number of levels for factor B is inconsistent
 * @throws std::invalid_argument If cell sizes are unequal
 * @throws std::invalid_argument If an empty cell exists
 */
inline two_way_anova_result two_way_anova(
    const std::vector<std::vector<std::vector<double>>>& data)
{
    std::size_t a = data.size();  // Number of levels for factor A
    if (a < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: need at least 2 levels for factor A");
    }

    std::size_t b = data[0].size();  // Number of levels for factor B
    if (b < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: need at least 2 levels for factor B");
    }

    // Check if all levels have the same number of replications
    std::size_t n_rep = data[0][0].size();  // Number of replications
    if (n_rep < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: at least 2 replications per cell are required (n_rep >= 2)");
    }
    std::size_t n_total = 0;
    double grand_sum = 0.0;

    for (std::size_t i = 0; i < a; ++i) {
        if (data[i].size() != b) {
            throw std::invalid_argument("statcpp::two_way_anova: inconsistent number of levels for factor B");
        }
        for (std::size_t j = 0; j < b; ++j) {
            if (data[i][j].size() != n_rep) {
                throw std::invalid_argument("statcpp::two_way_anova: unequal cell sizes not supported");
            }
            n_total += n_rep;
            for (double x : data[i][j]) {
                grand_sum += x;
            }
        }
    }

    double grand_mean = grand_sum / static_cast<double>(n_total);
    double n_rep_d = static_cast<double>(n_rep);
    double a_d = static_cast<double>(a);
    double b_d = static_cast<double>(b);

    // Calculate means for each level
    std::vector<double> mean_a(a, 0.0);  // Mean for each level of factor A
    std::vector<double> mean_b(b, 0.0);  // Mean for each level of factor B
    std::vector<std::vector<double>> mean_ab(a, std::vector<double>(b, 0.0));  // Cell means

    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            double cell_sum = std::accumulate(data[i][j].begin(), data[i][j].end(), 0.0);
            mean_ab[i][j] = cell_sum / n_rep_d;
            mean_a[i] += cell_sum;
            mean_b[j] += cell_sum;
        }
        mean_a[i] /= (b_d * n_rep_d);
    }
    for (std::size_t j = 0; j < b; ++j) {
        mean_b[j] /= (a_d * n_rep_d);
    }

    // Calculate sum of squares
    double ss_a = 0.0;      // Sum of squares for factor A
    double ss_b = 0.0;      // Sum of squares for factor B
    double ss_ab = 0.0;     // Sum of squares for interaction
    double ss_error = 0.0;  // Error sum of squares

    for (std::size_t i = 0; i < a; ++i) {
        double diff_a = mean_a[i] - grand_mean;
        ss_a += diff_a * diff_a;
    }
    ss_a *= b_d * n_rep_d;

    for (std::size_t j = 0; j < b; ++j) {
        double diff_b = mean_b[j] - grand_mean;
        ss_b += diff_b * diff_b;
    }
    ss_b *= a_d * n_rep_d;

    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            double interaction = mean_ab[i][j] - mean_a[i] - mean_b[j] + grand_mean;
            ss_ab += interaction * interaction;

            for (double x : data[i][j]) {
                double error = x - mean_ab[i][j];
                ss_error += error * error;
            }
        }
    }
    ss_ab *= n_rep_d;

    double ss_total = ss_a + ss_b + ss_ab + ss_error;

    // Degrees of freedom
    double df_a = a_d - 1.0;
    double df_b = b_d - 1.0;
    double df_ab = df_a * df_b;
    double df_error = static_cast<double>(n_total) - a_d * b_d;
    double df_total = static_cast<double>(n_total) - 1.0;

    // Mean squares
    double ms_a = ss_a / df_a;
    double ms_b = ss_b / df_b;
    double ms_ab = ss_ab / df_ab;
    double ms_error = ss_error / df_error;

    // F statistics and p-values
    double f_a = ms_a / ms_error;
    double f_b = ms_b / ms_error;
    double f_ab = ms_ab / ms_error;

    double p_a = 1.0 - f_cdf(f_a, df_a, df_error);
    double p_b = 1.0 - f_cdf(f_b, df_b, df_error);
    double p_ab = 1.0 - f_cdf(f_ab, df_ab, df_error);

    anova_row factor_a{"Factor A", ss_a, df_a, ms_a, f_a, p_a};
    anova_row factor_b{"Factor B", ss_b, df_b, ms_b, f_b, p_b};
    anova_row interaction{"A x B", ss_ab, df_ab, ms_ab, f_ab, p_ab};
    anova_row error{"Error", ss_error, df_error, ms_error, 0.0, 0.0};

    return {factor_a, factor_b, interaction, error,
            ss_total, df_total, a, b, n_total, grand_mean};
}

// ============================================================================
// Post-hoc Comparisons
// ============================================================================

/**
 * @brief Perform Tukey's Honestly Significant Difference (HSD) test
 *
 * Performs all pairwise comparisons among groups using the studentized range
 * distribution (Tukey-Kramer method for unequal sample sizes).
 *
 * The q statistic is computed as |mean_i - mean_j| / SE where
 * SE = sqrt(MSE/2 * (1/n_i + 1/n_j)), and p-values are obtained from
 * the studentized range distribution with k groups and df_error degrees
 * of freedom.
 *
 * @param anova_result Result from one-way ANOVA
 * @param groups Data for each group (currently unused; retained for API compatibility
 *               and potential future use such as input validation)
 * @param alpha Significance level (default: 0.05)
 * @return posthoc_result Post-hoc comparison results (statistic field contains q statistic)
 * @throws std::invalid_argument If alpha is outside the range (0, 1)
 */
inline posthoc_result tukey_hsd(const one_way_anova_result& anova_result,
                                 const std::vector<std::vector<double>>& groups,
                                 double alpha = 0.05)
{
    (void)groups;  // Currently unused; all statistics derived from anova_result

    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::tukey_hsd: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;
    double k_d = static_cast<double>(k);

    double q_crit = studentized_range_quantile(1.0 - alpha, k_d, df_error);

    std::vector<posthoc_comparison> comparisons;

    // Compare all pairs
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            // Tukey-Kramer standard error
            double se = std::sqrt(mse * 0.5 * (1.0 / n_i + 1.0 / n_j));

            double q_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // Degenerate case: within-group variance is zero
                if (mean_diff == 0.0) {
                    q_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    q_stat = std::numeric_limits<double>::infinity();
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                // q statistic (studentized range statistic)
                q_stat = std::abs(mean_diff) / se;

                // p-value from studentized range distribution
                p_value = 1.0 - studentized_range_cdf(q_stat, k_d, df_error);
                p_value = std::max(0.0, std::min(1.0, p_value));

                double margin = q_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (q_stat > q_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, q_stat, p_value, lower, upper, significant});
        }
    }

    return {"Tukey HSD", comparisons, alpha, mse, df_error};
}

/**
 * @brief Perform Bonferroni method for multiple comparisons
 *
 * Performs multiple comparisons among all pairs of groups with Bonferroni correction
 * as a post-hoc test following one-way ANOVA.
 *
 * @param anova_result Result from one-way ANOVA
 * @param alpha Significance level (default: 0.05)
 * @return posthoc_result Post-hoc comparison results
 * @throws std::invalid_argument If alpha is outside the range (0, 1)
 */
inline posthoc_result bonferroni_posthoc(const one_way_anova_result& anova_result,
                                          double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::bonferroni_posthoc: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;

    double n_comparisons = static_cast<double>(k * (k - 1) / 2);
    double alpha_adj = alpha / n_comparisons;
    double t_crit = t_quantile(1.0 - alpha_adj / 2.0, df_error);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_j));

            double t_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // Degenerate case: within-group variance is zero
                if (mean_diff == 0.0) {
                    t_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                t_stat = mean_diff / se;

                p_value = 2.0 * (1.0 - t_cdf(std::abs(t_stat), df_error));
                p_value = std::min(1.0, p_value * n_comparisons);

                double margin = t_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (std::abs(t_stat) > t_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, t_stat, p_value, lower, upper, significant});
        }
    }

    return {"Bonferroni", comparisons, alpha, mse, df_error};
}

/**
 * @brief Perform Dunnett's test for multiple comparisons against a control group
 *
 * Performs comparisons between a specified control group and all other groups
 * as a post-hoc test following one-way ANOVA. Uses Bonferroni approximation.
 *
 * @param anova_result Result from one-way ANOVA
 * @param control_group Index of the control group (default: 0)
 * @param alpha Significance level (default: 0.05)
 * @return posthoc_result Post-hoc comparison results
 * @throws std::invalid_argument If alpha is outside the range (0, 1)
 * @throws std::invalid_argument If control_group is an invalid index
 */
inline posthoc_result dunnett_posthoc(const one_way_anova_result& anova_result,
                                       std::size_t control_group = 0,
                                       double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::dunnett_posthoc: alpha must be in (0, 1)");
    }
    if (control_group >= anova_result.n_groups) {
        throw std::invalid_argument("statcpp::dunnett_posthoc: invalid control group index");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;

    double n_comparisons = static_cast<double>(k - 1);
    double alpha_adj = alpha / n_comparisons;  // Bonferroni approximation
    double t_crit = t_quantile(1.0 - alpha_adj / 2.0, df_error);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        if (i == control_group) continue;

        double mean_diff = anova_result.group_means[i] - anova_result.group_means[control_group];
        double n_i = static_cast<double>(anova_result.group_sizes[i]);
        double n_c = static_cast<double>(anova_result.group_sizes[control_group]);

        double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_c));

        double t_stat, p_value, lower, upper;
        bool significant;

        if (se == 0.0) {
            // Degenerate case: within-group variance is zero
            if (mean_diff == 0.0) {
                t_stat = 0.0; p_value = 1.0;
                lower = 0.0; upper = 0.0; significant = false;
            } else {
                t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                p_value = 0.0;
                lower = mean_diff; upper = mean_diff; significant = true;
            }
        } else {
            t_stat = mean_diff / se;

            p_value = 2.0 * (1.0 - t_cdf(std::abs(t_stat), df_error));
            p_value = std::min(1.0, p_value * n_comparisons);

            double margin = t_crit * se;
            lower = mean_diff - margin;
            upper = mean_diff + margin;
            significant = (std::abs(t_stat) > t_crit);
        }

        comparisons.push_back({i, control_group, mean_diff, se, t_stat, p_value, lower, upper, significant});
    }

    return {"Dunnett (Bonferroni approximation)", comparisons, alpha, mse, df_error};
}

/**
 * @brief Perform Scheffe's method for multiple comparisons
 *
 * Performs multiple comparisons among all pairs of groups using Scheffe's method
 * as a post-hoc test following one-way ANOVA. This is the most conservative method.
 *
 * Scheffe's method is valid for any linear contrast, applicable not only to
 * pairwise comparisons but also to complex contrasts.
 * The F statistic is calculated as t^2 / (k-1) and compared to F(k-1, df_error) distribution.
 * The p-value is computed as 1 - F_cdf(F_s, k-1, df_error).
 *
 * @param anova_result Result from one-way ANOVA
 * @param alpha Significance level (default: 0.05)
 * @return posthoc_result Post-hoc comparison results
 * @throws std::invalid_argument If alpha is outside the range (0, 1)
 */
inline posthoc_result scheffe_posthoc(const one_way_anova_result& anova_result,
                                       double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::scheffe_posthoc: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_between = anova_result.between.df;
    double df_error = anova_result.within.df;

    // Scheffe critical value
    double f_crit = f_quantile(1.0 - alpha, df_between, df_error);
    double scheffe_crit = std::sqrt(df_between * f_crit);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_j));

            double t_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // Degenerate case: within-group variance is zero
                if (mean_diff == 0.0) {
                    t_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                t_stat = mean_diff / se;

                // Scheffe's F statistic
                double f_stat = (t_stat * t_stat) / df_between;
                p_value = 1.0 - f_cdf(f_stat, df_between, df_error);

                double margin = scheffe_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (std::abs(t_stat) > scheffe_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, t_stat, p_value, lower, upper, significant});
        }
    }

    return {"Scheffe", comparisons, alpha, mse, df_error};
}

// ============================================================================
// ANCOVA (Analysis of Covariance)
// ============================================================================

/**
 * @brief Structure storing ANCOVA (Analysis of Covariance) results
 *
 * Holds all results from one-way ANCOVA (covariate effect, treatment effect, error, etc.).
 */
struct ancova_result {
    double ss_covariate;        ///< Sum of squares for covariate
    double ss_treatment;        ///< Sum of squares for treatment effect
    double ss_error;            ///< Error sum of squares
    double df_covariate;        ///< Degrees of freedom for covariate
    double df_treatment;        ///< Degrees of freedom for treatment
    double df_error;            ///< Degrees of freedom for error
    double ms_covariate;        ///< Mean square for covariate
    double ms_treatment;        ///< Mean square for treatment
    double ms_error;            ///< Mean square for error
    double f_covariate;         ///< F statistic for covariate
    double f_treatment;         ///< F statistic for treatment
    double p_covariate;         ///< p-value for covariate
    double p_treatment;         ///< p-value for treatment
    std::vector<double> adjusted_means;  ///< Adjusted group means
};

/**
 * @brief Perform one-way analysis of covariance
 *
 * Tests differences in group means while controlling for the effect of a covariate.
 * Each observation is specified as a pair of dependent variable (y) and covariate (x).
 *
 * @param groups Data for each group. groups[i] is a vector of (y, x) pairs for group i
 * @return ancova_result Results of the analysis of covariance
 * @throws std::invalid_argument If number of groups is less than 2
 * @throws std::invalid_argument If an empty group exists
 * @throws std::invalid_argument If number of observations is insufficient
 */
inline ancova_result one_way_ancova(
    const std::vector<std::vector<std::pair<double, double>>>& groups)
{
    std::size_t k = groups.size();
    if (k < 2) {
        throw std::invalid_argument("statcpp::one_way_ancova: need at least 2 groups");
    }

    // Organize data
    std::size_t n_total = 0;
    for (const auto& g : groups) {
        if (g.empty()) {
            throw std::invalid_argument("statcpp::one_way_ancova: empty group detected");
        }
        n_total += g.size();
    }

    if (n_total <= k + 1) {
        throw std::invalid_argument("statcpp::one_way_ancova: insufficient observations");
    }

    // Calculate overall means
    double sum_y = 0.0, sum_x = 0.0;
    for (const auto& g : groups) {
        for (const auto& pair : g) {
            sum_y += pair.first;
            sum_x += pair.second;
        }
    }
    double grand_mean_y = sum_y / static_cast<double>(n_total);
    double grand_mean_x = sum_x / static_cast<double>(n_total);

    // Calculate means for each group
    std::vector<double> group_mean_y(k);
    std::vector<double> group_mean_x(k);
    std::vector<std::size_t> group_sizes(k);

    for (std::size_t i = 0; i < k; ++i) {
        group_sizes[i] = groups[i].size();
        double sy = 0.0, sx = 0.0;
        for (const auto& pair : groups[i]) {
            sy += pair.first;
            sx += pair.second;
        }
        group_mean_y[i] = sy / static_cast<double>(group_sizes[i]);
        group_mean_x[i] = sx / static_cast<double>(group_sizes[i]);
    }

    // Calculate total sums of squares and cross-products
    double sst_y = 0.0;     // Total sum of squares (y)
    double sst_x = 0.0;     // Total sum of squares (x)
    double spt = 0.0;       // Total sum of cross-products
    double ssw_y = 0.0;     // Within-group sum of squares (y)
    double ssw_x = 0.0;     // Within-group sum of squares (x)
    double spw = 0.0;       // Within-group sum of cross-products

    for (std::size_t i = 0; i < k; ++i) {
        for (const auto& pair : groups[i]) {
            double dy_t = pair.first - grand_mean_y;
            double dx_t = pair.second - grand_mean_x;
            sst_y += dy_t * dy_t;
            sst_x += dx_t * dx_t;
            spt += dy_t * dx_t;

            double dy_w = pair.first - group_mean_y[i];
            double dx_w = pair.second - group_mean_x[i];
            ssw_y += dy_w * dy_w;
            ssw_x += dx_w * dx_w;
            spw += dy_w * dx_w;
        }
    }

    // Common regression coefficient (within groups)
    double b_within = (ssw_x > 0.0) ? spw / ssw_x : 0.0;

    // Calculate sum of squares
    double ss_error = ssw_y - b_within * spw;  // Adjusted within-group sum of squares
    double ss_covariate = b_within * spw;       // Variance explained by covariate

    // Overall regression coefficient
    double b_total = (sst_x > 0.0) ? spt / sst_x : 0.0;
    double ss_total_adj = sst_y - b_total * spt;

    // Sum of squares for treatment effect
    double ss_treatment = ss_total_adj - ss_error;

    // Degrees of freedom
    double df_covariate = 1.0;
    double df_treatment = static_cast<double>(k - 1);
    double df_error = static_cast<double>(n_total - k - 1);

    // Mean squares
    double ms_covariate = ss_covariate / df_covariate;
    double ms_treatment = ss_treatment / df_treatment;
    double ms_error = ss_error / df_error;

    // F statistics
    double f_covariate = ms_covariate / ms_error;
    double f_treatment = ms_treatment / ms_error;

    // p-values
    double p_covariate = 1.0 - f_cdf(f_covariate, df_covariate, df_error);
    double p_treatment = 1.0 - f_cdf(f_treatment, df_treatment, df_error);

    // Adjusted means
    std::vector<double> adjusted_means(k);
    for (std::size_t i = 0; i < k; ++i) {
        adjusted_means[i] = group_mean_y[i] - b_within * (group_mean_x[i] - grand_mean_x);
    }

    return {
        ss_covariate, ss_treatment, ss_error,
        df_covariate, df_treatment, df_error,
        ms_covariate, ms_treatment, ms_error,
        f_covariate, f_treatment,
        p_covariate, p_treatment,
        adjusted_means
    };
}

// ============================================================================
// Effect Size for ANOVA
// ============================================================================

/**
 * @brief Calculate Eta-squared for one-way ANOVA
 *
 * Effect size measure indicating the proportion of total variation
 * accounted for by between-group variation.
 *
 * @param result Result from one-way ANOVA
 * @return double Eta-squared value (range 0 to 1)
 */
inline double eta_squared(const one_way_anova_result& result)
{
    if (result.ss_total == 0.0) {
        return 0.0;
    }
    return result.between.ss / result.ss_total;
}

/**
 * @brief Calculate Partial eta-squared for factor A in two-way ANOVA
 *
 * Indicates the proportion of factor A effect relative to the sum of
 * factor A effect and error.
 *
 * @param result Result from two-way ANOVA
 * @return double Partial eta-squared value for factor A
 */
inline double partial_eta_squared_a(const two_way_anova_result& result)
{
    double denom = result.factor_a.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.factor_a.ss / denom;
}

/**
 * @brief Calculate Partial eta-squared for factor B in two-way ANOVA
 *
 * Indicates the proportion of factor B effect relative to the sum of
 * factor B effect and error.
 *
 * @param result Result from two-way ANOVA
 * @return double Partial eta-squared value for factor B
 */
inline double partial_eta_squared_b(const two_way_anova_result& result)
{
    double denom = result.factor_b.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.factor_b.ss / denom;
}

/**
 * @brief Calculate Partial eta-squared for interaction in two-way ANOVA
 *
 * Indicates the proportion of interaction effect relative to the sum of
 * interaction effect and error.
 *
 * @param result Result from two-way ANOVA
 * @return double Partial eta-squared value for interaction
 */
inline double partial_eta_squared_interaction(const two_way_anova_result& result)
{
    double denom = result.interaction.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.interaction.ss / denom;
}

/**
 * @brief Calculate Omega-squared for one-way ANOVA
 *
 * A less biased effect size estimate compared to Eta-squared.
 * Suitable for estimating population effect size.
 *
 * @param result Result from one-way ANOVA
 * @return double Omega-squared value
 */
inline double omega_squared(const one_way_anova_result& result)
{
    double ss_between = result.between.ss;
    double df_between = result.between.df;
    double ms_within = result.within.ms;
    double ss_total = result.ss_total;

    double denom = ss_total + ms_within;
    if (denom == 0.0) {
        return 0.0;
    }
    return (ss_between - df_between * ms_within) / denom;
}

/**
 * @brief Calculate Cohen's f for one-way ANOVA
 *
 * Standardized effect size measure where 0.10 indicates small,
 * 0.25 indicates medium, and 0.40 indicates large effect.
 *
 * @param result Result from one-way ANOVA
 * @return double Cohen's f value (infinity if eta_squared equals 1.0)
 */
inline double cohens_f(const one_way_anova_result& result)
{
    double eta_sq = eta_squared(result);
    if (eta_sq >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(eta_sq / (1.0 - eta_sq));
}

} // namespace statcpp
