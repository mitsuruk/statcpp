/**
 * @file categorical.hpp
 * @brief Categorical data analysis
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace statcpp {

// ============================================================================
// Contingency Table (Cross-tabulation)
// ============================================================================

/**
 * @brief Contingency table (cross-tabulation) result
 */
struct contingency_table_result {
    std::vector<std::vector<std::size_t>> table;  ///< Observed frequencies
    std::vector<std::size_t> row_totals;          ///< Row totals
    std::vector<std::size_t> col_totals;          ///< Column totals
    std::size_t total;                            ///< Grand total
    std::size_t n_rows;                           ///< Number of rows
    std::size_t n_cols;                           ///< Number of columns
};

/**
 * @brief Create a contingency table
 *
 * Creates a contingency table (cross-tabulation) from two categorical variables.
 *
 * @param row_data Row category values (integers starting from 0)
 * @param col_data Column category values (integers starting from 0)
 * @return Contingency table result
 * @throws std::invalid_argument If data lengths do not match or if data is empty
 */
inline contingency_table_result contingency_table(
    const std::vector<std::size_t>& row_data,
    const std::vector<std::size_t>& col_data)
{
    if (row_data.size() != col_data.size()) {
        throw std::invalid_argument("statcpp::contingency_table: data lengths must match");
    }
    if (row_data.empty()) {
        throw std::invalid_argument("statcpp::contingency_table: empty data");
    }

    // Determine number of categories
    std::size_t max_row = *std::max_element(row_data.begin(), row_data.end());
    std::size_t max_col = *std::max_element(col_data.begin(), col_data.end());
    std::size_t n_rows = max_row + 1;
    std::size_t n_cols = max_col + 1;

    // Create contingency table
    std::vector<std::vector<std::size_t>> table(n_rows, std::vector<std::size_t>(n_cols, 0));

    for (std::size_t i = 0; i < row_data.size(); ++i) {
        table[row_data[i]][col_data[i]]++;
    }

    // Calculate marginal frequencies
    std::vector<std::size_t> row_totals(n_rows, 0);
    std::vector<std::size_t> col_totals(n_cols, 0);
    std::size_t total = 0;

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            row_totals[i] += table[i][j];
            col_totals[j] += table[i][j];
            total += table[i][j];
        }
    }

    return {table, row_totals, col_totals, total, n_rows, n_cols};
}

// ============================================================================
// Odds Ratio
// ============================================================================

/**
 * @brief Odds ratio result
 */
struct odds_ratio_result {
    double odds_ratio;            ///< Odds ratio
    double log_odds_ratio;        ///< Log odds ratio
    double se_log_odds_ratio;     ///< Standard error of log odds ratio
    double ci_lower;              ///< 95% confidence interval lower bound
    double ci_upper;              ///< 95% confidence interval upper bound
};

/**
 * @brief Calculate odds ratio from a 2x2 contingency table
 *
 * Calculates the odds ratio and its confidence interval.
 * Odds ratio = (a * d) / (b * c)
 *
 * @param table 2x2 contingency table in the form [[a, b], [c, d]]
 *              - a: Exposed and diseased
 *              - b: Exposed and not diseased
 *              - c: Not exposed and diseased
 *              - d: Not exposed and not diseased
 * @return Odds ratio calculation result
 * @throws std::invalid_argument If table is not 2x2 or if cell count is zero
 *
 * @note Currently throws an exception when any cell count is zero. A future enhancement
 *       may add an optional Gart-Zweifel continuity correction (+0.5 to all cells)
 *       to handle zero-cell tables.
 */
inline odds_ratio_result odds_ratio(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::odds_ratio: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    if (a == 0 || b == 0 || c == 0 || d == 0) {
        throw std::invalid_argument("statcpp::odds_ratio: zero cell count not allowed");
    }

    double or_val = (a * d) / (b * c);
    double log_or = std::log(or_val);
    double se_log_or = std::sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d);

    // 95% confidence interval
    double z = 1.96;
    double ci_lower = std::exp(log_or - z * se_log_or);
    double ci_upper = std::exp(log_or + z * se_log_or);

    return {or_val, log_or, se_log_or, ci_lower, ci_upper};
}

/**
 * @brief Calculate odds ratio from a 2x2 contingency table (specifying cell values directly)
 *
 * @param a Exposed and diseased
 * @param b Exposed and not diseased
 * @param c Not exposed and diseased
 * @param d Not exposed and not diseased
 * @return Odds ratio calculation result
 * @throws std::invalid_argument If cell count is zero
 */
inline odds_ratio_result odds_ratio(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return odds_ratio({{a, b}, {c, d}});
}

// ============================================================================
// Relative Risk (Risk Ratio)
// ============================================================================

/**
 * @brief Relative risk result
 */
struct relative_risk_result {
    double relative_risk;         ///< Relative risk
    double log_relative_risk;     ///< Log relative risk
    double se_log_relative_risk;  ///< Standard error of log relative risk
    double ci_lower;              ///< 95% confidence interval lower bound
    double ci_upper;              ///< 95% confidence interval upper bound
};

/**
 * @brief Calculate relative risk (risk ratio) from a 2x2 contingency table
 *
 * Calculates the relative risk and its confidence interval.
 * Relative risk = (a/(a+b)) / (c/(c+d))
 *
 * Standard error is calculated on the log scale (Greenland-Robins method):
 * SE(log RR) = sqrt((1-p1)/(n1*p1) + (1-p0)/(n0*p0))
 *            = sqrt((1-p1)/a + (1-p0)/c)
 * where p1 = a/(a+b), p0 = c/(c+d)
 *
 * @param table 2x2 contingency table in the form [[a, b], [c, d]]
 *              - a: Exposed and diseased
 *              - b: Exposed and not diseased
 *              - c: Not exposed and diseased
 *              - d: Not exposed and not diseased
 * @return Relative risk calculation result
 * @throws std::invalid_argument If table is not 2x2, if row total is zero, or if risk is zero
 */
inline relative_risk_result relative_risk(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::relative_risk: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    double n1 = a + b;  // Size of exposed group
    double n0 = c + d;  // Size of unexposed group

    if (n1 == 0 || n0 == 0) {
        throw std::invalid_argument("statcpp::relative_risk: zero row total");
    }
    if (a == 0 || c == 0) {
        throw std::invalid_argument("statcpp::relative_risk: zero risk in a group");
    }

    double risk1 = a / n1;  // Risk in exposed group
    double risk0 = c / n0;  // Risk in unexposed group

    double rr = risk1 / risk0;
    double log_rr = std::log(rr);

    // Standard error (Greenland-Robins formula)
    double se_log_rr = std::sqrt((1.0 - risk1)/(a) + (1.0 - risk0)/(c));

    // 95% confidence interval
    double z = 1.96;
    double ci_lower = std::exp(log_rr - z * se_log_rr);
    double ci_upper = std::exp(log_rr + z * se_log_rr);

    return {rr, log_rr, se_log_rr, ci_lower, ci_upper};
}

/**
 * @brief Calculate relative risk from a 2x2 contingency table (specifying cell values directly)
 *
 * @param a Exposed and diseased
 * @param b Exposed and not diseased
 * @param c Not exposed and diseased
 * @param d Not exposed and not diseased
 * @return Relative risk calculation result
 * @throws std::invalid_argument If row total is zero or if risk is zero
 */
inline relative_risk_result relative_risk(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return relative_risk({{a, b}, {c, d}});
}

// ============================================================================
// Risk Difference (Attributable Risk)
// ============================================================================

/**
 * @brief Risk difference result
 */
struct risk_difference_result {
    double risk_difference;       ///< Risk difference (attributable risk)
    double se;                    ///< Standard error
    double ci_lower;              ///< 95% confidence interval lower bound
    double ci_upper;              ///< 95% confidence interval upper bound
};

/**
 * @brief Calculate risk difference from a 2x2 contingency table
 *
 * Calculates the risk difference (attributable risk) and its confidence interval.
 * Risk difference = (a/(a+b)) - (c/(c+d))
 *
 * @param table 2x2 contingency table in the form [[a, b], [c, d]]
 * @return Risk difference calculation result
 * @throws std::invalid_argument If table is not 2x2 or if row total is zero
 */
inline risk_difference_result risk_difference(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::risk_difference: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    double n1 = a + b;
    double n0 = c + d;

    if (n1 == 0 || n0 == 0) {
        throw std::invalid_argument("statcpp::risk_difference: zero row total");
    }

    double risk1 = a / n1;
    double risk0 = c / n0;

    double rd = risk1 - risk0;

    // Standard error
    double se = std::sqrt(risk1 * (1.0 - risk1) / n1 + risk0 * (1.0 - risk0) / n0);

    // 95% confidence interval
    double z = 1.96;
    double ci_lower = rd - z * se;
    double ci_upper = rd + z * se;

    return {rd, se, ci_lower, ci_upper};
}

/**
 * @brief Calculate risk difference from a 2x2 contingency table (specifying cell values directly)
 *
 * @param a Exposed and diseased
 * @param b Exposed and not diseased
 * @param c Not exposed and diseased
 * @param d Not exposed and not diseased
 * @return Risk difference calculation result
 * @throws std::invalid_argument If row total is zero
 */
inline risk_difference_result risk_difference(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return risk_difference({{a, b}, {c, d}});
}

// ============================================================================
// Number Needed to Treat (NNT)
// ============================================================================

/**
 * @brief Calculate Number Needed to Treat (NNT)
 *
 * Calculated as the reciprocal of the risk difference.
 * Represents the number of patients that need to be treated to prevent one outcome.
 *
 * @param table 2x2 contingency table
 * @return Number needed to treat
 * @throws std::invalid_argument If risk difference is zero
 */
inline double number_needed_to_treat(const std::vector<std::vector<std::size_t>>& table)
{
    auto rd = risk_difference(table);
    if (rd.risk_difference == 0.0) {
        throw std::invalid_argument("statcpp::number_needed_to_treat: risk difference is zero");
    }
    return 1.0 / std::abs(rd.risk_difference);
}

} // namespace statcpp
