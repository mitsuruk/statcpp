/**
 * @file missing_data.hpp
 * @brief Missing data handling functions
 *
 * Provides missing data handling capabilities including missing value detection,
 * pattern analysis, multiple imputation, and sensitivity analysis.
 *
 * Main features:
 * - Missing pattern analysis (MCAR/MAR/MNAR classification)
 * - Little's MCAR test
 * - Multiple imputation (PMM, Bootstrap EM)
 * - Sensitivity analysis (pattern mixture model, selection model)
 * - Complete case analysis
 */

#ifndef STATCPP_MISSING_DATA_HPP
#define STATCPP_MISSING_DATA_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/data_wrangling.hpp"
#include "statcpp/dispersion_spread.hpp"

namespace statcpp {

// ============================================================================
// Missing Data Pattern Classification (MCAR/MAR/MNAR)
// ============================================================================

/**
 * @brief Missing mechanism types
 *
 * Enumeration for classifying the mechanism of missing data.
 * - MCAR: Missing occurs completely at random
 * - MAR: Missing depends on observed variables
 * - MNAR: Missing depends on the missing value itself
 */
enum class missing_mechanism {
    mcar,       ///< Missing Completely At Random
    mar,        ///< Missing At Random
    mnar,       ///< Missing Not At Random
    unknown     ///< Cannot be determined
};

/**
 * @brief Little's MCAR test result
 *
 * Structure storing results of Little's MCAR test.
 * Contains chi-square statistic, p-value, degrees of freedom, and interpretation.
 */
struct mcar_test_result {
    double chi_square = 0.0;    ///< Chi-square statistic
    double p_value = 1.0;       ///< p-value
    std::size_t df = 0;         ///< Degrees of freedom
    bool is_mcar = true;        ///< Whether MCAR is concluded (p > 0.05)
    std::string interpretation; ///< Interpretation
};

/**
 * @brief Missing pattern information
 *
 * Structure storing information about missing patterns in a dataset.
 * Contains occurrence frequency of each pattern, missing rate per variable, and overall missing rate.
 */
struct missing_pattern_info {
    std::vector<std::vector<uint8_t>> patterns;    ///< Missing patterns (1 = missing, 0 = observed)
    std::vector<std::size_t> pattern_counts;       ///< Count of each pattern
    std::vector<double> missing_rates;             ///< Missing rate per variable
    double overall_missing_rate = 0.0;             ///< Overall missing rate
    std::size_t n_complete_cases = 0;              ///< Number of complete cases
    std::size_t n_patterns = 0;                    ///< Number of missing patterns
};

/**
 * @brief Analyze missing patterns
 *
 * Analyzes missing patterns in a dataset, calculating missing rate per variable,
 * overall missing rate, and types and frequencies of missing patterns.
 *
 * @param data 2D data for analysis (rows: observations, columns: variables)
 * @return missing_pattern_info Missing pattern information
 * @throws std::invalid_argument If data is empty or row sizes are inconsistent
 */
inline missing_pattern_info analyze_missing_patterns(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::analyze_missing_patterns: empty data");
    }

    missing_pattern_info result;
    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    // Calculate missing rate per variable
    result.missing_rates.resize(n_cols, 0.0);
    std::size_t total_missing = 0;

    for (const auto& row : data) {
        if (row.size() != n_cols) {
            throw std::invalid_argument(
                "statcpp::analyze_missing_patterns: inconsistent row sizes");
        }
        for (std::size_t j = 0; j < n_cols; ++j) {
            if (is_na(row[j])) {
                ++result.missing_rates[j];
                ++total_missing;
            }
        }
    }

    for (auto& rate : result.missing_rates) {
        rate /= static_cast<double>(n_rows);
    }
    result.overall_missing_rate = static_cast<double>(total_missing) /
                                  static_cast<double>(n_rows * n_cols);

    // Extract missing patterns (implemented with linear search instead of std::map)
    std::vector<std::vector<uint8_t>> unique_patterns;
    std::vector<std::size_t> pattern_counts;
    result.n_complete_cases = 0;

    for (const auto& row : data) {
        std::vector<uint8_t> pattern(n_cols);
        bool has_missing = false;
        for (std::size_t j = 0; j < n_cols; ++j) {
            pattern[j] = is_na(row[j]) ? 1 : 0;
            if (pattern[j]) {
                has_missing = true;
            }
        }
        if (!has_missing) {
            ++result.n_complete_cases;
        }

        // Search for existing pattern
        bool found = false;
        for (std::size_t i = 0; i < unique_patterns.size(); ++i) {
            if (unique_patterns[i] == pattern) {
                ++pattern_counts[i];
                found = true;
                break;
            }
        }

        // Add if new pattern
        if (!found) {
            unique_patterns.push_back(pattern);
            pattern_counts.push_back(1);
        }
    }

    // Store results
    result.patterns = unique_patterns;
    result.pattern_counts = pattern_counts;
    result.n_patterns = result.patterns.size();

    return result;
}

/**
 * @brief Create missing indicator variables
 *
 * Creates an indicator variable matrix where each element is 1.0 if missing
 * and 0.0 if observed.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @return std::vector<std::vector<double>> Missing indicator matrix (1 = missing, 0 = observed)
 */
inline std::vector<std::vector<double>> create_missing_indicator(
    const std::vector<std::vector<double>>& data)
{
    std::vector<std::vector<double>> indicator;
    indicator.reserve(data.size());

    for (const auto& row : data) {
        std::vector<double> ind_row;
        ind_row.reserve(row.size());
        for (double val : row) {
            ind_row.push_back(is_na(val) ? 1.0 : 0.0);
        }
        indicator.push_back(std::move(ind_row));
    }
    return indicator;
}

/**
 * @brief Little's MCAR test (simplified version)
 *
 * Tests whether the mean difference between complete and incomplete data
 * is significant, determining if data is MCAR (Missing Completely At Random).
 *
 * @note This is a simplified version; the complete Little's test requires
 *       covariance estimation via EM algorithm.
 *
 * @param data 2D data for testing (rows: observations, columns: variables)
 * @return mcar_test_result Test result (chi-square statistic, p-value, degrees of freedom, conclusion)
 * @throws std::invalid_argument If data is empty
 */
inline mcar_test_result test_mcar_simple(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::test_mcar_simple: empty data");
    }

    std::size_t n_cols = data[0].size();
    mcar_test_result result;

    // Check correlation between missingness in each variable and values of other variables
    // Under MCAR, missingness of one variable is unrelated to values of other variables
    double total_chi_sq = 0.0;
    std::size_t total_df = 0;

    for (std::size_t j = 0; j < n_cols; ++j) {
        // Missing indicator for variable j
        std::vector<double> missing_j;
        missing_j.reserve(data.size());
        for (const auto& row : data) {
            missing_j.push_back(is_na(row[j]) ? 1.0 : 0.0);
        }

        // Check correlation with other variables
        for (std::size_t k = 0; k < n_cols; ++k) {
            if (j == k) continue;

            // Observed values of variable k (only cases where both are observed)
            std::vector<double> obs_k;
            std::vector<double> miss_j_subset;

            for (std::size_t i = 0; i < data.size(); ++i) {
                if (!is_na(data[i][k])) {
                    obs_k.push_back(data[i][k]);
                    miss_j_subset.push_back(missing_j[i]);
                }
            }

            if (obs_k.size() < 5) continue;  // Sample size too small

            // Compare means between missing and observed groups
            std::vector<double> obs_when_j_missing;
            std::vector<double> obs_when_j_observed;

            for (std::size_t i = 0; i < obs_k.size(); ++i) {
                if (miss_j_subset[i] > 0.5) {
                    obs_when_j_missing.push_back(obs_k[i]);
                } else {
                    obs_when_j_observed.push_back(obs_k[i]);
                }
            }

            if (obs_when_j_missing.size() < 2 || obs_when_j_observed.size() < 2) {
                continue;
            }

            // Calculate two-sample t-test statistic
            double mean1 = mean(obs_when_j_missing.begin(), obs_when_j_missing.end());
            double mean2 = mean(obs_when_j_observed.begin(), obs_when_j_observed.end());
            double var1 = var(obs_when_j_missing.begin(), obs_when_j_missing.end(), 1);
            double var2 = var(obs_when_j_observed.begin(), obs_when_j_observed.end(), 1);

            double n1 = static_cast<double>(obs_when_j_missing.size());
            double n2 = static_cast<double>(obs_when_j_observed.size());

            double se = std::sqrt(var1 / n1 + var2 / n2);
            if (se > 1e-10) {
                double t_stat = (mean1 - mean2) / se;
                total_chi_sq += t_stat * t_stat;
                ++total_df;
            }
        }
    }

    result.chi_square = total_chi_sq;
    result.df = total_df;

    // Approximate p-value from chi-square distribution (Wilson-Hilferty approximation)
    if (total_df > 0) {
        double z = std::pow(total_chi_sq / static_cast<double>(total_df), 1.0 / 3.0) -
                   (1.0 - 2.0 / (9.0 * static_cast<double>(total_df)));
        z /= std::sqrt(2.0 / (9.0 * static_cast<double>(total_df)));
        // Upper tail probability of standard normal distribution (approximation)
        result.p_value = 0.5 * std::erfc(z / std::sqrt(2.0));
        result.p_value = std::max(0.0, std::min(1.0, result.p_value));
    } else {
        result.p_value = 1.0;
    }

    result.is_mcar = (result.p_value > 0.05);
    result.interpretation = result.is_mcar
        ? "MCAR assumption is not rejected (p > 0.05). "
          "Missing data may be completely random."
        : "MCAR assumption is rejected (p <= 0.05). "
          "Missing data is likely MAR or MNAR.";

    return result;
}

/**
 * @brief Simple diagnosis of missing mechanism
 *
 * Diagnoses the missing mechanism (MCAR, MAR, MNAR) of the data.
 * Internally performs Little's MCAR test and estimates the missing mechanism
 * based on the result.
 *
 * @note Distinguishing between MAR and MNAR is difficult with observed data alone,
 *       so if MCAR is rejected, it is conservatively classified as MAR.
 *
 * @param data 2D data for diagnosis (rows: observations, columns: variables)
 * @return missing_mechanism Estimated missing mechanism
 */
inline missing_mechanism diagnose_missing_mechanism(
    const std::vector<std::vector<double>>& data)
{
    auto mcar_result = test_mcar_simple(data);

    if (mcar_result.is_mcar) {
        return missing_mechanism::mcar;
    }

    // Distinguishing MAR vs MNAR is difficult with observed data alone
    // Here we assume MAR (conservative choice)
    return missing_mechanism::mar;
}

// ============================================================================
// Multiple Imputation
// ============================================================================

/**
 * @brief Multiple imputation result
 *
 * Structure storing results of multiple imputation.
 * Contains multiple imputed datasets, pooled statistics,
 * within-imputation variance, between-imputation variance, and fraction of missing information.
 */
struct multiple_imputation_result {
    std::vector<std::vector<std::vector<double>>> imputed_datasets;  ///< Imputed datasets
    std::size_t m = 0;                                               ///< Number of imputations
    std::vector<double> pooled_means;                                ///< Pooled means
    std::vector<double> pooled_vars;                                 ///< Pooled variances
    std::vector<double> within_vars;                                 ///< Within-imputation variances
    std::vector<double> between_vars;                                ///< Between-imputation variances
    std::vector<double> fraction_missing_info;                       ///< Fraction of missing information (FMI)
};

/**
 * @brief Single imputation by conditional mean
 *
 * Imputes missing values with conditional mean using simple linear regression with predictor variables.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @param target_col Column index for imputation target
 * @param predictor_cols Column indices for predictor variables
 * @return std::vector<double> Values of target_col after imputation
 */
inline std::vector<double> impute_conditional_mean(
    const std::vector<std::vector<double>>& data,
    std::size_t target_col,
    const std::vector<std::size_t>& predictor_cols)
{
    std::size_t n = data.size();
    std::vector<double> result;
    result.reserve(n);

    // Extract complete cases
    std::vector<std::vector<double>> complete_cases;
    for (const auto& row : data) {
        bool complete = !is_na(row[target_col]);
        for (std::size_t col : predictor_cols) {
            if (is_na(row[col])) {
                complete = false;
                break;
            }
        }
        if (complete) {
            complete_cases.push_back(row);
        }
    }

    if (complete_cases.empty()) {
        // If no complete cases, use simple mean imputation
        std::vector<double> non_na;
        for (const auto& row : data) {
            if (!is_na(row[target_col])) {
                non_na.push_back(row[target_col]);
            }
        }
        double fill_val = non_na.empty() ? 0.0 : mean(non_na.begin(), non_na.end());
        for (const auto& row : data) {
            result.push_back(is_na(row[target_col]) ? fill_val : row[target_col]);
        }
        return result;
    }

    // Estimate imputed values using simple linear regression
    // Y = target_col, X = predictor_cols (simplified version using only first predictor)
    if (predictor_cols.empty()) {
        double m = 0.0;
        for (const auto& row : complete_cases) {
            m += row[target_col];
        }
        m /= static_cast<double>(complete_cases.size());
        for (const auto& row : data) {
            result.push_back(is_na(row[target_col]) ? m : row[target_col]);
        }
        return result;
    }

    // Simple regression using first predictor
    std::size_t pred_col = predictor_cols[0];
    std::vector<double> x_vals, y_vals;
    for (const auto& row : complete_cases) {
        x_vals.push_back(row[pred_col]);
        y_vals.push_back(row[target_col]);
    }

    double x_mean = mean(x_vals.begin(), x_vals.end());
    double y_mean = mean(y_vals.begin(), y_vals.end());

    double cov_xy = 0.0;
    double var_x = 0.0;
    for (std::size_t i = 0; i < x_vals.size(); ++i) {
        double dx = x_vals[i] - x_mean;
        double dy = y_vals[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }

    double beta = (var_x > 1e-10) ? cov_xy / var_x : 0.0;
    double alpha = y_mean - beta * x_mean;

    // Imputation
    for (const auto& row : data) {
        if (is_na(row[target_col])) {
            if (!is_na(row[pred_col])) {
                result.push_back(alpha + beta * row[pred_col]);
            } else {
                result.push_back(y_mean);
            }
        } else {
            result.push_back(row[target_col]);
        }
    }

    return result;
}

/**
 * @brief Multiple imputation (PMM: Predictive Mean Matching)
 *
 * Performs multiple imputation using Predictive Mean Matching method.
 * Randomly selects a donor from k observations closest to the predicted value.
 * Results are pooled based on Rubin's rules.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @param m Number of imputations (default: 5)
 * @param seed Random seed (0 for random seed)
 * @return multiple_imputation_result Multiple imputation result
 * @throws std::invalid_argument If data is empty
 */
inline multiple_imputation_result multiple_imputation_pmm(
    const std::vector<std::vector<double>>& data,
    std::size_t m = 5,
    unsigned int seed = 0)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::multiple_imputation_pmm: empty data");
    }

    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    multiple_imputation_result result;
    result.m = m;
    result.imputed_datasets.resize(m);

    // Random number generator
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

    for (std::size_t imp = 0; imp < m; ++imp) {
        // Copy data
        auto imputed = data;

        // Impute each column
        for (std::size_t j = 0; j < n_cols; ++j) {
            // Collect missing indices
            std::vector<std::size_t> missing_indices;
            std::vector<std::size_t> observed_indices;
            std::vector<double> observed_values;

            for (std::size_t i = 0; i < n_rows; ++i) {
                if (is_na(imputed[i][j])) {
                    missing_indices.push_back(i);
                } else {
                    observed_indices.push_back(i);
                    observed_values.push_back(imputed[i][j]);
                }
            }

            if (missing_indices.empty() || observed_values.empty()) {
                continue;
            }

            // Use other columns as predictor variables
            std::vector<std::size_t> predictor_cols;
            for (std::size_t k = 0; k < n_cols; ++k) {
                if (k != j) {
                    predictor_cols.push_back(k);
                }
            }

            // Calculate conditional means
            auto cond_means = impute_conditional_mean(imputed, j, predictor_cols);

            // PMM: Randomly select from k observations closest to predicted value
            constexpr std::size_t k_donors = 5;

            for (std::size_t idx : missing_indices) {
                double pred_val = cond_means[idx];

                // Calculate distance to observed values
                std::vector<std::pair<double, std::size_t>> distances;
                distances.reserve(observed_indices.size());
                for (std::size_t oi = 0; oi < observed_indices.size(); ++oi) {
                    std::size_t obs_idx = observed_indices[oi];
                    double dist = std::abs(observed_values[oi] - pred_val);
                    distances.emplace_back(dist, obs_idx);
                }

                // Sort by distance
                std::partial_sort(distances.begin(),
                                  distances.begin() + std::min(k_donors, distances.size()),
                                  distances.end());

                // Randomly select from k donors
                std::size_t n_donors = std::min(k_donors, distances.size());
                std::uniform_int_distribution<std::size_t> dist(0, n_donors - 1);
                std::size_t donor_idx = distances[dist(rng)].second;

                imputed[idx][j] = imputed[donor_idx][j];
            }
        }

        result.imputed_datasets[imp] = std::move(imputed);
    }

    // Pooling using Rubin's rules
    result.pooled_means.resize(n_cols, 0.0);
    result.pooled_vars.resize(n_cols, 0.0);
    result.within_vars.resize(n_cols, 0.0);
    result.between_vars.resize(n_cols, 0.0);
    result.fraction_missing_info.resize(n_cols, 0.0);

    for (std::size_t j = 0; j < n_cols; ++j) {
        std::vector<double> means_j(m);
        std::vector<double> vars_j(m);

        for (std::size_t imp = 0; imp < m; ++imp) {
            std::vector<double> col_values;
            col_values.reserve(n_rows);
            for (std::size_t i = 0; i < n_rows; ++i) {
                col_values.push_back(result.imputed_datasets[imp][i][j]);
            }
            means_j[imp] = mean(col_values.begin(), col_values.end());
            vars_j[imp] = var(col_values.begin(), col_values.end(), 1);
        }

        // Pooled mean
        result.pooled_means[j] = mean(means_j.begin(), means_j.end());

        // Within-imputation variance (W)
        result.within_vars[j] = mean(vars_j.begin(), vars_j.end());

        // Between-imputation variance (B)
        double b = 0.0;
        for (double mj : means_j) {
            double diff = mj - result.pooled_means[j];
            b += diff * diff;
        }
        result.between_vars[j] = b / static_cast<double>(m - 1);

        // Total variance
        result.pooled_vars[j] = result.within_vars[j] +
                                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j];

        // Fraction of missing information (FMI)
        // Rubin (1987) definition: FMI = (r + 2/(df+3)) / (r+1)
        // where r = (1 + 1/m) * B / W
        // Simplified version: FMI ≈ (1 + 1/m) * B / T
        if (result.pooled_vars[j] > 1e-10) {
            result.fraction_missing_info[j] =
                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j] /
                result.pooled_vars[j];
        }
    }

    return result;
}

/**
 * @brief Multiple imputation (simplified Bootstrap EM method)
 *
 * Performs multiple imputation combining bootstrap sampling with
 * stochastic imputation from normal distribution.
 * Results are pooled based on Rubin's rules.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @param m Number of imputations (default: 5)
 * @param seed Random seed (0 for random seed)
 * @return multiple_imputation_result Multiple imputation result
 * @throws std::invalid_argument If data is empty
 */
inline multiple_imputation_result multiple_imputation_bootstrap(
    const std::vector<std::vector<double>>& data,
    std::size_t m = 5,
    unsigned int seed = 0)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::multiple_imputation_bootstrap: empty data");
    }

    std::size_t n_rows = data.size();
    std::size_t n_cols = data[0].size();

    multiple_imputation_result result;
    result.m = m;
    result.imputed_datasets.resize(m);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

    for (std::size_t imp = 0; imp < m; ++imp) {
        // Create bootstrap sample (from observations)
        std::vector<std::vector<double>> boot_sample;
        boot_sample.reserve(n_rows);

        std::uniform_int_distribution<std::size_t> dist(0, n_rows - 1);
        for (std::size_t i = 0; i < n_rows; ++i) {
            boot_sample.push_back(data[dist(rng)]);
        }

        // Calculate statistics for each column
        std::vector<double> col_means(n_cols);
        std::vector<double> col_stds(n_cols);

        for (std::size_t j = 0; j < n_cols; ++j) {
            std::vector<double> non_na;
            for (const auto& row : boot_sample) {
                if (!is_na(row[j])) {
                    non_na.push_back(row[j]);
                }
            }
            if (!non_na.empty()) {
                col_means[j] = mean(non_na.begin(), non_na.end());
                col_stds[j] = non_na.size() > 1 ?
                    std::sqrt(var(non_na.begin(), non_na.end(), 1)) : 0.0;
            }
        }

        // Imputation (stochastic imputation from normal distribution)
        auto imputed = data;
        std::normal_distribution<double> normal(0.0, 1.0);

        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = 0; j < n_cols; ++j) {
                if (is_na(imputed[i][j])) {
                    // Mean + random noise
                    imputed[i][j] = col_means[j] + col_stds[j] * normal(rng);
                }
            }
        }

        result.imputed_datasets[imp] = std::move(imputed);
    }

    // Pooling using Rubin's rules
    result.pooled_means.resize(n_cols, 0.0);
    result.pooled_vars.resize(n_cols, 0.0);
    result.within_vars.resize(n_cols, 0.0);
    result.between_vars.resize(n_cols, 0.0);
    result.fraction_missing_info.resize(n_cols, 0.0);

    for (std::size_t j = 0; j < n_cols; ++j) {
        std::vector<double> means_j(m);
        std::vector<double> vars_j(m);

        for (std::size_t imp = 0; imp < m; ++imp) {
            std::vector<double> col_values;
            col_values.reserve(n_rows);
            for (std::size_t i = 0; i < n_rows; ++i) {
                col_values.push_back(result.imputed_datasets[imp][i][j]);
            }
            means_j[imp] = mean(col_values.begin(), col_values.end());
            vars_j[imp] = var(col_values.begin(), col_values.end(), 1);
        }

        result.pooled_means[j] = mean(means_j.begin(), means_j.end());
        result.within_vars[j] = mean(vars_j.begin(), vars_j.end());

        double b = 0.0;
        for (double mj : means_j) {
            double diff = mj - result.pooled_means[j];
            b += diff * diff;
        }
        result.between_vars[j] = b / static_cast<double>(m - 1);

        result.pooled_vars[j] = result.within_vars[j] +
                                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j];

        // Fraction of missing information (FMI): Rubin (1987)
        if (result.pooled_vars[j] > 1e-10) {
            result.fraction_missing_info[j] =
                (1.0 + 1.0 / static_cast<double>(m)) * result.between_vars[j] /
                result.pooled_vars[j];
        }
    }

    return result;
}

// ============================================================================
// Sensitivity Analysis for Missing Data
// ============================================================================

/**
 * @brief Sensitivity analysis result (single parameter)
 *
 * Structure storing results of sensitivity analysis for missing data.
 * Contains estimated mean and variance for each value of sensitivity parameter, and interpretation.
 */
struct sensitivity_analysis_result {
    std::vector<double> delta_values;          ///< Sensitivity parameter values
    std::vector<double> estimated_means;       ///< Estimated means
    std::vector<double> estimated_vars;        ///< Estimated variances
    double original_mean = 0.0;                ///< Original estimated mean
    double original_var = 0.0;                 ///< Original estimated variance
    std::string interpretation;                ///< Interpretation
};

/**
 * @brief Sensitivity analysis using pattern mixture model
 *
 * Evaluates robustness of estimates by varying the difference (delta)
 * between missing and observed values under MNAR (Missing Not At Random) assumption.
 *
 * Basic formula of pattern mixture model:
 * - E[Y] = E[Y|R=1] * P(R=1) + E[Y|R=0] * P(R=0)
 * - where R=1 indicates observed, R=0 indicates missing
 * - Sensitivity parameter delta: E[Y|R=0] = E[Y|R=1] + delta
 *
 * Interpretation of delta:
 * - delta = 0: MAR (missing follows same distribution as observed)
 * - delta > 0: Missing values tend to be higher than observed
 * - delta < 0: Missing values tend to be lower than observed
 *
 * @note This implementation uses a simplified assumption for variance (equal to observed variance).
 *       More rigorous analysis requires pattern-specific variance estimation.
 *
 * @param data 1D data for analysis
 * @param delta_values Vector of sensitivity parameter delta values
 * @return sensitivity_analysis_result Sensitivity analysis result
 * @throws std::invalid_argument If data is empty or all values are missing
 */
inline sensitivity_analysis_result sensitivity_analysis_pattern_mixture(
    const std::vector<double>& data,
    const std::vector<double>& delta_values)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_pattern_mixture: empty data");
    }

    sensitivity_analysis_result result;
    result.delta_values = delta_values;
    result.estimated_means.reserve(delta_values.size());
    result.estimated_vars.reserve(delta_values.size());

    // Statistics of observed values
    std::vector<double> observed;
    std::size_t n_missing = 0;
    for (double val : data) {
        if (!is_na(val)) {
            observed.push_back(val);
        } else {
            ++n_missing;
        }
    }

    if (observed.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_pattern_mixture: all values are missing");
    }

    double obs_mean = mean(observed.begin(), observed.end());
    double obs_var = observed.size() > 1 ?
        var(observed.begin(), observed.end(), 1) : 0.0;

    result.original_mean = obs_mean;
    result.original_var = obs_var;

    double n_obs = static_cast<double>(observed.size());
    double n_total = static_cast<double>(data.size());
    double prop_obs = n_obs / n_total;
    double prop_miss = static_cast<double>(n_missing) / n_total;

    // Estimate for each delta value
    for (double delta : delta_values) {
        // Pattern mixture model: E[Y] = E[Y|R=1] * P(R=1) + E[Y|R=0] * P(R=0)
        // E[Y|R=0] = E[Y|R=1] + delta
        double imputed_mean = obs_mean + delta;
        double overall_mean = obs_mean * prop_obs + imputed_mean * prop_miss;

        // Variance estimation (simplified version)
        // True variance requires additional assumptions, but here we use observed variance
        double overall_var = obs_var;

        result.estimated_means.push_back(overall_mean);
        result.estimated_vars.push_back(overall_var);
    }

    result.interpretation =
        "Pattern mixture model sensitivity analysis. "
        "delta represents the hypothesized difference between "
        "missing and observed values. delta=0 corresponds to MAR assumption.";

    return result;
}

/**
 * @brief Sensitivity analysis using selection model
 *
 * Evaluates robustness of estimates by varying the degree (phi) to which
 * missingness depends on the response value.
 * phi = 0 corresponds to MAR assumption, and phi > 0 indicates that
 * missing values tend to be lower than observed values.
 *
 * @param data 1D data for analysis
 * @param phi_values Vector of sensitivity parameter phi values
 * @return sensitivity_analysis_result Sensitivity analysis result
 * @throws std::invalid_argument If data is empty or all values are missing
 */
inline sensitivity_analysis_result sensitivity_analysis_selection_model(
    const std::vector<double>& data,
    const std::vector<double>& phi_values)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_selection_model: empty data");
    }

    sensitivity_analysis_result result;
    result.delta_values = phi_values;  // Store phi as delta
    result.estimated_means.reserve(phi_values.size());
    result.estimated_vars.reserve(phi_values.size());

    // Statistics of observed values
    std::vector<double> observed;
    std::size_t n_missing = 0;
    for (double val : data) {
        if (!is_na(val)) {
            observed.push_back(val);
        } else {
            ++n_missing;
        }
    }

    if (observed.empty()) {
        throw std::invalid_argument(
            "statcpp::sensitivity_analysis_selection_model: all values are missing");
    }

    double obs_mean = mean(observed.begin(), observed.end());
    double obs_var = observed.size() > 1 ?
        var(observed.begin(), observed.end(), 1) : 0.0;
    double obs_std = std::sqrt(obs_var);

    result.original_mean = obs_mean;
    result.original_var = obs_var;

    // Selection model: logit(P(R=1|Y)) = alpha + phi * Y
    // phi > 0: Higher values more likely to be observed (missing values tend to be lower)
    // phi < 0: Lower values more likely to be observed (missing values tend to be higher)
    // phi = 0: MAR

    for (double phi : phi_values) {
        // Simplified correction: adjust expected value of missing based on phi
        // If phi is positive, missing values tend to be lower than observed
        double adjustment = -phi * obs_std * 0.5;  // Scaling factor
        double imputed_mean = obs_mean + adjustment;

        double n_obs = static_cast<double>(observed.size());
        double n_total = static_cast<double>(data.size());
        double prop_obs = n_obs / n_total;
        double prop_miss = static_cast<double>(n_missing) / n_total;

        double overall_mean = obs_mean * prop_obs + imputed_mean * prop_miss;
        double overall_var = obs_var;

        result.estimated_means.push_back(overall_mean);
        result.estimated_vars.push_back(overall_var);
    }

    result.interpretation =
        "Selection model sensitivity analysis. "
        "phi represents the dependence of missingness on the outcome value. "
        "phi=0 corresponds to MAR assumption. "
        "phi>0 implies missing values tend to be lower than observed values.";

    return result;
}

/**
 * @brief Tipping point analysis result
 *
 * Structure storing results of tipping point analysis.
 * Contains information about the critical point (tipping point) where conclusions change.
 */
struct tipping_point_result {
    double tipping_point = 0.0;        ///< Tipping point (critical delta value)
    bool found = false;                ///< Whether tipping point was found
    double threshold = 0.0;            ///< Threshold used
    std::string interpretation;        ///< Interpretation
};

/**
 * @brief Tipping point analysis
 *
 * Searches for the critical point (tipping point) where the estimated mean
 * crosses the specified threshold.
 * This allows assessment of how extreme the MNAR assumption must be to change conclusions.
 *
 * @param data 1D data for analysis
 * @param threshold Threshold (e.g., null hypothesis value)
 * @param delta_min Minimum delta value to search (default: -5.0)
 * @param delta_max Maximum delta value to search (default: 5.0)
 * @param n_points Number of search points (default: 100)
 * @return tipping_point_result Tipping point analysis result
 */
inline tipping_point_result find_tipping_point(
    const std::vector<double>& data,
    double threshold = 0.0,            // e.g., null hypothesis value
    double delta_min = -5.0,
    double delta_max = 5.0,
    std::size_t n_points = 100)
{
    tipping_point_result result;
    result.threshold = threshold;
    result.found = false;
    result.tipping_point = NA;

    std::vector<double> delta_values;
    double step = (delta_max - delta_min) / static_cast<double>(n_points - 1);
    for (std::size_t i = 0; i < n_points; ++i) {
        delta_values.push_back(delta_min + static_cast<double>(i) * step);
    }

    auto sens_result = sensitivity_analysis_pattern_mixture(data, delta_values);

    // Find point where threshold is crossed
    for (std::size_t i = 1; i < sens_result.estimated_means.size(); ++i) {
        double prev = sens_result.estimated_means[i - 1];
        double curr = sens_result.estimated_means[i];

        if ((prev <= threshold && curr > threshold) ||
            (prev >= threshold && curr < threshold)) {
            // Estimate tipping point by linear interpolation
            double delta_prev = sens_result.delta_values[i - 1];
            double delta_curr = sens_result.delta_values[i];
            double ratio = (threshold - prev) / (curr - prev);
            result.tipping_point = delta_prev + ratio * (delta_curr - delta_prev);
            result.found = true;
            break;
        }
    }

    if (result.found) {
        result.interpretation =
            "Tipping point found at delta = " + std::to_string(result.tipping_point) +
            ". At this value of delta (shift in missing values), "
            "the estimated mean crosses the threshold of " +
            std::to_string(threshold) + ".";
    } else {
        result.interpretation =
            "No tipping point found in the specified range. "
            "The conclusion is robust to MNAR assumptions within this range.";
    }

    return result;
}

// ============================================================================
// Complete Case Analysis Utilities
// ============================================================================

/**
 * @brief Complete case analysis result
 *
 * Structure storing results of complete case analysis (listwise deletion).
 * Contains complete case data, number of complete cases, and number of deleted cases.
 */
struct complete_case_result {
    std::vector<std::vector<double>> complete_data;    ///< Data with only complete cases
    std::size_t n_complete = 0;                        ///< Number of complete cases
    std::size_t n_dropped = 0;                         ///< Number of deleted cases
    double proportion_complete = 0.0;                  ///< Proportion of complete cases
};

/**
 * @brief Extract complete cases
 *
 * Extracts only rows without missing values (complete cases) from the dataset.
 * This is also called listwise deletion.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @return complete_case_result Complete case analysis result
 */
inline complete_case_result extract_complete_cases(
    const std::vector<std::vector<double>>& data)
{
    complete_case_result result;
    result.n_dropped = 0;

    for (const auto& row : data) {
        bool is_complete = true;
        for (double val : row) {
            if (is_na(val)) {
                is_complete = false;
                break;
            }
        }
        if (is_complete) {
            result.complete_data.push_back(row);
        } else {
            ++result.n_dropped;
        }
    }

    result.n_complete = result.complete_data.size();
    result.proportion_complete = data.empty() ? 0.0 :
        static_cast<double>(result.n_complete) / static_cast<double>(data.size());

    return result;
}

/**
 * @brief Correlation matrix using available case analysis (pairwise deletion)
 *
 * Calculates correlation coefficients using only cases where both variables
 * are observed for each pair of variables. Also called pairwise deletion.
 *
 * @param data Input data (rows: observations, columns: variables)
 * @return std::vector<std::vector<double>> Correlation matrix (NA if insufficient observations)
 */
inline std::vector<std::vector<double>> correlation_matrix_pairwise(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) {
        return {};
    }

    std::size_t n_cols = data[0].size();
    std::vector<std::vector<double>> corr_matrix(n_cols, std::vector<double>(n_cols, 1.0));

    for (std::size_t i = 0; i < n_cols; ++i) {
        for (std::size_t j = i + 1; j < n_cols; ++j) {
            // Extract cases where both are observed
            std::vector<double> x_vals, y_vals;
            for (const auto& row : data) {
                if (!is_na(row[i]) && !is_na(row[j])) {
                    x_vals.push_back(row[i]);
                    y_vals.push_back(row[j]);
                }
            }

            if (x_vals.size() >= 2) {
                double r = pearson_correlation(x_vals.begin(), x_vals.end(),
                                               y_vals.begin(), y_vals.end());
                corr_matrix[i][j] = r;
                corr_matrix[j][i] = r;
            } else {
                corr_matrix[i][j] = NA;
                corr_matrix[j][i] = NA;
            }
        }
    }

    return corr_matrix;
}

}  // namespace statcpp

#endif  // STATCPP_MISSING_DATA_HPP
