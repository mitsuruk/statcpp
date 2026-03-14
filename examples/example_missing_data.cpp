/**
 * @file example_missing_data.cpp
 * @brief Missing Data Analysis Sample Code
 *
 * Demonstrates the usage of missing pattern analysis, MCAR testing,
 * multiple imputation, sensitivity analysis, tipping point analysis,
 * and other missing data handling techniques.
 */

#include <iostream>
#include <iomanip>
#include "statcpp/missing_data.hpp"

int main() {
    std::cout << "=== Missing Data Analysis Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Missing Pattern Analysis
    // ============================================================================
    std::cout << "\n1. Missing Pattern Analysis" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {statcpp::NA, 2.0, 3.0},
        {1.0, statcpp::NA, 3.0},
        {1.0, 2.0, statcpp::NA},
        {1.0, 2.0, 3.0},
        {statcpp::NA, statcpp::NA, 3.0},
        {4.0, 5.0, 6.0}
    };

    auto pattern_info = statcpp::analyze_missing_patterns(data);

    std::cout << "Complete cases: " << pattern_info.n_complete_cases << std::endl;
    std::cout << "Overall missing rate: " << pattern_info.overall_missing_rate << std::endl;
    std::cout << "Number of unique patterns: " << pattern_info.n_patterns << std::endl;

    std::cout << "\nMissing rate by variable:" << std::endl;
    for (std::size_t i = 0; i < pattern_info.missing_rates.size(); ++i) {
        std::cout << "  Variable " << i << ": " << pattern_info.missing_rates[i] << std::endl;
    }

    std::cout << "\nMissing patterns:" << std::endl;
    for (std::size_t i = 0; i < pattern_info.patterns.size(); ++i) {
        std::cout << "  Pattern " << i << ": [";
        for (std::size_t j = 0; j < pattern_info.patterns[i].size(); ++j) {
            std::cout << static_cast<int>(pattern_info.patterns[i][j]);
            if (j + 1 < pattern_info.patterns[i].size()) std::cout << ", ";
        }
        std::cout << "] - Count: " << pattern_info.pattern_counts[i] << std::endl;
    }

    // ============================================================================
    // 2. MCAR Test
    // ============================================================================
    std::cout << "\n2. Little's MCAR Test (Simplified Version)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> test_data;
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(i % 10);

        // Insert missing values randomly
        if (i % 5 == 0) x = statcpp::NA;
        if (i % 7 == 0) y = statcpp::NA;

        test_data.push_back({x, y});
    }

    auto mcar_result = statcpp::test_mcar_simple(test_data);

    std::cout << "Chi-square statistic: " << mcar_result.chi_square << std::endl;
    std::cout << "Degrees of freedom: " << mcar_result.df << std::endl;
    std::cout << "P-value: " << mcar_result.p_value << std::endl;
    std::cout << "Is MCAR? " << (mcar_result.is_mcar ? "Yes" : "No") << std::endl;
    std::cout << "Interpretation: " << mcar_result.interpretation << std::endl;

    // ============================================================================
    // 3. Missing Mechanism Diagnosis
    // ============================================================================
    std::cout << "\n3. Missing Mechanism Diagnosis" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto mechanism = statcpp::diagnose_missing_mechanism(test_data);

    std::cout << "Detected mechanism: ";
    switch (mechanism) {
        case statcpp::missing_mechanism::mcar:
            std::cout << "MCAR (Missing Completely At Random)" << std::endl;
            break;
        case statcpp::missing_mechanism::mar:
            std::cout << "MAR (Missing At Random)" << std::endl;
            break;
        case statcpp::missing_mechanism::mnar:
            std::cout << "MNAR (Missing Not At Random)" << std::endl;
            break;
        case statcpp::missing_mechanism::unknown:
            std::cout << "Unknown" << std::endl;
            break;
    }

    // ============================================================================
    // 4. Multiple Imputation (PMM)
    // ============================================================================
    std::cout << "\n4. Multiple Imputation (PMM)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> impute_data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {statcpp::NA, 6.0},
        {4.0, 8.0},
        {5.0, statcpp::NA},
        {6.0, 12.0}
    };

    auto mi_result = statcpp::multiple_imputation_pmm(impute_data, 5, 42);

    std::cout << "Number of imputations: " << mi_result.m << std::endl;
    std::cout << "\nPooled means:" << std::endl;
    for (std::size_t i = 0; i < mi_result.pooled_means.size(); ++i) {
        std::cout << "  Variable " << i << ": " << mi_result.pooled_means[i] << std::endl;
    }

    std::cout << "\nPooled variances:" << std::endl;
    for (std::size_t i = 0; i < mi_result.pooled_vars.size(); ++i) {
        std::cout << "  Variable " << i << ": " << mi_result.pooled_vars[i] << std::endl;
    }

    std::cout << "\nFraction of Missing Information (FMI):" << std::endl;
    for (std::size_t i = 0; i < mi_result.fraction_missing_info.size(); ++i) {
        std::cout << "  Variable " << i << ": " << mi_result.fraction_missing_info[i] << std::endl;
    }

    // ============================================================================
    // 5. Sensitivity Analysis (Pattern Mixture Model)
    // ============================================================================
    std::cout << "\n5. Sensitivity Analysis (Pattern Mixture Model)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> sens_data = {1.0, 2.0, 3.0, statcpp::NA, 5.0, statcpp::NA};
    std::vector<double> delta_values = {-2.0, -1.0, 0.0, 1.0, 2.0};

    auto sens_result = statcpp::sensitivity_analysis_pattern_mixture(sens_data, delta_values);

    std::cout << "Original mean estimate: " << sens_result.original_mean << std::endl;
    std::cout << "\nEstimated means under different assumptions:" << std::endl;
    for (std::size_t i = 0; i < sens_result.delta_values.size(); ++i) {
        std::cout << "  delta = " << std::setw(5) << sens_result.delta_values[i]
                  << " -> mean = " << sens_result.estimated_means[i] << std::endl;
    }

    // ============================================================================
    // 6. Tipping Point Analysis
    // ============================================================================
    std::cout << "\n6. Tipping Point Analysis" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> tip_data = {1.0, 2.0, 3.0, statcpp::NA, statcpp::NA};
    auto tip_result = statcpp::find_tipping_point(tip_data, 0.0, -10.0, 10.0, 100);

    if (tip_result.found) {
        std::cout << "Tipping point found!" << std::endl;
        std::cout << "  Delta value: " << tip_result.tipping_point << std::endl;
        std::cout << "  Threshold: " << tip_result.threshold << std::endl;
    } else {
        std::cout << "No tipping point found within specified range." << std::endl;
    }
    std::cout << "Interpretation: " << tip_result.interpretation << std::endl;

    // ============================================================================
    // 7. Complete Case Analysis
    // ============================================================================
    std::cout << "\n7. Complete Case Analysis" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto cc_result = statcpp::extract_complete_cases(data);

    std::cout << "Complete cases: " << cc_result.n_complete << std::endl;
    std::cout << "Dropped cases: " << cc_result.n_dropped << std::endl;
    std::cout << "Proportion complete: " << cc_result.proportion_complete << std::endl;

    std::cout << "\nComplete case data:" << std::endl;
    for (const auto& row : cc_result.complete_data) {
        std::cout << "  [";
        for (std::size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i + 1 < row.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // ============================================================================
    // 8. Pairwise Correlation (with Missing Values)
    // ============================================================================
    std::cout << "\n8. Pairwise Correlation Matrix" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::vector<double>> corr_data = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {3.0, statcpp::NA, 9.0},
        {4.0, 8.0, statcpp::NA},
        {5.0, 10.0, 15.0}
    };

    auto corr_matrix = statcpp::correlation_matrix_pairwise(corr_data);

    std::cout << "Correlation matrix (pairwise deletion):" << std::endl;
    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < corr_matrix[i].size(); ++j) {
            if (statcpp::is_na(corr_matrix[i][j])) {
                std::cout << std::setw(8) << "NA";
            } else {
                std::cout << std::setw(8) << corr_matrix[i][j];
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
