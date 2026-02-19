/**
 * @file example_robust.cpp
 * @brief Sample code for robust statistics
 *
 * Demonstrates usage examples of robust statistical methods including
 * MAD (Median Absolute Deviation), outlier detection (IQR method, Z-score method),
 * Winsorization, and Hodges-Lehmann estimator.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/robust.hpp"
#include "statcpp/dispersion_spread.hpp"

int main() {
    std::cout << "=== Robust Statistics Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. MAD (Median Absolute Deviation)
    // ============================================================================
    std::cout << "\n1. MAD (Median Absolute Deviation)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double median_val = statcpp::median(data.begin(), data.end());
    double mad_val = statcpp::mad(data.begin(), data.end());
    double mad_scaled_val = statcpp::mad_scaled(data.begin(), data.end());
    double stddev_val = statcpp::sample_stddev(data.begin(), data.end());

    std::cout << "Data: [";
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\nMeasures of spread:" << std::endl;
    std::cout << "  Median: " << median_val << std::endl;
    std::cout << "  MAD: " << mad_val << std::endl;
    std::cout << "  MAD (scaled): " << mad_scaled_val << std::endl;
    std::cout << "  Standard deviation: " << stddev_val << std::endl;

    std::cout << "\nNote: For normally distributed data, scaled MAD ~ standard deviation" << std::endl;
    std::cout << "    Scale factor = 1.4826" << std::endl;

    // Compare MAD and SD with outlier
    std::vector<double> data_with_outlier = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double mad_scaled_outlier = statcpp::mad_scaled(data_with_outlier.begin(), data_with_outlier.end());
    double stddev_outlier = statcpp::sample_stddev(data_with_outlier.begin(), data_with_outlier.end());

    std::cout << "\nWith outlier (100):" << std::endl;
    std::cout << "  MAD (scaled): " << mad_scaled_outlier << std::endl;
    std::cout << "  Standard deviation: " << stddev_outlier << std::endl;
    std::cout << "  MAD is robust: change = " << (mad_scaled_outlier - mad_scaled_val) << std::endl;
    std::cout << "  Std dev is not robust: change = " << (stddev_outlier - stddev_val) << std::endl;

    // ============================================================================
    // 2. Outlier Detection (IQR Method)
    // ============================================================================
    std::cout << "\n2. Outlier Detection (IQR Method - Tukey's Fences)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> test_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 50, 55  // 50 and 55 are outliers
    };

    auto iqr_result = statcpp::detect_outliers_iqr(test_data.begin(), test_data.end(), 1.5);

    std::cout << "Quartiles:" << std::endl;
    std::cout << "  Q1: " << iqr_result.q1 << std::endl;
    std::cout << "  Q3: " << iqr_result.q3 << std::endl;
    std::cout << "  IQR: " << iqr_result.iqr_value << std::endl;

    std::cout << "\nFences (k=1.5):" << std::endl;
    std::cout << "  Lower fence: " << iqr_result.lower_fence << std::endl;
    std::cout << "  Upper fence: " << iqr_result.upper_fence << std::endl;

    std::cout << "\nDetected outliers (" << iqr_result.outliers.size() << "):" << std::endl;
    for (std::size_t i = 0; i < iqr_result.outliers.size(); ++i) {
        std::cout << "  Index " << iqr_result.outlier_indices[i]
                  << ": value = " << iqr_result.outliers[i] << std::endl;
    }

    // Extreme outlier detection (k=3.0)
    auto extreme_outliers = statcpp::detect_outliers_iqr(test_data.begin(), test_data.end(), 3.0);
    std::cout << "\nExtreme outliers (k=3.0): " << extreme_outliers.outliers.size() << std::endl;

    // ============================================================================
    // 3. Outlier Detection (Z-score Method)
    // ============================================================================
    std::cout << "\n3. Outlier Detection (Z-score Method)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto zscore_result = statcpp::detect_outliers_zscore(test_data.begin(), test_data.end(), 3.0);

    std::cout << "Threshold (Z = 3.0):" << std::endl;
    std::cout << "  Lower bound: " << zscore_result.lower_fence << std::endl;
    std::cout << "  Upper bound: " << zscore_result.upper_fence << std::endl;

    std::cout << "\nDetected outliers (" << zscore_result.outliers.size() << "):" << std::endl;
    for (std::size_t i = 0; i < zscore_result.outliers.size(); ++i) {
        std::cout << "  Index " << zscore_result.outlier_indices[i]
                  << ": value = " << zscore_result.outliers[i] << std::endl;
    }

    // ============================================================================
    // 4. Outlier Detection (Modified Z-score Method)
    // ============================================================================
    std::cout << "\n4. Outlier Detection (Modified Z-score Method - MAD-based)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto modified_zscore_result = statcpp::detect_outliers_modified_zscore(
        test_data.begin(), test_data.end(), 3.5
    );

    std::cout << "Modified Z-score method (more robust):" << std::endl;
    std::cout << "  Detected outliers: " << modified_zscore_result.outliers.size() << std::endl;
    for (std::size_t i = 0; i < modified_zscore_result.outliers.size(); ++i) {
        std::cout << "  Index " << modified_zscore_result.outlier_indices[i]
                  << ": value = " << modified_zscore_result.outliers[i] << std::endl;
    }

    // ============================================================================
    // 5. Comparison of Outlier Detection Methods
    // ============================================================================
    std::cout << "\n5. Comparison of Outlier Detection Methods" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Number of detected outliers:" << std::endl;
    std::cout << "  IQR method (k=1.5): " << iqr_result.outliers.size() << std::endl;
    std::cout << "  Z-score method (threshold=3.0): " << zscore_result.outliers.size() << std::endl;
    std::cout << "  Modified Z-score (threshold=3.5): " << modified_zscore_result.outliers.size() << std::endl;

    std::cout << "\nRecommendations:" << std::endl;
    std::cout << "  - IQR method: Best for skewed distributions" << std::endl;
    std::cout << "  - Z-score method: Assumes normality, sensitive to outliers" << std::endl;
    std::cout << "  - Modified Z-score: Robust, suitable for non-normal data" << std::endl;

    // ============================================================================
    // 6. Winsorization
    // ============================================================================
    std::cout << "\n6. Winsorization" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> winsor_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};

    auto winsorized = statcpp::winsorize(winsor_data.begin(), winsor_data.end(), 0.1);

    std::cout << "Original data: [";
    for (std::size_t i = 0; i < winsor_data.size(); ++i) {
        std::cout << winsor_data[i];
        if (i + 1 < winsor_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Winsorized (10%): [";
    for (std::size_t i = 0; i < winsorized.size(); ++i) {
        std::cout << winsorized[i];
        if (i + 1 < winsorized.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    double mean_original = statcpp::mean(winsor_data.begin(), winsor_data.end());
    double mean_winsorized = statcpp::mean(winsorized.begin(), winsorized.end());

    std::cout << "\nMean before Winsorization: " << mean_original << std::endl;
    std::cout << "Mean after Winsorization: " << mean_winsorized << std::endl;
    std::cout << "Change: " << (mean_original - mean_winsorized) << std::endl;

    std::cout << "\nNote: Winsorization replaces extreme values with less extreme" << std::endl;
    std::cout << "    values rather than removing them (unlike trimming)." << std::endl;

    // ============================================================================
    // 7. Cook's Distance (Regression Diagnostics)
    // ============================================================================
    std::cout << "\n7. Cook's Distance (Regression Diagnostics)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Hypothetical regression data
    std::vector<double> residuals = {0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 5.0, 0.3};
    std::vector<double> hat_values = {0.15, 0.12, 0.18, 0.10, 0.14, 0.11, 0.25, 0.13};
    double mse = 1.5;
    std::size_t p = 2;  // intercept + 1 variable

    auto cooks_d = statcpp::cooks_distance(residuals, hat_values, mse, p);

    std::cout << "Cook's Distance values:" << std::endl;
    for (std::size_t i = 0; i < cooks_d.size(); ++i) {
        std::cout << "  Observation " << (i + 1) << ": D = " << cooks_d[i];
        if (cooks_d[i] > 1.0) {
            std::cout << " (influential)";
        } else if (cooks_d[i] > 0.5) {
            std::cout << " (potentially influential)";
        }
        std::cout << std::endl;
    }

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  D > 1.0: Highly influential observation" << std::endl;
    std::cout << "  D > 0.5: Potentially influential" << std::endl;
    std::cout << "  D > 4/n: Common rule of thumb (here: " << (4.0 / cooks_d.size()) << ")" << std::endl;

    // ============================================================================
    // 8. DFFITS (Regression Diagnostics)
    // ============================================================================
    std::cout << "\n8. DFFITS (Regression Diagnostics)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto dffits_vals = statcpp::dffits(residuals, hat_values, mse);

    std::cout << "DFFITS values:" << std::endl;
    double dffits_cutoff = 2.0 * std::sqrt(static_cast<double>(p) / cooks_d.size());
    std::cout << "Cutoff: +-" << dffits_cutoff << std::endl;

    for (std::size_t i = 0; i < dffits_vals.size(); ++i) {
        std::cout << "  Observation " << (i + 1) << ": DFFITS = " << dffits_vals[i];
        if (std::abs(dffits_vals[i]) > dffits_cutoff) {
            std::cout << " (influential)";
        }
        std::cout << std::endl;
    }

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  |DFFITS| > 2*sqrt(p/n): Influential observation" << std::endl;

    // ============================================================================
    // 9. Hodges-Lehmann Estimator
    // ============================================================================
    std::cout << "\n9. Hodges-Lehmann Estimator (Robust Location Estimate)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> location_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double mean_loc = statcpp::mean(location_data.begin(), location_data.end());
    double median_loc = statcpp::median(location_data.begin(), location_data.end());
    double hl_loc = statcpp::hodges_lehmann(location_data.begin(), location_data.end());

    std::cout << "Data with outlier: [1, 2, ..., 9, 100]" << std::endl;
    std::cout << "\nLocation estimates:" << std::endl;
    std::cout << "  Mean: " << mean_loc << " (affected by outlier)" << std::endl;
    std::cout << "  Median: " << median_loc << " (robust)" << std::endl;
    std::cout << "  Hodges-Lehmann: " << hl_loc << " (robust)" << std::endl;

    std::cout << "\nNote: Hodges-Lehmann is the median of all pairwise averages." << std::endl;
    std::cout << "    For symmetric distributions, it is more efficient than the median." << std::endl;

    // ============================================================================
    // 10. Biweight Midvariance (Robust Variance Estimate)
    // ============================================================================
    std::cout << "\n10. Biweight Midvariance (Robust Variance Estimate)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> variance_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double sample_var = statcpp::sample_variance(variance_data.begin(), variance_data.end());
    double bw_midvar = statcpp::biweight_midvariance(variance_data.begin(), variance_data.end());

    std::cout << "Clean data (no outliers):" << std::endl;
    std::cout << "  Sample variance: " << sample_var << std::endl;
    std::cout << "  Biweight midvariance: " << bw_midvar << std::endl;

    // Add outlier
    std::vector<double> variance_data_outlier = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0};

    double sample_var_out = statcpp::sample_variance(variance_data_outlier.begin(), variance_data_outlier.end());
    double bw_midvar_out = statcpp::biweight_midvariance(variance_data_outlier.begin(), variance_data_outlier.end());

    std::cout << "\nWith outlier (100):" << std::endl;
    std::cout << "  Sample variance: " << sample_var_out << " (inflated)" << std::endl;
    std::cout << "  Biweight midvariance: " << bw_midvar_out << " (robust)" << std::endl;

    std::cout << "\nChange due to outlier:" << std::endl;
    std::cout << "  Sample variance: +" << (sample_var_out - sample_var) << std::endl;
    std::cout << "  Biweight midvariance: +" << (bw_midvar_out - bw_midvar) << std::endl;

    // ============================================================================
    // 11. Summary: Importance of Robust Statistics
    // ============================================================================
    std::cout << "\n11. Summary: Importance of Robust Statistics" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Classical vs Robust Estimators:" << std::endl;
    std::cout << "\nLocation:" << std::endl;
    std::cout << "  Classical: Mean (sensitive to outliers)" << std::endl;
    std::cout << "  Robust: Median, Hodges-Lehmann (resistant to outliers)" << std::endl;

    std::cout << "\nScale (spread):" << std::endl;
    std::cout << "  Classical: Standard deviation (sensitive)" << std::endl;
    std::cout << "  Robust: MAD, Biweight Midvariance (resistant)" << std::endl;

    std::cout << "\nOutlier detection:" << std::endl;
    std::cout << "  Non-robust: Z-score (uses mean and std dev)" << std::endl;
    std::cout << "  Robust: IQR method, Modified Z-score (uses median and MAD)" << std::endl;

    std::cout << "\nWhen to use robust methods:" << std::endl;
    std::cout << "  - Data may contain outliers or measurement errors" << std::endl;
    std::cout << "  - Distribution is non-normal or heavy-tailed" << std::endl;
    std::cout << "  - Want to reduce influence of extreme values" << std::endl;
    std::cout << "  - Exploratory data analysis" << std::endl;

    std::cout << "\n=== Examples completed ===" << std::endl;

    return 0;
}
