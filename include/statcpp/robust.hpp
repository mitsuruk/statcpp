/**
 * @file robust.hpp
 * @brief Robust statistics implementation
 *
 * Provides MAD, outlier detection, winsorization, Cook's distance, robust estimators, and more.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

namespace statcpp {

// ============================================================================
// Median Absolute Deviation (MAD)
// ============================================================================

/**
 * @brief Median Absolute Deviation (MAD)
 *
 * Computes the median of absolute deviations from the median.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return MAD value
 * @throws std::invalid_argument If range is empty
 *
 * @note Used as a robust estimator of variance.
 */
template <typename Iterator>
double mad(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mad: empty range");
    }

    // Copy and sort data (for median calculation)
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    double med = statcpp::median(sorted_data.begin(), sorted_data.end());

    // Calculate absolute deviations from median
    std::vector<double> abs_deviations;
    abs_deviations.reserve(n);
    for (double val : sorted_data) {
        abs_deviations.push_back(std::abs(val - med));
    }

    // Median of absolute deviations
    std::sort(abs_deviations.begin(), abs_deviations.end());
    return statcpp::median(abs_deviations.begin(), abs_deviations.end());
}

/**
 * @brief Scaled MAD for normal distribution
 *
 * Computes MAD scaled to estimate standard deviation for normal distribution.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return Scaled MAD value
 * @throws std::invalid_argument If range is empty
 *
 * @note Scale factor derivation:
 *       For standard normal N(0,1), the median of absolute deviations |X - 0| = |X|
 *       is the inverse of the normal quantile function at 0.75, approximately 0.6745.
 *       Therefore, to use MAD as an estimator of standard deviation sigma:
 *       sigma = MAD / 0.6745 = MAD * 1.4826
 *
 *       This factor is exact only when the data follows a normal distribution.
 *       For non-normal distributions, the standard deviation estimate using this
 *       factor is an approximation.
 */
template <typename Iterator>
double mad_scaled(Iterator first, Iterator last)
{
    return 1.4826 * mad(first, last);
}

// ============================================================================
// Outlier Detection (IQR Method / Tukey's Fences)
// ============================================================================

/**
 * @brief Outlier detection result
 */
struct outlier_detection_result {
    std::vector<double> outliers;                  ///< Outliers
    std::vector<std::size_t> outlier_indices;      ///< Outlier indices
    double lower_fence;                            ///< Lower fence
    double upper_fence;                            ///< Upper fence
    double q1;                                     ///< First quartile
    double q3;                                     ///< Third quartile
    double iqr_value;                              ///< Interquartile range
};

/**
 * @brief Outlier detection using IQR method (Tukey's Fences)
 *
 * Detects outliers using the interquartile range (Tukey's Fences).
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param k Fence multiplier (default: 1.5, use 3.0 for extreme outliers)
 * @return Outlier detection result
 * @throws std::invalid_argument If range is empty
 *
 * @note Standard box plots use k=1.5.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_iqr(Iterator first, Iterator last, double k = 1.5)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::detect_outliers_iqr: empty range");
    }

    // Copy and sort data
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    auto q = statcpp::quartiles(sorted_data.begin(), sorted_data.end());
    double iqr_val = q.q3 - q.q1;

    double lower_fence = q.q1 - k * iqr_val;
    double upper_fence = q.q3 + k * iqr_val;

    outlier_detection_result result;
    result.lower_fence = lower_fence;
    result.upper_fence = upper_fence;
    result.q1 = q.q1;
    result.q3 = q.q3;
    result.iqr_value = iqr_val;

    // Detect outliers
    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        if (val < lower_fence || val > upper_fence) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

// ============================================================================
// Z-score Outlier Detection
// ============================================================================

/**
 * @brief Outlier detection using Z-score
 *
 * Detects outliers using Z-score (standardized values).
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param threshold Z-score threshold (default: 3.0)
 * @return Outlier detection result
 * @throws std::invalid_argument If range has less than 2 elements or standard deviation is zero
 *
 * @note Assumes normal distribution. Sensitive to outliers.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_zscore(Iterator first, Iterator last, double threshold = 3.0)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 2) {
        throw std::invalid_argument("statcpp::detect_outliers_zscore: need at least 2 elements");
    }

    double m = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last, m);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::detect_outliers_zscore: zero standard deviation");
    }

    outlier_detection_result result;
    result.lower_fence = m - threshold * s;
    result.upper_fence = m + threshold * s;
    result.q1 = 0.0;  // Z-score method does not use quartiles
    result.q3 = 0.0;
    result.iqr_value = 0.0;

    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        double z = (val - m) / s;
        if (std::abs(z) > threshold) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

/**
 * @brief Outlier detection using Modified Z-score
 *
 * Detects outliers using MAD-based Modified Z-score (more robust).
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param threshold Modified Z-score threshold (default: 3.5)
 * @return Outlier detection result
 * @throws std::invalid_argument If range is empty or MAD is zero
 *
 * @note More robust than standard Z-score. Less affected by outliers.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_modified_zscore(Iterator first, Iterator last, double threshold = 3.5)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::detect_outliers_modified_zscore: empty range");
    }

    // Copy and sort data
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    double med = statcpp::median(sorted_data.begin(), sorted_data.end());
    double mad_val = mad(first, last);

    if (mad_val == 0.0) {
        throw std::invalid_argument("statcpp::detect_outliers_modified_zscore: zero MAD");
    }

    // Modified Z-score = 0.6745 * (x - median) / MAD
    double scale = 0.6745;

    outlier_detection_result result;
    result.lower_fence = med - threshold * mad_val / scale;
    result.upper_fence = med + threshold * mad_val / scale;
    result.q1 = 0.0;
    result.q3 = 0.0;
    result.iqr_value = 0.0;

    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        double modified_z = scale * (val - med) / mad_val;
        if (std::abs(modified_z) > threshold) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

// ============================================================================
// Winsorization
// ============================================================================

/**
 * @brief Winsorization
 *
 * Replaces extreme values with specified percentile values.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param limits Proportion to replace from each tail (e.g., 0.05 = replace top and bottom 5%)
 * @return Winsorized data
 * @throws std::invalid_argument If range is empty or limits is invalid
 *
 * @note Reduces the impact of outliers while preserving the number of data points.
 */
template <typename Iterator>
std::vector<double> winsorize(Iterator first, Iterator last, double limits = 0.05)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::winsorize: empty range");
    }
    if (limits < 0.0 || limits >= 0.5) {
        throw std::invalid_argument("statcpp::winsorize: limits must be in [0, 0.5)");
    }

    // Copy and sort data
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    // Calculate thresholds
    double lower_percentile = limits;
    double upper_percentile = 1.0 - limits;
    double lower_value = statcpp::percentile(sorted_data.begin(), sorted_data.end(), lower_percentile);
    double upper_value = statcpp::percentile(sorted_data.begin(), sorted_data.end(), upper_percentile);

    // Winsorize
    std::vector<double> result;
    result.reserve(n);
    for (auto it = first; it != last; ++it) {
        double val = static_cast<double>(*it);
        if (val < lower_value) {
            result.push_back(lower_value);
        } else if (val > upper_value) {
            result.push_back(upper_value);
        } else {
            result.push_back(val);
        }
    }

    return result;
}

// ============================================================================
// Cook's Distance (for Linear Regression Diagnostics)
// ============================================================================

/**
 * @brief Calculate Cook's Distance
 *
 * Calculates Cook's Distance to detect influential observations in linear regression.
 *
 * @param residuals Residuals
 * @param hat_values Leverage values (hat values)
 * @param mse Mean squared error
 * @param p Number of parameters (including intercept)
 * @return Cook's distances
 * @throws std::invalid_argument If parameters are invalid
 *
 * @note D_i > 1 indicates the observation is influential.
 * @note D_i > 4/n is also sometimes used as a threshold.
 */
inline std::vector<double> cooks_distance(
    const std::vector<double>& residuals,
    const std::vector<double>& hat_values,
    double mse,
    std::size_t p)
{
    if (residuals.size() != hat_values.size()) {
        throw std::invalid_argument("statcpp::cooks_distance: residuals and hat_values must have same length");
    }
    if (residuals.empty()) {
        throw std::invalid_argument("statcpp::cooks_distance: empty data");
    }
    if (mse <= 0) {
        throw std::invalid_argument("statcpp::cooks_distance: mse must be positive");
    }
    if (p == 0) {
        throw std::invalid_argument("statcpp::cooks_distance: p must be positive");
    }

    std::vector<double> result;
    result.reserve(residuals.size());

    for (std::size_t i = 0; i < residuals.size(); ++i) {
        double h = hat_values[i];
        if (h >= 1.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        double e = residuals[i];
        double d = (e * e / (static_cast<double>(p) * mse)) *
                   (h / ((1.0 - h) * (1.0 - h)));
        result.push_back(d);
    }

    return result;
}

// ============================================================================
// DFFITS (Difference in Fits)
// ============================================================================

/**
 * @brief Calculate DFFITS
 *
 * Calculates DFFITS to measure the influence of each observation on predicted values.
 *
 * @param residuals Residuals
 * @param hat_values Leverage values (hat values)
 * @param mse Mean squared error
 * @return DFFITS values
 * @throws std::invalid_argument If parameters are invalid
 *
 * @note |DFFITS_i| > 2*sqrt(p/n) indicates the observation is influential.
 */
inline std::vector<double> dffits(
    const std::vector<double>& residuals,
    const std::vector<double>& hat_values,
    double mse)
{
    if (residuals.size() != hat_values.size()) {
        throw std::invalid_argument("statcpp::dffits: residuals and hat_values must have same length");
    }
    if (residuals.empty()) {
        throw std::invalid_argument("statcpp::dffits: empty data");
    }

    std::vector<double> result;
    result.reserve(residuals.size());

    for (std::size_t i = 0; i < residuals.size(); ++i) {
        double h = hat_values[i];
        if (h >= 1.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        double e = residuals[i];
        // Studentized residual
        double se = std::sqrt(mse * (1.0 - h));
        if (se == 0.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }
        double t = e / se;

        // DFFITS
        double dffits_val = t * std::sqrt(h / (1.0 - h));
        result.push_back(dffits_val);
    }

    return result;
}

// ============================================================================
// Robust Location Estimators
// ============================================================================

/**
 * @brief Hodges-Lehmann estimator
 *
 * Calculates the median of all pairwise averages (Walsh average).
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @return Hodges-Lehmann estimate
 * @throws std::invalid_argument If range is empty
 *
 * @note A robust location estimator. Less affected by outliers.
 */
template <typename Iterator>
double hodges_lehmann(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::hodges_lehmann: empty range");
    }

    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }

    // Calculate all pairwise averages
    std::vector<double> pairwise_means;
    pairwise_means.reserve(n * (n + 1) / 2);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            pairwise_means.push_back((data[i] + data[j]) / 2.0);
        }
    }

    std::sort(pairwise_means.begin(), pairwise_means.end());
    return statcpp::median(pairwise_means.begin(), pairwise_means.end());
}

/**
 * @brief Biweight Midvariance
 *
 * Calculates the Biweight Midvariance, a robust variance estimator.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning of range
 * @param last End of range
 * @param c Tuning constant (default: 9.0)
 * @return Biweight midvariance
 * @throws std::invalid_argument If range has less than 2 elements
 *
 * @note A variance estimator less affected by outliers.
 */
template <typename Iterator>
double biweight_midvariance(Iterator first, Iterator last, double c = 9.0)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 2) {
        throw std::invalid_argument("statcpp::biweight_midvariance: need at least 2 elements");
    }

    // Copy and sort data
    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }
    std::sort(data.begin(), data.end());

    double med = statcpp::median(data.begin(), data.end());
    double mad_val = mad(first, last);

    if (mad_val == 0.0) {
        return 0.0;
    }

    double num = 0.0;
    double den = 0.0;
    std::size_t count = 0;

    for (double x : data) {
        double u = (x - med) / (c * mad_val);
        if (std::abs(u) < 1.0) {
            double u2 = u * u;
            double w = (1.0 - u2);
            double w2 = w * w;
            double w4 = w2 * w2;
            num += (x - med) * (x - med) * w4;
            den += w2 * (1.0 - 5.0 * u2);
            count++;
        }
    }

    if (den == 0.0 || count < 2) {
        return 0.0;
    }

    return static_cast<double>(n) * num / (den * den);
}

} // namespace statcpp
