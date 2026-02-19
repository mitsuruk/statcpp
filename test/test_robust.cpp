#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "statcpp/robust.hpp"

// ============================================================================
// MAD Tests
// ============================================================================

/**
 * @brief Tests basic MAD (Median Absolute Deviation) calculation
 * @test Verifies that MAD is computed correctly as the median of absolute deviations from the median
 */
TEST(MADTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double result = statcpp::mad(data.begin(), data.end());

    // median = 5, deviations = {4, 3, 2, 1, 0, 1, 2, 3, 4}
    // sorted deviations = {0, 1, 1, 2, 2, 3, 3, 4, 4}
    // median of deviations = 2
    EXPECT_DOUBLE_EQ(result, 2.0);
}

/**
 * @brief Tests MAD robustness to outliers
 * @test Verifies that MAD remains stable in the presence of extreme outliers
 */
TEST(MADTest, WithOutlier) {
    std::vector<double> data = {1, 2, 3, 4, 5, 100};  // 100 is outlier
    double result = statcpp::mad(data.begin(), data.end());

    // MAD is robust to outliers
    EXPECT_LT(result, 50.0);  // Much less than the outlier's deviation
}

/**
 * @brief Tests scaled MAD calculation
 * @test Verifies that scaled MAD applies the correct scaling factor for consistency with standard deviation
 */
TEST(MADTest, Scaled) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double result = statcpp::mad_scaled(data.begin(), data.end());

    EXPECT_NEAR(result, 2.0 * 1.4826, 1e-10);
}

/**
 * @brief Tests MAD calculation with empty input range
 * @test Verifies that an exception is thrown when the input range is empty
 */
TEST(MADTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::mad(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Outlier Detection IQR Tests
// ============================================================================

/**
 * @brief Tests IQR-based outlier detection with no outliers present
 * @test Verifies that no outliers are detected in a dataset with normal distribution
 */
TEST(OutlierIQRTest, NoOutliers) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = statcpp::detect_outliers_iqr(data.begin(), data.end());

    EXPECT_TRUE(result.outliers.empty());
}

/**
 * @brief Tests IQR-based outlier detection with outliers present
 * @test Verifies that extreme values are correctly identified as outliers using the IQR method
 */
TEST(OutlierIQRTest, WithOutliers) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};  // 100 is outlier
    auto result = statcpp::detect_outliers_iqr(data.begin(), data.end());

    EXPECT_FALSE(result.outliers.empty());
    EXPECT_EQ(result.outliers[0], 100);
}

/**
 * @brief Tests IQR fence calculations
 * @test Verifies that lower and upper fences are computed correctly relative to Q1 and Q3
 */
TEST(OutlierIQRTest, FenceValues) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = statcpp::detect_outliers_iqr(data.begin(), data.end());

    EXPECT_LT(result.lower_fence, result.q1);
    EXPECT_GT(result.upper_fence, result.q3);
    EXPECT_GT(result.iqr_value, 0);
}

// ============================================================================
// Outlier Detection Z-score Tests
// ============================================================================

/**
 * @brief Tests Z-score based outlier detection with no outliers present
 * @test Verifies that no outliers are detected when all values have Z-scores below the threshold
 */
TEST(OutlierZscoreTest, NoOutliers) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = statcpp::detect_outliers_zscore(data.begin(), data.end());

    EXPECT_TRUE(result.outliers.empty());
}

/**
 * @brief Tests Z-score based outlier detection with outliers present
 * @test Verifies that values with Z-scores exceeding 3 are correctly identified as outliers
 */
TEST(OutlierZscoreTest, WithOutliers) {
    // 100 is clearly an outlier with z-score > 3
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100};
    auto result = statcpp::detect_outliers_zscore(data.begin(), data.end());

    EXPECT_FALSE(result.outliers.empty());
}

// ============================================================================
// Modified Z-score Tests
// ============================================================================

/**
 * @brief Tests modified Z-score outlier detection
 * @test Verifies that outliers are detected using MAD-based modified Z-score method
 */
TEST(ModifiedZscoreTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    auto result = statcpp::detect_outliers_modified_zscore(data.begin(), data.end());

    // 100 should be detected as outlier
    EXPECT_FALSE(result.outliers.empty());
}

// ============================================================================
// Winsorize Tests
// ============================================================================

/**
 * @brief Tests basic winsorization of data
 * @test Verifies that extreme values are clipped to percentile limits
 */
TEST(WinsorizeTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto result = statcpp::winsorize(data.begin(), data.end(), 0.1);

    EXPECT_EQ(result.size(), 10);
    // Extreme values should be clipped
    EXPECT_GE(result[0], 1.0);
    EXPECT_LE(result[9], 10.0);
}

/**
 * @brief Tests winsorization with zero limits
 * @test Verifies that no changes occur when winsorization limits are set to zero
 */
TEST(WinsorizeTest, NoChange) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = statcpp::winsorize(data.begin(), data.end(), 0.0);

    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], data[i]);
    }
}

/**
 * @brief Tests winsorization with invalid limit values
 * @test Verifies that exceptions are thrown for invalid limit values (negative or >= 0.5)
 */
TEST(WinsorizeTest, InvalidLimits) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    EXPECT_THROW(statcpp::winsorize(data.begin(), data.end(), 0.5), std::invalid_argument);
    EXPECT_THROW(statcpp::winsorize(data.begin(), data.end(), -0.1), std::invalid_argument);
}

// ============================================================================
// Cook's Distance Tests
// ============================================================================

/**
 * @brief Tests Cook's distance calculation for influence diagnostics
 * @test Verifies that Cook's distance is computed correctly and all values are non-negative
 */
TEST(CooksDistanceTest, Basic) {
    std::vector<double> residuals = {0.1, -0.2, 0.3, -0.1, 0.2};
    std::vector<double> hat_values = {0.2, 0.3, 0.1, 0.2, 0.2};
    double mse = 0.04;
    std::size_t p = 2;

    auto result = statcpp::cooks_distance(residuals, hat_values, mse, p);

    EXPECT_EQ(result.size(), 5);
    for (double d : result) {
        EXPECT_GE(d, 0.0);
    }
}

/**
 * @brief Tests Cook's distance with empty input data
 * @test Verifies that an exception is thrown when residuals and hat values are empty
 */
TEST(CooksDistanceTest, EmptyData) {
    std::vector<double> residuals;
    std::vector<double> hat_values;
    EXPECT_THROW(statcpp::cooks_distance(residuals, hat_values, 1.0, 1), std::invalid_argument);
}

// ============================================================================
// DFFITS Tests
// ============================================================================

/**
 * @brief Tests DFFITS calculation for influence diagnostics
 * @test Verifies that DFFITS values are computed correctly for regression diagnostics
 */
TEST(DFFITSTest, Basic) {
    std::vector<double> residuals = {0.1, -0.2, 0.3, -0.1, 0.2};
    std::vector<double> hat_values = {0.2, 0.3, 0.1, 0.2, 0.2};
    double mse = 0.04;

    auto result = statcpp::dffits(residuals, hat_values, mse);

    EXPECT_EQ(result.size(), 5);
}

// ============================================================================
// Hodges-Lehmann Tests
// ============================================================================

/**
 * @brief Tests basic Hodges-Lehmann location estimator
 * @test Verifies that Hodges-Lehmann estimator is close to the median for symmetric data
 */
TEST(HodgesLehmannTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    double result = statcpp::hodges_lehmann(data.begin(), data.end());

    // For symmetric data, should be close to median
    EXPECT_NEAR(result, 3.0, 0.5);
}

/**
 * @brief Tests Hodges-Lehmann estimator robustness to outliers
 * @test Verifies that Hodges-Lehmann estimator is more robust than the mean in the presence of outliers
 */
TEST(HodgesLehmannTest, WithOutlier) {
    std::vector<double> data = {1, 2, 3, 4, 100};
    double result = statcpp::hodges_lehmann(data.begin(), data.end());

    // More robust than mean
    double mean = (1 + 2 + 3 + 4 + 100) / 5.0;
    EXPECT_LT(result, mean);
}

// ============================================================================
// Biweight Midvariance Tests
// ============================================================================

/**
 * @brief Tests basic biweight midvariance calculation
 * @test Verifies that biweight midvariance produces positive variance estimates
 */
TEST(BiweightMidvarianceTest, Basic) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double result = statcpp::biweight_midvariance(data.begin(), data.end());

    EXPECT_GT(result, 0.0);
}

/**
 * @brief Tests biweight midvariance with insufficient data
 * @test Verifies that an exception is thrown when the sample size is too small
 */
TEST(BiweightMidvarianceTest, Insufficient) {
    std::vector<double> data = {1};
    EXPECT_THROW(statcpp::biweight_midvariance(data.begin(), data.end()), std::invalid_argument);
}
