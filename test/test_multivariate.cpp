#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "statcpp/multivariate.hpp"

// ============================================================================
// Covariance Matrix Tests
// ============================================================================

/**
 * @brief Tests basic covariance matrix computation
 * @test Verifies that covariance matrix is computed with correct dimensions and symmetry
 */
TEST(CovarianceMatrixTest, Basic) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0}
    };

    auto cov = statcpp::covariance_matrix(data);

    EXPECT_EQ(cov.size(), 2);
    EXPECT_EQ(cov[0].size(), 2);

    // Covariance matrix is symmetric
    EXPECT_DOUBLE_EQ(cov[0][1], cov[1][0]);
}

/**
 * @brief Tests error handling for empty data in covariance matrix
 * @test Verifies that covariance matrix computation throws exception for empty input
 */
TEST(CovarianceMatrixTest, EmptyData) {
    std::vector<std::vector<double>> data;
    EXPECT_THROW(statcpp::covariance_matrix(data), std::invalid_argument);
}

/**
 * @brief Tests error handling for insufficient data points
 * @test Verifies that covariance matrix computation throws exception when n < 2
 */
TEST(CovarianceMatrixTest, InsufficientData) {
    std::vector<std::vector<double>> data = {{1.0, 2.0}};
    EXPECT_THROW(statcpp::covariance_matrix(data), std::invalid_argument);
}

// ============================================================================
// Correlation Matrix Tests
// ============================================================================

/**
 * @brief Tests correlation matrix computation for perfectly correlated variables
 * @test Verifies that diagonal elements are 1 and off-diagonal elements approach 1 for perfect correlation
 */
TEST(CorrelationMatrixTest, PerfectCorrelation) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0}
    };

    auto corr = statcpp::correlation_matrix(data);

    // Diagonal elements are 1
    EXPECT_DOUBLE_EQ(corr[0][0], 1.0);
    EXPECT_DOUBLE_EQ(corr[1][1], 1.0);

    // Perfect correlation
    EXPECT_NEAR(corr[0][1], 1.0, 1e-10);
}

// ============================================================================
// Standardize Tests
// ============================================================================

/**
 * @brief Tests basic data standardization (z-score normalization)
 * @test Verifies that standardized data has mean approximately zero for each variable
 */
TEST(StandardizeTest, Basic) {
    std::vector<std::vector<double>> data = {
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0}
    };

    auto result = statcpp::standardize(data);

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0].size(), 2);

    // Mean of each variable is close to 0
    double col0_mean = (result[0][0] + result[1][0] + result[2][0]) / 3.0;
    double col1_mean = (result[0][1] + result[1][1] + result[2][1]) / 3.0;
    EXPECT_NEAR(col0_mean, 0.0, 1e-10);
    EXPECT_NEAR(col1_mean, 0.0, 1e-10);
}

// ============================================================================
// Min-Max Scale Tests
// ============================================================================

/**
 * @brief Tests min-max scaling to [0,1] range
 * @test Verifies that scaled data has minimum value 0 and maximum value 1 for each variable
 */
TEST(MinMaxScaleTest, Basic) {
    std::vector<std::vector<double>> data = {
        {0.0, 100.0},
        {50.0, 200.0},
        {100.0, 300.0}
    };

    auto result = statcpp::min_max_scale(data);

    // Minimum is 0, maximum is 1
    EXPECT_DOUBLE_EQ(result[0][0], 0.0);
    EXPECT_DOUBLE_EQ(result[2][0], 1.0);
    EXPECT_DOUBLE_EQ(result[0][1], 0.0);
    EXPECT_DOUBLE_EQ(result[2][1], 1.0);
}

// ============================================================================
// PCA Tests
// ============================================================================

/**
 * @brief Tests basic Principal Component Analysis
 * @test Verifies that PCA produces correct number of components and explained variance ratios sum to approximately 1
 */
TEST(PCATest, Basic) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0},
        {4.0, 8.0},
        {5.0, 10.0}
    };

    auto result = statcpp::pca(data, 2);

    EXPECT_EQ(result.components.size(), 2);  // 2 variables
    EXPECT_EQ(result.explained_variance.size(), 2);
    EXPECT_EQ(result.explained_variance_ratio.size(), 2);

    // Sum of explained variance ratios is close to 1
    double total_ratio = result.explained_variance_ratio[0] + result.explained_variance_ratio[1];
    EXPECT_NEAR(total_ratio, 1.0, 0.1);  // Using larger tolerance due to approximation
}

/**
 * @brief Tests PCA transformation of data to principal component space
 * @test Verifies that PCA transform produces correct dimensions based on number of components
 */
TEST(PCATest, Transform) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0}
    };

    auto pca_result = statcpp::pca(data, 1);
    auto transformed = statcpp::pca_transform(data, pca_result);

    EXPECT_EQ(transformed.size(), 3);
    EXPECT_EQ(transformed[0].size(), 1);
}

// ============================================================================
// NaN Handling (v0.5.0, docs/NAN_POLICY.md)
// ============================================================================

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Columns a = c(1, 2, NA, 4, 5) and b = c(2, 1, 4, 3, 5)
const std::vector<std::vector<double>> kNanData = {{1, 2}, {2, 1}, {kNaN, 4}, {4, 3}, {5, 5}};

}  // namespace

/**
 * @brief Tests correlation_matrix with a NaN variable
 * @test Verifies R's cor(): NaN off the diagonal, exactly 1 on the diagonal
 */
TEST(MultivariateNanTest, CorrelationMatrixDiagonalIsOne) {
    auto r = statcpp::correlation_matrix(kNanData);
    EXPECT_DOUBLE_EQ(r[0][0], 1.0);
    EXPECT_DOUBLE_EQ(r[1][1], 1.0);
    EXPECT_TRUE(std::isnan(r[0][1]));
    EXPECT_TRUE(std::isnan(r[1][0]));
}

/**
 * @brief Tests that standardize follows R's scale()
 * @test Expected values: scale(d) in R 4.4.2; column statistics come from the non-NaN values
 */
TEST(MultivariateNanTest, StandardizeLikeRScale) {
    auto z = statcpp::standardize(kNanData);
    EXPECT_NEAR(z[0][0], -1.0954451150103321, 1e-12);
    EXPECT_NEAR(z[1][0], -0.54772255750516607, 1e-12);
    EXPECT_TRUE(std::isnan(z[2][0]));
    EXPECT_NEAR(z[3][0], 0.54772255750516607, 1e-12);
    EXPECT_NEAR(z[4][0], 1.0954451150103321, 1e-12);
    EXPECT_NEAR(z[2][1], 0.63245553203367588, 1e-12);
    EXPECT_NEAR(z[3][1], 0.0, 1e-12);
}

/**
 * @brief Tests that the eigen and PCA functions reject NaN
 * @test Verifies power_iteration, pca and pca_transform throw (they returned all-NaN results)
 */
TEST(MultivariateNanTest, PcaThrows) {
    std::vector<std::vector<double>> m = {{2.0, kNaN}, {kNaN, 1.0}};
    EXPECT_THROW(statcpp::power_iteration(m), std::invalid_argument);
    EXPECT_THROW(statcpp::pca(kNanData, 1), std::invalid_argument);
    std::vector<std::vector<double>> clean = {{1, 2}, {2, 1}, {3, 4}, {4, 3}, {5, 5}};
    auto model = statcpp::pca(clean, 1);
    EXPECT_THROW(statcpp::pca_transform(kNanData, model), std::invalid_argument);
}
