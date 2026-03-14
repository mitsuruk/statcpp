#include <gtest/gtest.h>
#include "statcpp/linear_regression.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// Simple Linear Regression Tests
// ============================================================================

/**
 * @brief Tests basic simple linear regression with perfect linear relationship.
 * @test Verifies that slope=3.0, intercept=2.0, and R²=1.0 for y = 2 + 3x.
 */
TEST(SimpleLinearRegressionTest, BasicComputation) {
    // y = 2 + 3x (perfect linear relationship)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 8.0, 11.0, 14.0, 17.0};

    auto result = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    EXPECT_NEAR(result.intercept, 2.0, 1e-10);
    EXPECT_NEAR(result.slope, 3.0, 1e-10);
    EXPECT_NEAR(result.r_squared, 1.0, 1e-10);
}

/**
 * @brief Tests simple linear regression with noisy data.
 * @test Verifies that regression parameters are close to true values with small random noise.
 */
TEST(SimpleLinearRegressionTest, WithNoise) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1};

    auto result = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    // Should be close to y = 2x
    EXPECT_NEAR(result.slope, 2.0, 0.1);
    EXPECT_NEAR(result.intercept, 0.0, 0.3);
    EXPECT_GT(result.r_squared, 0.99);
}

/**
 * @brief Tests that statistical tests (t, F) are computed correctly.
 * @test Verifies that t-statistics and p-values are significant for strong relationship.
 */
TEST(SimpleLinearRegressionTest, StatisticalTests) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.1, 7.2, 9.8, 12.3, 14.9, 17.4, 20.1, 22.5, 25.2};

    auto result = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    // t-statistics should be significant
    EXPECT_GT(std::abs(result.slope_t), 2.0);

    // p-values should be small for strong relationship
    EXPECT_LT(result.slope_p, 0.05);

    // F-statistic should be significant
    EXPECT_GT(result.f_statistic, 4.0);
    EXPECT_LT(result.f_p_value, 0.05);
}

/**
 * @brief Tests that adjusted R-squared is computed correctly.
 * @test Verifies that adjusted R² is less than or equal to R².
 */
TEST(SimpleLinearRegressionTest, AdjustedRSquared) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    // Adjusted R² should be slightly less than R² (or equal for perfect fit)
    EXPECT_LE(result.adj_r_squared, result.r_squared);
}

/**
 * @brief Tests that too few observations throw exception.
 * @test Verifies that regression with fewer than 3 observations throws std::invalid_argument.
 */
TEST(SimpleLinearRegressionTest, TooFewObservations) {
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> y = {3.0, 5.0};

    EXPECT_THROW(statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests that unequal length vectors throw exception.
 * @test Verifies that mismatched X and Y lengths throw std::invalid_argument.
 */
TEST(SimpleLinearRegressionTest, UnequalLength) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {3.0, 5.0};

    EXPECT_THROW(statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

/**
 * @brief Tests simple linear regression with constant response (zero variance in y).
 * @test Verifies that an exception is thrown when y has zero variance.
 */
TEST(SimpleLinearRegressionTest, ConstantResponse) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 5.0, 5.0, 5.0, 5.0};  // constant y

    EXPECT_THROW(statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end()),
                 std::invalid_argument);
}

// ============================================================================
// Multiple Linear Regression Tests
// ============================================================================

/**
 * @brief Tests basic multiple linear regression with perfect fit.
 * @test Verifies that coefficients match y = 1 + 2*x1 + 3*x2 with R²=1.0.
 */
TEST(MultipleLinearRegressionTest, BasicComputation) {
    // y = 1 + 2*x1 + 3*x2
    std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0},
        {1.0, 3.0},
        {3.0, 2.0},
        {2.0, 3.0}
    };
    std::vector<double> y;
    for (const auto& row : X) {
        y.push_back(1.0 + 2.0 * row[0] + 3.0 * row[1]);
    }

    auto result = statcpp::multiple_linear_regression(X, y);

    EXPECT_NEAR(result.coefficients[0], 1.0, 1e-8);  // intercept
    EXPECT_NEAR(result.coefficients[1], 2.0, 1e-8);  // x1
    EXPECT_NEAR(result.coefficients[2], 3.0, 1e-8);  // x2
    EXPECT_NEAR(result.r_squared, 1.0, 1e-10);
}

/**
 * @brief Tests multiple regression with noisy data.
 * @test Verifies that multiple regression achieves high R² with realistic noisy data.
 */
TEST(MultipleLinearRegressionTest, WithNoise) {
    // Two independent predictors (not perfectly collinear)
    std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {2.0, 3.0}, {3.0, 2.0}, {4.0, 4.0}, {5.0, 3.0},
        {6.0, 5.0}, {7.0, 4.0}, {8.0, 6.0}, {9.0, 5.0}, {10.0, 7.0}
    };
    std::vector<double> y = {4.1, 10.9, 10.1, 17.0, 16.2, 23.0, 22.1, 28.0, 27.9, 35.1};

    auto result = statcpp::multiple_linear_regression(X, y);

    // Should fit reasonably well
    EXPECT_GT(result.r_squared, 0.99);
    EXPECT_GT(result.f_statistic, 10.0);
}

/**
 * @brief Tests that too few observations throw exception in multiple regression.
 * @test Verifies that insufficient observations throw std::invalid_argument.
 */
TEST(MultipleLinearRegressionTest, TooFewObservations) {
    std::vector<std::vector<double>> X = {{1.0, 2.0}, {2.0, 3.0}};
    std::vector<double> y = {3.0, 5.0};

    EXPECT_THROW(statcpp::multiple_linear_regression(X, y), std::invalid_argument);
}

/**
 * @brief Tests multiple linear regression with constant response (zero variance in y).
 * @test Verifies that an exception is thrown when y has zero variance.
 */
TEST(MultipleLinearRegressionTest, ConstantResponse) {
    std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 1.0}
    };
    std::vector<double> y = {7.0, 7.0, 7.0, 7.0, 7.0};  // constant y

    EXPECT_THROW(statcpp::multiple_linear_regression(X, y), std::invalid_argument);
}

// ============================================================================
// Prediction Tests
// ============================================================================

/**
 * @brief Tests prediction from simple linear regression model.
 * @test Verifies that predictions match expected values from fitted model.
 */
TEST(PredictionTest, SimpleRegression) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 8.0, 11.0, 14.0, 17.0};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    EXPECT_NEAR(statcpp::predict(model, 6.0), 20.0, 1e-10);
    EXPECT_NEAR(statcpp::predict(model, 0.0), 2.0, 1e-10);
}

/**
 * @brief Tests prediction from multiple linear regression model.
 * @test Verifies that predictions match expected values from multiple regression model.
 */
TEST(PredictionTest, MultipleRegression) {
    std::vector<std::vector<double>> X = {
        {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0},
        {3.0, 1.0}, {1.0, 3.0}, {3.0, 2.0}, {2.0, 3.0}
    };
    std::vector<double> y;
    for (const auto& row : X) {
        y.push_back(1.0 + 2.0 * row[0] + 3.0 * row[1]);
    }

    auto model = statcpp::multiple_linear_regression(X, y);

    std::vector<double> new_x = {4.0, 5.0};
    EXPECT_NEAR(statcpp::predict(model, new_x), 1.0 + 2.0*4.0 + 3.0*5.0, 1e-8);
}

// ============================================================================
// Prediction Interval Tests
// ============================================================================

/**
 * @brief Tests basic prediction interval computation.
 * @test Verifies that prediction interval contains the point prediction with positive SE.
 */
TEST(PredictionIntervalTest, BasicComputation) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.1, 7.2, 9.8, 12.3, 14.9, 17.4, 20.1, 22.5, 25.2};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    auto pi = statcpp::prediction_interval_simple(model, x.begin(), x.end(), 5.5, 0.95);

    // Prediction should be between lower and upper bounds
    EXPECT_LT(pi.lower, pi.prediction);
    EXPECT_GT(pi.upper, pi.prediction);

    // Standard error should be positive
    EXPECT_GT(pi.se_prediction, 0.0);
}

/**
 * @brief Tests that confidence interval for mean is narrower than prediction interval.
 * @test Verifies that CI for mean response is narrower than PI for individual prediction.
 */
TEST(PredictionIntervalTest, ConfidenceIntervalMean) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.1, 7.2, 9.8, 12.3, 14.9, 17.4, 20.1, 22.5, 25.2};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    auto ci_mean = statcpp::confidence_interval_mean(model, x.begin(), x.end(), 5.5, 0.95);
    auto pi = statcpp::prediction_interval_simple(model, x.begin(), x.end(), 5.5, 0.95);

    // CI for mean should be narrower than prediction interval
    double ci_width = ci_mean.upper - ci_mean.lower;
    double pi_width = pi.upper - pi.lower;
    EXPECT_LT(ci_width, pi_width);
}

// ============================================================================
// Residual Diagnostics Tests
// ============================================================================

/**
 * @brief Tests basic residual diagnostics computation.
 * @test Verifies that all diagnostic statistics are computed with correct dimensions.
 */
TEST(ResidualDiagnosticsTest, BasicComputation) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.1, 7.2, 9.8, 12.3, 14.9, 17.4, 20.1, 22.5, 25.2};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    auto diag = statcpp::compute_residual_diagnostics(model, x.begin(), x.end(), y.begin(), y.end());

    // Check sizes
    EXPECT_EQ(diag.residuals.size(), 10u);
    EXPECT_EQ(diag.standardized_residuals.size(), 10u);
    EXPECT_EQ(diag.studentized_residuals.size(), 10u);
    EXPECT_EQ(diag.hat_values.size(), 10u);
    EXPECT_EQ(diag.cooks_distance.size(), 10u);

    // Durbin-Watson should be between 0 and 4
    EXPECT_GE(diag.durbin_watson, 0.0);
    EXPECT_LE(diag.durbin_watson, 4.0);
}

/**
 * @brief Tests that hat values are computed correctly.
 * @test Verifies that hat values sum to p and are bounded between 0 and 1.
 */
TEST(ResidualDiagnosticsTest, HatValues) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.5, 5.1, 7.2, 9.8, 12.3, 14.9, 17.4, 20.1, 22.5, 25.2};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    auto diag = statcpp::compute_residual_diagnostics(model, x.begin(), x.end(), y.begin(), y.end());

    // Sum of hat values should equal p (number of parameters)
    double sum_h = 0.0;
    for (double h : diag.hat_values) {
        sum_h += h;
        // Each hat value should be between 0 and 1
        EXPECT_GE(h, 0.0);
        EXPECT_LE(h, 1.0);
    }
    EXPECT_NEAR(sum_h, 2.0, 0.1);  // p = 2 for simple regression
}

// ============================================================================
// VIF Tests
// ============================================================================

/**
 * @brief Tests VIF calculation without perfect multicollinearity.
 * @test Verifies that VIF values are computed and greater than 1 for correlated predictors.
 */
TEST(VIFTest, NoPerfectCollinearity) {
    // Two independent variables with some correlation
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 5.0}, {4.0, 6.0}, {5.0, 8.0},
        {6.0, 9.0}, {7.0, 11.0}, {8.0, 12.0}, {9.0, 14.0}, {10.0, 15.0}
    };

    auto vif = statcpp::compute_vif(X);

    EXPECT_EQ(vif.size(), 2u);
    // VIF should be > 1 due to correlation
    EXPECT_GE(vif[0], 1.0);
    EXPECT_GE(vif[1], 1.0);
}

/**
 * @brief Tests VIF detection of high multicollinearity.
 * @test Verifies that VIF values are very high when predictors are nearly collinear.
 */
TEST(VIFTest, HighCollinearity) {
    // x2 is almost perfectly correlated with x1
    std::vector<std::vector<double>> X = {
        {1.0, 2.01}, {2.0, 4.00}, {3.0, 6.01}, {4.0, 8.00}, {5.0, 10.01},
        {6.0, 12.00}, {7.0, 14.01}, {8.0, 16.00}, {9.0, 18.01}, {10.0, 20.00}
    };

    auto vif = statcpp::compute_vif(X);

    // VIF should be very high (indicating multicollinearity)
    EXPECT_GT(vif[0], 100.0);
    EXPECT_GT(vif[1], 100.0);
}

// ============================================================================
// R-squared Functions Tests
// ============================================================================

/**
 * @brief Tests R-squared calculation with perfect fit.
 * @test Verifies that R² equals 1.0 when predictions exactly match actual values.
 */
TEST(RSquaredTest, PerfectFit) {
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> pred = {1.0, 2.0, 3.0, 4.0, 5.0};

    double r2 = statcpp::r_squared(y.begin(), y.end(), pred.begin(), pred.end());
    EXPECT_NEAR(r2, 1.0, 1e-10);
}

/**
 * @brief Tests R-squared calculation with poor fit.
 * @test Verifies that R² equals 0.0 when predictions are just the mean.
 */
TEST(RSquaredTest, PoorFit) {
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> pred = {3.0, 3.0, 3.0, 3.0, 3.0};  // predicting mean

    double r2 = statcpp::r_squared(y.begin(), y.end(), pred.begin(), pred.end());
    EXPECT_NEAR(r2, 0.0, 1e-10);
}

/**
 * @brief Tests that adjusted R-squared is less than R-squared.
 * @test Verifies that adjusted R² accounts for number of predictors and is smaller than R².
 */
TEST(RSquaredTest, AdjustedRSquared) {
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> pred = {1.1, 2.0, 2.9, 4.1, 5.0, 5.9, 7.1, 8.0, 8.9, 10.1};

    double r2 = statcpp::r_squared(y.begin(), y.end(), pred.begin(), pred.end());
    double adj_r2 = statcpp::adjusted_r_squared(y.begin(), y.end(), pred.begin(), pred.end(), 1);

    EXPECT_LT(adj_r2, r2);
    EXPECT_GT(adj_r2, 0.0);
}
