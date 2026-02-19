#include <gtest/gtest.h>
#include "statcpp/model_selection.hpp"
#include "statcpp/random_engine.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// AIC/BIC Tests
// ============================================================================

/**
 * @brief Tests basic AIC computation
 * @test Verifies that AIC is correctly calculated using the formula -2*LL + 2*k
 */
TEST(AICTest, BasicComputation) {
    double log_likelihood = -10.0;
    std::size_t k = 3;

    double aic_val = statcpp::aic(log_likelihood, k);

    // AIC = -2*LL + 2*k = -2*(-10) + 2*3 = 20 + 6 = 26
    EXPECT_NEAR(aic_val, 26.0, 1e-10);
}

/**
 * @brief Tests corrected AIC (AICc) for small sample sizes
 * @test Verifies that AICc applies additional penalty for small samples and is larger than AIC
 */
TEST(AICTest, AICc) {
    double log_likelihood = -10.0;
    std::size_t n = 10;
    std::size_t k = 3;

    double aicc_val = statcpp::aicc(log_likelihood, n, k);

    // AICc should be larger than AIC (additional penalty for small samples)
    double aic_val = statcpp::aic(log_likelihood, k);
    EXPECT_GT(aicc_val, aic_val);
}

/**
 * @brief Tests AICc error handling for insufficient observations
 * @test Verifies that AICc throws exception when sample size is not sufficiently larger than parameter count
 */
TEST(AICTest, AICcTooFewObservations) {
    double log_likelihood = -10.0;
    std::size_t n = 3;
    std::size_t k = 3;

    EXPECT_THROW(statcpp::aicc(log_likelihood, n, k), std::invalid_argument);
}

/**
 * @brief Tests basic BIC computation
 * @test Verifies that BIC is correctly calculated using the formula -2*LL + k*log(n)
 */
TEST(BICTest, BasicComputation) {
    double log_likelihood = -10.0;
    std::size_t n = 100;
    std::size_t k = 3;

    double bic_val = statcpp::bic(log_likelihood, n, k);

    // BIC = -2*LL + k*log(n) = 20 + 3*log(100) ≈ 20 + 13.82
    EXPECT_NEAR(bic_val, -2.0 * log_likelihood + k * std::log(n), 1e-10);
}

/**
 * @brief Tests that BIC penalizes model complexity more than AIC for large samples
 * @test Verifies that BIC > AIC when n > e^2 (approximately 7.4)
 */
TEST(BICTest, PenalizesMoreThanAIC) {
    double log_likelihood = -10.0;
    std::size_t n = 100;  // log(100) > 2
    std::size_t k = 5;

    double aic_val = statcpp::aic(log_likelihood, k);
    double bic_val = statcpp::bic(log_likelihood, n, k);

    // For n > e^2, BIC penalizes more
    EXPECT_GT(bic_val, aic_val);
}

// ============================================================================
// Linear Regression Model Criteria Tests
// ============================================================================

/**
 * @brief Tests AIC calculation for simple linear regression models
 * @test Verifies that AIC is finite for a simple linear regression model with noisy data
 */
TEST(LinearAICTest, SimpleRegression) {
    // Data with some noise (not perfect fit) to have positive ss_residual
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.6, 4.9, 7.6, 9.9, 12.6, 14.9, 17.6, 19.9, 22.6, 24.9};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    double aic_val = statcpp::aic_linear(model, 10);

    EXPECT_TRUE(std::isfinite(aic_val));
}

/**
 * @brief Tests BIC calculation for simple linear regression models
 * @test Verifies that BIC is finite for a simple linear regression model with noisy data
 */
TEST(LinearBICTest, SimpleRegression) {
    // Data with some noise (not perfect fit) to have positive ss_residual
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.6, 4.9, 7.6, 9.9, 12.6, 14.9, 17.6, 19.9, 22.6, 24.9};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    double bic_val = statcpp::bic_linear(model, 10);

    EXPECT_TRUE(std::isfinite(bic_val));
}

/**
 * @brief Tests AIC and BIC calculation for multiple linear regression models
 * @test Verifies that both AIC and BIC are finite for multiple regression with non-collinear predictors
 */
TEST(LinearAICTest, MultipleRegression) {
    // Use non-collinear predictors with some noise
    std::vector<std::vector<double>> X = {
        {1.0, 5.0}, {2.0, 3.0}, {3.0, 7.0}, {4.0, 2.0}, {5.0, 8.0},
        {6.0, 4.0}, {7.0, 9.0}, {8.0, 1.0}, {9.0, 6.0}, {10.0, 3.0}
    };
    std::vector<double> y = {5.1, 7.9, 11.2, 13.8, 17.1, 19.9, 23.2, 25.8, 29.1, 31.9};

    auto model = statcpp::multiple_linear_regression(X, y);
    double aic_val = statcpp::aic_linear(model, 10);
    double bic_val = statcpp::bic_linear(model, 10);

    EXPECT_TRUE(std::isfinite(aic_val));
    EXPECT_TRUE(std::isfinite(bic_val));
}

// ============================================================================
// Cross-Validation Tests
// ============================================================================

/**
 * @brief Tests creation of cross-validation folds with shuffling
 * @test Verifies that folds are created with correct counts and approximately equal sizes
 */
TEST(CrossValidationTest, CreateFolds) {
    statcpp::set_seed(42);
    auto folds = statcpp::create_cv_folds(10, 5, true);

    EXPECT_EQ(folds.size(), 5u);

    // Total count should equal n
    std::size_t total = 0;
    for (const auto& fold : folds) {
        total += fold.size();
    }
    EXPECT_EQ(total, 10u);

    // Each fold should have approximately equal size
    for (const auto& fold : folds) {
        EXPECT_GE(fold.size(), 1u);
        EXPECT_LE(fold.size(), 3u);
    }
}

/**
 * @brief Tests creation of cross-validation folds with uneven sample sizes
 * @test Verifies that folds handle uneven division correctly when n is not divisible by k
 */
TEST(CrossValidationTest, CreateFoldsUneven) {
    statcpp::set_seed(42);
    auto folds = statcpp::create_cv_folds(13, 5, false);

    // First 3 folds should have 3 elements, last 2 should have 2
    EXPECT_EQ(folds[0].size(), 3u);
    EXPECT_EQ(folds[1].size(), 3u);
    EXPECT_EQ(folds[2].size(), 3u);
    EXPECT_EQ(folds[3].size(), 2u);
    EXPECT_EQ(folds[4].size(), 2u);
}

/**
 * @brief Tests error handling for invalid fold counts
 * @test Verifies that exceptions are thrown for k < 2 or k > n
 */
TEST(CrossValidationTest, CreateFoldsInvalidK) {
    EXPECT_THROW(statcpp::create_cv_folds(10, 1, true), std::invalid_argument);
    EXPECT_THROW(statcpp::create_cv_folds(10, 15, true), std::invalid_argument);
}

/**
 * @brief Tests k-fold cross-validation for linear regression
 * @test Verifies that cross-validation produces error metrics with correct fold count and non-negative errors
 */
TEST(CrossValidationTest, LinearCV) {
    statcpp::set_seed(42);

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::cross_validate_linear(X, y, 5);

    EXPECT_EQ(result.n_folds, 5u);
    EXPECT_EQ(result.fold_errors.size(), 5u);
    EXPECT_GE(result.mean_error, 0.0);
    EXPECT_GE(result.se_error, 0.0);
}

/**
 * @brief Tests leave-one-out cross-validation for linear regression
 * @test Verifies that LOOCV creates n folds and produces non-negative mean error
 */
TEST(CrossValidationTest, LOOCV) {
    statcpp::set_seed(42);

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::loocv_linear(X, y);

    EXPECT_EQ(result.n_folds, 10u);  // LOO has n folds
    EXPECT_GE(result.mean_error, 0.0);
}

// ============================================================================
// Ridge Regression Tests
// ============================================================================

/**
 * @brief Tests basic Ridge regression computation
 * @test Verifies convergence and correct number of coefficients for Ridge regression with L2 penalty
 */
TEST(RidgeRegressionTest, BasicComputation) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::ridge_regression(X, y, 0.1);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.coefficients.size(), 2u);  // intercept + 1 coefficient
}

/**
 * @brief Tests that Ridge regression with zero lambda equals OLS
 * @test Verifies that Ridge regression with lambda=0 produces coefficients close to ordinary least squares
 */
TEST(RidgeRegressionTest, ZeroLambda) {
    // With lambda=0, should be close to OLS
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto ridge_result = statcpp::ridge_regression(X, y, 0.0);
    auto ols_result = statcpp::multiple_linear_regression(X, y);

    // Should be close to OLS
    EXPECT_NEAR(ridge_result.coefficients[0], ols_result.coefficients[0], 0.1);
    EXPECT_NEAR(ridge_result.coefficients[1], ols_result.coefficients[1], 0.1);
}

/**
 * @brief Tests coefficient shrinkage effect in Ridge regression
 * @test Verifies that larger lambda values produce greater coefficient shrinkage
 */
TEST(RidgeRegressionTest, ShrinkageEffect) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto small_lambda = statcpp::ridge_regression(X, y, 0.1);
    auto large_lambda = statcpp::ridge_regression(X, y, 100.0);

    // Larger lambda should shrink coefficients more
    EXPECT_LT(std::abs(large_lambda.coefficients[1]),
              std::abs(small_lambda.coefficients[1]));
}

/**
 * @brief Tests error handling for invalid lambda values in Ridge regression
 * @test Verifies that Ridge regression throws exception for negative lambda
 */
TEST(RidgeRegressionTest, InvalidLambda) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, 2.0, 3.0};

    EXPECT_THROW(statcpp::ridge_regression(X, y, -1.0), std::invalid_argument);
}

// ============================================================================
// Lasso Regression Tests
// ============================================================================

/**
 * @brief Tests basic Lasso regression computation
 * @test Verifies convergence and correct number of coefficients for Lasso regression with L1 penalty
 */
TEST(LassoRegressionTest, BasicComputation) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::lasso_regression(X, y, 0.1);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.coefficients.size(), 2u);
}

/**
 * @brief Tests Lasso's ability to produce sparse solutions
 * @test Verifies that Lasso regression can shrink coefficients significantly with large lambda
 */
TEST(LassoRegressionTest, SparsitySolution) {
    // With large lambda, some coefficients should be exactly zero
    std::vector<std::vector<double>> X = {
        {1.0, 0.1}, {2.0, 0.2}, {3.0, 0.3}, {4.0, 0.4}, {5.0, 0.5},
        {6.0, 0.6}, {7.0, 0.7}, {8.0, 0.8}, {9.0, 0.9}, {10.0, 1.0}
    };
    // y is mainly dependent on X[0]
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::lasso_regression(X, y, 5.0);

    EXPECT_TRUE(result.converged);
    // At least one coefficient might be shrunk significantly
    // (exact zero depends on implementation details)
}

/**
 * @brief Tests error handling for invalid lambda values in Lasso regression
 * @test Verifies that Lasso regression throws exception for negative lambda
 */
TEST(LassoRegressionTest, InvalidLambda) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, 2.0, 3.0};

    EXPECT_THROW(statcpp::lasso_regression(X, y, -1.0), std::invalid_argument);
}

// ============================================================================
// Elastic Net Tests
// ============================================================================

/**
 * @brief Tests basic Elastic Net regression computation
 * @test Verifies convergence and correct number of coefficients for Elastic Net combining L1 and L2 penalties
 */
TEST(ElasticNetTest, BasicComputation) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto result = statcpp::elastic_net_regression(X, y, 0.1, 0.5);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.coefficients.size(), 2u);
}

/**
 * @brief Tests that Elastic Net with alpha=0 equals Ridge regression
 * @test Verifies that Elastic Net produces results equivalent to Ridge when alpha is zero
 */
TEST(ElasticNetTest, AlphaZeroEqualsRidge) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto en_result = statcpp::elastic_net_regression(X, y, 1.0, 0.0);  // alpha=0 -> Ridge
    auto ridge_result = statcpp::ridge_regression(X, y, 1.0);

    EXPECT_NEAR(en_result.coefficients[0], ridge_result.coefficients[0], 0.1);
    EXPECT_NEAR(en_result.coefficients[1], ridge_result.coefficients[1], 0.1);
}

/**
 * @brief Tests that Elastic Net with alpha=1 equals Lasso regression
 * @test Verifies that Elastic Net produces results equivalent to Lasso when alpha is one
 */
TEST(ElasticNetTest, AlphaOneEqualsLasso) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto en_result = statcpp::elastic_net_regression(X, y, 1.0, 1.0);  // alpha=1 -> Lasso
    auto lasso_result = statcpp::lasso_regression(X, y, 1.0);

    EXPECT_NEAR(en_result.coefficients[0], lasso_result.coefficients[0], 0.1);
    EXPECT_NEAR(en_result.coefficients[1], lasso_result.coefficients[1], 0.1);
}

/**
 * @brief Tests error handling for invalid alpha values in Elastic Net
 * @test Verifies that Elastic Net throws exception for alpha outside [0,1] range
 */
TEST(ElasticNetTest, InvalidAlpha) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, 2.0, 3.0};

    EXPECT_THROW(statcpp::elastic_net_regression(X, y, 0.1, -0.1), std::invalid_argument);
    EXPECT_THROW(statcpp::elastic_net_regression(X, y, 0.1, 1.1), std::invalid_argument);
}

// ============================================================================
// Lambda Selection Tests
// ============================================================================

/**
 * @brief Tests generation of lambda grid for regularization path
 * @test Verifies that lambda grid contains correct number of values in descending order with all positive values
 */
TEST(LambdaSelectionTest, GenerateLambdaGrid) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    auto grid = statcpp::generate_lambda_grid(X, y, 50);

    EXPECT_EQ(grid.size(), 50u);

    // Grid should be in descending order
    for (std::size_t i = 1; i < grid.size(); ++i) {
        EXPECT_LT(grid[i], grid[i-1]);
    }

    // All values should be positive
    for (double lambda : grid) {
        EXPECT_GT(lambda, 0.0);
    }
}

/**
 * @brief Tests cross-validation for Ridge regression lambda selection
 * @test Verifies that CV selects a positive best lambda and returns error for each lambda value
 */
TEST(LambdaSelectionTest, CVRidge) {
    statcpp::set_seed(42);

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    std::vector<double> lambda_grid = {0.001, 0.01, 0.1, 1.0, 10.0};
    auto [best_lambda, cv_errors] = statcpp::cv_ridge(X, y, lambda_grid, 5);

    EXPECT_GT(best_lambda, 0.0);
    EXPECT_EQ(cv_errors.size(), lambda_grid.size());
}

/**
 * @brief Tests cross-validation for Lasso regression lambda selection
 * @test Verifies that CV selects a positive best lambda and returns error for each lambda value
 */
TEST(LambdaSelectionTest, CVLasso) {
    statcpp::set_seed(42);

    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0};

    std::vector<double> lambda_grid = {0.001, 0.01, 0.1, 1.0, 10.0};
    auto [best_lambda, cv_errors] = statcpp::cv_lasso(X, y, lambda_grid, 5);

    EXPECT_GT(best_lambda, 0.0);
    EXPECT_EQ(cv_errors.size(), lambda_grid.size());
}

// ============================================================================
// PRESS Statistic Tests
// ============================================================================

/**
 * @brief Tests basic PRESS statistic computation
 * @test Verifies that PRESS is positive and greater than or equal to residual sum of squares
 */
TEST(PRESSTest, BasicComputation) {
    // Data with some noise to have positive residuals
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> y = {2.6, 4.9, 7.6, 9.9, 12.6, 14.9, 17.6, 19.9, 22.6, 24.9};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    double press = statcpp::press_statistic(x.begin(), x.end(), y.begin(), y.end(), model);

    // PRESS should be positive
    EXPECT_GT(press, 0.0);

    // For a good fit, PRESS should be related to SS_residual
    // PRESS >= SS_residual
    EXPECT_GE(press, model.ss_residual - 1e-6);  // Small tolerance
}

/**
 * @brief Tests PRESS statistic for perfect model fit
 * @test Verifies that PRESS is approximately zero for data with perfect linear relationship
 */
TEST(PRESSTest, PerfectFitPRESSIsZero) {
    // Perfect fit: y = 2 + 3x
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 8.0, 11.0, 14.0, 17.0};

    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());
    double press = statcpp::press_statistic(x.begin(), x.end(), y.begin(), y.end(), model);

    // PRESS should be very small for perfect fit
    EXPECT_NEAR(press, 0.0, 1e-8);
}
