#include <gtest/gtest.h>
#include "statcpp/glm.hpp"
#include "statcpp/linear_regression.hpp"
#include <cmath>
#include <limits>
#include <vector>

// ============================================================================
// Logistic Regression Tests
// ============================================================================

/**
 * @brief Tests basic logistic regression with overlapping data
 * @test Verifies that logistic regression correctly identifies binomial family and logit link,
 *       and produces positive coefficients for positively associated predictors
 */
TEST(LogisticRegressionTest, BasicLogistic) {
    // Data with some overlap (not perfectly separated)
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::logistic_regression(X, y);

    EXPECT_EQ(result.family, statcpp::distribution_family::binomial);
    EXPECT_EQ(result.link, statcpp::link_function::logit);

    // Coefficient for X should be positive (higher X -> higher probability)
    EXPECT_GT(result.coefficients[1], 0.0);
}

/**
 * @brief Tests logistic regression with gradual probability changes
 * @test Verifies convergence and positive slope coefficient for data with gradual probability transitions
 */
TEST(LogisticRegressionTest, GradualProbability) {
    // More realistic data with gradual change
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0},
        {1.5}, {2.5}, {3.5}, {4.5}, {5.5}, {6.5}, {7.5}, {8.5}, {9.5}, {10.5}
    };
    std::vector<double> y = {
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0
    };

    auto result = statcpp::logistic_regression(X, y);

    EXPECT_TRUE(result.converged);
    // Slope coefficient should be positive
    EXPECT_GT(result.coefficients[1], 0.0);
}

/**
 * @brief Tests logistic regression with multiple predictor variables
 * @test Verifies that logistic regression correctly handles multiple predictors and returns the expected number of coefficients
 */
TEST(LogisticRegressionTest, MultipleVariables) {
    // More data points with variation
    std::vector<std::vector<double>> X = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0},
        {1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0},
        {1.5, 0.5}, {2.5, 0.5}, {3.5, 0.5}, {4.5, 0.5}, {5.5, 0.5}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                             0.0, 0.0, 1.0, 1.0, 1.0};

    auto result = statcpp::logistic_regression(X, y);

    // Should have 3 coefficients (intercept + 2 variables)
    EXPECT_EQ(result.coefficients.size(), 3u);
}

/**
 * @brief Tests probability prediction from logistic regression model
 * @test Verifies that predicted probabilities are within [0,1] range and increase with predictor values
 */
TEST(LogisticRegressionTest, PredictProbability) {
    std::vector<std::vector<double>> X = {
        {0.0}, {1.0}, {2.0}, {3.0}, {4.0},
        {5.0}, {6.0}, {7.0}, {8.0}, {9.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::logistic_regression(X, y);

    // Predict at midpoint
    double prob_mid = statcpp::predict_probability(result, {5.0});
    EXPECT_GE(prob_mid, 0.0);
    EXPECT_LE(prob_mid, 1.0);

    // Probability should increase with X
    double prob_low = statcpp::predict_probability(result, {0.0});
    double prob_high = statcpp::predict_probability(result, {10.0});
    EXPECT_LT(prob_low, prob_high);
}

/**
 * @brief Tests odds ratio calculation from logistic regression
 * @test Verifies that odds ratios are computed correctly and show positive association for positively correlated predictors
 */
TEST(LogisticRegressionTest, OddsRatios) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::logistic_regression(X, y);
    auto or_values = statcpp::odds_ratios(result);

    // Should have p-1 odds ratios (excluding intercept)
    EXPECT_EQ(or_values.size(), 1u);

    // OR should be > 1 (positive association)
    EXPECT_GT(or_values[0], 1.0);
}

/**
 * @brief Tests confidence interval calculation for odds ratios
 * @test Verifies that odds ratio confidence intervals are properly bounded around the point estimate
 */
TEST(LogisticRegressionTest, OddsRatiosCI) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::logistic_regression(X, y);
    auto ci = statcpp::odds_ratios_ci(result, 0.95);

    EXPECT_EQ(ci.size(), 1u);
    // Lower < OR < Upper
    auto or_values = statcpp::odds_ratios(result);
    EXPECT_LT(ci[0].first, or_values[0]);
    EXPECT_GT(ci[0].second, or_values[0]);
}

/**
 * @brief Tests error handling for invalid response variable values
 * @test Verifies that logistic regression throws exception for non-binary response values
 */
TEST(LogisticRegressionTest, InvalidY) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {0.0, 1.5, 1.0};  // 1.5 is invalid

    EXPECT_THROW(statcpp::logistic_regression(X, y), std::invalid_argument);
}

// ============================================================================
// Poisson Regression Tests
// ============================================================================

/**
 * @brief Tests basic Poisson regression computation with count data
 * @test Verifies convergence, correct family/link specification, and positive coefficients for increasing count patterns
 */
TEST(PoissonRegressionTest, BasicComputation) {
    // Count data
    std::vector<std::vector<double>> X = {
        {0.0}, {1.0}, {2.0}, {3.0}, {4.0},
        {5.0}, {6.0}, {7.0}, {8.0}, {9.0}
    };
    std::vector<double> y = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0};

    auto result = statcpp::poisson_regression(X, y);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.family, statcpp::distribution_family::poisson);
    EXPECT_EQ(result.link, statcpp::link_function::log);

    // Coefficient should be positive (increasing counts)
    EXPECT_GT(result.coefficients[1], 0.0);
}

/**
 * @brief Tests count prediction from Poisson regression model
 * @test Verifies that predicted counts are positive values
 */
TEST(PoissonRegressionTest, PredictCount) {
    std::vector<std::vector<double>> X = {
        {0.0}, {1.0}, {2.0}, {3.0}, {4.0}
    };
    std::vector<double> y = {1.0, 3.0, 7.0, 15.0, 31.0};

    auto result = statcpp::poisson_regression(X, y);

    double count = statcpp::predict_count(result, {2.5});

    // Predicted count should be positive
    EXPECT_GT(count, 0.0);
}

/**
 * @brief Tests incidence rate ratio calculation from Poisson regression
 * @test Verifies that incidence rate ratios are greater than 1 for predictors associated with increasing rates
 */
TEST(PoissonRegressionTest, IncidenceRateRatios) {
    std::vector<std::vector<double>> X = {
        {0.0}, {1.0}, {2.0}, {3.0}, {4.0},
        {5.0}, {6.0}, {7.0}, {8.0}, {9.0}
    };
    std::vector<double> y = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0};

    auto result = statcpp::poisson_regression(X, y);
    auto irr = statcpp::incidence_rate_ratios(result);

    // IRR should be > 1 (rate increases with X)
    EXPECT_EQ(irr.size(), 1u);
    EXPECT_GT(irr[0], 1.0);
}

/**
 * @brief Tests error handling for invalid count values
 * @test Verifies that Poisson regression throws exception for negative count values
 */
TEST(PoissonRegressionTest, InvalidY) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, -1.0, 2.0};  // -1 is invalid

    EXPECT_THROW(statcpp::poisson_regression(X, y), std::invalid_argument);
}

// ============================================================================
// GLM Fit Tests
// ============================================================================

/**
 * @brief Tests GLM with Gaussian family and identity link (equivalent to linear regression)
 * @test Verifies that GLM with Gaussian/identity produces results consistent with linear regression
 */
TEST(GLMFitTest, GaussianIdentity) {
    // This should behave like linear regression
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};  // y = 1 + 2x

    auto result = statcpp::glm_fit(X, y,
                                    statcpp::distribution_family::gaussian,
                                    statcpp::link_function::identity);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.coefficients[0], 1.0, 0.1);  // intercept
    EXPECT_NEAR(result.coefficients[1], 2.0, 0.1);  // slope
}

/**
 * @brief Tests GLM convergence behavior
 * @test Verifies that GLM with binomial family and logit link converges within reasonable iteration count
 */
TEST(GLMFitTest, Convergence) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::glm_fit(X, y,
                                    statcpp::distribution_family::binomial,
                                    statcpp::link_function::logit);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 100u);
}

// ============================================================================
// GLM Diagnostics Tests
// ============================================================================

/**
 * @brief Tests computation of different residual types for GLM
 * @test Verifies that all residual types (response, Pearson, deviance, working) are computed with correct dimensions
 */
TEST(GLMDiagnosticsTest, ResidualTypes) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);
    auto residuals = statcpp::compute_glm_residuals(model, X, y);

    EXPECT_EQ(residuals.response.size(), 10u);
    EXPECT_EQ(residuals.pearson.size(), 10u);
    EXPECT_EQ(residuals.deviance.size(), 10u);
    EXPECT_EQ(residuals.working.size(), 10u);
}

/**
 * @brief Tests overdispersion detection for Poisson models
 * @test Verifies that overdispersion parameter is positive for Poisson regression data
 */
TEST(GLMDiagnosticsTest, OverdispersionTest) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    // Poisson-like data
    std::vector<double> y = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0};

    auto model = statcpp::poisson_regression(X, y);
    double dispersion = statcpp::overdispersion_test(model, X, y);

    // Dispersion should be positive
    EXPECT_GT(dispersion, 0.0);
}

/**
 * @brief Tests McFadden's pseudo R-squared calculation
 * @test Verifies that McFadden's pseudo R-squared is bounded between 0 and 1
 */
TEST(GLMDiagnosticsTest, PseudoRSquaredMcFadden) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);
    double r2 = statcpp::pseudo_r_squared_mcfadden(model);

    // Pseudo R² should be between 0 and 1
    EXPECT_GE(r2, 0.0);
    EXPECT_LE(r2, 1.0);
}

/**
 * @brief Tests Nagelkerke's pseudo R-squared calculation
 * @test Verifies that Nagelkerke's pseudo R-squared is bounded between 0 and 1
 */
TEST(GLMDiagnosticsTest, PseudoRSquaredNagelkerke) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);
    double r2 = statcpp::pseudo_r_squared_nagelkerke(model, y, X.size());

    // Nagelkerke R² should be between 0 and 1
    EXPECT_GE(r2, 0.0);
    EXPECT_LE(r2, 1.0);
}

// ============================================================================
// Model Statistics Tests
// ============================================================================

/**
 * @brief Tests AIC and BIC computation for GLM models
 * @test Verifies that AIC and BIC values are finite for logistic regression model
 */
TEST(GLMStatisticsTest, AICAndBIC) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);

    // AIC and BIC should be finite
    EXPECT_TRUE(std::isfinite(model.aic));
    EXPECT_TRUE(std::isfinite(model.bic));

    // BIC penalizes more for n > e^2 ≈ 7.4
    // With n=10, BIC > AIC typically
}

/**
 * @brief Tests deviance calculations for GLM models
 * @test Verifies that residual deviance is less than null deviance and both are non-negative
 */
TEST(GLMStatisticsTest, Deviance) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);

    // Residual deviance should be less than null deviance
    EXPECT_LT(model.residual_deviance, model.null_deviance);

    // Both should be non-negative
    EXPECT_GE(model.residual_deviance, 0.0);
    EXPECT_GE(model.null_deviance, 0.0);
}

/**
 * @brief Tests degrees of freedom calculation for GLM models
 * @test Verifies that null and residual degrees of freedom are correctly computed based on sample size and parameters
 */
TEST(GLMStatisticsTest, DegreesOfFreedom) {
    // Use non-collinear predictors
    std::vector<std::vector<double>> X = {
        {1.0, 5.0}, {2.0, 3.0}, {3.0, 7.0}, {4.0, 2.0}, {5.0, 8.0},
        {6.0, 4.0}, {7.0, 9.0}, {8.0, 1.0}, {9.0, 6.0}, {10.0, 3.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    auto model = statcpp::logistic_regression(X, y);

    // df_null = n - 1 = 9
    EXPECT_NEAR(model.df_null, 9.0, 1e-10);

    // df_residual = n - p = 10 - 3 = 7 (3 = intercept + 2 predictors)
    EXPECT_NEAR(model.df_residual, 7.0, 1e-10);
}

// ============================================================================
// GLM AIC/BIC Parameter Count Tests
// ============================================================================

/**
 * @brief Tests that Gaussian GLM AIC/BIC includes sigma^2 as an estimated parameter
 * @test Verifies that k = p_full + 1 for Gaussian family (intercept + slope + sigma^2 = 3)
 */
TEST(GLMFitTest, GaussianAICIncludesSigma) {
    // y ≈ 1 + 2x with noise (not a perfect fit to avoid sigma2=0 and log(0)=-inf)
    // n=10, p_full=2 (intercept + 1 predictor), k should be 3 (includes sigma^2)
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {3.1, 4.8, 7.2, 8.9, 11.1, 12.8, 15.3, 16.9, 19.0, 21.2};

    auto result = statcpp::glm_fit(X, y,
                                    statcpp::distribution_family::gaussian,
                                    statcpp::link_function::identity);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::isfinite(result.log_likelihood));

    // For Gaussian: AIC = -2*LL + 2*k, k = p_full + 1 = 3 (includes sigma^2)
    double expected_aic = -2.0 * result.log_likelihood + 2.0 * 3.0;
    EXPECT_NEAR(result.aic, expected_aic, 1e-10);

    // BIC = -2*LL + k*log(n), k=3, n=10
    double expected_bic = -2.0 * result.log_likelihood + 3.0 * std::log(10.0);
    EXPECT_NEAR(result.bic, expected_bic, 1e-10);
}

/**
 * @brief Tests that non-Gaussian GLM AIC does NOT add sigma^2 parameter
 * @test Verifies that k = p_full for Binomial family (no sigma^2 parameter)
 */
TEST(GLMFitTest, BinomialAICNoSigma) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};

    auto result = statcpp::glm_fit(X, y,
                                    statcpp::distribution_family::binomial,
                                    statcpp::link_function::logit);

    // Binomial: k = p_full = 2 (intercept + 1 predictor), no sigma^2
    double expected_aic = -2.0 * result.log_likelihood + 2.0 * 2.0;
    EXPECT_NEAR(result.aic, expected_aic, 1e-10);
}

/**
 * @brief Regression guard for GLM coefficient SE dispersion (review #9).
 * @test For the gaussian family the coefficient covariance is phi * (X^T W X)^{-1}
 *       with phi = sigma^2. The SEs must match OLS (multiple_linear_regression).
 *       The pre-fix code omitted phi, making the GLM SEs a factor of sigma too small.
 */
TEST(GLMFitTest, GaussianCoefficientSeMatchesOLS) {
    // Non-perfect fit so sigma^2 > 0 (otherwise the bug is invisible).
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}
    };
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.3, 11.9, 14.2, 15.7};

    auto glm = statcpp::glm_fit(X, y, statcpp::distribution_family::gaussian,
                                statcpp::link_function::identity);
    auto ols = statcpp::multiple_linear_regression(X, y);

    ASSERT_EQ(glm.coefficient_se.size(), ols.coefficient_se.size());
    for (std::size_t j = 0; j < glm.coefficient_se.size(); ++j) {
        EXPECT_NEAR(glm.coefficient_se[j], ols.coefficient_se[j], 1e-6);
    }
}

// ============================================================================
// NaN Handling (v0.5.0, docs/NAN_POLICY.md)
// ============================================================================

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

/**
 * @brief Tests that logistic_regression and poisson_regression drop incomplete observations
 * @test Expected values: glm(family = binomial / poisson) in R 4.4.2 with the default na.omit
 */
TEST(GlmNanTest, RegressionsMatchR) {
    std::vector<std::vector<double>> X1 = {{0.5}, {1.2}, {kNaN}, {2.8}, {3.1}, {4.0}, {4.4}, {5.6}, {6.1}, {7.3}};
    std::vector<double> yb = {0, 0, 1, 0, 1, 0, 1, 1, kNaN, 1};
    auto g = statcpp::logistic_regression(X1, yb);
    EXPECT_TRUE(g.converged);
    EXPECT_NEAR(g.coefficients[0], -5.2381807357127519, 1e-6);
    EXPECT_NEAR(g.coefficients[1], 1.470074972983737, 1e-6);
    EXPECT_NEAR(g.residual_deviance, 5.5763158528398318, 1e-6);
    EXPECT_DOUBLE_EQ(g.df_residual, 6.0);

    std::vector<std::vector<double>> X2 = {{0.5}, {1.0}, {1.5}, {kNaN}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}, {5.0}};
    std::vector<double> yp = {1, 0, 2, 3, kNaN, 5, 4, 7, 9, 8};
    auto p = statcpp::poisson_regression(X2, yp);
    EXPECT_TRUE(p.converged);
    EXPECT_NEAR(p.coefficients[0], -0.30024671840961908, 1e-6);
    EXPECT_NEAR(p.coefficients[1], 0.52352659582539873, 1e-6);
}

/**
 * @brief Tests that glm_fit drops incomplete observations and the residual functions follow
 * @test Verifies glm_fit, compute_glm_residuals and overdispersion_test equal their results on the complete rows
 */
TEST(GlmNanTest, FitAndResidualsDropIncompleteRows) {
    std::vector<std::vector<double>> X = {{1}, {2}, {kNaN}, {4}, {5}, {6}, {7}, {8}};
    std::vector<double> y = {1, 3, 2, 4, kNaN, 7, 6, 9};
    std::vector<std::vector<double>> Xc = {{1}, {2}, {4}, {6}, {7}, {8}};
    std::vector<double> yc = {1, 3, 4, 7, 6, 9};

    auto a = statcpp::glm_fit(X, y);
    auto b = statcpp::glm_fit(Xc, yc);
    EXPECT_DOUBLE_EQ(a.coefficients[0], b.coefficients[0]);
    EXPECT_DOUBLE_EQ(a.coefficients[1], b.coefficients[1]);
    EXPECT_DOUBLE_EQ(a.residual_deviance, b.residual_deviance);

    auto p = statcpp::poisson_regression(Xc, yc);
    auto ra = statcpp::compute_glm_residuals(p, X, y);
    auto rb = statcpp::compute_glm_residuals(p, Xc, yc);
    ASSERT_EQ(ra.deviance.size(), 6u);
    EXPECT_EQ(ra.deviance, rb.deviance);
    EXPECT_EQ(ra.pearson, rb.pearson);
    EXPECT_DOUBLE_EQ(statcpp::overdispersion_test(p, X, y), statcpp::overdispersion_test(p, Xc, yc));
}

/**
 * @brief Tests that pseudo_r_squared_nagelkerke drops NaN responses and reduces n
 * @test Verifies the result equals the result for the response without NaN
 */
TEST(GlmNanTest, NagelkerkeDropsNaN) {
    std::vector<std::vector<double>> X = {{0.5}, {1.0}, {1.5}, {2.0}, {2.5}, {3.0}, {3.5}, {4.0}};
    std::vector<double> y = {1, 0, 2, 3, 4, 5, 7, 9};
    auto model = statcpp::poisson_regression(X, y);
    std::vector<double> y_nan = {1, 0, 2, kNaN, 3, 4, 5, 7, 9};
    EXPECT_DOUBLE_EQ(statcpp::pseudo_r_squared_nagelkerke(model, y_nan, 9),
                     statcpp::pseudo_r_squared_nagelkerke(model, y, 8));
}

/**
 * @brief Tests that the GLM fitting functions reject a NaN tolerance
 * @test Verifies policy section 5 for range-checked parameters
 */
TEST(GlmNanTest, NaNToleranceThrows) {
    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {0, 0, 1, 0, 1};
    EXPECT_THROW(statcpp::glm_fit(X, y, statcpp::distribution_family::gaussian,
                                  statcpp::link_function::identity, 100, kNaN),
                 std::invalid_argument);
    EXPECT_THROW(statcpp::logistic_regression(X, y, 100, kNaN), std::invalid_argument);
    EXPECT_THROW(statcpp::poisson_regression(X, y, 100, kNaN), std::invalid_argument);
}
