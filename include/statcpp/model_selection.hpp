/**
 * @file model_selection.hpp
 * @brief Model selection functions
 *
 * Provides model selection and evaluation metrics including AIC, BIC,
 * cross-validation, and regularized regression.
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/linear_regression.hpp"
#include "statcpp/random_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Model Selection Criteria
// ============================================================================

/**
 * @brief Calculate AIC (Akaike Information Criterion)
 *
 * Calculates the Akaike Information Criterion. Smaller values indicate better models.
 *
 * @param log_likelihood Log-likelihood
 * @param k Number of parameters
 * @return AIC value
 */
inline double aic(double log_likelihood, std::size_t k)
{
    return -2.0 * log_likelihood + 2.0 * static_cast<double>(k);
}

/**
 * @brief Calculate AIC from simple regression model
 *
 * Calculates log-likelihood from simple regression result and computes AIC.
 *
 * @param model Simple regression result
 * @param n Sample size
 * @return AIC value
 */
inline double aic_linear(const simple_regression_result& model, std::size_t n)
{
    // sigma^2 = SS_res / n (MLE version)
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    // Log-likelihood: -n/2 * (log(2*pi) + log(sigma^2) + 1)
    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    return aic(ll, 3);  // k = 2 (coefficients) + 1 (sigma^2)
}

/**
 * @brief Calculate AIC from multiple regression model
 *
 * Calculates log-likelihood from multiple regression result and computes AIC.
 *
 * @param model Multiple regression result
 * @param n Sample size
 * @return AIC value
 */
inline double aic_linear(const multiple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    std::size_t k = model.coefficients.size() + 1;  // coefficients + sigma^2
    return aic(ll, k);
}

/**
 * @brief Calculate AICc (corrected AIC)
 *
 * Calculates the corrected AIC for small samples.
 * Used when sample size is small relative to number of parameters.
 *
 * @param log_likelihood Log-likelihood
 * @param n Sample size
 * @param k Number of parameters
 * @return AICc value
 * @throws std::invalid_argument If n <= k + 1
 */
inline double aicc(double log_likelihood, std::size_t n, std::size_t k)
{
    double n_d = static_cast<double>(n);
    double k_d = static_cast<double>(k);

    if (n_d <= k_d + 1.0) {
        throw std::invalid_argument("statcpp::aicc: n must be greater than k + 1");
    }

    return aic(log_likelihood, k) + (2.0 * k_d * (k_d + 1.0)) / (n_d - k_d - 1.0);
}

/**
 * @brief Calculate BIC (Bayesian Information Criterion)
 *
 * Calculates the Bayesian Information Criterion (Schwarz criterion).
 * Penalizes complex models more heavily than AIC.
 *
 * @param log_likelihood Log-likelihood
 * @param n Sample size
 * @param k Number of parameters
 * @return BIC value
 */
inline double bic(double log_likelihood, std::size_t n, std::size_t k)
{
    return -2.0 * log_likelihood + static_cast<double>(k) * std::log(static_cast<double>(n));
}

/**
 * @brief Calculate BIC from simple regression model
 *
 * Calculates log-likelihood from simple regression result and computes BIC.
 *
 * @param model Simple regression result
 * @param n Sample size
 * @return BIC value
 */
inline double bic_linear(const simple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    return bic(ll, n, 3);
}

/**
 * @brief Calculate BIC from multiple regression model
 *
 * Calculates log-likelihood from multiple regression result and computes BIC.
 *
 * @param model Multiple regression result
 * @param n Sample size
 * @return BIC value
 */
inline double bic_linear(const multiple_regression_result& model, std::size_t n)
{
    double sigma2 = model.ss_residual / static_cast<double>(n);
    double n_d = static_cast<double>(n);

    double ll = -0.5 * n_d * (std::log(2.0 * pi) + std::log(sigma2) + 1.0);

    std::size_t k = model.coefficients.size() + 1;
    return bic(ll, n, k);
}

/**
 * @brief Calculate PRESS statistic
 *
 * Calculates the Prediction Sum of Squares (prediction residual sum of squares).
 * Used for efficient computation of leave-one-out cross-validation.
 *
 * @tparam IteratorX Iterator type for predictor variable
 * @tparam IteratorY Iterator type for response variable
 * @param x_first Beginning iterator for predictor variable
 * @param x_last Ending iterator for predictor variable
 * @param y_first Beginning iterator for response variable
 * @param y_last Ending iterator for response variable
 * @param model Simple regression model
 * @return PRESS statistic
 * @throws std::invalid_argument If x and y have different lengths
 */
template <typename IteratorX, typename IteratorY>
double press_statistic(IteratorX x_first, IteratorX x_last,
                        IteratorY y_first, IteratorY y_last,
                        const simple_regression_result& model)
{
    auto n = statcpp::count(x_first, x_last);
    auto n_y = statcpp::count(y_first, y_last);
    if (n != n_y) {
        throw std::invalid_argument("statcpp::press_statistic: x and y must have same length");
    }

    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Calculate Sxx
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double press = 0.0;
    auto it_x = x_first;
    auto it_y = y_first;
    for (; it_x != x_last; ++it_x, ++it_y) {
        double x_i = static_cast<double>(*it_x);
        double y_i = static_cast<double>(*it_y);

        double y_hat = predict(model, x_i);
        double residual = y_i - y_hat;

        // Leverage h_ii
        double dx = x_i - mean_x;
        double h_ii = 1.0 / n_d + dx * dx / sxx;

        // PRESS residual = residual / (1 - h_ii)
        double press_residual = residual / (1.0 - h_ii);
        press += press_residual * press_residual;
    }

    return press;
}

// ============================================================================
// Cross-Validation
// ============================================================================

/**
 * @brief Structure to store cross-validation results
 */
struct cv_result {
    double mean_error;              /**< Mean error (MSE, MAE, etc.) */
    double se_error;                /**< Standard error of error */
    std::vector<double> fold_errors; /**< Error for each fold */
    std::size_t n_folds;            /**< Number of folds */
};

/**
 * @brief Generate indices for k-fold cross-validation
 *
 * Generates indices for splitting data into k folds.
 *
 * @param n Data size
 * @param k Number of folds
 * @param shuffle Whether to shuffle (default: true)
 * @return Vector of indices belonging to each fold
 * @throws std::invalid_argument If k is less than 2 or k exceeds n
 */
inline std::vector<std::vector<std::size_t>> create_cv_folds(
    std::size_t n, std::size_t k, bool shuffle = true)
{
    if (k < 2) {
        throw std::invalid_argument("statcpp::create_cv_folds: k must be at least 2");
    }
    if (k > n) {
        throw std::invalid_argument("statcpp::create_cv_folds: k cannot exceed n");
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), get_random_engine());
    }

    std::vector<std::vector<std::size_t>> folds(k);
    std::size_t fold_size = n / k;
    std::size_t remainder = n % k;

    std::size_t current = 0;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t this_fold_size = fold_size + (i < remainder ? 1 : 0);
        for (std::size_t j = 0; j < this_fold_size; ++j) {
            folds[i].push_back(indices[current++]);
        }
    }

    return folds;
}

/**
 * @brief Perform k-fold cross-validation for multiple regression model
 *
 * Splits data into specified number of folds and evaluates model prediction performance
 * through cross-validation.
 *
 * @param X Predictor matrix (each row is one sample)
 * @param y Response variable vector
 * @param k Number of folds (default: 5)
 * @return Cross-validation result
 * @throws std::invalid_argument If X and y have different sizes
 */
inline cv_result cross_validate_linear(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t k = 5)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::cross_validate_linear: X and y must have same size");
    }

    auto folds = create_cv_folds(n, k, true);
    std::vector<double> fold_errors(k);

    for (std::size_t fold = 0; fold < k; ++fold) {
        // Separate test and training sets
        std::vector<std::size_t> test_idx = folds[fold];
        std::vector<std::size_t> train_idx;
        for (std::size_t f = 0; f < k; ++f) {
            if (f != fold) {
                train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
            }
        }

        // Training data
        std::vector<std::vector<double>> X_train(train_idx.size());
        std::vector<double> y_train(train_idx.size());
        for (std::size_t i = 0; i < train_idx.size(); ++i) {
            X_train[i] = X[train_idx[i]];
            y_train[i] = y[train_idx[i]];
        }

        // Test data
        std::vector<std::vector<double>> X_test(test_idx.size());
        std::vector<double> y_test(test_idx.size());
        for (std::size_t i = 0; i < test_idx.size(); ++i) {
            X_test[i] = X[test_idx[i]];
            y_test[i] = y[test_idx[i]];
        }

        // Fit model
        try {
            auto model = multiple_linear_regression(X_train, y_train);

            // Calculate test error
            double mse = 0.0;
            for (std::size_t i = 0; i < test_idx.size(); ++i) {
                double pred = predict(model, X_test[i]);
                double err = y_test[i] - pred;
                mse += err * err;
            }
            fold_errors[fold] = mse / static_cast<double>(test_idx.size());
        } catch (...) {
            fold_errors[fold] = std::numeric_limits<double>::infinity();
        }
    }

    double mean_error = statcpp::mean(fold_errors.begin(), fold_errors.end());
    double se_error = statcpp::sample_stddev(fold_errors.begin(), fold_errors.end())
                    / std::sqrt(static_cast<double>(k));

    return {mean_error, se_error, fold_errors, k};
}

/**
 * @brief Perform leave-one-out cross-validation
 *
 * Performs cross-validation using each sample as test data one at a time.
 *
 * @param X Predictor matrix (each row is one sample)
 * @param y Response variable vector
 * @return Cross-validation result
 */
inline cv_result loocv_linear(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    return cross_validate_linear(X, y, X.size());
}

// ============================================================================
// Regularized Regression
// ============================================================================

/**
 * @brief Structure to store regularized regression results
 */
struct regularized_regression_result {
    std::vector<double> coefficients;   /**< Regression coefficients (including intercept) */
    double lambda;                      /**< Regularization parameter */
    double mse;                         /**< Mean squared error */
    std::size_t iterations;             /**< Number of iterations */
    bool converged;                     /**< Convergence flag */
};

/**
 * @brief Perform Ridge regression (L2 regularization)
 *
 * Solves Ridge regression using coordinate descent.
 * L2 penalty shrinks coefficients and handles multicollinearity.
 *
 * @param X Predictor matrix (each row is one sample, no intercept column)
 * @param y Response variable vector
 * @param lambda Regularization parameter (>= 0)
 * @param standardize Whether to standardize data (default: true)
 * @param max_iter Maximum number of iterations (default: 1000)
 * @param tol Convergence tolerance (default: 1e-6)
 * @return Regularized regression result
 * @throws std::invalid_argument If lambda is negative, data is empty, or X and y have different sizes
 */
inline regularized_regression_result ridge_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // Verify that X does not contain an intercept column (from linear_regression.hpp)
    statcpp::detail::validate_no_intercept_column(X, "ridge_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::ridge_regression: lambda must be non-negative");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::ridge_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::ridge_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // Data standardization
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // Ridge closed-form solution: beta = (X'X + lambda*I)^{-1} X'y
    // Solve with coordinate descent
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            // Add back the contribution of this variable to residuals
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            // Calculate X_j'r
            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Ridge update
            double beta_new = xr / (xx + lambda);
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            // Update residuals
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // Transform coefficients back to original scale
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // Calculate MSE
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

/**
 * @brief Perform Lasso regression (L1 regularization)
 *
 * Solves Lasso regression using coordinate descent.
 * L1 penalty shrinks some coefficients exactly to zero, performing variable selection.
 *
 * @param X Predictor matrix (each row is one sample, no intercept column)
 * @param y Response variable vector
 * @param lambda Regularization parameter (>= 0)
 * @param standardize Whether to standardize data (default: true)
 * @param max_iter Maximum number of iterations (default: 1000)
 * @param tol Convergence tolerance (default: 1e-6)
 * @return Regularized regression result
 * @throws std::invalid_argument If lambda is negative, data is empty, or X and y have different sizes
 */
inline regularized_regression_result lasso_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // Verify that X does not contain an intercept column
    statcpp::detail::validate_no_intercept_column(X, "lasso_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::lasso_regression: lambda must be non-negative");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::lasso_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::lasso_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // Data standardization
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // Coordinate descent
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    // Soft thresholding function
    auto soft_threshold = [](double x, double t) -> double {
        if (x > t) return x - t;
        if (x < -t) return x + t;
        return 0.0;
    };

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            // Add back the contribution of this variable to residuals
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            // Calculate X_j'r
            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Lasso update (soft thresholding)
            double beta_new = soft_threshold(xr, lambda) / xx;
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            // Update residuals
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // Transform coefficients back to original scale
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // Calculate MSE
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

/**
 * @brief Perform Elastic Net regression (L1 + L2 regularization)
 *
 * Solves Elastic Net regression using coordinate descent.
 * Combines L1 and L2 penalties, achieving both Lasso's variable selection and Ridge's stability.
 *
 * @param X Predictor matrix (each row is one sample, no intercept column)
 * @param y Response variable vector
 * @param lambda Regularization parameter (>= 0)
 * @param alpha L1 penalty ratio (0 = Ridge, 1 = Lasso, default: 0.5)
 * @param standardize Whether to standardize data (default: true)
 * @param max_iter Maximum number of iterations (default: 1000)
 * @param tol Convergence tolerance (default: 1e-6)
 * @return Regularized regression result
 * @throws std::invalid_argument If lambda is negative, alpha is outside [0,1], data is empty, or X and y have different sizes
 */
inline regularized_regression_result elastic_net_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double lambda,
    double alpha = 0.5,  // L1 ratio (0 = Ridge, 1 = Lasso)
    bool standardize = true,
    std::size_t max_iter = 1000,
    double tol = 1e-6)
{
    // Verify that X does not contain an intercept column
    statcpp::detail::validate_no_intercept_column(X, "elastic_net_regression");

    if (lambda < 0.0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: lambda must be non-negative");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: alpha must be in [0, 1]");
    }

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::elastic_net_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::elastic_net_regression: X and y must have same size");
    }

    std::size_t p = X[0].size();

    // Data standardization
    std::vector<double> X_mean(p, 0.0);
    std::vector<double> X_std(p, 1.0);
    double y_mean = statcpp::mean(y.begin(), y.end());

    std::vector<std::vector<double>> X_scaled = X;
    std::vector<double> y_centered(n);

    if (standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += X[i][j];
            }
            X_mean[j] = sum / static_cast<double>(n);

            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X[i][j] - X_mean[j];
                ss += d * d;
            }
            X_std[j] = std::sqrt(ss / static_cast<double>(n));
            if (X_std[j] < 1e-10) X_std[j] = 1.0;

            for (std::size_t i = 0; i < n; ++i) {
                X_scaled[i][j] = (X[i][j] - X_mean[j]) / X_std[j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        y_centered[i] = y[i] - y_mean;
    }

    // Coordinate descent
    std::vector<double> beta(p, 0.0);
    std::vector<double> residuals = y_centered;

    auto soft_threshold = [](double x, double t) -> double {
        if (x > t) return x - t;
        if (x < -t) return x + t;
        return 0.0;
    };

    double lambda1 = alpha * lambda;        // L1 penalty
    double lambda2 = (1.0 - alpha) * lambda; // L2 penalty

    std::size_t iter = 0;
    bool converged = false;

    for (iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] += X_scaled[i][j] * beta[j];
            }

            double xr = 0.0;
            double xx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                xr += X_scaled[i][j] * residuals[i];
                xx += X_scaled[i][j] * X_scaled[i][j];
            }

            // Elastic Net update
            double beta_new = soft_threshold(xr, lambda1) / (xx + lambda2);
            double change = std::abs(beta_new - beta[j]);
            max_change = std::max(max_change, change);

            beta[j] = beta_new;

            for (std::size_t i = 0; i < n; ++i) {
                residuals[i] -= X_scaled[i][j] * beta[j];
            }
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // Transform coefficients back to original scale
    std::vector<double> coefficients(p + 1);
    if (standardize) {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j] / X_std[j];
            coefficients[0] -= coefficients[j + 1] * X_mean[j];
        }
    } else {
        coefficients[0] = y_mean;
        for (std::size_t j = 0; j < p; ++j) {
            coefficients[j + 1] = beta[j];
        }
    }

    // Calculate MSE
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pred = coefficients[0];
        for (std::size_t j = 0; j < p; ++j) {
            pred += coefficients[j + 1] * X[i][j];
        }
        double err = y[i] - pred;
        mse += err * err;
    }
    mse /= static_cast<double>(n);

    return {coefficients, lambda, mse, iter, converged};
}

// ============================================================================
// Lambda Selection (Regularization Parameter Selection)
// ============================================================================

/**
 * @brief Select optimal lambda for Ridge regression using cross-validation
 *
 * Performs k-fold cross-validation for a grid of lambda values and
 * selects the lambda with minimum cross-validation error.
 *
 * @param X Predictor matrix (each row is one sample)
 * @param y Response variable vector
 * @param lambda_grid Vector of lambda values to evaluate
 * @param k Number of folds (default: 5)
 * @return Pair of optimal lambda and cross-validation errors for each lambda
 */
inline std::pair<double, std::vector<double>> cv_ridge(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& lambda_grid,
    std::size_t k = 5)
{
    std::vector<double> cv_errors(lambda_grid.size());

    for (std::size_t l = 0; l < lambda_grid.size(); ++l) {
        double lambda = lambda_grid[l];
        auto folds = create_cv_folds(X.size(), k, true);

        double total_error = 0.0;
        for (std::size_t fold = 0; fold < k; ++fold) {
            std::vector<std::size_t> test_idx = folds[fold];
            std::vector<std::size_t> train_idx;
            for (std::size_t f = 0; f < k; ++f) {
                if (f != fold) {
                    train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
                }
            }

            std::vector<std::vector<double>> X_train(train_idx.size());
            std::vector<double> y_train(train_idx.size());
            for (std::size_t i = 0; i < train_idx.size(); ++i) {
                X_train[i] = X[train_idx[i]];
                y_train[i] = y[train_idx[i]];
            }

            try {
                auto model = ridge_regression(X_train, y_train, lambda);

                double mse = 0.0;
                for (std::size_t i : test_idx) {
                    double pred = model.coefficients[0];
                    for (std::size_t j = 0; j < X[i].size(); ++j) {
                        pred += model.coefficients[j + 1] * X[i][j];
                    }
                    double err = y[i] - pred;
                    mse += err * err;
                }
                total_error += mse / static_cast<double>(test_idx.size());
            } catch (...) {
                total_error += std::numeric_limits<double>::infinity();
            }
        }
        cv_errors[l] = total_error / static_cast<double>(k);
    }

    // Select lambda with minimum error
    auto min_it = std::min_element(cv_errors.begin(), cv_errors.end());
    double best_lambda = lambda_grid[std::distance(cv_errors.begin(), min_it)];

    return {best_lambda, cv_errors};
}

/**
 * @brief Select optimal lambda for Lasso regression using cross-validation
 *
 * Performs k-fold cross-validation for a grid of lambda values and
 * selects the lambda with minimum cross-validation error.
 *
 * @param X Predictor matrix (each row is one sample)
 * @param y Response variable vector
 * @param lambda_grid Vector of lambda values to evaluate
 * @param k Number of folds (default: 5)
 * @return Pair of optimal lambda and cross-validation errors for each lambda
 */
inline std::pair<double, std::vector<double>> cv_lasso(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& lambda_grid,
    std::size_t k = 5)
{
    std::vector<double> cv_errors(lambda_grid.size());

    for (std::size_t l = 0; l < lambda_grid.size(); ++l) {
        double lambda = lambda_grid[l];
        auto folds = create_cv_folds(X.size(), k, true);

        double total_error = 0.0;
        for (std::size_t fold = 0; fold < k; ++fold) {
            std::vector<std::size_t> test_idx = folds[fold];
            std::vector<std::size_t> train_idx;
            for (std::size_t f = 0; f < k; ++f) {
                if (f != fold) {
                    train_idx.insert(train_idx.end(), folds[f].begin(), folds[f].end());
                }
            }

            std::vector<std::vector<double>> X_train(train_idx.size());
            std::vector<double> y_train(train_idx.size());
            for (std::size_t i = 0; i < train_idx.size(); ++i) {
                X_train[i] = X[train_idx[i]];
                y_train[i] = y[train_idx[i]];
            }

            try {
                auto model = lasso_regression(X_train, y_train, lambda);

                double mse = 0.0;
                for (std::size_t i : test_idx) {
                    double pred = model.coefficients[0];
                    for (std::size_t j = 0; j < X[i].size(); ++j) {
                        pred += model.coefficients[j + 1] * X[i][j];
                    }
                    double err = y[i] - pred;
                    mse += err * err;
                }
                total_error += mse / static_cast<double>(test_idx.size());
            } catch (...) {
                total_error += std::numeric_limits<double>::infinity();
            }
        }
        cv_errors[l] = total_error / static_cast<double>(k);
    }

    auto min_it = std::min_element(cv_errors.begin(), cv_errors.end());
    double best_lambda = lambda_grid[std::distance(cv_errors.begin(), min_it)];

    return {best_lambda, cv_errors};
}

/**
 * @brief Automatically generate lambda grid for regularized regression
 *
 * Calculates maximum lambda based on data and generates a grid of lambda values
 * on a logarithmic scale.
 *
 * @param X Predictor matrix (each row is one sample)
 * @param y Response variable vector
 * @param n_lambda Grid size (default: 100)
 * @param lambda_min_ratio Ratio of lambda_min to lambda_max (default: 0.0001)
 * @return Vector of lambda values equally spaced on logarithmic scale
 */
inline std::vector<double> generate_lambda_grid(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t n_lambda = 100,
    double lambda_min_ratio = 0.0001)
{
    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // Center y
    double y_mean = statcpp::mean(y.begin(), y.end());

    // Calculate lambda_max (lambda where all coefficients are zero)
    double lambda_max = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        double xy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            xy += X[i][j] * (y[i] - y_mean);
        }
        lambda_max = std::max(lambda_max, std::abs(xy) / static_cast<double>(n));
    }

    double lambda_min = lambda_max * lambda_min_ratio;

    // Generate grid on logarithmic scale
    std::vector<double> grid(n_lambda);
    double log_max = std::log(lambda_max);
    double log_min = std::log(lambda_min);
    double step = (log_max - log_min) / static_cast<double>(n_lambda - 1);

    for (std::size_t i = 0; i < n_lambda; ++i) {
        grid[i] = std::exp(log_max - static_cast<double>(i) * step);
    }

    return grid;
}

} // namespace statcpp
