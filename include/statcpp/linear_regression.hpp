/**
 * @file linear_regression.hpp
 * @brief Linear regression analysis
 *
 * Provides linear regression analysis functionality including simple regression,
 * multiple regression, and polynomial regression.
 * Includes prediction, confidence intervals, residual diagnostics, and multicollinearity diagnostics.
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/correlation_covariance.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace statcpp {

// ============================================================================
// Linear Regression Result Structures
// ============================================================================

/**
 * @brief Structure to store simple regression analysis results
 *
 * Holds regression coefficients, standard errors, test statistics,
 * and coefficient of determination for simple regression y = b0 + b1*x.
 */
struct simple_regression_result {
    double intercept;           ///< Intercept (b0)
    double slope;               ///< Slope (b1)
    double intercept_se;        ///< Standard error of intercept
    double slope_se;            ///< Standard error of slope
    double intercept_t;         ///< t-statistic for intercept
    double slope_t;             ///< t-statistic for slope
    double intercept_p;         ///< p-value for intercept
    double slope_p;             ///< p-value for slope
    double r_squared;           ///< Coefficient of determination R^2
    double adj_r_squared;       ///< Adjusted R^2
    double residual_se;         ///< Residual standard error
    double f_statistic;         ///< F-statistic
    double f_p_value;           ///< p-value for F-test
    double df_regression;       ///< Regression degrees of freedom
    double df_residual;         ///< Residual degrees of freedom
    double ss_total;            ///< Total sum of squares
    double ss_regression;       ///< Regression sum of squares
    double ss_residual;         ///< Residual sum of squares
};

/**
 * @brief Structure to store multiple regression analysis results
 *
 * Holds regression coefficients, standard errors, test statistics,
 * and coefficient of determination for multiple regression y = b0 + b1*x1 + ... + bp*xp.
 */
struct multiple_regression_result {
    std::vector<double> coefficients;       ///< Regression coefficients (b0, b1, ..., bp)
    std::vector<double> coefficient_se;     ///< Standard errors of coefficients
    std::vector<double> t_statistics;       ///< t-statistics
    std::vector<double> p_values;           ///< p-values
    double r_squared;                       ///< Coefficient of determination R^2
    double adj_r_squared;                   ///< Adjusted R^2
    double residual_se;                     ///< Residual standard error
    double f_statistic;                     ///< F-statistic
    double f_p_value;                       ///< p-value for F-test
    double df_regression;                   ///< Regression degrees of freedom
    double df_residual;                     ///< Residual degrees of freedom
    double ss_total;                        ///< Total sum of squares
    double ss_regression;                   ///< Regression sum of squares
    double ss_residual;                     ///< Residual sum of squares
};

/**
 * @brief Structure to store prediction interval results
 *
 * Holds predicted values and their confidence intervals (or prediction intervals).
 */
struct prediction_interval {
    double prediction;      ///< Predicted value
    double lower;           ///< Lower bound
    double upper;           ///< Upper bound
    double se_prediction;   ///< Standard error of prediction
};

/**
 * @brief Structure to store residual diagnostics results
 *
 * Holds various diagnostic statistics for regression model residuals.
 * Used for detecting outliers, influential points, and testing for autocorrelation.
 */
struct residual_diagnostics {
    std::vector<double> residuals;              ///< Residuals
    std::vector<double> standardized_residuals; ///< Standardized residuals
    std::vector<double> studentized_residuals;  ///< Studentized residuals
    std::vector<double> hat_values;             ///< Leverage values
    std::vector<double> cooks_distance;         ///< Cook's distance
    double durbin_watson;                       ///< Durbin-Watson statistic
};

// ============================================================================
// Simple Linear Regression
// ============================================================================

/**
 * @brief Perform simple linear regression
 *
 * Estimates simple regression model y = b0 + b1*x using least squares method.
 * Calculates regression coefficients, standard errors, t-tests, F-test, and coefficient of determination.
 *
 * @tparam IteratorX Iterator type for predictor variable
 * @tparam IteratorY Iterator type for response variable
 * @param x_first Beginning iterator for predictor variable
 * @param x_last Ending iterator for predictor variable
 * @param y_first Beginning iterator for response variable
 * @param y_last Ending iterator for response variable
 * @return simple_regression_result Regression analysis results
 * @throws std::invalid_argument If x and y have different lengths
 * @throws std::invalid_argument If there are fewer than 3 observations
 * @throws std::invalid_argument If x has zero variance
 */
template <typename IteratorX, typename IteratorY>
simple_regression_result simple_linear_regression(IteratorX x_first, IteratorX x_last,
                                                   IteratorY y_first, IteratorY y_last)
{
    auto n_x = statcpp::count(x_first, x_last);
    auto n_y = statcpp::count(y_first, y_last);

    if (n_x != n_y) {
        throw std::invalid_argument("statcpp::simple_linear_regression: x and y must have same length");
    }
    if (n_x < 3) {
        throw std::invalid_argument("statcpp::simple_linear_regression: need at least 3 observations");
    }

    std::size_t n = n_x;
    double n_d = static_cast<double>(n);

    // Calculate means
    double mean_x = statcpp::mean(x_first, x_last);
    double mean_y = statcpp::mean(y_first, y_last);

    // Calculate Sxx, Syy, Sxy
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    auto it_x = x_first;
    auto it_y = y_first;
    for (; it_x != x_last; ++it_x, ++it_y) {
        double dx = static_cast<double>(*it_x) - mean_x;
        double dy = static_cast<double>(*it_y) - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if (sxx == 0.0) {
        throw std::invalid_argument("statcpp::simple_linear_regression: zero variance in x");
    }

    // Regression coefficients
    double slope = sxy / sxx;
    double intercept = mean_y - slope * mean_x;

    // Sum of squares
    double ss_total = syy;
    double ss_regression = slope * sxy;
    double ss_residual = ss_total - ss_regression;

    if (ss_total == 0.0) {
        throw std::invalid_argument("statcpp::simple_linear_regression: zero variance in y (constant response)");
    }

    // Residual variance and standard error
    double df_reg = 1.0;
    double df_res = n_d - 2.0;
    double mse = ss_residual / df_res;
    double residual_se = std::sqrt(mse);

    // Standard errors of coefficients
    double slope_se = residual_se / std::sqrt(sxx);
    double intercept_se = residual_se * std::sqrt(1.0 / n_d + mean_x * mean_x / sxx);

    // t-statistics and p-values
    double slope_t = slope / slope_se;
    double intercept_t = intercept / intercept_se;
    double slope_p = 2.0 * (1.0 - t_cdf(std::abs(slope_t), df_res));
    double intercept_p = 2.0 * (1.0 - t_cdf(std::abs(intercept_t), df_res));

    // Coefficient of determination
    double r_squared = ss_regression / ss_total;
    double adj_r_squared = 1.0 - (1.0 - r_squared) * (n_d - 1.0) / df_res;

    // F-statistic
    double f_statistic = (ss_regression / df_reg) / mse;
    double f_p_value = 1.0 - f_cdf(f_statistic, df_reg, df_res);

    return {
        intercept, slope,
        intercept_se, slope_se,
        intercept_t, slope_t,
        intercept_p, slope_p,
        r_squared, adj_r_squared,
        residual_se,
        f_statistic, f_p_value,
        df_reg, df_res,
        ss_total, ss_regression, ss_residual
    };
}

// ============================================================================
// Multiple Linear Regression
// ============================================================================

namespace detail {

/**
 * @brief Validate 2D matrix structure
 *
 * Verifies that the matrix is not empty and all rows have the same number of columns.
 *
 * @param data Matrix to validate
 * @param func_name Function name for error messages
 * @throws std::invalid_argument If matrix is empty
 * @throws std::invalid_argument If first row is empty
 * @throws std::invalid_argument If rows have different column counts
 */
inline void validate_matrix_structure(const std::vector<std::vector<double>>& data,
                                      const char* func_name)
{
    if (data.empty()) {
        std::string msg = "statcpp::";
        msg += func_name;
        msg += ": empty data";
        throw std::invalid_argument(msg);
    }

    std::size_t p = data[0].size();
    if (p == 0) {
        std::string msg = "statcpp::";
        msg += func_name;
        msg += ": first row is empty (0 columns)";
        throw std::invalid_argument(msg);
    }

    // Verify all rows have the same number of columns
    for (std::size_t i = 1; i < data.size(); ++i) {
        if (data[i].size() != p) {
            std::string msg = "statcpp::";
            msg += func_name;
            msg += ": inconsistent row dimensions (row 0 has ";
            msg += std::to_string(p);
            msg += " columns, but row ";
            msg += std::to_string(i);
            msg += " has ";
            msg += std::to_string(data[i].size());
            msg += " columns)";
            throw std::invalid_argument(msg);
        }
    }
}

/**
 * @brief Check that X data does not contain an intercept column
 *
 * Warns if the first column is all 1.0, as the user may have mistakenly included an intercept column.
 * The intercept is added automatically within the function, so users don't need to include it.
 *
 * @param X Predictor matrix
 * @param func_name Function name for error messages
 * @throws std::invalid_argument If the first column is all 1.0
 */
inline void validate_no_intercept_column(const std::vector<std::vector<double>>& X, const char* func_name)
{
    if (X.empty()) {
        return;
    }

    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // Check if first column is all 1.0
    if (p > 0) {
        bool all_ones = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (X[i].size() == 0 || std::abs(X[i][0] - 1.0) > 1e-10) {
                all_ones = false;
                break;
            }
        }

        if (all_ones) {
            std::string msg = "statcpp::";
            msg += func_name;
            msg += ": X should not contain intercept column (all 1s in first column detected). ";
            msg += "The intercept is added automatically.";
            throw std::invalid_argument(msg);
        }
    }
}

// Simple matrix operations (for small matrices)

/**
 * @brief Calculate transpose matrix
 *
 * @param A Input matrix
 * @return std::vector<std::vector<double>> Transposed matrix
 * @throws std::invalid_argument If rows have different column counts
 */
inline std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A)
{
    if (A.empty()) return {};

    // Validate matrix structure (all rows have same size)
    std::size_t rows = A.size();
    std::size_t cols = A[0].size();
    for (std::size_t i = 1; i < rows; ++i) {
        if (A[i].size() != cols) {
            throw std::invalid_argument("statcpp::detail::transpose: inconsistent row dimensions");
        }
    }

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

/**
 * @brief Calculate matrix product
 *
 * @param A Left matrix (m x n)
 * @param B Right matrix (n x p)
 * @return std::vector<std::vector<double>> Product matrix (m x p)
 * @throws std::invalid_argument If matrix dimensions are incompatible
 * @throws std::invalid_argument If matrix rows are inconsistent
 */
inline std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B)
{
    if (A.empty() || B.empty()) return {};
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t p = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("statcpp::detail::matrix_multiply: incompatible dimensions");
    }

    // Check that each row has consistent size
    for (std::size_t i = 0; i < m; ++i) {
        if (A[i].size() != n) {
            throw std::invalid_argument("statcpp::detail::matrix_multiply: matrix A has inconsistent row dimensions");
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (B[k].size() != p) {
            throw std::invalid_argument("statcpp::detail::matrix_multiply: matrix B has inconsistent row dimensions");
        }
    }

    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

/**
 * @brief Calculate matrix-vector product
 *
 * @param A Matrix (m x n)
 * @param v Vector (n-dimensional)
 * @return std::vector<double> Result vector (m-dimensional)
 * @throws std::invalid_argument If matrix and vector dimensions are incompatible
 */
inline std::vector<double> matrix_vector_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v)
{
    if (A.empty()) return {};
    std::size_t m = A.size();
    std::size_t n = A[0].size();

    if (n != v.size()) {
        throw std::invalid_argument("statcpp::detail::matrix_vector_multiply: incompatible dimensions");
    }

    std::vector<double> result(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

/**
 * @brief Perform Cholesky decomposition
 *
 * Computes lower triangular matrix L such that L * L^T = A for positive definite symmetric matrix A.
 *
 * @param A Positive definite symmetric matrix
 * @return std::vector<std::vector<double>> Lower triangular matrix L
 * @throws std::runtime_error If matrix is not positive definite
 */
inline std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>>& A)
{
    std::size_t n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double val = A[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("statcpp::detail::cholesky: matrix is not positive definite");
                }
                L[i][j] = std::sqrt(val);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

/**
 * @brief Solve system of equations using Cholesky decomposition
 *
 * Solves A * x = b (where A = L * L^T).
 * Efficiently finds the solution using forward and back substitution.
 *
 * @param L Lower triangular matrix from Cholesky decomposition
 * @param b Right-hand side vector
 * @return std::vector<double> Solution vector x
 */
inline std::vector<double> solve_cholesky(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& b)
{
    std::size_t n = L.size();

    // Forward substitution: L * y = b
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Back substitution: L^T * x = y
    std::vector<double> x(n);
    for (std::size_t i = n; i > 0; --i) {
        std::size_t idx = i - 1;
        double sum = 0.0;
        for (std::size_t j = idx + 1; j < n; ++j) {
            sum += L[j][idx] * x[j];
        }
        x[idx] = (y[idx] - sum) / L[idx][idx];
    }
    return x;
}

/**
 * @brief Calculate inverse matrix using Cholesky decomposition
 *
 * @param L Lower triangular matrix from Cholesky decomposition
 * @return std::vector<std::vector<double>> Inverse of the original matrix
 */
inline std::vector<std::vector<double>> inverse_cholesky(
    const std::vector<std::vector<double>>& L)
{
    std::size_t n = L.size();
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));

    // Solve each column
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> e(n, 0.0);
        e[j] = 1.0;
        std::vector<double> col = solve_cholesky(L, e);
        for (std::size_t i = 0; i < n; ++i) {
            inv[i][j] = col[i];
        }
    }
    return inv;
}

} // namespace detail

/**
 * @brief Perform multiple linear regression
 *
 * Estimates multiple regression model y = b0 + b1*x1 + ... + bp*xp using least squares method.
 * The intercept is added automatically, so X should not contain an intercept column.
 *
 * @param X Predictor matrix (n x p). Each row is one observation, each column is one predictor
 * @param y Response variable vector (n-dimensional)
 * @return multiple_regression_result Regression analysis results
 * @throws std::invalid_argument If data is empty
 * @throws std::invalid_argument If X and Y have different numbers of observations
 * @throws std::invalid_argument If number of predictors is greater than or equal to number of observations
 * @throws std::invalid_argument If X contains an intercept column (all 1s column)
 */
inline multiple_regression_result multiple_linear_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    // Verify that X does not contain an intercept column
    detail::validate_no_intercept_column(X, "multiple_linear_regression");

    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: X and y must have same number of observations");
    }

    std::size_t p = X[0].size();  // Number of predictors (excluding intercept)
    for (const auto& row : X) {
        if (row.size() != p) {
            throw std::invalid_argument("statcpp::multiple_linear_regression: inconsistent number of predictors");
        }
    }

    std::size_t p_full = p + 1;  // Number of coefficients including intercept
    if (n <= p_full) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: need more observations than predictors");
    }

    double n_d = static_cast<double>(n);

    // Create design matrix with intercept term
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;  // Intercept
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // Calculate X^T * X
    auto Xt = detail::transpose(X_design);
    auto XtX = detail::matrix_multiply(Xt, X_design);

    // Calculate X^T * y
    auto Xty = detail::matrix_vector_multiply(Xt, y);

    // Solve for coefficients using Cholesky decomposition
    auto L = detail::cholesky(XtX);
    auto coefficients = detail::solve_cholesky(L, Xty);

    // Calculate (X^T X)^{-1}
    auto XtX_inv = detail::inverse_cholesky(L);

    // Calculate predicted values and residuals
    std::vector<double> y_hat(n);
    std::vector<double> residuals(n);
    double mean_y = statcpp::mean(y.begin(), y.end());

    double ss_total = 0.0;
    double ss_residual = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double pred = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            pred += X_design[i][j] * coefficients[j];
        }
        y_hat[i] = pred;
        residuals[i] = y[i] - pred;

        ss_total += (y[i] - mean_y) * (y[i] - mean_y);
        ss_residual += residuals[i] * residuals[i];
    }

    double ss_regression = ss_total - ss_residual;

    if (ss_total == 0.0) {
        throw std::invalid_argument("statcpp::multiple_linear_regression: zero variance in y (constant response)");
    }

    // Degrees of freedom
    double df_reg = static_cast<double>(p);
    double df_res = n_d - static_cast<double>(p_full);

    // Residual variance
    double mse = ss_residual / df_res;
    double residual_se = std::sqrt(mse);

    // Standard errors, t-statistics, and p-values of coefficients
    std::vector<double> coefficient_se(p_full);
    std::vector<double> t_statistics(p_full);
    std::vector<double> p_values(p_full);

    for (std::size_t j = 0; j < p_full; ++j) {
        coefficient_se[j] = std::sqrt(mse * XtX_inv[j][j]);
        t_statistics[j] = coefficients[j] / coefficient_se[j];
        p_values[j] = 2.0 * (1.0 - t_cdf(std::abs(t_statistics[j]), df_res));
    }

    // Coefficient of determination
    double r_squared = ss_regression / ss_total;
    double adj_r_squared = 1.0 - (1.0 - r_squared) * (n_d - 1.0) / df_res;

    // F-statistic
    double f_statistic = (ss_regression / df_reg) / mse;
    double f_p_value = 1.0 - f_cdf(f_statistic, df_reg, df_res);

    return {
        coefficients, coefficient_se, t_statistics, p_values,
        r_squared, adj_r_squared,
        residual_se,
        f_statistic, f_p_value,
        df_reg, df_res,
        ss_total, ss_regression, ss_residual
    };
}

// ============================================================================
// Prediction
// ============================================================================

/**
 * @brief Make prediction using simple regression model
 *
 * @param model Simple regression analysis results
 * @param x Predictor variable value
 * @return double Predicted value
 */
inline double predict(const simple_regression_result& model, double x)
{
    return model.intercept + model.slope * x;
}

/**
 * @brief Make prediction using multiple regression model
 *
 * @param model Multiple regression analysis results
 * @param x Predictor variable vector (do not include intercept)
 * @return double Predicted value
 * @throws std::invalid_argument If x dimension doesn't match model's number of predictors
 */
inline double predict(const multiple_regression_result& model, const std::vector<double>& x)
{
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict: x dimension mismatch");
    }

    double pred = model.coefficients[0];  // Intercept
    for (std::size_t i = 0; i < x.size(); ++i) {
        pred += model.coefficients[i + 1] * x[i];
    }
    return pred;
}

// ============================================================================
// Prediction Interval
// ============================================================================

/**
 * @brief Calculate prediction interval for simple regression model
 *
 * Calculates the prediction interval for a new observation.
 * The prediction interval shows the probability that a future individual observation falls within this interval.
 *
 * @tparam IteratorX Iterator type for predictor variable
 * @param model Simple regression analysis results
 * @param x_first Beginning iterator for original predictor variable data
 * @param x_last Ending iterator for original predictor variable data
 * @param x_new x value of the point to predict
 * @param confidence Confidence level (default: 0.95)
 * @return prediction_interval Predicted value and prediction interval
 * @throws std::invalid_argument If confidence is outside (0, 1) range
 */
template <typename IteratorX>
prediction_interval prediction_interval_simple(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    double x_new,
    double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::prediction_interval_simple: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(x_first, x_last);
    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Calculate Sxx
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double y_hat = predict(model, x_new);

    // Standard error of prediction (for new observation)
    double dx_new = x_new - mean_x;
    double se_pred = model.residual_se * std::sqrt(1.0 + 1.0 / n_d + dx_new * dx_new / sxx);

    double t_crit = t_quantile(1.0 - (1.0 - confidence) / 2.0, model.df_residual);
    double margin = t_crit * se_pred;

    return {y_hat, y_hat - margin, y_hat + margin, se_pred};
}

/**
 * @brief Calculate confidence interval for mean of simple regression model
 *
 * Calculates the confidence interval for the mean response at a specific x value.
 * The confidence interval shows the probability that the true regression line is within this interval.
 *
 * @tparam IteratorX Iterator type for predictor variable
 * @param model Simple regression analysis results
 * @param x_first Beginning iterator for original predictor variable data
 * @param x_last Ending iterator for original predictor variable data
 * @param x_new x value of the point to predict
 * @param confidence Confidence level (default: 0.95)
 * @return prediction_interval Predicted value and confidence interval
 * @throws std::invalid_argument If confidence is outside (0, 1) range
 */
template <typename IteratorX>
prediction_interval confidence_interval_mean(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    double x_new,
    double confidence = 0.95)
{
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::confidence_interval_mean: confidence must be in (0, 1)");
    }

    auto n = statcpp::count(x_first, x_last);
    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Calculate Sxx
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    double y_hat = predict(model, x_new);

    // Standard error of mean prediction
    double dx_new = x_new - mean_x;
    double se_mean = model.residual_se * std::sqrt(1.0 / n_d + dx_new * dx_new / sxx);

    double t_crit = t_quantile(1.0 - (1.0 - confidence) / 2.0, model.df_residual);
    double margin = t_crit * se_mean;

    return {y_hat, y_hat - margin, y_hat + margin, se_mean};
}

// ============================================================================
// Residual Diagnostics
// ============================================================================

/**
 * @brief Perform residual diagnostics for simple regression model
 *
 * Calculates residuals, standardized residuals, studentized residuals, leverage values,
 * Cook's distance, and Durbin-Watson statistic.
 *
 * @tparam IteratorX Iterator type for predictor variable
 * @tparam IteratorY Iterator type for response variable
 * @param model Simple regression analysis results
 * @param x_first Beginning iterator for predictor variable
 * @param x_last Ending iterator for predictor variable
 * @param y_first Beginning iterator for response variable
 * @param y_last Ending iterator for response variable
 * @return residual_diagnostics Residual diagnostics results
 * @throws std::invalid_argument If x and y have different lengths
 */
template <typename IteratorX, typename IteratorY>
residual_diagnostics compute_residual_diagnostics(
    const simple_regression_result& model,
    IteratorX x_first, IteratorX x_last,
    IteratorY y_first, IteratorY y_last)
{
    auto n = statcpp::count(x_first, x_last);
    if (n != statcpp::count(y_first, y_last)) {
        throw std::invalid_argument("statcpp::compute_residual_diagnostics: x and y must have same length");
    }

    double n_d = static_cast<double>(n);
    double mean_x = statcpp::mean(x_first, x_last);

    // Calculate Sxx
    double sxx = 0.0;
    for (auto it = x_first; it != x_last; ++it) {
        double dx = static_cast<double>(*it) - mean_x;
        sxx += dx * dx;
    }

    std::vector<double> residuals(n);
    std::vector<double> hat_values(n);
    std::vector<double> standardized_residuals(n);

    auto it_x = x_first;
    auto it_y = y_first;
    for (std::size_t i = 0; it_x != x_last; ++it_x, ++it_y, ++i) {
        double x_i = static_cast<double>(*it_x);
        double y_i = static_cast<double>(*it_y);
        double y_hat = predict(model, x_i);
        residuals[i] = y_i - y_hat;

        // Leverage h_ii = 1/n + (x_i - mean_x)^2 / Sxx
        double dx = x_i - mean_x;
        hat_values[i] = 1.0 / n_d + dx * dx / sxx;

        // Standardized residuals
        standardized_residuals[i] = residuals[i] / model.residual_se;
    }

    // Studentized residuals and Cook's distance
    std::vector<double> studentized_residuals(n);
    std::vector<double> cooks_distance(n);
    double p = 2.0;  // Number of coefficients (intercept + slope)

    for (std::size_t i = 0; i < n; ++i) {
        double h_i = hat_values[i];
        double se_i = model.residual_se * std::sqrt(1.0 - h_i);
        studentized_residuals[i] = (se_i > 0.0) ? residuals[i] / se_i : 0.0;
        cooks_distance[i] = (standardized_residuals[i] * standardized_residuals[i] / p)
                          * (h_i / (1.0 - h_i));
    }

    // Durbin-Watson statistic
    double dw_num = 0.0;
    double dw_den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dw_den += residuals[i] * residuals[i];
        if (i > 0) {
            double diff = residuals[i] - residuals[i - 1];
            dw_num += diff * diff;
        }
    }
    double durbin_watson = (dw_den > 0.0) ? dw_num / dw_den : 0.0;

    return {residuals, standardized_residuals, studentized_residuals,
            hat_values, cooks_distance, durbin_watson};
}

/**
 * @brief Perform residual diagnostics for multiple regression model
 *
 * Calculates residuals, standardized residuals, studentized residuals, leverage values,
 * Cook's distance, and Durbin-Watson statistic.
 *
 * @param model Multiple regression analysis results
 * @param X Predictor matrix
 * @param y Response variable vector
 * @return residual_diagnostics Residual diagnostics results
 * @throws std::invalid_argument If X and y have different lengths
 */
inline residual_diagnostics compute_residual_diagnostics(
    const multiple_regression_result& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::compute_residual_diagnostics: X and y must have same length");
    }

    std::size_t p = X[0].size();
    std::size_t p_full = p + 1;

    // Create design matrix
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // Calculate (X^T X)^{-1}
    auto Xt = detail::transpose(X_design);
    auto XtX = detail::matrix_multiply(Xt, X_design);
    auto L = detail::cholesky(XtX);
    auto XtX_inv = detail::inverse_cholesky(L);

    // H = X(X^T X)^{-1}X^T leverage values (diagonal elements only)
    std::vector<double> hat_values(n);
    for (std::size_t i = 0; i < n; ++i) {
        double h_ii = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            for (std::size_t k = 0; k < p_full; ++k) {
                h_ii += X_design[i][j] * XtX_inv[j][k] * X_design[i][k];
            }
        }
        hat_values[i] = h_ii;
    }

    // Calculate residuals
    std::vector<double> residuals(n);
    for (std::size_t i = 0; i < n; ++i) {
        double pred = predict(model, X[i]);
        residuals[i] = y[i] - pred;
    }

    // Standardized residuals, studentized residuals, Cook's distance
    std::vector<double> standardized_residuals(n);
    std::vector<double> studentized_residuals(n);
    std::vector<double> cooks_distance(n);

    double p_d = static_cast<double>(p_full);

    for (std::size_t i = 0; i < n; ++i) {
        standardized_residuals[i] = residuals[i] / model.residual_se;

        double h_i = hat_values[i];
        double se_i = model.residual_se * std::sqrt(1.0 - h_i);
        studentized_residuals[i] = (se_i > 0.0) ? residuals[i] / se_i : 0.0;

        cooks_distance[i] = (standardized_residuals[i] * standardized_residuals[i] / p_d)
                          * (h_i / (1.0 - h_i));
    }

    // Durbin-Watson statistic
    double dw_num = 0.0;
    double dw_den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dw_den += residuals[i] * residuals[i];
        if (i > 0) {
            double diff = residuals[i] - residuals[i - 1];
            dw_num += diff * diff;
        }
    }
    double durbin_watson = (dw_den > 0.0) ? dw_num / dw_den : 0.0;

    return {residuals, standardized_residuals, studentized_residuals,
            hat_values, cooks_distance, durbin_watson};
}

// ============================================================================
// VIF (Variance Inflation Factor)
// ============================================================================

/**
 * @brief Calculate VIF (Variance Inflation Factor) for each predictor
 *
 * VIF is an indicator of multicollinearity, showing the degree to which
 * each predictor is explained by other predictors. Generally VIF > 10
 * suggests a multicollinearity problem.
 *
 * @param X Predictor matrix (n x p)
 * @return std::vector<double> VIF for each predictor
 * @throws std::invalid_argument If there are fewer than 3 observations
 * @throws std::invalid_argument If there are fewer than 2 predictors
 */
inline std::vector<double> compute_vif(const std::vector<std::vector<double>>& X)
{
    std::size_t n = X.size();
    if (n < 3) {
        throw std::invalid_argument("statcpp::compute_vif: need at least 3 observations");
    }

    std::size_t p = X[0].size();
    if (p < 2) {
        throw std::invalid_argument("statcpp::compute_vif: need at least 2 predictors");
    }

    std::vector<double> vif(p);

    for (std::size_t j = 0; j < p; ++j) {
        // Regress j-th variable as response on other variables as predictors
        std::vector<double> y_j(n);
        std::vector<std::vector<double>> X_others(n, std::vector<double>(p - 1));

        for (std::size_t i = 0; i < n; ++i) {
            y_j[i] = X[i][j];
            std::size_t col = 0;
            for (std::size_t k = 0; k < p; ++k) {
                if (k != j) {
                    X_others[i][col++] = X[i][k];
                }
            }
        }

        auto result = multiple_linear_regression(X_others, y_j);
        double r_sq = result.r_squared;

        // VIF = 1 / (1 - R^2)
        if (r_sq >= 1.0) {
            vif[j] = std::numeric_limits<double>::infinity();
        } else {
            vif[j] = 1.0 / (1.0 - r_sq);
        }
    }

    return vif;
}

// ============================================================================
// Multicollinearity Diagnostics (Extended)
// ============================================================================

/**
 * @brief Calculate determinant of correlation matrix
 *
 * Used for multicollinearity diagnostics.
 * det(R) close to 0 indicates strong multicollinearity,
 * det(R) = 1 indicates no correlation between variables.
 *
 * @param X Predictor matrix (n x p)
 * @return double Determinant of correlation matrix
 * @throws std::invalid_argument If there are fewer than 2 observations
 * @throws std::invalid_argument If there are fewer than 2 or more than 3 predictors (only 2 or 3 supported)
 */
inline double correlation_matrix_determinant(const std::vector<std::vector<double>>& X)
{
    std::size_t n = X.size();
    if (n < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: need at least 2 observations");
    }

    std::size_t p = X[0].size();
    if (p < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: need at least 2 predictors");
    }

    // Calculate correlation matrix
    std::vector<std::vector<double>> corr_matrix(p, std::vector<double>(p));
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            if (i == j) {
                corr_matrix[i][j] = 1.0;
            } else if (j > i) {
                // Extract each column
                std::vector<double> col_i(n), col_j(n);
                for (std::size_t k = 0; k < n; ++k) {
                    col_i[k] = X[k][i];
                    col_j[k] = X[k][j];
                }
                double corr = pearson_correlation(col_i.begin(), col_i.end(),
                                                   col_j.begin(), col_j.end());
                corr_matrix[i][j] = corr;
                corr_matrix[j][i] = corr;
            }
        }
    }

    // Calculate determinant (only supports small matrices)
    if (p == 2) {
        // 2x2 matrix: det = a11*a22 - a12*a21
        return corr_matrix[0][0] * corr_matrix[1][1] - corr_matrix[0][1] * corr_matrix[1][0];
    } else if (p == 3) {
        // 3x3 matrix: Sarrus' rule
        double a = corr_matrix[0][0] * corr_matrix[1][1] * corr_matrix[2][2];
        double b = corr_matrix[0][1] * corr_matrix[1][2] * corr_matrix[2][0];
        double c = corr_matrix[0][2] * corr_matrix[1][0] * corr_matrix[2][1];
        double d = corr_matrix[0][2] * corr_matrix[1][1] * corr_matrix[2][0];
        double e = corr_matrix[0][0] * corr_matrix[1][2] * corr_matrix[2][1];
        double f = corr_matrix[0][1] * corr_matrix[1][0] * corr_matrix[2][2];
        return a + b + c - d - e - f;
    } else {
        // Larger matrices would require LU decomposition, not supported here
        throw std::invalid_argument("statcpp::correlation_matrix_determinant: only 2 or 3 predictors supported");
    }
}

/**
 * @brief Calculate multicollinearity score
 *
 * Score ranges from 0 to 1, with values closer to 1 indicating stronger multicollinearity.
 * Calculated as Score = 1 - |det(R)|.
 *
 * @param X Predictor matrix (n x p)
 * @return double Multicollinearity score (0: no correlation, 1: perfect multicollinearity)
 * @throws std::invalid_argument If there are fewer than 2 observations
 * @throws std::invalid_argument If there are fewer than 2 or more than 3 predictors
 */
inline double multicollinearity_score(const std::vector<std::vector<double>>& X)
{
    double det = correlation_matrix_determinant(X);
    // Determinant can be negative, but what matters is how close to 0
    return 1.0 - std::abs(det);
}

// ============================================================================
// R-squared and Related Measures
// ============================================================================

/**
 * @brief Calculate coefficient of determination from observed and predicted values
 *
 * Calculated as R^2 = 1 - SS_residual / SS_total.
 *
 * @tparam IteratorY Iterator type for observed values
 * @tparam IteratorPred Iterator type for predicted values
 * @param y_first Beginning iterator for observed values
 * @param y_last Ending iterator for observed values
 * @param pred_first Beginning iterator for predicted values
 * @param pred_last Ending iterator for predicted values
 * @return double Coefficient of determination R^2
 * @throws std::invalid_argument If y and predictions have different lengths
 * @throws std::invalid_argument If there are fewer than 2 observations
 */
template <typename IteratorY, typename IteratorPred>
double r_squared(IteratorY y_first, IteratorY y_last,
                  IteratorPred pred_first, IteratorPred pred_last)
{
    auto n_y = statcpp::count(y_first, y_last);
    auto n_pred = statcpp::count(pred_first, pred_last);

    if (n_y != n_pred) {
        throw std::invalid_argument("statcpp::r_squared: y and predictions must have same length");
    }
    if (n_y < 2) {
        throw std::invalid_argument("statcpp::r_squared: need at least 2 observations");
    }

    double mean_y = statcpp::mean(y_first, y_last);

    double ss_total = 0.0;
    double ss_residual = 0.0;

    auto it_y = y_first;
    auto it_pred = pred_first;
    for (; it_y != y_last; ++it_y, ++it_pred) {
        double y_i = static_cast<double>(*it_y);
        double pred_i = static_cast<double>(*it_pred);
        ss_total += (y_i - mean_y) * (y_i - mean_y);
        ss_residual += (y_i - pred_i) * (y_i - pred_i);
    }

    if (ss_total == 0.0) {
        return 1.0;  // Completely constant case
    }

    return 1.0 - ss_residual / ss_total;
}

/**
 * @brief Calculate adjusted coefficient of determination
 *
 * Coefficient of determination adjusted for overfitting due to number of predictors.
 * Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
 *
 * @tparam IteratorY Iterator type for observed values
 * @tparam IteratorPred Iterator type for predicted values
 * @param y_first Beginning iterator for observed values
 * @param y_last Ending iterator for observed values
 * @param pred_first Beginning iterator for predicted values
 * @param pred_last Ending iterator for predicted values
 * @param num_predictors Number of predictors (excluding intercept)
 * @return double Adjusted coefficient of determination
 * @throws std::invalid_argument If y and predictions have different lengths
 * @throws std::invalid_argument If number of observations is not greater than number of predictors + 1
 */
template <typename IteratorY, typename IteratorPred>
double adjusted_r_squared(IteratorY y_first, IteratorY y_last,
                           IteratorPred pred_first, IteratorPred pred_last,
                           std::size_t num_predictors)
{
    auto n = statcpp::count(y_first, y_last);
    double n_d = static_cast<double>(n);
    double p = static_cast<double>(num_predictors);

    if (n_d <= p + 1.0) {
        throw std::invalid_argument("statcpp::adjusted_r_squared: need more observations than predictors");
    }

    double r_sq = r_squared(y_first, y_last, pred_first, pred_last);
    return 1.0 - (1.0 - r_sq) * (n_d - 1.0) / (n_d - p - 1.0);
}

} // namespace statcpp
