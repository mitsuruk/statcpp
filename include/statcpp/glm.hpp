/**
 * @file glm.hpp
 * @brief Generalized Linear Models (GLM)
 *
 * Provides generalized linear models including logistic regression and Poisson regression.
 * Uses the IRLS (Iteratively Reweighted Least Squares) algorithm for parameter estimation.
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/linear_regression.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// GLM Result Structures
// ============================================================================

/**
 * @brief Link function types
 *
 * Specifies the link function to use in generalized linear models.
 */
enum class link_function {
    identity,       ///< Identity link (linear regression)
    logit,          ///< Logit link (logistic regression)
    probit,         ///< Probit link
    log,            ///< Log link (Poisson regression)
    inverse,        ///< Inverse link (Gamma regression)
    cloglog         ///< Complementary log-log link
};

/**
 * @brief Distribution family
 *
 * Specifies the probability distribution family to use in generalized linear models.
 */
enum class distribution_family {
    gaussian,       ///< Gaussian (normal) distribution
    binomial,       ///< Binomial distribution
    poisson,        ///< Poisson distribution
    gamma_family    ///< Gamma distribution (gamma_family because gamma is a reserved word)
};

/**
 * @brief GLM result structure
 *
 * Stores the results of generalized linear model fitting.
 */
struct glm_result {
    std::vector<double> coefficients;       ///< Regression coefficients
    std::vector<double> coefficient_se;     ///< Standard errors of coefficients
    std::vector<double> z_statistics;       ///< z-statistics (or Wald statistics)
    std::vector<double> p_values;           ///< p-values
    double null_deviance;                   ///< Null deviance
    double residual_deviance;               ///< Residual deviance
    double df_null;                         ///< Null model degrees of freedom
    double df_residual;                     ///< Residual degrees of freedom
    double aic;                             ///< AIC
    double bic;                             ///< BIC
    double log_likelihood;                  ///< Log-likelihood
    std::size_t iterations;                 ///< Number of iterations until convergence
    bool converged;                         ///< Whether convergence was achieved
    link_function link;                     ///< Link function used
    distribution_family family;             ///< Distribution family used
};

// ============================================================================
// Link Functions
// ============================================================================

namespace detail {

/**
 * @brief Link function g(mu) -> eta
 *
 * Transforms the expected value mu using the link function to return the linear predictor eta.
 *
 * @param mu Expected value
 * @param link Link function to use
 * @return Linear predictor eta
 */
inline double link_transform(double mu, link_function link)
{
    switch (link) {
        case link_function::identity:
            return mu;
        case link_function::logit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return std::log(mu / (1.0 - mu));
        case link_function::probit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return norm_quantile(mu);
        case link_function::log:
            return std::log(std::max(1e-10, mu));
        case link_function::inverse:
            return 1.0 / std::max(1e-10, mu);
        case link_function::cloglog:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return std::log(-std::log(1.0 - mu));
        default:
            return mu;
    }
}

/**
 * @brief Inverse link function g^{-1}(eta) -> mu
 *
 * Transforms the linear predictor eta using the inverse link function to return the expected value mu.
 *
 * @param eta Linear predictor
 * @param link Link function to use
 * @return Expected value mu
 */
inline double inverse_link(double eta, link_function link)
{
    switch (link) {
        case link_function::identity:
            return eta;
        case link_function::logit:
            return 1.0 / (1.0 + std::exp(-eta));
        case link_function::probit:
            return norm_cdf(eta);
        case link_function::log:
            return std::exp(eta);
        case link_function::inverse:
            return 1.0 / eta;
        case link_function::cloglog:
            return 1.0 - std::exp(-std::exp(eta));
        default:
            return eta;
    }
}

/**
 * @brief Derivative of link function d(eta)/d(mu) = g'(mu)
 *
 * Calculates the derivative of the link function with respect to the expected value mu.
 *
 * @param mu Expected value
 * @param link Link function to use
 * @return Derivative of the link function
 * @throws std::runtime_error If mu is close to 0 for cloglog link
 */
inline double link_derivative(double mu, link_function link)
{
    switch (link) {
        case link_function::identity:
            return 1.0;
        case link_function::logit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return 1.0 / (mu * (1.0 - mu));
        case link_function::probit:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return 1.0 / normal_pdf(norm_quantile(mu));
        case link_function::log:
            return 1.0 / std::max(1e-10, mu);
        case link_function::inverse:
            mu = std::max(1e-10, mu);
            return -1.0 / (mu * mu);
        case link_function::cloglog: {
            mu = std::max(1e-8, std::min(1.0 - 1e-8, mu));
            double neg_log_term = -std::log(1.0 - mu);  // -log(1-mu) > 0 for 0 < mu < 1
            // Protection when -log(1-mu) is close to 0 (mu near 0)
            if (neg_log_term < 1e-10) {
                throw std::runtime_error("statcpp::link_derivative: cloglog derivative undefined near mu=0");
            }
            // g'(mu) = 1 / ((1-mu) * (-log(1-mu)))
            return 1.0 / ((1.0 - mu) * neg_log_term);
        }
        default:
            return 1.0;
    }
}

/**
 * @brief Variance function V(mu)
 *
 * Calculates the variance function according to the distribution family.
 *
 * @param mu Expected value
 * @param family Distribution family
 * @return Variance function value
 */
inline double variance_function(double mu, distribution_family family)
{
    switch (family) {
        case distribution_family::gaussian:
            return 1.0;
        case distribution_family::binomial:
            mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
            return mu * (1.0 - mu);
        case distribution_family::poisson:
            return std::max(1e-10, mu);
        case distribution_family::gamma_family:
            mu = std::max(1e-10, mu);
            return mu * mu;
        default:
            return 1.0;
    }
}

/**
 * @brief Calculate deviance (for a single observation)
 *
 * Calculates the deviance residual according to the distribution family.
 *
 * @param y Observed value
 * @param mu Expected value
 * @param family Distribution family
 * @return Deviance residual
 */
inline double deviance_residual(double y, double mu, distribution_family family)
{
    switch (family) {
        case distribution_family::gaussian:
            return (y - mu) * (y - mu);
        case distribution_family::binomial:
            {
                mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
                double d = 0.0;
                if (y > 0.0) {
                    d += y * std::log(y / mu);
                }
                if (y < 1.0) {
                    d += (1.0 - y) * std::log((1.0 - y) / (1.0 - mu));
                }
                return 2.0 * d;
            }
        case distribution_family::poisson:
            {
                mu = std::max(1e-10, mu);
                if (y > 0.0) {
                    return 2.0 * (y * std::log(y / mu) - (y - mu));
                } else {
                    return 2.0 * mu;
                }
            }
        case distribution_family::gamma_family:
            {
                mu = std::max(1e-10, mu);
                return 2.0 * ((y - mu) / mu - std::log(y / mu));
            }
        default:
            return (y - mu) * (y - mu);
    }
}

/**
 * @brief Solve weighted least squares
 *
 * Computes (X'WX)^{-1} X'Wz using Cholesky decomposition.
 *
 * @param X Design matrix
 * @param z Working variable vector
 * @param w Weight vector
 * @param XtWX_inv Output: inverse of (X'WX)
 * @return Weighted least squares solution (coefficient vector)
 * @throws std::runtime_error If matrix is not positive definite
 */
inline std::vector<double> solve_weighted_least_squares(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& z,
    const std::vector<double>& w,
    std::vector<std::vector<double>>& XtWX_inv)  // Output: (X'WX)^{-1}
{
    std::size_t n = X.size();
    std::size_t p = X[0].size();

    // Calculate X'WX
    std::vector<std::vector<double>> XtWX(p, std::vector<double>(p, 0.0));
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k < p; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
                XtWX[j][k] += X[i][j] * w[i] * X[i][k];
            }
        }
    }

    // Calculate X'Wz
    std::vector<double> XtWz(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            XtWz[j] += X[i][j] * w[i] * z[i];
        }
    }

    // Solve using Cholesky decomposition
    std::vector<std::vector<double>> L(p, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double val = XtWX[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("statcpp::glm: matrix is not positive definite");
                }
                L[i][j] = std::sqrt(val);
            } else {
                L[i][j] = (XtWX[i][j] - sum) / L[j][j];
            }
        }
    }

    // Forward substitution
    std::vector<double> y(p);
    for (std::size_t i = 0; i < p; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (XtWz[i] - sum) / L[i][i];
    }

    // Back substitution
    std::vector<double> beta(p);
    for (std::size_t i = p; i > 0; --i) {
        std::size_t idx = i - 1;
        double sum = 0.0;
        for (std::size_t j = idx + 1; j < p; ++j) {
            sum += L[j][idx] * beta[j];
        }
        beta[idx] = (y[idx] - sum) / L[idx][idx];
    }

    // Calculate (X'WX)^{-1}
    XtWX_inv.assign(p, std::vector<double>(p, 0.0));
    for (std::size_t col = 0; col < p; ++col) {
        std::vector<double> e(p, 0.0);
        e[col] = 1.0;

        // Forward substitution
        std::vector<double> y_inv(p);
        for (std::size_t i = 0; i < p; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += L[i][j] * y_inv[j];
            }
            y_inv[i] = (e[i] - sum) / L[i][i];
        }

        // Back substitution
        for (std::size_t i = p; i > 0; --i) {
            std::size_t idx = i - 1;
            double sum = 0.0;
            for (std::size_t j = idx + 1; j < p; ++j) {
                sum += L[j][idx] * XtWX_inv[j][col];
            }
            XtWX_inv[idx][col] = (y_inv[idx] - sum) / L[idx][idx];
        }
    }

    return beta;
}

} // namespace detail

// ============================================================================
// IRLS Algorithm (Iteratively Reweighted Least Squares)
// ============================================================================

/**
 * @brief Fit a generalized linear model
 *
 * Estimates parameters of a generalized linear model using the
 * IRLS (Iteratively Reweighted Least Squares) algorithm.
 *
 * @param X Predictor matrix (intercept is added automatically)
 * @param y Response variable vector
 * @param family Distribution family (default: gaussian)
 * @param link Link function (default: identity)
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tol Convergence tolerance (default: 1e-8)
 * @return GLM estimation results
 * @throws std::invalid_argument If data is empty, X and Y sizes don't match,
 *         number of predictors is inconsistent, or number of observations is not greater than number of predictors
 * @note If the IRLS algorithm does not converge within max_iter iterations,
 *       converged is set to false and coefficient_se, z_statistics, and
 *       p_values may contain NaN values. NaN indicates that the corresponding
 *       estimate is undefined due to numerical issues (not a bug).
 *       Always check glm_result::converged before using these fields.
 */
inline glm_result glm_fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    distribution_family family = distribution_family::gaussian,
    link_function link = link_function::identity,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    std::size_t n = X.size();
    if (n == 0) {
        throw std::invalid_argument("statcpp::glm_fit: empty data");
    }
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::glm_fit: X and y must have same number of observations");
    }

    std::size_t p = X[0].size();
    for (const auto& row : X) {
        if (row.size() != p) {
            throw std::invalid_argument("statcpp::glm_fit: inconsistent number of predictors");
        }
    }

    std::size_t p_full = p + 1;  // Including intercept
    if (n <= p_full) {
        throw std::invalid_argument("statcpp::glm_fit: need more observations than predictors");
    }

    // Design matrix (add intercept)
    std::vector<std::vector<double>> X_design(n, std::vector<double>(p_full));
    for (std::size_t i = 0; i < n; ++i) {
        X_design[i][0] = 1.0;
        for (std::size_t j = 0; j < p; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }

    // Set initial values
    std::vector<double> mu(n);
    std::vector<double> eta(n);

    // Set initial values from mean of response variable
    double y_mean = statcpp::mean(y.begin(), y.end());
    double eta_init;

    switch (family) {
        case distribution_family::binomial:
            y_mean = std::max(0.01, std::min(0.99, y_mean));
            eta_init = detail::link_transform(y_mean, link);
            break;
        case distribution_family::poisson:
            y_mean = std::max(0.1, y_mean);
            eta_init = detail::link_transform(y_mean, link);
            break;
        default:
            eta_init = y_mean;
            break;
    }

    for (std::size_t i = 0; i < n; ++i) {
        eta[i] = eta_init;
        mu[i] = detail::inverse_link(eta[i], link);
    }

    // Initial coefficients
    std::vector<double> beta(p_full, 0.0);
    beta[0] = eta_init;

    std::vector<std::vector<double>> XtWX_inv;
    bool converged = false;
    std::size_t iter = 0;

    // IRLS iteration
    for (iter = 0; iter < max_iter; ++iter) {
        // Calculate weights and working variable
        std::vector<double> w(n);
        std::vector<double> z(n);

        for (std::size_t i = 0; i < n; ++i) {
            double var_i = detail::variance_function(mu[i], family);
            double g_prime = detail::link_derivative(mu[i], link);

            // Weight w_i = 1 / (V(mu_i) * g'(mu_i)^2)
            w[i] = 1.0 / (var_i * g_prime * g_prime);

            // Working variable z_i = eta_i + (y_i - mu_i) * g'(mu_i)
            z[i] = eta[i] + (y[i] - mu[i]) * g_prime;
        }

        // Solve weighted least squares
        std::vector<double> beta_new;
        try {
            beta_new = detail::solve_weighted_least_squares(X_design, z, w, XtWX_inv);
        } catch (const std::exception&) {
            break;  // Exit if numerically unstable
        }

        // Convergence check
        double max_change = 0.0;
        for (std::size_t j = 0; j < p_full; ++j) {
            double change = std::abs(beta_new[j] - beta[j]);
            if (std::abs(beta[j]) > 1.0) {
                change /= std::abs(beta[j]);
            }
            max_change = std::max(max_change, change);
        }

        beta = beta_new;

        // Update eta and mu
        eta = detail::matrix_vector_multiply(X_design, beta);
        for (std::size_t i = 0; i < n; ++i) {
            mu[i] = detail::inverse_link(eta[i], link);
        }

        if (max_change < tol) {
            converged = true;
            ++iter;
            break;
        }
    }

    // Calculate standard errors
    std::vector<double> coefficient_se(p_full, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> z_statistics(p_full, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> p_values(p_full, std::numeric_limits<double>::quiet_NaN());

    if (!XtWX_inv.empty()) {
        for (std::size_t j = 0; j < p_full; ++j) {
            coefficient_se[j] = std::sqrt(XtWX_inv[j][j]);
        }

        // z-statistics and p-values
        for (std::size_t j = 0; j < p_full; ++j) {
            z_statistics[j] = beta[j] / coefficient_se[j];
            p_values[j] = 2.0 * (1.0 - norm_cdf(std::abs(z_statistics[j])));
        }
    }

    // Calculate deviance
    double residual_deviance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        residual_deviance += detail::deviance_residual(y[i], mu[i], family);
    }

    // Null deviance (intercept-only model)
    double null_deviance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        null_deviance += detail::deviance_residual(y[i], y_mean, family);
    }

    // Calculate log-likelihood
    double log_likelihood = 0.0;
    switch (family) {
        case distribution_family::gaussian:
            {
                double sigma2 = residual_deviance / static_cast<double>(n);
                log_likelihood = -0.5 * static_cast<double>(n) *
                    (std::log(2.0 * pi) + std::log(sigma2) + 1.0);
            }
            break;
        case distribution_family::binomial:
            for (std::size_t i = 0; i < n; ++i) {
                double mu_i = std::max(1e-10, std::min(1.0 - 1e-10, mu[i]));
                if (y[i] > 0.0) {
                    log_likelihood += y[i] * std::log(mu_i);
                }
                if (y[i] < 1.0) {
                    log_likelihood += (1.0 - y[i]) * std::log(1.0 - mu_i);
                }
            }
            break;
        case distribution_family::poisson:
            for (std::size_t i = 0; i < n; ++i) {
                log_likelihood += y[i] * std::log(std::max(1e-10, mu[i])) - mu[i]
                                  - std::lgamma(y[i] + 1.0);
            }
            break;
        default:
            log_likelihood = -0.5 * residual_deviance;
            break;
    }

    // AIC and BIC
    double n_d = static_cast<double>(n);
    double k = static_cast<double>(p_full);
    double aic = -2.0 * log_likelihood + 2.0 * k;
    double bic = -2.0 * log_likelihood + k * std::log(n_d);

    return {
        beta, coefficient_se, z_statistics, p_values,
        null_deviance, residual_deviance,
        static_cast<double>(n - 1), static_cast<double>(n - p_full),
        aic, bic, log_likelihood,
        iter, converged,
        link, family
    };
}

// ============================================================================
// Logistic Regression
// ============================================================================

/**
 * @brief Logistic regression
 *
 * Fits a generalized linear model using binomial distribution and logit link function.
 *
 * @param X Predictor matrix (intercept is added automatically)
 * @param y Response variable vector (range 0 to 1)
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tol Convergence tolerance (default: 1e-8)
 * @return GLM estimation results
 * @throws std::invalid_argument If y is outside [0,1] range or X contains an intercept column
 */
inline glm_result logistic_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    // Verify that X does not contain an intercept column (from linear_regression.hpp)
    statcpp::detail::validate_no_intercept_column(X, "logistic_regression");

    // Verify that y is in [0, 1] range
    for (double yi : y) {
        if (yi < 0.0 || yi > 1.0) {
            throw std::invalid_argument("statcpp::logistic_regression: y must be in [0, 1]");
        }
    }

    return glm_fit(X, y, distribution_family::binomial, link_function::logit, max_iter, tol);
}

/**
 * @brief Probability prediction with logistic regression
 *
 * Uses a fitted logistic regression model to predict probabilities
 * for new data points.
 *
 * @param model Fitted GLM model (binomial distribution)
 * @param x Predictor variable vector
 * @return Predicted probability (range 0 to 1)
 * @throws std::invalid_argument If model is not binomial or x dimension doesn't match
 */
inline double predict_probability(const glm_result& model, const std::vector<double>& x)
{
    if (model.family != distribution_family::binomial) {
        throw std::invalid_argument("statcpp::predict_probability: model must be binomial");
    }
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict_probability: x dimension mismatch");
    }

    double eta = model.coefficients[0];
    for (std::size_t i = 0; i < x.size(); ++i) {
        eta += model.coefficients[i + 1] * x[i];
    }

    return detail::inverse_link(eta, model.link);
}

/**
 * @brief Calculate odds ratios
 *
 * Calculates odds ratios from logistic regression model coefficients.
 *
 * @param model Fitted logistic regression model
 * @return Vector of odds ratios for each predictor (excluding intercept)
 * @throws std::invalid_argument If model is not logistic regression
 */
inline std::vector<double> odds_ratios(const glm_result& model)
{
    if (model.family != distribution_family::binomial || model.link != link_function::logit) {
        throw std::invalid_argument("statcpp::odds_ratios: requires logistic regression model");
    }

    std::vector<double> or_values(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        or_values[i - 1] = std::exp(model.coefficients[i]);
    }
    return or_values;
}

/**
 * @brief Confidence intervals for odds ratios
 *
 * Calculates confidence intervals for odds ratios from a logistic regression model.
 *
 * @param model Fitted logistic regression model
 * @param confidence Confidence level (default: 0.95)
 * @return Vector of (lower, upper) pairs for each predictor's odds ratio
 * @throws std::invalid_argument If model is not logistic regression or confidence level is outside (0,1)
 */
inline std::vector<std::pair<double, double>> odds_ratios_ci(
    const glm_result& model, double confidence = 0.95)
{
    if (model.family != distribution_family::binomial || model.link != link_function::logit) {
        throw std::invalid_argument("statcpp::odds_ratios_ci: requires logistic regression model");
    }
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("statcpp::odds_ratios_ci: confidence must be in (0, 1)");
    }

    double z = norm_quantile(1.0 - (1.0 - confidence) / 2.0);

    std::vector<std::pair<double, double>> ci(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        double beta = model.coefficients[i];
        double se = model.coefficient_se[i];
        double lower = std::exp(beta - z * se);
        double upper = std::exp(beta + z * se);
        ci[i - 1] = {lower, upper};
    }
    return ci;
}

// ============================================================================
// Poisson Regression
// ============================================================================

/**
 * @brief Poisson regression
 *
 * Fits a generalized linear model using Poisson distribution and log link function.
 * Used for regression analysis of count data.
 *
 * @param X Predictor matrix (intercept is added automatically)
 * @param y Response variable vector (non-negative count data)
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tol Convergence tolerance (default: 1e-8)
 * @return GLM estimation results
 * @throws std::invalid_argument If y is negative or X contains an intercept column
 */
inline glm_result poisson_regression(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::size_t max_iter = 100,
    double tol = 1e-8)
{
    // Verify that X does not contain an intercept column
    statcpp::detail::validate_no_intercept_column(X, "poisson_regression");

    // Verify that y is non-negative
    for (double yi : y) {
        if (yi < 0.0) {
            throw std::invalid_argument("statcpp::poisson_regression: y must be non-negative");
        }
    }

    return glm_fit(X, y, distribution_family::poisson, link_function::log, max_iter, tol);
}

/**
 * @brief Expected count prediction with Poisson regression
 *
 * Uses a fitted Poisson regression model to predict expected counts
 * for new data points.
 *
 * @param model Fitted GLM model (Poisson distribution)
 * @param x Predictor variable vector
 * @return Predicted expected count
 * @throws std::invalid_argument If model is not Poisson or x dimension doesn't match
 */
inline double predict_count(const glm_result& model, const std::vector<double>& x)
{
    if (model.family != distribution_family::poisson) {
        throw std::invalid_argument("statcpp::predict_count: model must be Poisson");
    }
    if (x.size() + 1 != model.coefficients.size()) {
        throw std::invalid_argument("statcpp::predict_count: x dimension mismatch");
    }

    double eta = model.coefficients[0];
    for (std::size_t i = 0; i < x.size(); ++i) {
        eta += model.coefficients[i + 1] * x[i];
    }

    return detail::inverse_link(eta, model.link);
}

/**
 * @brief Calculate Incidence Rate Ratios
 *
 * Calculates incidence rate ratios from Poisson regression model coefficients.
 *
 * @param model Fitted Poisson regression model
 * @return Vector of incidence rate ratios for each predictor (excluding intercept)
 * @throws std::invalid_argument If model is not Poisson regression
 */
inline std::vector<double> incidence_rate_ratios(const glm_result& model)
{
    if (model.family != distribution_family::poisson || model.link != link_function::log) {
        throw std::invalid_argument("statcpp::incidence_rate_ratios: requires Poisson regression model");
    }

    std::vector<double> irr(model.coefficients.size() - 1);
    for (std::size_t i = 1; i < model.coefficients.size(); ++i) {
        irr[i - 1] = std::exp(model.coefficients[i]);
    }
    return irr;
}

// ============================================================================
// GLM Diagnostics
// ============================================================================

/**
 * @brief GLM residuals structure
 *
 * Stores various types of residuals for generalized linear models.
 */
struct glm_residuals {
    std::vector<double> response;       ///< Response residuals (y - mu)
    std::vector<double> pearson;        ///< Pearson residuals
    std::vector<double> deviance;       ///< Deviance residuals
    std::vector<double> working;        ///< Working residuals
};

/**
 * @brief Calculate GLM residuals
 *
 * Calculates various types of residuals from a fitted GLM model.
 *
 * @param model Fitted GLM model
 * @param X Predictor matrix
 * @param y Response variable vector
 * @return Structure containing various residuals
 * @throws std::invalid_argument If X and Y sizes don't match
 */
inline glm_residuals compute_glm_residuals(
    const glm_result& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y)
{
    std::size_t n = X.size();
    if (n != y.size()) {
        throw std::invalid_argument("statcpp::compute_glm_residuals: X and y must have same length");
    }

    std::vector<double> response(n);
    std::vector<double> pearson(n);
    std::vector<double> deviance_res(n);
    std::vector<double> working(n);

    for (std::size_t i = 0; i < n; ++i) {
        // Calculate linear predictor
        double eta = model.coefficients[0];
        for (std::size_t j = 0; j < X[i].size(); ++j) {
            eta += model.coefficients[j + 1] * X[i][j];
        }

        double mu = detail::inverse_link(eta, model.link);

        // Response residuals
        response[i] = y[i] - mu;

        // Pearson residuals
        double var = detail::variance_function(mu, model.family);
        pearson[i] = response[i] / std::sqrt(var);

        // Deviance residuals
        double d = detail::deviance_residual(y[i], mu, model.family);
        int sign = (y[i] >= mu) ? 1 : -1;
        deviance_res[i] = sign * std::sqrt(d);

        // Working residuals
        double g_prime = detail::link_derivative(mu, model.link);
        working[i] = response[i] * g_prime;
    }

    return {response, pearson, deviance_res, working};
}

/**
 * @brief Overdispersion test (for Poisson regression)
 *
 * Calculates the overdispersion parameter for a Poisson regression model.
 * Values greater than 1 suggest the presence of overdispersion.
 *
 * @param model Fitted Poisson regression model
 * @param X Predictor matrix
 * @param y Response variable vector
 * @return Overdispersion parameter (Pearson chi-square statistic / residual degrees of freedom)
 * @throws std::invalid_argument If model is not Poisson
 */
inline double overdispersion_test(const glm_result& model,
                                   const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y)
{
    if (model.family != distribution_family::poisson) {
        throw std::invalid_argument("statcpp::overdispersion_test: requires Poisson model");
    }

    auto residuals = compute_glm_residuals(model, X, y);

    // Pearson chi-square statistic
    double pearson_chi2 = 0.0;
    for (double r : residuals.pearson) {
        pearson_chi2 += r * r;
    }

    // Variance estimate (overdispersion parameter)
    double dispersion = pearson_chi2 / model.df_residual;

    return dispersion;
}

/**
 * @brief McFadden's pseudo R-squared
 *
 * Calculates McFadden's pseudo R-squared.
 * Defined as 1 - (residual deviance / null deviance).
 *
 * @param model Fitted GLM model
 * @return McFadden's pseudo R-squared
 */
inline double pseudo_r_squared_mcfadden(const glm_result& model)
{
    return 1.0 - (model.residual_deviance / model.null_deviance);
}

/**
 * @brief Nagelkerke's pseudo R-squared
 *
 * Calculates Nagelkerke's pseudo R-squared.
 * Cox-Snell pseudo R-squared adjusted to have a maximum value of 1.
 *
 * Uses the relationship: deviance = -2 * (LL_model - LL_saturated),
 * so LL_null = LL_saturated - null_deviance / 2.
 * For binomial (0/1 responses) and Poisson, the saturated model
 * log-likelihood is computed explicitly.
 *
 * @param model Fitted GLM model
 * @param y Response variable vector (needed to compute saturated LL for non-Gaussian)
 * @param n Sample size
 * @return Nagelkerke's pseudo R-squared
 */
inline double pseudo_r_squared_nagelkerke(const glm_result& model,
                                           const std::vector<double>& y,
                                           std::size_t n)
{
    double n_d = static_cast<double>(n);
    double ll_model = model.log_likelihood;

    // Compute saturated model log-likelihood
    double ll_saturated = 0.0;
    switch (model.family) {
        case distribution_family::gaussian:
            // For Gaussian, saturated LL = 0 (perfect fit, residual = 0)
            // Not exactly 0 but the deviance relationship gives us:
            // ll_null = ll_saturated - null_deviance/2
            // For Gaussian with MLE sigma^2: ll_saturated ≈ -n/2*(log(2*pi) + 1) when sigma -> 0
            // Use the simpler relationship directly
            break;
        case distribution_family::binomial:
            // Saturated LL for 0/1 data is 0
            ll_saturated = 0.0;
            break;
        case distribution_family::poisson:
            for (std::size_t i = 0; i < n; ++i) {
                if (y[i] > 0.0) {
                    ll_saturated += y[i] * std::log(y[i]) - y[i] - std::lgamma(y[i] + 1.0);
                }
                // y[i] == 0 contributes 0
            }
            break;
        default:
            break;
    }

    double ll_null;
    if (model.family == distribution_family::gaussian) {
        // For Gaussian: use -null_deviance/2 as approximation (exact for Gaussian)
        ll_null = -model.null_deviance / 2.0;
    } else {
        ll_null = ll_saturated - model.null_deviance / 2.0;
    }

    double r2_cox_snell = 1.0 - std::exp(2.0 * (ll_null - ll_model) / n_d);
    double r2_max = 1.0 - std::exp(2.0 * ll_null / n_d);

    if (r2_max == 0.0) return 0.0;

    return r2_cox_snell / r2_max;
}

} // namespace statcpp
