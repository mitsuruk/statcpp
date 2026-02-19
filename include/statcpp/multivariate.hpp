/**
 * @file multivariate.hpp
 * @brief Multivariate analysis functions
 *
 * Provides functions for multivariate data analysis including covariance matrices,
 * correlation matrices, principal component analysis (PCA), and data standardization.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "statcpp/linear_regression.hpp"  // for detail::validate_matrix_structure

namespace statcpp {

// ============================================================================
// Covariance Matrix
// ============================================================================

/**
 * @brief Calculate sample covariance matrix
 *
 * Calculates the covariance matrix for multivariate data.
 * Data is passed with rows=observations and columns=variables.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @return Covariance matrix (p x p)
 * @throws std::invalid_argument If data is empty, rows have different column counts, or there are fewer than 2 observations
 */
inline std::vector<std::vector<double>> covariance_matrix(
    const std::vector<std::vector<double>>& data)
{
    // Validate matrix structure (empty check and column count consistency)
    detail::validate_matrix_structure(data, "covariance_matrix");

    std::size_t n = data.size();      // Number of observations
    std::size_t p = data[0].size();   // Number of variables

    if (n < 2) {
        throw std::invalid_argument("statcpp::covariance_matrix: need at least 2 observations");
    }

    // Calculate mean for each variable
    std::vector<double> means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);
    }

    // Calculate covariance matrix
    std::vector<std::vector<double>> cov(p, std::vector<double>(p, 0.0));
    for (std::size_t j1 = 0; j1 < p; ++j1) {
        for (std::size_t j2 = j1; j2 < p; ++j2) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += (data[i][j1] - means[j1]) * (data[i][j2] - means[j2]);
            }
            cov[j1][j2] = sum / static_cast<double>(n - 1);
            cov[j2][j1] = cov[j1][j2];  // Symmetric matrix
        }
    }

    return cov;
}

// ============================================================================
// Correlation Matrix
// ============================================================================

/**
 * @brief Calculate Pearson correlation matrix
 *
 * Calculates the correlation matrix for multivariate data.
 * Standardizes the covariance matrix to obtain correlation coefficients.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @return Correlation matrix (p x p)
 * @throws std::invalid_argument If data is empty, rows have different column counts, there are fewer than 2 observations, or any variable has zero variance
 */
inline std::vector<std::vector<double>> correlation_matrix(
    const std::vector<std::vector<double>>& data)
{
    // Validate matrix structure
    detail::validate_matrix_structure(data, "correlation_matrix");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    if (n < 2) {
        throw std::invalid_argument("statcpp::correlation_matrix: need at least 2 observations");
    }

    // Calculate covariance matrix
    auto cov = covariance_matrix(data);

    // Calculate standard deviations
    std::vector<double> stddevs(p);
    for (std::size_t j = 0; j < p; ++j) {
        stddevs[j] = std::sqrt(cov[j][j]);
        if (stddevs[j] == 0.0) {
            throw std::invalid_argument("statcpp::correlation_matrix: zero variance variable");
        }
    }

    // Convert to correlation matrix
    std::vector<std::vector<double>> corr(p, std::vector<double>(p, 0.0));
    for (std::size_t j1 = 0; j1 < p; ++j1) {
        for (std::size_t j2 = 0; j2 < p; ++j2) {
            corr[j1][j2] = cov[j1][j2] / (stddevs[j1] * stddevs[j2]);
        }
    }

    return corr;
}

// ============================================================================
// Standardization (Z-score)
// ============================================================================

/**
 * @brief Z-score standardization
 *
 * Standardizes each variable to mean 0 and standard deviation 1.
 * Data is passed with rows=observations and columns=variables.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @return Standardized data
 * @throws std::invalid_argument If data is empty, rows have different column counts, there are fewer than 2 observations, or any variable has zero variance
 */
inline std::vector<std::vector<double>> standardize(
    const std::vector<std::vector<double>>& data)
{
    // Validate matrix structure
    detail::validate_matrix_structure(data, "standardize");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    if (n < 2) {
        throw std::invalid_argument("statcpp::standardize: need at least 2 observations");
    }

    // Calculate mean and standard deviation for each variable
    std::vector<double> means(p, 0.0);
    std::vector<double> stddevs(p, 0.0);

    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);

        for (std::size_t i = 0; i < n; ++i) {
            double diff = data[i][j] - means[j];
            stddevs[j] += diff * diff;
        }
        stddevs[j] = std::sqrt(stddevs[j] / static_cast<double>(n - 1));

        if (stddevs[j] == 0.0) {
            throw std::invalid_argument("statcpp::standardize: zero variance variable");
        }
    }

    // Standardize
    std::vector<std::vector<double>> result(n, std::vector<double>(p));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            result[i][j] = (data[i][j] - means[j]) / stddevs[j];
        }
    }

    return result;
}

// ============================================================================
// Min-Max Scaling
// ============================================================================

/**
 * @brief Min-Max normalization (0-1 scaling)
 *
 * Scales each variable to the [0, 1] range.
 * Data is passed with rows=observations and columns=variables.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @return Scaled data
 * @throws std::invalid_argument If data is empty, rows have different column counts, or any variable has zero range
 */
inline std::vector<std::vector<double>> min_max_scale(
    const std::vector<std::vector<double>>& data)
{
    // Validate matrix structure
    detail::validate_matrix_structure(data, "min_max_scale");

    std::size_t n = data.size();
    std::size_t p = data[0].size();

    // Calculate minimum and maximum for each variable
    std::vector<double> mins(p, std::numeric_limits<double>::max());
    std::vector<double> maxs(p, std::numeric_limits<double>::lowest());

    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            mins[j] = std::min(mins[j], data[i][j]);
            maxs[j] = std::max(maxs[j], data[i][j]);
        }

        if (maxs[j] == mins[j]) {
            throw std::invalid_argument("statcpp::min_max_scale: zero range variable");
        }
    }

    // Scale
    std::vector<std::vector<double>> result(n, std::vector<double>(p));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            result[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
        }
    }

    return result;
}

// ============================================================================
// Principal Component Analysis (PCA)
// ============================================================================

/**
 * @brief PCA result
 *
 * Holds the results of principal component analysis.
 */
struct pca_result {
    std::vector<std::vector<double>> components;  ///< Principal components (eigenvectors) p x n_components
    std::vector<double> explained_variance;       ///< Explained variance (eigenvalues)
    std::vector<double> explained_variance_ratio; ///< Variance explained ratio
};

/**
 * @brief Find largest eigenvalue and eigenvector using power iteration
 *
 * Finds the largest eigenvalue and corresponding eigenvector of a matrix
 * through iterative computation.
 *
 * @param matrix Symmetric matrix
 * @param max_iter Maximum number of iterations (default: 1000)
 * @param tol Convergence threshold (default: 1e-10)
 * @return Pair of (eigenvalue, eigenvector)
 */
inline std::pair<double, std::vector<double>> power_iteration(
    const std::vector<std::vector<double>>& matrix,
    std::size_t max_iter = 1000,
    double tol = 1e-10)
{
    std::size_t n = matrix.size();
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));

    double eigenvalue = 0.0;

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Calculate Av
        std::vector<double> Av(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Av[i] += matrix[i][j] * v[j];
            }
        }

        // Calculate norm
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            norm += Av[i] * Av[i];
        }
        norm = std::sqrt(norm);

        if (norm == 0.0) break;

        // Estimate eigenvalue
        double new_eigenvalue = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            new_eigenvalue += v[i] * Av[i];
        }

        // Normalize
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = Av[i] / norm;
        }

        // Convergence check
        if (std::abs(new_eigenvalue - eigenvalue) < tol) {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    return {eigenvalue, v};
}

/**
 * @brief Principal Component Analysis
 *
 * Finds principal components through eigenvalue decomposition of the covariance matrix.
 * Uses deflation method to compute multiple principal components.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @param n_components Number of principal components to extract
 * @return PCA result
 * @throws std::invalid_argument If data is empty or rows have different column counts
 */
inline pca_result pca(const std::vector<std::vector<double>>& data,
                      std::size_t n_components)
{
    // Validate matrix structure
    detail::validate_matrix_structure(data, "pca");

    std::size_t p = data[0].size();

    if (n_components > p) {
        n_components = p;
    }

    // Calculate covariance matrix
    auto cov = covariance_matrix(data);

    pca_result result;
    result.components.resize(p, std::vector<double>(n_components));
    result.explained_variance.resize(n_components);
    result.explained_variance_ratio.resize(n_components);

    // Total variance (trace)
    double total_variance = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        total_variance += cov[j][j];
    }

    // Calculate each principal component (deflation method)
    auto working_cov = cov;

    for (std::size_t k = 0; k < n_components; ++k) {
        auto [eigenvalue, eigenvector] = power_iteration(working_cov);

        result.explained_variance[k] = eigenvalue;
        result.explained_variance_ratio[k] = eigenvalue / total_variance;

        for (std::size_t j = 0; j < p; ++j) {
            result.components[j][k] = eigenvector[j];
        }

        // Deflation: remove processed eigenvalue component
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                working_cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    return result;
}

/**
 * @brief Project data onto principal component space
 *
 * Transforms data to principal component space using PCA results.
 *
 * @param data Matrix data (rows=observations, columns=variables)
 * @param pca PCA result
 * @return Data projected onto principal component space (n x n_components)
 * @throws std::invalid_argument If data is empty, PCA components are empty, or dimensions don't match
 */
inline std::vector<std::vector<double>> pca_transform(
    const std::vector<std::vector<double>>& data,
    const pca_result& pca)
{
    // Validate matrix structure
    detail::validate_matrix_structure(data, "pca_transform");

    // Also validate PCA components structure
    if (pca.components.empty() || pca.components[0].empty()) {
        throw std::invalid_argument("statcpp::pca_transform: pca.components is empty");
    }

    std::size_t n = data.size();
    std::size_t p = data[0].size();
    std::size_t n_components = pca.components[0].size();

    // Check that data and PCA components dimensions match
    if (p != pca.components.size()) {
        throw std::invalid_argument("statcpp::pca_transform: dimension mismatch between data and pca.components");
    }

    // Calculate means
    std::vector<double> means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= static_cast<double>(n);
    }

    // Project
    std::vector<std::vector<double>> result(n, std::vector<double>(n_components, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n_components; ++k) {
            for (std::size_t j = 0; j < p; ++j) {
                result[i][k] += (data[i][j] - means[j]) * pca.components[j][k];
            }
        }
    }

    return result;
}

} // namespace statcpp
