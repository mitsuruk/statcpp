/**
 * @file distance_metrics.hpp
 * @brief Distance and similarity metrics
 */

#ifndef STATCPP_DISTANCE_METRICS_HPP
#define STATCPP_DISTANCE_METRICS_HPP

#include <cmath>
#include <stdexcept>
#include <iterator>
#include <vector>

namespace statcpp {

/**
 * @brief Euclidean distance (L2 norm)
 *
 * Computes the Euclidean distance between two points in n-dimensional space:
 * d(x, y) = sqrt(sum((x_i - y_i)^2))
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @return Euclidean distance
 * @throw std::invalid_argument if sequences have different lengths
 */
template <typename Iterator1, typename Iterator2>
double euclidean_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::euclidean_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::euclidean_distance: empty sequences");
    }

    double sum_sq = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
        sum_sq += diff * diff;
        ++it1;
        ++it2;
    }

    return std::sqrt(sum_sq);
}

/**
 * @brief Euclidean distance with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double euclidean_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::euclidean_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::euclidean_distance: empty sequences");
    }

    double sum_sq = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = v1 - v2;
        sum_sq += diff * diff;
        ++it1;
        ++it2;
    }

    return std::sqrt(sum_sq);
}

/**
 * @brief Manhattan distance (L1 norm, taxicab distance)
 *
 * Computes the Manhattan distance between two points:
 * d(x, y) = sum(|x_i - y_i|)
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @return Manhattan distance
 * @throw std::invalid_argument if sequences have different lengths
 */
template <typename Iterator1, typename Iterator2>
double manhattan_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::manhattan_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::manhattan_distance: empty sequences");
    }

    double sum_abs = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
        sum_abs += std::abs(diff);
        ++it1;
        ++it2;
    }

    return sum_abs;
}

/**
 * @brief Manhattan distance with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double manhattan_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::manhattan_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::manhattan_distance: empty sequences");
    }

    double sum_abs = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = v1 - v2;
        sum_abs += std::abs(diff);
        ++it1;
        ++it2;
    }

    return sum_abs;
}

/**
 * @brief Cosine similarity
 *
 * Computes the cosine of the angle between two vectors:
 * similarity(x, y) = (x · y) / (||x|| * ||y||)
 *
 * Returns a value in [-1, 1]:
 * - 1: vectors point in the same direction
 * - 0: vectors are orthogonal
 * - -1: vectors point in opposite directions
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @return Cosine similarity
 * @throw std::invalid_argument if sequences have different lengths or if either vector has zero norm
 */
template <typename Iterator1, typename Iterator2>
double cosine_similarity(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::cosine_similarity: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::cosine_similarity: empty sequences");
    }

    double dot_product = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;

    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(*it1);
        double v2 = static_cast<double>(*it2);

        dot_product += v1 * v2;
        norm1_sq += v1 * v1;
        norm2_sq += v2 * v2;

        ++it1;
        ++it2;
    }

    if (norm1_sq == 0.0 || norm2_sq == 0.0) {
        throw std::invalid_argument("statcpp::cosine_similarity: zero vector encountered");
    }

    return dot_product / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
}

/**
 * @brief Cosine similarity with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double cosine_similarity(Iterator1 first1, Iterator1 last1,
                         Iterator2 first2, Iterator2 last2,
                         Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::cosine_similarity: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::cosine_similarity: empty sequences");
    }

    double dot_product = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;

    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));

        dot_product += v1 * v2;
        norm1_sq += v1 * v1;
        norm2_sq += v2 * v2;

        ++it1;
        ++it2;
    }

    if (norm1_sq == 0.0 || norm2_sq == 0.0) {
        throw std::invalid_argument("statcpp::cosine_similarity: zero vector encountered");
    }

    return dot_product / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
}

/**
 * @brief Cosine distance
 *
 * Defined as: distance = 1 - similarity
 * Returns a value in [0, 2], where 0 means identical direction
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @return Cosine distance
 */
template <typename Iterator1, typename Iterator2>
double cosine_distance(Iterator1 first1, Iterator1 last1,
                       Iterator2 first2, Iterator2 last2)
{
    return 1.0 - cosine_similarity(first1, last1, first2, last2);
}

/**
 * @brief Cosine distance with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double cosine_distance(Iterator1 first1, Iterator1 last1,
                       Iterator2 first2, Iterator2 last2,
                       Proj1 proj1, Proj2 proj2)
{
    return 1.0 - cosine_similarity(first1, last1, first2, last2, proj1, proj2);
}

/**
 * @brief Mahalanobis distance
 *
 * Computes the Mahalanobis distance between a point and a distribution.
 * This accounts for correlations in the data and is scale-invariant.
 *
 * For a point x and a distribution with mean μ and covariance matrix Σ:
 * d(x, μ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
 *
 * Current implementation supports 2-dimensional data only.
 *
 * @param x Point to measure distance from
 * @param mean Mean of the distribution
 * @param cov_matrix Covariance matrix (2x2)
 * @return Mahalanobis distance
 * @throw std::invalid_argument if dimensions mismatch or covariance matrix is singular
 */
inline double mahalanobis_distance(const std::vector<double>& x,
                                   const std::vector<double>& mean,
                                   const std::vector<std::vector<double>>& cov_matrix)
{
    if (x.size() != mean.size()) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: x and mean must have same dimension");
    }
    if (x.size() != 2) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: only 2D is currently supported");
    }
    if (cov_matrix.size() != 2 || cov_matrix[0].size() != 2 || cov_matrix[1].size() != 2) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: covariance matrix must be 2x2");
    }

    // Compute difference vector: d = x - mean
    double d0 = x[0] - mean[0];
    double d1 = x[1] - mean[1];

    // Compute inverse of 2x2 covariance matrix
    // Σ = [a b]
    //     [c d]
    // Σ⁻¹ = (1/det) * [d  -b]
    //                 [-c  a]

    double a = cov_matrix[0][0];
    double b = cov_matrix[0][1];
    double c = cov_matrix[1][0];
    double d = cov_matrix[1][1];

    double det = a * d - b * c;

    if (std::abs(det) < 1e-10) {
        throw std::invalid_argument("statcpp::mahalanobis_distance: covariance matrix is singular");
    }

    // Inverse matrix elements
    double inv_a = d / det;
    double inv_b = -b / det;
    double inv_c = -c / det;
    double inv_d = a / det;

    // Compute dᵀ Σ⁻¹ d
    // First compute Σ⁻¹ d
    double temp0 = inv_a * d0 + inv_b * d1;
    double temp1 = inv_c * d0 + inv_d * d1;

    // Then compute dᵀ (Σ⁻¹ d)
    double mahalanobis_sq = d0 * temp0 + d1 * temp1;

    if (mahalanobis_sq < 0.0) {
        // Due to numerical errors, ensure non-negative
        mahalanobis_sq = 0.0;
    }

    return std::sqrt(mahalanobis_sq);
}

/**
 * @brief Minkowski distance (generalized Lp distance)
 *
 * Computes the Minkowski distance with parameter p:
 * d(x, y) = (sum(|x_i - y_i|^p))^(1/p)
 *
 * Special cases:
 * - p = 1: Manhattan distance
 * - p = 2: Euclidean distance
 * - p → ∞: Chebyshev distance (max |x_i - y_i|)
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @param p The order of the Minkowski distance (must be >= 1)
 * @return Minkowski distance
 * @throw std::invalid_argument if sequences have different lengths or p < 1
 */
template <typename Iterator1, typename Iterator2>
double minkowski_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          double p)
{
    if (p < 1.0) {
        throw std::invalid_argument("statcpp::minkowski_distance: p must be >= 1");
    }

    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::minkowski_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::minkowski_distance: empty sequences");
    }

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = std::abs(static_cast<double>(*it1) - static_cast<double>(*it2));
        sum += std::pow(diff, p);
        ++it1;
        ++it2;
    }

    return std::pow(sum, 1.0 / p);
}

/**
 * @brief Minkowski distance with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double minkowski_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          double p, Proj1 proj1, Proj2 proj2)
{
    if (p < 1.0) {
        throw std::invalid_argument("statcpp::minkowski_distance: p must be >= 1");
    }

    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::minkowski_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::minkowski_distance: empty sequences");
    }

    double sum = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = std::abs(v1 - v2);
        sum += std::pow(diff, p);
        ++it1;
        ++it2;
    }

    return std::pow(sum, 1.0 / p);
}

/**
 * @brief Chebyshev distance (L∞ norm, maximum metric)
 *
 * Computes the maximum absolute difference:
 * d(x, y) = max(|x_i - y_i|)
 *
 * @tparam Iterator1 RandomAccessIterator type for first sequence
 * @tparam Iterator2 RandomAccessIterator type for second sequence
 * @param first1 Beginning of first sequence
 * @param last1 End of first sequence
 * @param first2 Beginning of second sequence
 * @param last2 End of second sequence
 * @return Chebyshev distance
 * @throw std::invalid_argument if sequences have different lengths
 */
template <typename Iterator1, typename Iterator2>
double chebyshev_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::chebyshev_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::chebyshev_distance: empty sequences");
    }

    double max_diff = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double diff = std::abs(static_cast<double>(*it1) - static_cast<double>(*it2));
        if (diff > max_diff) {
            max_diff = diff;
        }
        ++it1;
        ++it2;
    }

    return max_diff;
}

/**
 * @brief Chebyshev distance with projection
 */
template <typename Iterator1, typename Iterator2, typename Proj1, typename Proj2>
double chebyshev_distance(Iterator1 first1, Iterator1 last1,
                          Iterator2 first2, Iterator2 last2,
                          Proj1 proj1, Proj2 proj2)
{
    auto n1 = std::distance(first1, last1);
    auto n2 = std::distance(first2, last2);

    if (n1 != n2) {
        throw std::invalid_argument("statcpp::chebyshev_distance: sequences must have the same length");
    }
    if (n1 == 0) {
        throw std::invalid_argument("statcpp::chebyshev_distance: empty sequences");
    }

    double max_diff = 0.0;
    auto it1 = first1;
    auto it2 = first2;

    while (it1 != last1) {
        double v1 = static_cast<double>(proj1(*it1));
        double v2 = static_cast<double>(proj2(*it2));
        double diff = std::abs(v1 - v2);
        if (diff > max_diff) {
            max_diff = diff;
        }
        ++it1;
        ++it2;
    }

    return max_diff;
}

} // namespace statcpp

#endif // STATCPP_DISTANCE_METRICS_HPP
