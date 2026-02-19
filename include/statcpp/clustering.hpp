/**
 * @file clustering.hpp
 * @brief Clustering algorithms implementation
 *
 * Provides K-means, hierarchical clustering, silhouette score, and related algorithms.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "statcpp/random_engine.hpp"

namespace statcpp {

// ============================================================================
// Distance Functions
// ============================================================================

/**
 * @brief Euclidean distance
 *
 * Computes the Euclidean distance between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @return Euclidean distance
 * @throws std::invalid_argument If vector dimensions mismatch
 */
inline double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("statcpp::euclidean_distance: dimension mismatch");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Manhattan distance
 *
 * Computes the Manhattan distance between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @return Manhattan distance
 * @throws std::invalid_argument If vector dimensions mismatch
 */
inline double manhattan_distance(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("statcpp::manhattan_distance: dimension mismatch");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

// ============================================================================
// K-means Clustering
// ============================================================================

/**
 * @brief K-means clustering result
 */
struct kmeans_result {
    std::vector<std::size_t> labels;              ///< Cluster assignments
    std::vector<std::vector<double>> centroids;   ///< Cluster centroids
    double inertia;                               ///< Inertia (within-cluster sum of squares)
    std::size_t n_iter;                           ///< Number of iterations until convergence
};

/**
 * @brief K-means++ initialization
 *
 * Selects initial cluster centroids using the K-means++ algorithm.
 *
 * @param data Vector of data points
 * @param k Number of clusters
 * @return Initial cluster centroids
 *
 * @note Arthur & Vassilvitskii (2007) "k-means++: the advantages of careful seeding"
 */
inline std::vector<std::vector<double>> kmeans_plusplus_init(
    const std::vector<std::vector<double>>& data,
    std::size_t k)
{
    std::size_t n = data.size();

    std::vector<std::vector<double>> centroids;
    centroids.reserve(k);

    auto& rng = get_random_engine();
    std::uniform_int_distribution<std::size_t> init_dist(0, n - 1);

    // Randomly select the first centroid
    centroids.push_back(data[init_dist(rng)]);

    // Probabilistically select remaining centroids
    std::vector<double> distances(n);

    for (std::size_t c = 1; c < k; ++c) {
        double total_dist = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& centroid : centroids) {
                double dist = euclidean_distance(data[i], centroid);
                min_dist = std::min(min_dist, dist);
            }
            distances[i] = min_dist * min_dist;
            total_dist += distances[i];
        }

        // Probabilistically select the next centroid
        std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
        double threshold = prob_dist(rng);
        double cumsum = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            cumsum += distances[i];
            if (cumsum >= threshold) {
                centroids.push_back(data[i]);
                break;
            }
        }
    }

    return centroids;
}

/**
 * @brief K-means clustering
 *
 * Performs clustering using the K-means algorithm.
 *
 * @param data Vector of data points
 * @param k Number of clusters
 * @param max_iter Maximum number of iterations (default: 100)
 * @param tol Convergence tolerance (default: 1e-6)
 * @return Clustering result
 * @throws std::invalid_argument If data is empty, k is 0, or k exceeds number of data points
 */
inline kmeans_result kmeans(
    const std::vector<std::vector<double>>& data,
    std::size_t k,
    std::size_t max_iter = 100,
    double tol = 1e-6)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::kmeans: empty data");
    }
    if (k == 0) {
        throw std::invalid_argument("statcpp::kmeans: k must be positive");
    }
    if (k > data.size()) {
        throw std::invalid_argument("statcpp::kmeans: k exceeds number of data points");
    }

    std::size_t n = data.size();
    std::size_t dim = data[0].size();

    // K-means++ initialization
    auto centroids = kmeans_plusplus_init(data, k);

    std::vector<std::size_t> labels(n);
    std::size_t iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        // Assignment step
        for (std::size_t i = 0; i < n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            std::size_t best_cluster = 0;

            for (std::size_t c = 0; c < k; ++c) {
                double dist = euclidean_distance(data[i], centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            labels[i] = best_cluster;
        }

        // Update step
        std::vector<std::vector<double>> new_centroids(k, std::vector<double>(dim, 0.0));
        std::vector<std::size_t> cluster_sizes(k, 0);

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t c = labels[i];
            cluster_sizes[c]++;
            for (std::size_t d = 0; d < dim; ++d) {
                new_centroids[c][d] += data[i][d];
            }
        }

        for (std::size_t c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (std::size_t d = 0; d < dim; ++d) {
                    new_centroids[c][d] /= static_cast<double>(cluster_sizes[c]);
                }
            }
        }

        // Convergence check
        double max_shift = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            double shift = euclidean_distance(centroids[c], new_centroids[c]);
            max_shift = std::max(max_shift, shift);
        }

        centroids = std::move(new_centroids);

        if (max_shift < tol) {
            iter++;
            break;
        }
    }

    // Calculate inertia
    double inertia = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dist = euclidean_distance(data[i], centroids[labels[i]]);
        inertia += dist * dist;
    }

    return {labels, centroids, inertia, iter};
}

// ============================================================================
// Hierarchical Clustering
// ============================================================================

/**
 * @brief Linkage types
 */
enum class linkage_type {
    single,    ///< Single linkage
    complete,  ///< Complete linkage
    average,   ///< Average linkage
    ward       ///< Ward's method
};

/**
 * @brief Dendrogram node
 */
struct dendrogram_node {
    std::size_t left;      ///< Left child (cluster or data point)
    std::size_t right;     ///< Right child
    double distance;       ///< Merge distance
    std::size_t count;     ///< Number of data points in cluster
};

/**
 * @brief Hierarchical clustering
 *
 * Performs hierarchical clustering and generates a dendrogram.
 *
 * @param data Vector of data points
 * @param linkage Linkage type (default: single)
 * @return Dendrogram
 * @throws std::invalid_argument If data is empty
 *
 * @note Has O(n^3) time complexity. Not suitable for large datasets.
 */
inline std::vector<dendrogram_node> hierarchical_clustering(
    const std::vector<std::vector<double>>& data,
    linkage_type linkage = linkage_type::single)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::hierarchical_clustering: empty data");
    }

    std::size_t n = data.size();

    // Calculate distance matrix
    std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double d = euclidean_distance(data[i], data[j]);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }

    // Active clusters
    std::vector<bool> active(2 * n - 1, false);
    for (std::size_t i = 0; i < n; ++i) {
        active[i] = true;
    }

    // Cluster sizes
    std::vector<std::size_t> cluster_size(2 * n - 1, 1);

    // Dendrogram
    std::vector<dendrogram_node> dendrogram;
    dendrogram.reserve(n - 1);

    // Extended cluster distance matrix
    // For Ward's method, store squared distances; for others, store distances.
    bool use_squared = (linkage == linkage_type::ward);
    std::vector<std::vector<double>> cluster_dist(2 * n - 1, std::vector<double>(2 * n - 1, std::numeric_limits<double>::max()));
    // Copy initial distance matrix
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double d = dist_matrix[i][j];
            cluster_dist[i][j] = use_squared ? d * d : d;
        }
    }

    for (std::size_t step = 0; step < n - 1; ++step) {
        // Find the closest pair
        double min_dist = std::numeric_limits<double>::max();
        std::size_t min_i = 0, min_j = 0;

        for (std::size_t i = 0; i < n + step; ++i) {
            if (!active[i]) continue;
            for (std::size_t j = i + 1; j < n + step; ++j) {
                if (!active[j]) continue;
                if (cluster_dist[i][j] < min_dist) {
                    min_dist = cluster_dist[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Create new cluster
        std::size_t new_cluster = n + step;
        active[min_i] = false;
        active[min_j] = false;
        active[new_cluster] = true;

        cluster_size[new_cluster] = cluster_size[min_i] + cluster_size[min_j];

        // Store Euclidean distance in dendrogram (take sqrt for Ward's squared distances)
        double dendro_dist = use_squared ? std::sqrt(min_dist) : min_dist;
        dendrogram.push_back({min_i, min_j, dendro_dist, cluster_size[new_cluster]});

        // Update distances between new cluster and other clusters
        for (std::size_t k = 0; k < new_cluster; ++k) {
            if (!active[k]) continue;

            double new_dist;
            switch (linkage) {
                case linkage_type::single:
                    new_dist = std::min(cluster_dist[min_i][k], cluster_dist[min_j][k]);
                    break;
                case linkage_type::complete:
                    new_dist = std::max(cluster_dist[min_i][k], cluster_dist[min_j][k]);
                    break;
                case linkage_type::average:
                    new_dist = (cluster_size[min_i] * cluster_dist[min_i][k] +
                               cluster_size[min_j] * cluster_dist[min_j][k]) /
                              static_cast<double>(cluster_size[min_i] + cluster_size[min_j]);
                    break;
                case linkage_type::ward: {
                    // Lance-Williams recurrence for Ward's method on squared distances
                    double ni = static_cast<double>(cluster_size[min_i]);
                    double nj = static_cast<double>(cluster_size[min_j]);
                    double nk = static_cast<double>(cluster_size[k]);
                    double nijk = ni + nj + nk;
                    new_dist = ((ni + nk) * cluster_dist[min_i][k] +
                                (nj + nk) * cluster_dist[min_j][k] -
                                nk * min_dist) / nijk;
                    break;
                }
            }

            cluster_dist[new_cluster][k] = new_dist;
            cluster_dist[k][new_cluster] = new_dist;
        }
    }

    return dendrogram;
}

/**
 * @brief Extract k clusters from dendrogram
 *
 * Cuts the dendrogram to extract k clusters.
 *
 * @param dendrogram Dendrogram
 * @param n_data Number of data points
 * @param k Number of clusters
 * @return Cluster labels
 * @throws std::invalid_argument If k is invalid
 */
inline std::vector<std::size_t> cut_dendrogram(
    const std::vector<dendrogram_node>& dendrogram,
    std::size_t n_data,
    std::size_t k)
{
    if (k == 0 || k > n_data) {
        throw std::invalid_argument("statcpp::cut_dendrogram: invalid k");
    }

    std::vector<std::size_t> labels(n_data);
    std::iota(labels.begin(), labels.end(), 0);

    if (k == n_data) {
        return labels;
    }

    // Apply n_data - k merges
    std::size_t n_merges = n_data - k;

    std::vector<std::size_t> cluster_map(2 * n_data - 1);
    std::iota(cluster_map.begin(), cluster_map.end(), 0);

    for (std::size_t i = 0; i < n_merges; ++i) {
        const auto& node = dendrogram[i];
        std::size_t new_cluster = n_data + i;

        // Assign merged clusters the same label
        std::size_t left_label = cluster_map[node.left];
        std::size_t right_label = cluster_map[node.right];

        for (std::size_t j = 0; j < 2 * n_data - 1; ++j) {
            if (cluster_map[j] == right_label) {
                cluster_map[j] = left_label;
            }
        }
        cluster_map[new_cluster] = left_label;
    }

    // Normalize labels to 0 through k-1
    for (std::size_t i = 0; i < n_data; ++i) {
        labels[i] = cluster_map[i];
    }

    // Convert labels to consecutive integers
    std::map<std::size_t, std::size_t> label_map;
    std::size_t next_label = 0;
    for (std::size_t i = 0; i < n_data; ++i) {
        if (label_map.find(labels[i]) == label_map.end()) {
            label_map[labels[i]] = next_label++;
        }
        labels[i] = label_map[labels[i]];
    }

    return labels;
}

// ============================================================================
// Silhouette Score
// ============================================================================

/**
 * @brief Calculate silhouette score
 *
 * Calculates the silhouette score to evaluate clustering quality.
 *
 * @param data Vector of data points
 * @param labels Cluster labels
 * @return Silhouette score (-1 to 1, closer to 1 is better)
 * @throws std::invalid_argument If data is empty or sizes don't match
 *
 * @note Score interpretation: 0.7-1.0: strong structure, 0.5-0.7: reasonable structure,
 * 0.25-0.5: weak structure, < 0.25: no substantial structure
 */
inline double silhouette_score(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::size_t>& labels)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::silhouette_score: empty data");
    }
    if (data.size() != labels.size()) {
        throw std::invalid_argument("statcpp::silhouette_score: data and labels size mismatch");
    }

    std::size_t n = data.size();

    // Check number of clusters
    std::size_t k = *std::max_element(labels.begin(), labels.end()) + 1;
    if (k == 1) {
        return 0.0;  // Single cluster case
    }

    double total_silhouette = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        // a(i): Average distance to points in the same cluster
        double a = 0.0;
        std::size_t same_cluster_count = 0;

        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && labels[j] == labels[i]) {
                a += euclidean_distance(data[i], data[j]);
                same_cluster_count++;
            }
        }
        if (same_cluster_count > 0) {
            a /= static_cast<double>(same_cluster_count);
        }

        // b(i): Average distance to points in the nearest other cluster
        double b = std::numeric_limits<double>::max();

        for (std::size_t c = 0; c < k; ++c) {
            if (c == labels[i]) continue;

            double cluster_dist = 0.0;
            std::size_t cluster_count = 0;

            for (std::size_t j = 0; j < n; ++j) {
                if (labels[j] == c) {
                    cluster_dist += euclidean_distance(data[i], data[j]);
                    cluster_count++;
                }
            }

            if (cluster_count > 0) {
                cluster_dist /= static_cast<double>(cluster_count);
                b = std::min(b, cluster_dist);
            }
        }

        // Silhouette value
        double s = 0.0;
        if (same_cluster_count > 0) {
            s = (b - a) / std::max(a, b);
        }
        total_silhouette += s;
    }

    return total_silhouette / static_cast<double>(n);
}

} // namespace statcpp
