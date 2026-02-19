#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "statcpp/clustering.hpp"

// ============================================================================
// Distance Tests
// ============================================================================

/**
 * @brief Tests Euclidean distance calculation
 * @test Verifies that Euclidean distance is computed correctly using the standard formula
 */
TEST(DistanceTest, Euclidean) {
    std::vector<double> a = {0, 0};
    std::vector<double> b = {3, 4};

    double result = statcpp::euclidean_distance(a, b);
    EXPECT_DOUBLE_EQ(result, 5.0);
}

/**
 * @brief Tests Manhattan distance calculation
 * @test Verifies that Manhattan distance is computed as sum of absolute differences
 */
TEST(DistanceTest, Manhattan) {
    std::vector<double> a = {0, 0};
    std::vector<double> b = {3, 4};

    double result = statcpp::manhattan_distance(a, b);
    EXPECT_DOUBLE_EQ(result, 7.0);
}

/**
 * @brief Tests distance calculation for identical points
 * @test Verifies that distance between identical points is zero for both metrics
 */
TEST(DistanceTest, SamePoint) {
    std::vector<double> a = {1, 2, 3};
    EXPECT_DOUBLE_EQ(statcpp::euclidean_distance(a, a), 0.0);
    EXPECT_DOUBLE_EQ(statcpp::manhattan_distance(a, a), 0.0);
}

/**
 * @brief Tests distance calculation with mismatched dimensions
 * @test Verifies that an exception is thrown when points have different dimensions
 */
TEST(DistanceTest, DimensionMismatch) {
    std::vector<double> a = {1, 2};
    std::vector<double> b = {1, 2, 3};
    EXPECT_THROW(statcpp::euclidean_distance(a, b), std::invalid_argument);
}

// ============================================================================
// K-means Tests
// ============================================================================

/**
 * @brief Tests K-means clustering with two well-separated clusters
 * @test Verifies that K-means correctly separates two distinct groups of points
 */
TEST(KmeansTest, TwoClusters) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {0, 1}, {1, 1},
        {10, 10}, {11, 10}, {10, 11}, {11, 11}
    };

    statcpp::set_seed(42);
    auto result = statcpp::kmeans(data, 2);

    EXPECT_EQ(result.labels.size(), 8);
    EXPECT_EQ(result.centroids.size(), 2);
    EXPECT_GT(result.inertia, 0.0);

    // Points in same cluster should have same label
    EXPECT_EQ(result.labels[0], result.labels[1]);
    EXPECT_EQ(result.labels[4], result.labels[5]);
    // Points in different clusters should have different labels
    EXPECT_NE(result.labels[0], result.labels[4]);
}

/**
 * @brief Tests K-means clustering with a single cluster
 * @test Verifies that all points are assigned to the same cluster when K=1
 */
TEST(KmeansTest, SingleCluster) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {0, 1}, {1, 1}
    };

    auto result = statcpp::kmeans(data, 1);

    EXPECT_EQ(result.labels.size(), 4);
    // All should have same label
    for (auto label : result.labels) {
        EXPECT_EQ(label, 0);
    }
}

/**
 * @brief Tests K-means clustering with empty input data
 * @test Verifies that an exception is thrown when the input dataset is empty
 */
TEST(KmeansTest, EmptyData) {
    std::vector<std::vector<double>> data;
    EXPECT_THROW(statcpp::kmeans(data, 2), std::invalid_argument);
}

/**
 * @brief Tests K-means clustering with K larger than the number of data points
 * @test Verifies that an exception is thrown when K exceeds the dataset size
 */
TEST(KmeansTest, KTooLarge) {
    std::vector<std::vector<double>> data = {{0, 0}, {1, 1}};
    EXPECT_THROW(statcpp::kmeans(data, 5), std::invalid_argument);
}

/**
 * @brief Tests K-means convergence behavior
 * @test Verifies that K-means converges within the maximum number of iterations
 */
TEST(KmeansTest, Convergence) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {5, 5}, {5, 6}, {6, 5}, {6, 6}
    };

    statcpp::set_seed(42);
    auto result = statcpp::kmeans(data, 2, 1000, 1e-10);

    EXPECT_LE(result.n_iter, 1000);
}

// ============================================================================
// Hierarchical Clustering Tests
// ============================================================================

/**
 * @brief Tests basic hierarchical clustering
 * @test Verifies that hierarchical clustering produces n-1 merges for n data points
 */
TEST(HierarchicalTest, Basic) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {5, 5}, {6, 5}
    };

    auto dendrogram = statcpp::hierarchical_clustering(data);

    EXPECT_EQ(dendrogram.size(), 3);  // n-1 merges
}

/**
 * @brief Tests hierarchical clustering with single linkage
 * @test Verifies that single linkage merges clusters based on minimum distance
 */
TEST(HierarchicalTest, SingleLinkage) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {2, 0}
    };

    auto dendrogram = statcpp::hierarchical_clustering(data, statcpp::linkage_type::single);

    EXPECT_EQ(dendrogram.size(), 2);
    // First merge should be distance 1
    EXPECT_DOUBLE_EQ(dendrogram[0].distance, 1.0);
}

/**
 * @brief Tests hierarchical clustering with complete linkage
 * @test Verifies that complete linkage merges clusters based on maximum distance
 */
TEST(HierarchicalTest, CompleteLinkage) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {2, 0}
    };

    auto dendrogram = statcpp::hierarchical_clustering(data, statcpp::linkage_type::complete);

    EXPECT_EQ(dendrogram.size(), 2);
}

// ============================================================================
// Cut Dendrogram Tests
// ============================================================================

/**
 * @brief Tests cutting dendrogram to produce two clusters
 * @test Verifies that dendrogram can be cut to produce the desired number of clusters
 */
TEST(CutDendrogramTest, TwoClusters) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {1, 0}, {10, 10}, {11, 10}
    };

    auto dendrogram = statcpp::hierarchical_clustering(data);
    auto labels = statcpp::cut_dendrogram(dendrogram, 4, 2);

    EXPECT_EQ(labels.size(), 4);

    // First two should be in same cluster
    EXPECT_EQ(labels[0], labels[1]);
    // Last two should be in same cluster
    EXPECT_EQ(labels[2], labels[3]);
    // Different clusters
    EXPECT_NE(labels[0], labels[2]);
}

/**
 * @brief Tests cutting dendrogram with invalid K values
 * @test Verifies that exceptions are thrown for invalid number of clusters (K <= 0 or K > n)
 */
TEST(CutDendrogramTest, InvalidK) {
    std::vector<std::vector<double>> data = {{0, 0}, {1, 1}};
    auto dendrogram = statcpp::hierarchical_clustering(data);

    EXPECT_THROW(statcpp::cut_dendrogram(dendrogram, 2, 0), std::invalid_argument);
    EXPECT_THROW(statcpp::cut_dendrogram(dendrogram, 2, 5), std::invalid_argument);
}

// ============================================================================
// Silhouette Score Tests
// ============================================================================

/**
 * @brief Tests silhouette score for well-separated clusters
 * @test Verifies that silhouette score is high for clearly separated clusters
 */
TEST(SilhouetteTest, PerfectClusters) {
    std::vector<std::vector<double>> data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {10, 10}, {10, 11}, {11, 10}, {11, 11}
    };
    std::vector<std::size_t> labels = {0, 0, 0, 0, 1, 1, 1, 1};

    double score = statcpp::silhouette_score(data, labels);

    // Well-separated clusters should have high silhouette score
    EXPECT_GT(score, 0.5);
}

/**
 * @brief Tests silhouette score for a single cluster
 * @test Verifies that silhouette score is zero when all points belong to one cluster
 */
TEST(SilhouetteTest, SingleCluster) {
    std::vector<std::vector<double>> data = {{0, 0}, {1, 1}, {2, 2}};
    std::vector<std::size_t> labels = {0, 0, 0};

    double score = statcpp::silhouette_score(data, labels);
    EXPECT_DOUBLE_EQ(score, 0.0);
}

/**
 * @brief Tests silhouette score with empty input data
 * @test Verifies that an exception is thrown when computing silhouette score for empty data
 */
TEST(SilhouetteTest, EmptyData) {
    std::vector<std::vector<double>> data;
    std::vector<std::size_t> labels;
    EXPECT_THROW(statcpp::silhouette_score(data, labels), std::invalid_argument);
}
