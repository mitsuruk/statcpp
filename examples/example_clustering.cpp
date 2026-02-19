/**
 * @file example_clustering.cpp
 * @brief Sample code for clustering methods
 *
 * Demonstrates usage of k-means clustering, hierarchical clustering,
 * silhouette score, and distance functions.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/clustering.hpp"

void print_data_points(const std::vector<std::vector<double>>& data, const std::string& label) {
    std::cout << label << ":" << std::endl;
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::cout << "  Point " << i << ": [";
        for (std::size_t j = 0; j < data[i].size(); ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << data[i][j];
            if (j + 1 < data[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    std::cout << "=== Clustering Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Distance Functions
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "1. Distance Functions" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Basic methods for measuring similarity between data points" << std::endl;

    std::vector<double> point_a = {1.0, 2.0, 3.0};
    std::vector<double> point_b = {4.0, 5.0, 6.0};

    double euclidean_dist = statcpp::euclidean_distance(point_a, point_b);
    double manhattan_dist = statcpp::manhattan_distance(point_a, point_b);

    std::cout << "\n--- Distance Between Two Points ---" << std::endl;
    std::cout << "Point A: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "Point B: [4.0, 5.0, 6.0]" << std::endl;
    std::cout << "\nEuclidean Distance: " << euclidean_dist << std::endl;
    std::cout << "  -> sqrt[(4-1)^2 + (5-2)^2 + (6-3)^2] = " << euclidean_dist << std::endl;
    std::cout << "\nManhattan Distance: " << manhattan_dist << std::endl;
    std::cout << "  -> |4-1| + |5-2| + |6-3| = " << manhattan_dist << std::endl;

    // ============================================================================
    // 2. K-means Clustering
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "2. K-means Clustering" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "A representative method that partitions data into K clusters" << std::endl;
    std::cout << "Optimizes clusters by iteratively updating cluster centers (centroids)" << std::endl;

    statcpp::set_seed(42);

    // Create data consisting of 3 clusters
    std::vector<std::vector<double>> data = {
        // Cluster 1 (around 0, 0)
        {0.5, 0.3}, {0.8, 0.5}, {0.3, 0.7}, {0.6, 0.4},
        {0.4, 0.6}, {0.7, 0.2}, {0.2, 0.5},
        // Cluster 2 (around 5, 5)
        {5.2, 5.1}, {5.5, 5.3}, {5.1, 5.5}, {4.9, 5.2},
        {5.3, 4.8}, {5.4, 5.4}, {4.8, 5.0},
        // Cluster 3 (around 10, 0)
        {10.1, 0.2}, {10.3, 0.5}, {9.8, 0.3}, {10.2, 0.6},
        {9.9, 0.4}, {10.4, 0.1}, {10.0, 0.5}
    };

    std::cout << "\n[Example: Grouping Customer Data]" << std::endl;
    std::cout << "Partitioning 21 data points into 3 clusters" << std::endl;

    std::size_t k = 3;
    auto kmeans_result = statcpp::kmeans(data, k);

    std::cout << "\n--- K-means Results ---" << std::endl;
    std::cout << "Number of clusters k = " << k << std::endl;
    std::cout << "Iterations until convergence: " << kmeans_result.n_iter << std::endl;
    std::cout << "Inertia (within-cluster sum of squares): " << kmeans_result.inertia << std::endl;
    std::cout << "  -> Smaller values indicate more compact clusters" << std::endl;

    std::cout << "\nCluster Centers (Centroids):" << std::endl;
    for (std::size_t i = 0; i < kmeans_result.centroids.size(); ++i) {
        std::cout << "  Cluster " << i << ": [";
        for (std::size_t j = 0; j < kmeans_result.centroids[i].size(); ++j) {
            std::cout << std::setw(6) << kmeans_result.centroids[i][j];
            if (j + 1 < kmeans_result.centroids[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nPoints per Cluster:" << std::endl;
    std::vector<std::size_t> cluster_counts(k, 0);
    for (std::size_t i = 0; i < kmeans_result.labels.size(); ++i) {
        std::size_t cluster = kmeans_result.labels[i];
        cluster_counts[cluster]++;
    }

    for (std::size_t i = 0; i < k; ++i) {
        std::cout << "  Cluster " << i << ": " << cluster_counts[i] << " points" << std::endl;
    }

    // ============================================================================
    // 3. Silhouette Score
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "3. Silhouette Score (Cluster Quality Evaluation)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "A metric for evaluating clustering quality (-1 to 1)" << std::endl;
    std::cout << "Higher values indicate better cluster separation" << std::endl;

    double silhouette = statcpp::silhouette_score(data, kmeans_result.labels);

    std::cout << "\n--- Silhouette Score ---" << std::endl;
    std::cout << "Score: " << silhouette << std::endl;
    std::cout << "\nInterpretation Guidelines:" << std::endl;
    std::cout << "  > 0.7: Strong structure (clear clusters)" << std::endl;
    std::cout << "  > 0.5: Reasonable structure" << std::endl;
    std::cout << "  > 0.25: Weak structure" << std::endl;
    std::cout << "  < 0.25: No clear structure" << std::endl;

    std::cout << "\nCurrent result: ";
    if (silhouette > 0.7) {
        std::cout << "Strong cluster structure" << std::endl;
    } else if (silhouette > 0.5) {
        std::cout << "Reasonable cluster structure" << std::endl;
    } else if (silhouette > 0.25) {
        std::cout << "Weak cluster structure" << std::endl;
    } else {
        std::cout << "No clear cluster structure" << std::endl;
    }

    // ============================================================================
    // 4. Elbow Method (Selecting Optimal Number of Clusters)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "4. Elbow Method (Selecting Optimal Number of Clusters)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Plot inertia for different k values and" << std::endl;
    std::cout << "select the 'elbow' point as the optimal k" << std::endl;

    std::cout << "\n[Example: Finding Optimal Number of Clusters]" << std::endl;
    std::cout << "Evaluating k from 2 to 5" << std::endl;

    std::cout << "\n--- Evaluation for Each k ---" << std::endl;
    std::cout << "   k  Inertia     Silhouette" << std::endl;

    for (std::size_t test_k = 2; test_k <= 5; ++test_k) {
        auto result = statcpp::kmeans(data, test_k);
        double sil = statcpp::silhouette_score(data, result.labels);
        std::cout << "  " << std::setw(2) << test_k
                  << "  " << std::setw(10) << result.inertia
                  << "  " << std::setw(10) << sil << std::endl;
    }

    std::cout << "\n[Interpretation]" << std::endl;
    std::cout << "  -> Find the 'elbow' in the inertia plot" << std::endl;
    std::cout << "  -> Select k that maximizes silhouette score" << std::endl;
    std::cout << "  -> Inertia decreases with larger k, but risk of overfitting" << std::endl;

    // ============================================================================
    // 5. Hierarchical Clustering (Single Linkage)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "5. Hierarchical Clustering (Single Linkage)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "A method that sequentially merges nearest points" << std::endl;
    std::cout << "Creates a dendrogram (tree diagram) to visualize hierarchical structure" << std::endl;

    std::vector<std::vector<double>> small_data = {
        {0.0, 0.0}, {1.0, 0.5}, {5.0, 5.0}, {5.5, 5.2}, {10.0, 0.0}
    };

    std::cout << "\n[Example: Hierarchical Clustering of Small Dataset]" << std::endl;
    print_data_points(small_data, "Input Data");

    auto dendrogram = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::single);

    std::cout << "\n--- Dendrogram (Merge History) ---" << std::endl;
    for (std::size_t i = 0; i < dendrogram.size(); ++i) {
        std::cout << "  Step " << (i + 1) << ": Merge cluster "
                  << dendrogram[i].left << " and " << dendrogram[i].right
                  << " at distance " << dendrogram[i].distance << std::endl;
    }
    std::cout << "  -> Merging nearest clusters sequentially" << std::endl;

    // ============================================================================
    // 6. Extracting Clusters from Dendrogram
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "6. Extracting Clusters from Dendrogram" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Cut the dendrogram at an appropriate height to generate clusters" << std::endl;
    std::cout << "The cut position determines the number of clusters" << std::endl;

    std::size_t n_clusters = 3;
    auto hier_labels = statcpp::cut_dendrogram(dendrogram, small_data.size(), n_clusters);

    std::cout << "\n[Example: Partitioning into " << n_clusters << " Clusters]" << std::endl;
    std::cout << "\n--- Cluster Assignment for Each Point ---" << std::endl;
    for (std::size_t i = 0; i < hier_labels.size(); ++i) {
        std::cout << "  Point " << i << " -> Cluster " << hier_labels[i] << std::endl;
    }
    std::cout << "  -> Cutting dendrogram to obtain flat clustering" << std::endl;

    // ============================================================================
    // 7. Comparison of Complete and Average Linkage
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "7. Comparison of Linkage Methods (Single / Complete / Average)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Three methods defined by how inter-cluster distance is calculated" << std::endl;

    auto dend_complete = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::complete);
    auto dend_average = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::average);

    std::cout << "\n[Example: Comparison of Three Linkage Methods]" << std::endl;
    std::cout << "\n--- Final Merge Distance ---" << std::endl;
    std::cout << "Single Linkage:   "
              << dendrogram.back().distance << std::endl;
    std::cout << "Complete Linkage: "
              << dend_complete.back().distance << std::endl;
    std::cout << "Average Linkage:  "
              << dend_average.back().distance << std::endl;

    std::cout << "\n[Linkage Method Characteristics]" << std::endl;
    std::cout << "  Single Linkage:   Nearest point distance" << std::endl;
    std::cout << "    -> Tends to create chain-like clusters" << std::endl;
    std::cout << "  Complete Linkage: Farthest point distance" << std::endl;
    std::cout << "    -> Creates compact clusters" << std::endl;
    std::cout << "  Average Linkage:  Average of all pairwise distances" << std::endl;
    std::cout << "    -> Balanced results" << std::endl;

    // ============================================================================
    // 8. Practical Example: Customer Segmentation
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "8. Practical Example: Customer Segmentation" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Group customers by purchasing behavior to optimize marketing strategies" << std::endl;

    // Customer data: [Purchase Frequency, Average Purchase Amount] (normalized)
    std::vector<std::vector<double>> customers = {
        {0.2, 0.3}, {0.3, 0.2}, {0.1, 0.4},  // Low frequency, low amount
        {0.7, 0.8}, {0.8, 0.7}, {0.6, 0.9},  // High frequency, high amount
        {0.9, 0.2}, {0.8, 0.3}, {0.7, 0.1},  // High frequency, low amount
        {0.1, 0.9}, {0.2, 0.8}, {0.3, 0.9}   // Low frequency, high amount
    };

    std::size_t customer_k = 4;
    auto customer_result = statcpp::kmeans(customers, customer_k);

    std::cout << "\n[Example: Classifying Customers into 4 Segments]" << std::endl;
    std::cout << "\n--- Segment Centers [Purchase Frequency, Avg Amount] ---" << std::endl;
    for (std::size_t i = 0; i < customer_result.centroids.size(); ++i) {
        std::cout << "  Segment " << i << ": ["
                  << std::setprecision(2)
                  << customer_result.centroids[i][0] << ", "
                  << customer_result.centroids[i][1] << "]";

        // Interpret segments
        double freq = customer_result.centroids[i][0];
        double amt = customer_result.centroids[i][1];
        std::cout << " - ";
        if (freq > 0.5 && amt > 0.5) std::cout << "VIP Customers (High Freq, High Amount)";
        else if (freq > 0.5 && amt < 0.5) std::cout << "Frequent Buyers (High Freq, Low Amount)";
        else if (freq < 0.5 && amt > 0.5) std::cout << "Occasional Big Spenders (Low Freq, High Amount)";
        else std::cout << "Low Engagement Customers";
        std::cout << std::endl;
    }

    std::cout << "\n--- Customer Distribution ---" << std::endl;
    std::vector<std::size_t> seg_counts(customer_k, 0);
    for (std::size_t i = 0; i < customer_result.labels.size(); ++i) {
        seg_counts[customer_result.labels[i]]++;
    }
    for (std::size_t i = 0; i < customer_k; ++i) {
        std::cout << "  Segment " << i << ": " << seg_counts[i] << " customers" << std::endl;
    }

    double customer_silhouette = statcpp::silhouette_score(customers, customer_result.labels);
    std::cout << "\nSegmentation Quality (Silhouette): " << customer_silhouette << std::endl;
    std::cout << "  -> Can deploy targeted marketing strategies for each segment" << std::endl;

    // ============================================================================
    // 9. Clustering Best Practices
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "9. Clustering Best Practices and Summary" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Preparation Before Clustering]" << std::endl;
    std::cout << "  1. Feature standardization (essential for K-means)" << std::endl;
    std::cout << "     -> Unify features with different scales" << std::endl;
    std::cout << "  2. Remove outliers or use robust methods" << std::endl;
    std::cout << "     -> Prevent results from being skewed by extreme values" << std::endl;
    std::cout << "  3. Consider feature selection" << std::endl;
    std::cout << "     -> Use only relevant features" << std::endl;

    std::cout << "\n[Algorithm Selection]" << std::endl;
    std::cout << "  K-means:" << std::endl;
    std::cout << "    + Fast and scalable to large datasets" << std::endl;
    std::cout << "    + Suitable for spherical clusters" << std::endl;
    std::cout << "    - Requires specifying k in advance" << std::endl;
    std::cout << "  Hierarchical Clustering:" << std::endl;
    std::cout << "    + No need to specify k in advance" << std::endl;
    std::cout << "    + Visualize hierarchy with dendrogram" << std::endl;
    std::cout << "    - High computational cost (O(n^2) or O(n^3))" << std::endl;

    std::cout << "\n[Methods for Selecting k]" << std::endl;
    std::cout << "  1. Elbow method (inertia vs k plot)" << std::endl;
    std::cout << "  2. Maximize silhouette score" << std::endl;
    std::cout << "  3. Domain knowledge" << std::endl;
    std::cout << "  4. Business requirements (e.g., number of marketing segments)" << std::endl;

    std::cout << "\n[Result Validation]" << std::endl;
    std::cout << "  - Silhouette score (cluster quality)" << std::endl;
    std::cout << "  - Examine cluster centers (understand each cluster's characteristics)" << std::endl;
    std::cout << "  - Check cluster sizes (avoid extreme imbalance)" << std::endl;
    std::cout << "  - Visualize if possible (2D or 3D plot)" << std::endl;

    std::cout << "\n[Practical Applications]" << std::endl;
    std::cout << "  - Customer segmentation (marketing)" << std::endl;
    std::cout << "  - Anomaly detection (isolated clusters)" << std::endl;
    std::cout << "  - Image compression (color clustering)" << std::endl;
    std::cout << "  - Document classification (topic grouping)" << std::endl;
    std::cout << "  - Gene expression pattern analysis" << std::endl;

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
