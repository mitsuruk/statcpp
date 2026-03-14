/**
 * @file example_distance_metrics.cpp
 * @brief Sample code for distance and similarity metrics
 *
 * Demonstrates usage of Euclidean distance, Manhattan distance, cosine similarity,
 * Minkowski distance, Mahalanobis distance, and other distance/similarity measures.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/distance_metrics.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=== Distance and Similarity Metrics Examples ===\n\n";

    // Sample vectors
    std::vector<double> v1 = {1.0, 2.0, 3.0};
    std::vector<double> v2 = {4.0, 5.0, 6.0};
    std::vector<double> v3 = {1.0, 0.0};
    std::vector<double> v4 = {0.0, 1.0};

    // Euclidean Distance
    std::cout << "1. Euclidean Distance\n";
    std::cout << "   Distance between [1,2,3] and [4,5,6]: "
              << statcpp::euclidean_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Manhattan Distance
    std::cout << "2. Manhattan Distance\n";
    std::cout << "   Distance between [1,2,3] and [4,5,6]: "
              << statcpp::manhattan_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Cosine Similarity
    std::cout << "3. Cosine Similarity\n";
    std::cout << "   Similarity between [1,0] and [0,1]: "
              << statcpp::cosine_similarity(v3.begin(), v3.end(), v4.begin(), v4.end()) << "\n";
    std::cout << "   Cosine distance: "
              << statcpp::cosine_distance(v3.begin(), v3.end(), v4.begin(), v4.end()) << "\n\n";

    // Minkowski Distance
    std::cout << "4. Minkowski Distance\n";
    std::cout << "   p=1 (Manhattan): "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 1.0) << "\n";
    std::cout << "   p=2 (Euclidean): "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 2.0) << "\n";
    std::cout << "   p=3: "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 3.0) << "\n\n";

    // Chebyshev Distance
    std::cout << "5. Chebyshev Distance (L-infinity)\n";
    std::cout << "   Distance between [1,2,3] and [4,5,6]: "
              << statcpp::chebyshev_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Mahalanobis Distance
    std::cout << "6. Mahalanobis Distance (2D only)\n";
    std::vector<double> point = {2.0, 2.0};
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> cov = {{1.0, 0.0}, {0.0, 1.0}};

    std::cout << "   Distance from [2,2] to distribution (mean=[0,0], cov=I): "
              << statcpp::mahalanobis_distance(point, mean, cov) << "\n\n";

    return 0;
}
