/**
 * @file example_distance_metrics.cpp
 * @brief 距離・類似度メトリクスのサンプルコード
 *
 * ユークリッド距離、マンハッタン距離、コサイン類似度、
 * ミンコフスキー距離、マハラノビス距離等の距離・類似度指標の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/distance_metrics.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=== 距離・類似度メトリクスの例 ===\n\n";

    // Sample vectors
    std::vector<double> v1 = {1.0, 2.0, 3.0};
    std::vector<double> v2 = {4.0, 5.0, 6.0};
    std::vector<double> v3 = {1.0, 0.0};
    std::vector<double> v4 = {0.0, 1.0};

    // Euclidean Distance
    std::cout << "1. ユークリッド距離\n";
    std::cout << "   [1,2,3] と [4,5,6] 間の距離: "
              << statcpp::euclidean_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Manhattan Distance
    std::cout << "2. マンハッタン距離\n";
    std::cout << "   [1,2,3] と [4,5,6] 間の距離: "
              << statcpp::manhattan_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Cosine Similarity
    std::cout << "3. コサイン類似度\n";
    std::cout << "   [1,0] と [0,1] 間の類似度: "
              << statcpp::cosine_similarity(v3.begin(), v3.end(), v4.begin(), v4.end()) << "\n";
    std::cout << "   コサイン距離: "
              << statcpp::cosine_distance(v3.begin(), v3.end(), v4.begin(), v4.end()) << "\n\n";

    // Minkowski Distance
    std::cout << "4. ミンコフスキー距離\n";
    std::cout << "   p=1 (マンハッタン): "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 1.0) << "\n";
    std::cout << "   p=2 (ユークリッド): "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 2.0) << "\n";
    std::cout << "   p=3: "
              << statcpp::minkowski_distance(v1.begin(), v1.end(), v2.begin(), v2.end(), 3.0) << "\n\n";

    // Chebyshev Distance
    std::cout << "5. チェビシェフ距離 (L∞)\n";
    std::cout << "   [1,2,3] と [4,5,6] 間の距離: "
              << statcpp::chebyshev_distance(v1.begin(), v1.end(), v2.begin(), v2.end()) << "\n\n";

    // Mahalanobis Distance
    std::cout << "6. マハラノビス距離 (2次元のみ)\n";
    std::vector<double> point = {2.0, 2.0};
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> cov = {{1.0, 0.0}, {0.0, 1.0}};

    std::cout << "   [2,2] から分布 (平均=[0,0], 共分散=I) までの距離: "
              << statcpp::mahalanobis_distance(point, mean, cov) << "\n\n";

    return 0;
}
