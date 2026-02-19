/**
 * @file clustering.hpp
 * @brief クラスタリングアルゴリズムの実装 (Clustering algorithms implementation)
 *
 * K-means、階層的クラスタリング、シルエットスコアなどを提供します。
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
 * @brief ユークリッド距離 (Euclidean distance)
 *
 * 2つのベクトル間のユークリッド距離を計算します。
 * Computes the Euclidean distance between two vectors.
 *
 * @param a 第1ベクトル (first vector)
 * @param b 第2ベクトル (second vector)
 * @return ユークリッド距離 (Euclidean distance)
 * @throws std::invalid_argument ベクトルの次元が一致しない場合 (if vector dimensions mismatch)
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
 * @brief マンハッタン距離 (Manhattan distance)
 *
 * 2つのベクトル間のマンハッタン距離を計算します。
 * Computes the Manhattan distance between two vectors.
 *
 * @param a 第1ベクトル (first vector)
 * @param b 第2ベクトル (second vector)
 * @return マンハッタン距離 (Manhattan distance)
 * @throws std::invalid_argument ベクトルの次元が一致しない場合 (if vector dimensions mismatch)
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
 * @brief K-means の結果 (K-means clustering result)
 */
struct kmeans_result {
    std::vector<std::size_t> labels;              ///< クラスタ割り当て (cluster assignments)
    std::vector<std::vector<double>> centroids;   ///< クラスタ中心 (cluster centroids)
    double inertia;                               ///< 慣性（クラスタ内平方和）(inertia, within-cluster sum of squares)
    std::size_t n_iter;                           ///< 収束までの反復回数 (number of iterations until convergence)
};

/**
 * @brief K-means++ 初期化 (K-means++ initialization)
 *
 * K-means++ アルゴリズムによる初期クラスタ中心の選択を行います。
 * Selects initial cluster centroids using the K-means++ algorithm.
 *
 * @param data データ点のベクトル (vector of data points)
 * @param k クラスタ数 (number of clusters)
 * @return 初期クラスタ中心 (initial cluster centroids)
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

    // 最初のセントロイドをランダムに選択
    centroids.push_back(data[init_dist(rng)]);

    // 残りのセントロイドを確率的に選択
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

        // 確率的に次のセントロイドを選択
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
 * @brief K-means クラスタリング (K-means clustering)
 *
 * K-means アルゴリズムによるクラスタリングを実行します。
 * Performs clustering using the K-means algorithm.
 *
 * @param data データ点のベクトル (vector of data points)
 * @param k クラスタ数 (number of clusters)
 * @param max_iter 最大反復回数 (maximum number of iterations, default: 100)
 * @param tol 収束判定の閾値 (convergence tolerance, default: 1e-6)
 * @return クラスタリング結果 (clustering result)
 * @throws std::invalid_argument データが空、kが0、またはkがデータ数を超える場合
 *         (if data is empty, k is 0, or k exceeds number of data points)
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

    // K-means++ 初期化
    auto centroids = kmeans_plusplus_init(data, k);

    std::vector<std::size_t> labels(n);
    std::size_t iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        // 割り当てステップ
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

        // 更新ステップ
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

        // 収束判定
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

    // 慣性を計算
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
 * @brief 連結法の種類 (Linkage types)
 */
enum class linkage_type {
    single,    ///< 最短距離法 (single linkage)
    complete,  ///< 最長距離法 (complete linkage)
    average,   ///< 群平均法 (average linkage)
    ward       ///< Ward 法 (Ward's method)
};

/**
 * @brief デンドログラムのノード (Dendrogram node)
 */
struct dendrogram_node {
    std::size_t left;      ///< 左子（クラスタまたはデータ点）(left child)
    std::size_t right;     ///< 右子 (right child)
    double distance;       ///< 結合距離 (merge distance)
    std::size_t count;     ///< クラスタ内のデータ点数 (number of data points in cluster)
};

/**
 * @brief 階層的クラスタリング (Hierarchical clustering)
 *
 * 階層的クラスタリングを実行し、デンドログラムを生成します。
 * Performs hierarchical clustering and generates a dendrogram.
 *
 * @param data データ点のベクトル (vector of data points)
 * @param linkage 連結法の種類 (linkage type, default: single)
 * @return デンドログラム (dendrogram)
 * @throws std::invalid_argument データが空の場合 (if data is empty)
 *
 * @note O(n³) の計算量を持ちます。大規模データには適していません。
 * Has O(n³) time complexity. Not suitable for large datasets.
 */
inline std::vector<dendrogram_node> hierarchical_clustering(
    const std::vector<std::vector<double>>& data,
    linkage_type linkage = linkage_type::single)
{
    if (data.empty()) {
        throw std::invalid_argument("statcpp::hierarchical_clustering: empty data");
    }

    std::size_t n = data.size();

    // 距離行列を計算
    std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double d = euclidean_distance(data[i], data[j]);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }

    // アクティブなクラスタ
    std::vector<bool> active(2 * n - 1, false);
    for (std::size_t i = 0; i < n; ++i) {
        active[i] = true;
    }

    // クラスタサイズ
    std::vector<std::size_t> cluster_size(2 * n - 1, 1);

    // デンドログラム
    std::vector<dendrogram_node> dendrogram;
    dendrogram.reserve(n - 1);

    // クラスタ間距離を拡張した行列
    // Ward法では二乗距離で管理し、その他では距離をそのまま管理する
    bool use_squared = (linkage == linkage_type::ward);
    std::vector<std::vector<double>> cluster_dist(2 * n - 1, std::vector<double>(2 * n - 1, std::numeric_limits<double>::max()));
    // 初期距離行列をコピー
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double d = dist_matrix[i][j];
            cluster_dist[i][j] = use_squared ? d * d : d;
        }
    }

    for (std::size_t step = 0; step < n - 1; ++step) {
        // 最も近いペアを見つける
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

        // 新しいクラスタを作成
        std::size_t new_cluster = n + step;
        active[min_i] = false;
        active[min_j] = false;
        active[new_cluster] = true;

        cluster_size[new_cluster] = cluster_size[min_i] + cluster_size[min_j];

        // デンドログラムにはユークリッド距離を格納（Ward法の場合は平方根を取る）
        double dendro_dist = use_squared ? std::sqrt(min_dist) : min_dist;
        dendrogram.push_back({min_i, min_j, dendro_dist, cluster_size[new_cluster]});

        // 新しいクラスタと他のクラスタ間の距離を更新
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
                    // Ward法のLance-Williams漸化式（二乗距離に対して適用）
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
 * @brief デンドログラムから k 個のクラスタを抽出 (Extract k clusters from dendrogram)
 *
 * デンドログラムを切断して k 個のクラスタを抽出します。
 * Cuts the dendrogram to extract k clusters.
 *
 * @param dendrogram デンドログラム (dendrogram)
 * @param n_data データ点の数 (number of data points)
 * @param k クラスタ数 (number of clusters)
 * @return クラスタラベル (cluster labels)
 * @throws std::invalid_argument k が無効な場合 (if k is invalid)
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

    // n_data - k 回のマージまで適用
    std::size_t n_merges = n_data - k;

    std::vector<std::size_t> cluster_map(2 * n_data - 1);
    std::iota(cluster_map.begin(), cluster_map.end(), 0);

    for (std::size_t i = 0; i < n_merges; ++i) {
        const auto& node = dendrogram[i];
        std::size_t new_cluster = n_data + i;

        // マージされるクラスタを同じラベルに
        std::size_t left_label = cluster_map[node.left];
        std::size_t right_label = cluster_map[node.right];

        for (std::size_t j = 0; j < 2 * n_data - 1; ++j) {
            if (cluster_map[j] == right_label) {
                cluster_map[j] = left_label;
            }
        }
        cluster_map[new_cluster] = left_label;
    }

    // ラベルを 0 から k-1 に正規化
    for (std::size_t i = 0; i < n_data; ++i) {
        labels[i] = cluster_map[i];
    }

    // ラベルを連続した整数に変換
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
 * @brief シルエットスコアを計算 (Calculate silhouette score)
 *
 * クラスタリングの品質を評価するシルエットスコアを計算します。
 * Calculates the silhouette score to evaluate clustering quality.
 *
 * @param data データ点のベクトル (vector of data points)
 * @param labels クラスタラベル (cluster labels)
 * @return シルエットスコア (-1〜1、1に近いほど良好) (silhouette score, -1 to 1, closer to 1 is better)
 * @throws std::invalid_argument データが空、またはサイズが一致しない場合
 *         (if data is empty or sizes don't match)
 *
 * @note スコアの解釈：0.7-1.0: 強い構造、0.5-0.7: 妥当な構造、0.25-0.5: 弱い構造、< 0.25: 構造なし
 * Score interpretation: 0.7-1.0: strong structure, 0.5-0.7: reasonable structure,
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

    // クラスタ数を確認
    std::size_t k = *std::max_element(labels.begin(), labels.end()) + 1;
    if (k == 1) {
        return 0.0;  // 単一クラスタの場合
    }

    double total_silhouette = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        // a(i): 同じクラスタ内の平均距離
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

        // b(i): 最も近い他クラスタへの平均距離
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

        // シルエット値
        double s = 0.0;
        if (same_cluster_count > 0) {
            s = (b - a) / std::max(a, b);
        }
        total_silhouette += s;
    }

    return total_silhouette / static_cast<double>(n);
}

} // namespace statcpp
