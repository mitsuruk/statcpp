/**
 * @file example_clustering.cpp
 * @brief クラスタリング手法のサンプルコード
 *
 * k-means法、階層的クラスタリング、シルエットスコア等の
 * クラスタリングアルゴリズムと距離関数の使用例を示します。
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
    std::cout << "=== クラスタリング（Clustering）の例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 距離関数
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "1. 距離関数（Distance Functions）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "データポイント間の類似度を測定する基本的な方法" << std::endl;

    std::vector<double> point_a = {1.0, 2.0, 3.0};
    std::vector<double> point_b = {4.0, 5.0, 6.0};

    double euclidean_dist = statcpp::euclidean_distance(point_a, point_b);
    double manhattan_dist = statcpp::manhattan_distance(point_a, point_b);

    std::cout << "\n--- 2つのポイント間の距離 ---" << std::endl;
    std::cout << "ポイント A: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "ポイント B: [4.0, 5.0, 6.0]" << std::endl;
    std::cout << "\nユークリッド距離: " << euclidean_dist << std::endl;
    std::cout << "  → √[(4-1)² + (5-2)² + (6-3)²] = " << euclidean_dist << std::endl;
    std::cout << "\nマンハッタン距離: " << manhattan_dist << std::endl;
    std::cout << "  → |4-1| + |5-2| + |6-3| = " << manhattan_dist << std::endl;

    // ============================================================================
    // 2. K-means クラスタリング
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "2. K-means クラスタリング" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "データを K 個のクラスタに分割する代表的な手法" << std::endl;
    std::cout << "各クラスタの中心（重心）を繰り返し更新してクラスタを最適化" << std::endl;

    statcpp::set_seed(42);

    // 3つのクラスタからなるデータを作成
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

    std::cout << "\n【実例: 顧客データのグループ分け】" << std::endl;
    std::cout << "21個のデータポイントを3つのクラスタに分割" << std::endl;

    std::size_t k = 3;
    auto kmeans_result = statcpp::kmeans(data, k);

    std::cout << "\n--- K-means の結果 ---" << std::endl;
    std::cout << "クラスタ数 k = " << k << std::endl;
    std::cout << "収束までの反復回数: " << kmeans_result.n_iter << std::endl;
    std::cout << "慣性（クラスタ内平方和）: " << kmeans_result.inertia << std::endl;
    std::cout << "  → 小さいほどクラスタが密集している" << std::endl;

    std::cout << "\nクラスタ中心（重心）:" << std::endl;
    for (std::size_t i = 0; i < kmeans_result.centroids.size(); ++i) {
        std::cout << "  クラスタ " << i << ": [";
        for (std::size_t j = 0; j < kmeans_result.centroids[i].size(); ++j) {
            std::cout << std::setw(6) << kmeans_result.centroids[i][j];
            if (j + 1 < kmeans_result.centroids[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\n各クラスタのポイント数:" << std::endl;
    std::vector<std::size_t> cluster_counts(k, 0);
    for (std::size_t i = 0; i < kmeans_result.labels.size(); ++i) {
        std::size_t cluster = kmeans_result.labels[i];
        cluster_counts[cluster]++;
    }

    for (std::size_t i = 0; i < k; ++i) {
        std::cout << "  クラスタ " << i << ": " << cluster_counts[i] << " ポイント" << std::endl;
    }

    // ============================================================================
    // 3. シルエットスコア
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "3. シルエットスコア（クラスタの品質評価）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "クラスタリングの品質を評価する指標（-1 〜 1）" << std::endl;
    std::cout << "高いほどクラスタが明確に分離されている" << std::endl;

    double silhouette = statcpp::silhouette_score(data, kmeans_result.labels);

    std::cout << "\n--- シルエットスコア ---" << std::endl;
    std::cout << "スコア: " << silhouette << std::endl;
    std::cout << "\n解釈の目安:" << std::endl;
    std::cout << "  > 0.7: 強い構造（クラスタが明確）" << std::endl;
    std::cout << "  > 0.5: 妥当な構造" << std::endl;
    std::cout << "  > 0.25: 弱い構造" << std::endl;
    std::cout << "  < 0.25: 明確な構造なし" << std::endl;

    std::cout << "\n今回の結果: ";
    if (silhouette > 0.7) {
        std::cout << "強いクラスタ構造" << std::endl;
    } else if (silhouette > 0.5) {
        std::cout << "妥当なクラスタ構造" << std::endl;
    } else if (silhouette > 0.25) {
        std::cout << "弱いクラスタ構造" << std::endl;
    } else {
        std::cout << "明確なクラスタ構造なし" << std::endl;
    }

    // ============================================================================
    // 4. エルボー法（最適なクラスタ数の選択）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "4. エルボー法（最適なクラスタ数の選択）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "異なる k 値で慣性（inertia）をプロットし、" << std::endl;
    std::cout << "「肘」のように曲がるポイントを最適な k として選ぶ" << std::endl;

    std::cout << "\n【実例: 最適なクラスタ数の探索】" << std::endl;
    std::cout << "k を 2 から 5 まで変化させて評価" << std::endl;

    std::cout << "\n--- 各 k 値の評価 ---" << std::endl;
    std::cout << "   k  慣性        シルエット" << std::endl;

    for (std::size_t test_k = 2; test_k <= 5; ++test_k) {
        auto result = statcpp::kmeans(data, test_k);
        double sil = statcpp::silhouette_score(data, result.labels);
        std::cout << "  " << std::setw(2) << test_k
                  << "  " << std::setw(10) << result.inertia
                  << "  " << std::setw(10) << sil << std::endl;
    }

    std::cout << "\n【解釈】" << std::endl;
    std::cout << "  → 慣性のプロットで「肘」の位置を探す" << std::endl;
    std::cout << "  → シルエットスコアが最大となる k を選ぶ" << std::endl;
    std::cout << "  → k が大きいほど慣性は減少するが、過適合のリスク" << std::endl;

    // ============================================================================
    // 5. 階層的クラスタリング（単連結法）
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "5. 階層的クラスタリング（単連結法）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "最も近いポイント同士を順次結合していく手法" << std::endl;
    std::cout << "デンドログラム（樹形図）を作成し、階層構造を可視化" << std::endl;

    std::vector<std::vector<double>> small_data = {
        {0.0, 0.0}, {1.0, 0.5}, {5.0, 5.0}, {5.5, 5.2}, {10.0, 0.0}
    };

    std::cout << "\n【実例: 小規模データの階層的クラスタリング】" << std::endl;
    print_data_points(small_data, "入力データ");

    auto dendrogram = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::single);

    std::cout << "\n--- デンドログラム（結合履歴）---" << std::endl;
    for (std::size_t i = 0; i < dendrogram.size(); ++i) {
        std::cout << "  ステップ " << (i + 1) << ": クラスタ "
                  << dendrogram[i].left << " と " << dendrogram[i].right
                  << " を距離 " << dendrogram[i].distance << " で結合" << std::endl;
    }
    std::cout << "  → 最も近いクラスタから順に結合していく" << std::endl;

    // ============================================================================
    // 6. デンドログラムからクラスタを抽出
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "6. デンドログラムからクラスタを抽出" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "デンドログラムを適切な高さで切断し、クラスタを生成" << std::endl;
    std::cout << "切断位置によってクラスタ数が決まる" << std::endl;

    std::size_t n_clusters = 3;
    auto hier_labels = statcpp::cut_dendrogram(dendrogram, small_data.size(), n_clusters);

    std::cout << "\n【実例: " << n_clusters << "つのクラスタに分割】" << std::endl;
    std::cout << "\n--- 各ポイントのクラスタ割り当て ---" << std::endl;
    for (std::size_t i = 0; i < hier_labels.size(); ++i) {
        std::cout << "  ポイント " << i << " → クラスタ " << hier_labels[i] << std::endl;
    }
    std::cout << "  → デンドログラムを切断して平坦なクラスタリングを得る" << std::endl;

    // ============================================================================
    // 7. 完全連結法と平均連結法の比較
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "7. 連結法の比較（Single / Complete / Average）" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "クラスタ間の距離の定義方法による3つの手法" << std::endl;

    auto dend_complete = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::complete);
    auto dend_average = statcpp::hierarchical_clustering(small_data, statcpp::linkage_type::average);

    std::cout << "\n【実例: 3つの連結法の比較】" << std::endl;
    std::cout << "\n--- 最終結合時の距離 ---" << std::endl;
    std::cout << "単連結法 (Single):   "
              << dendrogram.back().distance << std::endl;
    std::cout << "完全連結法 (Complete): "
              << dend_complete.back().distance << std::endl;
    std::cout << "平均連結法 (Average):  "
              << dend_average.back().distance << std::endl;

    std::cout << "\n【連結法の特徴】" << std::endl;
    std::cout << "  単連結法 (Single):   最近接点間の距離" << std::endl;
    std::cout << "    → 鎖状のクラスタを作りやすい" << std::endl;
    std::cout << "  完全連結法 (Complete): 最遠点間の距離" << std::endl;
    std::cout << "    → コンパクトなクラスタを作る" << std::endl;
    std::cout << "  平均連結法 (Average):  全ペアの平均距離" << std::endl;
    std::cout << "    → バランスの取れた結果" << std::endl;

    // ============================================================================
    // 8. 実用例：顧客セグメンテーション
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "8. 実用例：顧客セグメンテーション" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【概念】" << std::endl;
    std::cout << "顧客を購買行動でグループ分けし、マーケティング戦略を最適化" << std::endl;

    // 顧客データ：[購入頻度, 平均購入額]（正規化済み）
    std::vector<std::vector<double>> customers = {
        {0.2, 0.3}, {0.3, 0.2}, {0.1, 0.4},  // 低頻度・低額
        {0.7, 0.8}, {0.8, 0.7}, {0.6, 0.9},  // 高頻度・高額
        {0.9, 0.2}, {0.8, 0.3}, {0.7, 0.1},  // 高頻度・低額
        {0.1, 0.9}, {0.2, 0.8}, {0.3, 0.9}   // 低頻度・高額
    };

    std::size_t customer_k = 4;
    auto customer_result = statcpp::kmeans(customers, customer_k);

    std::cout << "\n【実例: 顧客を4つのセグメントに分類】" << std::endl;
    std::cout << "\n--- セグメント中心 [購入頻度, 平均購入額] ---" << std::endl;
    for (std::size_t i = 0; i < customer_result.centroids.size(); ++i) {
        std::cout << "  セグメント " << i << ": ["
                  << std::setprecision(2)
                  << customer_result.centroids[i][0] << ", "
                  << customer_result.centroids[i][1] << "]";

        // セグメントを解釈
        double freq = customer_result.centroids[i][0];
        double amt = customer_result.centroids[i][1];
        std::cout << " - ";
        if (freq > 0.5 && amt > 0.5) std::cout << "VIP顧客（高頻度・高額）";
        else if (freq > 0.5 && amt < 0.5) std::cout << "頻繁購入者（高頻度・低額）";
        else if (freq < 0.5 && amt > 0.5) std::cout << "高額時々購入（低頻度・高額）";
        else std::cout << "低頻度・低額顧客";
        std::cout << std::endl;
    }

    std::cout << "\n--- 顧客分布 ---" << std::endl;
    std::vector<std::size_t> seg_counts(customer_k, 0);
    for (std::size_t i = 0; i < customer_result.labels.size(); ++i) {
        seg_counts[customer_result.labels[i]]++;
    }
    for (std::size_t i = 0; i < customer_k; ++i) {
        std::cout << "  セグメント " << i << ": " << seg_counts[i] << " 人の顧客" << std::endl;
    }

    double customer_silhouette = statcpp::silhouette_score(customers, customer_result.labels);
    std::cout << "\nセグメンテーション品質（シルエット）: " << customer_silhouette << std::endl;
    std::cout << "  → 各セグメントに適したマーケティング戦略を展開可能" << std::endl;

    // ============================================================================
    // 9. クラスタリングのベストプラクティス
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "9. クラスタリングのベストプラクティスとまとめ" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n【クラスタリング前の準備】" << std::endl;
    std::cout << "  1. 特徴量の標準化（特にK-meansでは必須）" << std::endl;
    std::cout << "     → 異なるスケールの特徴量を統一" << std::endl;
    std::cout << "  2. 外れ値の除去またはロバストな手法の使用" << std::endl;
    std::cout << "     → 結果が極端な値に左右されないように" << std::endl;
    std::cout << "  3. 特徴量選択の検討" << std::endl;
    std::cout << "     → 関連性の高い特徴量のみを使用" << std::endl;

    std::cout << "\n【アルゴリズムの選択】" << std::endl;
    std::cout << "  K-means:" << std::endl;
    std::cout << "    + 高速で大規模データに適用可能" << std::endl;
    std::cout << "    + 球状のクラスタに適している" << std::endl;
    std::cout << "    - クラスタ数 k を事前に指定する必要がある" << std::endl;
    std::cout << "  階層的クラスタリング:" << std::endl;
    std::cout << "    + k を事前に指定不要" << std::endl;
    std::cout << "    + デンドログラムで階層構造を可視化" << std::endl;
    std::cout << "    - 計算コストが高い（O(n²) または O(n³)）" << std::endl;

    std::cout << "\n【クラスタ数 k の選択方法】" << std::endl;
    std::cout << "  1. エルボー法（慣性 vs k のプロット）" << std::endl;
    std::cout << "  2. シルエットスコアの最大化" << std::endl;
    std::cout << "  3. ドメイン知識の活用" << std::endl;
    std::cout << "  4. ビジネス要件（マーケティングセグメント数など）" << std::endl;

    std::cout << "\n【結果の検証】" << std::endl;
    std::cout << "  - シルエットスコア（クラスタの品質評価）" << std::endl;
    std::cout << "  - クラスタ中心の確認（各クラスタの特徴を理解）" << std::endl;
    std::cout << "  - クラスタサイズの確認（極端な偏りがないか）" << std::endl;
    std::cout << "  - 可能であれば可視化（2次元または3次元プロット）" << std::endl;

    std::cout << "\n【実務への応用】" << std::endl;
    std::cout << "  ✓ 顧客セグメンテーション（マーケティング）" << std::endl;
    std::cout << "  ✓ 異常検知（孤立したクラスタの検出）" << std::endl;
    std::cout << "  ✓ 画像圧縮（色のクラスタリング）" << std::endl;
    std::cout << "  ✓ 文書分類（トピックのグループ化）" << std::endl;
    std::cout << "  ✓ 遺伝子発現パターンの分析" << std::endl;

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
