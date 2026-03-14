/**
 * @file example_multivariate.cpp
 * @brief 多変量解析のサンプルコード
 *
 * 共分散行列、相関行列、データ標準化、Min-Maxスケーリング、
 * 主成分分析（PCA）等の多変量解析手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/multivariate.hpp"

// ============================================================================
// 結果表示用のヘルパー関数
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 共分散行列と相関行列
    // ============================================================================
    print_section("1. 共分散行列と相関行列 (Covariance and Correlation Matrices)");

    std::cout << R"(
【概念】
共分散行列: 複数の変数間の共分散をまとめた行列
相関行列: 変数間の相関係数（標準化された共分散）をまとめた行列

【実例: 健康データ】
8人の被験者について、身長・体重・年齢の3変数を測定
→ 変数間の関係性を分析
)";

    // サンプルデータ: 3変数（身長、体重、年齢）
    std::vector<std::vector<double>> data = {
        {170, 65, 25},
        {175, 70, 30},
        {165, 60, 22},
        {180, 75, 35},
        {172, 68, 28},
        {168, 63, 24},
        {177, 72, 32},
        {171, 67, 27}
    };

    std::cout << "\nサンプルデータ (n=" << data.size() << "人, p=" << data[0].size() << "変数):\n";
    std::cout << "  身長(cm), 体重(kg), 年齢(歳)\n";

    auto cov_matrix = statcpp::covariance_matrix(data);
    auto corr_matrix = statcpp::correlation_matrix(data);

    print_subsection("共分散行列");
    for (std::size_t i = 0; i < cov_matrix.size(); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < cov_matrix[i].size(); ++j) {
            std::cout << std::setw(10) << cov_matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "→ 対角要素は各変数の分散、非対角要素は共分散\n";

    print_subsection("相関行列");
    const char* var_names[] = {"身長", "体重", "年齢"};
    std::cout << "           ";
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << std::setw(10) << var_names[i];
    }
    std::cout << std::endl;

    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        std::cout << std::setw(10) << var_names[i] << " ";
        for (std::size_t j = 0; j < corr_matrix[i].size(); ++j) {
            std::cout << std::setw(10) << corr_matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "→ -1〜1の範囲で、1に近いほど強い正の相関\n";

    // ============================================================================
    // 2. データの標準化
    // ============================================================================
    print_section("2. データの標準化 (Data Standardization / Z-score)");

    std::cout << R"(
【概念】
各変数を平均0、標準偏差1に変換する処理
スケールの異なる変数を比較可能にする

【実例: 異なる単位の変数】
身長(cm)、体重(kg)、年齢(歳)のように異なる単位の変数を
同じスケールに揃えて比較・分析する
)";

    auto standardized = statcpp::standardize(data);

    print_subsection("元のデータ（最初の3行）");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < data[i].size(); ++j) {
            std::cout << std::setw(7) << data[i][j];
            if (j + 1 < data[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    print_subsection("標準化後のデータ（最初の3行）");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < standardized[i].size(); ++j) {
            std::cout << std::setw(7) << std::setprecision(2) << standardized[i][j];
            if (j + 1 < standardized[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << std::setprecision(4);
    std::cout << "→ 各変数が平均=0、標準偏差=1に変換される\n";
    std::cout << "→ 外れ値の影響を受けやすい点に注意\n";

    // ============================================================================
    // 3. Min-Max スケーリング
    // ============================================================================
    print_section("3. Min-Maxスケーリング (0-1正規化)");

    std::cout << R"(
【概念】
各変数を最小値0、最大値1に変換する処理
データの範囲を[0,1]に統一する

【実例: 機械学習の前処理】
ニューラルネットワークや距離ベースのアルゴリズムで
変数のスケールを揃える
)";

    auto scaled = statcpp::min_max_scale(data);

    print_subsection("Min-Maxスケーリング後（最初の3行）");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < scaled[i].size(); ++j) {
            std::cout << std::setw(7) << scaled[i][j];
            if (j + 1 < scaled[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "→ すべての値が[0, 1]の範囲に収まる\n";
    std::cout << "→ 外れ値の影響を強く受ける点に注意\n";

    // ============================================================================
    // 4. 主成分分析 (PCA)
    // ============================================================================
    print_section("4. 主成分分析 (Principal Component Analysis)");

    std::cout << R"(
【概念】
多変量データを少数の主成分（線形結合）に要約する手法
データの分散を最大化する新しい軸を見つける

【実例: 次元削減】
3変数のデータを2つの主成分に圧縮
→ 可視化や計算効率の向上
)";

    std::size_t n_components = 2;
    auto pca_result = statcpp::pca(data, n_components);

    print_subsection("PCA結果（" + std::to_string(n_components) + "主成分）");

    std::cout << "\n主成分負荷量 (loadings):\n";
    for (std::size_t i = 0; i < pca_result.components.size(); ++i) {
        std::cout << "  第" << (i + 1) << "主成分: [";
        for (std::size_t j = 0; j < pca_result.components[i].size(); ++j) {
            std::cout << std::setw(8) << pca_result.components[i][j];
            if (j + 1 < pca_result.components[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "→ 各変数の主成分への寄与度\n";

    std::cout << "\n説明された分散:\n";
    for (std::size_t i = 0; i < pca_result.explained_variance.size(); ++i) {
        std::cout << "  第" << (i + 1) << "主成分: " << pca_result.explained_variance[i] << std::endl;
    }

    std::cout << "\n分散寄与率:\n";
    for (std::size_t i = 0; i < pca_result.explained_variance_ratio.size(); ++i) {
        std::cout << "  第" << (i + 1) << "主成分: " << (pca_result.explained_variance_ratio[i] * 100)
                  << "%" << std::endl;
    }

    double total_explained = 0.0;
    for (double ratio : pca_result.explained_variance_ratio) {
        total_explained += ratio;
    }
    std::cout << "\n累積寄与率: " << (total_explained * 100) << "%\n";
    std::cout << "→ " << n_components << "主成分で全体の" << (total_explained * 100) << "%の情報を保持\n";

    // ============================================================================
    // 5. PCA変換（次元削減）
    // ============================================================================
    print_section("5. PCA変換（次元削減）");

    std::cout << R"(
【概念】
元のデータを主成分空間に変換
高次元データを低次元に射影

【実例: 可視化のための次元削減】
3次元データを2次元に変換して散布図を作成
)";

    auto transformed = statcpp::pca_transform(data, pca_result);

    print_subsection("次元削減の結果");
    std::cout << "元の次元数: " << data[0].size() << "変数\n";
    std::cout << "削減後の次元数: " << transformed[0].size() << "主成分\n";

    std::cout << "\n変換後のデータ（最初の5観測）:\n";
    std::cout << "   第1主成分  第2主成分\n";
    for (std::size_t i = 0; i < std::min(std::size_t(5), transformed.size()); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < transformed[i].size(); ++j) {
            std::cout << std::setw(11) << transformed[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "→ 元の3変数が2つの主成分に集約された\n";

    // ============================================================================
    // 6. PCA の実用例
    // ============================================================================
    print_section("6. 実用例: 特徴抽出と成分数の決定");

    std::cout << R"(
【概念】
必要な主成分数を決定する方法
累積寄与率が目標値（例: 90%）を超える成分数を選択

【実例: 多変量データの圧縮】
4変数のデータから最小限の主成分を選択
)";

    // 多次元データの例
    std::vector<std::vector<double>> high_dim_data = {
        {2.5, 2.4, 3.1, 2.9},
        {0.5, 0.7, 0.9, 0.6},
        {2.2, 2.9, 2.7, 2.5},
        {1.9, 2.2, 2.4, 2.0},
        {3.1, 3.0, 3.5, 3.2},
        {2.3, 2.7, 2.8, 2.4},
        {2.0, 1.6, 1.8, 2.1},
        {1.0, 1.1, 1.3, 0.9},
        {1.5, 1.6, 1.7, 1.4},
        {1.1, 0.9, 1.0, 1.2}
    };

    std::cout << "\n元のデータ: " << high_dim_data.size() << "観測、"
              << high_dim_data[0].size() << "変数\n";

    // 分散の累積寄与率が90%以上になる成分数を決定
    auto full_pca = statcpp::pca(high_dim_data, high_dim_data[0].size());

    print_subsection("各主成分の寄与率");
    double cumulative_var = 0.0;
    std::size_t components_for_90 = 0;
    for (std::size_t i = 0; i < full_pca.explained_variance_ratio.size(); ++i) {
        cumulative_var += full_pca.explained_variance_ratio[i];
        std::cout << "  第" << (i + 1) << "主成分: "
                  << (full_pca.explained_variance_ratio[i] * 100) << "% "
                  << "(累積: " << (cumulative_var * 100) << "%)" << std::endl;
        if (cumulative_var >= 0.9 && components_for_90 == 0) {
            components_for_90 = i + 1;
        }
    }

    std::cout << "\n結論: 分散の90%を捕捉するには " << components_for_90 << "主成分が必要\n";
    std::cout << "→ 次元数を " << high_dim_data[0].size()
              << "から" << components_for_90 << "に削減可能\n";

    // ============================================================================
    // 7. 相関分析の解釈
    // ============================================================================
    print_section("7. 相関行列の解釈");

    std::cout << R"(
【概念】
相関係数の強さの目安
変数間の関係性を定量的に評価

【解釈の基準】
|r| < 0.3: 弱い相関
0.3 ≤ |r| < 0.7: 中程度の相関
|r| ≥ 0.7: 強い相関
)";

    print_subsection("変数間の関係分析");
    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        for (std::size_t j = i + 1; j < corr_matrix[i].size(); ++j) {
            double r = corr_matrix[i][j];
            std::cout << "  " << var_names[i] << " vs " << var_names[j] << ": r = " << r;
            if (std::abs(r) >= 0.7) {
                std::cout << " (強い相関)";
            } else if (std::abs(r) >= 0.3) {
                std::cout << " (中程度の相関)";
            } else {
                std::cout << " (弱い相関)";
            }
            std::cout << std::endl;
        }
    }

    // ============================================================================
    // 8. まとめ：多変量解析の使用ガイドライン
    // ============================================================================
    print_section("まとめ：多変量解析の使用ガイドライン");

    std::cout << R"(
【PCAを使うべき場面】
- 変数が多すぎる（次元の呪い）
- 変数間に相関がある
- 高次元データを可視化したい
- ノイズを削減したい
- 他のアルゴリズムの前処理として

【PCA適用前の準備】
1. 特徴量を標準化（特にスケールが異なる場合）
2. 外れ値をチェック
3. 線形関係を確認

【主成分数の選び方】
- 累積寄与率（例: 80-95%）
- スクリープロット（エルボー法）
- Kaiser基準（固有値 > 1）
- 分野知識と解釈可能性

【主成分の解釈】
- 負荷量は各変数の寄与度を示す
- 主成分は互いに直交（無相関）
- 第1主成分が最も分散を捕捉

【データ変換の使い分け】
┌──────────────┬────────────────────────────────┐
│ 手法         │ 使用場面                       │
├──────────────┼────────────────────────────────┤
│ 標準化       │ 異なるスケールの変数を比較     │
│ (Z-score)    │ 外れ値に敏感                   │
├──────────────┼────────────────────────────────┤
│ Min-Max      │ [0,1]範囲への正規化            │
│ スケーリング │ ニューラルネット等の前処理     │
├──────────────┼────────────────────────────────┤
│ PCA          │ 次元削減、ノイズ除去           │
│              │ 可視化、計算効率向上           │
└──────────────┴────────────────────────────────┘
)";

    return 0;
}
