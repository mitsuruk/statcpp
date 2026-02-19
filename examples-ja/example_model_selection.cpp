/**
 * @file example_model_selection.cpp
 * @brief モデル選択のサンプルコード
 *
 * AIC、BIC、交差検証、LOOCV、リッジ回帰、Lasso回帰、
 * Elastic Net回帰等のモデル選択・正則化手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/model_selection.hpp"
#include "statcpp/linear_regression.hpp"

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
    // 1. 情報量基準（AIC, BIC）
    // ============================================================================
    print_section("1. 情報量基準 (AIC, BIC)");

    std::cout << R"(
【概念】
AIC (赤池情報量基準): モデルの適合度と複雑さのバランスを評価
BIC (ベイズ情報量基準): AICより複雑さにペナルティを課す

【実例: 単回帰モデルの評価】
データへの当てはまりとパラメータ数のトレードオフを評価
→ 値が小さいほど良いモデル
)";

    // サンプルデータ
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {2.1, 4.2, 5.9, 8.1, 10.3, 12.0, 14.2, 16.1, 18.0, 20.1};

    // 2D matrix format for cross-validation (with intercept column)
    std::vector<std::vector<double>> X_simple;
    for (double val : x) {
        X_simple.push_back({1.0, val});  // intercept, x
    }

    // 単回帰モデル
    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    print_subsection("単回帰モデル");
    std::cout << "  推定式: y = " << model.intercept << " + " << model.slope << "x\n";
    std::cout << "  決定係数 R² = " << model.r_squared << "\n";

    double aic_value = statcpp::aic_linear(model, x.size());
    double bic_value = statcpp::bic_linear(model, x.size());

    print_subsection("情報量基準");
    std::cout << "  AIC: " << aic_value << "\n";
    std::cout << "  BIC: " << bic_value << "\n";

    std::cout << "\n解釈:\n";
    std::cout << "  - 小さいほど良いモデル\n";
    std::cout << "  - BICはAICより複雑さにペナルティを課す\n";
    std::cout << "  - 複数モデルの比較に使用\n";

    // ============================================================================
    // 2. 交差検証（K分割）
    // ============================================================================
    print_section("2. K分割交差検証 (K-Fold Cross-Validation)");

    std::cout << R"(
【概念】
データをK個に分割し、K-1個で学習、1個で検証を繰り返す
予測性能を評価し、過学習を検出する

【実例: 5分割交差検証】
データを5分割し、各分割を1回ずつ検証用に使用
→ モデルの汎化性能を評価
)";

    std::size_t k_folds = 5;
    auto cv_result = statcpp::cross_validate_linear(X_simple, y, k_folds);

    print_subsection(std::to_string(k_folds) + "分割交差検証の結果");
    std::cout << "  平均MSE: " << cv_result.mean_error << "\n";
    std::cout << "  標準誤差: " << cv_result.se_error << "\n";

    std::cout << "\n各分割のスコア:\n";
    for (std::size_t i = 0; i < cv_result.fold_errors.size(); ++i) {
        std::cout << "  分割 " << (i + 1) << ": MSE = " << cv_result.fold_errors[i] << std::endl;
    }

    std::cout << "\n→ MSEが小さいほど予測性能が高い\n";

    // ============================================================================
    // 3. Leave-One-Out 交差検証（LOOCV）
    // ============================================================================
    print_section("3. Leave-One-Out交差検証 (LOOCV)");

    std::cout << R"(
【概念】
1つのデータを除いて学習、除いた1つで検証を全データで繰り返す
ほぼ不偏な推定値が得られるが計算コストが高い

【実例: データ数と同じ分割数】
n個のデータに対してn分割の交差検証
)";

    auto loocv_result = statcpp::loocv_linear(X_simple, y);

    print_subsection("LOOCV結果");
    std::cout << "  平均MSE: " << loocv_result.mean_error << "\n";
    std::cout << "  標準誤差: " << loocv_result.se_error << "\n";

    std::cout << "\n→ LOOCV = " << x.size() << "分割交差検証（各観測を1回ずつ検証用に使用）\n";
    std::cout << "→ ほぼ不偏だが計算コストが高い\n";

    // ============================================================================
    // 4. リッジ回帰（L2正則化）
    // ============================================================================
    print_section("4. リッジ回帰 (Ridge Regression / L2正則化)");

    std::cout << R"(
【概念】
係数の二乗和にペナルティを課して過学習を防ぐ
多重共線性のある変数でも安定した推定が可能

【実例: 多重共線性のあるデータ】
高度に相関する変数x2とx3を含むデータで
係数を縮小して安定化
)";

    // 多重共線性のあるデータ (切片は自動的に追加されます)
    std::vector<std::vector<double>> X = {
        {2.0, 2.1},  // x1とx2は高度に相関
        {3.0, 3.2},
        {4.0, 4.1},
        {5.0, 5.0},
        {6.0, 6.2},
        {7.0, 7.1},
        {8.0, 8.0},
        {9.0, 9.1}
    };
    std::vector<double> y_multi = {5, 8, 11, 14, 17, 20, 23, 26};

    double lambda = 1.0;
    auto ridge_model = statcpp::ridge_regression(X, y_multi, lambda);

    print_subsection("リッジ回帰モデル (λ = " + std::to_string(lambda) + ")");
    std::cout << "  係数: [";
    for (std::size_t i = 0; i < ridge_model.coefficients.size(); ++i) {
        std::cout << ridge_model.coefficients[i];
        if (i + 1 < ridge_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // λの値を変えて比較
    print_subsection("λの値が係数に与える影響");
    std::cout << "   λ       β₀       β₁       β₂\n";
    for (double test_lambda : {0.0, 0.5, 1.0, 5.0, 10.0}) {
        auto test_model = statcpp::ridge_regression(X, y_multi, test_lambda);
        std::cout << std::setw(4) << test_lambda << "  ";
        for (std::size_t i = 0; i < test_model.coefficients.size(); ++i) {
            std::cout << std::setw(8) << test_model.coefficients[i];
        }
        std::cout << std::endl;
    }

    std::cout << "\n→ λが大きいほど係数が0に近づく（縮小）\n";

    // ============================================================================
    // 5. Lasso回帰（L1正則化）
    // ============================================================================
    print_section("5. Lasso回帰 (Lasso Regression / L1正則化)");

    std::cout << R"(
【概念】
係数の絶対値和にペナルティを課す
一部の係数をちょうど0にする（変数選択）

【実例: 自動的な変数選択】
重要でない変数の係数を0にして
スパースなモデルを作成
)";

    auto lasso_model = statcpp::lasso_regression(X, y_multi, lambda);

    print_subsection("Lasso回帰モデル (λ = " + std::to_string(lambda) + ")");
    std::cout << "  係数: [";
    for (std::size_t i = 0; i < lasso_model.coefficients.size(); ++i) {
        std::cout << lasso_model.coefficients[i];
        if (i + 1 < lasso_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    print_subsection("λの値が係数に与える影響（変数選択）");
    std::cout << "   λ       β₀       β₁       β₂\n";
    for (double test_lambda : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        auto test_model = statcpp::lasso_regression(X, y_multi, test_lambda);
        std::cout << std::setw(4) << test_lambda << "  ";
        for (std::size_t i = 0; i < test_model.coefficients.size(); ++i) {
            std::cout << std::setw(8) << test_model.coefficients[i];
        }
        std::cout << std::endl;
    }

    std::cout << "\n→ Lassoは係数をちょうど0にできる（変数選択）\n";

    // ============================================================================
    // 6. Elastic Net回帰（L1+L2正則化）
    // ============================================================================
    print_section("6. Elastic Net回帰 (L1 + L2正則化)");

    std::cout << R"(
【概念】
RidgeとLassoの長所を組み合わせた手法
L1とL2ペナルティを同時に使用

【実例: ハイブリッドアプローチ】
変数選択と安定性を両立
相関する変数グループを同時に選択/除外
)";

    double alpha = 0.5;  // L1とL2のバランス
    auto enet_model = statcpp::elastic_net_regression(X, y_multi, lambda, alpha);

    print_subsection("Elastic Netモデル (λ = " + std::to_string(lambda) + ", α = " + std::to_string(alpha) + ")");
    std::cout << "  係数: [";
    for (std::size_t i = 0; i < enet_model.coefficients.size(); ++i) {
        std::cout << enet_model.coefficients[i];
        if (i + 1 < enet_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\nαパラメータの意味:\n";
    std::cout << "  α = 0: 純粋なRidge (L2のみ)\n";
    std::cout << "  α = 1: 純粋なLasso (L1のみ)\n";
    std::cout << "  0 < α < 1: 両方の混合\n";

    // ============================================================================
    // 7. 正則化パラメータの選択（交差検証）
    // ============================================================================
    print_section("7. 正則化パラメータλの選択（交差検証）");

    std::cout << R"(
【概念】
交差検証を使って最適なλを自動選択
予測誤差を最小化するλを探索

【実例: グリッドサーチ】
複数のλ候補から最良のものを選択
)";

    std::vector<double> lambda_grid = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    auto ridge_cv_result = statcpp::cv_ridge(X, y_multi, lambda_grid, 5);

    print_subsection("Ridge回帰の交差検証結果");
    std::cout << "  最適λ: " << ridge_cv_result.first << "\n";
    std::cout << "  最適係数: [";
    for (std::size_t i = 0; i < ridge_cv_result.second.size(); ++i) {
        std::cout << ridge_cv_result.second[i];
        if (i + 1 < ridge_cv_result.second.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\n→ このλが交差検証誤差を最小化\n";

    // ============================================================================
    // 8. まとめ：モデル選択のガイドライン
    // ============================================================================
    print_section("まとめ：モデル選択のガイドライン");

    std::cout << R"(
【正則化手法の使い分け】

Ridge回帰:
  - 適用場面: 多重共線性、すべての変数が重要
  - 効果: 係数を縮小、すべての変数を保持
  - 使用例: 多くの相関変数での予測

Lasso回帰:
  - 適用場面: 変数選択が必要、スパースなモデル
  - 効果: 一部の係数をちょうど0にする
  - 使用例: 高次元データ、解釈可能性重視

Elastic Net:
  - 適用場面: グループ選択、多数の変数
  - 効果: 両方の利点を組み合わせ
  - 使用例: p >> n、相関する変数グループ

【検証戦略】
- K分割交差検証: 汎用的（k=5または10）
- LOOCV: 小規模データ、不偏推定
- 訓練/テスト分割: 大規模データ、高速

【モデル比較基準】
- AIC: 予測重視、ペナルティ小
- BIC: 一致性、複雑さへのペナルティ大
- 交差検証: 予測性能を直接測定

【実用上のアドバイス】
1. まず標準化を適用
2. 交差検証でλを選択
3. 複数手法を比較
4. 解釈可能性も考慮
5. ドメイン知識を活用
)";

    return 0;
}
