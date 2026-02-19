/**
 * @file example_glm.cpp
 * @brief 一般化線形モデル（GLM）のサンプルコード
 *
 * ロジスティック回帰、ポアソン回帰、リンク関数、残差計算、
 * 過分散検定等の一般化線形モデルの使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include "statcpp/glm.hpp"

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
    // 1. ロジスティック回帰
    // ============================================================================
    print_section("1. ロジスティック回帰 (Logistic Regression)");

    std::cout << R"(
【概念】
二値応答変数（0/1）をモデル化
ロジット変換により線形モデルを適用

【実例: 試験の合否予測】
勉強時間から合格確率を予測
→ P(合格) = 1 / (1 + exp(-(β₀ + β₁×時間)))
)";

    // 例: 試験の合否予測（説明変数: 勉強時間）
    std::vector<std::vector<double>> X_logistic = {
        {1.0},  // study hours
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0},
        {7.0},
        {8.0},
        {9.0},
        {10.0}
    };

    std::vector<double> y_binary = {0, 0, 0, 1, 0, 1, 1, 1, 1, 1};  // pass/fail

    auto logistic_model = statcpp::logistic_regression(X_logistic, y_binary);

    print_subsection("ロジスティック回帰モデル");
    std::cout << "  係数:\n";
    std::cout << "    切片: " << logistic_model.coefficients[0] << "\n";
    std::cout << "    勉強時間: " << logistic_model.coefficients[1] << "\n";

    print_subsection("モデルの適合度");
    std::cout << "  残差逸脱度: " << logistic_model.residual_deviance << "\n";
    std::cout << "  ヌル逸脱度: " << logistic_model.null_deviance << "\n";
    std::cout << "  収束: " << (logistic_model.converged ? "はい" : "いいえ") << "\n";
    std::cout << "  反復回数: " << logistic_model.iterations << "\n";

    // McFadden's Pseudo R²
    double pseudo_r2 = statcpp::pseudo_r_squared_mcfadden(logistic_model);
    std::cout << "  McFaddenの疑似R²: " << pseudo_r2 << "\n";
    std::cout << "  → 1に近いほど良い適合\n";

    // オッズ比を計算
    auto odds_ratios = statcpp::odds_ratios(logistic_model);
    print_subsection("オッズ比");
    for (std::size_t i = 0; i < odds_ratios.size(); ++i) {
        std::cout << "  係数 " << (i + 1) << ": OR = " << odds_ratios[i];
        if (i == 0) {
            std::cout << " (勉強時間1時間増加あたり)";
        }
        std::cout << std::endl;
    }
    std::cout << "  → OR > 1: 正の効果、OR < 1: 負の効果\n";

    // 予測確率
    print_subsection("予測確率");
    for (double hours = 3.0; hours <= 8.0; hours += 1.0) {
        std::vector<double> x_new = {hours};
        double prob = statcpp::predict_probability(logistic_model, x_new);
        std::cout << "  勉強時間 " << hours << "時間: P(合格) = " << prob << std::endl;
    }
    std::cout << "  → 勉強時間が増えるほど合格確率が上昇\n";

    // ============================================================================
    // 2. ポアソン回帰
    // ============================================================================
    print_section("2. ポアソン回帰 (Poisson Regression)");

    std::cout << R"(
【概念】
カウントデータ（非負整数）をモデル化
対数リンク関数により線形モデルを適用

【実例: ウェブサイト訪問数の予測】
広告費から訪問者数を予測
→ log(E[訪問数]) = β₀ + β₁×広告費
)";

    // 例: ウェブサイト訪問数の予測（説明変数: 広告費）
    std::vector<std::vector<double>> X_poisson = {
        {10.0},  // ad budget (in thousands)
        {15.0},
        {20.0},
        {25.0},
        {30.0},
        {35.0},
        {40.0},
        {45.0}
    };

    std::vector<double> y_count = {12, 18, 25, 30, 38, 42, 50, 55};  // visitor counts

    auto poisson_model = statcpp::poisson_regression(X_poisson, y_count);

    print_subsection("ポアソン回帰モデル");
    std::cout << "  係数:\n";
    std::cout << "    切片: " << poisson_model.coefficients[0] << "\n";
    std::cout << "    広告費: " << poisson_model.coefficients[1] << "\n";

    print_subsection("モデルの適合度");
    std::cout << "  残差逸脱度: " << poisson_model.residual_deviance << "\n";
    std::cout << "  収束: " << (poisson_model.converged ? "はい" : "いいえ") << "\n";

    // 発生率比（Incidence Rate Ratios）
    auto irr = statcpp::incidence_rate_ratios(poisson_model);
    print_subsection("発生率比 (IRR)");
    for (std::size_t i = 0; i < irr.size(); ++i) {
        std::cout << "  係数 " << (i + 1) << ": IRR = " << irr[i];
        if (i == 0) {
            std::cout << " (広告費$1000増加あたり)";
        }
        std::cout << std::endl;
    }
    std::cout << "  → IRR > 1: 正の効果（カウント増加）\n";

    // カウント予測
    print_subsection("予測カウント");
    for (double budget = 20.0; budget <= 40.0; budget += 5.0) {
        std::vector<double> x_new = {budget};
        double count = statcpp::predict_count(poisson_model, x_new);
        std::cout << "  広告費$" << budget << "k: E[訪問者数] = " << count << "\n";
    }
    std::cout << "  → 広告費が増えるほど訪問者数が増加\n";

    // ============================================================================
    // 3. リンク関数の比較
    // ============================================================================
    print_section("3. リンク関数 (Link Functions)");

    std::cout << R"(
【概念】
応答変数の期待値と線形予測子を結びつける関数
異なるリンク関数で異なる仮定をモデル化

【主なリンク関数】
- Logit: 二項分布に最適
- Probit: 正規分布の累積密度関数
- Identity: 通常の線形回帰
- Log: ポアソン回帰
)";

    double mu_value = 0.7;

    print_subsection("順変換（μ → η）");
    std::cout << "μ = " << mu_value << " のとき:\n";
    std::cout << "  Logit: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::logit) << "\n";
    std::cout << "  Probit: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::probit) << "\n";
    std::cout << "  Identity: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::identity) << "\n";

    double eta_value = 0.5;
    print_subsection("逆変換（η → μ）");
    std::cout << "η = " << eta_value << " のとき:\n";
    std::cout << "  逆Logit: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::logit) << "\n";
    std::cout << "  逆Probit: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::probit) << "\n";
    std::cout << "  Identity: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::identity) << "\n";

    // ============================================================================
    // 4. GLM残差の計算
    // ============================================================================
    print_section("4. GLM残差 (Residuals)");

    std::cout << R"(
【概念】
GLMの残差には複数の種類がある
各タイプが異なる診断情報を提供

【残差の種類】
- 応答残差: 観測値と予測値の差
- ピアソン残差: 標準化残差
- 逸脱度残差: 逸脱度への寄与
)";

    auto residuals = statcpp::compute_glm_residuals(logistic_model, X_logistic, y_binary);

    print_subsection("ロジスティック回帰の残差");
    std::cout << "  観測   応答残差  ピアソン  逸脱度\n";
    for (std::size_t i = 0; i < std::min(std::size_t(5), residuals.response.size()); ++i) {
        std::cout << "   " << std::setw(2) << i
                  << "    " << std::setw(8) << residuals.response[i]
                  << "  " << std::setw(8) << residuals.pearson[i]
                  << "  " << std::setw(8) << residuals.deviance[i]
                  << std::endl;
    }

    std::cout << "\n残差の解釈:\n";
    std::cout << "  - 応答残差: y - 予測値\n";
    std::cout << "  - ピアソン残差: 標準化された残差\n";
    std::cout << "  - 逸脱度残差: モデル逸脱度への寄与\n";

    // ============================================================================
    // 5. 過分散の検定
    // ============================================================================
    print_section("5. 過分散の検定 (Overdispersion Test)");

    std::cout << R"(
【概念】
ポアソン分布の仮定が適切かを評価
分散が期待値より大きい場合、過分散が発生

【実例: ポアソンモデルの診断】
分散パラメータが1に近いかを確認
→ 1より大きい場合は負の二項分布を検討
)";

    double disp_stat = statcpp::overdispersion_test(poisson_model, X_poisson, y_count);

    print_subsection("分散統計量");
    std::cout << "  分散パラメータ: " << disp_stat << "\n";

    std::cout << "\n解釈:\n";
    std::cout << "  ≈ 1.0: 過分散なし（ポアソンが適切）\n";
    std::cout << "  > 1.0: 過分散あり（負の二項分布を検討）\n";
    std::cout << "  < 1.0: 過小分散（稀）\n";

    if (disp_stat > 1.5) {
        std::cout << "\n結果: 有意な過分散が検出されました\n";
    } else {
        std::cout << "\n結果: 有意な過分散はありません\n";
    }

    // ============================================================================
    // 6. まとめ：GLMファミリーの選択
    // ============================================================================
    print_section("まとめ：GLMファミリーとリンク関数の選択");

    std::cout << R"(
【応答変数のタイプ別のGLM選択】

二値/二項データ (0/1, 成功/失敗):
  - ファミリー: 二項分布 (Binomial)
  - リンク関数: Logit（最も一般的）またはProbit
  - 例: 疾患の有無、ローン債務不履行、クリック有無

カウントデータ (0, 1, 2, ...):
  - ファミリー: ポアソン分布 (Poisson)
  - リンク関数: Log
  - 例: イベント発生回数、来店客数、エラー数
  - 注意: 過分散の有無を確認

連続データ（正の値のみ）:
  - ファミリー: ガンマ分布 (Gamma)
  - リンク関数: LogまたはInverse
  - 例: 保険金請求額、生存時間、待ち時間

連続データ（任意の実数）:
  - ファミリー: 正規分布 (Gaussian/Normal)
  - リンク関数: Identity
  - 備考: これが通常の線形回帰

【モデル診断のチェックリスト】

1. 収束の確認:
   - アルゴリズムは収束したか？
   - ロジスティック: )" << (logistic_model.converged ? "収束" : "未収束") << R"(
   - ポアソン: )" << (poisson_model.converged ? "収束" : "未収束") << R"(

2. 残差の検証:
   - 残差プロットにパターンがないか確認
   - 外れ値をチェック

3. 過分散の検定（カウントモデル）:
   - 分散パラメータ ≈ 1 か？

4. モデルの適合度評価:
   - 逸脱度と自由度を比較
   - 二項モデルでは疑似R²を使用

5. 予測の検証:
   - 予測値が妥当か確認
   - データが十分なら交差検証を実施

【GLMの利点】
- 応答変数の分布を適切にモデル化
- 線形モデルの柔軟な拡張
- 最尤推定による統計的推論
- 多様な応用分野で利用可能
)";

    return 0;
}
