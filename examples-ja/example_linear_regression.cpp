/**
 * @file example_linear_regression.cpp
 * @brief statcpp::linear_regression.hpp のサンプルコード
 *
 * このファイルでは、linear_regression.hpp で提供される
 * 線形回帰分析を行う関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - simple_linear_regression()      : 単回帰分析
 * - multiple_linear_regression()    : 重回帰分析
 * - predict()                       : 予測
 * - prediction_interval_simple()    : 予測区間（単回帰）
 * - confidence_interval_mean()      : 平均の信頼区間（単回帰）
 * - compute_residual_diagnostics()  : 残差診断
 * - compute_vif()                   : 分散膨張係数(VIF)
 * - r_squared()                     : 決定係数
 * - adjusted_r_squared()            : 自由度調整済み決定係数
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_linear_regression.cpp -o example_linear_regression
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// statcpp の線形回帰ヘッダー
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

template <typename T>
void print_data(const std::string& label, const std::vector<T>& data) {
    std::cout << label << ": ";
    for (const auto& d : data) std::cout << d << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. simple_linear_regression() - 単回帰分析
// ============================================================================

/**
 * @brief simple_linear_regression() の使用例
 *
 * 【目的】
 * 単回帰分析は、1つの説明変数(x)から目的変数(y)を予測するモデルを構築します。
 *
 * 【数式】
 * y = β₀ + β₁x + ε
 * - β₀: 切片（intercept）
 * - β₁: 傾き（slope）
 * - ε: 誤差項
 *
 * 【戻り値】
 * simple_regression_result 構造体:
 * - intercept, slope: 回帰係数
 * - intercept_se, slope_se: 標準誤差
 * - intercept_t, slope_t: t統計量
 * - intercept_p, slope_p: p値
 * - r_squared: 決定係数 R²
 * - adj_r_squared: 自由度調整済みR²
 * - residual_se: 残差の標準誤差
 * - f_statistic, f_p_value: F検定の結果
 *
 * 【使用場面】
 * - 2変数間の関係のモデル化
 * - 予測モデルの構築
 * - 因果関係の分析（因果関係の証明には注意が必要）
 */
void example_simple_regression() {
    print_section("1. simple_linear_regression() - 単回帰分析");

    // 広告費用と売上の関係
    std::vector<double> ad_spend = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};  // 万円
    std::vector<double> sales = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450}; // 万円

    std::cout << "データ: 広告費用と売上の関係\n";
    print_data("広告費用 (万円)", ad_spend);
    print_data("売上 (万円)", sales);

    auto result = statcpp::simple_linear_regression(
        ad_spend.begin(), ad_spend.end(),
        sales.begin(), sales.end());

    print_subsection("回帰結果");
    std::cout << "回帰式: 売上 = " << result.intercept << " + "
              << result.slope << " × 広告費用\n\n";

    std::cout << "係数の詳細:\n";
    std::cout << std::setw(12) << "係数"
              << std::setw(12) << "推定値"
              << std::setw(12) << "標準誤差"
              << std::setw(10) << "t値"
              << std::setw(12) << "p値" << "\n";
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::setw(12) << "切片"
              << std::setw(12) << result.intercept
              << std::setw(12) << result.intercept_se
              << std::setw(10) << result.intercept_t
              << std::setw(12) << result.intercept_p << "\n";
    std::cout << std::setw(12) << "傾き"
              << std::setw(12) << result.slope
              << std::setw(12) << result.slope_se
              << std::setw(10) << result.slope_t
              << std::setw(12) << result.slope_p << "\n";

    print_subsection("モデルの適合度");
    std::cout << "決定係数 (R²):           " << result.r_squared << "\n";
    std::cout << "自由度調整済みR²:        " << result.adj_r_squared << "\n";
    std::cout << "残差の標準誤差:          " << result.residual_se << "万円\n";
    std::cout << "F統計量:                 " << result.f_statistic << "\n";
    std::cout << "F検定のp値:              " << result.f_p_value << "\n";

    print_subsection("解釈");
    std::cout << "- 広告費用が1万円増えると、売上は約"
              << result.slope << "万円増加\n";
    std::cout << "- R² = " << result.r_squared
              << " → 売上の変動の" << (result.r_squared * 100)
              << "%を広告費用で説明可能\n";
    std::cout << "- p値 < 0.05 → 傾きは統計的に有意\n";
}

// ============================================================================
// 2. predict() と予測区間
// ============================================================================

/**
 * @brief predict() と予測区間の使用例
 *
 * 【目的】
 * モデルを使って新しいxの値に対するyを予測します。
 * 予測区間は、個々の観測値がどの範囲に収まるかを示します。
 * 信頼区間は、平均的なyがどの範囲に収まるかを示します。
 */
void example_prediction() {
    print_section("2. predict() と予測区間");

    std::vector<double> x = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> y = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450};

    auto model = statcpp::simple_linear_regression(
        x.begin(), x.end(),
        y.begin(), y.end());

    // 点予測
    print_subsection("点予測");
    double x_new = 55.0;  // 広告費55万円
    double y_pred = statcpp::predict(model, x_new);
    std::cout << "広告費用 " << x_new << "万円での売上予測: " << y_pred << "万円\n";

    // 予測区間
    print_subsection("予測区間（95%）");
    auto pred_int = statcpp::prediction_interval_simple(
        model, x.begin(), x.end(), x_new, 0.95);
    std::cout << "予測値: " << pred_int.prediction << "万円\n";
    std::cout << "95%予測区間: [" << pred_int.lower << ", " << pred_int.upper << "]万円\n";
    std::cout << "予測の標準誤差: " << pred_int.se_prediction << "万円\n";
    std::cout << "→ 新しい観測値が95%の確率でこの区間に収まる\n";

    // 平均の信頼区間
    print_subsection("平均の信頼区間（95%）");
    auto conf_int = statcpp::confidence_interval_mean(
        model, x.begin(), x.end(), x_new, 0.95);
    std::cout << "予測値: " << conf_int.prediction << "万円\n";
    std::cout << "95%信頼区間: [" << conf_int.lower << ", " << conf_int.upper << "]万円\n";
    std::cout << "→ 広告費" << x_new << "万円時の平均売上が95%の確率でこの区間に収まる\n";

    print_subsection("予測区間 vs 信頼区間");
    std::cout << "予測区間: 個々の観測値の範囲（より広い）\n";
    std::cout << "信頼区間: 平均値の範囲（より狭い）\n";
}

// ============================================================================
// 3. multiple_linear_regression() - 重回帰分析
// ============================================================================

/**
 * @brief multiple_linear_regression() の使用例
 *
 * 【目的】
 * 重回帰分析は、複数の説明変数から目的変数を予測するモデルを構築します。
 *
 * 【数式】
 * y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
 *
 * 【注意】
 * - 切片(β₀)は自動的に追加されます
 * - Xは n×p 行列（n: 観測数、p: 説明変数の数）
 */
void example_multiple_regression() {
    print_section("3. multiple_linear_regression() - 重回帰分析");

    // 住宅価格予測: 面積、築年数、最寄り駅からの距離
    // X[i] = {面積(㎡), 築年数(年), 駅距離(分)}
    std::vector<std::vector<double>> X = {
        {60, 10, 5},   {70, 5, 8},   {80, 15, 3},  {65, 8, 10},
        {75, 12, 6},   {90, 3, 4},   {55, 20, 12}, {85, 7, 2},
        {72, 10, 7},   {68, 6, 9},   {95, 2, 5},   {58, 18, 15}
    };
    std::vector<double> price = {
        3200, 3800, 3500, 2900,
        3400, 4500, 2500, 4200,
        3300, 3100, 4800, 2200
    };  // 万円

    std::cout << "データ: 住宅価格予測\n";
    std::cout << "説明変数: 面積(㎡), 築年数(年), 駅距離(分)\n";
    std::cout << "目的変数: 価格(万円)\n\n";

    for (std::size_t i = 0; i < X.size(); ++i) {
        std::cout << "  物件" << (i + 1) << ": 面積=" << X[i][0]
                  << "㎡, 築年数=" << X[i][1] << "年, 駅距離=" << X[i][2]
                  << "分 → " << price[i] << "万円\n";
    }

    auto result = statcpp::multiple_linear_regression(X, price);

    print_subsection("回帰結果");
    std::cout << "回帰式: 価格 = " << result.coefficients[0]
              << " + " << result.coefficients[1] << " × 面積"
              << " + " << result.coefficients[2] << " × 築年数"
              << " + " << result.coefficients[3] << " × 駅距離\n\n";

    std::cout << "係数の詳細:\n";
    std::cout << std::setw(12) << "変数"
              << std::setw(12) << "係数"
              << std::setw(12) << "標準誤差"
              << std::setw(10) << "t値"
              << std::setw(12) << "p値" << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<std::string> var_names = {"切片", "面積", "築年数", "駅距離"};
    for (std::size_t i = 0; i < result.coefficients.size(); ++i) {
        std::cout << std::setw(12) << var_names[i]
                  << std::setw(12) << result.coefficients[i]
                  << std::setw(12) << result.coefficient_se[i]
                  << std::setw(10) << result.t_statistics[i]
                  << std::setw(12) << result.p_values[i] << "\n";
    }

    print_subsection("モデルの適合度");
    std::cout << "決定係数 (R²):           " << result.r_squared << "\n";
    std::cout << "自由度調整済みR²:        " << result.adj_r_squared << "\n";
    std::cout << "F統計量:                 " << result.f_statistic << "\n";
    std::cout << "F検定のp値:              " << result.f_p_value << "\n";

    // 新しい物件の予測
    print_subsection("新しい物件の価格予測");
    std::vector<double> new_property = {75, 8, 6};  // 75㎡, 築8年, 駅6分
    double pred_price = statcpp::predict(result, new_property);
    std::cout << "物件: 面積=" << new_property[0] << "㎡, 築年数=" << new_property[1]
              << "年, 駅距離=" << new_property[2] << "分\n";
    std::cout << "予測価格: " << pred_price << "万円\n";

    print_subsection("係数の解釈");
    std::cout << "- 面積が1㎡増えると価格は約" << result.coefficients[1] << "万円上昇\n";
    std::cout << "- 築年数が1年増えると価格は約" << result.coefficients[2] << "万円変化\n";
    std::cout << "- 駅距離が1分増えると価格は約" << result.coefficients[3] << "万円変化\n";
}

// ============================================================================
// 4. compute_residual_diagnostics() - 残差診断
// ============================================================================

/**
 * @brief compute_residual_diagnostics() の使用例
 *
 * 【目的】
 * 残差診断は、回帰モデルの仮定が満たされているかを確認します。
 *
 * 【診断項目】
 * - 残差: 実測値 - 予測値
 * - 標準化残差: 残差 / 残差の標準誤差
 * - スチューデント化残差: 各点を除いて計算した標準化残差
 * - てこ比(leverage): その観測値がモデルに与える影響度
 * - クックの距離: 外れ値の影響度
 * - ダービン・ワトソン統計量: 残差の自己相関
 */
void example_residual_diagnostics() {
    print_section("4. compute_residual_diagnostics() - 残差診断");

    std::vector<double> x = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> y = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450};

    auto model = statcpp::simple_linear_regression(
        x.begin(), x.end(),
        y.begin(), y.end());

    auto diag = statcpp::compute_residual_diagnostics(
        model, x.begin(), x.end(), y.begin(), y.end());

    print_subsection("残差の詳細");
    std::cout << std::setw(8) << "X"
              << std::setw(10) << "Y"
              << std::setw(12) << "残差"
              << std::setw(12) << "標準化残差"
              << std::setw(10) << "てこ比"
              << std::setw(12) << "Cook's D" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << std::setw(8) << x[i]
                  << std::setw(10) << y[i]
                  << std::setw(12) << diag.residuals[i]
                  << std::setw(12) << diag.standardized_residuals[i]
                  << std::setw(10) << diag.hat_values[i]
                  << std::setw(12) << diag.cooks_distance[i] << "\n";
    }

    print_subsection("診断統計量");
    std::cout << "ダービン・ワトソン統計量: " << diag.durbin_watson << "\n";

    print_subsection("診断の解釈");
    std::cout << R"(
【標準化残差】
- |標準化残差| > 2 の観測値は外れ値の可能性
- |標準化残差| > 3 の観測値は重大な外れ値

【てこ比 (Leverage)】
- 高いてこ比は回帰に大きな影響を与える
- 目安: 2(p+1)/n より大きい場合は注意
  (p: 説明変数の数、n: 観測数)

【クックの距離】
- Cook's D > 1 または > 4/n の観測値は影響力が大きい
- その観測値を除くとモデルが大きく変わる可能性

【ダービン・ワトソン統計量】
- DW ≈ 2: 残差に自己相関なし（理想的）
- DW < 2: 正の自己相関の可能性
- DW > 2: 負の自己相関の可能性
)";
}

// ============================================================================
// 5. compute_vif() - 分散膨張係数
// ============================================================================

/**
 * @brief compute_vif() の使用例
 *
 * 【目的】
 * VIF (Variance Inflation Factor) は、説明変数間の多重共線性を検出します。
 * 多重共線性があると、係数の推定が不安定になります。
 *
 * 【数式】
 * VIF_j = 1 / (1 - R²_j)
 * R²_j は j番目の変数を他の変数で回帰した時の決定係数
 *
 * 【目安】
 * - VIF < 5: 問題なし
 * - 5 ≤ VIF < 10: やや問題あり
 * - VIF ≥ 10: 重大な多重共線性
 */
void example_vif() {
    print_section("5. compute_vif() - 分散膨張係数（多重共線性）");

    // 多重共線性がある例
    // 面積(㎡)と部屋数は強く相関
    std::vector<std::vector<double>> X = {
        {60, 2, 10},   {70, 3, 5},    {80, 3, 15},   {65, 2, 8},
        {75, 3, 12},   {90, 4, 3},    {55, 2, 20},   {85, 4, 7},
        {72, 3, 10},   {68, 3, 6},    {95, 4, 2},    {58, 2, 18}
    };  // 面積, 部屋数, 築年数

    std::cout << "説明変数: 面積(㎡), 部屋数, 築年数(年)\n";
    std::cout << "（面積と部屋数は強く相関している可能性あり）\n\n";

    auto vif = statcpp::compute_vif(X);

    std::vector<std::string> var_names = {"面積", "部屋数", "築年数"};

    std::cout << "VIF (分散膨張係数):\n";
    std::cout << std::setw(12) << "変数" << std::setw(12) << "VIF" << "\n";
    std::cout << std::string(24, '-') << "\n";

    for (std::size_t i = 0; i < vif.size(); ++i) {
        std::cout << std::setw(12) << var_names[i]
                  << std::setw(12) << vif[i] << "\n";
    }

    std::cout << "\n【判定基準】\n";
    std::cout << "- VIF < 5: 問題なし\n";
    std::cout << "- 5 ≤ VIF < 10: やや問題あり\n";
    std::cout << "- VIF ≥ 10: 重大な多重共線性\n";

    bool has_problem = false;
    for (std::size_t i = 0; i < vif.size(); ++i) {
        if (vif[i] >= 5.0) {
            std::cout << "\n警告: " << var_names[i] << " のVIFが高い ("
                      << vif[i] << ")\n";
            has_problem = true;
        }
    }
    if (!has_problem) {
        std::cout << "\n→ 多重共線性の問題はなさそうです\n";
    }
}

// ============================================================================
// 6. R²の比較と自由度調整
// ============================================================================

/**
 * @brief 決定係数の計算と比較
 *
 * 【R²（決定係数）】
 * モデルがデータの変動をどれだけ説明できるかを示す
 * R² = 1 - SS_residual / SS_total
 *
 * 【調整済みR²】
 * 変数の数を考慮して調整
 * 変数を追加しても必ずしも増加しない
 */
void example_r_squared() {
    print_section("6. 決定係数 R² と調整済み R²");

    std::vector<double> y_actual = {100, 120, 140, 160, 180};
    std::vector<double> y_pred1 = {105, 115, 145, 155, 180};  // 良い予測
    std::vector<double> y_pred2 = {130, 130, 130, 130, 130};  // 悪い予測（平均のみ）

    print_data("実測値", y_actual);
    print_data("予測値（良いモデル）", y_pred1);
    print_data("予測値（悪いモデル）", y_pred2);

    double r2_good = statcpp::r_squared(
        y_actual.begin(), y_actual.end(),
        y_pred1.begin(), y_pred1.end());
    double r2_bad = statcpp::r_squared(
        y_actual.begin(), y_actual.end(),
        y_pred2.begin(), y_pred2.end());

    std::cout << "\n決定係数 R²:\n";
    std::cout << "  良いモデル: " << r2_good << "\n";
    std::cout << "  悪いモデル: " << r2_bad << "\n";

    std::cout << R"(
【決定係数の解釈】
- R² = 1.0: 完璧な予測
- R² = 0.0: 平均を予測するのと同じ
- R² < 0:   平均より悪い予測

【R² vs 調整済み R²】
- R² は変数を追加すると必ず増加（過学習の危険）
- 調整済み R² は不要な変数のペナルティを含む
- モデル比較には調整済み R² を使用
)";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：線形回帰分析の関数");

    std::cout << R"(
┌────────────────────────────────┬─────────────────────────────────────────┐
│ 関数                           │ 説明                                    │
├────────────────────────────────┼─────────────────────────────────────────┤
│ simple_linear_regression()     │ 単回帰分析 (y = β₀ + β₁x)              │
│ multiple_linear_regression()   │ 重回帰分析 (y = β₀ + β₁x₁ + ...)       │
│ predict()                      │ 点予測                                  │
│ prediction_interval_simple()   │ 予測区間（個々の値の範囲）              │
│ confidence_interval_mean()     │ 平均の信頼区間                          │
│ compute_residual_diagnostics() │ 残差診断                                │
│ compute_vif()                  │ 分散膨張係数（多重共線性チェック）      │
│ r_squared()                    │ 決定係数                                │
│ adjusted_r_squared()           │ 自由度調整済み決定係数                  │
└────────────────────────────────┴─────────────────────────────────────────┘

【戻り値の構造体】
- simple_regression_result: 単回帰の詳細結果
- multiple_regression_result: 重回帰の詳細結果
- prediction_interval: 予測値と区間
- residual_diagnostics: 残差診断情報

【モデル評価のポイント】
1. R²: モデルの説明力（1に近いほど良い）
2. p値: 係数の統計的有意性（< 0.05 で有意）
3. 残差診断: モデルの仮定が満たされているか
4. VIF: 多重共線性のチェック（< 5 が望ましい）

【注意事項】
- 相関は因果を意味しない
- 外挿（データ範囲外の予測）は信頼性が低い
- 多重共線性があると係数が不安定になる
- 残差が正規分布に従わない場合は別の手法を検討
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_simple_regression();
    example_prediction();
    example_multiple_regression();
    example_residual_diagnostics();
    example_vif();
    example_r_squared();

    // まとめを表示
    print_summary();

    return 0;
}
