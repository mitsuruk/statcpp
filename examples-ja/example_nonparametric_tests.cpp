/**
 * @file example_nonparametric_tests.cpp
 * @brief statcpp ノンパラメトリック検定関数のサンプルコード
 *
 * このファイルでは nonparametric_tests.hpp で提供される以下の関数を解説します：
 * - shapiro_wilk_test(): Shapiro-Wilk正規性検定
 * - ks_test_normal(): Kolmogorov-Smirnov正規性検定
 * - levene_test(): Levene等分散性検定（Brown-Forsythe版）
 * - bartlett_test(): Bartlett等分散性検定
 * - wilcoxon_signed_rank_test(): Wilcoxon符号付順位検定
 * - mann_whitney_u_test(): Mann-Whitney U検定
 * - kruskal_wallis_test(): Kruskal-Wallis検定
 * - fisher_exact_test(): Fisher正確確率検定
 *
 * コンパイル方法:
 *   g++ -std=c++17 -I../statcpp/include example_nonparametric_tests.cpp -o example_nonparametric_tests
 *
 * 実行方法:
 *   ./example_nonparametric_tests
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#include "statcpp/nonparametric_tests.hpp"

// ============================================================================
// ヘルパー関数
// ============================================================================

void print_section(const std::string& title)
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << " " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title)
{
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// 各関数のサンプル
// ============================================================================

/**
 * @brief shapiro_wilk_test() のサンプル
 *
 * 【目的】
 * Shapiro-Wilk検定は、データが正規分布に従うかどうかを検定します。
 * 正規性検定の中で最も検出力が高いとされ、広く使用されています。
 *
 * 【数式】
 * W = (Σ aᵢ x₍ᵢ₎)² / Σ (xᵢ - x̄)²
 * ここで x₍ᵢ₎ は順序統計量、aᵢ は係数（正規分布の期待値から計算）
 * W は 0〜1 の値を取り、1 に近いほど正規分布に近い
 *
 * 【使用場面】
 * - パラメトリック検定の前提条件（正規性）の確認
 * - t検定、分散分析を行う前のデータチェック
 * - 品質管理における製品特性の分布確認
 *
 * 【注意点】
 * - サンプルサイズ n ≤ 50 で最適化されている
 * - n > 5000 はサポートされない
 * - 帰無仮説 H₀: データは正規分布に従う
 * - p値が小さい（通常 < 0.05）場合、正規性を棄却
 */
void example_shapiro_wilk_test()
{
    print_section("shapiro_wilk_test() - Shapiro-Wilk正規性検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 正規分布に近いデータ（テストの点数）
    print_subsection("ケース1: 正規分布に近いデータ");
    std::cout << "シナリオ: 100人の学生のテスト点数（平均70点、標準偏差10点程度）\n";

    std::mt19937 gen(42);
    std::normal_distribution<> normal(70.0, 10.0);

    std::vector<double> normal_data;
    for (int i = 0; i < 30; ++i) {
        normal_data.push_back(normal(gen));
    }

    std::cout << "データ（最初の10件）: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << normal_data[i] << " ";
    }
    std::cout << "...\n";

    auto result1 = statcpp::shapiro_wilk_test(normal_data.begin(), normal_data.end());
    std::cout << "W統計量: " << result1.statistic << "\n";
    std::cout << "p値: " << result1.p_value << "\n";
    std::cout << "有意水準 0.05 での判定: "
              << (result1.p_value > 0.05 ? "正規性を棄却できない（正規分布と見なせる）"
                                          : "正規性を棄却（正規分布ではない）") << "\n";

    // ケース2: 非正規分布データ（指数分布的な待ち時間）
    print_subsection("ケース2: 非正規分布データ");
    std::cout << "シナリオ: コールセンターの待ち時間（秒）- 指数分布的\n";

    std::exponential_distribution<> expo(0.1);
    std::vector<double> expo_data;
    for (int i = 0; i < 30; ++i) {
        expo_data.push_back(expo(gen));
    }

    std::cout << "データ（最初の10件）: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << expo_data[i] << " ";
    }
    std::cout << "...\n";

    auto result2 = statcpp::shapiro_wilk_test(expo_data.begin(), expo_data.end());
    std::cout << "W統計量: " << result2.statistic << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "有意水準 0.05 での判定: "
              << (result2.p_value > 0.05 ? "正規性を棄却できない"
                                          : "正規性を棄却（正規分布ではない）") << "\n";

    // ケース3: 一様分布データ
    print_subsection("ケース3: 一様分布データ");
    std::cout << "シナリオ: サイコロを30回振った結果\n";

    std::uniform_int_distribution<> uniform(1, 6);
    std::vector<double> uniform_data;
    for (int i = 0; i < 30; ++i) {
        uniform_data.push_back(static_cast<double>(uniform(gen)));
    }

    auto result3 = statcpp::shapiro_wilk_test(uniform_data.begin(), uniform_data.end());
    std::cout << "W統計量: " << result3.statistic << "\n";
    std::cout << "p値: " << result3.p_value << "\n";
    std::cout << "有意水準 0.05 での判定: "
              << (result3.p_value > 0.05 ? "正規性を棄却できない"
                                          : "正規性を棄却（正規分布ではない）") << "\n";
}

/**
 * @brief ks_test_normal() のサンプル
 *
 * 【目的】
 * Kolmogorov-Smirnov検定（KS検定）は、データの経験分布関数と
 * 理論的な正規分布関数との最大乖離を検定します。
 *
 * 【数式】
 * D = max|Fₙ(x) - F(x)|
 * ここで Fₙ(x) は経験分布関数、F(x) は標準正規分布の累積分布関数
 * データは標準化されてから検定される
 *
 * 【使用場面】
 * - 大規模データの正規性検定
 * - Shapiro-Wilk検定が使えない場合（n > 5000）
 * - 分布の形状の大まかな確認
 *
 * 【注意点】
 * - Shapiro-Wilk検定より検出力が低い傾向がある
 * - 分布の中心部分よりも裾部分に敏感
 * - 平均と分散を推定して使用（Lilliefors補正）
 */
void example_ks_test_normal()
{
    print_section("ks_test_normal() - Kolmogorov-Smirnov正規性検定");

    std::cout << std::fixed << std::setprecision(4);

    std::mt19937 gen(123);

    // ケース1: 正規分布データ
    print_subsection("ケース1: 正規分布データ");
    std::cout << "シナリオ: 製品の重量測定値（100個）\n";

    std::normal_distribution<> normal(500.0, 5.0);
    std::vector<double> normal_data;
    for (int i = 0; i < 100; ++i) {
        normal_data.push_back(normal(gen));
    }

    auto result1 = statcpp::ks_test_normal(normal_data.begin(), normal_data.end());
    std::cout << "D統計量（最大乖離）: " << result1.statistic << "\n";
    std::cout << "p値: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value > 0.05 ? "正規性を棄却できない" : "正規性を棄却") << "\n";

    // ケース2: 二峰性分布
    print_subsection("ケース2: 二峰性分布");
    std::cout << "シナリオ: 男女混合の身長データ\n";

    std::normal_distribution<> male(170.0, 6.0);
    std::normal_distribution<> female(158.0, 5.0);
    std::bernoulli_distribution gender(0.5);

    std::vector<double> bimodal_data;
    for (int i = 0; i < 100; ++i) {
        if (gender(gen)) {
            bimodal_data.push_back(male(gen));
        } else {
            bimodal_data.push_back(female(gen));
        }
    }

    auto result2 = statcpp::ks_test_normal(bimodal_data.begin(), bimodal_data.end());
    std::cout << "D統計量: " << result2.statistic << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "正規性を棄却できない" : "正規性を棄却") << "\n";

    // Shapiro-Wilkとの比較
    print_subsection("Shapiro-Wilk検定との比較");
    std::vector<double> test_data;
    for (int i = 0; i < 50; ++i) {
        test_data.push_back(normal(gen));
    }

    auto sw_result = statcpp::shapiro_wilk_test(test_data.begin(), test_data.end());
    auto ks_result = statcpp::ks_test_normal(test_data.begin(), test_data.end());

    std::cout << "同じ正規分布データに対する検定結果:\n";
    std::cout << "  Shapiro-Wilk: W = " << sw_result.statistic
              << ", p = " << sw_result.p_value << "\n";
    std::cout << "  KS検定:       D = " << ks_result.statistic
              << ", p = " << ks_result.p_value << "\n";
}

/**
 * @brief levene_test() のサンプル
 *
 * 【目的】
 * Levene検定は、複数の群の分散が等しいか（等分散性）を検定します。
 * 中央値からの偏差を使用するBrown-Forsythe版で、外れ値に頑健です。
 *
 * 【数式】
 * W = [(N-k) / (k-1)] × [Σ nᵢ(Z̄ᵢ - Z̄)² / Σ Σ(Zᵢⱼ - Z̄ᵢ)²]
 * ここで Zᵢⱼ = |Xᵢⱼ - Medianᵢ|
 * W は F分布に従う
 *
 * 【使用場面】
 * - 分散分析（ANOVA）の前提条件確認
 * - 2群以上のデータの分散比較
 * - 正規性が疑われる場合の等分散性検定
 *
 * 【注意点】
 * - Bartlett検定より正規性からの逸脱に頑健
 * - 各群に最低2要素必要
 * - 帰無仮説 H₀: すべての群の分散は等しい
 */
void example_levene_test()
{
    print_section("levene_test() - Levene等分散性検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 等分散のグループ
    print_subsection("ケース1: 等分散のグループ");
    std::cout << "シナリオ: 3つの工場の製品重量のばらつき比較\n";

    std::vector<std::vector<double>> equal_var_groups = {
        {100.2, 99.8, 100.5, 99.5, 100.1, 99.9, 100.3, 99.7},  // 工場A
        {100.1, 99.9, 100.4, 99.6, 100.0, 99.8, 100.2, 99.8},  // 工場B
        {100.3, 99.7, 100.2, 99.8, 100.1, 99.9, 100.0, 100.0}  // 工場C
    };

    std::cout << "工場A: ";
    for (double v : equal_var_groups[0]) std::cout << v << " ";
    std::cout << "\n工場B: ";
    for (double v : equal_var_groups[1]) std::cout << v << " ";
    std::cout << "\n工場C: ";
    for (double v : equal_var_groups[2]) std::cout << v << " ";
    std::cout << "\n\n";

    auto result1 = statcpp::levene_test(equal_var_groups);
    std::cout << "F統計量: " << result1.statistic << "\n";
    std::cout << "p値: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value > 0.05 ? "等分散性を棄却できない（分散は等しいと見なせる）"
                                          : "等分散性を棄却（分散に差がある）") << "\n";

    // ケース2: 不等分散のグループ
    print_subsection("ケース2: 不等分散のグループ");
    std::cout << "シナリオ: 経験年数別の作業時間のばらつき\n";

    std::vector<std::vector<double>> unequal_var_groups = {
        {45, 52, 48, 55, 42, 58, 40, 60, 38, 62},  // 新人（ばらつき大）
        {30, 32, 31, 33, 29, 34, 28, 35, 30, 31},  // 中堅（ばらつき中）
        {25, 26, 25, 27, 24, 26, 25, 26, 25, 26}   // ベテラン（ばらつき小）
    };

    std::cout << "新人:   ";
    for (double v : unequal_var_groups[0]) std::cout << v << " ";
    std::cout << "\n中堅:   ";
    for (double v : unequal_var_groups[1]) std::cout << v << " ";
    std::cout << "\nベテラン: ";
    for (double v : unequal_var_groups[2]) std::cout << v << " ";
    std::cout << "\n\n";

    // 各群の分散を計算
    for (std::size_t i = 0; i < unequal_var_groups.size(); ++i) {
        double var = statcpp::sample_variance(unequal_var_groups[i].begin(),
                                               unequal_var_groups[i].end());
        std::cout << "群" << i + 1 << "の分散: " << var << "\n";
    }

    auto result2 = statcpp::levene_test(unequal_var_groups);
    std::cout << "\nF統計量: " << result2.statistic << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "等分散性を棄却できない"
                                          : "等分散性を棄却（分散に差がある）") << "\n";
}

/**
 * @brief bartlett_test() のサンプル
 *
 * 【目的】
 * Bartlett検定は、複数の群の分散が等しいかを検定します。
 * 正規分布を仮定し、Levene検定より検出力が高いですが、
 * 正規性からの逸脱に敏感です。
 *
 * 【数式】
 * χ² = [(N-k)ln(S²ₚ) - Σ(nᵢ-1)ln(S²ᵢ)] / C
 * ここで S²ₚ はプール分散、C は補正係数
 * 検定統計量はχ²分布に従う
 *
 * 【使用場面】
 * - データの正規性が確認されている場合の等分散性検定
 * - ANOVAの前提条件チェック（正規データの場合）
 *
 * 【注意点】
 * - 正規性からの逸脱に非常に敏感
 * - 非正規データにはLevene検定を推奨
 * - 分散が0または負の群があるとエラー
 */
void example_bartlett_test()
{
    print_section("bartlett_test() - Bartlett等分散性検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 等分散（正規分布データ）
    print_subsection("ケース1: 等分散の正規分布データ");
    std::cout << "シナリオ: 3種類の肥料を使った植物の成長（cm）\n";

    std::mt19937 gen(456);
    std::normal_distribution<> dist(15.0, 2.0);

    std::vector<std::vector<double>> equal_groups(3);
    for (int g = 0; g < 3; ++g) {
        for (int i = 0; i < 15; ++i) {
            equal_groups[g].push_back(dist(gen) + g * 0.5);  // わずかに平均が異なる
        }
    }

    for (int g = 0; g < 3; ++g) {
        std::cout << "肥料" << static_cast<char>('A' + g) << "の分散: "
                  << statcpp::sample_variance(equal_groups[g].begin(), equal_groups[g].end())
                  << "\n";
    }

    auto result1 = statcpp::bartlett_test(equal_groups);
    std::cout << "\nχ²統計量: " << result1.statistic << "\n";
    std::cout << "自由度: " << result1.df << "\n";
    std::cout << "p値: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value > 0.05 ? "等分散性を棄却できない" : "等分散性を棄却") << "\n";

    // ケース2: 不等分散
    print_subsection("ケース2: 明らかに不等分散のデータ");
    std::cout << "シナリオ: 品質管理における異なる機械からの出力\n";

    std::normal_distribution<> dist1(100.0, 1.0);   // 分散 1
    std::normal_distribution<> dist2(100.0, 3.0);   // 分散 9
    std::normal_distribution<> dist3(100.0, 5.0);   // 分散 25

    std::vector<std::vector<double>> unequal_groups(3);
    for (int i = 0; i < 20; ++i) {
        unequal_groups[0].push_back(dist1(gen));
        unequal_groups[1].push_back(dist2(gen));
        unequal_groups[2].push_back(dist3(gen));
    }

    for (int g = 0; g < 3; ++g) {
        std::cout << "機械" << g + 1 << "の分散: "
                  << statcpp::sample_variance(unequal_groups[g].begin(), unequal_groups[g].end())
                  << "\n";
    }

    auto result2 = statcpp::bartlett_test(unequal_groups);
    std::cout << "\nχ²統計量: " << result2.statistic << "\n";
    std::cout << "自由度: " << result2.df << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "等分散性を棄却できない" : "等分散性を棄却") << "\n";

    // LeveneとBartlettの比較
    print_subsection("Levene検定との比較");
    auto levene_result = statcpp::levene_test(unequal_groups);
    std::cout << "同じ不等分散データに対する結果:\n";
    std::cout << "  Bartlett検定: χ² = " << result2.statistic
              << ", p = " << result2.p_value << "\n";
    std::cout << "  Levene検定:   F  = " << levene_result.statistic
              << ", p = " << levene_result.p_value << "\n";
}

/**
 * @brief wilcoxon_signed_rank_test() のサンプル
 *
 * 【目的】
 * Wilcoxon符号付順位検定は、対応のある2群の差または
 * 1群の中央値が特定の値と等しいかを検定するノンパラメトリック検定です。
 * 対応のあるt検定のノンパラメトリック版です。
 *
 * 【数式】
 * W⁺ = Σ Rᵢ (差が正の場合の順位の和)
 * z = (W⁺ - E[W⁺]) / √Var(W⁺)  （正規近似）
 *
 * 【使用場面】
 * - 正規性が仮定できない対応のあるデータの比較
 * - 処理前後の効果測定
 * - 順序尺度データの分析
 *
 * 【注意点】
 * - 差が0のデータは除外される
 * - 最低2つの非ゼロの差が必要
 * - デフォルトは両側検定
 */
void example_wilcoxon_signed_rank_test()
{
    print_section("wilcoxon_signed_rank_test() - Wilcoxon符号付順位検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 対応のあるデータ（処理前後）
    print_subsection("ケース1: ダイエットプログラムの効果");
    std::cout << "シナリオ: 10人の参加者のダイエット前後の体重変化\n";

    std::vector<double> before = {75.2, 82.1, 68.5, 90.3, 77.8, 85.6, 72.4, 88.9, 79.1, 83.2};
    std::vector<double> after  = {73.1, 79.5, 67.8, 86.2, 75.9, 82.3, 71.0, 85.4, 76.8, 80.1};

    // 差を計算
    std::vector<double> diff;
    std::cout << "参加者: 前   後   差\n";
    for (std::size_t i = 0; i < before.size(); ++i) {
        double d = after[i] - before[i];
        diff.push_back(d);
        std::cout << "   " << i + 1 << ":   " << before[i] << " " << after[i]
                  << " " << (d >= 0 ? "+" : "") << d << "\n";
    }

    // H₀: 中央値 = 0（差がない）
    auto result1 = statcpp::wilcoxon_signed_rank_test(diff.begin(), diff.end(), 0.0);
    std::cout << "\nW⁺統計量: " << result1.statistic << "\n";
    std::cout << "p値（両側）: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value < 0.05 ? "有意な体重減少あり" : "有意な差なし") << "\n";

    // 片側検定
    print_subsection("ケース1b: 片側検定（減少方向）");
    auto result1_less = statcpp::wilcoxon_signed_rank_test(
        diff.begin(), diff.end(), 0.0, statcpp::alternative_hypothesis::less);
    std::cout << "p値（片側、less）: " << result1_less.p_value << "\n";
    std::cout << "判定: "
              << (result1_less.p_value < 0.05 ? "有意な体重減少" : "有意な減少なし") << "\n";

    // ケース2: 1標本の中央値検定
    print_subsection("ケース2: 1標本の中央値検定");
    std::cout << "シナリオ: 製品の重量が規格値500gを満たすか\n";

    std::vector<double> weights = {498.5, 501.2, 499.8, 502.1, 497.6, 500.5, 499.1, 501.8, 498.2, 500.9};

    std::cout << "測定値: ";
    for (double w : weights) std::cout << w << " ";
    std::cout << "\n";

    // H₀: 中央値 = 500
    auto result2 = statcpp::wilcoxon_signed_rank_test(weights.begin(), weights.end(), 500.0);
    std::cout << "\nW⁺統計量: " << result2.statistic << "\n";
    std::cout << "p値（両側）: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "規格値500gと有意差なし" : "規格値500gと有意差あり") << "\n";

    // ケース3: 効果なしのケース
    print_subsection("ケース3: 効果なしのケース");
    std::cout << "シナリオ: プラセボ群の変化\n";

    std::vector<double> placebo_diff = {-0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, 0.0, -0.2};
    std::cout << "差: ";
    for (double d : placebo_diff) std::cout << d << " ";
    std::cout << "\n";

    auto result3 = statcpp::wilcoxon_signed_rank_test(placebo_diff.begin(), placebo_diff.end(), 0.0);
    std::cout << "W⁺統計量: " << result3.statistic << "\n";
    std::cout << "p値: " << result3.p_value << "\n";
    std::cout << "判定: "
              << (result3.p_value > 0.05 ? "有意な変化なし（期待通り）" : "有意な変化あり") << "\n";
}

/**
 * @brief mann_whitney_u_test() のサンプル
 *
 * 【目的】
 * Mann-Whitney U検定は、独立した2群の分布が同じかを検定する
 * ノンパラメトリック検定です。対応のないt検定のノンパラメトリック版。
 *
 * 【数式】
 * U₁ = R₁ - n₁(n₁+1)/2
 * ここで R₁ は群1の順位和、n₁ は群1のサンプルサイズ
 * z = (U₁ - E[U₁]) / √Var(U₁)  （正規近似）
 *
 * 【使用場面】
 * - 正規性が仮定できない2群の比較
 * - 順序尺度データの比較
 * - サンプルサイズが小さい場合
 *
 * 【注意点】
 * - 各群に最低2要素必要
 * - 同順位（タイ）の補正あり
 * - 分布の形状が似ていることが望ましい
 */
void example_mann_whitney_u_test()
{
    print_section("mann_whitney_u_test() - Mann-Whitney U検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 差があるケース
    print_subsection("ケース1: 2つの治療法の効果比較");
    std::cout << "シナリオ: 新薬と従来薬の痛み軽減スコア（0-10）\n";

    std::vector<double> new_drug  = {8, 7, 9, 6, 8, 7, 9, 8, 7, 8};
    std::vector<double> old_drug  = {5, 4, 6, 5, 4, 5, 6, 4, 5, 5};

    std::cout << "新薬群: ";
    for (double v : new_drug) std::cout << v << " ";
    std::cout << "\n従来薬群: ";
    for (double v : old_drug) std::cout << v << " ";
    std::cout << "\n\n";

    std::cout << "新薬群の中央値: " << statcpp::median(new_drug.begin(), new_drug.end()) << "\n";
    std::cout << "従来薬群の中央値: " << statcpp::median(old_drug.begin(), old_drug.end()) << "\n";

    auto result1 = statcpp::mann_whitney_u_test(
        new_drug.begin(), new_drug.end(),
        old_drug.begin(), old_drug.end());

    std::cout << "\nU統計量: " << result1.statistic << "\n";
    std::cout << "p値（両側）: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value < 0.05 ? "有意差あり（新薬が効果的）" : "有意差なし") << "\n";

    // 片側検定
    print_subsection("ケース1b: 片側検定（新薬 > 従来薬）");
    auto result1_greater = statcpp::mann_whitney_u_test(
        new_drug.begin(), new_drug.end(),
        old_drug.begin(), old_drug.end(),
        statcpp::alternative_hypothesis::greater);
    std::cout << "p値（片側、greater）: " << result1_greater.p_value << "\n";

    // ケース2: 差がないケース
    print_subsection("ケース2: 差がないケース");
    std::cout << "シナリオ: 2つのクラスのテスト成績\n";

    std::vector<double> class_a = {72, 85, 78, 90, 82, 75, 88, 79};
    std::vector<double> class_b = {74, 81, 76, 89, 84, 77, 86, 80};

    std::cout << "クラスA: ";
    for (double v : class_a) std::cout << v << " ";
    std::cout << "\nクラスB: ";
    for (double v : class_b) std::cout << v << " ";
    std::cout << "\n";

    auto result2 = statcpp::mann_whitney_u_test(
        class_a.begin(), class_a.end(),
        class_b.begin(), class_b.end());

    std::cout << "\nU統計量: " << result2.statistic << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "有意差なし" : "有意差あり") << "\n";

    // ケース3: 順序尺度データ
    print_subsection("ケース3: 順序尺度データ");
    std::cout << "シナリオ: 顧客満足度調査（1=非常に不満 〜 5=非常に満足）\n";

    std::vector<double> store_a = {4, 5, 4, 3, 5, 4, 4, 5, 3, 4};
    std::vector<double> store_b = {3, 2, 3, 4, 2, 3, 3, 2, 4, 3};

    std::cout << "店舗A: ";
    for (double v : store_a) std::cout << v << " ";
    std::cout << "\n店舗B: ";
    for (double v : store_b) std::cout << v << " ";
    std::cout << "\n";

    auto result3 = statcpp::mann_whitney_u_test(
        store_a.begin(), store_a.end(),
        store_b.begin(), store_b.end());

    std::cout << "\nU統計量: " << result3.statistic << "\n";
    std::cout << "p値: " << result3.p_value << "\n";
    std::cout << "判定: "
              << (result3.p_value < 0.05 ? "店舗間で満足度に有意差あり" : "有意差なし") << "\n";
}

/**
 * @brief kruskal_wallis_test() のサンプル
 *
 * 【目的】
 * Kruskal-Wallis検定は、3群以上の独立した群の分布が同じかを検定する
 * ノンパラメトリック検定です。一元配置分散分析のノンパラメトリック版。
 *
 * 【数式】
 * H = [12 / (N(N+1))] × Σ(nᵢR̄ᵢ²) - 3(N+1)
 * ここで R̄ᵢ は群iの平均順位
 * H は近似的にχ²(k-1)分布に従う
 *
 * 【使用場面】
 * - 3群以上の比較で正規性が仮定できない場合
 * - 順序尺度データの群間比較
 * - ANOVAの前提条件が満たされない場合
 *
 * 【注意点】
 * - 帰無仮説の棄却は「どこかに差がある」ことを示す
 * - どの群間に差があるかは多重比較が必要
 * - 各群は空でないことが必要
 */
void example_kruskal_wallis_test()
{
    print_section("kruskal_wallis_test() - Kruskal-Wallis検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 有意差があるケース
    print_subsection("ケース1: 3種類の教授法の効果比較");
    std::cout << "シナリオ: 講義型、演習型、オンライン型のテスト成績\n";

    std::vector<std::vector<double>> teaching_methods = {
        {65, 70, 68, 72, 69, 71, 67, 73},  // 講義型
        {78, 82, 80, 85, 79, 83, 81, 84},  // 演習型
        {72, 75, 74, 77, 73, 76, 74, 78}   // オンライン
    };

    std::vector<std::string> method_names = {"講義型", "演習型", "オンライン"};
    for (std::size_t i = 0; i < teaching_methods.size(); ++i) {
        std::cout << method_names[i] << ": ";
        for (double v : teaching_methods[i]) std::cout << v << " ";
        std::cout << "\n  中央値: " << statcpp::median(teaching_methods[i].begin(),
                                                        teaching_methods[i].end()) << "\n";
    }

    auto result1 = statcpp::kruskal_wallis_test(teaching_methods);
    std::cout << "\nH統計量: " << result1.statistic << "\n";
    std::cout << "自由度: " << result1.df << "\n";
    std::cout << "p値: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value < 0.05 ? "教授法間で有意差あり" : "有意差なし") << "\n";

    // ケース2: 有意差がないケース
    print_subsection("ケース2: 差がないケース");
    std::cout << "シナリオ: 3つの店舗の売上（万円）\n";

    std::vector<std::vector<double>> stores = {
        {120, 135, 128, 142, 131, 138, 125, 140},
        {125, 132, 130, 138, 128, 135, 127, 136},
        {122, 137, 126, 140, 133, 134, 129, 139}
    };

    for (std::size_t i = 0; i < stores.size(); ++i) {
        std::cout << "店舗" << static_cast<char>('A' + i) << ": ";
        for (double v : stores[i]) std::cout << v << " ";
        std::cout << "\n";
    }

    auto result2 = statcpp::kruskal_wallis_test(stores);
    std::cout << "\nH統計量: " << result2.statistic << "\n";
    std::cout << "自由度: " << result2.df << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "店舗間で有意差なし" : "有意差あり") << "\n";

    // ケース3: 4群以上
    print_subsection("ケース3: 4群の比較");
    std::cout << "シナリオ: 4つの年齢層の運動習慣（週あたりの運動時間）\n";

    std::vector<std::vector<double>> age_groups = {
        {5.5, 6.0, 4.5, 7.0, 5.0, 6.5, 4.0, 5.5},  // 20代
        {4.0, 3.5, 5.0, 3.0, 4.5, 3.5, 4.0, 4.5},  // 30代
        {2.5, 3.0, 2.0, 3.5, 2.5, 2.0, 3.0, 2.5},  // 40代
        {1.5, 2.0, 1.0, 2.5, 1.5, 2.0, 1.0, 1.5}   // 50代
    };

    std::vector<std::string> age_names = {"20代", "30代", "40代", "50代"};
    for (std::size_t i = 0; i < age_groups.size(); ++i) {
        std::cout << age_names[i] << "の中央値: "
                  << statcpp::median(age_groups[i].begin(), age_groups[i].end())
                  << " 時間/週\n";
    }

    auto result3 = statcpp::kruskal_wallis_test(age_groups);
    std::cout << "\nH統計量: " << result3.statistic << "\n";
    std::cout << "自由度: " << result3.df << "\n";
    std::cout << "p値: " << result3.p_value << "\n";
    std::cout << "判定: "
              << (result3.p_value < 0.05 ? "年齢層間で運動時間に有意差あり"
                                          : "有意差なし") << "\n";
}

/**
 * @brief fisher_exact_test() のサンプル
 *
 * 【目的】
 * Fisher正確確率検定は、2×2分割表において2つのカテゴリ変数が
 * 独立かどうかを検定します。χ²検定と異なり、期待度数が小さくても適用可能。
 *
 * 【数式】
 * P(a) = C(a+c,a) × C(b+d,b) / C(n,a+b)
 * 超幾何分布に基づく確率計算
 * オッズ比 = (a×d) / (b×c)
 *
 * 【使用場面】
 * - サンプルサイズが小さい2×2分割表
 * - 期待度数が5未満のセルがある場合
 * - 2つのカテゴリ変数の関連性検定
 *
 * 【注意点】
 * - 2×2表専用
 * - 計算コストが高い（大きな数では注意）
 * - 統計量としてオッズ比を返す
 */
void example_fisher_exact_test()
{
    print_section("fisher_exact_test() - Fisher正確確率検定");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 関連があるケース
    print_subsection("ケース1: 治療と回復の関連");
    std::cout << "シナリオ: 新薬の効果（回復/非回復）\n";

    std::cout << "分割表:\n";
    std::cout << "           回復  非回復  計\n";
    std::cout << "新薬群      12      3    15\n";
    std::cout << "対照群       5     10    15\n";
    std::cout << "計          17     13    30\n\n";

    // a=12, b=3, c=5, d=10
    auto result1 = statcpp::fisher_exact_test(12, 3, 5, 10);
    std::cout << "オッズ比: " << result1.statistic << "\n";
    std::cout << "p値（両側）: " << result1.p_value << "\n";
    std::cout << "判定: "
              << (result1.p_value < 0.05 ? "治療と回復に有意な関連あり"
                                          : "有意な関連なし") << "\n";

    std::cout << "\n解釈: オッズ比 " << result1.statistic
              << " は、新薬群の回復オッズが対照群の約"
              << result1.statistic << "倍であることを示す\n";

    // 片側検定
    print_subsection("ケース1b: 片側検定");
    auto result1_greater = statcpp::fisher_exact_test(12, 3, 5, 10,
                                                       statcpp::alternative_hypothesis::greater);
    std::cout << "p値（片側、greater）: " << result1_greater.p_value << "\n";
    std::cout << "H₁: 新薬は対照より効果が高い\n";

    // ケース2: 関連がないケース
    print_subsection("ケース2: 関連がないケース");
    std::cout << "シナリオ: 性別と製品選好\n";

    std::cout << "分割表:\n";
    std::cout << "           製品A  製品B  計\n";
    std::cout << "男性         8      7    15\n";
    std::cout << "女性         7      8    15\n";
    std::cout << "計          15     15    30\n\n";

    auto result2 = statcpp::fisher_exact_test(8, 7, 7, 8);
    std::cout << "オッズ比: " << result2.statistic << "\n";
    std::cout << "p値: " << result2.p_value << "\n";
    std::cout << "判定: "
              << (result2.p_value > 0.05 ? "性別と製品選好に有意な関連なし"
                                          : "有意な関連あり") << "\n";

    // ケース3: 極端なケース
    print_subsection("ケース3: 極端に偏ったデータ");
    std::cout << "シナリオ: 稀な副作用の発生\n";

    std::cout << "分割表:\n";
    std::cout << "           副作用あり  副作用なし  計\n";
    std::cout << "投薬群          5          95    100\n";
    std::cout << "対照群          0         100    100\n";
    std::cout << "計              5         195    200\n\n";

    auto result3 = statcpp::fisher_exact_test(5, 95, 0, 100);
    std::cout << "オッズ比: " << result3.statistic << " (無限大: 対照群でゼロ)\n";
    std::cout << "p値: " << result3.p_value << "\n";
    std::cout << "判定: "
              << (result3.p_value < 0.05 ? "投薬と副作用に有意な関連あり"
                                          : "有意な関連なし") << "\n";

    // ケース4: 小さなサンプルサイズ
    print_subsection("ケース4: 小さなサンプル");
    std::cout << "シナリオ: パイロットスタディの結果\n";

    std::cout << "分割表:\n";
    std::cout << "           成功  失敗  計\n";
    std::cout << "実験群       4     1    5\n";
    std::cout << "対照群       1     4    5\n";
    std::cout << "計           5     5   10\n\n";

    auto result4 = statcpp::fisher_exact_test(4, 1, 1, 4);
    std::cout << "オッズ比: " << result4.statistic << "\n";
    std::cout << "p値: " << result4.p_value << "\n";
    std::cout << "判定: "
              << (result4.p_value < 0.05 ? "有意な関連あり"
                                          : "サンプルサイズが小さく有意差検出困難") << "\n";
}

/**
 * @brief 複合的な例：パラメトリック vs ノンパラメトリック検定の選択
 */
void example_test_selection()
{
    print_section("検定の選択：パラメトリック vs ノンパラメトリック");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: データの特性に応じた検定選択");
    std::cout << "2つの治療法の効果を比較する際の検定選択フロー\n\n";

    // 歪んだ分布のデータを生成
    std::mt19937 gen(789);
    std::exponential_distribution<> expo1(0.1);
    std::exponential_distribution<> expo2(0.15);

    std::vector<double> treatment_a, treatment_b;
    for (int i = 0; i < 20; ++i) {
        treatment_a.push_back(expo1(gen));
        treatment_b.push_back(expo2(gen));
    }

    std::cout << "Step 1: 正規性の確認\n";
    auto sw_a = statcpp::shapiro_wilk_test(treatment_a.begin(), treatment_a.end());
    auto sw_b = statcpp::shapiro_wilk_test(treatment_b.begin(), treatment_b.end());

    std::cout << "  治療A群: W = " << sw_a.statistic << ", p = " << sw_a.p_value << "\n";
    std::cout << "  治療B群: W = " << sw_b.statistic << ", p = " << sw_b.p_value << "\n";

    bool normal_a = sw_a.p_value > 0.05;
    bool normal_b = sw_b.p_value > 0.05;

    std::cout << "  結果: " << (normal_a ? "A群は正規" : "A群は非正規")
              << ", " << (normal_b ? "B群は正規" : "B群は非正規") << "\n\n";

    if (normal_a && normal_b) {
        std::cout << "Step 2: 両群とも正規なので等分散性を確認\n";
        std::vector<std::vector<double>> groups = {treatment_a, treatment_b};
        auto bartlett = statcpp::bartlett_test(groups);
        std::cout << "  Bartlett検定: χ² = " << bartlett.statistic
                  << ", p = " << bartlett.p_value << "\n";

        if (bartlett.p_value > 0.05) {
            std::cout << "  → 等分散なのでStudent's t検定を使用\n";
        } else {
            std::cout << "  → 不等分散なのでWelch's t検定を使用\n";
        }
    } else {
        std::cout << "Step 2: 正規性が満たされないのでノンパラメトリック検定を使用\n";

        auto mw = statcpp::mann_whitney_u_test(
            treatment_a.begin(), treatment_a.end(),
            treatment_b.begin(), treatment_b.end());

        std::cout << "  Mann-Whitney U検定: U = " << mw.statistic
                  << ", p = " << mw.p_value << "\n";
        std::cout << "  判定: "
                  << (mw.p_value < 0.05 ? "2群間に有意差あり" : "有意差なし") << "\n";
    }

    print_subsection("検定選択のガイドライン");
    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ 状況                          推奨検定                      │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ 1群の正規性検定               Shapiro-Wilk検定              │\n";
    std::cout << "│ 等分散性検定（正規）          Bartlett検定                  │\n";
    std::cout << "│ 等分散性検定（非正規）        Levene検定                    │\n";
    std::cout << "│ 2群比較（正規・等分散）       Student's t検定               │\n";
    std::cout << "│ 2群比較（正規・不等分散）     Welch's t検定                 │\n";
    std::cout << "│ 2群比較（非正規）             Mann-Whitney U検定            │\n";
    std::cout << "│ 対応あり2群（非正規）         Wilcoxon符号付順位検定        │\n";
    std::cout << "│ 3群以上（非正規）             Kruskal-Wallis検定            │\n";
    std::cout << "│ 2×2分割表（小サンプル）      Fisher正確確率検定            │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// サマリー出力
// ============================================================================

void print_summary()
{
    print_section("サマリー: nonparametric_tests.hpp の関数一覧");

    std::cout << R"(
┌────────────────────────────────────────────────────────────────────────────┐
│ 関数名                          用途                                       │
├────────────────────────────────────────────────────────────────────────────┤
│ shapiro_wilk_test()             正規性検定（高検出力、n≤5000）             │
│ ks_test_normal()                正規性検定（KS検定）                       │
│ levene_test()                   等分散性検定（頑健）                       │
│ bartlett_test()                 等分散性検定（正規性仮定）                 │
│ wilcoxon_signed_rank_test()     1標本/対応あり2標本の位置検定              │
│ mann_whitney_u_test()           独立2標本の位置検定                        │
│ kruskal_wallis_test()           独立k標本の位置検定                        │
│ fisher_exact_test()             2×2分割表の独立性検定                     │
└────────────────────────────────────────────────────────────────────────────┘

【ノンパラメトリック検定の利点】
  - 正規性などの分布の仮定が不要
  - 外れ値に頑健
  - 順序尺度データにも適用可能
  - 小サンプルでも適用可能

【ノンパラメトリック検定の欠点】
  - パラメトリック検定より検出力が低い傾向
  - 信頼区間の計算が複雑

【検定結果の解釈】
  - test_result 構造体には statistic, p_value, df, alternative が含まれる
  - p値 < 有意水準 → 帰無仮説を棄却
  - p値 ≥ 有意水準 → 帰無仮説を棄却できない（採択ではない）
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main()
{
    std::cout << "==========================================================\n";
    std::cout << " statcpp ノンパラメトリック検定関数 サンプルコード\n";
    std::cout << "==========================================================\n";

    example_shapiro_wilk_test();
    example_ks_test_normal();
    example_levene_test();
    example_bartlett_test();
    example_wilcoxon_signed_rank_test();
    example_mann_whitney_u_test();
    example_kruskal_wallis_test();
    example_fisher_exact_test();
    example_test_selection();
    print_summary();

    return 0;
}
