/**
 * @file example_anova.cpp
 * @brief statcpp 分散分析（ANOVA）関数のサンプルコード
 *
 * このファイルでは anova.hpp で提供される以下の関数を解説します：
 * - one_way_anova(): 一元配置分散分析
 * - two_way_anova(): 二元配置分散分析
 * - tukey_hsd(): Tukey HSD検定（Studentized range分布による事後比較）
 * - bonferroni_posthoc(): Bonferroni法（事後比較）
 * - dunnett_posthoc(): Dunnett法（対照群との比較）
 * - scheffe_posthoc(): Scheffe法（事後比較）
 * - one_way_ancova(): 一元配置共分散分析
 * - eta_squared(), omega_squared(), cohens_f(): 効果量
 *
 * コンパイル方法:
 *   g++ -std=c++17 -I../statcpp/include example_anova.cpp -o example_anova
 *
 * 実行方法:
 *   ./example_anova
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "statcpp/anova.hpp"
#include "statcpp/basic_statistics.hpp"

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

void print_anova_table_oneway(const statcpp::one_way_anova_result& result)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nANOVA表:\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
    std::cout << "変動要因           SS        df      MS        F       p値\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
    std::cout << "群間(Between)   " << std::setw(8) << result.between.ss
              << "  " << std::setw(4) << result.between.df
              << "  " << std::setw(8) << result.between.ms
              << "  " << std::setw(6) << result.between.f_statistic
              << "  " << std::setw(6) << result.between.p_value << "\n";
    std::cout << "群内(Within)    " << std::setw(8) << result.within.ss
              << "  " << std::setw(4) << result.within.df
              << "  " << std::setw(8) << result.within.ms << "\n";
    std::cout << "総計(Total)     " << std::setw(8) << result.ss_total
              << "  " << std::setw(4) << result.df_total << "\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
}

void print_anova_table_twoway(const statcpp::two_way_anova_result& result)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nANOVA表:\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
    std::cout << "変動要因         SS        df      MS        F       p値\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
    std::cout << "要因A         " << std::setw(8) << result.factor_a.ss
              << "  " << std::setw(4) << result.factor_a.df
              << "  " << std::setw(8) << result.factor_a.ms
              << "  " << std::setw(6) << result.factor_a.f_statistic
              << "  " << std::setw(6) << result.factor_a.p_value << "\n";
    std::cout << "要因B         " << std::setw(8) << result.factor_b.ss
              << "  " << std::setw(4) << result.factor_b.df
              << "  " << std::setw(8) << result.factor_b.ms
              << "  " << std::setw(6) << result.factor_b.f_statistic
              << "  " << std::setw(6) << result.factor_b.p_value << "\n";
    std::cout << "交互作用(A×B) " << std::setw(8) << result.interaction.ss
              << "  " << std::setw(4) << result.interaction.df
              << "  " << std::setw(8) << result.interaction.ms
              << "  " << std::setw(6) << result.interaction.f_statistic
              << "  " << std::setw(6) << result.interaction.p_value << "\n";
    std::cout << "誤差(Error)   " << std::setw(8) << result.error.ss
              << "  " << std::setw(4) << result.error.df
              << "  " << std::setw(8) << result.error.ms << "\n";
    std::cout << "総計(Total)   " << std::setw(8) << result.ss_total
              << "  " << std::setw(4) << result.df_total << "\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
}

// ============================================================================
// 各関数のサンプル
// ============================================================================

/**
 * @brief one_way_anova() のサンプル
 *
 * 【目的】
 * 一元配置分散分析は、3群以上の平均値に差があるかを検定します。
 * 1つの独立変数（要因）が従属変数に及ぼす影響を分析します。
 *
 * 【数式】
 * F = MS_between / MS_within
 * SS_between = Σ nᵢ(x̄ᵢ - x̄)²  （群間変動）
 * SS_within = Σ Σ(xᵢⱼ - x̄ᵢ)²   （群内変動）
 *
 * 【使用場面】
 * - 複数の処理法・教授法の効果比較
 * - 複数群の平均値比較
 * - 実験計画法における要因の効果検定
 *
 * 【注意点】
 * - 正規性と等分散性の仮定が必要
 * - 有意な結果は「どこかに差がある」ことを示すのみ
 * - 具体的にどの群間に差があるかは事後比較が必要
 */
void example_one_way_anova()
{
    print_section("one_way_anova() - 一元配置分散分析");

    std::cout << std::fixed << std::setprecision(4);

    // ケース1: 有意差があるケース
    print_subsection("ケース1: 3つの教授法の効果比較");
    std::cout << "シナリオ: 講義型、演習型、PBL型のテスト成績（100点満点）\n";

    std::vector<std::vector<double>> teaching_methods = {
        {65, 70, 68, 72, 69, 71, 67, 73, 66, 70},  // 講義型
        {78, 82, 80, 85, 79, 83, 81, 84, 77, 80},  // 演習型
        {72, 75, 74, 77, 73, 76, 74, 78, 71, 75}   // PBL型
    };

    std::vector<std::string> method_names = {"講義型", "演習型", "PBL型"};
    for (std::size_t i = 0; i < teaching_methods.size(); ++i) {
        std::cout << method_names[i] << "の平均: "
                  << statcpp::mean(teaching_methods[i].begin(), teaching_methods[i].end())
                  << "\n";
    }

    auto result1 = statcpp::one_way_anova(teaching_methods);
    print_anova_table_oneway(result1);

    std::cout << "\n判定: "
              << (result1.between.p_value < 0.05 ? "教授法間で有意差あり（H₀を棄却）"
                                                  : "有意差なし") << "\n";
    std::cout << "総平均: " << result1.grand_mean << "\n";

    // ケース2: 有意差がないケース
    print_subsection("ケース2: 差がないケース");
    std::cout << "シナリオ: 3つの店舗の売上（万円）\n";

    std::vector<std::vector<double>> stores = {
        {120, 135, 128, 142, 131},
        {125, 132, 130, 138, 128},
        {122, 137, 126, 140, 133}
    };

    for (std::size_t i = 0; i < stores.size(); ++i) {
        std::cout << "店舗" << static_cast<char>('A' + i) << "の平均: "
                  << statcpp::mean(stores[i].begin(), stores[i].end()) << "\n";
    }

    auto result2 = statcpp::one_way_anova(stores);
    print_anova_table_oneway(result2);

    std::cout << "\n判定: "
              << (result2.between.p_value > 0.05 ? "店舗間で有意差なし" : "有意差あり") << "\n";

    // ケース3: 4群の比較
    print_subsection("ケース3: 4つの肥料の効果比較");
    std::cout << "シナリオ: 4種類の肥料による植物の成長（cm）\n";

    std::vector<std::vector<double>> fertilizers = {
        {15.2, 16.1, 14.8, 15.9, 15.5, 16.0},  // 肥料A
        {18.3, 19.1, 17.8, 18.5, 18.9, 18.2},  // 肥料B
        {16.5, 17.2, 16.8, 17.0, 16.7, 17.1},  // 肥料C
        {14.1, 15.0, 14.5, 14.8, 14.3, 14.7}   // 肥料D（対照）
    };

    std::vector<std::string> fert_names = {"A", "B", "C", "D(対照)"};
    for (std::size_t i = 0; i < fertilizers.size(); ++i) {
        std::cout << "肥料" << fert_names[i] << "の平均: "
                  << statcpp::mean(fertilizers[i].begin(), fertilizers[i].end()) << "\n";
    }

    auto result3 = statcpp::one_way_anova(fertilizers);
    print_anova_table_oneway(result3);

    std::cout << "\n判定: "
              << (result3.between.p_value < 0.05 ? "肥料間で有意差あり" : "有意差なし") << "\n";

    // 効果量の計算
    print_subsection("ケース1の効果量");
    double eta_sq = statcpp::eta_squared(result1);
    double omega_sq = statcpp::omega_squared(result1);
    double cohens_f_val = statcpp::cohens_f(result1);

    std::cout << "η² (Eta-squared):     " << eta_sq << "\n";
    std::cout << "ω² (Omega-squared):   " << omega_sq << "\n";
    std::cout << "Cohen's f:            " << cohens_f_val << "\n";
    std::cout << "\n効果量の解釈:\n";
    std::cout << "  η² - 説明された分散の割合（" << (eta_sq * 100) << "%）\n";
    if (cohens_f_val < 0.10) std::cout << "  Cohen's f: 小さい効果\n";
    else if (cohens_f_val < 0.25) std::cout << "  Cohen's f: 中程度の効果\n";
    else if (cohens_f_val < 0.40) std::cout << "  Cohen's f: 大きい効果\n";
    else std::cout << "  Cohen's f: 非常に大きい効果\n";
}

/**
 * @brief two_way_anova() のサンプル
 *
 * 【目的】
 * 二元配置分散分析は、2つの独立変数（要因）とその交互作用が
 * 従属変数に及ぼす影響を同時に分析します。
 *
 * 【数式】
 * F_A = MS_A / MS_error  （要因Aの主効果）
 * F_B = MS_B / MS_error  （要因Bの主効果）
 * F_AB = MS_AB / MS_error（交互作用効果）
 *
 * 【使用場面】
 * - 2要因の効果を同時に検証したい場合
 * - 要因間の交互作用を調べたい場合
 * - 実験計画法における多要因実験
 *
 * 【注意点】
 * - 交互作用が有意な場合、主効果の解釈は慎重に
 * - 各セルに同数のデータが必要（均衡計画）
 * - 正規性と等分散性の仮定
 */
void example_two_way_anova()
{
    print_section("two_way_anova() - 二元配置分散分析");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("ケース: 温度×湿度による製品品質");
    std::cout << "シナリオ: 温度(低/高)と湿度(低/中/高)が製品品質に与える影響\n";
    std::cout << "各条件で4回測定\n\n";

    // data[温度][湿度][繰り返し]
    // 温度: 0=低温, 1=高温
    // 湿度: 0=低湿度, 1=中湿度, 2=高湿度
    std::vector<std::vector<std::vector<double>>> data = {
        {   // 低温
            {85, 87, 86, 88},  // 低湿度
            {82, 84, 83, 85},  // 中湿度
            {78, 80, 79, 81}   // 高湿度
        },
        {   // 高温
            {88, 90, 89, 91},  // 低湿度
            {90, 92, 91, 93},  // 中湿度
            {87, 89, 88, 90}   // 高湿度
        }
    };

    // セル平均を表示
    std::cout << "セル平均:\n";
    std::cout << "         低湿度  中湿度  高湿度\n";
    for (std::size_t temp = 0; temp < 2; ++temp) {
        std::cout << (temp == 0 ? "低温    " : "高温    ");
        for (std::size_t hum = 0; hum < 3; ++hum) {
            double mean = statcpp::mean(data[temp][hum].begin(), data[temp][hum].end());
            std::cout << std::setw(7) << mean << " ";
        }
        std::cout << "\n";
    }

    auto result = statcpp::two_way_anova(data);
    print_anova_table_twoway(result);

    std::cout << "\n判定:\n";
    std::cout << "  要因A（温度）: "
              << (result.factor_a.p_value < 0.05 ? "有意（p < 0.05）" : "有意でない") << "\n";
    std::cout << "  要因B（湿度）: "
              << (result.factor_b.p_value < 0.05 ? "有意（p < 0.05）" : "有意でない") << "\n";
    std::cout << "  交互作用（温度×湿度）: "
              << (result.interaction.p_value < 0.05 ? "有意（p < 0.05）" : "有意でない") << "\n";

    // 効果量（偏η²）
    print_subsection("効果量（偏η²）");
    double partial_eta_a = statcpp::partial_eta_squared_a(result);
    double partial_eta_b = statcpp::partial_eta_squared_b(result);
    double partial_eta_ab = statcpp::partial_eta_squared_interaction(result);

    std::cout << "温度の偏η²:       " << partial_eta_a << "\n";
    std::cout << "湿度の偏η²:       " << partial_eta_b << "\n";
    std::cout << "交互作用の偏η²:   " << partial_eta_ab << "\n";

    // 交互作用なしのケース
    print_subsection("ケース2: 交互作用なし");
    std::cout << "シナリオ: 薬剤A×薬剤Bの組み合わせ効果（加法的）\n";

    std::vector<std::vector<std::vector<double>>> data2 = {
        {   // 薬剤A: なし
            {10, 11, 10, 11},  // 薬剤B: なし
            {15, 16, 15, 16}   // 薬剤B: あり
        },
        {   // 薬剤A: あり
            {20, 21, 20, 21},  // 薬剤B: なし
            {25, 26, 25, 26}   // 薬剤B: あり
        }
    };

    std::cout << "\nセル平均:\n";
    std::cout << "          薬剤B:なし  薬剤B:あり\n";
    for (std::size_t a_level = 0; a_level < 2; ++a_level) {
        std::cout << (a_level == 0 ? "薬剤A:なし  " : "薬剤A:あり  ");
        for (std::size_t b_level = 0; b_level < 2; ++b_level) {
            double mean = statcpp::mean(data2[a_level][b_level].begin(),
                                        data2[a_level][b_level].end());
            std::cout << std::setw(10) << mean << "  ";
        }
        std::cout << "\n";
    }

    auto result2 = statcpp::two_way_anova(data2);
    print_anova_table_twoway(result2);

    std::cout << "\n解釈: 交互作用が有意でない場合、各要因の効果は独立（加法的）\n";
}

/**
 * @brief tukey_hsd() のサンプル
 *
 * 【目的】
 * Tukeyの正直有意差（HSD）検定により、
 * ANOVAで有意差が出た後、どの群間に差があるかを調べる事後比較です。
 * Studentized range分布（q分布）を使用します
 * （不等サンプルサイズにはTukey-Kramer法）。
 *
 * 【数式】
 * q = |mean_i - mean_j| / SE
 * ここで SE = sqrt(MSE/2 × (1/n_i + 1/n_j))、
 * p値はk群・df_error自由度のStudentized range分布から算出
 *
 * 【使用場面】
 * - ANOVA後の全ての群間比較
 * - 等サンプルサイズ・不等サンプルサイズの両方に対応（Tukey-Kramer）
 * - 全ペア比較において族単位の第一種過誤率を正確に制御
 *
 * 【注意点】
 * - ANOVAが有意な場合のみ実施
 * - 全ての可能なペア比較を行う
 * - 全ペア比較ではBonferroni法より検出力が高い
 */
void example_tukey_hsd()
{
    print_section("tukey_hsd() - Tukey HSD（事後比較）");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: 4つのダイエット法の効果");
    std::cout << "体重減少量（kg）を比較\n";

    std::vector<std::vector<double>> diets = {
        {5.2, 6.1, 4.8, 5.9, 5.5, 6.0},  // ダイエットA
        {8.3, 9.1, 7.8, 8.5, 8.9, 8.2},  // ダイエットB
        {6.5, 7.2, 6.8, 7.0, 6.7, 7.1},  // ダイエットC
        {4.1, 5.0, 4.5, 4.8, 4.3, 4.7}   // ダイエットD
    };

    std::vector<std::string> diet_names = {"A", "B", "C", "D"};
    for (std::size_t i = 0; i < diets.size(); ++i) {
        std::cout << "ダイエット" << diet_names[i] << "の平均: "
                  << statcpp::mean(diets[i].begin(), diets[i].end()) << " kg\n";
    }

    // まずANOVAを実施
    auto anova_result = statcpp::one_way_anova(diets);
    print_anova_table_oneway(anova_result);

    if (anova_result.between.p_value < 0.05) {
        std::cout << "\nANOVAで有意差が検出されたため、事後比較を実施します。\n";

        auto tukey_result = statcpp::tukey_hsd(anova_result, diets, 0.05);

        std::cout << "\n" << tukey_result.method << " の結果:\n";
        std::cout << "────────────────────────────────────────────────────────────\n";
        std::cout << "比較       平均差     SE      統計量   p値     95%CI        判定\n";
        std::cout << "────────────────────────────────────────────────────────────\n";

        for (const auto& comp : tukey_result.comparisons) {
            std::cout << diet_names[comp.group1] << " vs " << diet_names[comp.group2]
                      << "  " << std::setw(7) << comp.mean_diff
                      << "  " << std::setw(6) << comp.se
                      << "  " << std::setw(7) << comp.statistic
                      << "  " << std::setw(6) << comp.p_value
                      << "  [" << std::setw(6) << comp.lower << ", " << std::setw(6) << comp.upper << "]"
                      << "  " << (comp.significant ? "有意*" : "n.s.") << "\n";
        }
        std::cout << "────────────────────────────────────────────────────────────\n";
        std::cout << "* 有意水準 α = " << tukey_result.alpha << "\n";
    }
}

/**
 * @brief bonferroni_posthoc() のサンプル
 *
 * 【目的】
 * Bonferroni法は、多重比較の際に有意水準をペア数で調整する
 * 保守的な事後比較法です。
 *
 * 【数式】
 * 調整済みα = α / m  （m はペア数）
 * または p_adjusted = p × m
 *
 * 【使用場面】
 * - 比較する群が少ない場合
 * - 保守的な判定を望む場合
 * - 計算が簡単でわかりやすい
 *
 * 【注意点】
 * - 非常に保守的（検出力が低い）
 * - 群数が多いと過度に保守的になる
 */
void example_bonferroni_posthoc()
{
    print_section("bonferroni_posthoc() - Bonferroni法（事後比較）");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: 3つの治療法の効果");

    std::vector<std::vector<double>> treatments = {
        {72, 75, 74, 77, 73, 76, 74, 78},  // 治療A
        {65, 68, 67, 70, 66, 69, 67, 71},  // 治療B（対照）
        {78, 81, 80, 83, 79, 82, 80, 84}   // 治療C
    };

    auto anova_result = statcpp::one_way_anova(treatments);
    auto bonf_result = statcpp::bonferroni_posthoc(anova_result, 0.05);

    std::cout << "\n" << bonf_result.method << " の結果:\n";
    std::cout << "比較数: " << bonf_result.comparisons.size() << "\n";
    std::cout << "調整済みα: " << (0.05 / bonf_result.comparisons.size()) << "\n\n";

    for (const auto& comp : bonf_result.comparisons) {
        std::cout << "群" << comp.group1 + 1 << " vs 群" << comp.group2 + 1
                  << ": 平均差 = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[有意]" : "[n.s.]") << "\n";
    }
}

/**
 * @brief dunnett_posthoc() のサンプル
 *
 * 【目的】
 * Dunnett法は、対照群と他の各群を比較する専用の事後比較法です。
 * 全ペア比較より検出力が高い。
 *
 * 【数式】
 * t = (x̄ᵢ - x̄_control) / SE
 * 臨界値はDunnett分布から求める
 *
 * 【使用場面】
 * - 対照群（プラセボ、標準治療）との比較
 * - 複数の実験群 vs 1つの対照群
 * - 薬剤開発における用量反応試験
 *
 * 【注意点】
 * - 対照群が明確に定義されている必要
 * - 対照群以外の群間比較はしない
 */
void example_dunnett_posthoc()
{
    print_section("dunnett_posthoc() - Dunnett法（対照群との比較）");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: 3種類の新薬 vs プラセボ");
    std::cout << "痛み軽減スコア（0-100）\n";

    std::vector<std::vector<double>> drugs = {
        {35, 38, 36, 40, 37, 39, 36, 38},  // プラセボ（対照群）
        {52, 55, 54, 57, 53, 56, 54, 58},  // 新薬A
        {48, 51, 50, 53, 49, 52, 50, 54},  // 新薬B
        {45, 48, 47, 50, 46, 49, 47, 51}   // 新薬C
    };

    std::vector<std::string> drug_names = {"プラセボ", "新薬A", "新薬B", "新薬C"};
    for (std::size_t i = 0; i < drugs.size(); ++i) {
        std::cout << drug_names[i] << "の平均: "
                  << statcpp::mean(drugs[i].begin(), drugs[i].end()) << "\n";
    }

    auto anova_result = statcpp::one_way_anova(drugs);
    auto dunnett_result = statcpp::dunnett_posthoc(anova_result, 0, 0.05);  // 0 = プラセボ

    std::cout << "\n" << dunnett_result.method << " の結果（対照群 = プラセボ）:\n";
    std::cout << "────────────────────────────────────────────────────\n";

    for (const auto& comp : dunnett_result.comparisons) {
        std::cout << drug_names[comp.group1] << " vs " << drug_names[comp.group2]
                  << ": 差 = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[プラセボより有意に効果的]" : "[有意差なし]")
                  << "\n";
    }
}

/**
 * @brief scheffe_posthoc() のサンプル
 *
 * 【目的】
 * Scheffe法は、全ての可能な対比（線形結合）を検定できる
 * 最も柔軟だが保守的な事後比較法です。
 *
 * 【数式】
 * S = √[(k-1) × F_critical]
 * 対比 C が有意 ⇔ |C/SE(C)| > S
 *
 * 【使用場面】
 * - 複雑な対比（例: A vs (B+C)/2）を検定したい場合
 * - データを見た後で対比を決める場合
 * - 最も柔軟な事後比較が必要な場合
 *
 * 【注意点】
 * - 非常に保守的（検出力が低い）
 * - 単純なペア比較には不向き
 */
void example_scheffe_posthoc()
{
    print_section("scheffe_posthoc() - Scheffe法（事後比較）");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: 4つの広告手法の効果");

    std::vector<std::vector<double>> ads = {
        {45, 48, 47, 50, 46, 49},  // TV広告
        {52, 55, 54, 57, 53, 56},  // Web広告
        {42, 45, 44, 47, 43, 46},  // 新聞広告
        {38, 41, 40, 43, 39, 42}   // ラジオ広告
    };

    auto anova_result = statcpp::one_way_anova(ads);
    auto scheffe_result = statcpp::scheffe_posthoc(anova_result, 0.05);

    std::cout << "\n" << scheffe_result.method << " の結果:\n";
    std::cout << "Scheffe法は全ての対比に対して保護的\n\n";

    std::vector<std::string> ad_names = {"TV", "Web", "新聞", "ラジオ"};
    for (const auto& comp : scheffe_result.comparisons) {
        std::cout << ad_names[comp.group1] << " vs " << ad_names[comp.group2]
                  << ": 差 = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[有意]" : "[n.s.]") << "\n";
    }

    std::cout << "\nScheffe法の利点: 事後に複雑な対比も検定可能\n";
    std::cout << "例: (TV + Web) / 2 vs (新聞 + ラジオ) / 2 の比較も可能\n";
}

/**
 * @brief one_way_ancova() のサンプル
 *
 * 【目的】
 * 共分散分析（ANCOVA）は、共変量（交絡変数）の影響を統制して
 * 群間の差を検定します。
 *
 * 【数式】
 * 調整済み平均 = ȳᵢ - b(x̄ᵢ - x̄_grand)
 * ここで b は共通回帰係数、x は共変量
 *
 * 【使用場面】
 * - 処理前のベースライン値を統制したい場合
 * - 年齢、性別などの共変量の影響を除去したい場合
 * - 実験統制が不十分な場合の補正
 *
 * 【注意点】
 * - 群間で回帰の傾きが等しいことが前提
 * - 共変量と従属変数の線形関係を仮定
 * - 共変量は処理と独立であるべき
 */
void example_one_way_ancova()
{
    print_section("one_way_ancova() - 一元配置共分散分析");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("シナリオ: 3つの学習法の効果（事前テストで調整）");
    std::cout << "従属変数: 事後テスト得点\n";
    std::cout << "共変量:   事前テスト得点\n\n";

    // (事後得点, 事前得点)のペア
    std::vector<std::vector<std::pair<double, double>>> learning_methods = {
        {   // 学習法A
            {75, 60}, {80, 65}, {78, 62}, {82, 68}, {77, 63}, {79, 64}
        },
        {   // 学習法B
            {85, 70}, {90, 75}, {88, 72}, {92, 78}, {87, 73}, {89, 74}
        },
        {   // 学習法C
            {78, 65}, {83, 70}, {81, 67}, {85, 73}, {80, 68}, {82, 69}
        }
    };

    std::vector<std::string> method_labels = {"学習法A", "学習法B", "学習法C"};

    // 各群の生の平均を表示
    std::cout << "各群の平均:\n";
    for (std::size_t i = 0; i < learning_methods.size(); ++i) {
        double sum_post = 0.0, sum_pre = 0.0;
        for (const auto& [post, pre] : learning_methods[i]) {
            sum_post += post;
            sum_pre += pre;
        }
        std::cout << method_labels[i]
                  << " - 事前: " << (sum_pre / learning_methods[i].size())
                  << ", 事後: " << (sum_post / learning_methods[i].size()) << "\n";
    }

    auto ancova_result = statcpp::one_way_ancova(learning_methods);

    std::cout << "\nANCOVA表:\n";
    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "変動要因          SS       df      MS       F      p値\n";
    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "共変量(事前)   " << std::setw(8) << ancova_result.ss_covariate
              << "  " << std::setw(4) << ancova_result.df_covariate
              << "  " << std::setw(8) << ancova_result.ms_covariate
              << "  " << std::setw(6) << ancova_result.f_covariate
              << "  " << std::setw(6) << ancova_result.p_covariate << "\n";
    std::cout << "処理(学習法)   " << std::setw(8) << ancova_result.ss_treatment
              << "  " << std::setw(4) << ancova_result.df_treatment
              << "  " << std::setw(8) << ancova_result.ms_treatment
              << "  " << std::setw(6) << ancova_result.f_treatment
              << "  " << std::setw(6) << ancova_result.p_treatment << "\n";
    std::cout << "誤差           " << std::setw(8) << ancova_result.ss_error
              << "  " << std::setw(4) << ancova_result.df_error
              << "  " << std::setw(8) << ancova_result.ms_error << "\n";
    std::cout << "────────────────────────────────────────────────────────────\n";

    std::cout << "\n判定:\n";
    std::cout << "  共変量（事前得点）: "
              << (ancova_result.p_covariate < 0.05 ? "有意（事前得点は事後に影響）" : "有意でない")
              << "\n";
    std::cout << "  処理（学習法）: "
              << (ancova_result.p_treatment < 0.05 ? "有意（学習法間で差あり）" : "有意でない")
              << "\n";

    std::cout << "\n調整済み平均（事前得点の影響を除去）:\n";
    for (std::size_t i = 0; i < ancova_result.adjusted_means.size(); ++i) {
        std::cout << "  " << method_labels[i] << ": " << ancova_result.adjusted_means[i] << "\n";
    }

    std::cout << "\n解釈: 事前得点の差を統制した上で学習法の効果を評価\n";
}

// ============================================================================
// サマリー出力
// ============================================================================

void print_summary()
{
    print_section("サマリー: anova.hpp の関数一覧");

    std::cout << R"(
┌──────────────────────────────────────────────────────────────────────┐
│ 関数名                      用途                                     │
├──────────────────────────────────────────────────────────────────────┤
│ one_way_anova()             一元配置分散分析（1要因）                │
│ two_way_anova()             二元配置分散分析（2要因 + 交互作用）     │
│ tukey_hsd()                 Tukey HSD法（全ペア比較）                │
│ bonferroni_posthoc()        Bonferroni法（保守的な多重比較）         │
│ dunnett_posthoc()           Dunnett法（対照群との比較）              │
│ scheffe_posthoc()           Scheffe法（最も柔軟な事後比較）          │
│ one_way_ancova()            一元配置共分散分析（共変量で調整）       │
│ eta_squared()               η²（効果量）                            │
│ omega_squared()             ω²（不偏効果量推定）                     │
│ cohens_f()                  Cohen's f（効果量）                      │
└──────────────────────────────────────────────────────────────────────┘

【ANOVA実施の流れ】
  1. 正規性と等分散性の確認
  2. ANOVAの実施（F検定）
  3. F検定が有意 → 事後比較で具体的な差を特定
  4. 効果量の報告

【事後比較法の選択】
  ┌─────────────────────────────────────────────┐
  │ 目的                   推奨法               │
  ├─────────────────────────────────────────────┤
  │ 全ペア比較             Tukey HSD            │
  │ 少数の比較             Bonferroni           │
  │ 対照群との比較         Dunnett              │
  │ 複雑な対比             Scheffe              │
  └─────────────────────────────────────────────┘

【効果量の解釈】
  η² (Eta-squared):
    - 0.01: 小さい効果
    - 0.06: 中程度の効果
    - 0.14: 大きい効果

  Cohen's f:
    - 0.10: 小さい効果
    - 0.25: 中程度の効果
    - 0.40: 大きい効果

【注意事項】
  - ANOVAの前提: 正規性、等分散性、独立性
  - 事後比較はANOVAが有意な場合のみ
  - 交互作用が有意な場合、主効果の解釈は慎重に
  - 効果量も必ず報告すること
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main()
{
    std::cout << "==========================================================\n";
    std::cout << " statcpp 分散分析（ANOVA）関数 サンプルコード\n";
    std::cout << "==========================================================\n";

    example_one_way_anova();
    example_two_way_anova();
    example_tukey_hsd();
    example_bonferroni_posthoc();
    example_dunnett_posthoc();
    example_scheffe_posthoc();
    example_one_way_ancova();
    print_summary();

    return 0;
}
