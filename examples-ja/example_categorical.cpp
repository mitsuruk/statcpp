/**
 * @file example_categorical.cpp
 * @brief カテゴリカルデータ分析のサンプルコード
 *
 * 分割表（クロス集計表）、オッズ比、相対リスク、リスク差、
 * 治療必要数(NNT)等のカテゴリカルデータ分析手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/categorical.hpp"

int main() {
    std::cout << "=== カテゴリカルデータ分析の例 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. 分割表（クロス集計表）の作成
    // ============================================================================
    std::cout << "\n1. 分割表（クロス集計表）の作成" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // 例: 性別（0=男性, 1=女性）と製品の好み（0=製品A, 1=製品B, 2=製品C）
    std::vector<std::size_t> gender = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                        0, 0, 1, 1, 0, 1, 0, 1, 0, 1};
    std::vector<std::size_t> product = {0, 0, 1, 1, 2, 0, 1, 1, 2, 2,
                                         0, 1, 0, 1, 2, 2, 1, 0, 2, 1};

    auto ct = statcpp::contingency_table(gender, product);

    std::cout << "性別 vs 製品の好み:" << std::endl;
    std::cout << "           製品A     製品B     製品C     合計" << std::endl;

    const char* row_names[] = {"男性  ", "女性  "};
    for (std::size_t i = 0; i < ct.n_rows; ++i) {
        std::cout << row_names[i] << "     ";
        for (std::size_t j = 0; j < ct.n_cols; ++j) {
            std::cout << std::setw(9) << ct.table[i][j] << "  ";
        }
        std::cout << std::setw(5) << ct.row_totals[i] << std::endl;
    }

    std::cout << "合計         ";
    for (std::size_t j = 0; j < ct.n_cols; ++j) {
        std::cout << std::setw(9) << ct.col_totals[j] << "  ";
    }
    std::cout << std::setw(5) << ct.total << std::endl;

    // ============================================================================
    // 2. オッズ比（Odds Ratio）
    // ============================================================================
    std::cout << "\n2. オッズ比の分析" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // 2x2分割表の例: 治療効果の研究
    //              Success  Failure
    // Treatment       50       10
    // Control         30       20

    std::vector<std::vector<std::size_t>> treatment_table = {
        {50, 10},  // Treatment: 50 successes, 10 failures
        {30, 20}   // Control:   30 successes, 20 failures
    };

    auto or_result = statcpp::odds_ratio(treatment_table);

    std::cout << "治療効果の研究:" << std::endl;
    std::cout << "             成功     失敗" << std::endl;
    std::cout << "治療群         50       10" << std::endl;
    std::cout << "対照群         30       20" << std::endl;

    std::cout << "\nオッズ比の結果:" << std::endl;
    std::cout << "  オッズ比: " << or_result.odds_ratio << std::endl;
    std::cout << "  対数オッズ比: " << or_result.log_odds_ratio << std::endl;
    std::cout << "  SE(log OR): " << or_result.se_log_odds_ratio << std::endl;
    std::cout << "  95% 信頼区間: [" << or_result.ci_lower << ", " << or_result.ci_upper << "]" << std::endl;

    std::cout << "\n解釈:" << std::endl;
    if (or_result.odds_ratio > 1.0) {
        std::cout << "  治療群は対照群より" << or_result.odds_ratio
                  << "倍高い成功のオッズを持つ" << std::endl;
    } else if (or_result.odds_ratio < 1.0) {
        std::cout << "  治療群は対照群より成功のオッズが低い" << std::endl;
    } else {
        std::cout << "  グループ間でオッズに差はない" << std::endl;
    }

    // セル値を直接指定する方法
    auto or_direct = statcpp::odds_ratio(50, 10, 30, 20);
    std::cout << "  (検証: OR = " << or_direct.odds_ratio << ")" << std::endl;

    // ============================================================================
    // 3. 相対リスク（Risk Ratio）
    // ============================================================================
    std::cout << "\n3. 相対リスク（リスク比）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto rr_result = statcpp::relative_risk(treatment_table);

    std::cout << "相対リスクの結果:" << std::endl;
    std::cout << "  相対リスク: " << rr_result.relative_risk << std::endl;
    std::cout << "  対数RR: " << rr_result.log_relative_risk << std::endl;
    std::cout << "  SE(log RR): " << rr_result.se_log_relative_risk << std::endl;
    std::cout << "  95% 信頼区間: [" << rr_result.ci_lower << ", " << rr_result.ci_upper << "]" << std::endl;

    double risk_treatment = 50.0 / (50.0 + 10.0);
    double risk_control = 30.0 / (30.0 + 20.0);

    std::cout << "\nリスクの計算:" << std::endl;
    std::cout << "  治療群のリスク: " << risk_treatment << " (" << (risk_treatment * 100) << "%)" << std::endl;
    std::cout << "  対照群のリスク: " << risk_control << " (" << (risk_control * 100) << "%)" << std::endl;

    std::cout << "\n解釈:" << std::endl;
    if (rr_result.relative_risk > 1.0) {
        std::cout << "  治療群は対照群より" << rr_result.relative_risk
                  << "倍高い成功のリスクを持つ" << std::endl;
    } else if (rr_result.relative_risk < 1.0) {
        std::cout << "  治療群は対照群より成功のリスクが低い" << std::endl;
    }

    // ============================================================================
    // 4. リスク差（Attributable Risk）
    // ============================================================================
    std::cout << "\n4. リスク差（寄与リスク）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto rd_result = statcpp::risk_difference(treatment_table);

    std::cout << "リスク差の結果:" << std::endl;
    std::cout << "  リスク差: " << rd_result.risk_difference << std::endl;
    std::cout << "  標準誤差: " << rd_result.se << std::endl;
    std::cout << "  95% 信頼区間: [" << rd_result.ci_lower << ", " << rd_result.ci_upper << "]" << std::endl;

    std::cout << "\n解釈:" << std::endl;
    std::cout << "  リスクの絶対差: " << (rd_result.risk_difference * 100) << "%" << std::endl;
    if (rd_result.risk_difference > 0) {
        std::cout << "  治療により成功が" << (rd_result.risk_difference * 100)
                  << "パーセントポイント増加" << std::endl;
    }

    // ============================================================================
    // 5. 治療必要数（Number Needed to Treat, NNT）
    // ============================================================================
    std::cout << "\n5. 治療必要数（NNT）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double nnt = statcpp::number_needed_to_treat(treatment_table);

    std::cout << "NNT: " << nnt << std::endl;
    std::cout << "\n解釈:" << std::endl;
    std::cout << "  対照群と比較して1人の追加的な成功を得るために" << std::endl;
    std::cout << "  約" << std::ceil(nnt) << "人の患者を治療する必要がある" << std::endl;

    // ============================================================================
    // 6. 疾病研究の例
    // ============================================================================
    std::cout << "\n6. 疾病曝露研究の例" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // 喫煙と肺がんの関連性（仮想データ）
    //              Lung Cancer+  Lung Cancer-
    // Smoker            80              120
    // Non-smoker        20              180

    std::vector<std::vector<std::size_t>> smoking_table = {
        {80, 120},   // Smokers
        {20, 180}    // Non-smokers
    };

    std::cout << "喫煙と肺がんの研究:" << std::endl;
    std::cout << "             がん+    がん-" << std::endl;
    std::cout << "喫煙者          80      120" << std::endl;
    std::cout << "非喫煙者        20      180" << std::endl;

    auto smoking_or = statcpp::odds_ratio(smoking_table);
    auto smoking_rr = statcpp::relative_risk(smoking_table);
    auto smoking_rd = statcpp::risk_difference(smoking_table);

    std::cout << "\n結果:" << std::endl;
    std::cout << "  オッズ比: " << smoking_or.odds_ratio
              << " (95% 信頼区間: [" << smoking_or.ci_lower << ", " << smoking_or.ci_upper << "])" << std::endl;
    std::cout << "  相対リスク: " << smoking_rr.relative_risk
              << " (95% 信頼区間: [" << smoking_rr.ci_lower << ", " << smoking_rr.ci_upper << "])" << std::endl;
    std::cout << "  リスク差: " << smoking_rd.risk_difference
              << " (95% 信頼区間: [" << smoking_rd.ci_lower << ", " << smoking_rd.ci_upper << "])" << std::endl;

    double smoker_risk = 80.0 / (80.0 + 120.0);
    double nonsmoker_risk = 20.0 / (20.0 + 180.0);

    std::cout << "\nがん発生率:" << std::endl;
    std::cout << "  喫煙者: " << (smoker_risk * 100) << "%" << std::endl;
    std::cout << "  非喫煙者: " << (nonsmoker_risk * 100) << "%" << std::endl;

    std::cout << "\n解釈:" << std::endl;
    std::cout << "  喫煙者は非喫煙者の" << smoking_or.odds_ratio << "倍の肺がんのオッズを持つ" << std::endl;
    std::cout << "  喫煙者は非喫煙者の" << smoking_rr.relative_risk << "倍の肺がんのリスクを持つ" << std::endl;

    // ============================================================================
    // 7. ワクチン効果の例
    // ============================================================================
    std::cout << "\n7. ワクチン効果の研究" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    //              Infected  Not Infected
    // Vaccinated       5          495
    // Unvaccinated    50          450

    std::vector<std::vector<std::size_t>> vaccine_table = {
        {5, 495},     // Vaccinated
        {50, 450}     // Unvaccinated
    };

    std::cout << "ワクチン試験の結果:" << std::endl;
    std::cout << "             感染あり  感染なし" << std::endl;
    std::cout << "接種群           5          495" << std::endl;
    std::cout << "非接種群        50          450" << std::endl;

    auto vaccine_rr = statcpp::relative_risk(vaccine_table);

    double attack_rate_vaccinated = 5.0 / 500.0;
    double attack_rate_unvaccinated = 50.0 / 500.0;
    double vaccine_efficacy = 1.0 - vaccine_rr.relative_risk;

    std::cout << "\n罹患率:" << std::endl;
    std::cout << "  接種群: " << (attack_rate_vaccinated * 100) << "%" << std::endl;
    std::cout << "  非接種群: " << (attack_rate_unvaccinated * 100) << "%" << std::endl;

    std::cout << "\n相対リスク: " << vaccine_rr.relative_risk << std::endl;
    std::cout << "ワクチン効果: " << (vaccine_efficacy * 100) << "%" << std::endl;

    std::cout << "\n解釈:" << std::endl;
    std::cout << "  ワクチンは感染リスクを" << (vaccine_efficacy * 100) << "%減少させる" << std::endl;

    try {
        double vaccine_nnt = statcpp::number_needed_to_treat(vaccine_table);
        std::cout << "  NNT: " << std::ceil(vaccine_nnt)
                  << " (1回の感染を予防するためにこの人数に接種が必要)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  (NNTの計算には正のリスク差が必要)" << std::endl;
    }

    // ============================================================================
    // 8. オッズ比とリスク比の比較
    // ============================================================================
    std::cout << "\n8. オッズ比とリスク比の比較" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "どの指標を使用するか:" << std::endl;
    std::cout << "  - 相対リスク (RR): コホート研究およびランダム化試験に使用" << std::endl;
    std::cout << "    より直感的な解釈が可能" << std::endl;
    std::cout << "  - オッズ比 (OR): 症例対照研究に使用" << std::endl;
    std::cout << "    アウトカムが稀(<10%)な場合、RRに近似" << std::endl;
    std::cout << "  - リスク差 (RD): 絶対的な効果量を示す" << std::endl;
    std::cout << "    臨床的意思決定に有用" << std::endl;

    std::cout << "\n治療研究の場合:" << std::endl;
    std::cout << "  オッズ比: " << or_result.odds_ratio << std::endl;
    std::cout << "  リスク比: " << rr_result.relative_risk << std::endl;
    std::cout << "  注意: 成功率が稀でないため、ORとRRは異なる" << std::endl;

    std::cout << "\n=== 例が正常に完了しました ===" << std::endl;

    return 0;
}
