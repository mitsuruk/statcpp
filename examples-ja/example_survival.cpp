/**
 * @file example_survival.cpp
 * @brief 生存時間解析のサンプルコード
 *
 * Kaplan-Meier生存曲線、ログランク検定、Nelson-Aalen累積ハザード推定等の
 * 生存時間解析手法の使用例を示します。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include "statcpp/survival.hpp"

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
    // 1. Kaplan-Meier 生存曲線
    // ============================================================================
    print_section("1. Kaplan-Meier生存曲線");

    std::cout << R"(
【概念】
打ち切りデータを考慮した生存関数の推定
時間とともに生存確率がどう変化するかを示す

【実例: 患者の生存期間分析】
10人の患者の生存期間（月単位）を追跡
→ 打ち切り（censored）データも含む
)";

    // 生存時間データ（月）
    std::vector<double> times = {1, 3, 4, 5, 8, 10, 12, 15, 18, 20};
    // イベント発生（1=死亡, 0=打ち切り）
    std::vector<bool> events = {1, 1, 0, 1, 1, 0, 1, 1, 0, 1};

    auto km_result = statcpp::kaplan_meier(times, events);

    print_subsection("データの概要");
    std::cout << "  サンプルサイズ: " << times.size() << "人\n";
    std::cout << "  イベント発生: " << std::count(events.begin(), events.end(), 1) << "件\n";
    std::cout << "  打ち切り: " << std::count(events.begin(), events.end(), 0) << "件\n";

    print_subsection("Kaplan-Meier生存表");
    std::cout << "  時点  リスク数  イベント  生存率      95%信頼区間\n";

    for (std::size_t i = 0; i < km_result.times.size(); ++i) {
        std::cout << std::setw(6) << km_result.times[i]
                  << std::setw(9) << km_result.n_at_risk[i]
                  << std::setw(9) << km_result.n_events[i]
                  << std::setw(10) << km_result.survival[i];
        if (km_result.ci_lower[i] > 0) {
            std::cout << "  [" << km_result.ci_lower[i] << ", " << km_result.ci_upper[i] << "]";
        }
        std::cout << std::endl;
    }
    std::cout << "→ 時間経過とともに生存率が低下\n";

    // ============================================================================
    // 2. 生存時間の中央値
    // ============================================================================
    print_section("2. 生存時間の中央値 (Median Survival Time)");

    std::cout << R"(
【概念】
生存関数が0.5になる時点
50%の被験者がこの時点を超えて生存

【実例: 中央生存期間】
患者の半数が生存している期間
→ 平均より頑健な指標
)";

    double median_time = statcpp::median_survival_time(km_result);

    print_subsection("中央生存時間");
    std::cout << "  中央値: " << median_time << "ヶ月\n";
    std::cout << "\n解釈: 被験者の50%が" << median_time << "ヶ月を超えて生存\n";

    // ============================================================================
    // 3. 特定時点での生存率
    // ============================================================================
    print_section("3. 特定時点での生存確率");

    std::cout << R"(
【概念】
指定した時点における生存確率を推定
臨床的に重要な時点（例: 6ヶ月、1年、5年）の評価

【実例: 重要時点の生存率】
6ヶ月、12ヶ月、18ヶ月時点での生存確率を評価
)";

    std::vector<double> time_points = {6, 12, 18};

    print_subsection("各時点の生存確率");
    for (double t : time_points) {
        // 指定時点以前の最後の観測値を見つける
        double survival_prob = 1.0;
        for (std::size_t i = 0; i < km_result.times.size(); ++i) {
            if (km_result.times[i] <= t) {
                survival_prob = km_result.survival[i];
            } else {
                break;
            }
        }
        std::cout << "  S(" << t << "ヶ月) = " << survival_prob
                  << " (" << (survival_prob * 100) << "%)" << std::endl;
    }
    std::cout << "→ 時間経過とともに生存確率が低下\n";

    // ============================================================================
    // 4. Log-rank 検定（2群比較）
    // ============================================================================
    print_section("4. Log-rank検定（2群の生存曲線比較）");

    std::cout << R"(
【概念】
2つのグループの生存曲線全体を比較
帰無仮説: 両群の生存曲線は同じ

【実例: 治療効果の評価】
治療群と対照群の生存期間を比較
→ 治療が生存率を改善するか検定
)";

    // グループ1（治療群）
    std::vector<double> times1 = {1, 3, 5, 8, 12, 15, 20};
    std::vector<bool> events1 = {1, 1, 1, 1, 0, 1, 0};

    // グループ2（対照群）
    std::vector<double> times2 = {2, 4, 6, 7, 9, 10, 11};
    std::vector<bool> events2 = {1, 1, 1, 1, 1, 1, 1};

    auto logrank_result = statcpp::logrank_test(times1, events1, times2, events2);

    print_subsection("グループ情報");
    std::cout << "  治療群:\n";
    std::cout << "    n = " << times1.size() << ", イベント = "
              << std::count(events1.begin(), events1.end(), 1) << "\n";

    std::cout << "\n  対照群:\n";
    std::cout << "    n = " << times2.size() << ", イベント = "
              << std::count(events2.begin(), events2.end(), 1) << "\n";

    print_subsection("Log-rank検定結果");
    std::cout << "  カイ二乗統計量: " << logrank_result.statistic << "\n";
    std::cout << "  自由度: " << logrank_result.df << "\n";
    std::cout << "  P値: " << logrank_result.p_value << "\n";

    std::cout << "\n解釈 (α = 0.05):\n";
    if (logrank_result.p_value < 0.05) {
        std::cout << "  → 両群の生存曲線に有意な差がある\n";
    } else {
        std::cout << "  → 両群の生存曲線に有意な差はない\n";
    }

    // ============================================================================
    // 5. Kaplan-Meier曲線の比較
    // ============================================================================
    print_section("5. 生存曲線の比較");

    std::cout << R"(
【概念】
各群の中央生存時間を比較
治療効果の大きさを評価

【実例: 中央生存期間の差】
治療による生存期間の延長を定量化
)";

    auto km1 = statcpp::kaplan_meier(times1, events1);
    auto km2 = statcpp::kaplan_meier(times2, events2);

    print_subsection("中央生存時間の比較");
    std::cout << "  治療群の中央生存時間: " << statcpp::median_survival_time(km1) << "ヶ月\n";
    std::cout << "  対照群の中央生存時間: " << statcpp::median_survival_time(km2) << "ヶ月\n";

    double diff = statcpp::median_survival_time(km1) - statcpp::median_survival_time(km2);
    std::cout << "  差: " << diff << "ヶ月\n";
    if (diff > 0) {
        std::cout << "  → 治療群の方が生存期間が長い\n";
    }

    // ============================================================================
    // 6. Nelson-Aalen 累積ハザード推定量
    // ============================================================================
    print_section("6. Nelson-Aalen累積ハザード推定");

    std::cout << R"(
【概念】
時間経過とともに累積するリスク（ハザード）を推定
生存関数との関係: S(t) = exp(-H(t))

【実例: 累積リスクの評価】
時間とともにリスクがどう蓄積するかを分析
)";

    auto na_result = statcpp::nelson_aalen(times, events);

    print_subsection("累積ハザード推定値");
    std::cout << "  時点    累積ハザード\n";

    for (std::size_t i = 0; i < std::min(std::size_t(5), na_result.times.size()); ++i) {
        std::cout << std::setw(6) << na_result.times[i]
                  << std::setw(16) << na_result.cumulative_hazard[i] << std::endl;
    }

    std::cout << "\n→ 累積ハザードは時間とともに増加\n";
    std::cout << "→ 値が大きいほどリスクが高い\n";

    // ============================================================================
    // 7. 実用例：臨床試験データ
    // ============================================================================
    print_section("7. 実用例：臨床試験データ分析");

    std::cout << R"(
【概念】
新薬と標準治療の生存期間を比較
臨床的有用性を統計的に評価

【実例: 新薬の有効性評価】
新薬群と標準治療群の生存分析
→ 治療効果を定量的に示す
)";

    // 新薬群
    std::vector<double> new_drug_times = {6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    std::vector<bool> new_drug_events = {0, 1, 0, 1, 0, 1, 0, 0, 1, 0};

    // 標準治療群
    std::vector<double> standard_times = {4, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    std::vector<bool> standard_events = {1, 1, 1, 1, 1, 0, 1, 1, 1, 0};

    print_subsection("各治療群の生存時間");
    auto km_new = statcpp::kaplan_meier(new_drug_times, new_drug_events);
    std::cout << "  新薬群:\n";
    std::cout << "    中央生存時間: " << statcpp::median_survival_time(km_new) << "ヶ月\n";

    auto km_std = statcpp::kaplan_meier(standard_times, standard_events);
    std::cout << "\n  標準治療群:\n";
    std::cout << "    中央生存時間: " << statcpp::median_survival_time(km_std) << "ヶ月\n";

    auto trial_logrank = statcpp::logrank_test(new_drug_times, new_drug_events,
                                                standard_times, standard_events);

    print_subsection("統計的検定");
    std::cout << "  Log-rank検定のP値: " << trial_logrank.p_value << "\n";

    if (trial_logrank.p_value < 0.05) {
        double improvement = statcpp::median_survival_time(km_new) - statcpp::median_survival_time(km_std);
        std::cout << "\n結論: 新薬は生存期間を有意に改善する\n";
        std::cout << "      中央生存時間の改善: " << improvement << "ヶ月\n";
    } else {
        std::cout << "\n結論: 両治療間に有意差なし\n";
    }

    // ============================================================================
    // 8. まとめ：生存分析の解釈ガイド
    // ============================================================================
    print_section("まとめ：生存分析の解釈ガイド");

    std::cout << R"(
【重要な概念】

生存関数 S(t):
  - 時点tを超えて生存する確率
  - 1.0 (100%)から始まり時間とともに減少
  - 曲線が高いほど生存率が良い

打ち切り (Censoring):
  - 追跡不能（脱落）
  - イベント発生前の研究終了
  - 競合リスク
  → 打ち切りデータも情報として活用

中央生存時間:
  - S(t) = 0.5となる時点
  - 被験者の50%がこの時点を超えて生存
  - 外れ値に頑健

Log-rank検定:
  - 生存曲線全体を比較
  - 帰無仮説: 両曲線は等しい
  - 比例ハザードを仮定

累積ハザード:
  - 累積された総リスク
  - 生存関数との関係: S(t) = exp(-H(t))
  - ハザード比の推定に有用

【生存分析の使用場面】
┌─────────────────┬────────────────────────────────┐
│ 分野             │ 応用例                         │
├─────────────────┼────────────────────────────────┤
│ 医療・臨床       │ 治療効果評価、患者予後         │
├─────────────────┼────────────────────────────────┤
│ 工学             │ 製品寿命、信頼性分析           │
├─────────────────┼────────────────────────────────┤
│ ビジネス         │ 顧客離脱、サブスク解約         │
├─────────────────┼────────────────────────────────┤
│ 社会科学         │ 失業期間、結婚持続期間         │
└─────────────────┴────────────────────────────────┘

【分析の手順】
1. データの準備（生存時間、イベント発生、打ち切り）
2. Kaplan-Meier曲線の作成
3. 中央生存時間の算出
4. グループ間比較（Log-rank検定）
5. Cox比例ハザードモデル（より高度な分析）

【注意点】
- 打ち切りは情報的でないこと（ランダム）
- 比例ハザードの仮定を確認
- サンプルサイズの考慮
- 多重比較の補正
)";

    return 0;
}
