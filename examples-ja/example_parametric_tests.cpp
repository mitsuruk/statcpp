/**
 * @file example_parametric_tests.cpp
 * @brief statcpp::parametric_tests.hpp のサンプルコード
 *
 * このファイルでは、parametric_tests.hpp で提供される
 * パラメトリック統計検定の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - z_test()                     : 1標本z検定（既知の分散）
 * - z_test_proportion()          : 1標本比率z検定
 * - z_test_proportion_two_sample(): 2標本比率z検定
 * - t_test()                     : 1標本t検定
 * - t_test_two_sample()          : 2標本t検定（プール分散）
 * - t_test_welch()               : Welch のt検定（等分散を仮定しない）
 * - t_test_paired()              : 対応のあるt検定
 * - chisq_test_gof()             : カイ二乗適合度検定
 * - chisq_test_independence()    : カイ二乗独立性検定
 * - f_test()                     : F検定（分散比較）
 * - bonferroni_correction()      : Bonferroni補正
 * - benjamini_hochberg_correction(): BH補正（FDR制御）
 * - holm_correction()            : Holm補正
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_parametric_tests.cpp -o example_parametric_tests
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// statcpp のパラメトリック検定ヘッダー
#include "statcpp/parametric_tests.hpp"
#include "statcpp/basic_statistics.hpp"

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

// 検定結果を表示
void print_test_result(const statcpp::test_result& result, const std::string& stat_name = "統計量") {
    std::cout << stat_name << ": " << result.statistic << "\n";
    std::cout << "自由度: " << result.df << "\n";
    std::cout << "p値: " << result.p_value << "\n";

    std::string alt_str;
    switch (result.alternative) {
        case statcpp::alternative_hypothesis::less:
            alt_str = "片側（小さい方）";
            break;
        case statcpp::alternative_hypothesis::greater:
            alt_str = "片側（大きい方）";
            break;
        case statcpp::alternative_hypothesis::two_sided:
        default:
            alt_str = "両側";
            break;
    }
    std::cout << "対立仮説: " << alt_str << "\n";

    std::cout << "判定 (α=0.05): ";
    if (result.p_value < 0.05) {
        std::cout << "有意 (帰無仮説を棄却)\n";
    } else {
        std::cout << "有意でない (帰無仮説を棄却できない)\n";
    }
}

// ============================================================================
// 1. t_test() - 1標本t検定
// ============================================================================

/**
 * @brief t_test() の使用例
 *
 * 【目的】
 * 1標本t検定は、サンプルの平均が特定の値（帰無仮説の値）と
 * 統計的に有意に異なるかどうかを検定します。
 *
 * 【帰無仮説】H₀: μ = μ₀
 * 【対立仮説】H₁: μ ≠ μ₀（両側）、μ < μ₀（片側下）、μ > μ₀（片側上）
 *
 * 【使用場面】
 * - 新しい治療法の効果が基準値と異なるか
 * - 製品の品質が規格値を満たしているか
 * - 実験結果が理論値と一致するか
 *
 * 【注意点】
 * - データが正規分布に従うことを仮定
 * - 標本サイズが小さい場合（n < 30）、この仮定は重要
 */
void example_t_test() {
    print_section("1. t_test() - 1標本t検定");

    // 例: ある高校の生徒の平均身長が全国平均170cmと異なるか？
    std::vector<double> heights = {168, 172, 175, 165, 170, 178, 169, 173, 167, 174};

    print_data("身長データ (cm)", heights);
    double mu0 = 170.0;  // 帰無仮説: 平均 = 170cm
    std::cout << "帰無仮説の平均: " << mu0 << " cm\n\n";

    std::cout << "サンプル統計量:\n";
    std::cout << "  サンプルサイズ: " << statcpp::count(heights.begin(), heights.end()) << "\n";
    std::cout << "  サンプル平均: " << statcpp::mean(heights.begin(), heights.end()) << " cm\n";
    std::cout << "  サンプル標準偏差: " << statcpp::sample_stddev(heights.begin(), heights.end()) << " cm\n\n";

    // 両側検定
    print_subsection("両側検定: H₁: μ ≠ 170");
    auto result_two = statcpp::t_test(heights.begin(), heights.end(), mu0,
                                      statcpp::alternative_hypothesis::two_sided);
    print_test_result(result_two, "t統計量");

    // 片側検定（大きい）
    print_subsection("片側検定: H₁: μ > 170");
    auto result_greater = statcpp::t_test(heights.begin(), heights.end(), mu0,
                                          statcpp::alternative_hypothesis::greater);
    print_test_result(result_greater, "t統計量");
}

// ============================================================================
// 2. t_test_two_sample() と t_test_welch() - 2標本t検定
// ============================================================================

/**
 * @brief 2標本t検定の使用例
 *
 * 【目的】
 * 2つの独立したサンプルの平均が統計的に有意に異なるかを検定します。
 *
 * 【2つの方法】
 * - t_test_two_sample(): プール分散を使用（等分散を仮定）
 * - t_test_welch(): Welch法（等分散を仮定しない、より頑健）
 *
 * 【使用場面】
 * - A/Bテストの効果比較
 * - 治療群と対照群の比較
 * - 2つのグループの成績比較
 */
void example_two_sample_t_test() {
    print_section("2. 2標本t検定");

    // 例: 新しい教材を使ったグループと従来の教材を使ったグループの成績比較
    std::vector<double> group_new = {85, 90, 78, 92, 88, 95, 82, 91, 87, 89};
    std::vector<double> group_old = {75, 82, 70, 85, 78, 80, 72, 83, 76, 79};

    std::cout << "新教材グループ (n=" << group_new.size() << "):\n";
    print_data("  データ", group_new);
    std::cout << "  平均: " << statcpp::mean(group_new.begin(), group_new.end()) << "点\n\n";

    std::cout << "旧教材グループ (n=" << group_old.size() << "):\n";
    print_data("  データ", group_old);
    std::cout << "  平均: " << statcpp::mean(group_old.begin(), group_old.end()) << "点\n\n";

    // プール分散版
    print_subsection("等分散を仮定したt検定 (t_test_two_sample)");
    auto result_pooled = statcpp::t_test_two_sample(
        group_new.begin(), group_new.end(),
        group_old.begin(), group_old.end());
    print_test_result(result_pooled, "t統計量");

    // Welch版
    print_subsection("Welchのt検定 (t_test_welch) - 等分散を仮定しない");
    auto result_welch = statcpp::t_test_welch(
        group_new.begin(), group_new.end(),
        group_old.begin(), group_old.end());
    print_test_result(result_welch, "t統計量");

    std::cout << "\n【どちらを使うべきか】\n";
    std::cout << "- 等分散かどうか不明な場合: Welch法を推奨\n";
    std::cout << "- Welch法はより保守的で、Type I エラーを制御しやすい\n";
}

// ============================================================================
// 3. t_test_paired() - 対応のあるt検定
// ============================================================================

/**
 * @brief t_test_paired() の使用例
 *
 * 【目的】
 * 対になったデータ（同じ被験者の前後比較など）の差が0と
 * 有意に異なるかを検定します。
 *
 * 【使用場面】
 * - 治療前後の比較
 * - 同じ人への2つの条件の比較
 * - 双子研究
 */
void example_paired_t_test() {
    print_section("3. t_test_paired() - 対応のあるt検定");

    // 例: ダイエットプログラム前後の体重比較
    std::vector<double> weight_before = {70, 75, 68, 82, 78, 65, 72, 80, 74, 77};
    std::vector<double> weight_after =  {68, 72, 67, 78, 75, 64, 70, 76, 72, 74};

    std::cout << "ダイエットプログラムの効果検証\n\n";

    std::cout << "被験者データ:\n";
    std::cout << std::setw(10) << "被験者" << std::setw(12) << "前(kg)"
              << std::setw(12) << "後(kg)" << std::setw(12) << "差(kg)" << "\n";
    std::cout << std::string(46, '-') << "\n";

    double sum_diff = 0;
    for (std::size_t i = 0; i < weight_before.size(); ++i) {
        double diff = weight_before[i] - weight_after[i];
        sum_diff += diff;
        std::cout << std::setw(10) << (i + 1)
                  << std::setw(12) << weight_before[i]
                  << std::setw(12) << weight_after[i]
                  << std::setw(12) << diff << "\n";
    }

    double mean_diff = sum_diff / weight_before.size();
    std::cout << std::string(46, '-') << "\n";
    std::cout << "平均変化: " << mean_diff << " kg\n\n";

    print_subsection("対応のあるt検定: H₁: 差 ≠ 0");
    auto result = statcpp::t_test_paired(
        weight_before.begin(), weight_before.end(),
        weight_after.begin(), weight_after.end());
    print_test_result(result, "t統計量");

    std::cout << "\n→ ダイエットプログラムの効果は統計的に"
              << (result.p_value < 0.05 ? "有意" : "有意でない") << "\n";
}

// ============================================================================
// 4. z_test_proportion() - 比率の検定
// ============================================================================

/**
 * @brief 比率のz検定の使用例
 *
 * 【目的】
 * サンプルの比率が特定の値と有意に異なるかを検定します。
 *
 * 【使用場面】
 * - 不良品率が基準値を超えているか
 * - 選挙での支持率が50%を超えているか
 * - 合格率の検証
 */
void example_proportion_test() {
    print_section("4. z_test_proportion() - 比率のz検定");

    // 例: ある政党の支持率が50%を超えているか？
    print_subsection("1標本比率検定");
    std::size_t supporters = 54;   // 支持者数
    std::size_t sample_size = 100; // 調査対象者数
    double p0 = 0.5;               // 帰無仮説: 支持率 = 50%

    std::cout << "調査結果:\n";
    std::cout << "  調査対象者数: " << sample_size << "人\n";
    std::cout << "  支持者数: " << supporters << "人\n";
    std::cout << "  支持率: " << (100.0 * supporters / sample_size) << "%\n";
    std::cout << "  帰無仮説: 支持率 = " << (p0 * 100) << "%\n\n";

    auto result_prop = statcpp::z_test_proportion(supporters, sample_size, p0,
                                                  statcpp::alternative_hypothesis::greater);
    print_test_result(result_prop, "z統計量");

    // 2標本比率検定
    print_subsection("2標本比率検定");
    std::size_t success_A = 60, n_A = 100;  // 広告A
    std::size_t success_B = 45, n_B = 100;  // 広告B

    std::cout << "A/Bテスト結果:\n";
    std::cout << "  広告A: " << success_A << "/" << n_A << " = " << (100.0 * success_A / n_A) << "%\n";
    std::cout << "  広告B: " << success_B << "/" << n_B << " = " << (100.0 * success_B / n_B) << "%\n\n";

    auto result_two_prop = statcpp::z_test_proportion_two_sample(
        success_A, n_A, success_B, n_B);
    print_test_result(result_two_prop, "z統計量");
}

// ============================================================================
// 5. chisq_test_gof() - カイ二乗適合度検定
// ============================================================================

/**
 * @brief カイ二乗適合度検定の使用例
 *
 * 【目的】
 * 観測された度数分布が期待される分布と一致するかを検定します。
 *
 * 【使用場面】
 * - サイコロが公正かどうか
 * - 観測データが理論分布に従うか
 * - メンデルの遺伝法則の検証
 */
void example_chisq_gof() {
    print_section("5. chisq_test_gof() - カイ二乗適合度検定");

    // 例: サイコロの公正性検定
    std::vector<double> observed = {18, 20, 16, 22, 14, 30};  // 観測度数
    std::vector<double> expected = {20, 20, 20, 20, 20, 20};  // 期待度数（公正な場合）

    std::cout << "サイコロを120回振った結果:\n";
    std::cout << std::setw(8) << "出目" << std::setw(12) << "観測度数"
              << std::setw(12) << "期待度数" << "\n";
    std::cout << std::string(32, '-') << "\n";
    for (int i = 0; i < 6; ++i) {
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << observed[i]
                  << std::setw(12) << expected[i] << "\n";
    }
    std::cout << "\n";

    auto result = statcpp::chisq_test_gof(
        observed.begin(), observed.end(),
        expected.begin(), expected.end());
    print_test_result(result, "χ²統計量");

    std::cout << "\n→ サイコロは" << (result.p_value < 0.05 ? "公正でない可能性がある" : "公正と言える") << "\n";
}

// ============================================================================
// 6. chisq_test_independence() - カイ二乗独立性検定
// ============================================================================

/**
 * @brief カイ二乗独立性検定の使用例
 *
 * 【目的】
 * 分割表の行変数と列変数が独立かどうかを検定します。
 *
 * 【使用場面】
 * - 性別と製品嗜好の関係
 * - 治療法と回復状況の関係
 * - 学歴と所得の関係
 */
void example_chisq_independence() {
    print_section("6. chisq_test_independence() - カイ二乗独立性検定");

    // 例: 性別と製品選好の関係
    std::vector<std::vector<double>> contingency_table = {
        {30, 20, 10},  // 男性: A, B, C
        {20, 25, 35}   // 女性: A, B, C
    };

    std::cout << "性別と製品選好の分割表:\n";
    std::cout << std::setw(10) << "" << std::setw(12) << "製品A"
              << std::setw(12) << "製品B" << std::setw(12) << "製品C"
              << std::setw(12) << "合計" << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<std::string> rows = {"男性", "女性"};
    for (std::size_t i = 0; i < contingency_table.size(); ++i) {
        double row_total = 0;
        std::cout << std::setw(10) << rows[i];
        for (double val : contingency_table[i]) {
            std::cout << std::setw(12) << val;
            row_total += val;
        }
        std::cout << std::setw(12) << row_total << "\n";
    }

    std::cout << std::string(58, '-') << "\n";
    std::cout << std::setw(10) << "合計";
    double grand_total = 0;
    for (std::size_t j = 0; j < contingency_table[0].size(); ++j) {
        double col_total = 0;
        for (std::size_t i = 0; i < contingency_table.size(); ++i) {
            col_total += contingency_table[i][j];
        }
        std::cout << std::setw(12) << col_total;
        grand_total += col_total;
    }
    std::cout << std::setw(12) << grand_total << "\n\n";

    auto result = statcpp::chisq_test_independence(contingency_table);
    print_test_result(result, "χ²統計量");

    std::cout << "\n→ 性別と製品選好は"
              << (result.p_value < 0.05 ? "関連がある" : "独立である") << "\n";
}

// ============================================================================
// 7. f_test() - F検定（分散の比較）
// ============================================================================

/**
 * @brief F検定の使用例
 *
 * 【目的】
 * 2つのサンプルの分散が等しいかを検定します。
 * 等分散の検定は、t検定を適用する前の前提条件確認に使用されます。
 *
 * 【注意点】
 * - 正規分布を強く仮定する
 * - より頑健な方法としてLevene検定やBartlett検定がある
 */
void example_f_test() {
    print_section("7. f_test() - F検定（分散の比較）");

    // 例: 2つの製造ラインの品質のばらつき比較
    std::vector<double> line_A = {10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1, 10.4};
    std::vector<double> line_B = {10.0, 10.5, 9.5, 10.3, 9.7, 10.2, 9.8, 10.6};

    std::cout << "製造ラインの品質データ:\n";
    print_data("ラインA", line_A);
    std::cout << "  分散: " << statcpp::sample_variance(line_A.begin(), line_A.end()) << "\n";
    print_data("ラインB", line_B);
    std::cout << "  分散: " << statcpp::sample_variance(line_B.begin(), line_B.end()) << "\n\n";

    auto result = statcpp::f_test(
        line_A.begin(), line_A.end(),
        line_B.begin(), line_B.end());
    print_test_result(result, "F統計量");

    std::cout << "\n→ 2つのラインのばらつきは"
              << (result.p_value < 0.05 ? "異なる" : "同程度") << "\n";
}

// ============================================================================
// 8. 多重検定補正
// ============================================================================

/**
 * @brief 多重検定補正の使用例
 *
 * 【目的】
 * 複数の検定を同時に行うと、偶然に有意になる確率（Type I エラー率）が
 * 上昇します。多重検定補正はこの問題に対処します。
 *
 * 【補正方法】
 * - Bonferroni: 最も保守的、α/n で調整
 * - Holm: Bonferroniの段階的版、より検出力が高い
 * - Benjamini-Hochberg: FDR（偽発見率）を制御
 */
void example_multiple_testing_correction() {
    print_section("8. 多重検定補正");

    // 例: 複数の遺伝子発現の検定
    std::vector<double> p_values = {0.01, 0.04, 0.03, 0.005, 0.12, 0.02};

    std::cout << "複数の検定を行った場合（例: 遺伝子発現解析）\n\n";
    std::cout << "元のp値: ";
    for (double p : p_values) std::cout << p << " ";
    std::cout << "\n\n";

    auto bonf = statcpp::bonferroni_correction(p_values);
    auto holm = statcpp::holm_correction(p_values);
    auto bh = statcpp::benjamini_hochberg_correction(p_values);

    std::cout << std::setw(8) << "検定"
              << std::setw(12) << "元のp値"
              << std::setw(12) << "Bonferroni"
              << std::setw(12) << "Holm"
              << std::setw(12) << "BH (FDR)" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (std::size_t i = 0; i < p_values.size(); ++i) {
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << p_values[i]
                  << std::setw(12) << bonf[i]
                  << std::setw(12) << holm[i]
                  << std::setw(12) << bh[i] << "\n";
    }

    std::cout << R"(
【補正方法の選び方】
- Bonferroni: 最も保守的、偽陽性を厳しく制御したい場合
- Holm: Bonferroniより検出力が高い、推奨される方法
- Benjamini-Hochberg: 偽発見率(FDR)を制御、探索的研究に適する
)";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：パラメトリック検定");

    std::cout << R"(
┌────────────────────────────┬────────────────────────────────────────────┐
│ 関数                       │ 使用場面                                   │
├────────────────────────────┼────────────────────────────────────────────┤
│ t_test()                   │ 1標本の平均が特定の値と異なるか            │
│ t_test_two_sample()        │ 2群の平均比較（等分散を仮定）              │
│ t_test_welch()             │ 2群の平均比較（等分散を仮定しない）※推奨  │
│ t_test_paired()            │ 対応のあるデータの比較                     │
│ z_test()                   │ 1標本平均（分散既知）                      │
│ z_test_proportion()        │ 比率が特定の値と異なるか                   │
│ chisq_test_gof()           │ 度数分布が期待値と一致するか               │
│ chisq_test_independence()  │ 2変数が独立か（分割表）                    │
│ f_test()                   │ 2群の分散が等しいか                        │
└────────────────────────────┴────────────────────────────────────────────┘

【対立仮説の指定】
- alternative_hypothesis::two_sided  → 両側検定（デフォルト）
- alternative_hypothesis::less       → 片側検定（小さい方）
- alternative_hypothesis::greater    → 片側検定（大きい方）

【多重検定補正】
- bonferroni_correction()        → 最も保守的
- holm_correction()              → Bonferroniの改良版
- benjamini_hochberg_correction()→ FDR制御

【パラメトリック検定の前提条件】
- データが正規分布に従う（または十分なサンプルサイズ）
- 分散の等質性（一部の検定で必要）
- 前提が満たされない場合はノンパラメトリック検定を検討
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_t_test();
    example_two_sample_t_test();
    example_paired_t_test();
    example_proportion_test();
    example_chisq_gof();
    example_chisq_independence();
    example_f_test();
    example_multiple_testing_correction();

    // まとめを表示
    print_summary();

    return 0;
}
