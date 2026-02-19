/**
 * @file example_shape_of_distribution.cpp
 * @brief statcpp::shape_of_distribution.hpp のサンプルコード
 *
 * このファイルでは、shape_of_distribution.hpp で提供される分布の形状を
 * 測定する関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - population_skewness() : 母歪度
 * - sample_skewness()     : 標本歪度（バイアス補正版）
 * - skewness()            : 歪度（sample_skewness のエイリアス）
 * - population_kurtosis() : 母尖度（超過尖度）
 * - sample_kurtosis()     : 標本尖度（バイアス補正版）
 * - kurtosis()            : 尖度（sample_kurtosis のエイリアス）
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_shape_of_distribution.cpp -o example_shape_of_distribution
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>

// statcpp の分布形状ヘッダー
#include "statcpp/shape_of_distribution.hpp"
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

// ============================================================================
// 1. skewness() - 歪度（概念の説明）
// ============================================================================

/**
 * @brief skewness() の概念説明
 *
 * 【目的】
 * 歪度(skewness)は、分布の非対称性を測る指標です。
 * 正規分布のような対称な分布では歪度は0になります。
 *
 * 【数式】
 * 母歪度: γ₁ = E[(X - μ)³] / σ³
 * 標本歪度: バイアス補正を適用した推定量
 *
 * 【解釈】
 * - skewness = 0 : 対称な分布
 * - skewness > 0 : 右に歪んだ分布（正の歪度、右裾が長い）
 * - skewness < 0 : 左に歪んだ分布（負の歪度、左裾が長い）
 *
 * 【目安】
 * |skewness| < 0.5  : ほぼ対称
 * 0.5 ≤ |skewness| < 1.0 : 中程度の歪み
 * |skewness| ≥ 1.0 : 強い歪み
 */
void example_skewness_concept() {
    print_section("1. 歪度（Skewness）の概念");

    std::cout << R"(
【歪度とは】
分布の非対称性を測る指標。平均を中心に左右対称かどうかを数値化。

【視覚的イメージ】

正の歪度（右に歪む）:
    │
  ▄▄█▄
 ▄█████▄▄▄▄▄▄___
 ← 平均は中央値より右 →
 所得分布、家屋価格など

負の歪度（左に歪む）:
           │
         ▄▄█▄▄
___▄▄▄▄▄██████▄
 ← 平均は中央値より左 →
 試験の点数（満点に近い場合）など

対称（歪度≈0）:
       │
     ▄▄█▄▄
   ▄███████▄
 平均 ≈ 中央値
 正規分布など
)";
}

// ============================================================================
// 2. 具体的なデータでの歪度計算
// ============================================================================

/**
 * @brief 歪度の計算例
 *
 * 【使用場面】
 * - データの分布形状の確認
 * - 正規性の簡易チェック
 * - データ変換の必要性判断（対数変換など）
 *
 * 【注意点】
 * - sample_skewness() は3つ以上のデータが必要
 * - 外れ値の影響を受けやすい
 */
void example_skewness_calculation() {
    print_section("2. 歪度の計算例");

    // 対称に近いデータ
    std::vector<double> symmetric = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // 正の歪度を持つデータ（右に歪む）
    std::vector<double> right_skewed = {1, 2, 2, 3, 3, 3, 4, 4, 5, 10};

    // 負の歪度を持つデータ（左に歪む）
    std::vector<double> left_skewed = {1, 6, 7, 7, 7, 8, 8, 9, 9, 10};

    print_subsection("対称に近いデータ");
    print_data("データ", symmetric);
    double skew_sym = statcpp::skewness(symmetric.begin(), symmetric.end());
    std::cout << "歪度: " << skew_sym << " (≈0: 対称)\n";
    std::cout << "平均: " << statcpp::mean(symmetric.begin(), symmetric.end()) << "\n";

    print_subsection("正の歪度を持つデータ（右裾が長い）");
    print_data("データ", right_skewed);
    double skew_right = statcpp::skewness(right_skewed.begin(), right_skewed.end());
    std::cout << "歪度: " << skew_right << " (>0: 右に歪む)\n";
    std::cout << "平均: " << statcpp::mean(right_skewed.begin(), right_skewed.end()) << "\n";
    std::cout << "→ 外れ値10が平均を右に引っ張る\n";

    print_subsection("負の歪度を持つデータ（左裾が長い）");
    print_data("データ", left_skewed);
    double skew_left = statcpp::skewness(left_skewed.begin(), left_skewed.end());
    std::cout << "歪度: " << skew_left << " (<0: 左に歪む)\n";
    std::cout << "平均: " << statcpp::mean(left_skewed.begin(), left_skewed.end()) << "\n";
    std::cout << "→ 外れ値1が平均を左に引っ張る\n";
}

// ============================================================================
// 3. population_skewness() vs sample_skewness()
// ============================================================================

/**
 * @brief 母歪度と標本歪度の違い
 *
 * 【目的】
 * population_skewness() と sample_skewness() の違いを説明します。
 * 標本から母集団を推定する場合は sample_skewness() を使用します。
 */
void example_population_vs_sample_skewness() {
    print_section("3. 母歪度 vs 標本歪度");

    std::vector<double> data = {2, 3, 5, 7, 8, 9, 10, 12, 15, 20};
    print_data("データ (n=10)", data);

    double pop_skew = statcpp::population_skewness(data.begin(), data.end());
    double samp_skew = statcpp::sample_skewness(data.begin(), data.end());

    std::cout << "\n母歪度 (population_skewness): " << pop_skew << "\n";
    std::cout << "標本歪度 (sample_skewness):   " << samp_skew << "\n";
    std::cout << "skewness():                   " << statcpp::skewness(data.begin(), data.end())
              << " (= sample_skewness)\n";

    std::cout << R"(
【使い分け】
- population_skewness(): データが母集団全体の場合
  例: 会社の全社員の勤務年数分布

- sample_skewness(): データが標本の場合（母集団からの抽出）
  例: アンケート調査の回答
  バイアス補正が適用され、より正確な推定が可能
)";
}

// ============================================================================
// 4. kurtosis() - 尖度（概念の説明）
// ============================================================================

/**
 * @brief kurtosis() の概念説明
 *
 * 【目的】
 * 尖度(kurtosis)は、分布の「裾の重さ」や「尖り具合」を測る指標です。
 * 正規分布と比較してどの程度裾が重いか（外れ値が出やすいか）を示します。
 *
 * 【数式】
 * 母尖度（超過尖度）: γ₂ = E[(X - μ)⁴] / σ⁴ - 3
 * -3 することで正規分布の尖度が0になる（超過尖度）
 *
 * 【解釈】
 * - kurtosis = 0  : 正規分布と同程度（mesokurtic）
 * - kurtosis > 0  : 正規分布より裾が重い（leptokurtic）
 *                   外れ値が出やすい、分布がより尖っている
 * - kurtosis < 0  : 正規分布より裾が軽い（platykurtic）
 *                   外れ値が出にくい、分布がより平坦
 */
void example_kurtosis_concept() {
    print_section("4. 尖度（Kurtosis）の概念");

    std::cout << R"(
【尖度とは】
分布の裾の重さ（外れ値の出やすさ）を測る指標。
正規分布を基準(0)として比較。

【超過尖度（Excess Kurtosis）】
本ライブラリは超過尖度を使用（正規分布で0になるよう調整）

【視覚的イメージ】

正の尖度（leptokurtic）:
      ▲
     ███
     ███   ← 中央が尖り、裾が重い
   ▄█████▄
__▄▄██████▄▄___▄__
   外れ値が出やすい
   例: t分布、金融リターン

負の尖度（platykurtic）:
   ▄▄▄▄▄▄▄▄▄
  ██████████
 ████████████  ← 中央が平坦、裾が軽い
████████████████
   外れ値が出にくい
   例: 一様分布

正規分布（mesokurtic, 尖度≈0）:
     ▄▄
   ▄████▄
 ▄████████▄
████████████████
)";
}

// ============================================================================
// 5. 具体的なデータでの尖度計算
// ============================================================================

/**
 * @brief 尖度の計算例
 *
 * 【使用場面】
 * - 分布の裾の重さの評価
 * - リスク評価（外れ値の発生頻度）
 * - 正規性の簡易チェック
 *
 * 【注意点】
 * - sample_kurtosis() は4つ以上のデータが必要
 * - 外れ値の影響を非常に受けやすい（4乗を使うため）
 */
void example_kurtosis_calculation() {
    print_section("5. 尖度の計算例");

    // 一様分布に近いデータ（負の尖度）
    std::vector<double> uniform_like = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 正規分布に近いデータ
    std::vector<double> normal_like = {2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8};

    // 外れ値を含むデータ（正の尖度）
    std::vector<double> heavy_tails = {1, 4, 5, 5, 5, 5, 5, 5, 6, 10};

    print_subsection("一様分布に近いデータ");
    print_data("データ", uniform_like);
    double kurt_uniform = statcpp::kurtosis(uniform_like.begin(), uniform_like.end());
    std::cout << "尖度: " << kurt_uniform << " (<0: 裾が軽い)\n";

    print_subsection("正規分布に近いデータ");
    print_data("データ", normal_like);
    double kurt_normal = statcpp::kurtosis(normal_like.begin(), normal_like.end());
    std::cout << "尖度: " << kurt_normal << " (≈0: 正規分布と同程度)\n";

    print_subsection("外れ値を含むデータ");
    print_data("データ", heavy_tails);
    double kurt_heavy = statcpp::kurtosis(heavy_tails.begin(), heavy_tails.end());
    std::cout << "尖度: " << kurt_heavy << " (>0: 裾が重い)\n";
    std::cout << "→ 両端の1と10が尖度を大きくする\n";
}

// ============================================================================
// 6. population_kurtosis() vs sample_kurtosis()
// ============================================================================

/**
 * @brief 母尖度と標本尖度の違い
 */
void example_population_vs_sample_kurtosis() {
    print_section("6. 母尖度 vs 標本尖度");

    std::vector<double> data = {2, 3, 5, 7, 8, 9, 10, 12, 15, 20};
    print_data("データ (n=10)", data);

    double pop_kurt = statcpp::population_kurtosis(data.begin(), data.end());
    double samp_kurt = statcpp::sample_kurtosis(data.begin(), data.end());

    std::cout << "\n母尖度 (population_kurtosis): " << pop_kurt << "\n";
    std::cout << "標本尖度 (sample_kurtosis):   " << samp_kurt << "\n";
    std::cout << "kurtosis():                   " << statcpp::kurtosis(data.begin(), data.end())
              << " (= sample_kurtosis)\n";

    std::cout << R"(
【使い分け】
- population_kurtosis(): データが母集団全体の場合

- sample_kurtosis(): データが標本の場合（母集団からの抽出）
  バイアス補正が適用される
)";
}

// ============================================================================
// 7. 実践例：所得分布の分析
// ============================================================================

/**
 * @brief 実践的な使用例：所得分布
 *
 * 所得分布は典型的に右に歪んだ分布（正の歪度）を示します。
 */
void example_income_distribution() {
    print_section("7. 実践例：所得分布の分析");

    // 仮想的な所得データ（万円）
    std::vector<double> income = {
        300, 320, 350, 380, 400, 420, 450, 480, 500,
        520, 550, 600, 650, 700, 800, 1000, 1500, 2000, 5000
    };

    std::cout << "所得データ（万円）: ";
    for (size_t i = 0; i < income.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << income[i];
    }
    std::cout << "\n\n";

    double mean_val = statcpp::mean(income.begin(), income.end());
    std::vector<double> sorted_income = income;
    std::sort(sorted_income.begin(), sorted_income.end());
    double median_val = statcpp::median(sorted_income.begin(), sorted_income.end());
    double skew = statcpp::skewness(income.begin(), income.end());
    double kurt = statcpp::kurtosis(income.begin(), income.end());

    std::cout << "平均: " << mean_val << " 万円\n";
    std::cout << "中央値: " << median_val << " 万円\n";
    std::cout << "歪度: " << skew << "\n";
    std::cout << "尖度: " << kurt << "\n";

    std::cout << "\n【分析結果】\n";
    std::cout << "- 平均 > 中央値 → 高所得者により平均が引き上げられている\n";
    std::cout << "- 歪度 > 0 → 右に歪んだ分布（少数の高所得者）\n";
    std::cout << "- 尖度 > 0 → 裾が重い（極端な高所得者が存在）\n";
    std::cout << "\n→ 所得の「代表値」には中央値の方が適切\n";
}

// ============================================================================
// 8. 正規性の簡易チェック
// ============================================================================

/**
 * @brief 正規性の簡易チェック
 *
 * 歪度と尖度を使って、データが正規分布に従っているかを
 * 簡易的にチェックできます。
 */
void example_normality_check() {
    print_section("8. 正規性の簡易チェック");

    // 乱数生成器を使って正規分布に近いデータを生成
    std::mt19937 gen(42);  // 再現性のためシードを固定
    std::normal_distribution<> normal_dist(100, 15);

    std::vector<double> normal_data;
    normal_data.reserve(100);
    for (int i = 0; i < 100; ++i) {
        normal_data.push_back(normal_dist(gen));
    }

    double skew = statcpp::skewness(normal_data.begin(), normal_data.end());
    double kurt = statcpp::kurtosis(normal_data.begin(), normal_data.end());

    std::cout << "正規分布から生成したデータ（n=100, μ=100, σ=15）\n\n";
    std::cout << "歪度: " << skew << "\n";
    std::cout << "尖度: " << kurt << "\n";

    std::cout << R"(
【正規性の目安】
歪度：|skewness| < 2 であれば許容範囲
尖度：|kurtosis| < 7 であれば許容範囲
（Kline, 2015 の基準による）

より厳密なチェックには Shapiro-Wilk 検定などを使用
)";

    // 判定
    std::cout << "\n【判定】\n";
    bool skew_ok = std::abs(skew) < 2.0;
    bool kurt_ok = std::abs(kurt) < 7.0;
    std::cout << "歪度の基準: " << (skew_ok ? "OK" : "NG") << "\n";
    std::cout << "尖度の基準: " << (kurt_ok ? "OK" : "NG") << "\n";
    std::cout << "総合判定: " << ((skew_ok && kurt_ok) ? "正規分布に近い" : "正規分布から乖離") << "\n";
}

// ============================================================================
// 9. ラムダ式（射影）を使った使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 */
void example_projection() {
    print_section("9. ラムダ式（射影）を使った使用例");

    // 構造体の例
    struct ExamResult {
        std::string name;
        double score;
    };

    std::vector<ExamResult> results = {
        {"Alice", 85}, {"Bob", 78}, {"Charlie", 92}, {"Diana", 65},
        {"Eve", 88}, {"Frank", 72}, {"Grace", 95}, {"Henry", 80},
        {"Ivy", 58}, {"Jack", 90}, {"Kate", 82}, {"Leo", 75}
    };

    std::cout << "試験結果:\n";
    for (const auto& r : results) {
        std::cout << "  " << r.name << ": " << r.score << "点\n";
    }

    auto score_proj = [](const ExamResult& r) { return r.score; };

    double mean_score = statcpp::mean(results.begin(), results.end(), score_proj);
    double skew = statcpp::skewness(results.begin(), results.end(), score_proj);
    double kurt = statcpp::kurtosis(results.begin(), results.end(), score_proj);

    std::cout << "\n点数の統計:\n";
    std::cout << "  平均: " << mean_score << "点\n";
    std::cout << "  歪度: " << skew << "\n";
    std::cout << "  尖度: " << kurt << "\n";

    // 解釈
    std::cout << "\n【分布の解釈】\n";
    if (skew < -0.5) {
        std::cout << "- 負の歪度: 高得点に偏った分布（試験が易しめ）\n";
    } else if (skew > 0.5) {
        std::cout << "- 正の歪度: 低得点に偏った分布（試験が難しめ）\n";
    } else {
        std::cout << "- 歪度≈0: ほぼ対称な分布（適切な難易度）\n";
    }
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：分布形状の指標");

    std::cout << R"(
┌─────────────────────────┬──────────────────────────────────────────┐
│ 関数                    │ 説明                                     │
├─────────────────────────┼──────────────────────────────────────────┤
│ skewness()              │ 標本歪度（バイアス補正版）               │
│ population_skewness()   │ 母歪度                                   │
│ sample_skewness()       │ = skewness()                             │
│ kurtosis()              │ 標本尖度（超過尖度、バイアス補正版）     │
│ population_kurtosis()   │ 母尖度（超過尖度）                       │
│ sample_kurtosis()       │ = kurtosis()                             │
└─────────────────────────┴──────────────────────────────────────────┘

【歪度の解釈】
┌──────────────────┬────────────────────────────────────────────────┐
│ 値               │ 意味                                           │
├──────────────────┼────────────────────────────────────────────────┤
│ ≈ 0              │ 対称な分布                                     │
│ > 0              │ 右に歪む（右裾が長い）例：所得分布             │
│ < 0              │ 左に歪む（左裾が長い）例：高得点が多い試験     │
└──────────────────┴────────────────────────────────────────────────┘

【尖度の解釈】（超過尖度）
┌──────────────────┬────────────────────────────────────────────────┐
│ 値               │ 意味                                           │
├──────────────────┼────────────────────────────────────────────────┤
│ ≈ 0              │ 正規分布と同程度                               │
│ > 0              │ 裾が重い（外れ値が出やすい）                   │
│ < 0              │ 裾が軽い（外れ値が出にくい）                   │
└──────────────────┴────────────────────────────────────────────────┘

【注意事項】
- sample_skewness() は n≥3、sample_kurtosis() は n≥4 が必要
- 外れ値の影響を受けやすい（特に尖度は4乗を使うため）
- 正規性の厳密なチェックには統計検定を使用
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_skewness_concept();
    example_skewness_calculation();
    example_population_vs_sample_skewness();
    example_kurtosis_concept();
    example_kurtosis_calculation();
    example_population_vs_sample_kurtosis();
    example_income_distribution();
    example_normality_check();
    example_projection();

    // まとめを表示
    print_summary();

    return 0;
}
