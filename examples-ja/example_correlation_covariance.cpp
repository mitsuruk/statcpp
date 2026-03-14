/**
 * @file example_correlation_covariance.cpp
 * @brief statcpp::correlation_covariance.hpp のサンプルコード
 *
 * このファイルでは、correlation_covariance.hpp で提供される
 * 共分散・相関係数を計算する関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - population_covariance() : 母共分散
 * - sample_covariance()     : 標本共分散（不偏共分散）
 * - covariance()            : 共分散（sample_covariance のエイリアス）
 * - pearson_correlation()   : ピアソン相関係数
 * - spearman_correlation()  : スピアマン順位相関係数
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_correlation_covariance.cpp -o example_correlation_covariance
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// statcpp の相関・共分散ヘッダー
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

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
// 1. 共分散の概念説明
// ============================================================================

/**
 * @brief 共分散の概念説明
 *
 * 【目的】
 * 共分散(Covariance)は、2つの変数がどの程度一緒に変動するかを示す指標です。
 *
 * 【数式】
 * 母共分散: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = (1/n)Σ(xᵢ-x̄)(yᵢ-ȳ)
 * 標本共分散: s_xy = (1/(n-1))Σ(xᵢ-x̄)(yᵢ-ȳ)
 *
 * 【解釈】
 * - 共分散 > 0: Xが大きいときYも大きい傾向（正の関連）
 * - 共分散 < 0: Xが大きいときYは小さい傾向（負の関連）
 * - 共分散 ≈ 0: XとYに線形関連がない
 *
 * 【注意点】
 * - 共分散は単位に依存する（スケールに影響される）
 * - 値の大きさだけで関連の強さを判断できない
 */
void example_covariance_concept() {
    print_section("1. 共分散の概念");

    std::cout << R"(
【共分散とは】
2つの変数が「一緒に変動する度合い」を測る指標。

【視覚的イメージ】

正の共分散:
Y
│     ●  ●
│   ●  ●
│  ●
│●
└────────── X
Xが↑ならYも↑

負の共分散:
Y
│●
│  ●
│    ●
│      ●  ●
└────────── X
Xが↑ならYは↓

共分散≈0:
Y
│  ●    ●
│●  ●
│    ●
│  ●    ●
└────────── X
XとYに線形関係なし

)";
}

// ============================================================================
// 2. covariance() - 共分散の計算
// ============================================================================

/**
 * @brief covariance() の使用例
 *
 * 【使用場面】
 * - 2変数間の関連の方向（正/負）を調べる
 * - 分散分析や回帰分析の中間計算
 * - ポートフォリオのリスク計算
 */
void example_covariance() {
    print_section("2. covariance() - 共分散");

    // 勉強時間と試験の点数
    std::vector<double> study_hours = {2, 4, 6, 8, 10};
    std::vector<double> test_scores = {50, 60, 70, 80, 90};

    print_data("勉強時間 (h)", study_hours);
    print_data("試験の点数", test_scores);

    double pop_cov = statcpp::population_covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double samp_cov = statcpp::sample_covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double cov = statcpp::covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());

    std::cout << "\n母共分散 (population_covariance): " << pop_cov << "\n";
    std::cout << "標本共分散 (sample_covariance):   " << samp_cov << "\n";
    std::cout << "covariance():                     " << cov << " (= sample_covariance)\n";

    std::cout << "\n→ 正の共分散: 勉強時間が長いほど点数も高い傾向\n";

    // 負の共分散の例
    print_subsection("負の共分散の例");
    std::vector<double> absences = {0, 2, 4, 6, 8};    // 欠席日数
    std::vector<double> grades = {90, 80, 70, 60, 50};  // 成績

    print_data("欠席日数", absences);
    print_data("成績", grades);

    double neg_cov = statcpp::covariance(
        absences.begin(), absences.end(),
        grades.begin(), grades.end());
    std::cout << "共分散: " << neg_cov << "\n";
    std::cout << "→ 負の共分散: 欠席が多いほど成績が低い傾向\n";
}

// ============================================================================
// 3. pearson_correlation() - ピアソン相関係数
// ============================================================================

/**
 * @brief pearson_correlation() の使用例
 *
 * 【目的】
 * ピアソン相関係数は、2変数間の線形関連の強さと方向を -1〜+1 の範囲で表します。
 * 共分散を標準化したもので、スケールに依存しません。
 *
 * 【数式】
 * r = Cov(X,Y) / (σₓ × σᵧ) = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² × Σ(yᵢ-ȳ)²]
 *
 * 【解釈】
 * - r = +1 : 完全な正の線形関係
 * - r = -1 : 完全な負の線形関係
 * - r = 0  : 線形関係なし
 *
 * 【目安】
 * |r| < 0.2 : ほぼ無相関
 * 0.2 ≤ |r| < 0.4 : 弱い相関
 * 0.4 ≤ |r| < 0.6 : 中程度の相関
 * 0.6 ≤ |r| < 0.8 : 強い相関
 * |r| ≥ 0.8 : 非常に強い相関
 *
 * 【使用場面】
 * - 2変数間の関連の強さを数値化
 * - 予測モデルの評価
 * - 変数選択
 *
 * 【注意点】
 * - 線形関係のみを測定（非線形関係は検出できない）
 * - 外れ値の影響を受けやすい
 * - 相関≠因果（見せかけの相関に注意）
 */
void example_pearson_correlation() {
    print_section("3. pearson_correlation() - ピアソン相関係数");

    // 強い正の相関
    print_subsection("強い正の相関");
    std::vector<double> height = {160, 165, 170, 175, 180};
    std::vector<double> weight = {50, 55, 62, 68, 75};

    print_data("身長 (cm)", height);
    print_data("体重 (kg)", weight);

    double r_positive = statcpp::pearson_correlation(
        height.begin(), height.end(),
        weight.begin(), weight.end());
    std::cout << "ピアソン相関係数: " << r_positive << "\n";
    std::cout << "→ 強い正の相関: 身長が高いほど体重も重い\n";

    // 強い負の相関
    print_subsection("強い負の相関");
    std::vector<double> distance = {1, 2, 3, 4, 5};      // 通勤距離
    std::vector<double> satisfaction = {90, 75, 60, 45, 30};  // 通勤満足度

    print_data("通勤距離 (km)", distance);
    print_data("通勤満足度", satisfaction);

    double r_negative = statcpp::pearson_correlation(
        distance.begin(), distance.end(),
        satisfaction.begin(), satisfaction.end());
    std::cout << "ピアソン相関係数: " << r_negative << "\n";
    std::cout << "→ 強い負の相関: 距離が長いほど満足度が低い\n";

    // 相関がほぼない例
    print_subsection("相関がほぼない例");
    std::vector<double> shoe_size = {24, 26, 25, 27, 24};
    std::vector<double> iq = {100, 95, 110, 100, 105};

    print_data("靴のサイズ", shoe_size);
    print_data("IQ", iq);

    double r_none = statcpp::pearson_correlation(
        shoe_size.begin(), shoe_size.end(),
        iq.begin(), iq.end());
    std::cout << "ピアソン相関係数: " << r_none << "\n";
    std::cout << "→ ほぼ無相関: 靴のサイズとIQに関連なし\n";
}

// ============================================================================
// 4. spearman_correlation() - スピアマン順位相関係数
// ============================================================================

/**
 * @brief spearman_correlation() の使用例
 *
 * 【目的】
 * スピアマン順位相関係数は、2変数の順位（ランク）間のピアソン相関係数です。
 * 線形関係だけでなく、単調な関係も検出できます。
 *
 * 【数式】
 * ρ = Pearson(rank(X), rank(Y))
 *
 * 【特徴】
 * - 外れ値の影響を受けにくい
 * - 順序尺度のデータにも適用可能
 * - 単調関係（常に増加/減少）を検出
 *
 * 【使用場面】
 * - 順位データの相関
 * - 外れ値がある場合
 * - 非線形だが単調な関係の検出
 * - 順序尺度のデータ（満足度1〜5など）
 */
void example_spearman_correlation() {
    print_section("4. spearman_correlation() - スピアマン順位相関係数");

    // 線形に近い関係（ピアソンとスピアマンが近い）
    print_subsection("線形に近い関係");
    std::vector<double> study_hours = {2, 4, 6, 8, 10};
    std::vector<double> test_scores = {50, 60, 70, 80, 90};

    print_data("勉強時間", study_hours);
    print_data("テスト点数", test_scores);

    double r_pearson = statcpp::pearson_correlation(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double r_spearman = statcpp::spearman_correlation(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());

    std::cout << "ピアソン相関係数:     " << r_pearson << "\n";
    std::cout << "スピアマン順位相関係数: " << r_spearman << "\n";
    std::cout << "→ 線形関係では両者はほぼ同じ\n";

    // 外れ値がある場合
    print_subsection("外れ値がある場合");
    std::vector<double> x_outlier = {1, 2, 3, 4, 5, 100};  // 100は外れ値
    std::vector<double> y_outlier = {10, 20, 30, 40, 50, 55};

    print_data("X（外れ値あり）", x_outlier);
    print_data("Y", y_outlier);

    double r_pearson_outlier = statcpp::pearson_correlation(
        x_outlier.begin(), x_outlier.end(),
        y_outlier.begin(), y_outlier.end());
    double r_spearman_outlier = statcpp::spearman_correlation(
        x_outlier.begin(), x_outlier.end(),
        y_outlier.begin(), y_outlier.end());

    std::cout << "ピアソン相関係数:     " << r_pearson_outlier << "\n";
    std::cout << "スピアマン順位相関係数: " << r_spearman_outlier << "\n";
    std::cout << "→ スピアマンは外れ値の影響を受けにくい\n";

    // 単調だが非線形な関係
    print_subsection("単調だが非線形な関係");
    std::vector<double> x_exp = {1, 2, 3, 4, 5};
    std::vector<double> y_exp = {2, 4, 8, 16, 32};  // 指数関数的増加

    print_data("X", x_exp);
    print_data("Y（指数的増加）", y_exp);

    double r_pearson_exp = statcpp::pearson_correlation(
        x_exp.begin(), x_exp.end(),
        y_exp.begin(), y_exp.end());
    double r_spearman_exp = statcpp::spearman_correlation(
        x_exp.begin(), x_exp.end(),
        y_exp.begin(), y_exp.end());

    std::cout << "ピアソン相関係数:     " << r_pearson_exp << "\n";
    std::cout << "スピアマン順位相関係数: " << r_spearman_exp << "\n";
    std::cout << "→ スピアマンは単調関係を正しく検出\n";
}

// ============================================================================
// 5. ピアソン vs スピアマン の使い分け
// ============================================================================

/**
 * @brief ピアソンとスピアマンの使い分け
 */
void example_correlation_comparison() {
    print_section("5. ピアソン vs スピアマン の使い分け");

    std::cout << R"(
【使い分けの指針】

┌──────────────────────┬───────────────────────────────────────────┐
│ 状況                 │ 推奨される相関係数                        │
├──────────────────────┼───────────────────────────────────────────┤
│ 線形関係を調べたい   │ pearson_correlation()                     │
│ 正規分布に近いデータ │                                           │
│ 外れ値がない         │                                           │
├──────────────────────┼───────────────────────────────────────────┤
│ 単調関係を調べたい   │ spearman_correlation()                    │
│ 外れ値がある         │                                           │
│ 順序尺度のデータ     │                                           │
│ 分布が歪んでいる     │                                           │
└──────────────────────┴───────────────────────────────────────────┘

【両方を計算して比較する場合】
- 両者が近い → 線形関係
- スピアマン > ピアソン → 非線形だが単調な関係
- 両者が大きく異なる → 外れ値の影響か、複雑な関係
)";
}

// ============================================================================
// 6. ラムダ式（射影）を使った使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 *
 * 構造体のメンバー間の相関を計算します。
 */
void example_projection() {
    print_section("6. ラムダ式（射影）を使った使用例");

    struct Student {
        std::string name;
        double math_score;
        double english_score;
        double study_hours;
    };

    std::vector<Student> students = {
        {"Alice", 85, 90, 5},
        {"Bob", 70, 75, 3},
        {"Charlie", 95, 85, 7},
        {"Diana", 60, 70, 2},
        {"Eve", 80, 80, 4}
    };

    std::cout << "学生データ:\n";
    for (const auto& s : students) {
        std::cout << "  " << s.name << ": 数学=" << s.math_score
                  << ", 英語=" << s.english_score
                  << ", 勉強時間=" << s.study_hours << "h\n";
    }

    // 数学と英語の相関
    auto math_proj = [](const Student& s) { return s.math_score; };
    auto eng_proj = [](const Student& s) { return s.english_score; };
    auto hours_proj = [](const Student& s) { return s.study_hours; };

    double r_math_eng = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        math_proj, eng_proj);

    double r_math_hours = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        math_proj, hours_proj);

    double r_eng_hours = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        eng_proj, hours_proj);

    std::cout << "\n相関係数:\n";
    std::cout << "  数学 - 英語:    " << r_math_eng << "\n";
    std::cout << "  数学 - 勉強時間: " << r_math_hours << "\n";
    std::cout << "  英語 - 勉強時間: " << r_eng_hours << "\n";

    // 相関行列（概念表示）
    std::cout << "\n相関行列:\n";
    std::cout << "           数学   英語   勉強時間\n";
    std::cout << "  数学     1.0000  " << std::setw(7) << r_math_eng
              << "  " << std::setw(7) << r_math_hours << "\n";
    std::cout << "  英語     " << std::setw(7) << r_math_eng << "  1.0000  "
              << std::setw(7) << r_eng_hours << "\n";
    std::cout << "  勉強時間 " << std::setw(7) << r_math_hours
              << "  " << std::setw(7) << r_eng_hours << "  1.0000\n";
}

// ============================================================================
// 7. 相関係数の解釈の注意点
// ============================================================================

/**
 * @brief 相関係数の解釈の注意点
 */
void example_correlation_caveats() {
    print_section("7. 相関係数の解釈の注意点");

    std::cout << R"(
【注意点 1: 相関≠因果】

例: アイスクリームの売上と溺死事故の数には正の相関がある
→ アイスクリームが溺死を引き起こすわけではない
→ 両方とも「気温」という第三の変数に影響されている（疑似相関）

【注意点 2: 非線形関係の見落とし】

例: U字型の関係
  Y
  │
  │●        ●
  │  ●    ●
  │    ●●
  └────────── X

ピアソン相関係数 ≈ 0 になるが、明らかに関係はある

【注意点 3: 外れ値の影響】

たった1つの外れ値が相関係数を大きく変える可能性がある
→ 散布図を描いて確認することが重要

【注意点 4: サンプルサイズ】

小さなサンプルでは偶然による高い相関が出やすい
→ 統計的検定を行い、有意性を確認する

【注意点 5: 制限された範囲】

変数の範囲が制限されると、相関係数は過小評価される
例: 高身長の人だけを対象にすると、身長と体重の相関が弱く見える
)";

    // 実例：非線形関係
    print_subsection("実例: U字型の関係");
    std::vector<double> age = {20, 30, 40, 50, 60, 70};
    std::vector<double> happiness = {80, 70, 60, 65, 75, 85};  // U字型

    print_data("年齢", age);
    print_data("幸福度", happiness);

    double r = statcpp::pearson_correlation(
        age.begin(), age.end(),
        happiness.begin(), happiness.end());
    std::cout << "ピアソン相関係数: " << r << "\n";
    std::cout << "→ 相関係数は小さいが、U字型の関係が存在する\n";
    std::cout << "  （中年期に幸福度が最低で、若年と高齢で高い）\n";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：相関・共分散の関数");

    std::cout << R"(
┌─────────────────────────┬──────────────────────────────────────────┐
│ 関数                    │ 説明                                     │
├─────────────────────────┼──────────────────────────────────────────┤
│ covariance()            │ 標本共分散（不偏共分散）                 │
│ population_covariance() │ 母共分散                                 │
│ sample_covariance()     │ = covariance()                           │
│ pearson_correlation()   │ ピアソン相関係数（線形関係）             │
│ spearman_correlation()  │ スピアマン順位相関係数（単調関係）       │
└─────────────────────────┴──────────────────────────────────────────┘

【相関係数の解釈】
┌─────────────┬───────────────────────────────────────────────────────┐
│ |r| の範囲  │ 解釈                                                  │
├─────────────┼───────────────────────────────────────────────────────┤
│ 0.0 - 0.2   │ ほぼ無相関                                            │
│ 0.2 - 0.4   │ 弱い相関                                              │
│ 0.4 - 0.6   │ 中程度の相関                                          │
│ 0.6 - 0.8   │ 強い相関                                              │
│ 0.8 - 1.0   │ 非常に強い相関                                        │
└─────────────┴───────────────────────────────────────────────────────┘

【重要な注意事項】
- 相関は因果関係を示さない
- ピアソンは線形関係、スピアマンは単調関係を測定
- 外れ値がある場合はスピアマンを推奨
- 散布図を描いて視覚的に確認することが重要
- 統計的有意性の検定も行うべき
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_covariance_concept();
    example_covariance();
    example_pearson_correlation();
    example_spearman_correlation();
    example_correlation_comparison();
    example_projection();
    example_correlation_caveats();

    // まとめを表示
    print_summary();

    return 0;
}
