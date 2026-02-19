/**
 * @file example_estimation.cpp
 * @brief statcpp::estimation.hpp のサンプルコード
 *
 * このファイルでは、estimation.hpp で提供される
 * 統計的推定（信頼区間、誤差範囲）の関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - standard_error()              : 標準誤差
 * - ci_mean()                     : 平均の信頼区間
 * - margin_of_error_mean()        : 平均の誤差範囲
 * - ci_proportion()               : 比率の信頼区間
 * - margin_of_error_proportion()  : 比率の誤差範囲
 * - ci_mean_diff_*()              : 2標本平均差の信頼区間
 * - ci_variance()                 : 分散の信頼区間
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_estimation.cpp -o example_estimation
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include "statcpp/estimation.hpp"
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
// 1. 信頼区間の概念説明
// ============================================================================

/**
 * @brief 信頼区間の概念説明
 *
 * 【目的】
 * 信頼区間（Confidence Interval, CI）は、母集団パラメータの推定値の
 * 不確実性を表現する区間です。
 *
 * 【解釈】
 * 95%信頼区間 [L, U] の意味：
 * - 「母数が [L, U] の中にある確率が95%」ではない（よくある誤解）
 * - 正しくは：「同じ方法で100回標本抽出して区間を作ると、
 *   約95回は母数を含む区間が得られる」
 *
 * 【信頼水準】
 * - 90% CI: より狭い区間、精度は低い
 * - 95% CI: 標準的、バランスが良い
 * - 99% CI: より広い区間、より保守的
 */
void example_ci_concept() {
    print_section("1. 信頼区間の概念");

    std::cout << R"(
【信頼区間とは】
母集団パラメータ（平均、比率など）の真の値が含まれると
期待される範囲を示す指標

【視覚的イメージ】

真の母平均 μ
      ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━
    [----]  ← 標本1の95%CI（μを含む）
       [----]  ← 標本2の95%CI（μを含む）
 [----]        ← 標本3の95%CI（μを含まない！）
     [----]    ← 標本4の95%CI（μを含む）
        [----] ← 標本5の95%CI（μを含む）

100回抽出すると、約95回は母平均μを含む区間が得られる
（5回程度は含まない区間も出る）

【信頼水準の選択】
┌──────────┬─────────┬───────────────────────────┐
│ 信頼水準   │ 区間の幅 │ 使用場面                   │
├──────────┼─────────┼───────────────────────────┤
│ 90%        │ 狭い     │ 探索的分析                 │
│ 95%        │ 標準     │ 一般的な研究（最も一般的） │
│ 99%        │ 広い     │ 医薬品、安全性評価         │
└──────────┴─────────┴───────────────────────────┘
)";
}

// ============================================================================
// 2. 標準誤差 (Standard Error)
// ============================================================================

/**
 * @brief 標準誤差の使用例
 *
 * 【概念】
 * 標準誤差（Standard Error, SE）は、統計量（平均など）の
 * 標本抽出による変動の大きさを表します。
 *
 * 【数式】
 * SE = s / √n
 * s: 標本標準偏差、n: サンプルサイズ
 *
 * 【解釈】
 * - SE が小さい → 推定が精密
 * - SE が大きい → 推定が不精密
 * - n を増やすと SE は小さくなる（√n に比例して減少）
 *
 * 【使用場面】
 * - 信頼区間の計算
 * - 仮説検定の検定統計量の計算
 * - 推定値の精度評価
 */
void example_standard_error() {
    print_section("2. 標準誤差 (Standard Error)");

    std::cout << R"(
【概念】
統計量（平均など）の標本抽出による変動の大きさ

【実例: 身長測定】
20人の成人男性の身長（cm）を測定
)";

    std::vector<double> heights = {
        172, 168, 175, 171, 169, 173, 170, 174,
        168, 172, 171, 169, 173, 170, 172, 171,
        169, 174, 170, 172
    };

    print_data("身長データ", heights);

    double mean_height = statcpp::mean(heights.begin(), heights.end());
    double sd = statcpp::sample_stddev(heights.begin(), heights.end());
    double se = statcpp::standard_error(heights.begin(), heights.end());

    std::cout << "\n標本平均: " << mean_height << " cm\n";
    std::cout << "標本標準偏差: " << sd << " cm\n";
    std::cout << "標準誤差: " << se << " cm\n";

    std::cout << "\n→ 標準誤差が小さいほど、母平均の推定が精密\n";

    print_subsection("サンプルサイズの影響");
    std::cout << R"(
標準誤差 SE = s / √n なので、n を増やすと SE は減少：

n=20  → SE = )";
    std::cout << se << " cm\n";
    std::cout << "n=80  → SE = " << sd / std::sqrt(80) << " cm (1/2に減少)\n";
    std::cout << "n=320 → SE = " << sd / std::sqrt(320) << " cm (1/4に減少)\n";

    std::cout << "\n→ nを4倍にすると、SEは半分になる\n";
}

// ============================================================================
// 3. 平均の信頼区間
// ============================================================================

/**
 * @brief 平均の信頼区間の使用例
 *
 * 【概念】
 * 母平均 μ の推定値の範囲を示す
 *
 * 【数式】
 * CI = x̄ ± t(α/2, n-1) × SE
 * x̄: 標本平均、t: t分布の分位点、SE: 標準誤差
 *
 * 【使用場面】
 * - 平均身長、平均年収、平均スコアの推定
 * - 実験結果の効果量の推定
 * - A/Bテストの平均差の評価
 */
void example_ci_mean() {
    print_section("3. 平均の信頼区間");

    std::cout << R"(
【実例: 製品の寿命テスト】
新しいバッテリーの寿命（時間）を8個テスト
)";

    std::vector<double> lifetimes = {23.1, 25.3, 22.8, 24.5, 26.1, 23.7, 24.9, 25.5};
    print_data("寿命データ", lifetimes);

    double mean_life = statcpp::mean(lifetimes.begin(), lifetimes.end());
    double se_life = statcpp::standard_error(lifetimes.begin(), lifetimes.end());

    std::cout << "\n標本平均: " << mean_life << " 時間\n";
    std::cout << "標準誤差: " << se_life << " 時間\n";

    // 95% CI
    print_subsection("95% 信頼区間");
    auto ci_95 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.95);
    std::cout << "95% CI: [" << ci_95.lower << ", " << ci_95.upper << "]\n";
    std::cout << "点推定値: " << ci_95.point_estimate << "\n";
    std::cout << "\n解釈: 母平均の寿命は95%の確率で\n";
    std::cout << "      " << ci_95.lower << "〜" << ci_95.upper << " 時間の範囲にある\n";

    // 99% CI
    print_subsection("99% 信頼区間（より保守的）");
    auto ci_99 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.99);
    std::cout << "99% CI: [" << ci_99.lower << ", " << ci_99.upper << "]\n";
    std::cout << "\n→ 99% CI は 95% CI より広い（より保守的）\n";

    // 90% CI
    print_subsection("90% 信頼区間（より積極的）");
    auto ci_90 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.90);
    std::cout << "90% CI: [" << ci_90.lower << ", " << ci_90.upper << "]\n";
    std::cout << "\n→ 90% CI は 95% CI より狭い（探索的分析で使用）\n";

    print_subsection("誤差範囲 (Margin of Error)");
    double moe_95 = statcpp::margin_of_error_mean(lifetimes.begin(), lifetimes.end(), 0.95);
    std::cout << "誤差範囲（95%）: ±" << moe_95 << " 時間\n";
    std::cout << "→ 平均 = " << mean_life << " ± " << moe_95 << " 時間\n";
}

// ============================================================================
// 4. 比率の信頼区間
// ============================================================================

/**
 * @brief 比率の信頼区間の使用例
 *
 * 【概念】
 * 母比率 p の推定値の範囲を示す
 *
 * 【数式】
 * CI = p̂ ± z(α/2) × √[p̂(1-p̂)/n]
 * p̂: 標本比率、z: 標準正規分布の分位点
 *
 * 【使用場面】
 * - 支持率調査
 * - 不良品率の推定
 * - コンバージョン率の推定
 * - 有病率の推定
 */
void example_ci_proportion() {
    print_section("4. 比率の信頼区間");

    std::cout << R"(
【実例: 製品の不良品率調査】
1000個の製品を検査したところ、65個が不良品だった
→ 全体の不良品率は？
)";

    std::size_t defects = 65;
    std::size_t n = 1000;
    double p_hat = static_cast<double>(defects) / n;

    std::cout << "\n標本サイズ: " << n << " 個\n";
    std::cout << "不良品数: " << defects << " 個\n";
    std::cout << "標本比率: " << p_hat << " (" << p_hat * 100 << "%)\n";

    // 95% CI
    print_subsection("95% 信頼区間");
    auto ci_prop = statcpp::ci_proportion(defects, n, 0.95);
    std::cout << "95% CI: [" << ci_prop.lower << ", " << ci_prop.upper << "]\n";
    std::cout << "       ([" << ci_prop.lower * 100 << "%, "
              << ci_prop.upper * 100 << "%])\n";
    std::cout << "\n解釈: 母不良品率は95%の確率で\n";
    std::cout << "      " << ci_prop.lower * 100 << "〜"
              << ci_prop.upper * 100 << "% の範囲\n";

    // 誤差範囲
    print_subsection("誤差範囲 (Margin of Error)");
    double moe_prop = statcpp::margin_of_error_proportion(defects, n, 0.95);
    std::cout << "誤差範囲（95%）: ±" << moe_prop << " (±"
              << moe_prop * 100 << "%)\n";
    std::cout << "→ 不良品率 = " << p_hat * 100 << "% ± "
              << moe_prop * 100 << "%\n";

    // 最悪ケースの誤差範囲
    print_subsection("最悪ケースの誤差範囲");
    double moe_worst = statcpp::margin_of_error_proportion_worst_case(n, 0.95);
    std::cout << R"(
最悪ケース（p=0.5のとき）の誤差範囲: ±)" << moe_worst * 100 << "%\n";
    std::cout << "\n→ 比率がわからない場合、この値をサンプルサイズ設計に使用\n";

    print_subsection("実用例: 世論調査");
    std::cout << R"(
支持率調査で n=1000 人にアンケート
「はい」と答えた人: 520人（52%）

95% CI はいくつか？
)";
    auto poll_ci = statcpp::ci_proportion(520, 1000, 0.95);
    std::cout << "95% CI: [" << poll_ci.lower * 100 << "%, "
              << poll_ci.upper * 100 << "%]\n";
    std::cout << "\n→ 真の支持率は " << poll_ci.lower * 100 << "〜"
              << poll_ci.upper * 100 << "% の範囲\n";
    std::cout << "   （「過半数を超えている」と言えるか？）\n";
}

// ============================================================================
// 5. 2標本平均差の信頼区間
// ============================================================================

/**
 * @brief 2標本平均差の信頼区間の使用例
 *
 * 【概念】
 * 2つのグループの母平均の差 μ1 - μ2 の推定値の範囲
 *
 * 【種類】
 * - Pooled (等分散仮定): 両群の分散が等しいと仮定
 * - Welch (不等分散): 分散が異なる可能性がある
 * - Paired (対応あり): 同一個体の前後比較など
 *
 * 【使用場面】
 * - A/Bテスト（新旧バージョンの比較）
 * - 臨床試験（治療群 vs 対照群）
 * - 教育介入の効果測定
 */
void example_ci_mean_diff() {
    print_section("5. 2標本平均差の信頼区間");

    std::cout << R"(
【実例: A/Bテスト】
Webサイトの新デザイン（B）と旧デザイン（A）で
ページ滞在時間（秒）を比較
)";

    std::vector<double> design_a = {45, 52, 48, 50, 47, 49, 51, 46, 48, 50};  // 旧デザイン
    std::vector<double> design_b = {52, 58, 55, 57, 53, 56, 54, 55, 57, 56};  // 新デザイン

    print_data("旧デザイン（A）", design_a);
    print_data("新デザイン（B）", design_b);

    double mean_a = statcpp::mean(design_a.begin(), design_a.end());
    double mean_b = statcpp::mean(design_b.begin(), design_b.end());
    double observed_diff = mean_b - mean_a;

    std::cout << "\n平均滞在時間:\n";
    std::cout << "  旧デザイン（A）: " << mean_a << " 秒\n";
    std::cout << "  新デザイン（B）: " << mean_b << " 秒\n";
    std::cout << "  差（B - A）: " << observed_diff << " 秒\n";

    print_subsection("等分散を仮定（Pooled t-test）");
    auto ci_pooled = statcpp::ci_mean_diff_pooled(
        design_b.begin(), design_b.end(),
        design_a.begin(), design_a.end(),
        0.95
    );
    std::cout << "95% CI: [" << ci_pooled.lower << ", " << ci_pooled.upper << "]\n";
    std::cout << "点推定値: " << ci_pooled.point_estimate << "\n";
    std::cout << "\n解釈: 新デザインは旧デザインより\n";
    std::cout << "      " << ci_pooled.lower << "〜" << ci_pooled.upper
              << " 秒長く滞在させる（95% CI）\n";

    if (ci_pooled.lower > 0) {
        std::cout << "\n→ CI がすべて正なので、新デザインの効果は統計的に有意\n";
    }

    print_subsection("不等分散を仮定（Welch's t-test）");
    auto ci_welch = statcpp::ci_mean_diff_welch(
        design_b.begin(), design_b.end(),
        design_a.begin(), design_a.end(),
        0.95
    );
    std::cout << "95% CI: [" << ci_welch.lower << ", " << ci_welch.upper << "]\n";
    std::cout << "\n→ 分散が等しいか不明な場合、Welch を使う方が安全\n";

    print_subsection("実用例: 対応のあるデータ");
    std::cout << R"(
同じ10人の患者に対して、治療前後で血圧を測定
)";
    std::vector<double> before = {140, 135, 142, 138, 145, 137, 141, 139, 143, 136};
    std::vector<double> after =  {132, 130, 135, 133, 138, 131, 134, 132, 136, 130};

    print_data("治療前", before);
    print_data("治療後", after);

    // 対応のあるデータでは、差分を計算してから ci_mean() を使用
    std::vector<double> differences;
    for (size_t i = 0; i < before.size(); ++i) {
        differences.push_back(before[i] - after[i]);
    }

    print_data("差分（前 - 後）", differences);

    auto ci_paired = statcpp::ci_mean(differences.begin(), differences.end(), 0.95);

    std::cout << "\n治療効果（前 - 後）の 95% CI: ["
              << ci_paired.lower << ", " << ci_paired.upper << "]\n";
    std::cout << "点推定値: " << ci_paired.point_estimate << " mmHg の低下\n";

    if (ci_paired.lower > 0) {
        std::cout << "\n→ 治療により血圧が有意に低下した\n";
    }
}

// ============================================================================
// 6. 分散の信頼区間
// ============================================================================

/**
 * @brief 分散の信頼区間の使用例
 *
 * 【概念】
 * 母分散 σ² の推定値の範囲を示す
 *
 * 【数式】
 * カイ二乗分布を使用
 * CI = [(n-1)s² / χ²(α/2), (n-1)s² / χ²(1-α/2)]
 *
 * 【使用場面】
 * - 製造プロセスのばらつき評価
 * - 品質管理
 * - リスク評価
 * - 測定器の精度評価
 */
void example_ci_variance() {
    print_section("6. 分散の信頼区間");

    std::cout << R"(
【実例: 製造プロセスのばらつき評価】
製品の重量（g）のばらつきを評価
)";

    std::vector<double> weights = {
        100.2, 99.8, 100.1, 99.9, 100.3,
        99.7, 100.0, 100.2, 99.8, 100.1
    };

    print_data("重量データ", weights);

    double mean_weight = statcpp::mean(weights.begin(), weights.end());
    double var_weight = statcpp::var(weights.begin(), weights.end(), 1);
    double sd_weight = std::sqrt(var_weight);

    std::cout << "\n標本平均: " << mean_weight << " g\n";
    std::cout << "標本分散: " << var_weight << " g²\n";
    std::cout << "標本標準偏差: " << sd_weight << " g\n";

    print_subsection("分散の95% 信頼区間");
    auto ci_var = statcpp::ci_variance(weights.begin(), weights.end(), 0.95);
    std::cout << "95% CI: [" << ci_var.lower << ", " << ci_var.upper << "]\n";
    std::cout << "\n解釈: 母分散は95%の確率で\n";
    std::cout << "      " << ci_var.lower << "〜" << ci_var.upper << " g² の範囲\n";

    print_subsection("標準偏差の95% 信頼区間");
    std::cout << "95% CI: [" << std::sqrt(ci_var.lower) << ", "
              << std::sqrt(ci_var.upper) << "] g\n";
    std::cout << "\n→ 標準偏差の CI は、分散の CI の平方根\n";

    print_subsection("実用例: 品質管理");
    std::cout << R"(
品質管理では、ばらつき（分散）の評価が重要：
- 分散が大きい → 製品品質が不安定
- 分散が小さい → 製品品質が安定

目標: 標準偏差 < 0.3 g

現在の95% CI: [)" << std::sqrt(ci_var.lower) << ", "
              << std::sqrt(ci_var.upper) << "] g\n";

    if (std::sqrt(ci_var.upper) < 0.3) {
        std::cout << "\n→ 目標を達成している（CI全体が0.3未満）\n";
    } else if (std::sqrt(ci_var.lower) > 0.3) {
        std::cout << "\n→ 目標未達成（CI全体が0.3超過）\n";
    } else {
        std::cout << "\n→ 判断保留（CIが0.3をまたいでいる）\n";
    }
}

// ============================================================================
// 7. サンプルサイズ設計
// ============================================================================

/**
 * @brief サンプルサイズ設計の例
 *
 * 【概念】
 * 希望する誤差範囲を達成するために必要なサンプルサイズを計算
 *
 * 【数式（比率の場合）】
 * n = (z(α/2) / MOE)² × p(1-p)
 *
 * p が不明な場合、p=0.5 を使用（最大分散）
 */
void example_sample_size() {
    print_section("7. サンプルサイズ設計（MOE方式）");

    std::cout << R"(
【実例1: 国政選挙の出口調査】
選挙で候補者Aの得票率を ±3% の精度で推定したい（95% CI）
必要なサンプルサイズは？
)";

    double desired_moe = 0.03;  // ±3%

    print_subsection("事前情報がない場合（保守的推定）");
    std::cout << "得票率が不明な場合、p=0.5（最大分散）を仮定\n";

    // 新しい関数を使用
    std::size_t n_conservative = statcpp::sample_size_for_moe_proportion(
        desired_moe, 0.95, 0.5);

    std::cout << "必要サンプル数: " << n_conservative << " 人\n";
    std::cout << "\n→ 約1068人に出口調査を行う必要がある\n";
    std::cout << "  （事前情報なしの場合、最も安全な見積もり）\n";

    print_subsection("事前の世論調査データがある場合");
    std::cout << R"(
事前の世論調査で候補者Aの支持率が約40%とわかっている場合:
)";
    double p_prior = 0.40;
    std::size_t n_with_prior = statcpp::sample_size_for_moe_proportion(
        desired_moe, 0.95, p_prior);

    std::cout << "必要サンプル数: " << n_with_prior << " 人\n";
    std::cout << "\n→ 約1025人（保守的推定より少し少ない）\n";
    std::cout << "  事前情報を使うことで、調査コストを削減できる\n";

    // 新しい例: 地方選挙
    print_subsection("【実例2: 地方選挙・首長選】");
    std::cout << R"(
市長選で現職の支持率を ±5% の精度で推定したい（95% CI）
地方選挙では予算が限られるため、精度をやや緩和
)";

    std::size_t n_local = statcpp::sample_size_for_moe_proportion(
        0.05, 0.95, 0.5);

    std::cout << "必要サンプル数: " << n_local << " 人\n";
    std::cout << "\n→ 約385人（±5%精度なら実施可能な規模）\n";

    // 新しい例: 接戦の選挙
    print_subsection("【実例3: 接戦選挙でより高精度が必要な場合】");
    std::cout << R"(
2候補が拮抗しており、±2% の精度で推定したい（95% CI）
接戦では高い精度が求められる
)";

    std::size_t n_tight = statcpp::sample_size_for_moe_proportion(
        0.02, 0.95, 0.5);

    std::cout << "必要サンプル数: " << n_tight << " 人\n";
    std::cout << "\n→ 約2401人（高精度には大規模調査が必要）\n";

    // 新しい例: 信頼水準を変更
    print_subsection("【実例4: 信頼水準を変更する場合】");
    std::cout << R"(
99% CI で ±3% の精度が欲しい場合（より確実な推定）
)";

    std::size_t n_high_conf = statcpp::sample_size_for_moe_proportion(
        0.03, 0.99, 0.5);

    std::cout << "必要サンプル数（99% CI）: " << n_high_conf << " 人\n";

    std::size_t n_normal_conf = statcpp::sample_size_for_moe_proportion(
        0.03, 0.95, 0.5);

    std::cout << "必要サンプル数（95% CI）: " << n_normal_conf << " 人\n";
    std::cout << "\n→ 信頼水準を上げると、必要サンプル数も増加\n";

    // 新しい例: 国民投票・住民投票
    print_subsection("【実例5: 国民投票・住民投票の支持率調査】");
    std::cout << R"(
憲法改正や基地移設などの賛否を調査
賛成率を ±2.5% の精度で推定（95% CI）
)";

    std::size_t n_referendum = statcpp::sample_size_for_moe_proportion(
        0.025, 0.95, 0.5);

    std::cout << "必要サンプル数: " << n_referendum << " 人\n";
    std::cout << "\n→ 約1537人（重要な意思決定には十分な精度が必要）\n";

    // 誤差範囲とサンプルサイズの関係表
    print_subsection("誤差範囲とサンプルサイズの関係（95% CI）");
    std::cout << "\n選挙・世論調査で一般的な精度レベル:\n";
    std::cout << "┌──────────┬────────────┬────────────────┐\n";
    std::cout << "│ 誤差範囲 │ サンプル数 │ 用途例         │\n";
    std::cout << "├──────────┼────────────┼────────────────┤\n";

    struct MOEExample {
        double moe;
        const char* use_case;
    };

    std::vector<MOEExample> moe_examples = {
        {0.01, "大規模全国調査"},
        {0.02, "国政選挙・接戦"},
        {0.03, "標準的な出口調査"},
        {0.04, "中規模世論調査"},
        {0.05, "地方選挙・予備調査"},
        {0.10, "小規模探索調査"}
    };

    for (const auto& ex : moe_examples) {
        std::size_t n = statcpp::sample_size_for_moe_proportion(ex.moe, 0.95, 0.5);
        std::cout << "│ ±" << std::setw(5) << ex.moe * 100 << "% │ "
                  << std::setw(10) << n << " │ "
                  << std::setw(14) << ex.use_case << " │\n";
    }
    std::cout << "└──────────┴────────────┴────────────────┘\n";

    std::cout << "\n→ 重要な法則: 精度を2倍にするには、サンプルサイズを4倍にする必要\n";
    std::cout << "  例: ±6% → ±3% にするには 4倍のコストがかかる\n";
}

// ============================================================================
// まとめ
// ============================================================================

void print_summary() {
    print_section("まとめ：統計的推定の関数");

    std::cout << R"(
┌────────────────────────────────┬─────────────────────────────────────┐
│ 関数                           │ 説明                                │
├────────────────────────────────┼─────────────────────────────────────┤
│ standard_error()               │ 標準誤差（推定の精度）              │
│ ci_mean()                      │ 平均の信頼区間                      │
│ margin_of_error_mean()         │ 平均の誤差範囲                      │
│ ci_proportion()                │ 比率の信頼区間                      │
│ margin_of_error_*()            │ 比率の誤差範囲                      │
│ sample_size_for_moe_proportion │ 比率推定に必要なサンプル数計算      │
│ sample_size_for_moe_mean       │ 平均推定に必要なサンプル数計算      │
│ ci_mean_diff_pooled()          │ 2標本平均差（等分散）               │
│ ci_mean_diff_welch()           │ 2標本平均差（不等分散）             │
│ ci_mean_diff_paired()          │ 対応のある平均差                    │
│ ci_variance()                  │ 分散の信頼区間                      │
└────────────────────────────────┴─────────────────────────────────────┘

【信頼区間の解釈】
✅ 正しい解釈:
   「同じ方法で100回標本抽出すると、約95回は母数を含む区間が得られる」

❌ 間違った解釈:
   「母数がこの区間に含まれる確率が95%」
   （母数は固定値であり、確率的ではない）

【実用上のヒント】
1. 信頼水準の選択:
   - 探索的分析 → 90%
   - 一般的な研究 → 95%（標準）
   - 医薬品・安全性 → 99%

2. 平均差の推定:
   - 分散が等しい → ci_mean_diff_pooled()
   - 分散が不明/異なる → ci_mean_diff_welch()（より安全）
   - 対応あり → ci_mean_diff_paired()

3. サンプルサイズ設計（MOE方式）:
   - 選挙の出口調査、世論調査などで使用
   - 比率: sample_size_for_moe_proportion(moe, conf, p)
   - 平均: sample_size_for_moe_mean(moe, sigma, conf)
   - 精度を2倍にする → サンプルサイズは4倍必要
   - 比率の場合、p=0.5 を仮定すると保守的

   典型的な選挙調査の精度:
   - ±1%: 約9604人（大規模全国調査）
   - ±2%: 約2401人（国政選挙・接戦）
   - ±3%: 約1068人（標準的な出口調査）
   - ±5%: 約385人（地方選挙）

【統計的有意性の判定】
信頼区間を使った判定:
- CI が 0 を含まない → 統計的に有意
- CI が 0 を含む → 統計的に有意でない

例: 平均差の95% CI が [2.3, 5.7] → 有意（0を含まない）
    平均差の95% CI が [-1.2, 3.4] → 非有意（0を含む）
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_ci_concept();
    example_standard_error();
    example_ci_mean();
    example_ci_proportion();
    example_ci_mean_diff();
    example_ci_variance();
    example_sample_size();

    // まとめを表示
    print_summary();

    return 0;
}
