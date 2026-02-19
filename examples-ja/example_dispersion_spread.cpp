/**
 * @file example_dispersion_spread.cpp
 * @brief statcpp::dispersion_spread.hpp のサンプルコード
 *
 * このファイルでは、dispersion_spread.hpp で提供される散らばり（分散）の
 * 指標を計算する関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - range()                   : 範囲（最大値 - 最小値）
 * - var()                     : 分散（ddof指定可能）
 * - population_variance()     : 母分散（ddof=0）
 * - sample_variance()         : 標本分散（ddof=1）
 * - variance()                : 分散（sample_variance のエイリアス）
 * - stdev()                   : 標準偏差（ddof指定可能）
 * - population_stddev()       : 母標準偏差（ddof=0）
 * - sample_stddev()           : 標本標準偏差（ddof=1）
 * - stddev()                  : 標準偏差（sample_stddev のエイリアス）
 * - coefficient_of_variation(): 変動係数
 * - iqr()                     : 四分位範囲（※要ソート済みデータ）
 * - mean_absolute_deviation() : 平均絶対偏差
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_dispersion_spread.cpp -o example_dispersion_spread
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// statcpp の散らばりヘッダー
#include "statcpp/dispersion_spread.hpp"
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
// 1. range() - 範囲
// ============================================================================

/**
 * @brief range() の使用例
 *
 * 【目的】
 * range() は、データの最大値と最小値の差（範囲）を計算します。
 * データの散らばりの最も単純な指標です。
 *
 * 【数式】
 * range = max(x) - min(x)
 *
 * 【使用場面】
 * - データの全体的な広がりを素早く把握
 * - 品質管理での許容範囲の確認
 * - 温度変動幅、株価変動幅の測定
 *
 * 【注意点】
 * - 外れ値の影響を非常に受けやすい
 * - 分布の詳細な情報は得られない
 */
void example_range() {
    print_section("1. range() - 範囲");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("テストの点数", scores);

    double r = statcpp::range(scores.begin(), scores.end());
    std::cout << "範囲: " << r << "点 (最高95点 - 最低75点)\n";

    // 外れ値の影響を確認
    print_subsection("外れ値の影響");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    print_data("外れ値あり", with_outlier);
    double r_outlier = statcpp::range(with_outlier.begin(), with_outlier.end());
    std::cout << "外れ値なしの範囲: " << r << "\n";
    std::cout << "外れ値ありの範囲: " << r_outlier << " (200点により大幅に拡大)\n";
}

// ============================================================================
// 2. var() と population_variance(), sample_variance() - 分散
// ============================================================================

/**
 * @brief var() と関連関数の使用例
 *
 * 【目的】
 * 分散は、データが平均からどれだけ散らばっているかを表す指標です。
 * 各データと平均との差（偏差）の二乗の平均です。
 *
 * 【数式】
 * 母分散: σ² = Σ(xᵢ - μ)² / N
 * 標本分散: s² = Σ(xᵢ - x̄)² / (N - 1)
 *
 * 【ddof (Delta Degrees of Freedom)】
 * - ddof = 0: N で割る（母分散）
 * - ddof = 1: N-1 で割る（標本分散、不偏分散）
 *
 * 【使用場面】
 * - データのばらつきの定量化
 * - リスク測定（投資収益率の分散）
 * - 統計検定の基礎計算
 *
 * 【注意点】
 * - 標本から母集団を推定する場合は ddof=1（不偏分散）を使用
 * - 単位は元データの二乗になる
 */
void example_variance() {
    print_section("2. var() - 分散");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("テストの点数", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    std::cout << "平均: " << mean_val << "点\n\n";

    // var() with ddof
    print_subsection("ddof の違い");
    double var_population = statcpp::var(scores.begin(), scores.end(), 0);  // ddof=0
    double var_sample = statcpp::var(scores.begin(), scores.end(), 1);      // ddof=1
    std::cout << "母分散 (ddof=0, N=" << scores.size() << "で割る): " << var_population << "\n";
    std::cout << "標本分散 (ddof=1, N-1=" << scores.size() - 1 << "で割る): " << var_sample << "\n";

    // エイリアス関数
    print_subsection("エイリアス関数");
    double pop_var = statcpp::population_variance(scores.begin(), scores.end());
    double samp_var = statcpp::sample_variance(scores.begin(), scores.end());
    double variance_val = statcpp::variance(scores.begin(), scores.end());  // = sample_variance
    std::cout << "population_variance(): " << pop_var << "\n";
    std::cout << "sample_variance():     " << samp_var << "\n";
    std::cout << "variance():            " << variance_val << " (= sample_variance)\n";

    // 事前計算済み平均を使う場合
    print_subsection("事前計算済み平均を使う場合（効率化）");
    double var_with_mean = statcpp::var(scores.begin(), scores.end(), mean_val, 1);
    std::cout << "平均を再利用した標本分散: " << var_with_mean << "\n";
    std::cout << "→ 平均を既に計算済みの場合、再計算を避けられる\n";
}

// ============================================================================
// 3. stdev() と population_stddev(), sample_stddev() - 標準偏差
// ============================================================================

/**
 * @brief stdev() と関連関数の使用例
 *
 * 【目的】
 * 標準偏差は、分散の平方根です。
 * 元データと同じ単位を持つため、解釈が容易です。
 *
 * 【数式】
 * 母標準偏差: σ = √(Σ(xᵢ - μ)² / N)
 * 標本標準偏差: s = √(Σ(xᵢ - x̄)² / (N - 1))
 *
 * 【使用場面】
 * - 平均からの典型的な散らばりの大きさ
 * - 正規分布では約68%のデータが平均±1σに収まる
 * - 品質管理、偏差値の計算
 *
 * 【注意点】
 * - 分散と同様、母標準偏差と標本標準偏差を区別する
 */
void example_stddev() {
    print_section("3. stdev() - 標準偏差");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("テストの点数", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    std::cout << "平均: " << mean_val << "点\n\n";

    // stdev() with ddof
    print_subsection("ddof の違い");
    double stdev_pop = statcpp::stdev(scores.begin(), scores.end(), 0);  // ddof=0
    double stdev_samp = statcpp::stdev(scores.begin(), scores.end(), 1); // ddof=1
    std::cout << "母標準偏差 (ddof=0): " << stdev_pop << "点\n";
    std::cout << "標本標準偏差 (ddof=1): " << stdev_samp << "点\n";

    // エイリアス関数
    print_subsection("エイリアス関数");
    double pop_sd = statcpp::population_stddev(scores.begin(), scores.end());
    double samp_sd = statcpp::sample_stddev(scores.begin(), scores.end());
    double stddev_val = statcpp::stddev(scores.begin(), scores.end());  // = sample_stddev
    std::cout << "population_stddev(): " << pop_sd << "点\n";
    std::cout << "sample_stddev():     " << samp_sd << "点\n";
    std::cout << "stddev():            " << stddev_val << "点 (= sample_stddev)\n";

    // 解釈の例
    print_subsection("標準偏差の解釈");
    std::cout << "平均 " << mean_val << "点 ± 標準偏差 " << stddev_val << "点\n";
    std::cout << "→ 約68%の学生が " << (mean_val - stddev_val)
              << "点 〜 " << (mean_val + stddev_val) << "点 の範囲に収まる（正規分布の場合）\n";
}

// ============================================================================
// 4. coefficient_of_variation() - 変動係数
// ============================================================================

/**
 * @brief coefficient_of_variation() の使用例
 *
 * 【目的】
 * 変動係数(CV)は、標準偏差を平均で割った値です。
 * 異なるスケールのデータの散らばりを比較できます。
 *
 * 【数式】
 * CV = σ / |μ| （比率として返される。%表示には ×100）
 *
 * 【使用場面】
 * - 異なる単位や規模のデータの比較
 *   例：身長(cm)と体重(kg)の散らばりを比較
 * - 測定の精度評価
 * - 株価のボラティリティ比較
 *
 * 【注意点】
 * - 平均が0に近いデータでは使用不可（0除算）
 * - 比率データや対数変換後のデータに向いている
 */
void example_coefficient_of_variation() {
    print_section("4. coefficient_of_variation() - 変動係数");

    // 異なるスケールのデータ比較
    std::vector<double> heights = {160, 165, 170, 175, 180};  // cm
    std::vector<double> weights = {50, 55, 60, 65, 70};       // kg

    print_data("身長 (cm)", heights);
    print_data("体重 (kg)", weights);

    double height_mean = statcpp::mean(heights.begin(), heights.end());
    double height_sd = statcpp::stddev(heights.begin(), heights.end());
    double height_cv = statcpp::coefficient_of_variation(heights.begin(), heights.end());

    double weight_mean = statcpp::mean(weights.begin(), weights.end());
    double weight_sd = statcpp::stddev(weights.begin(), weights.end());
    double weight_cv = statcpp::coefficient_of_variation(weights.begin(), weights.end());

    std::cout << "\n身長:\n";
    std::cout << "  平均: " << height_mean << " cm\n";
    std::cout << "  標準偏差: " << height_sd << " cm\n";
    std::cout << "  変動係数: " << height_cv << " (" << height_cv * 100 << "%)\n";

    std::cout << "\n体重:\n";
    std::cout << "  平均: " << weight_mean << " kg\n";
    std::cout << "  標準偏差: " << weight_sd << " kg\n";
    std::cout << "  変動係数: " << weight_cv << " (" << weight_cv * 100 << "%)\n";

    std::cout << "\n→ 単位が異なるため標準偏差では比較できないが、\n";
    std::cout << "  変動係数により相対的な散らばりを比較可能\n";
    if (height_cv < weight_cv) {
        std::cout << "  体重の方が相対的に散らばりが大きい\n";
    } else {
        std::cout << "  身長の方が相対的に散らばりが大きい\n";
    }
}

// ============================================================================
// 5. iqr() - 四分位範囲
// ============================================================================

/**
 * @brief iqr() の使用例
 *
 * 【目的】
 * 四分位範囲(IQR)は、第3四分位数(Q3)と第1四分位数(Q1)の差です。
 * 中央50%のデータの広がりを示します。
 *
 * 【数式】
 * IQR = Q3 - Q1
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【使用場面】
 * - 外れ値の影響を受けにくい散らばりの指標
 * - 箱ひげ図の作成
 * - 外れ値の検出（1.5×IQR ルール）
 *
 * 【注意点】
 * - 中央50%のみを見るため、極端な値を無視
 * - 分布の裾の情報は失われる
 */
void example_iqr() {
    print_section("5. iqr() - 四分位範囲");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // ソートしてから計算（重要）
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("ソート済みデータ", sorted_scores);

    double iqr_val = statcpp::iqr(sorted_scores.begin(), sorted_scores.end());
    std::cout << "四分位範囲 (IQR): " << iqr_val << "点\n";

    // 外れ値の影響を確認
    print_subsection("外れ値に対するロバスト性");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_outlier = with_outlier;
    std::sort(sorted_outlier.begin(), sorted_outlier.end());
    print_data("外れ値ありデータ (ソート済み)", sorted_outlier);

    double iqr_outlier = statcpp::iqr(sorted_outlier.begin(), sorted_outlier.end());
    double range_outlier = statcpp::range(sorted_outlier.begin(), sorted_outlier.end());

    std::cout << "外れ値なしの IQR: " << iqr_val << "\n";
    std::cout << "外れ値ありの IQR: " << iqr_outlier << " (Q1-Q3は変化しにくい)\n";
    std::cout << "外れ値ありの範囲: " << range_outlier << " (大きく変化)\n";
    std::cout << "→ IQR は外れ値に対してロバスト\n";

    // 外れ値検出への応用
    print_subsection("外れ値検出（1.5×IQRルール）");
    double q1 = statcpp::percentile(sorted_outlier.begin(), sorted_outlier.end(), 0.25);
    double q3 = statcpp::percentile(sorted_outlier.begin(), sorted_outlier.end(), 0.75);
    double lower_fence = q1 - 1.5 * iqr_outlier;
    double upper_fence = q3 + 1.5 * iqr_outlier;
    std::cout << "Q1: " << q1 << ", Q3: " << q3 << ", IQR: " << iqr_outlier << "\n";
    std::cout << "下限フェンス: " << lower_fence << "\n";
    std::cout << "上限フェンス: " << upper_fence << "\n";
    std::cout << "→ 200点はフェンス外にあるため外れ値と判定\n";
}

// ============================================================================
// 6. mean_absolute_deviation() - 平均絶対偏差
// ============================================================================

/**
 * @brief mean_absolute_deviation() の使用例
 *
 * 【目的】
 * 平均絶対偏差(MAD)は、各データと平均との差の絶対値の平均です。
 * 標準偏差より外れ値に強い散らばりの指標です。
 *
 * 【数式】
 * MAD = Σ|xᵢ - x̄| / n
 *
 * 【使用場面】
 * - 外れ値に対してロバストな散らばりの指標が必要な場合
 * - 予測誤差の評価（MAE: Mean Absolute Error）
 * - 単位が元データと同じなので解釈が容易
 *
 * 【注意点】
 * - 数学的には標準偏差の方が扱いやすい性質を持つ
 * - 分散分析などの統計検定では標準偏差を使用
 */
void example_mean_absolute_deviation() {
    print_section("6. mean_absolute_deviation() - 平均絶対偏差");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("テストの点数", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    double mad = statcpp::mean_absolute_deviation(scores.begin(), scores.end());
    double sd = statcpp::stddev(scores.begin(), scores.end());

    std::cout << "平均: " << mean_val << "点\n";
    std::cout << "平均絶対偏差 (MAD): " << mad << "点\n";
    std::cout << "標準偏差 (SD): " << sd << "点\n";
    std::cout << "→ MAD ≈ 0.8 × SD（正規分布の場合）\n";

    // 外れ値の影響を比較
    print_subsection("外れ値に対する頑健性");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    print_data("外れ値ありデータ", with_outlier);

    double mad_outlier = statcpp::mean_absolute_deviation(with_outlier.begin(), with_outlier.end());
    double sd_outlier = statcpp::stddev(with_outlier.begin(), with_outlier.end());

    std::cout << "\n外れ値なし:\n";
    std::cout << "  MAD: " << mad << ", SD: " << sd << "\n";
    std::cout << "外れ値あり:\n";
    std::cout << "  MAD: " << mad_outlier << ", SD: " << sd_outlier << "\n";
    std::cout << "\n→ 外れ値により:\n";
    std::cout << "  SDの増加率: " << (sd_outlier / sd - 1) * 100 << "%\n";
    std::cout << "  MADの増加率: " << (mad_outlier / mad - 1) * 100 << "%\n";
    std::cout << "  → MAD の方が外れ値の影響が小さい\n";
}

// ============================================================================
// 7. ラムダ式（射影）を使った使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 *
 * 【目的】
 * すべての分散関数は、ラムダ式（射影関数）を受け取るオーバーロードを持ちます。
 * 構造体のメンバーや変換後の値に対して散らばりを計算できます。
 */
void example_projection() {
    print_section("7. ラムダ式（射影）を使った使用例");

    // 構造体の例
    struct Product {
        std::string name;
        double price;
        int quantity;
    };

    std::vector<Product> products = {
        {"商品A", 1000, 50},
        {"商品B", 1500, 30},
        {"商品C", 800, 70},
        {"商品D", 2000, 20},
        {"商品E", 1200, 40}
    };

    std::cout << "商品データ:\n";
    for (const auto& p : products) {
        std::cout << "  " << p.name << ": 価格=" << p.price
                  << "円, 数量=" << p.quantity << "個\n";
    }

    // 価格の統計
    double price_mean = statcpp::mean(products.begin(), products.end(),
                                      [](const Product& p) { return p.price; });
    double price_sd = statcpp::stddev(products.begin(), products.end(),
                                      [](const Product& p) { return p.price; });
    double price_cv = statcpp::coefficient_of_variation(products.begin(), products.end(),
                                                        [](const Product& p) { return p.price; });

    std::cout << "\n価格の統計:\n";
    std::cout << "  平均: " << price_mean << "円\n";
    std::cout << "  標準偏差: " << price_sd << "円\n";
    std::cout << "  変動係数: " << price_cv * 100 << "%\n";

    // 数量の統計
    double qty_mean = statcpp::mean(products.begin(), products.end(),
                                    [](const Product& p) { return p.quantity; });
    double qty_sd = statcpp::stddev(products.begin(), products.end(),
                                    [](const Product& p) { return p.quantity; });
    double qty_cv = statcpp::coefficient_of_variation(products.begin(), products.end(),
                                                      [](const Product& p) { return p.quantity; });

    std::cout << "\n数量の統計:\n";
    std::cout << "  平均: " << qty_mean << "個\n";
    std::cout << "  標準偏差: " << qty_sd << "個\n";
    std::cout << "  変動係数: " << qty_cv * 100 << "%\n";

    std::cout << "\n→ 変動係数で比較: ";
    if (price_cv > qty_cv) {
        std::cout << "価格の方が相対的に散らばりが大きい\n";
    } else {
        std::cout << "数量の方が相対的に散らばりが大きい\n";
    }
}

// ============================================================================
// 8. 母集団 vs 標本：どちらを使うべきか
// ============================================================================

/**
 * @brief 母分散と標本分散の使い分け
 *
 * 【目的】
 * 母分散(ddof=0)と標本分散(ddof=1)の違いと使い分けを説明します。
 */
void example_population_vs_sample() {
    print_section("8. 母集団 vs 標本：どちらを使うべきか");

    std::vector<double> data = {10, 12, 14, 16, 18};
    print_data("データ", data);

    double pop_var = statcpp::population_variance(data.begin(), data.end());
    double samp_var = statcpp::sample_variance(data.begin(), data.end());

    std::cout << "母分散 (N=" << data.size() << "で割る): " << pop_var << "\n";
    std::cout << "標本分散 (N-1=" << data.size() - 1 << "で割る): " << samp_var << "\n";

    std::cout << R"(
【使い分けの指針】

┌─────────────────────┬───────────────────────────────────┐
│ 状況                │ 使用する関数                      │
├─────────────────────┼───────────────────────────────────┤
│ データが母集団全体  │ population_variance()             │
│ 例：クラス全員の点数│ population_stddev()               │
│     を分析          │ var(first, last, 0)               │
├─────────────────────┼───────────────────────────────────┤
│ データが標本        │ sample_variance()                 │
│ （母集団から抽出）  │ sample_stddev()                   │
│ 例：アンケート調査  │ variance()                        │
│     臨床試験        │ stddev()                          │
│                     │ var(first, last, 1)               │
└─────────────────────┴───────────────────────────────────┘

【理由】
標本分散で N-1 で割るのは「不偏推定量」を得るため。
標本平均を使うと自由度が1減るため、N-1 で割ることで
母分散の期待値に一致する推定量（不偏推定量）になります。
)";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：散らばりの指標の選び方");

    std::cout << R"(
┌──────────────────────────┬────────────────────────────────────────┐
│ 関数                     │ 使用場面                               │
├──────────────────────────┼────────────────────────────────────────┤
│ range()                  │ 最も単純な散らばり。外れ値に弱い       │
│ variance() / stddev()    │ 一般的な散らばりの指標                 │
│ population_*()           │ データが母集団全体の場合               │
│ sample_*()               │ データが標本の場合（推定）             │
│ coefficient_of_variation()│異なるスケールのデータ比較             │
│ iqr()                    │ 外れ値にロバスト。箱ひげ図用           │
│ mean_absolute_deviation()│ 外れ値にある程度ロバスト               │
└──────────────────────────┴────────────────────────────────────────┘

【注意事項】
- iqr() は事前にソートが必要
- coefficient_of_variation() は平均が0のデータに使用不可
- 推定目的では sample_variance() / sample_stddev() を使用
- 外れ値がある場合は iqr() や MAD を検討
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_range();
    example_variance();
    example_stddev();
    example_coefficient_of_variation();
    example_iqr();
    example_mean_absolute_deviation();
    example_projection();
    example_population_vs_sample();

    // まとめを表示
    print_summary();

    return 0;
}
