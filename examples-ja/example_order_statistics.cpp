/**
 * @file example_order_statistics.cpp
 * @brief statcpp::order_statistics.hpp のサンプルコード
 *
 * このファイルでは、order_statistics.hpp で提供される順序統計量を
 * 計算する関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - minimum()             : 最小値
 * - maximum()             : 最大値
 * - quartiles()           : 四分位数（Q1, Q2, Q3）※要ソート済み
 * - percentile()          : パーセンタイル ※要ソート済み
 * - five_number_summary() : 五数要約 ※要ソート済み
 * - interpolate_at()      : 線形補間によるパーセンタイル計算（内部関数）
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_order_statistics.cpp -o example_order_statistics
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

// statcpp の順序統計量ヘッダー
#include "statcpp/order_statistics.hpp"
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
// 1. minimum() / maximum() - 最小値・最大値
// ============================================================================

/**
 * @brief minimum() / maximum() の使用例
 *
 * 【目的】
 * minimum() は最小値、maximum() は最大値を返します。
 * ソートは不要です。
 *
 * 【使用場面】
 * - データの範囲の確認
 * - 異常値の検出
 * - 基本的なデータ検証
 */
void example_min_max() {
    print_section("1. minimum() / maximum() - 最小値・最大値");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("テストの点数", scores);

    double min_val = statcpp::minimum(scores.begin(), scores.end());
    double max_val = statcpp::maximum(scores.begin(), scores.end());

    std::cout << "最小値: " << min_val << "点\n";
    std::cout << "最大値: " << max_val << "点\n";
    std::cout << "範囲: " << (max_val - min_val) << "点\n";

    // 整数データでも動作
    print_subsection("整数データの場合");
    std::vector<int> integers = {3, 1, 4, 1, 5, 9, 2, 6};
    print_data("整数データ", integers);
    std::cout << "最小値: " << statcpp::minimum(integers.begin(), integers.end()) << "\n";
    std::cout << "最大値: " << statcpp::maximum(integers.begin(), integers.end()) << "\n";
}

// ============================================================================
// 2. quartiles() - 四分位数
// ============================================================================

/**
 * @brief quartiles() の使用例
 *
 * 【目的】
 * quartiles() は、データを4等分する3つの値（Q1, Q2, Q3）を返します。
 *
 * 【数式】
 * Q1（第1四分位数）: データの25%点
 * Q2（第2四分位数）: データの50%点（= 中央値）
 * Q3（第3四分位数）: データの75%点
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【使用場面】
 * - データの分布を把握
 * - 箱ひげ図の作成
 * - 外れ値の検出（IQRルール）
 *
 * 【補間方法】
 * R type=7 / Excel QUARTILE.INC 相当の線形補間を使用
 */
void example_quartiles() {
    print_section("2. quartiles() - 四分位数");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // ソートが必須
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("ソート済みデータ", sorted_scores);

    statcpp::quartile_result q = statcpp::quartiles(sorted_scores.begin(), sorted_scores.end());

    std::cout << "\n四分位数:\n";
    std::cout << "  Q1（25%点）: " << q.q1 << "点\n";
    std::cout << "  Q2（50%点）: " << q.q2 << "点 (= 中央値)\n";
    std::cout << "  Q3（75%点）: " << q.q3 << "点\n";
    std::cout << "  IQR（四分位範囲）: " << (q.q3 - q.q1) << "点\n";

    // 解釈
    std::cout << "\n【解釈】\n";
    std::cout << "- 下位25%の学生は " << q.q1 << "点以下\n";
    std::cout << "- 中央50%の学生は " << q.q1 << "点 〜 " << q.q3 << "点\n";
    std::cout << "- 上位25%の学生は " << q.q3 << "点以上\n";
}

// ============================================================================
// 3. percentile() - パーセンタイル
// ============================================================================

/**
 * @brief percentile() の使用例
 *
 * 【目的】
 * percentile() は、指定した割合に対応するデータの値を返します。
 * パーセンタイルは0から100ではなく、0.0から1.0の割合で指定します。
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【使用場面】
 * - 偏差値の計算
 * - 成績の順位付け
 * - 応答時間の p99 の計算
 */
void example_percentile() {
    print_section("3. percentile() - パーセンタイル");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("ソート済みデータ", sorted_scores);

    std::cout << "\n主要なパーセンタイル:\n";
    std::cout << "  0パーセンタイル（最小値）: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.00) << "点\n";
    std::cout << "  10パーセンタイル: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.10) << "点\n";
    std::cout << "  25パーセンタイル（Q1）: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.25) << "点\n";
    std::cout << "  50パーセンタイル（中央値）: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.50) << "点\n";
    std::cout << "  75パーセンタイル（Q3）: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.75) << "点\n";
    std::cout << "  90パーセンタイル: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.90) << "点\n";
    std::cout << "  100パーセンタイル（最大値）: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 1.00) << "点\n";

    // 実用例: 応答時間の分析
    print_subsection("実用例: API応答時間の分析");
    std::vector<double> response_times = {
        45, 52, 48, 51, 49, 47, 55, 120, 46, 50,
        48, 53, 47, 49, 150, 51, 48, 52, 46, 200
    };
    std::sort(response_times.begin(), response_times.end());

    std::cout << "応答時間（ms）: ";
    for (double t : response_times) std::cout << t << " ";
    std::cout << "\n\n";

    std::cout << "p50（中央値）: "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.50) << " ms\n";
    std::cout << "p95: "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.95) << " ms\n";
    std::cout << "p99: "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.99) << " ms\n";
    std::cout << "\n→ p50 は良好だが、p99 が高い（外れ値の影響）\n";
}

// ============================================================================
// 4. five_number_summary() - 五数要約
// ============================================================================

/**
 * @brief five_number_summary() の使用例
 *
 * 【目的】
 * five_number_summary() は、データを要約する5つの値を返します：
 * - 最小値 (min)
 * - 第1四分位数 (Q1)
 * - 中央値 (median)
 * - 第3四分位数 (Q3)
 * - 最大値 (max)
 *
 * 【重要】
 * ※ 入力データは事前にソートされている必要があります！
 *
 * 【使用場面】
 * - データの全体像を素早く把握
 * - 箱ひげ図の作成
 * - レポートでのデータ要約
 */
void example_five_number_summary() {
    print_section("4. five_number_summary() - 五数要約");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("ソート済みデータ", sorted_scores);

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(sorted_scores.begin(), sorted_scores.end());

    std::cout << "\n五数要約:\n";
    std::cout << "  最小値: " << summary.min << "点\n";
    std::cout << "  Q1:     " << summary.q1 << "点\n";
    std::cout << "  中央値: " << summary.median << "点\n";
    std::cout << "  Q3:     " << summary.q3 << "点\n";
    std::cout << "  最大値: " << summary.max << "点\n";

    // テキストベースの箱ひげ図（概念図）
    print_subsection("箱ひげ図（概念図）");
    std::cout << R"(
    最小値  Q1    中央値   Q3    最大値
      │     ├─────┼─────┤     │
      ├─────┤     │     ├─────┤
      │     └─────┼─────┘     │
     )" << summary.min << "   " << summary.q1 << "   "
              << summary.median << "   " << summary.q3 << "   " << summary.max << "\n";
}

// ============================================================================
// 5. 外れ値の検出（IQRルール）
// ============================================================================

/**
 * @brief 外れ値検出への応用
 *
 * 【IQRルール】
 * 外れ値の定義:
 * - 軽度の外れ値: Q1 - 1.5×IQR より小さい、または Q3 + 1.5×IQR より大きい
 * - 重度の外れ値: Q1 - 3.0×IQR より小さい、または Q3 + 3.0×IQR より大きい
 */
void example_outlier_detection() {
    print_section("5. 外れ値の検出（IQRルール）");

    std::vector<double> data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 50};  // 50は外れ値

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    print_data("データ", sorted_data);

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(sorted_data.begin(), sorted_data.end());

    double iqr = summary.q3 - summary.q1;
    double lower_fence = summary.q1 - 1.5 * iqr;
    double upper_fence = summary.q3 + 1.5 * iqr;
    double lower_extreme = summary.q1 - 3.0 * iqr;
    double upper_extreme = summary.q3 + 3.0 * iqr;

    std::cout << "\n五数要約:\n";
    std::cout << "  最小値: " << summary.min << "\n";
    std::cout << "  Q1: " << summary.q1 << "\n";
    std::cout << "  中央値: " << summary.median << "\n";
    std::cout << "  Q3: " << summary.q3 << "\n";
    std::cout << "  最大値: " << summary.max << "\n";

    std::cout << "\nIQR: " << iqr << "\n";
    std::cout << "\n外れ値の判定基準:\n";
    std::cout << "  軽度の外れ値: < " << lower_fence << " または > " << upper_fence << "\n";
    std::cout << "  重度の外れ値: < " << lower_extreme << " または > " << upper_extreme << "\n";

    std::cout << "\n外れ値の検出:\n";
    for (double val : sorted_data) {
        if (val < lower_extreme || val > upper_extreme) {
            std::cout << "  " << val << " → 重度の外れ値\n";
        } else if (val < lower_fence || val > upper_fence) {
            std::cout << "  " << val << " → 軽度の外れ値\n";
        }
    }
}

// ============================================================================
// 6. ラムダ式（射影）を使った使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 *
 * 構造体のメンバーに対して順序統計量を計算します。
 */
void example_projection() {
    print_section("6. ラムダ式（射影）を使った使用例");

    struct Employee {
        std::string name;
        int salary;
        int years;
    };

    std::vector<Employee> employees = {
        {"Alice", 450, 3},
        {"Bob", 520, 5},
        {"Charlie", 380, 2},
        {"Diana", 680, 8},
        {"Eve", 420, 4},
        {"Frank", 550, 6},
        {"Grace", 490, 5},
        {"Henry", 350, 1}
    };

    std::cout << "従業員データ:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": 給与=" << e.salary
                  << "万円, 勤続=" << e.years << "年\n";
    }

    // 給与の最小・最大
    auto salary_proj = [](const Employee& e) { return e.salary; };

    int min_salary = statcpp::minimum(employees.begin(), employees.end(), salary_proj);
    int max_salary = statcpp::maximum(employees.begin(), employees.end(), salary_proj);

    std::cout << "\n給与の範囲:\n";
    std::cout << "  最小: " << min_salary << "万円\n";
    std::cout << "  最大: " << max_salary << "万円\n";

    // 給与でソートしてから四分位数を計算
    std::sort(employees.begin(), employees.end(),
              [](const Employee& a, const Employee& b) { return a.salary < b.salary; });

    std::cout << "\n給与順にソート:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": " << e.salary << "万円\n";
    }

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(employees.begin(), employees.end(), salary_proj);

    std::cout << "\n給与の五数要約:\n";
    std::cout << "  最小値: " << summary.min << "万円\n";
    std::cout << "  Q1: " << summary.q1 << "万円\n";
    std::cout << "  中央値: " << summary.median << "万円\n";
    std::cout << "  Q3: " << summary.q3 << "万円\n";
    std::cout << "  最大値: " << summary.max << "万円\n";
}

// ============================================================================
// 7. 補間方法の説明
// ============================================================================

/**
 * @brief 補間方法の説明
 *
 * 本ライブラリは R type=7 / Excel QUARTILE.INC 相当の
 * 線形補間を使用しています。
 */
void example_interpolation() {
    print_section("7. 補間方法の説明");

    std::cout << R"(
【パーセンタイルの補間方法】

本ライブラリは R の quantile() 関数の type=7
および Excel の QUARTILE.INC / PERCENTILE.INC と同じ
線形補間方法を使用しています。

【計算方法】
1. インデックス = p × (n - 1)  （p: パーセンタイル割合, n: データ数）
2. インデックスを整数部 lo と小数部 frac に分解
3. 結果 = data[lo] × (1 - frac) + data[lo + 1] × frac

【例: n=10 のデータで 25パーセンタイル（Q1）を計算】
インデックス = 0.25 × (10 - 1) = 2.25
lo = 2, frac = 0.25
結果 = data[2] × 0.75 + data[3] × 0.25

)";

    // 具体例
    std::vector<double> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    print_data("データ (n=10)", data);

    std::cout << "\nQ1の計算:\n";
    std::cout << "  インデックス = 0.25 × 9 = 2.25\n";
    std::cout << "  lo = 2, frac = 0.25\n";
    std::cout << "  結果 = data[2] × 0.75 + data[3] × 0.25\n";
    std::cout << "       = 30 × 0.75 + 40 × 0.25 = 22.5 + 10 = 32.5\n";
    std::cout << "  実際の計算結果: "
              << statcpp::percentile(data.begin(), data.end(), 0.25) << "\n";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：順序統計量の関数");

    std::cout << R"(
┌────────────────────────┬───────────────────────────────────────────┐
│ 関数                   │ 説明                                      │
├────────────────────────┼───────────────────────────────────────────┤
│ minimum()              │ 最小値（ソート不要）                      │
│ maximum()              │ 最大値（ソート不要）                      │
│ quartiles()            │ 四分位数 Q1, Q2, Q3（要ソート）           │
│ percentile(first,last,p)│任意のパーセンタイル（要ソート）          │
│ five_number_summary()  │ 五数要約（要ソート）                      │
└────────────────────────┴───────────────────────────────────────────┘

【戻り値の構造体】
- quartile_result: q1, q2, q3
- five_number_summary_result: min, q1, median, q3, max

【重要な注意事項】
- quartiles(), percentile(), five_number_summary() は
  事前にソートされたデータを前提とする
- percentile() の引数 p は 0.0〜1.0 の割合で指定
  （例: 90パーセンタイル → p = 0.90）
- 補間方法は R type=7 / Excel QUARTILE.INC 相当

【外れ値検出（IQRルール）】
- IQR = Q3 - Q1
- 軽度の外れ値: Q1 - 1.5×IQR より小、または Q3 + 1.5×IQR より大
- 重度の外れ値: Q1 - 3.0×IQR より小、または Q3 + 3.0×IQR より大
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_min_max();
    example_quartiles();
    example_percentile();
    example_five_number_summary();
    example_outlier_detection();
    example_projection();
    example_interpolation();

    // まとめを表示
    print_summary();

    return 0;
}
