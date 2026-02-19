/**
 * @file example_frequency_distribution.cpp
 * @brief statcpp::frequency_distribution.hpp のサンプルコード
 *
 * このファイルでは、frequency_distribution.hpp で提供される
 * 度数分布を計算する関数の使い方を実践的な例を通じて説明します。
 *
 * 【提供される関数】
 * - frequency_table()               : 度数表（ソート済み、全情報含む）
 * - frequency_count()               : 度数（高速、unordered_map版）
 * - relative_frequency()            : 相対度数
 * - cumulative_frequency()          : 累積度数
 * - cumulative_relative_frequency() : 累積相対度数
 *
 * 【コンパイル方法】
 * g++ -std=c++17 -I/path/to/statcpp/include example_frequency_distribution.cpp -o example_frequency_distribution
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

// statcpp の度数分布ヘッダー
#include "statcpp/frequency_distribution.hpp"

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
// 1. frequency_table() - 度数表
// ============================================================================

/**
 * @brief frequency_table() の使用例
 *
 * 【目的】
 * frequency_table() は、各値の度数、相対度数、累積度数、累積相対度数を
 * 一度に計算し、ソート済みの度数表として返します。
 *
 * 【戻り値の構造】
 * frequency_table_result<T>:
 *   - entries: vector<frequency_entry<T>>
 *     - value: 値
 *     - count: 度数（出現回数）
 *     - relative_frequency: 相対度数（比率）
 *     - cumulative_count: 累積度数
 *     - cumulative_relative_frequency: 累積相対度数
 *   - total_count: 総データ数
 *
 * 【使用場面】
 * - データの分布を把握
 * - ヒストグラムの作成
 * - カテゴリカルデータの集計
 */
void example_frequency_table() {
    print_section("1. frequency_table() - 度数表");

    // テストの点数（離散値）
    std::vector<int> scores = {80, 85, 90, 80, 75, 85, 90, 95, 80, 85,
                               85, 90, 80, 75, 85, 90, 80, 85, 80, 90};

    std::cout << "テストの点数データ:\n";
    print_data("データ", scores);
    std::cout << "\n";

    auto table = statcpp::frequency_table(scores.begin(), scores.end());

    std::cout << "度数表:\n";
    std::cout << std::setw(8) << "値"
              << std::setw(10) << "度数"
              << std::setw(12) << "相対度数"
              << std::setw(12) << "累積度数"
              << std::setw(15) << "累積相対度数" << "\n";
    std::cout << std::string(57, '-') << "\n";

    for (const auto& entry : table.entries) {
        std::cout << std::setw(8) << entry.value
                  << std::setw(10) << entry.count
                  << std::setw(12) << entry.relative_frequency
                  << std::setw(12) << entry.cumulative_count
                  << std::setw(15) << entry.cumulative_relative_frequency << "\n";
    }
    std::cout << std::string(57, '-') << "\n";
    std::cout << "合計: " << table.total_count << "件\n";
}

// ============================================================================
// 2. frequency_count() - 度数（高速版）
// ============================================================================

/**
 * @brief frequency_count() の使用例
 *
 * 【目的】
 * frequency_count() は、各値の出現回数のみを高速に計算します。
 * 結果は unordered_map で返されるため、ソートされていません。
 *
 * 【特徴】
 * - frequency_table() より高速（相対度数等の計算をしない）
 * - 結果はソートされていない
 * - 単純に度数のみが必要な場合に適切
 */
void example_frequency_count() {
    print_section("2. frequency_count() - 度数（高速版）");

    std::vector<std::string> responses = {
        "賛成", "反対", "賛成", "どちらでもない", "賛成",
        "反対", "賛成", "賛成", "反対", "どちらでもない"
    };

    std::cout << "アンケート回答データ:\n";
    for (const auto& r : responses) std::cout << r << " ";
    std::cout << "\n\n";

    auto freq = statcpp::frequency_count(responses.begin(), responses.end());

    std::cout << "度数:\n";
    // ソートして表示（unordered_map は順序が不定なため）
    std::map<std::string, std::size_t> sorted_freq(freq.begin(), freq.end());
    for (const auto& [value, count] : sorted_freq) {
        std::cout << "  " << value << ": " << count << "回\n";
    }
}

// ============================================================================
// 3. relative_frequency() - 相対度数
// ============================================================================

/**
 * @brief relative_frequency() の使用例
 *
 * 【目的】
 * relative_frequency() は、各値の相対度数（比率）を計算します。
 * 相対度数 = その値の度数 / 全データ数
 *
 * 【数式】
 * 相対度数(x) = count(x) / n
 *
 * 【使用場面】
 * - データを比率で比較したい場合
 * - 確率分布の推定
 * - パーセンテージ表示が必要な場合
 */
void example_relative_frequency() {
    print_section("3. relative_frequency() - 相対度数");

    std::vector<int> dice_rolls = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4,
                                   5, 6, 1, 2, 3, 4, 5, 6, 6, 6};

    print_data("サイコロの出目", dice_rolls);
    std::cout << "試行回数: " << dice_rolls.size() << "回\n\n";

    auto rel_freq = statcpp::relative_frequency(dice_rolls.begin(), dice_rolls.end());

    // ソートして表示
    std::map<int, double> sorted_freq(rel_freq.begin(), rel_freq.end());

    std::cout << "相対度数:\n";
    std::cout << std::setw(8) << "出目" << std::setw(15) << "相対度数"
              << std::setw(15) << "パーセント" << "\n";
    std::cout << std::string(38, '-') << "\n";

    for (const auto& [value, freq] : sorted_freq) {
        std::cout << std::setw(8) << value
                  << std::setw(15) << freq
                  << std::setw(14) << (freq * 100) << "%\n";
    }

    std::cout << "\n理論値（公正なサイコロ）: 各16.67%\n";
    std::cout << "→ 6が多く出ており、偏りがある可能性\n";
}

// ============================================================================
// 4. cumulative_frequency() - 累積度数
// ============================================================================

/**
 * @brief cumulative_frequency() の使用例
 *
 * 【目的】
 * cumulative_frequency() は、各値までの累積度数を計算します。
 * 値の小さい順にソートされた結果が返されます。
 *
 * 【数式】
 * 累積度数(x) = Σ count(y) for all y ≤ x
 *
 * 【使用場面】
 * - 「x以下のデータが何件あるか」を調べる
 * - 累積分布関数の近似
 * - パーセンタイルの計算
 */
void example_cumulative_frequency() {
    print_section("4. cumulative_frequency() - 累積度数");

    std::vector<int> ages = {20, 25, 30, 35, 40, 25, 30, 35, 30, 30,
                             25, 35, 40, 45, 30, 35, 25, 30, 35, 40};

    print_data("参加者の年齢", ages);
    std::cout << "\n";

    auto cum_freq = statcpp::cumulative_frequency(ages.begin(), ages.end());

    std::cout << "累積度数:\n";
    std::cout << std::setw(10) << "年齢以下" << std::setw(15) << "累積人数" << "\n";
    std::cout << std::string(25, '-') << "\n";

    for (const auto& [value, cum] : cum_freq) {
        std::cout << std::setw(8) << value << "歳"
                  << std::setw(12) << cum << "人\n";
    }

    std::cout << "\n→ 35歳以下の参加者は " << cum_freq[4].second << "人\n";
}

// ============================================================================
// 5. cumulative_relative_frequency() - 累積相対度数
// ============================================================================

/**
 * @brief cumulative_relative_frequency() の使用例
 *
 * 【目的】
 * cumulative_relative_frequency() は、各値までの累積相対度数を計算します。
 * 経験的累積分布関数(ECDF)と同等の値を返します。
 *
 * 【数式】
 * 累積相対度数(x) = 累積度数(x) / n
 *
 * 【使用場面】
 * - 「x以下のデータが全体の何%か」を調べる
 * - 経験的累積分布関数(ECDF)の作成
 * - パーセンタイル順位の計算
 */
void example_cumulative_relative_frequency() {
    print_section("5. cumulative_relative_frequency() - 累積相対度数");

    std::vector<int> test_scores = {60, 65, 70, 75, 80, 85, 90, 95,
                                    70, 75, 80, 80, 85, 85, 85, 90};

    print_data("テストの点数", test_scores);
    std::cout << "受験者数: " << test_scores.size() << "人\n\n";

    auto cum_rel = statcpp::cumulative_relative_frequency(
        test_scores.begin(), test_scores.end());

    std::cout << "累積相対度数:\n";
    std::cout << std::setw(10) << "点数以下"
              << std::setw(15) << "累積比率"
              << std::setw(15) << "パーセント" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& [value, cum_rel_freq] : cum_rel) {
        std::cout << std::setw(8) << value << "点"
                  << std::setw(15) << cum_rel_freq
                  << std::setw(14) << (cum_rel_freq * 100) << "%\n";
    }

    // 特定のパーセンタイル位置を表示
    std::cout << "\n【解釈の例】\n";
    for (const auto& [value, cum_rel_freq] : cum_rel) {
        if (cum_rel_freq >= 0.5 && cum_rel_freq < 0.7) {
            std::cout << "- " << value << "点以下は全体の "
                      << (cum_rel_freq * 100) << "% (約上位"
                      << ((1.0 - cum_rel_freq) * 100) << "%)\n";
        }
    }
}

// ============================================================================
// 6. ラムダ式（射影）を使った使用例
// ============================================================================

/**
 * @brief ラムダ式（射影）を使った高度な使用例
 *
 * 構造体のメンバーに対して度数分布を計算します。
 */
void example_projection() {
    print_section("6. ラムダ式（射影）を使った使用例");

    struct Employee {
        std::string name;
        std::string department;
        int years;
    };

    std::vector<Employee> employees = {
        {"Alice", "営業", 3},
        {"Bob", "開発", 5},
        {"Charlie", "営業", 2},
        {"Diana", "開発", 8},
        {"Eve", "人事", 4},
        {"Frank", "営業", 5},
        {"Grace", "開発", 5},
        {"Henry", "人事", 1},
        {"Ivy", "開発", 3},
        {"Jack", "営業", 7}
    };

    std::cout << "従業員データ:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": " << e.department
                  << ", 勤続" << e.years << "年\n";
    }

    // 部署別の度数
    print_subsection("部署別の度数");
    auto dept_proj = [](const Employee& e) { return e.department; };
    auto dept_freq = statcpp::frequency_count(
        employees.begin(), employees.end(), dept_proj);

    std::map<std::string, std::size_t> sorted_dept(dept_freq.begin(), dept_freq.end());
    for (const auto& [dept, count] : sorted_dept) {
        std::cout << "  " << dept << ": " << count << "人\n";
    }

    // 勤続年数の相対度数
    print_subsection("勤続年数の相対度数");
    auto years_proj = [](const Employee& e) { return e.years; };
    auto years_rel = statcpp::relative_frequency(
        employees.begin(), employees.end(), years_proj);

    std::map<int, double> sorted_years(years_rel.begin(), years_rel.end());
    for (const auto& [years, freq] : sorted_years) {
        std::cout << "  " << years << "年: "
                  << (freq * 100) << "%\n";
    }

    // 勤続年数の度数表
    print_subsection("勤続年数の度数表");
    auto years_table = statcpp::frequency_table(
        employees.begin(), employees.end(), years_proj);

    std::cout << std::setw(10) << "勤続年数"
              << std::setw(10) << "人数"
              << std::setw(15) << "累積比率" << "\n";
    std::cout << std::string(35, '-') << "\n";

    for (const auto& entry : years_table.entries) {
        std::cout << std::setw(8) << entry.value << "年"
                  << std::setw(10) << entry.count
                  << std::setw(14) << (entry.cumulative_relative_frequency * 100) << "%\n";
    }
}

// ============================================================================
// 7. 文字列データの度数分布
// ============================================================================

/**
 * @brief 文字列データでの使用例
 */
void example_string_data() {
    print_section("7. 文字列データの度数分布");

    std::vector<std::string> blood_types = {
        "A", "B", "O", "AB", "A", "A", "O", "B", "A", "O",
        "A", "B", "A", "O", "A", "AB", "A", "O", "B", "A"
    };

    std::cout << "血液型データ:\n";
    for (const auto& bt : blood_types) std::cout << bt << " ";
    std::cout << "\n\n";

    auto table = statcpp::frequency_table(blood_types.begin(), blood_types.end());

    std::cout << "血液型分布:\n";
    std::cout << std::setw(8) << "血液型"
              << std::setw(10) << "人数"
              << std::setw(15) << "割合" << "\n";
    std::cout << std::string(33, '-') << "\n";

    for (const auto& entry : table.entries) {
        std::cout << std::setw(8) << entry.value
                  << std::setw(10) << entry.count
                  << std::setw(14) << (entry.relative_frequency * 100) << "%\n";
    }

    std::cout << "\n日本人の血液型分布（参考）:\n";
    std::cout << "  A型: 約40%, O型: 約30%, B型: 約20%, AB型: 約10%\n";
}

// ============================================================================
// まとめ
// ============================================================================

/**
 * @brief まとめを表示する
 */
void print_summary() {
    print_section("まとめ：度数分布の関数");

    std::cout << R"(
┌───────────────────────────────────┬────────────────────────────────────┐
│ 関数                              │ 説明                               │
├───────────────────────────────────┼────────────────────────────────────┤
│ frequency_table()                 │ 完全な度数表（ソート済み）         │
│                                   │ 度数、相対度数、累積度数を含む     │
├───────────────────────────────────┼────────────────────────────────────┤
│ frequency_count()                 │ 度数のみ（高速、unordered_map）    │
├───────────────────────────────────┼────────────────────────────────────┤
│ relative_frequency()              │ 相対度数（比率）                   │
├───────────────────────────────────┼────────────────────────────────────┤
│ cumulative_frequency()            │ 累積度数                           │
├───────────────────────────────────┼────────────────────────────────────┤
│ cumulative_relative_frequency()   │ 累積相対度数（ECDF）               │
└───────────────────────────────────┴────────────────────────────────────┘

【戻り値の構造体】
- frequency_table_result<T>:
    - entries: vector<frequency_entry<T>>
    - total_count: 総データ数

- frequency_entry<T>:
    - value: 値
    - count: 度数
    - relative_frequency: 相対度数
    - cumulative_count: 累積度数
    - cumulative_relative_frequency: 累積相対度数

【使い分け】
- 全情報が必要 → frequency_table()
- 度数のみ必要（高速） → frequency_count()
- 比率が必要 → relative_frequency()
- 累積情報が必要 → cumulative_* 系関数

【対応するデータ型】
- 数値型（int, double など）
- 文字列型（std::string）
- 比較演算子 < が定義された任意の型
)";
}

// ============================================================================
// メイン関数
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 各サンプルを実行
    example_frequency_table();
    example_frequency_count();
    example_relative_frequency();
    example_cumulative_frequency();
    example_cumulative_relative_frequency();
    example_projection();
    example_string_data();

    // まとめを表示
    print_summary();

    return 0;
}
