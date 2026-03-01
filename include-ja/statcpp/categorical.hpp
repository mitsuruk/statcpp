/**
 * @file categorical.hpp
 * @brief カテゴリカルデータの分析
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace statcpp {

// ============================================================================
// Contingency Table (Cross-tabulation)
// ============================================================================

/**
 * @brief 分割表（クロス集計表）の結果
 */
struct contingency_table_result {
    std::vector<std::vector<std::size_t>> table;  ///< 観測度数
    std::vector<std::size_t> row_totals;          ///< 行合計
    std::vector<std::size_t> col_totals;          ///< 列合計
    std::size_t total;                            ///< 総計
    std::size_t n_rows;                           ///< 行数
    std::size_t n_cols;                           ///< 列数
};

/**
 * @brief 分割表を作成
 *
 * 2つのカテゴリカル変数から分割表（クロス集計表）を作成します。
 *
 * @param row_data 行カテゴリ値（0から始まる整数）
 * @param col_data 列カテゴリ値（0から始まる整数）
 * @return 分割表の結果
 * @throws std::invalid_argument データ長が一致しない場合、または空データの場合
 */
inline contingency_table_result contingency_table(
    const std::vector<std::size_t>& row_data,
    const std::vector<std::size_t>& col_data)
{
    if (row_data.size() != col_data.size()) {
        throw std::invalid_argument("statcpp::contingency_table: data lengths must match");
    }
    if (row_data.empty()) {
        throw std::invalid_argument("statcpp::contingency_table: empty data");
    }

    // カテゴリ数を決定
    std::size_t max_row = *std::max_element(row_data.begin(), row_data.end());
    std::size_t max_col = *std::max_element(col_data.begin(), col_data.end());
    std::size_t n_rows = max_row + 1;
    std::size_t n_cols = max_col + 1;

    // 分割表を作成
    std::vector<std::vector<std::size_t>> table(n_rows, std::vector<std::size_t>(n_cols, 0));

    for (std::size_t i = 0; i < row_data.size(); ++i) {
        table[row_data[i]][col_data[i]]++;
    }

    // 周辺度数を計算
    std::vector<std::size_t> row_totals(n_rows, 0);
    std::vector<std::size_t> col_totals(n_cols, 0);
    std::size_t total = 0;

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            row_totals[i] += table[i][j];
            col_totals[j] += table[i][j];
            total += table[i][j];
        }
    }

    return {table, row_totals, col_totals, total, n_rows, n_cols};
}

// ============================================================================
// Odds Ratio
// ============================================================================

/**
 * @brief オッズ比の結果
 */
struct odds_ratio_result {
    double odds_ratio;            ///< オッズ比
    double log_odds_ratio;        ///< 対数オッズ比
    double se_log_odds_ratio;     ///< 対数オッズ比の標準誤差
    double ci_lower;              ///< 95%信頼区間下限
    double ci_upper;              ///< 95%信頼区間上限
};

/**
 * @brief 2x2分割表からオッズ比を計算
 *
 * オッズ比とその信頼区間を計算します。
 * オッズ比 = (a * d) / (b * c)
 *
 * @param table 2x2分割表 [[a, b], [c, d]]の形式
 *              - a: 曝露あり・疾患あり
 *              - b: 曝露あり・疾患なし
 *              - c: 曝露なし・疾患あり
 *              - d: 曝露なし・疾患なし
 * @return オッズ比の計算結果
 * @throws std::invalid_argument テーブルが2x2でない場合、またはセル度数が0の場合
 *
 * @note 現在、セル度数が0の場合は例外をスローします。将来のバージョンでは、
 *       ゼロセルテーブルを扱うためのGart-Zweifel連続補正（全セルに+0.5）を
 *       オプション引数で選択可能にする予定です。
 */
inline odds_ratio_result odds_ratio(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::odds_ratio: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    if (a == 0 || b == 0 || c == 0 || d == 0) {
        throw std::invalid_argument("statcpp::odds_ratio: zero cell count not allowed");
    }

    double or_val = (a * d) / (b * c);
    double log_or = std::log(or_val);
    double se_log_or = std::sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d);

    // 95% 信頼区間
    double z = 1.96;
    double ci_lower = std::exp(log_or - z * se_log_or);
    double ci_upper = std::exp(log_or + z * se_log_or);

    return {or_val, log_or, se_log_or, ci_lower, ci_upper};
}

/**
 * @brief 2x2分割表からオッズ比を計算（セル値を直接指定）
 *
 * @param a 曝露あり・疾患あり
 * @param b 曝露あり・疾患なし
 * @param c 曝露なし・疾患あり
 * @param d 曝露なし・疾患なし
 * @return オッズ比の計算結果
 * @throws std::invalid_argument セル度数が0の場合
 */
inline odds_ratio_result odds_ratio(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return odds_ratio({{a, b}, {c, d}});
}

// ============================================================================
// Relative Risk (Risk Ratio)
// ============================================================================

/**
 * @brief 相対リスクの結果
 */
struct relative_risk_result {
    double relative_risk;         ///< 相対リスク
    double log_relative_risk;     ///< 対数相対リスク
    double se_log_relative_risk;  ///< 対数相対リスクの標準誤差
    double ci_lower;              ///< 95%信頼区間下限
    double ci_upper;              ///< 95%信頼区間上限
};

/**
 * @brief 2x2分割表から相対リスク（リスク比）を計算
 *
 * 相対リスクとその信頼区間を計算します。
 * 相対リスク = (a/(a+b)) / (c/(c+d))
 *
 * 標準誤差は対数スケールで計算されます（Greenland-Robins 法）：
 * SE(log RR) = sqrt((1-p1)/(n1*p1) + (1-p0)/(n0*p0))
 *            = sqrt((1-p1)/a + (1-p0)/c)
 * ここで p1 = a/(a+b), p0 = c/(c+d)
 *
 * @param table 2x2分割表 [[a, b], [c, d]]の形式
 *              - a: 曝露あり・疾患あり
 *              - b: 曝露あり・疾患なし
 *              - c: 曝露なし・疾患あり
 *              - d: 曝露なし・疾患なし
 * @return 相対リスクの計算結果
 * @throws std::invalid_argument テーブルが2x2でない場合、行合計が0の場合、またはリスクが0の場合
 */
inline relative_risk_result relative_risk(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::relative_risk: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    double n1 = a + b;  // 曝露群のサイズ
    double n0 = c + d;  // 非曝露群のサイズ

    if (n1 == 0 || n0 == 0) {
        throw std::invalid_argument("statcpp::relative_risk: zero row total");
    }
    if (a == 0 || c == 0) {
        throw std::invalid_argument("statcpp::relative_risk: zero risk in a group");
    }

    double risk1 = a / n1;  // 曝露群のリスク
    double risk0 = c / n0;  // 非曝露群のリスク

    double rr = risk1 / risk0;
    double log_rr = std::log(rr);

    // 標準誤差（Greenland-Robins formula）
    double se_log_rr = std::sqrt((1.0 - risk1)/(a) + (1.0 - risk0)/(c));

    // 95% 信頼区間
    double z = 1.96;
    double ci_lower = std::exp(log_rr - z * se_log_rr);
    double ci_upper = std::exp(log_rr + z * se_log_rr);

    return {rr, log_rr, se_log_rr, ci_lower, ci_upper};
}

/**
 * @brief 2x2分割表から相対リスクを計算（セル値を直接指定）
 *
 * @param a 曝露あり・疾患あり
 * @param b 曝露あり・疾患なし
 * @param c 曝露なし・疾患あり
 * @param d 曝露なし・疾患なし
 * @return 相対リスクの計算結果
 * @throws std::invalid_argument 行合計が0の場合、またはリスクが0の場合
 */
inline relative_risk_result relative_risk(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return relative_risk({{a, b}, {c, d}});
}

// ============================================================================
// Risk Difference (Attributable Risk)
// ============================================================================

/**
 * @brief リスク差の結果
 */
struct risk_difference_result {
    double risk_difference;       ///< リスク差（寄与危険度）
    double se;                    ///< 標準誤差
    double ci_lower;              ///< 95%信頼区間下限
    double ci_upper;              ///< 95%信頼区間上限
};

/**
 * @brief 2x2分割表からリスク差を計算
 *
 * リスク差（寄与危険度）とその信頼区間を計算します。
 * リスク差 = (a/(a+b)) - (c/(c+d))
 *
 * @param table 2x2分割表 [[a, b], [c, d]]の形式
 * @return リスク差の計算結果
 * @throws std::invalid_argument テーブルが2x2でない場合、または行合計が0の場合
 */
inline risk_difference_result risk_difference(const std::vector<std::vector<std::size_t>>& table)
{
    if (table.size() != 2 || table[0].size() != 2 || table[1].size() != 2) {
        throw std::invalid_argument("statcpp::risk_difference: table must be 2x2");
    }

    double a = static_cast<double>(table[0][0]);
    double b = static_cast<double>(table[0][1]);
    double c = static_cast<double>(table[1][0]);
    double d = static_cast<double>(table[1][1]);

    double n1 = a + b;
    double n0 = c + d;

    if (n1 == 0 || n0 == 0) {
        throw std::invalid_argument("statcpp::risk_difference: zero row total");
    }

    double risk1 = a / n1;
    double risk0 = c / n0;

    double rd = risk1 - risk0;

    // 標準誤差
    double se = std::sqrt(risk1 * (1.0 - risk1) / n1 + risk0 * (1.0 - risk0) / n0);

    // 95% 信頼区間
    double z = 1.96;
    double ci_lower = rd - z * se;
    double ci_upper = rd + z * se;

    return {rd, se, ci_lower, ci_upper};
}

/**
 * @brief 2x2分割表からリスク差を計算（セル値を直接指定）
 *
 * @param a 曝露あり・疾患あり
 * @param b 曝露あり・疾患なし
 * @param c 曝露なし・疾患あり
 * @param d 曝露なし・疾患なし
 * @return リスク差の計算結果
 * @throws std::invalid_argument 行合計が0の場合
 */
inline risk_difference_result risk_difference(std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    return risk_difference({{a, b}, {c, d}});
}

// ============================================================================
// Number Needed to Treat (NNT)
// ============================================================================

/**
 * @brief 治療必要数 (NNT) を計算
 *
 * リスク差の逆数として計算されます。
 * ある結果を1つ防ぐために治療が必要な患者数を表します。
 *
 * @param table 2x2分割表
 * @return 治療必要数
 * @throws std::invalid_argument リスク差が0の場合
 */
inline double number_needed_to_treat(const std::vector<std::vector<std::size_t>>& table)
{
    auto rd = risk_difference(table);
    if (rd.risk_difference == 0.0) {
        throw std::invalid_argument("statcpp::number_needed_to_treat: risk difference is zero");
    }
    return 1.0 / std::abs(rd.risk_difference);
}

} // namespace statcpp
