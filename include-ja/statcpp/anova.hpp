/**
 * @file anova.hpp
 * @brief 分散分析（ANOVA）
 *
 * 一元配置分散分析、二元配置分散分析、共分散分析、事後検定などの
 * 分散分析関連関数を提供します。
 */

#pragma once

#include "statcpp/continuous_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// ANOVA Result Structures
// ============================================================================

/**
 * @brief ANOVA表の一行を表す構造体
 *
 * 分散分析表の各変動要因（群間、群内など）に対応する統計量を保持します。
 */
struct anova_row {
    std::string source;     ///< 変動要因の名称
    double ss;              ///< 平方和 (Sum of Squares)
    double df;              ///< 自由度
    double ms;              ///< 平均平方 (Mean Square)
    double f_statistic;     ///< F統計量
    double p_value;         ///< p値
};

/**
 * @brief 一元配置分散分析の結果を格納する構造体
 *
 * 一元配置ANOVAの全ての結果（群間・群内変動、各種統計量、群情報）を保持します。
 */
struct one_way_anova_result {
    anova_row between;      ///< 群間変動
    anova_row within;       ///< 群内変動（残差）
    double ss_total;        ///< 総平方和
    double df_total;        ///< 総自由度
    std::size_t n_groups;   ///< 群数
    std::size_t n_total;    ///< 総観測数
    double grand_mean;      ///< 総平均
    std::vector<double> group_means;       ///< 各群の平均
    std::vector<std::size_t> group_sizes;  ///< 各群のサイズ
};

/**
 * @brief 二元配置分散分析の結果を格納する構造体
 *
 * 二元配置ANOVAの全ての結果（要因A・B効果、交互作用、誤差など）を保持します。
 */
struct two_way_anova_result {
    anova_row factor_a;         ///< 要因Aの効果
    anova_row factor_b;         ///< 要因Bの効果
    anova_row interaction;      ///< 交互作用
    anova_row error;            ///< 誤差（残差）
    double ss_total;            ///< 総平方和
    double df_total;            ///< 総自由度
    std::size_t levels_a;       ///< 要因Aの水準数
    std::size_t levels_b;       ///< 要因Bの水準数
    std::size_t n_total;        ///< 総観測数
    double grand_mean;          ///< 総平均
};

/**
 * @brief 事後比較の個別ペア比較結果を格納する構造体
 *
 * 2群間の事後比較における統計量と判定結果を保持します。
 */
struct posthoc_comparison {
    std::size_t group1;     ///< 群1のインデックス
    std::size_t group2;     ///< 群2のインデックス
    double mean_diff;       ///< 平均値の差
    double se;              ///< 標準誤差
    double statistic;       ///< 検定統計量
    double p_value;         ///< p値
    double lower;           ///< 信頼区間下限
    double upper;           ///< 信頼区間上限
    bool significant;       ///< 有意かどうか
};

/**
 * @brief 事後比較の結果一式を格納する構造体
 *
 * 事後検定の手法名、全ての比較結果、および検定に使用したパラメータを保持します。
 */
struct posthoc_result {
    std::string method;                          ///< 手法名
    std::vector<posthoc_comparison> comparisons; ///< 全ての比較
    double alpha;                                ///< 有意水準
    double mse;                                  ///< 誤差平均平方
    double df_error;                             ///< 誤差自由度
};

// ============================================================================
// One-Way ANOVA (一元配置分散分析)
// ============================================================================

/**
 * @brief 一元配置分散分析を実行する
 *
 * 複数の群間で平均値に有意な差があるかを検定します。
 * F検定を用いて群間変動と群内変動を比較します。
 *
 * @param groups 各群のデータを格納したベクトルのベクトル
 * @return one_way_anova_result 分散分析の結果
 * @throws std::invalid_argument 群数が2未満の場合
 * @throws std::invalid_argument 空の群が存在する場合
 * @throws std::invalid_argument 総観測数が群数以下の場合
 */
inline one_way_anova_result one_way_anova(const std::vector<std::vector<double>>& groups)
{
    std::size_t k = groups.size();  // 群数
    if (k < 2) {
        throw std::invalid_argument("statcpp::one_way_anova: need at least 2 groups");
    }

    // 各群のサイズと平均を計算
    std::vector<std::size_t> group_sizes(k);
    std::vector<double> group_means(k);
    std::size_t n_total = 0;
    double grand_sum = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        if (groups[i].empty()) {
            throw std::invalid_argument("statcpp::one_way_anova: empty group detected");
        }
        group_sizes[i] = groups[i].size();
        n_total += group_sizes[i];
        double group_sum = std::accumulate(groups[i].begin(), groups[i].end(), 0.0);
        group_means[i] = group_sum / static_cast<double>(group_sizes[i]);
        grand_sum += group_sum;
    }

    if (n_total <= k) {
        throw std::invalid_argument("statcpp::one_way_anova: need more observations than groups");
    }

    double grand_mean = grand_sum / static_cast<double>(n_total);

    // 平方和を計算
    double ss_between = 0.0;  // 群間平方和
    double ss_within = 0.0;   // 群内平方和

    for (std::size_t i = 0; i < k; ++i) {
        double diff = group_means[i] - grand_mean;
        ss_between += static_cast<double>(group_sizes[i]) * diff * diff;

        for (double x : groups[i]) {
            double diff_within = x - group_means[i];
            ss_within += diff_within * diff_within;
        }
    }

    double ss_total = ss_between + ss_within;

    // 自由度
    double df_between = static_cast<double>(k - 1);
    double df_within = static_cast<double>(n_total - k);
    double df_total = static_cast<double>(n_total - 1);

    // 平均平方
    double ms_between = ss_between / df_between;
    double ms_within = ss_within / df_within;

    // F統計量とp値（退化ケース ms_within == 0 の処理）
    double f_statistic;
    double p_value;
    if (ms_within == 0.0) {
        f_statistic = (ms_between == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        p_value = (ms_between == 0.0) ? 1.0 : 0.0;
    } else {
        f_statistic = ms_between / ms_within;
        p_value = 1.0 - f_cdf(f_statistic, df_between, df_within);
    }

    anova_row between{"Between Groups", ss_between, df_between, ms_between, f_statistic, p_value};
    anova_row within{"Within Groups", ss_within, df_within, ms_within, 0.0, 0.0};

    return {between, within, ss_total, df_total, k, n_total, grand_mean, group_means, group_sizes};
}

// ============================================================================
// Two-Way ANOVA (二元配置分散分析)
// ============================================================================

/**
 * @brief 二元配置分散分析（繰り返しあり）を実行する
 *
 * 2つの要因（A, B）とその交互作用が従属変数に与える効果を検定します。
 * 等サイズのセルを前提とします。
 *
 * @param data 3次元データ配列。data[i][j]は要因A=i、要因B=jでの観測値のベクトル
 * @return two_way_anova_result 分散分析の結果
 * @throws std::invalid_argument 要因Aの水準数が2未満の場合
 * @throws std::invalid_argument 要因Bの水準数が2未満の場合
 * @throws std::invalid_argument 要因Bの水準数が一貫していない場合
 * @throws std::invalid_argument セルサイズが不均等な場合
 * @throws std::invalid_argument 空のセルが存在する場合
 */
inline two_way_anova_result two_way_anova(
    const std::vector<std::vector<std::vector<double>>>& data)
{
    std::size_t a = data.size();  // 要因Aの水準数
    if (a < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: need at least 2 levels for factor A");
    }

    std::size_t b = data[0].size();  // 要因Bの水準数
    if (b < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: need at least 2 levels for factor B");
    }

    // 全ての水準で同じ繰り返し数か確認
    std::size_t n_rep = data[0][0].size();  // 繰り返し数
    if (n_rep < 2) {
        throw std::invalid_argument("statcpp::two_way_anova: at least 2 replications per cell are required (n_rep >= 2)");
    }
    std::size_t n_total = 0;
    double grand_sum = 0.0;

    for (std::size_t i = 0; i < a; ++i) {
        if (data[i].size() != b) {
            throw std::invalid_argument("statcpp::two_way_anova: inconsistent number of levels for factor B");
        }
        for (std::size_t j = 0; j < b; ++j) {
            if (data[i][j].size() != n_rep) {
                throw std::invalid_argument("statcpp::two_way_anova: unequal cell sizes not supported");
            }
            n_total += n_rep;
            for (double x : data[i][j]) {
                grand_sum += x;
            }
        }
    }

    double grand_mean = grand_sum / static_cast<double>(n_total);
    double n_rep_d = static_cast<double>(n_rep);
    double a_d = static_cast<double>(a);
    double b_d = static_cast<double>(b);

    // 各水準の平均を計算
    std::vector<double> mean_a(a, 0.0);  // 要因Aの各水準の平均
    std::vector<double> mean_b(b, 0.0);  // 要因Bの各水準の平均
    std::vector<std::vector<double>> mean_ab(a, std::vector<double>(b, 0.0));  // セル平均

    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            double cell_sum = std::accumulate(data[i][j].begin(), data[i][j].end(), 0.0);
            mean_ab[i][j] = cell_sum / n_rep_d;
            mean_a[i] += cell_sum;
            mean_b[j] += cell_sum;
        }
        mean_a[i] /= (b_d * n_rep_d);
    }
    for (std::size_t j = 0; j < b; ++j) {
        mean_b[j] /= (a_d * n_rep_d);
    }

    // 平方和を計算
    double ss_a = 0.0;      // 要因Aの平方和
    double ss_b = 0.0;      // 要因Bの平方和
    double ss_ab = 0.0;     // 交互作用の平方和
    double ss_error = 0.0;  // 誤差平方和

    for (std::size_t i = 0; i < a; ++i) {
        double diff_a = mean_a[i] - grand_mean;
        ss_a += diff_a * diff_a;
    }
    ss_a *= b_d * n_rep_d;

    for (std::size_t j = 0; j < b; ++j) {
        double diff_b = mean_b[j] - grand_mean;
        ss_b += diff_b * diff_b;
    }
    ss_b *= a_d * n_rep_d;

    for (std::size_t i = 0; i < a; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
            double interaction = mean_ab[i][j] - mean_a[i] - mean_b[j] + grand_mean;
            ss_ab += interaction * interaction;

            for (double x : data[i][j]) {
                double error = x - mean_ab[i][j];
                ss_error += error * error;
            }
        }
    }
    ss_ab *= n_rep_d;

    double ss_total = ss_a + ss_b + ss_ab + ss_error;

    // 自由度
    double df_a = a_d - 1.0;
    double df_b = b_d - 1.0;
    double df_ab = df_a * df_b;
    double df_error = static_cast<double>(n_total) - a_d * b_d;
    double df_total = static_cast<double>(n_total) - 1.0;

    // 平均平方
    double ms_a = ss_a / df_a;
    double ms_b = ss_b / df_b;
    double ms_ab = ss_ab / df_ab;
    double ms_error = ss_error / df_error;

    // F統計量とp値
    double f_a = ms_a / ms_error;
    double f_b = ms_b / ms_error;
    double f_ab = ms_ab / ms_error;

    double p_a = 1.0 - f_cdf(f_a, df_a, df_error);
    double p_b = 1.0 - f_cdf(f_b, df_b, df_error);
    double p_ab = 1.0 - f_cdf(f_ab, df_ab, df_error);

    anova_row factor_a{"Factor A", ss_a, df_a, ms_a, f_a, p_a};
    anova_row factor_b{"Factor B", ss_b, df_b, ms_b, f_b, p_b};
    anova_row interaction{"A x B", ss_ab, df_ab, ms_ab, f_ab, p_ab};
    anova_row error{"Error", ss_error, df_error, ms_error, 0.0, 0.0};

    return {factor_a, factor_b, interaction, error,
            ss_total, df_total, a, b, n_total, grand_mean};
}

// ============================================================================
// Post-hoc Comparisons (事後比較)
// ============================================================================

/**
 * @brief Tukeyの正直有意差（HSD）検定を実行する
 *
 * Studentized range 分布（スチューデント化された範囲分布）を使用して、
 * 全ての群ペア間の比較を行います（不等サンプルサイズにはTukey-Kramer法）。
 *
 * q統計量は |mean_i - mean_j| / SE として計算され、
 * SE = sqrt(MSE/2 * (1/n_i + 1/n_j)) です。p値はk群・df_error自由度の
 * Studentized range 分布から求められます。
 *
 * @param anova_result 一元配置分散分析の結果
 * @param groups 各群のデータ（現在未使用。API互換性のため保持。
 *               将来の入力検証等での使用を想定）
 * @param alpha 有意水準（デフォルト: 0.05）
 * @return posthoc_result 事後比較の結果（statistic フィールドはq統計量）
 * @throws std::invalid_argument alphaが(0, 1)の範囲外の場合
 */
inline posthoc_result tukey_hsd(const one_way_anova_result& anova_result,
                                 const std::vector<std::vector<double>>& groups,
                                 double alpha = 0.05)
{
    (void)groups;  // 現在未使用。全ての統計量は anova_result から導出

    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::tukey_hsd: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;
    double k_d = static_cast<double>(k);

    double q_crit = studentized_range_quantile(1.0 - alpha, k_d, df_error);

    std::vector<posthoc_comparison> comparisons;

    // 全てのペアを比較
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            // Tukey-Kramer 標準誤差
            double se = std::sqrt(mse * 0.5 * (1.0 / n_i + 1.0 / n_j));

            double q_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // 退化ケース: 群内分散がゼロ
                if (mean_diff == 0.0) {
                    q_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    q_stat = std::numeric_limits<double>::infinity();
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                // q統計量（スチューデント化された範囲統計量）
                q_stat = std::abs(mean_diff) / se;

                // Studentized range 分布からのp値
                p_value = 1.0 - studentized_range_cdf(q_stat, k_d, df_error);
                p_value = std::max(0.0, std::min(1.0, p_value));

                double margin = q_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (q_stat > q_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, q_stat, p_value, lower, upper, significant});
        }
    }

    return {"Tukey HSD", comparisons, alpha, mse, df_error};
}

/**
 * @brief Bonferroni法による多重比較を実行する
 *
 * 一元配置分散分析後の事後比較として、全ての群ペア間で
 * Bonferroni補正を適用した多重比較を行います。
 *
 * @param anova_result 一元配置分散分析の結果
 * @param alpha 有意水準（デフォルト: 0.05）
 * @return posthoc_result 事後比較の結果
 * @throws std::invalid_argument alphaが(0, 1)の範囲外の場合
 */
inline posthoc_result bonferroni_posthoc(const one_way_anova_result& anova_result,
                                          double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::bonferroni_posthoc: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;

    double n_comparisons = static_cast<double>(k * (k - 1) / 2);
    double alpha_adj = alpha / n_comparisons;
    double t_crit = t_quantile(1.0 - alpha_adj / 2.0, df_error);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_j));

            double t_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // 退化ケース: 群内分散がゼロ
                if (mean_diff == 0.0) {
                    t_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                t_stat = mean_diff / se;

                p_value = 2.0 * (1.0 - t_cdf(std::abs(t_stat), df_error));
                p_value = std::min(1.0, p_value * n_comparisons);

                double margin = t_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (std::abs(t_stat) > t_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, t_stat, p_value, lower, upper, significant});
        }
    }

    return {"Bonferroni", comparisons, alpha, mse, df_error};
}

/**
 * @brief Dunnett法による対照群との多重比較を実行する
 *
 * 一元配置分散分析後の事後比較として、指定した対照群と
 * 他の全ての群との比較を行います。Bonferroni近似を使用します。
 *
 * @param anova_result 一元配置分散分析の結果
 * @param control_group 対照群のインデックス（デフォルト: 0）
 * @param alpha 有意水準（デフォルト: 0.05）
 * @return posthoc_result 事後比較の結果
 * @throws std::invalid_argument alphaが(0, 1)の範囲外の場合
 * @throws std::invalid_argument control_groupが無効なインデックスの場合
 */
inline posthoc_result dunnett_posthoc(const one_way_anova_result& anova_result,
                                       std::size_t control_group = 0,
                                       double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::dunnett_posthoc: alpha must be in (0, 1)");
    }
    if (control_group >= anova_result.n_groups) {
        throw std::invalid_argument("statcpp::dunnett_posthoc: invalid control group index");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_error = anova_result.within.df;

    double n_comparisons = static_cast<double>(k - 1);
    double alpha_adj = alpha / n_comparisons;  // Bonferroni近似
    double t_crit = t_quantile(1.0 - alpha_adj / 2.0, df_error);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        if (i == control_group) continue;

        double mean_diff = anova_result.group_means[i] - anova_result.group_means[control_group];
        double n_i = static_cast<double>(anova_result.group_sizes[i]);
        double n_c = static_cast<double>(anova_result.group_sizes[control_group]);

        double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_c));

        double t_stat, p_value, lower, upper;
        bool significant;

        if (se == 0.0) {
            // 退化ケース: 群内分散がゼロ
            if (mean_diff == 0.0) {
                t_stat = 0.0; p_value = 1.0;
                lower = 0.0; upper = 0.0; significant = false;
            } else {
                t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                p_value = 0.0;
                lower = mean_diff; upper = mean_diff; significant = true;
            }
        } else {
            t_stat = mean_diff / se;

            p_value = 2.0 * (1.0 - t_cdf(std::abs(t_stat), df_error));
            p_value = std::min(1.0, p_value * n_comparisons);

            double margin = t_crit * se;
            lower = mean_diff - margin;
            upper = mean_diff + margin;
            significant = (std::abs(t_stat) > t_crit);
        }

        comparisons.push_back({i, control_group, mean_diff, se, t_stat, p_value, lower, upper, significant});
    }

    return {"Dunnett (Bonferroni approximation)", comparisons, alpha, mse, df_error};
}

/**
 * @brief Scheffe法による多重比較を実行する
 *
 * 一元配置分散分析後の事後比較として、全ての群ペア間で
 * Scheffe法を用いた多重比較を行います。最も保守的な手法です。
 *
 * Scheffe法は任意の線形対比（コントラスト）に対して有効であり、
 * ペア比較だけでなく複雑な対比にも適用可能です。
 * F統計量は t² / (k-1) として計算され、F(k-1, df_error) 分布と比較されます。
 * p値は 1 - F_cdf(F_s, k-1, df_error) として計算されます。
 *
 * @param anova_result 一元配置分散分析の結果
 * @param alpha 有意水準（デフォルト: 0.05）
 * @return posthoc_result 事後比較の結果
 * @throws std::invalid_argument alphaが(0, 1)の範囲外の場合
 */
inline posthoc_result scheffe_posthoc(const one_way_anova_result& anova_result,
                                       double alpha = 0.05)
{
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::scheffe_posthoc: alpha must be in (0, 1)");
    }

    std::size_t k = anova_result.n_groups;
    double mse = anova_result.within.ms;
    double df_between = anova_result.between.df;
    double df_error = anova_result.within.df;

    // Scheffe の臨界値
    double f_crit = f_quantile(1.0 - alpha, df_between, df_error);
    double scheffe_crit = std::sqrt(df_between * f_crit);

    std::vector<posthoc_comparison> comparisons;

    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            double mean_diff = anova_result.group_means[i] - anova_result.group_means[j];
            double n_i = static_cast<double>(anova_result.group_sizes[i]);
            double n_j = static_cast<double>(anova_result.group_sizes[j]);

            double se = std::sqrt(mse * (1.0 / n_i + 1.0 / n_j));

            double t_stat, p_value, lower, upper;
            bool significant;

            if (se == 0.0) {
                // 退化ケース: 群内分散がゼロ
                if (mean_diff == 0.0) {
                    t_stat = 0.0; p_value = 1.0;
                    lower = 0.0; upper = 0.0; significant = false;
                } else {
                    t_stat = std::copysign(std::numeric_limits<double>::infinity(), mean_diff);
                    p_value = 0.0;
                    lower = mean_diff; upper = mean_diff; significant = true;
                }
            } else {
                t_stat = mean_diff / se;

                // ScheffeのF統計量
                double f_stat = (t_stat * t_stat) / df_between;
                p_value = 1.0 - f_cdf(f_stat, df_between, df_error);

                double margin = scheffe_crit * se;
                lower = mean_diff - margin;
                upper = mean_diff + margin;
                significant = (std::abs(t_stat) > scheffe_crit);
            }

            comparisons.push_back({i, j, mean_diff, se, t_stat, p_value, lower, upper, significant});
        }
    }

    return {"Scheffe", comparisons, alpha, mse, df_error};
}

// ============================================================================
// ANCOVA (Analysis of Covariance / 共分散分析)
// ============================================================================

/**
 * @brief 共分散分析（ANCOVA）の結果を格納する構造体
 *
 * 一元配置共分散分析の全ての結果（共変量効果、処理効果、誤差など）を保持します。
 */
struct ancova_result {
    double ss_covariate;        ///< 共変量の平方和
    double ss_treatment;        ///< 処理効果の平方和
    double ss_error;            ///< 誤差平方和
    double df_covariate;        ///< 共変量の自由度
    double df_treatment;        ///< 処理の自由度
    double df_error;            ///< 誤差の自由度
    double ms_covariate;        ///< 共変量の平均平方
    double ms_treatment;        ///< 処理の平均平方
    double ms_error;            ///< 誤差の平均平方
    double f_covariate;         ///< 共変量のF統計量
    double f_treatment;         ///< 処理のF統計量
    double p_covariate;         ///< 共変量のp値
    double p_treatment;         ///< 処理のp値
    std::vector<double> adjusted_means;  ///< 調整済み群平均
};

/**
 * @brief 一元配置共分散分析を実行する
 *
 * 共変量の影響を統制した上で、群間の平均値差を検定します。
 * 各観測値は従属変数(y)と共変量(x)のペアで指定します。
 *
 * @param groups 各群のデータ。groups[i]は群iの(y, x)ペアのベクトル
 * @return ancova_result 共分散分析の結果
 * @throws std::invalid_argument 群数が2未満の場合
 * @throws std::invalid_argument 空の群が存在する場合
 * @throws std::invalid_argument 観測数が不十分な場合
 */
inline ancova_result one_way_ancova(
    const std::vector<std::vector<std::pair<double, double>>>& groups)
{
    std::size_t k = groups.size();
    if (k < 2) {
        throw std::invalid_argument("statcpp::one_way_ancova: need at least 2 groups");
    }

    // データを整理
    std::size_t n_total = 0;
    for (const auto& g : groups) {
        if (g.empty()) {
            throw std::invalid_argument("statcpp::one_way_ancova: empty group detected");
        }
        n_total += g.size();
    }

    if (n_total <= k + 1) {
        throw std::invalid_argument("statcpp::one_way_ancova: insufficient observations");
    }

    // 全体の平均を計算
    double sum_y = 0.0, sum_x = 0.0;
    for (const auto& g : groups) {
        for (const auto& pair : g) {
            sum_y += pair.first;
            sum_x += pair.second;
        }
    }
    double grand_mean_y = sum_y / static_cast<double>(n_total);
    double grand_mean_x = sum_x / static_cast<double>(n_total);

    // 各群の平均を計算
    std::vector<double> group_mean_y(k);
    std::vector<double> group_mean_x(k);
    std::vector<std::size_t> group_sizes(k);

    for (std::size_t i = 0; i < k; ++i) {
        group_sizes[i] = groups[i].size();
        double sy = 0.0, sx = 0.0;
        for (const auto& pair : groups[i]) {
            sy += pair.first;
            sx += pair.second;
        }
        group_mean_y[i] = sy / static_cast<double>(group_sizes[i]);
        group_mean_x[i] = sx / static_cast<double>(group_sizes[i]);
    }

    // 総平方和と積和を計算
    double sst_y = 0.0;     // 総平方和（y）
    double sst_x = 0.0;     // 総平方和（x）
    double spt = 0.0;       // 総積和
    double ssw_y = 0.0;     // 群内平方和（y）
    double ssw_x = 0.0;     // 群内平方和（x）
    double spw = 0.0;       // 群内積和

    for (std::size_t i = 0; i < k; ++i) {
        for (const auto& pair : groups[i]) {
            double dy_t = pair.first - grand_mean_y;
            double dx_t = pair.second - grand_mean_x;
            sst_y += dy_t * dy_t;
            sst_x += dx_t * dx_t;
            spt += dy_t * dx_t;

            double dy_w = pair.first - group_mean_y[i];
            double dx_w = pair.second - group_mean_x[i];
            ssw_y += dy_w * dy_w;
            ssw_x += dx_w * dx_w;
            spw += dy_w * dx_w;
        }
    }

    // 共通の回帰係数（群内）
    double b_within = (ssw_x > 0.0) ? spw / ssw_x : 0.0;

    // 平方和の計算
    double ss_error = ssw_y - b_within * spw;  // 調整済み群内平方和
    double ss_covariate = b_within * spw;       // 共変量による説明された分散

    // 全体の回帰係数
    double b_total = (sst_x > 0.0) ? spt / sst_x : 0.0;
    double ss_total_adj = sst_y - b_total * spt;

    // 処理効果の平方和
    double ss_treatment = ss_total_adj - ss_error;

    // 自由度
    double df_covariate = 1.0;
    double df_treatment = static_cast<double>(k - 1);
    double df_error = static_cast<double>(n_total - k - 1);

    // 平均平方
    double ms_covariate = ss_covariate / df_covariate;
    double ms_treatment = ss_treatment / df_treatment;
    double ms_error = ss_error / df_error;

    // F統計量
    double f_covariate = ms_covariate / ms_error;
    double f_treatment = ms_treatment / ms_error;

    // p値
    double p_covariate = 1.0 - f_cdf(f_covariate, df_covariate, df_error);
    double p_treatment = 1.0 - f_cdf(f_treatment, df_treatment, df_error);

    // 調整済み平均
    std::vector<double> adjusted_means(k);
    for (std::size_t i = 0; i < k; ++i) {
        adjusted_means[i] = group_mean_y[i] - b_within * (group_mean_x[i] - grand_mean_x);
    }

    return {
        ss_covariate, ss_treatment, ss_error,
        df_covariate, df_treatment, df_error,
        ms_covariate, ms_treatment, ms_error,
        f_covariate, f_treatment,
        p_covariate, p_treatment,
        adjusted_means
    };
}

// ============================================================================
// Effect Size for ANOVA (ANOVAの効果量)
// ============================================================================

/**
 * @brief 一元配置ANOVAのEta-squared（イータ二乗）を計算する
 *
 * 総変動のうち群間変動が占める割合を示す効果量指標です。
 *
 * @param result 一元配置分散分析の結果
 * @return double Eta-squared値（0から1の範囲）
 */
inline double eta_squared(const one_way_anova_result& result)
{
    if (result.ss_total == 0.0) {
        return 0.0;
    }
    return result.between.ss / result.ss_total;
}

/**
 * @brief 二元配置ANOVAの要因Aに対するPartial eta-squaredを計算する
 *
 * 要因Aの効果と誤差の合計に対する要因Aの効果の割合を示します。
 *
 * @param result 二元配置分散分析の結果
 * @return double 要因AのPartial eta-squared値
 */
inline double partial_eta_squared_a(const two_way_anova_result& result)
{
    double denom = result.factor_a.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.factor_a.ss / denom;
}

/**
 * @brief 二元配置ANOVAの要因Bに対するPartial eta-squaredを計算する
 *
 * 要因Bの効果と誤差の合計に対する要因Bの効果の割合を示します。
 *
 * @param result 二元配置分散分析の結果
 * @return double 要因BのPartial eta-squared値
 */
inline double partial_eta_squared_b(const two_way_anova_result& result)
{
    double denom = result.factor_b.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.factor_b.ss / denom;
}

/**
 * @brief 二元配置ANOVAの交互作用に対するPartial eta-squaredを計算する
 *
 * 交互作用の効果と誤差の合計に対する交互作用の効果の割合を示します。
 *
 * @param result 二元配置分散分析の結果
 * @return double 交互作用のPartial eta-squared値
 */
inline double partial_eta_squared_interaction(const two_way_anova_result& result)
{
    double denom = result.interaction.ss + result.error.ss;
    if (denom == 0.0) {
        return 0.0;
    }
    return result.interaction.ss / denom;
}

/**
 * @brief 一元配置ANOVAのOmega-squared（オメガ二乗）を計算する
 *
 * Eta-squaredよりも偏りの少ない効果量推定値です。
 * 母集団における効果量の推定に適しています。
 *
 * @param result 一元配置分散分析の結果
 * @return double Omega-squared値
 */
inline double omega_squared(const one_way_anova_result& result)
{
    double ss_between = result.between.ss;
    double df_between = result.between.df;
    double ms_within = result.within.ms;
    double ss_total = result.ss_total;

    double denom = ss_total + ms_within;
    if (denom == 0.0) {
        return 0.0;
    }
    return (ss_between - df_between * ms_within) / denom;
}

/**
 * @brief 一元配置ANOVAのCohen's fを計算する
 *
 * 標準化された効果量指標で、0.10が小、0.25が中、0.40が大の効果を示します。
 *
 * @param result 一元配置分散分析の結果
 * @return double Cohen's f値（eta_squaredが1.0の場合は無限大）
 */
inline double cohens_f(const one_way_anova_result& result)
{
    double eta_sq = eta_squared(result);
    if (eta_sq >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(eta_sq / (1.0 - eta_sq));
}

} // namespace statcpp
