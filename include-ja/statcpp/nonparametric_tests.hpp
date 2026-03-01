/**
 * @file nonparametric_tests.hpp
 * @brief ノンパラメトリック検定関数
 *
 * Wilcoxon検定、Mann-Whitney U検定、Kruskal-Wallis検定など、
 * 分布を仮定しない統計検定を提供します。
 * 正規性検定（Shapiro-Wilk検定、Kolmogorov-Smirnov検定）や
 * 等分散性検定（Levene検定、Bartlett検定）も含みます。
 *
 * @note タイ（同順位）検出: タイの検出には浮動小数点の完全一致（operator==）を
 *       使用します。これは観測データ（整数、固定精度の小数など）においてはビット
 *       表現が一致するため適切です。R と同じ挙動です。浮動小数点演算の結果
 *       （例: sqrt(2)*sqrt(2) vs 2.0）を直接入力した場合、タイとして認識
 *       されない可能性があります。
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/order_statistics.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/discrete_distributions.hpp"
#include "statcpp/parametric_tests.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Helper: Compute Ranks with Tie Handling
// ============================================================================

/**
 * @brief 同順位（タイ）を考慮した順位を計算する
 *
 * 入力データに対して順位を割り当てます。同じ値を持つ要素には
 * 平均順位（average rank）が割り当てられます。
 *
 * @tparam Iterator 入力イテレータの型
 * @param first 範囲の先頭イテレータ
 * @param last 範囲の終端イテレータ
 * @return std::vector<double> 各要素に対応する順位のベクトル
 *
 * @note 空の範囲を渡した場合は空のベクトルを返します。
 */
template <typename Iterator>
std::vector<double> compute_ranks_with_ties(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n == 0) return {};

    // Create index-value pairs
    std::vector<std::pair<double, std::size_t>> indexed(n);
    std::size_t i = 0;
    for (auto it = first; it != last; ++it, ++i) {
        indexed[i] = {static_cast<double>(*it), i};
    }

    // Sort by value
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Assign ranks with tie handling (average rank)
    std::vector<double> ranks(n);
    std::size_t j = 0;
    while (j < n) {
        std::size_t k = j;
        // Find all elements with same value
        while (k < n && indexed[k].first == indexed[j].first) {
            ++k;
        }
        // Average rank for tied elements
        double avg_rank = (static_cast<double>(j + 1) + static_cast<double>(k)) / 2.0;
        for (std::size_t m = j; m < k; ++m) {
            ranks[indexed[m].second] = avg_rank;
        }
        j = k;
    }

    return ranks;
}

/**
 * @brief ソート済みデータから同順位グループのサイズを計算する
 *
 * 各同順位グループのサイズ t_j を返します（t_j > 1 のグループのみ）。
 * ノンパラメトリック検定の同順位補正に使用します。
 *
 * @param sorted_values ソート済みの値のベクトル
 * @return std::vector<std::size_t> 同順位グループサイズのベクトル（t > 1 のみ）
 */
inline std::vector<std::size_t> compute_tie_groups(const std::vector<double>& sorted_values)
{
    std::vector<std::size_t> tie_groups;
    std::size_t n = sorted_values.size();
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j < n && sorted_values[j] == sorted_values[i]) {
            ++j;
        }
        std::size_t t = j - i;
        if (t > 1) {
            tie_groups.push_back(t);
        }
        i = j;
    }
    return tie_groups;
}

// ============================================================================
// Shapiro-Wilk Test for Normality
// ============================================================================

/**
 * @brief Shapiro-Wilk検定を実行する
 *
 * データが正規分布に従うかどうかを検定します。
 * Roystonのアルゴリズムによる近似を使用しています。
 * サンプルサイズ n <= 50 に最適化されていますが、n <= 5000 まで対応可能です。
 *
 * @tparam Iterator 入力イテレータの型
 * @param first 範囲の先頭イテレータ
 * @param last 範囲の終端イテレータ
 * @return test_result 検定統計量W、p値、サンプルサイズを含む検定結果
 *
 * @throws std::invalid_argument 要素数が3未満の場合
 * @throws std::invalid_argument 要素数が5000を超える場合
 * @throws std::invalid_argument 分散がゼロの場合
 *
 * @note 帰無仮説: データは正規分布に従う
 * @note W統計量が1に近いほど正規性が高いことを示します
 */
template <typename Iterator>
test_result shapiro_wilk_test(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n < 3) {
        throw std::invalid_argument("statcpp::shapiro_wilk_test: need at least 3 elements");
    }
    if (n > 5000) {
        throw std::invalid_argument("statcpp::shapiro_wilk_test: n > 5000 not supported");
    }

    // Copy and sort data
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    double mean_val = statcpp::mean(sorted_data.begin(), sorted_data.end());

    // Compute SS (sum of squared deviations)
    double ss = 0.0;
    for (double x : sorted_data) {
        double d = x - mean_val;
        ss += d * d;
    }

    if (ss == 0.0) {
        throw std::invalid_argument("statcpp::shapiro_wilk_test: zero variance");
    }

    // Compute W statistic using Royston's approximation
    // First compute all m_i values (expected order statistics of standard normal)
    std::vector<double> m_vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        double p = (static_cast<double>(i + 1) - 0.375) / (static_cast<double>(n) + 0.25);
        m_vals[i] = norm_quantile(p);
    }

    // Compute sum of squared m values
    double sum_m2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_m2 += m_vals[i] * m_vals[i];
    }

    // Compute a coefficients using Royston's algorithm
    std::vector<double> a(n, 0.0);

    if (n <= 5) {
        // For very small n, use simple approximation
        double sqrt_sum_m2 = std::sqrt(sum_m2);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = m_vals[i] / sqrt_sum_m2;
        }
    } else {
        // Royston's polynomial approximation for a_n
        double sqrt_n = std::sqrt(static_cast<double>(n));
        double u = 1.0 / sqrt_n;

        // a_n coefficient (for largest order statistic)
        double a_n = -2.706056 * std::pow(u, 5) + 4.434685 * std::pow(u, 4)
                     - 2.071190 * std::pow(u, 3) - 0.147981 * std::pow(u, 2)
                     + 0.221157 * u + m_vals[n - 1] / std::sqrt(sum_m2);

        // a_{n-1} coefficient
        double a_n1 = -3.582633 * std::pow(u, 5) + 5.682633 * std::pow(u, 4)
                      - 1.752461 * std::pow(u, 3) - 0.293762 * std::pow(u, 2)
                      + 0.042981 * u + m_vals[n - 2] / std::sqrt(sum_m2);

        // Set the extreme coefficients
        a[n - 1] = a_n;
        a[0] = -a_n;
        if (n > 3) {
            a[n - 2] = a_n1;
            a[1] = -a_n1;
        }

        // Compute phi for intermediate coefficients
        double phi = (sum_m2 - 2.0 * m_vals[n - 1] * m_vals[n - 1]
                      - 2.0 * m_vals[n - 2] * m_vals[n - 2])
                     / (1.0 - 2.0 * a_n * a_n - 2.0 * a_n1 * a_n1);

        if (phi > 0) {
            double sqrt_phi = std::sqrt(phi);
            for (std::size_t i = 2; i < n - 2; ++i) {
                a[i] = m_vals[i] / sqrt_phi;
            }
        }
    }

    // Compute W = (sum(a_i * x_(i)))^2 / SS
    double b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        b += a[i] * sorted_data[i];
    }

    double w = (b * b) / ss;

    // Clamp W to valid range [0, 1]
    w = std::max(0.0, std::min(1.0, w));

    // Approximation for p-value using transformation to normal
    double ln_n = std::log(static_cast<double>(n));
    double mu, sigma, gamma;

    if (n <= 11) {
        gamma = 0.459 * static_cast<double>(n) - 2.273;
        mu = -0.0006714 * std::pow(static_cast<double>(n), 3)
             + 0.025054 * std::pow(static_cast<double>(n), 2)
             - 0.39978 * static_cast<double>(n) + 0.5440;
        sigma = std::exp(-0.0020322 * std::pow(static_cast<double>(n), 3)
                         + 0.062767 * std::pow(static_cast<double>(n), 2)
                         - 0.77857 * static_cast<double>(n) + 1.3822);
    } else {
        gamma = 0.0;
        mu = 0.0038915 * std::pow(ln_n, 3) - 0.083751 * std::pow(ln_n, 2)
             - 0.31082 * ln_n - 1.5861;
        sigma = std::exp(0.0030302 * std::pow(ln_n, 2) - 0.082676 * ln_n - 0.4803);
    }

    double z;
    if (gamma != 0.0 && w < 1.0) {
        double arg = gamma - std::log(1.0 - w);
        if (arg > 0) {
            z = (-std::log(arg) - mu) / sigma;
        } else {
            z = 3.0;  // Very high W, very normal
        }
    } else if (w < 1.0) {
        z = (std::log(1.0 - w) - mu) / sigma;
    } else {
        z = 3.0;  // Perfect W = 1
    }

    double p_value = 1.0 - norm_cdf(z);
    p_value = std::max(0.0, std::min(1.0, p_value));

    return {w, p_value, static_cast<double>(n), alternative_hypothesis::less};
}

// ============================================================================
// Lilliefors検定（正規性検定） (Lilliefors Test for Normality)
// ============================================================================

/**
 * @brief Lilliefors検定（正規性検定）を実行する (Perform Lilliefors test for normality)
 *
 * データが正規分布に従うかどうかを検定します。これはLilliefors検定であり、
 * 標準KS検定ではありません。パラメータ（平均、分散）はデータから推定され、
 * Lilliefors補正がp値に適用されます。
 *
 * @tparam Iterator 入力イテレータの型
 * @param first 範囲の先頭イテレータ
 * @param last 範囲の終端イテレータ
 * @return test_result 検定統計量D、p値、サンプルサイズを含む検定結果
 *
 * @throws std::invalid_argument 要素数が2未満の場合
 * @throws std::invalid_argument 分散がゼロの場合
 *
 * @note 帰無仮説: データは正規分布に従う
 * @note D統計量が大きいほど正規分布からの乖離が大きいことを示します
 * @note 現在の実装は漸近近似式を使用しています。将来のバージョンでは、より精密な
 *       臨界値（例: Dallal & Wilkinson 1986）を採用する可能性があります。
 * @note Lilliefors漸近近似は小標本（n < 20）や極端な裾領域（非常に小さいp値）では
 *       精度が低下する場合があります。小標本ではShapiro-Wilk検定の使用を検討してください。
 */
template <typename Iterator>
test_result lilliefors_test(Iterator first, Iterator last)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::lilliefors_test: need at least 2 elements");
    }

    // Standardize data
    double mean_val = statcpp::mean(first, last);
    double sd = statcpp::sample_stddev(first, last);

    if (sd == 0.0) {
        throw std::invalid_argument("statcpp::lilliefors_test: zero variance");
    }

    std::vector<double> standardized;
    standardized.reserve(n);
    for (auto it = first; it != last; ++it) {
        standardized.push_back((static_cast<double>(*it) - mean_val) / sd);
    }
    std::sort(standardized.begin(), standardized.end());

    // Compute D statistic
    double d_plus = 0.0;
    double d_minus = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double f_x = norm_cdf(standardized[i]);
        double f_n_upper = static_cast<double>(i + 1) / static_cast<double>(n);
        double f_n_lower = static_cast<double>(i) / static_cast<double>(n);

        d_plus = std::max(d_plus, f_n_upper - f_x);
        d_minus = std::max(d_minus, f_x - f_n_lower);
    }

    double d = std::max(d_plus, d_minus);

    // Lilliefors correction for estimated parameters
    // Use asymptotic approximation
    double sqrt_n = std::sqrt(static_cast<double>(n));
    double d_adj = (d - 0.01 + 0.85 / sqrt_n) * (sqrt_n + 0.05 + 0.82 / sqrt_n);

    // Asymptotic p-value approximation
    double p_value = 2.0 * std::exp(-2.0 * d_adj * d_adj);
    p_value = std::max(0.0, std::min(1.0, p_value));

    return {d, p_value, static_cast<double>(n), alternative_hypothesis::greater};
}

/**
 * @brief Kolmogorov-Smirnov検定（正規性検定）を実行（非推奨） (deprecated)
 *
 * @deprecated lilliefors_test() を使用してください。この関数は将来のバージョンで削除されます。
 *             実装はデータからパラメータを推定するLilliefors検定であり、
 *             既知パラメータの標準KS検定ではありません。
 */
template <typename Iterator>
[[deprecated("Use lilliefors_test() instead. ks_test_normal() will be removed in a future version.")]]
test_result ks_test_normal(Iterator first, Iterator last)
{
    return lilliefors_test(first, last);
}

// ============================================================================
// Levene's Test for Homogeneity of Variance
// ============================================================================

/**
 * @brief Levene検定（等分散性検定）を実行する
 *
 * 複数のグループ間で分散が等しいかどうかを検定します。
 * 中央値ベースの Brown-Forsythe 版を使用しており、
 * 正規分布からの乖離に対してロバストです。
 *
 * @param groups 各グループのデータを含むベクトルのベクトル
 * @return test_result F統計量、p値、自由度を含む検定結果
 *
 * @throws std::invalid_argument グループ数が2未満の場合
 * @throws std::invalid_argument いずれかのグループの要素数が2未満の場合
 *
 * @note 帰無仮説: 全てのグループの分散は等しい
 * @note F統計量が大きいほど分散の不均一性が大きいことを示します
 */
inline test_result levene_test(const std::vector<std::vector<double>>& groups)
{
    std::size_t k = groups.size();
    if (k < 2) {
        throw std::invalid_argument("statcpp::levene_test: need at least 2 groups");
    }

    // Compute group medians and deviations from median
    std::vector<std::vector<double>> z_values(k);
    std::size_t total_n = 0;

    for (std::size_t i = 0; i < k; ++i) {
        if (groups[i].size() < 2) {
            throw std::invalid_argument("statcpp::levene_test: each group needs at least 2 elements");
        }

        std::vector<double> sorted = groups[i];
        std::sort(sorted.begin(), sorted.end());
        double med = statcpp::median(sorted.begin(), sorted.end());

        for (double x : groups[i]) {
            z_values[i].push_back(std::abs(x - med));
        }
        total_n += groups[i].size();
    }

    // Compute group means of z values
    std::vector<double> z_means(k);
    double z_grand_mean = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        z_means[i] = statcpp::mean(z_values[i].begin(), z_values[i].end());
        z_grand_mean += z_means[i] * static_cast<double>(z_values[i].size());
    }
    z_grand_mean /= static_cast<double>(total_n);

    // Compute test statistic
    double ss_between = 0.0;
    double ss_within = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        double ni = static_cast<double>(z_values[i].size());
        ss_between += ni * (z_means[i] - z_grand_mean) * (z_means[i] - z_grand_mean);

        for (double z : z_values[i]) {
            ss_within += (z - z_means[i]) * (z - z_means[i]);
        }
    }

    double df1 = static_cast<double>(k - 1);
    double df2 = static_cast<double>(total_n - k);

    // ガード: 全偏差がゼロ（各グループ内で全値が同一）の場合、
    // 分散は自明に等しい → F = 0, p = 1
    if (ss_within == 0.0) {
        return {0.0, 1.0, df1, alternative_hypothesis::greater};
    }

    double f = (ss_between / df1) / (ss_within / df2);
    double p_value = 1.0 - f_cdf(f, df1, df2);

    return {f, p_value, df1, alternative_hypothesis::greater};
}

// ============================================================================
// Bartlett's Test for Homogeneity of Variance
// ============================================================================

/**
 * @brief Bartlett検定（等分散性検定）を実行する
 *
 * 複数のグループ間で分散が等しいかどうかを検定します。
 * 正規分布を仮定した尤度比検定に基づいています。
 * データが正規分布に従わない場合は Levene 検定の使用を推奨します。
 *
 * @param groups 各グループのデータを含むベクトルのベクトル
 * @return test_result カイ二乗統計量、p値、自由度を含む検定結果
 *
 * @throws std::invalid_argument グループ数が2未満の場合
 * @throws std::invalid_argument いずれかのグループの要素数が2未満の場合
 * @throws std::invalid_argument いずれかのグループの分散がゼロ以下の場合
 *
 * @note 帰無仮説: 全てのグループの分散は等しい
 * @note 正規性からの乖離に敏感なため、正規性が疑わしい場合は levene_test を使用してください
 */
inline test_result bartlett_test(const std::vector<std::vector<double>>& groups)
{
    std::size_t k = groups.size();
    if (k < 2) {
        throw std::invalid_argument("statcpp::bartlett_test: need at least 2 groups");
    }

    std::vector<double> vars(k);
    std::vector<std::size_t> ns(k);
    std::size_t total_n = 0;
    double pooled_var_num = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        if (groups[i].size() < 2) {
            throw std::invalid_argument("statcpp::bartlett_test: each group needs at least 2 elements");
        }
        ns[i] = groups[i].size();
        total_n += ns[i];
        vars[i] = statcpp::sample_variance(groups[i].begin(), groups[i].end());

        if (vars[i] <= 0.0) {
            throw std::invalid_argument("statcpp::bartlett_test: zero or negative variance in group");
        }

        pooled_var_num += (ns[i] - 1) * vars[i];
    }

    double pooled_var = pooled_var_num / static_cast<double>(total_n - k);

    // Compute Bartlett's statistic
    double sum_log = 0.0;
    double sum_inv = 0.0;

    for (std::size_t i = 0; i < k; ++i) {
        double df_i = static_cast<double>(ns[i] - 1);
        sum_log += df_i * std::log(vars[i]);
        sum_inv += 1.0 / df_i;
    }

    double df_total = static_cast<double>(total_n - k);
    double chi2 = df_total * std::log(pooled_var) - sum_log;

    // Correction factor
    double c = 1.0 + (sum_inv - 1.0 / df_total) / (3.0 * (k - 1));
    chi2 /= c;

    double df = static_cast<double>(k - 1);
    double p_value = 1.0 - chisq_cdf(chi2, df);

    return {chi2, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// Wilcoxon Signed-Rank Test
// ============================================================================

/**
 * @brief Wilcoxon符号付順位検定（1標本）を実行する
 *
 * データの中央値が指定された値と等しいかどうかを検定します。
 * 正規分布を仮定しないノンパラメトリック検定です。
 * 正規近似（連続性補正付き）を使用してp値を計算します。
 *
 * @tparam Iterator 入力イテレータの型
 * @param first 範囲の先頭イテレータ
 * @param last 範囲の終端イテレータ
 * @param mu0 帰無仮説における中央値（デフォルト: 0.0）
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result W+統計量（正の差の順位和）、p値、非ゼロ差のサンプルサイズを含む検定結果
 *
 * @throws std::invalid_argument 要素数が2未満の場合
 * @throws std::invalid_argument 非ゼロの差が2未満の場合
 *
 * @note 帰無仮説: 中央値は mu0 に等しい
 * @note mu0 との差がゼロの観測値は除外されます
 */
template <typename Iterator>
test_result wilcoxon_signed_rank_test(Iterator first, Iterator last, double mu0 = 0.0,
                                       alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::wilcoxon_signed_rank_test: need at least 2 elements");
    }

    // Compute differences from mu0, excluding zeros
    std::vector<double> diffs;
    for (auto it = first; it != last; ++it) {
        double d = static_cast<double>(*it) - mu0;
        if (d != 0.0) {
            diffs.push_back(d);
        }
    }

    std::size_t n_nonzero = diffs.size();
    if (n_nonzero < 2) {
        throw std::invalid_argument("statcpp::wilcoxon_signed_rank_test: need at least 2 non-zero differences");
    }

    // Compute ranks of absolute differences
    std::vector<double> abs_diffs(n_nonzero);
    for (std::size_t i = 0; i < n_nonzero; ++i) {
        abs_diffs[i] = std::abs(diffs[i]);
    }

    auto ranks = compute_ranks_with_ties(abs_diffs.begin(), abs_diffs.end());

    // Compute W+ (sum of ranks of positive differences)
    double w = 0.0;

    for (std::size_t i = 0; i < n_nonzero; ++i) {
        if (diffs[i] > 0.0) {
            w += ranks[i];
        }
    }

    // 正規近似によるp値計算（連続補正あり）
    double nn = static_cast<double>(n_nonzero);
    double mean_w = nn * (nn + 1.0) / 4.0;
    double var_w = nn * (nn + 1.0) * (2.0 * nn + 1.0) / 24.0;

    // 同順位補正：分散から sum(t^3 - t) / 48 を引く
    std::vector<double> sorted_abs(abs_diffs.begin(), abs_diffs.end());
    std::sort(sorted_abs.begin(), sorted_abs.end());
    auto tie_groups = compute_tie_groups(sorted_abs);
    double tie_correction = 0.0;
    for (auto t : tie_groups) {
        double td = static_cast<double>(t);
        tie_correction += td * td * td - td;
    }
    var_w -= tie_correction / 48.0;

    double se = std::sqrt(std::max(0.0, var_w));

    double z;
    double p_value;

    switch (alt) {
        case alternative_hypothesis::less:
            z = (w - mean_w + 0.5) / se;
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            z = (w - mean_w - 0.5) / se;
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            z = (w - mean_w) / se;
            if (z < 0) z = (w - mean_w + 0.5) / se;
            else z = (w - mean_w - 0.5) / se;
            p_value = 2.0 * std::min(norm_cdf(z), 1.0 - norm_cdf(z));
            break;
    }

    return {w, p_value, static_cast<double>(n_nonzero), alt};
}

// ============================================================================
// Mann-Whitney U Test
// ============================================================================

/**
 * @brief Mann-Whitney U検定（2標本）を実行する
 *
 * 2つの独立した標本が同じ分布からのものかどうかを検定します。
 * Wilcoxon順位和検定とも呼ばれ、正規分布を仮定しないノンパラメトリック検定です。
 * 正規近似を使用してp値を計算します。
 *
 * @tparam Iterator1 第1標本のイテレータ型
 * @tparam Iterator2 第2標本のイテレータ型
 * @param first1 第1標本の先頭イテレータ
 * @param last1 第1標本の終端イテレータ
 * @param first2 第2標本の先頭イテレータ
 * @param last2 第2標本の終端イテレータ
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result U1統計量、p値、合計サンプルサイズを含む検定結果
 *
 * @throws std::invalid_argument いずれかの標本の要素数が2未満の場合
 *
 * @note 帰無仮説: 2つの標本は同じ分布から抽出されている
 * @note U1 は第1標本の順位和から計算されます
 */
template <typename Iterator1, typename Iterator2>
test_result mann_whitney_u_test(Iterator1 first1, Iterator1 last1,
                                 Iterator2 first2, Iterator2 last2,
                                 alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::mann_whitney_u_test: need at least 2 elements in each sample");
    }

    // Combine and rank all observations
    std::vector<std::pair<double, int>> combined;
    combined.reserve(n1 + n2);

    for (auto it = first1; it != last1; ++it) {
        combined.push_back({static_cast<double>(*it), 1}); // group 1
    }
    for (auto it = first2; it != last2; ++it) {
        combined.push_back({static_cast<double>(*it), 2}); // group 2
    }

    // Sort by value
    std::sort(combined.begin(), combined.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Assign ranks with tie handling
    std::size_t total_n = n1 + n2;
    std::vector<double> ranks(total_n);
    std::size_t i = 0;
    while (i < total_n) {
        std::size_t j = i;
        while (j < total_n && combined[j].first == combined[i].first) {
            ++j;
        }
        double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Compute R1 (sum of ranks in group 1)
    double r1 = 0.0;
    for (std::size_t k = 0; k < total_n; ++k) {
        if (combined[k].second == 1) {
            r1 += ranks[k];
        }
    }

    // Compute U1
    double u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    (void)(static_cast<double>(n1 * n2) - u1);  // u2 not used but computed for reference

    // 同順位補正付き正規近似
    double N = static_cast<double>(total_n);
    double mean_u = static_cast<double>(n1 * n2) / 2.0;
    double var_u = static_cast<double>(n1 * n2) / 12.0 * (N + 1.0);

    // 同順位補正：ソート済み結合データから同順位グループを計算
    {
        std::vector<double> sorted_vals;
        sorted_vals.reserve(total_n);
        for (const auto& p : combined) {
            sorted_vals.push_back(p.first);
        }
        // combined はすでに値でソート済み
        auto tie_groups = compute_tie_groups(sorted_vals);
        double tie_sum = 0.0;
        for (auto t : tie_groups) {
            double td = static_cast<double>(t);
            tie_sum += td * td * td - td;
        }
        // 補正後の分散: n1*n2/(12*N*(N-1)) * (N^3 - N - tie_sum)
        if (N > 1.0 && tie_sum > 0.0) {
            var_u = static_cast<double>(n1 * n2) / (12.0 * N * (N - 1.0))
                    * (N * N * N - N - tie_sum);
        }
    }

    double se = std::sqrt(std::max(0.0, var_u));

    double z = (u1 - mean_u) / se;

    double p_value;
    switch (alt) {
        case alternative_hypothesis::less:
            p_value = norm_cdf(z);
            break;
        case alternative_hypothesis::greater:
            p_value = 1.0 - norm_cdf(z);
            break;
        case alternative_hypothesis::two_sided:
        default:
            p_value = 2.0 * std::min(norm_cdf(z), 1.0 - norm_cdf(z));
            break;
    }

    return {u1, p_value, static_cast<double>(n1 + n2), alt};
}

// ============================================================================
// Kruskal-Wallis Test
// ============================================================================

/**
 * @brief Kruskal-Wallis検定（k標本）を実行する
 *
 * 3つ以上の独立した標本が同じ分布からのものかどうかを検定します。
 * 一元配置分散分析のノンパラメトリック版であり、正規分布を仮定しません。
 * カイ二乗近似を使用してp値を計算します。
 *
 * @param groups 各グループのデータを含むベクトルのベクトル
 * @return test_result H統計量、p値、自由度（k-1）を含む検定結果
 *
 * @throws std::invalid_argument グループ数が2未満の場合
 * @throws std::invalid_argument いずれかのグループが空の場合
 *
 * @note 帰無仮説: 全てのグループは同じ分布から抽出されている
 * @note H統計量が大きいほどグループ間の差が大きいことを示します
 */
inline test_result kruskal_wallis_test(const std::vector<std::vector<double>>& groups)
{
    std::size_t k = groups.size();
    if (k < 2) {
        throw std::invalid_argument("statcpp::kruskal_wallis_test: need at least 2 groups");
    }

    // Combine all observations with group labels
    std::vector<std::pair<double, std::size_t>> combined;
    std::vector<std::size_t> ns(k);
    std::size_t total_n = 0;

    for (std::size_t i = 0; i < k; ++i) {
        if (groups[i].empty()) {
            throw std::invalid_argument("statcpp::kruskal_wallis_test: empty group");
        }
        ns[i] = groups[i].size();
        total_n += ns[i];
        for (double x : groups[i]) {
            combined.push_back({x, i});
        }
    }

    // Sort by value
    std::sort(combined.begin(), combined.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Assign ranks with tie handling
    std::vector<double> ranks(total_n);
    std::size_t i = 0;
    while (i < total_n) {
        std::size_t j = i;
        while (j < total_n && combined[j].first == combined[i].first) {
            ++j;
        }
        double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t m = i; m < j; ++m) {
            ranks[m] = avg_rank;
        }
        i = j;
    }

    // Compute sum of ranks for each group
    std::vector<double> rank_sums(k, 0.0);
    for (std::size_t m = 0; m < total_n; ++m) {
        rank_sums[combined[m].second] += ranks[m];
    }

    // Compute H statistic
    double n_d = static_cast<double>(total_n);
    double sum_term = 0.0;
    for (std::size_t g = 0; g < k; ++g) {
        double r_bar = rank_sums[g] / static_cast<double>(ns[g]);
        sum_term += static_cast<double>(ns[g]) * r_bar * r_bar;
    }

    double h = (12.0 / (n_d * (n_d + 1.0))) * sum_term - 3.0 * (n_d + 1.0);

    // 同順位補正: H_corrected = H / (1 - sum(t^3 - t) / (N^3 - N))
    {
        std::vector<double> sorted_vals;
        sorted_vals.reserve(total_n);
        for (const auto& p : combined) {
            sorted_vals.push_back(p.first);
        }
        auto tie_groups = compute_tie_groups(sorted_vals);
        double tie_sum = 0.0;
        for (auto t : tie_groups) {
            double td = static_cast<double>(t);
            tie_sum += td * td * td - td;
        }
        double denom = n_d * n_d * n_d - n_d;
        if (tie_sum > 0.0 && denom > 0.0) {
            h /= (1.0 - tie_sum / denom);
        }
    }

    // カイ二乗分布を用いた近似p値
    double df = static_cast<double>(k - 1);
    double p_value = 1.0 - chisq_cdf(h, df);

    return {h, p_value, df, alternative_hypothesis::greater};
}

// ============================================================================
// Fisher's Exact Test (2x2 table)
// ============================================================================

/**
 * @brief Fisher正確確率検定（2x2分割表）を実行する
 *
 * 2x2分割表において、行と列の変数が独立かどうかを検定します。
 * 超幾何分布に基づく正確な確率計算を行うため、
 * 期待度数が小さい場合でも適用可能です。
 *
 * 分割表の形式:
 * @code
 *       | Col1 | Col2 | Row Total
 * Row1  |  a   |  b   |   a+b
 * Row2  |  c   |  d   |   c+d
 * Col   | a+c  | b+d  |   n
 * @endcode
 *
 * @param a 分割表の(1,1)セルの度数
 * @param b 分割表の(1,2)セルの度数
 * @param c 分割表の(2,1)セルの度数
 * @param d 分割表の(2,2)セルの度数
 * @param alt 対立仮説の種類（デフォルト: 両側検定）
 * @return test_result オッズ比、p値、自由度（NaN）を含む検定結果
 *
 * @note 帰無仮説: 行と列の変数は独立である（オッズ比 = 1）
 * @note 両側検定では、観測された確率以下の全ての表の確率を合計します
 * @note b または c がゼロの場合、オッズ比は無限大となります
 */
inline test_result fisher_exact_test(std::uint64_t a, std::uint64_t b,
                                      std::uint64_t c, std::uint64_t d,
                                      alternative_hypothesis alt = alternative_hypothesis::two_sided)
{
    // 2x2 contingency table:
    //       | Col1 | Col2 | Row Total
    // Row1  |  a   |  b   |   a+b
    // Row2  |  c   |  d   |   c+d
    // Col   | a+c  | b+d  |   n

    std::uint64_t n = a + b + c + d;
    std::uint64_t row1 = a + b;
    (void)(c + d);  // row2 not used
    std::uint64_t col1 = a + c;
    std::uint64_t col2 = b + d;

    // Compute p-value under hypergeometric distribution
    // P(X = a) where X ~ Hypergeometric(n, col1, row1)

    // Probability of observed table
    double log_p_obs = log_binomial_coef(col1, a) + log_binomial_coef(col2, b) - log_binomial_coef(n, row1);
    double p_obs = std::exp(log_p_obs);

    double p_value = 0.0;

    // Compute range of possible values for a
    std::uint64_t a_min = (row1 > col2) ? row1 - col2 : 0;
    std::uint64_t a_max = std::min(row1, col1);

    switch (alt) {
        case alternative_hypothesis::less:
            for (std::uint64_t x = a_min; x <= a; ++x) {
                double log_p = log_binomial_coef(col1, x) + log_binomial_coef(col2, row1 - x) - log_binomial_coef(n, row1);
                p_value += std::exp(log_p);
            }
            break;
        case alternative_hypothesis::greater:
            for (std::uint64_t x = a; x <= a_max; ++x) {
                double log_p = log_binomial_coef(col1, x) + log_binomial_coef(col2, row1 - x) - log_binomial_coef(n, row1);
                p_value += std::exp(log_p);
            }
            break;
        case alternative_hypothesis::two_sided:
        default:
            // Sum probabilities of tables as extreme or more extreme
            for (std::uint64_t x = a_min; x <= a_max; ++x) {
                double log_p = log_binomial_coef(col1, x) + log_binomial_coef(col2, row1 - x) - log_binomial_coef(n, row1);
                double p = std::exp(log_p);
                if (p <= p_obs + 1e-10) {
                    p_value += p;
                }
            }
            break;
    }

    p_value = std::min(1.0, p_value);

    // Odds ratio as test statistic
    double odds_ratio = (b == 0 || c == 0) ? std::numeric_limits<double>::infinity()
                                           : (static_cast<double>(a) * d) / (static_cast<double>(b) * c);

    return {odds_ratio, p_value, std::numeric_limits<double>::quiet_NaN(), alt};
}

} // namespace statcpp
