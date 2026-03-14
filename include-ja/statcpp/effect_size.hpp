/**
 * @file effect_size.hpp
 * @brief 効果量の計算と解釈
 */

#pragma once

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

#include <cmath>
#include <stdexcept>

namespace statcpp {

// ============================================================================
// Cohen's d (Standardized Mean Difference)
// ============================================================================

/**
 * @brief Cohen's d（1標本、母標準偏差が既知）
 *
 * 標準化された平均差を計算します。母標準偏差が既知の場合に使用します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param mu0 比較する母平均
 * @param sigma 母標準偏差
 * @return Cohen's d
 * @throws std::invalid_argument 空の範囲の場合、またはsigmaが正でない場合
 */
template <typename Iterator>
double cohens_d(Iterator first, Iterator last, double mu0, double sigma)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument("statcpp::cohens_d: sigma must be positive");
    }

    auto n = statcpp::count(first, last);
    if (n == 0) {
        throw std::invalid_argument("statcpp::cohens_d: empty range");
    }

    double mean_val = statcpp::mean(first, last);
    return (mean_val - mu0) / sigma;
}

/**
 * @brief Cohen's d（1標本、標本標準偏差を使用）
 *
 * 標準化された平均差を計算します。標本標準偏差を使用します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param mu0 比較する母平均
 * @return Cohen's d
 * @throws std::invalid_argument 要素数が2未満の場合、または分散が0の場合
 */
template <typename Iterator>
double cohens_d(Iterator first, Iterator last, double mu0)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::cohens_d: need at least 2 elements");
    }

    double mean_val = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::cohens_d: zero variance");
    }

    return (mean_val - mu0) / s;
}

/**
 * @brief Cohen's d（2標本、プール標準偏差）
 *
 * 2つのグループ間の標準化された平均差を計算します。
 * プールされた標準偏差を使用します。
 *
 * @tparam Iterator1 第1サンプルのRandomAccessIterator型
 * @tparam Iterator2 第2サンプルのRandomAccessIterator型
 * @param first1 第1サンプルの開始イテレータ
 * @param last1 第1サンプルの終了イテレータ
 * @param first2 第2サンプルの開始イテレータ
 * @param last2 第2サンプルの終了イテレータ
 * @return Cohen's d
 * @throws std::invalid_argument 各サンプルの要素数が2未満の場合、またはプール分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
double cohens_d_two_sample(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::cohens_d_two_sample: need at least 2 elements in each sample");
    }

    double mean1 = statcpp::mean(first1, last1);
    double mean2 = statcpp::mean(first2, last2);
    double var1 = statcpp::sample_variance(first1, last1);
    double var2 = statcpp::sample_variance(first2, last2);

    // Pooled standard deviation
    double sp = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));

    if (sp == 0.0) {
        throw std::invalid_argument("statcpp::cohens_d_two_sample: zero pooled variance");
    }

    return (mean1 - mean2) / sp;
}

// ============================================================================
// Hedges' g (Bias-corrected Cohen's d)
// ============================================================================

/**
 * @brief Hedgesのバイアス補正係数J
 *
 * 小サンプルサイズのバイアスを補正するための係数を計算します。
 *
 * @param df 自由度
 * @return バイアス補正係数
 */
inline double hedges_correction_factor(double df)
{
    // J ≈ 1 - 3 / (4 * df - 1)
    return 1.0 - 3.0 / (4.0 * df - 1.0);
}

/**
 * @brief Hedges' g（1標本）
 *
 * バイアス補正されたCohen's dを計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param mu0 比較する母平均
 * @return Hedges' g
 * @throws std::invalid_argument 要素数が2未満の場合
 */
template <typename Iterator>
double hedges_g(Iterator first, Iterator last, double mu0)
{
    auto n = statcpp::count(first, last);
    if (n < 2) {
        throw std::invalid_argument("statcpp::hedges_g: need at least 2 elements");
    }

    double d = cohens_d(first, last, mu0);
    double df = static_cast<double>(n - 1);
    return d * hedges_correction_factor(df);
}

/**
 * @brief Hedges' g（2標本）
 *
 * 2つのグループ間のバイアス補正された標準化平均差を計算します。
 *
 * @tparam Iterator1 第1サンプルのRandomAccessIterator型
 * @tparam Iterator2 第2サンプルのRandomAccessIterator型
 * @param first1 第1サンプルの開始イテレータ
 * @param last1 第1サンプルの終了イテレータ
 * @param first2 第2サンプルの開始イテレータ
 * @param last2 第2サンプルの終了イテレータ
 * @return Hedges' g
 * @throws std::invalid_argument 各サンプルの要素数が2未満の場合
 */
template <typename Iterator1, typename Iterator2>
double hedges_g_two_sample(Iterator1 first1, Iterator1 last1,
                           Iterator2 first2, Iterator2 last2)
{
    auto n1 = statcpp::count(first1, last1);
    auto n2 = statcpp::count(first2, last2);

    if (n1 < 2 || n2 < 2) {
        throw std::invalid_argument("statcpp::hedges_g_two_sample: need at least 2 elements in each sample");
    }

    double d = cohens_d_two_sample(first1, last1, first2, last2);
    double df = static_cast<double>(n1 + n2 - 2);
    return d * hedges_correction_factor(df);
}

// ============================================================================
// Glass's Delta (using control group's SD)
// ============================================================================

/**
 * @brief Glass's Δ（対照群の標準偏差を使用）
 *
 * 対照群の標準偏差のみを使用した効果量を計算します。
 * 実験群と対照群の分散が大きく異なる場合に有用です。
 *
 * @tparam Iterator1 対照群のRandomAccessIterator型
 * @tparam Iterator2 実験群のRandomAccessIterator型
 * @param control_first 対照群の開始イテレータ
 * @param control_last 対照群の終了イテレータ
 * @param treatment_first 実験群の開始イテレータ
 * @param treatment_last 実験群の終了イテレータ
 * @return Glass's Δ
 * @throws std::invalid_argument 対照群の要素数が2未満の場合、実験群が空の場合、または対照群の分散が0の場合
 */
template <typename Iterator1, typename Iterator2>
double glass_delta(Iterator1 control_first, Iterator1 control_last,
                   Iterator2 treatment_first, Iterator2 treatment_last)
{
    auto n1 = statcpp::count(control_first, control_last);
    auto n2 = statcpp::count(treatment_first, treatment_last);

    if (n1 < 2) {
        throw std::invalid_argument("statcpp::glass_delta: control group needs at least 2 elements");
    }
    if (n2 == 0) {
        throw std::invalid_argument("statcpp::glass_delta: treatment group is empty");
    }

    double mean1 = statcpp::mean(control_first, control_last);
    double mean2 = statcpp::mean(treatment_first, treatment_last);
    double s1 = statcpp::sample_stddev(control_first, control_last);

    if (s1 == 0.0) {
        throw std::invalid_argument("statcpp::glass_delta: control group has zero variance");
    }

    return (mean2 - mean1) / s1;
}

// ============================================================================
// Correlation-based Effect Size (r)
// ============================================================================

/**
 * @brief t値から相関係数への変換
 *
 * @param t t統計量
 * @param df 自由度
 * @return 相関係数
 */
inline double t_to_r(double t, double df)
{
    return t / std::sqrt(t * t + df);
}

/**
 * @brief Cohen's dから相関係数への変換
 *
 * @param d Cohen's d
 * @return 相関係数
 */
inline double d_to_r(double d)
{
    // r = d / sqrt(d^2 + 4)
    return d / std::sqrt(d * d + 4.0);
}

/**
 * @brief 相関係数からCohen's dへの変換
 *
 * @param r 相関係数
 * @return Cohen's d
 * @throws std::invalid_argument |r| >= 1の場合
 */
inline double r_to_d(double r)
{
    // d = 2r / sqrt(1 - r^2)
    if (std::abs(r) >= 1.0) {
        throw std::invalid_argument("statcpp::r_to_d: |r| must be less than 1");
    }
    return 2.0 * r / std::sqrt(1.0 - r * r);
}

// ============================================================================
// Eta-squared and Partial Eta-squared
// ============================================================================

/**
 * @brief η²（イータ二乗）をF検定から計算
 *
 * 効果の平方和と総平方和から効果量を計算します。
 *
 * @param ss_effect 効果の平方和
 * @param ss_total 総平方和
 * @return η²
 * @throws std::invalid_argument ss_totalが正でない場合
 */
inline double eta_squared(double ss_effect, double ss_total)
{
    if (ss_total <= 0.0) {
        throw std::invalid_argument("statcpp::eta_squared: ss_total must be positive");
    }
    return ss_effect / ss_total;
}

/**
 * @brief 偏η²をF検定から計算
 *
 * @param f F統計量
 * @param df1 分子の自由度
 * @param df2 分母の自由度
 * @return 偏η²
 */
inline double partial_eta_squared(double f, double df1, double df2)
{
    return (f * df1) / (f * df1 + df2);
}

// ============================================================================
// Omega-squared (less biased than eta-squared)
// ============================================================================

/**
 * @brief ω²（オメガ二乗）
 *
 * η²よりもバイアスの少ない効果量の推定値です。
 *
 * @param ss_effect 効果の平方和
 * @param ss_total 総平方和
 * @param ms_error 誤差平均平方
 * @param df_effect 効果の自由度
 * @return ω²
 * @throws std::invalid_argument ss_totalが正でない場合
 */
inline double omega_squared(double ss_effect, double ss_total, double ms_error, double df_effect)
{
    if (ss_total <= 0.0) {
        throw std::invalid_argument("statcpp::omega_squared: ss_total must be positive");
    }
    return (ss_effect - df_effect * ms_error) / (ss_total + ms_error);
}

// ============================================================================
// Cohen's h (Effect Size for Proportions)
// ============================================================================

/**
 * @brief Cohen's h（2つの比率の差の効果量）
 *
 * 2つの比率（割合）の差を表す効果量を計算します。
 *
 * @param p1 第1グループの比率
 * @param p2 第2グループの比率
 * @return Cohen's h
 * @throws std::invalid_argument 比率が[0, 1]の範囲外の場合
 */
inline double cohens_h(double p1, double p2)
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::cohens_h: proportions must be in [0, 1]");
    }

    // h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    return 2.0 * (std::asin(std::sqrt(p1)) - std::asin(std::sqrt(p2)));
}

// ============================================================================
// Odds Ratio and Risk Ratio
// ============================================================================

/**
 * @brief オッズ比
 *
 * 2x2分割表からオッズ比を計算します。
 *
 * @param a セル(1,1)の度数（曝露あり・疾患あり）
 * @param b セル(1,2)の度数（曝露あり・疾患なし）
 * @param c セル(2,1)の度数（曝露なし・疾患あり）
 * @param d セル(2,2)の度数（曝露なし・疾患なし）
 * @return オッズ比
 * @throws std::invalid_argument bまたはcが0の場合
 */
inline double odds_ratio(double a, double b, double c, double d)
{
    // 2x2 table: a, b, c, d
    if (b == 0.0 || c == 0.0) {
        throw std::invalid_argument("statcpp::odds_ratio: cell b or c is zero");
    }
    return (a * d) / (b * c);
}

/**
 * @brief 相対リスク（リスク比）
 *
 * 2x2分割表から相対リスクを計算します。
 *
 * @param a セル(1,1)の度数（曝露あり・疾患あり）
 * @param b セル(1,2)の度数（曝露あり・疾患なし）
 * @param c セル(2,1)の度数（曝露なし・疾患あり）
 * @param d セル(2,2)の度数（曝露なし・疾患なし）
 * @return 相対リスク
 * @throws std::invalid_argument 行合計が0の場合、またはグループ2のリスクが0の場合
 */
inline double risk_ratio(double a, double b, double c, double d)
{
    // Risk in group 1: a / (a + b)
    // Risk in group 2: c / (c + d)
    if (a + b == 0.0 || c + d == 0.0) {
        throw std::invalid_argument("statcpp::risk_ratio: row total is zero");
    }
    double risk1 = a / (a + b);
    double risk2 = c / (c + d);

    if (risk2 == 0.0) {
        throw std::invalid_argument("statcpp::risk_ratio: risk in group 2 is zero");
    }

    return risk1 / risk2;
}

// ============================================================================
// Effect Size Interpretation
// ============================================================================

/**
 * @brief 効果量の大きさを表す列挙型
 */
enum class effect_size_magnitude {
    negligible,  ///< 無視できる
    small,       ///< 小
    medium,      ///< 中
    large        ///< 大
};

/**
 * @brief Cohen's dの解釈
 *
 * Cohen (1988)の基準に基づいて効果量の大きさを判定します。
 *
 * @param d Cohen's d
 * @return 効果量の大きさ
 */
inline effect_size_magnitude interpret_cohens_d(double d)
{
    double abs_d = std::abs(d);
    if (abs_d < 0.2) return effect_size_magnitude::negligible;
    if (abs_d < 0.5) return effect_size_magnitude::small;
    if (abs_d < 0.8) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

/**
 * @brief 相関係数の解釈
 *
 * Cohen (1988)の基準に基づいて相関の強さを判定します。
 *
 * @param r 相関係数
 * @return 効果量の大きさ
 */
inline effect_size_magnitude interpret_correlation(double r)
{
    double abs_r = std::abs(r);
    if (abs_r < 0.1) return effect_size_magnitude::negligible;
    if (abs_r < 0.3) return effect_size_magnitude::small;
    if (abs_r < 0.5) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

/**
 * @brief η²の解釈
 *
 * Cohen (1988)の基準に基づいて効果量の大きさを判定します。
 *
 * @param eta2 η²
 * @return 効果量の大きさ
 */
inline effect_size_magnitude interpret_eta_squared(double eta2)
{
    if (eta2 < 0.01) return effect_size_magnitude::negligible;
    if (eta2 < 0.06) return effect_size_magnitude::small;
    if (eta2 < 0.14) return effect_size_magnitude::medium;
    return effect_size_magnitude::large;
}

} // namespace statcpp
