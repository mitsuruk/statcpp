/**
 * @file power_analysis.hpp
 * @brief 検出力分析の実装 (Power analysis implementation)
 *
 * t検定、比率検定などの検出力計算とサンプルサイズ計算を提供します。
 * Provides power calculation and sample size determination for t-tests, proportion tests, and more.
 *
 * @note 実装上の注意（正規分布近似について）:
 *       この実装ではt検定の検出力計算に正規分布近似を使用しています。
 *       厳密には非心t分布（noncentral t-distribution）を使用すべきですが、
 *       正規近似は以下の条件で十分な精度を提供します：
 *       - サンプルサイズが大きい場合（n > 30程度）
 *       - 中程度以上の効果量の場合
 *
 *       小サンプルや小さい効果量の場合、この近似は検出力をやや過大評価する
 *       可能性があります。より正確な計算が必要な場合は、R の pwr パッケージや
 *       G*Power などの専用ソフトウェアの使用を検討してください。
 */

#pragma once

#include "statcpp/special_functions.hpp"
#include "statcpp/continuous_distributions.hpp"
#include "statcpp/parametric_tests.hpp"  // for alternative_hypothesis enum

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace statcpp {

// ============================================================================
// Power Analysis Result Structure
// ============================================================================

/**
 * @brief 検出力分析の結果 (Power analysis result)
 */
struct power_result {
    double power;          ///< 検出力 (1-β) (statistical power, 1-β)
    double sample_size;    ///< サンプルサイズ (sample size)
    double effect_size;    ///< 効果量 (effect size)
    double alpha;          ///< 有意水準 (significance level)
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief 内部ヘルパー関数 (Internal helper functions)
 */
namespace detail {

/**
 * @brief 非心t分布の非心度パラメータを計算 (Calculate noncentrality parameter for t-distribution)
 *
 * @param effect_size 効果量 (effect size)
 * @param n サンプルサイズ (sample size)
 * @return 非心度パラメータ (noncentrality parameter)
 */
inline double noncentrality_parameter_t(double effect_size, double n) {
    return effect_size * std::sqrt(n);
}

/**
 * @brief 2標本の場合の非心度パラメータ (Calculate noncentrality parameter for two-sample case)
 *
 * @param effect_size 効果量 (effect size)
 * @param n1 群1のサンプルサイズ (sample size of group 1)
 * @param n2 群2のサンプルサイズ (sample size of group 2)
 * @return 非心度パラメータ (noncentrality parameter)
 */
inline double noncentrality_parameter_t_two_sample(double effect_size, double n1, double n2) {
    return effect_size * std::sqrt((n1 * n2) / (n1 + n2));
}

/**
 * @brief t分布の臨界値を計算（両側検定）(Calculate critical value for two-sided t-test)
 *
 * @param alpha 有意水準 (significance level)
 * @param df 自由度 (degrees of freedom)
 * @return 臨界値 (critical value)
 */
inline double critical_t_two_sided(double alpha, double df) {
    return t_quantile(1.0 - alpha / 2.0, df);
}

/**
 * @brief t分布の臨界値を計算（片側検定）(Calculate critical value for one-sided t-test)
 *
 * @param alpha 有意水準 (significance level)
 * @param df 自由度 (degrees of freedom)
 * @return 臨界値 (critical value)
 */
inline double critical_t_one_sided(double alpha, double df) {
    return t_quantile(1.0 - alpha, df);
}

/**
 * @brief 正規分布の臨界値を計算（両側検定）(Calculate critical value for two-sided normal test)
 *
 * @param alpha 有意水準 (significance level)
 * @return 臨界値 (critical value)
 */
inline double critical_z_two_sided(double alpha) {
    return norm_quantile(1.0 - alpha / 2.0);
}

/**
 * @brief 正規分布の臨界値を計算（片側検定）(Calculate critical value for one-sided normal test)
 *
 * @param alpha 有意水準 (significance level)
 * @return 臨界値 (critical value)
 */
inline double critical_z_one_sided(double alpha) {
    return norm_quantile(1.0 - alpha);
}

/**
 * @brief alternative_hypothesis列挙型を文字列に変換
 *        (Convert alternative_hypothesis enum to string)
 *
 * @param alt 対立仮説の列挙値 (alternative hypothesis enum value)
 * @return 対応する文字列 ("two.sided", "greater", "less")
 */
inline const char* alternative_to_string(alternative_hypothesis alt) {
    switch (alt) {
        case alternative_hypothesis::two_sided: return "two.sided";
        case alternative_hypothesis::greater:   return "greater";
        case alternative_hypothesis::less:      return "less";
        default:                                return "two.sided";
    }
}

} // namespace detail

// ============================================================================
// 1標本t検定のパワー解析 (One-sample t-test power analysis)
// ============================================================================

/**
 * @brief 1標本t検定の検出力を計算 (Calculate power for one-sample t-test)
 *
 * @param effect_size 効果量（Cohen's d）(effect size, Cohen's d)
 * @param n サンプルサイズ (sample size)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param alternative 対立仮説の種類: "two.sided", "greater", "less"（デフォルト: "two.sided"）
 *                    (type of alternative hypothesis, default: "two.sided")
 * @return 検出力 (0.0〜1.0) (statistical power, 0.0 to 1.0)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 *
 * @note この関数は正規分布近似を使用します。厳密には非心t分布を使用すべきですが、
 *       サンプルサイズが大きい場合（n > 30程度）は十分な精度が得られます。
 *       小サンプルでは検出力がやや過大評価される可能性があります。
 */
inline double power_t_test_one_sample(double effect_size, std::size_t n,
                                      double alpha = 0.05,
                                      const std::string& alternative = "two.sided")
{
    if (n == 0) {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: sample size must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: alpha must be in (0, 1)");
    }

    double ncp = detail::noncentrality_parameter_t(effect_size, static_cast<double>(n));

    if (alternative == "two.sided") {
        // 両側検定: P(|T| > t_crit | ncp)
        // 簡易近似: 正規分布で近似
        double z_crit = detail::critical_z_two_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_t_test_one_sample: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief 1標本t検定の必要サンプルサイズを計算 (Calculate required sample size for one-sample t-test)
 *
 * @param effect_size 効果量（Cohen's d）(effect size, Cohen's d)
 * @param power 目標検出力（デフォルト: 0.80）(target power, default: 0.80)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param alternative 対立仮説の種類（デフォルト: "two.sided"）(type of alternative hypothesis, default: "two.sided")
 * @return 必要なサンプルサイズ (required sample size)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
inline std::size_t sample_size_t_test_one_sample(double effect_size, double power = 0.80,
                                                  double alpha = 0.05,
                                                  const std::string& alternative = "two.sided")
{
    if (effect_size == 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: effect size must be non-zero");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_one_sample: alpha must be in (0, 1)");
    }

    // 正規分布近似を使用した初期推定
    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    // 初期推定: n = ((z_alpha + z_beta) / d)^2
    double n_approx = std::pow((z_alpha + z_beta) / std::abs(effect_size), 2.0);

    // 最低サンプルサイズ
    std::size_t n = std::max(2.0, std::ceil(n_approx));

    // 反復法で精密化（最大100回）
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_t_test_one_sample(effect_size, n, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n++;
    }

    return n;
}

// ============================================================================
// 2標本t検定のパワー解析 (Two-sample t-test power analysis)
// ============================================================================

/**
 * @brief 2標本t検定の検出力を計算 (Calculate power for two-sample t-test)
 *
 * @param effect_size 効果量（Cohen's d）(effect size, Cohen's d)
 * @param n1 群1のサンプルサイズ (sample size of group 1)
 * @param n2 群2のサンプルサイズ (sample size of group 2)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param alternative 対立仮説の種類（デフォルト: "two.sided"）(type of alternative hypothesis, default: "two.sided")
 * @return 検出力 (0.0〜1.0) (statistical power, 0.0 to 1.0)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 *
 * @note この関数は正規分布近似を使用します。厳密には非心t分布を使用すべきですが、
 *       合計サンプルサイズが大きい場合（n1 + n2 > 60程度）は十分な精度が得られます。
 *       小サンプルでは検出力がやや過大評価される可能性があります。
 */
inline double power_t_test_two_sample(double effect_size, std::size_t n1, std::size_t n2,
                                      double alpha = 0.05,
                                      const std::string& alternative = "two.sided")
{
    if (n1 == 0 || n2 == 0) {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: sample sizes must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: alpha must be in (0, 1)");
    }

    double ncp = detail::noncentrality_parameter_t_two_sample(effect_size,
                                                                static_cast<double>(n1),
                                                                static_cast<double>(n2));

    if (alternative == "two.sided") {
        double z_crit = detail::critical_z_two_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_t_test_two_sample: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief 2標本t検定の必要サンプルサイズを計算（各群のサイズ）
 *        (Calculate required sample size for two-sample t-test, per group)
 *
 * @param effect_size 効果量（Cohen's d）(effect size, Cohen's d)
 * @param power 目標検出力（デフォルト: 0.80）(target power, default: 0.80)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param ratio n2/n1の比率（デフォルト: 1.0 = 等サイズ）
 *              (ratio n2/n1, default: 1.0 = equal sizes)
 * @param alternative 対立仮説の種類（デフォルト: "two.sided"）
 *                    (type of alternative hypothesis, default: "two.sided")
 * @return 群1の必要なサンプルサイズ（群2のサイズは n1 * ratio）
 *         (required sample size for group 1, group 2 size is n1 * ratio)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
inline std::size_t sample_size_t_test_two_sample(double effect_size, double power = 0.80,
                                                  double alpha = 0.05, double ratio = 1.0,
                                                  const std::string& alternative = "two.sided")
{
    if (effect_size == 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: effect size must be non-zero");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: alpha must be in (0, 1)");
    }
    if (ratio <= 0.0) {
        throw std::invalid_argument("statcpp::sample_size_t_test_two_sample: ratio must be positive");
    }

    // 正規分布近似
    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    // 初期推定: n1 = ((z_alpha + z_beta) / d)^2 * (1 + 1/r)
    double n1_approx = std::pow((z_alpha + z_beta) / std::abs(effect_size), 2.0) * (1.0 + 1.0 / ratio);

    std::size_t n1 = std::max(2.0, std::ceil(n1_approx));
    std::size_t n2 = std::max(2.0, std::ceil(n1 * ratio));

    // 反復法で精密化
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_t_test_two_sample(effect_size, n1, n2, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n1++;
        n2 = std::max(2.0, std::ceil(n1 * ratio));
    }

    return n1;
}

// ============================================================================
// 比率検定のパワー解析 (Proportion test power analysis)
// ============================================================================

/**
 * @brief 2標本比率検定の検出力を計算 (Calculate power for two-sample proportion test)
 *
 * @param p1 群1の比率 (proportion of group 1)
 * @param p2 群2の比率 (proportion of group 2)
 * @param n サンプルサイズ（各群）(sample size per group)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param alternative 対立仮説の種類（デフォルト: "two.sided"）
 *                    (type of alternative hypothesis, default: "two.sided")
 * @return 検出力 (0.0〜1.0) (statistical power, 0.0 to 1.0)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
inline double power_prop_test(double p1, double p2, std::size_t n,
                              double alpha = 0.05,
                              const std::string& alternative = "two.sided")
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::power_prop_test: proportions must be in [0, 1]");
    }
    if (n == 0) {
        throw std::invalid_argument("statcpp::power_prop_test: sample size must be positive");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::power_prop_test: alpha must be in (0, 1)");
    }

    // 標準誤差（対立仮説下）
    double se_alt = std::sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n);

    double diff = p1 - p2;

    if (alternative == "two.sided") {
        double z_crit = detail::critical_z_two_sided(alpha);
        double ncp = diff / se_alt;
        return 1.0 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp);
    } else if (alternative == "greater") {
        double z_crit = detail::critical_z_one_sided(alpha);
        double ncp = diff / se_alt;
        return 1.0 - norm_cdf(z_crit - ncp);
    } else if (alternative == "less") {
        double z_crit = detail::critical_z_one_sided(alpha);
        double ncp = diff / se_alt;
        return norm_cdf(-z_crit - ncp);
    } else {
        throw std::invalid_argument("statcpp::power_prop_test: alternative must be 'two.sided', 'greater', or 'less'");
    }
}

/**
 * @brief 2標本比率検定の必要サンプルサイズを計算 (Calculate required sample size for two-sample proportion test)
 *
 * @param p1 群1の比率 (proportion of group 1)
 * @param p2 群2の比率 (proportion of group 2)
 * @param power 目標検出力（デフォルト: 0.80）(target power, default: 0.80)
 * @param alpha 有意水準（デフォルト: 0.05）(significance level, default: 0.05)
 * @param alternative 対立仮説の種類（デフォルト: "two.sided"）
 *                    (type of alternative hypothesis, default: "two.sided")
 * @return 各群の必要なサンプルサイズ (required sample size per group)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 */
inline std::size_t sample_size_prop_test(double p1, double p2, double power = 0.80,
                                         double alpha = 0.05,
                                         const std::string& alternative = "two.sided")
{
    if (p1 < 0.0 || p1 > 1.0 || p2 < 0.0 || p2 > 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: proportions must be in [0, 1]");
    }
    if (std::abs(p1 - p2) < 1e-10) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: proportions must be different");
    }
    if (power <= 0.0 || power >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: power must be in (0, 1)");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("statcpp::sample_size_prop_test: alpha must be in (0, 1)");
    }

    double z_alpha, z_beta;
    if (alternative == "two.sided") {
        z_alpha = detail::critical_z_two_sided(alpha);
    } else {
        z_alpha = detail::critical_z_one_sided(alpha);
    }
    z_beta = norm_quantile(power);

    double p_pooled = (p1 + p2) / 2.0;
    double diff = std::abs(p1 - p2);

    // 初期推定
    double n_approx = (std::pow(z_alpha * std::sqrt(2.0 * p_pooled * (1.0 - p_pooled)) +
                                z_beta * std::sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2)), 2.0)) /
                      std::pow(diff, 2.0);

    std::size_t n = std::max(2.0, std::ceil(n_approx));

    // 反復法で精密化
    for (int iter = 0; iter < 100; ++iter) {
        double current_power = power_prop_test(p1, p2, n, alpha, alternative);
        if (current_power >= power) {
            break;
        }
        n++;
    }

    return n;
}

// ============================================================================
// パワー解析のラッパー関数（構造体を返す）
// Power analysis wrapper functions (returning structs)
// ============================================================================

/**
 * @brief 1標本t検定のパワー解析（検出力を計算）
 *        (Power analysis for one-sample t-test, calculate power)
 *
 * @param effect_size 効果量 (effect size)
 * @param n サンプルサイズ (sample size)
 * @param alpha 有意水準 (significance level, default: 0.05)
 * @param alternative 対立仮説の種類 (type of alternative hypothesis, default: "two.sided")
 * @return 検出力分析の結果 (power analysis result)
 */
inline power_result power_analysis_t_one_sample(double effect_size, std::size_t n,
                                                 double alpha = 0.05,
                                                 const std::string& alternative = "two.sided")
{
    power_result result;
    result.effect_size = effect_size;
    result.sample_size = static_cast<double>(n);
    result.alpha = alpha;
    result.power = power_t_test_one_sample(effect_size, n, alpha, alternative);
    return result;
}

/**
 * @brief 1標本t検定のパワー解析（サンプルサイズを計算）
 *        (Power analysis for one-sample t-test, calculate sample size)
 *
 * @param effect_size 効果量 (effect size)
 * @param power 目標検出力 (target power, default: 0.80)
 * @param alpha 有意水準 (significance level, default: 0.05)
 * @param alternative 対立仮説の種類 (type of alternative hypothesis, default: "two.sided")
 * @return 検出力分析の結果 (power analysis result)
 */
inline power_result power_analysis_t_one_sample_n(double effect_size, double power = 0.80,
                                                   double alpha = 0.05,
                                                   const std::string& alternative = "two.sided")
{
    power_result result;
    result.effect_size = effect_size;
    result.power = power;
    result.alpha = alpha;
    result.sample_size = static_cast<double>(sample_size_t_test_one_sample(effect_size, power, alpha, alternative));
    return result;
}

// ============================================================================
// 列挙型オーバーロード（文字列APIの型安全な代替）
// Enum overloads (type-safe alternative to string-based API)
// ============================================================================

/**
 * @brief 1標本t検定の検出力を計算（enum版）
 *        (Calculate power for one-sample t-test, enum overload)
 */
inline double power_t_test_one_sample(double effect_size, std::size_t n,
                                      double alpha,
                                      alternative_hypothesis alt)
{
    return power_t_test_one_sample(effect_size, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 1標本t検定の必要サンプルサイズを計算（enum版）
 *        (Calculate required sample size for one-sample t-test, enum overload)
 */
inline std::size_t sample_size_t_test_one_sample(double effect_size, double power,
                                                  double alpha,
                                                  alternative_hypothesis alt)
{
    return sample_size_t_test_one_sample(effect_size, power, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 2標本t検定の検出力を計算（enum版）
 *        (Calculate power for two-sample t-test, enum overload)
 */
inline double power_t_test_two_sample(double effect_size, std::size_t n1, std::size_t n2,
                                      double alpha,
                                      alternative_hypothesis alt)
{
    return power_t_test_two_sample(effect_size, n1, n2, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 2標本t検定の必要サンプルサイズを計算（enum版）
 *        (Calculate required sample size for two-sample t-test, enum overload)
 */
inline std::size_t sample_size_t_test_two_sample(double effect_size, double power,
                                                  double alpha, double ratio,
                                                  alternative_hypothesis alt)
{
    return sample_size_t_test_two_sample(effect_size, power, alpha, ratio, detail::alternative_to_string(alt));
}

/**
 * @brief 2標本比率検定の検出力を計算（enum版）
 *        (Calculate power for two-sample proportion test, enum overload)
 */
inline double power_prop_test(double p1, double p2, std::size_t n,
                              double alpha,
                              alternative_hypothesis alt)
{
    return power_prop_test(p1, p2, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 2標本比率検定の必要サンプルサイズを計算（enum版）
 *        (Calculate required sample size for two-sample proportion test, enum overload)
 */
inline std::size_t sample_size_prop_test(double p1, double p2, double power,
                                         double alpha,
                                         alternative_hypothesis alt)
{
    return sample_size_prop_test(p1, p2, power, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 1標本t検定のパワー解析（enum版、構造体を返す）
 *        (Power analysis for one-sample t-test, enum overload, returning struct)
 */
inline power_result power_analysis_t_one_sample(double effect_size, std::size_t n,
                                                 double alpha,
                                                 alternative_hypothesis alt)
{
    return power_analysis_t_one_sample(effect_size, n, alpha, detail::alternative_to_string(alt));
}

/**
 * @brief 1標本t検定のパワー解析・サンプルサイズ計算（enum版、構造体を返す）
 *        (Power analysis for one-sample t-test, sample size, enum overload, returning struct)
 */
inline power_result power_analysis_t_one_sample_n(double effect_size, double power,
                                                   double alpha,
                                                   alternative_hypothesis alt)
{
    return power_analysis_t_one_sample_n(effect_size, power, alpha, detail::alternative_to_string(alt));
}

} // namespace statcpp
