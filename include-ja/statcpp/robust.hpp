/**
 * @file robust.hpp
 * @brief ロバスト統計の実装 (Robust statistics implementation)
 *
 * MAD、外れ値検出、ウィンザー化、Cook距離、ロバスト推定量などを提供します。
 * Provides MAD, outlier detection, winsorization, Cook's distance, robust estimators, and more.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/order_statistics.hpp"

namespace statcpp {

// ============================================================================
// Median Absolute Deviation (MAD)
// ============================================================================

/**
 * @brief 中央絶対偏差 (Median Absolute Deviation)
 *
 * 中央値からの絶対偏差の中央値を計算します。
 * Computes the median of absolute deviations from the median.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @return MAD 値 (MAD value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note ロバストな分散の推定量として使用されます。
 * Used as a robust estimator of variance.
 */
template <typename Iterator>
double mad(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mad: empty range");
    }

    // データをコピーしてソート（中央値計算のため）
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    double med = statcpp::median(sorted_data.begin(), sorted_data.end());

    // 中央値からの絶対偏差を計算
    std::vector<double> abs_deviations;
    abs_deviations.reserve(n);
    for (double val : sorted_data) {
        abs_deviations.push_back(std::abs(val - med));
    }

    // 絶対偏差の中央値
    std::sort(abs_deviations.begin(), abs_deviations.end());
    return statcpp::median(abs_deviations.begin(), abs_deviations.end());
}

/**
 * @brief 正規分布に対するスケーリングされた MAD (Scaled MAD for normal distribution)
 *
 * 正規分布の標準偏差の推定量として使用できるようスケーリングした MAD を計算します。
 * Computes MAD scaled to estimate standard deviation for normal distribution.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @return スケーリングされた MAD 値 (scaled MAD value)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note スケール係数の導出：
 *       標準正規分布 N(0,1) において、中央値からの絶対偏差 |X - 0| = |X| の
 *       中央値は Φ⁻¹(0.75) ≈ 0.6745 です（Φ⁻¹ は標準正規分布の分位点関数）。
 *       したがって、MAD を標準偏差 σ の推定量として使用するには：
 *       σ ≈ MAD / 0.6745 = MAD × 1.4826
 *
 *       この係数はデータが正規分布に従う場合にのみ厳密に成り立ちます。
 *       非正規分布のデータに対しては、この係数による標準偏差の推定は
 *       近似値となります。
 *
 *       Scale factor derivation: For standard normal N(0,1), the median of |X|
 *       is Φ⁻¹(0.75) ≈ 0.6745, so σ ≈ MAD × 1.4826.
 */
template <typename Iterator>
double mad_scaled(Iterator first, Iterator last)
{
    return 1.4826 * mad(first, last);
}

// ============================================================================
// Outlier Detection (IQR Method / Tukey's Fences)
// ============================================================================

/**
 * @brief 外れ値検出の結果 (Outlier detection result)
 */
struct outlier_detection_result {
    std::vector<double> outliers;                  ///< 外れ値 (outliers)
    std::vector<std::size_t> outlier_indices;      ///< 外れ値のインデックス (outlier indices)
    double lower_fence;                            ///< 下限フェンス (lower fence)
    double upper_fence;                            ///< 上限フェンス (upper fence)
    double q1;                                     ///< 第1四分位数 (first quartile)
    double q3;                                     ///< 第3四分位数 (third quartile)
    double iqr_value;                              ///< 四分位範囲 (interquartile range)
};

/**
 * @brief IQR 法による外れ値検出 (Outlier detection using IQR method / Tukey's Fences)
 *
 * 四分位範囲を用いて外れ値を検出します（Tukey's Fences）。
 * Detects outliers using the interquartile range (Tukey's Fences).
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param k フェンスの倍率（デフォルト 1.5、極端な外れ値には 3.0）
 *          (fence multiplier, default: 1.5, use 3.0 for extreme outliers)
 * @return 外れ値検出結果 (outlier detection result)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note 標準的な箱ひげ図では k=1.5 を使用します。
 * Standard box plots use k=1.5.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_iqr(Iterator first, Iterator last, double k = 1.5)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::detect_outliers_iqr: empty range");
    }

    // データをコピーしてソート
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    auto q = statcpp::quartiles(sorted_data.begin(), sorted_data.end());
    double iqr_val = q.q3 - q.q1;

    double lower_fence = q.q1 - k * iqr_val;
    double upper_fence = q.q3 + k * iqr_val;

    outlier_detection_result result;
    result.lower_fence = lower_fence;
    result.upper_fence = upper_fence;
    result.q1 = q.q1;
    result.q3 = q.q3;
    result.iqr_value = iqr_val;

    // 外れ値を検出
    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        if (val < lower_fence || val > upper_fence) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

// ============================================================================
// Z-score Outlier Detection
// ============================================================================

/**
 * @brief Z-score による外れ値検出 (Outlier detection using Z-score)
 *
 * Z-score（標準化された値）を用いて外れ値を検出します。
 * Detects outliers using Z-score (standardized values).
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param threshold Z-score の閾値（デフォルト 3.0）
 *                  (Z-score threshold, default: 3.0)
 * @return 外れ値検出結果 (outlier detection result)
 * @throws std::invalid_argument 範囲が2未満、または標準偏差が0の場合
 *         (if range has less than 2 elements or standard deviation is zero)
 *
 * @note 正規分布を仮定します。外れ値に対して敏感です。
 * Assumes normal distribution. Sensitive to outliers.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_zscore(Iterator first, Iterator last, double threshold = 3.0)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 2) {
        throw std::invalid_argument("statcpp::detect_outliers_zscore: need at least 2 elements");
    }

    double m = statcpp::mean(first, last);
    double s = statcpp::sample_stddev(first, last, m);

    if (s == 0.0) {
        throw std::invalid_argument("statcpp::detect_outliers_zscore: zero standard deviation");
    }

    outlier_detection_result result;
    result.lower_fence = m - threshold * s;
    result.upper_fence = m + threshold * s;
    result.q1 = 0.0;  // Z-score 法では四分位数は使用しない
    result.q3 = 0.0;
    result.iqr_value = 0.0;

    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        double z = (val - m) / s;
        if (std::abs(z) > threshold) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

/**
 * @brief Modified Z-score による外れ値検出 (Outlier detection using Modified Z-score)
 *
 * MAD ベースの Modified Z-score を用いて外れ値を検出します（より堅牢）。
 * Detects outliers using MAD-based Modified Z-score (more robust).
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param threshold Modified Z-score の閾値（デフォルト 3.5）
 *                  (Modified Z-score threshold, default: 3.5)
 * @return 外れ値検出結果 (outlier detection result)
 * @throws std::invalid_argument 範囲が空、または MAD が 0 の場合
 *         (if range is empty or MAD is zero)
 *
 * @note 通常の Z-score よりもロバストです。外れ値の影響を受けにくいです。
 * More robust than standard Z-score. Less affected by outliers.
 */
template <typename Iterator>
outlier_detection_result detect_outliers_modified_zscore(Iterator first, Iterator last, double threshold = 3.5)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::detect_outliers_modified_zscore: empty range");
    }

    // データをコピーしてソート
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    double med = statcpp::median(sorted_data.begin(), sorted_data.end());
    double mad_val = mad(first, last);

    if (mad_val == 0.0) {
        throw std::invalid_argument("statcpp::detect_outliers_modified_zscore: zero MAD");
    }

    // Modified Z-score = 0.6745 * (x - median) / MAD
    double scale = 0.6745;

    outlier_detection_result result;
    result.lower_fence = med - threshold * mad_val / scale;
    result.upper_fence = med + threshold * mad_val / scale;
    result.q1 = 0.0;
    result.q3 = 0.0;
    result.iqr_value = 0.0;

    std::size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
        double val = static_cast<double>(*it);
        double modified_z = scale * (val - med) / mad_val;
        if (std::abs(modified_z) > threshold) {
            result.outliers.push_back(val);
            result.outlier_indices.push_back(idx);
        }
    }

    return result;
}

// ============================================================================
// Winsorization
// ============================================================================

/**
 * @brief ウィンザー化 (Winsorization)
 *
 * 極端値を一定パーセンタイルの値に置き換えます。
 * Replaces extreme values with specified percentile values.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param limits 両端から置き換える割合（例: 0.05 = 上下 5% を置き換え）
 *               (proportion to replace from each tail, e.g., 0.05 = replace top and bottom 5%)
 * @return ウィンザー化されたデータ (winsorized data)
 * @throws std::invalid_argument 範囲が空、または limits が無効な場合
 *         (if range is empty or limits is invalid)
 *
 * @note 外れ値の影響を軽減しながらデータポイント数を保持します。
 * Reduces the impact of outliers while preserving the number of data points.
 */
template <typename Iterator>
std::vector<double> winsorize(Iterator first, Iterator last, double limits = 0.05)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::winsorize: empty range");
    }
    if (limits < 0.0 || limits >= 0.5) {
        throw std::invalid_argument("statcpp::winsorize: limits must be in [0, 0.5)");
    }

    // データをコピーしてソート
    std::vector<double> sorted_data;
    sorted_data.reserve(n);
    for (auto it = first; it != last; ++it) {
        sorted_data.push_back(static_cast<double>(*it));
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    // 閾値を計算
    double lower_percentile = limits;
    double upper_percentile = 1.0 - limits;
    double lower_value = statcpp::percentile(sorted_data.begin(), sorted_data.end(), lower_percentile);
    double upper_value = statcpp::percentile(sorted_data.begin(), sorted_data.end(), upper_percentile);

    // ウィンザー化
    std::vector<double> result;
    result.reserve(n);
    for (auto it = first; it != last; ++it) {
        double val = static_cast<double>(*it);
        if (val < lower_value) {
            result.push_back(lower_value);
        } else if (val > upper_value) {
            result.push_back(upper_value);
        } else {
            result.push_back(val);
        }
    }

    return result;
}

// ============================================================================
// Cook's Distance (for Linear Regression Diagnostics)
// ============================================================================

/**
 * @brief Cook の距離を計算 (Calculate Cook's Distance)
 *
 * 線形回帰における影響力のある観測値を検出するための Cook の距離を計算します。
 * Calculates Cook's Distance to detect influential observations in linear regression.
 *
 * @param residuals 残差 (residuals)
 * @param hat_values てこ比（レバレッジ）(leverage values / hat values)
 * @param mse 平均二乗誤差 (mean squared error)
 * @param p パラメータ数（切片を含む）(number of parameters including intercept)
 * @return Cook の距離 (Cook's distances)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 *
 * @note D_i > 1 の場合、その観測値は影響力が大きいと判断されます。
 * D_i > 1 indicates the observation is influential.
 * @note D_i > 4/n の場合も影響力があると判断されることがあります。
 * D_i > 4/n is also sometimes used as a threshold.
 */
inline std::vector<double> cooks_distance(
    const std::vector<double>& residuals,
    const std::vector<double>& hat_values,
    double mse,
    std::size_t p)
{
    if (residuals.size() != hat_values.size()) {
        throw std::invalid_argument("statcpp::cooks_distance: residuals and hat_values must have same length");
    }
    if (residuals.empty()) {
        throw std::invalid_argument("statcpp::cooks_distance: empty data");
    }
    if (mse <= 0) {
        throw std::invalid_argument("statcpp::cooks_distance: mse must be positive");
    }
    if (p == 0) {
        throw std::invalid_argument("statcpp::cooks_distance: p must be positive");
    }

    std::vector<double> result;
    result.reserve(residuals.size());

    for (std::size_t i = 0; i < residuals.size(); ++i) {
        double h = hat_values[i];
        if (h >= 1.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        double e = residuals[i];
        double d = (e * e / (static_cast<double>(p) * mse)) *
                   (h / ((1.0 - h) * (1.0 - h)));
        result.push_back(d);
    }

    return result;
}

// ============================================================================
// DFFITS (Difference in Fits)
// ============================================================================

/**
 * @brief DFFITS を計算 (Calculate DFFITS)
 *
 * 各観測値が予測値に与える影響を測定する DFFITS を計算します。
 * Calculates DFFITS to measure the influence of each observation on predicted values.
 *
 * @param residuals 残差 (residuals)
 * @param hat_values てこ比 (leverage values / hat values)
 * @param mse 平均二乗誤差 (mean squared error)
 * @return DFFITS 値 (DFFITS values)
 * @throws std::invalid_argument パラメータが無効な場合 (if parameters are invalid)
 *
 * @note |DFFITS_i| > 2√(p/n) の場合、その観測値は影響力が大きいと判断されます。
 * |DFFITS_i| > 2√(p/n) indicates the observation is influential.
 */
inline std::vector<double> dffits(
    const std::vector<double>& residuals,
    const std::vector<double>& hat_values,
    double mse)
{
    if (residuals.size() != hat_values.size()) {
        throw std::invalid_argument("statcpp::dffits: residuals and hat_values must have same length");
    }
    if (residuals.empty()) {
        throw std::invalid_argument("statcpp::dffits: empty data");
    }

    std::vector<double> result;
    result.reserve(residuals.size());

    for (std::size_t i = 0; i < residuals.size(); ++i) {
        double h = hat_values[i];
        if (h >= 1.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        double e = residuals[i];
        // スチューデント化残差
        double se = std::sqrt(mse * (1.0 - h));
        if (se == 0.0) {
            result.push_back(std::numeric_limits<double>::infinity());
            continue;
        }
        double t = e / se;

        // DFFITS
        double dffits_val = t * std::sqrt(h / (1.0 - h));
        result.push_back(dffits_val);
    }

    return result;
}

// ============================================================================
// Robust Location Estimators
// ============================================================================

/**
 * @brief Hodges-Lehmann 推定量 (Hodges-Lehmann estimator)
 *
 * すべてのペアの平均の中央値を計算します（Walsh 平均）。
 * Calculates the median of all pairwise averages (Walsh average).
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @return Hodges-Lehmann 推定値 (Hodges-Lehmann estimate)
 * @throws std::invalid_argument 範囲が空の場合 (if range is empty)
 *
 * @note ロバストな位置推定量です。外れ値の影響を受けにくいです。
 * A robust location estimator. Less affected by outliers.
 */
template <typename Iterator>
double hodges_lehmann(Iterator first, Iterator last)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::hodges_lehmann: empty range");
    }

    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }

    // すべてのペアの平均を計算
    std::vector<double> pairwise_means;
    pairwise_means.reserve(n * (n + 1) / 2);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            pairwise_means.push_back((data[i] + data[j]) / 2.0);
        }
    }

    std::sort(pairwise_means.begin(), pairwise_means.end());
    return statcpp::median(pairwise_means.begin(), pairwise_means.end());
}

/**
 * @brief Biweight Midvariance (Biweight Midvariance)
 *
 * ロバストな分散推定量である Biweight Midvariance を計算します。
 * Calculates the Biweight Midvariance, a robust variance estimator.
 *
 * @tparam Iterator イテレータ型 (iterator type)
 * @param first 範囲の開始 (beginning of range)
 * @param last 範囲の終了 (end of range)
 * @param c チューニング定数（デフォルト 9.0）(tuning constant, default: 9.0)
 * @return Biweight Midvariance (biweight midvariance)
 * @throws std::invalid_argument 範囲が2未満の場合 (if range has less than 2 elements)
 *
 * @note 外れ値の影響を受けにくい分散推定量です。
 * A variance estimator less affected by outliers.
 */
template <typename Iterator>
double biweight_midvariance(Iterator first, Iterator last, double c = 9.0)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 2) {
        throw std::invalid_argument("statcpp::biweight_midvariance: need at least 2 elements");
    }

    // データをコピーしてソート
    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }
    std::sort(data.begin(), data.end());

    double med = statcpp::median(data.begin(), data.end());
    double mad_val = mad(first, last);

    if (mad_val == 0.0) {
        return 0.0;
    }

    double num = 0.0;
    double den = 0.0;
    std::size_t count = 0;

    for (double x : data) {
        double u = (x - med) / (c * mad_val);
        if (std::abs(u) < 1.0) {
            double u2 = u * u;
            double w = (1.0 - u2);
            double w2 = w * w;
            double w4 = w2 * w2;
            num += (x - med) * (x - med) * w4;
            den += w2 * (1.0 - 5.0 * u2);
            count++;
        }
    }

    if (den == 0.0 || count < 2) {
        return 0.0;
    }

    return static_cast<double>(n) * num / (den * den);
}

} // namespace statcpp
