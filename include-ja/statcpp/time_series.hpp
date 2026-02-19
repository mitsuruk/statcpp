/**
 * @file time_series.hpp
 * @brief 時系列分析
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "statcpp/basic_statistics.hpp"

namespace statcpp {

// ============================================================================
// Autocorrelation Function (ACF)
// ============================================================================

/**
 * @brief 自己相関係数（lag k）を計算
 *
 * 時系列データの指定されたラグにおける自己相関係数を計算します。
 *
 * @note 正規化について：
 *       この実装ではBox-Jenkins（1976）に従い、全データの分散で正規化します：
 *       ρ(k) = γ(k) / γ(0)
 *       ここで γ(k) = (1/n) Σ(x_t - μ)(x_{t+k} - μ) は自己共分散です。
 *
 *       分母は常に γ(0)（全データの分散）を使用するため、ρ(0) = 1 となり、
 *       すべてのラグで |ρ(k)| ≤ 1 が保証されます。
 *       これはR の acf() や Python statsmodels と同じ方法です。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param lag ラグ（時間差）
 * @return 自己相関係数 [-1, 1]
 * @throws std::invalid_argument 空の範囲の場合、ラグがデータ長以上の場合、または分散が0の場合
 */
template <typename Iterator>
double autocorrelation(Iterator first, Iterator last, std::size_t lag)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::autocorrelation: empty range");
    }
    if (lag >= n) {
        throw std::invalid_argument("statcpp::autocorrelation: lag must be less than n");
    }
    if (lag == 0) {
        return 1.0;  // 自己相関のlag 0は常に1
    }

    double m = statcpp::mean(first, last);

    // 分散（N で割る）
    double var = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - m;
        var += diff * diff;
    }

    if (var == 0.0) {
        throw std::invalid_argument("statcpp::autocorrelation: zero variance");
    }

    // 自己共分散（lag k）
    double cov = 0.0;
    auto it1 = first;
    auto it2 = first;
    std::advance(it2, lag);

    for (std::size_t i = 0; i < n - lag; ++i) {
        cov += (static_cast<double>(*it1) - m) * (static_cast<double>(*it2) - m);
        ++it1;
        ++it2;
    }

    return cov / var;
}

/**
 * @brief 自己相関関数（ACF）を計算（lag 0からmax_lagまで）
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param max_lag 最大ラグ
 * @return 自己相関係数のベクトル
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
std::vector<double> acf(Iterator first, Iterator last, std::size_t max_lag)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::acf: empty range");
    }
    if (max_lag >= n) {
        max_lag = n - 1;
    }

    std::vector<double> result;
    result.reserve(max_lag + 1);

    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        result.push_back(autocorrelation(first, last, lag));
    }

    return result;
}

// ============================================================================
// Partial Autocorrelation Function (PACF)
// ============================================================================

/**
 * @brief 偏自己相関関数（PACF）を計算（Durbin-Levinsonアルゴリズム）
 *
 * Durbin-Levinsonアルゴリズムを使用してPACFを計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param max_lag 最大ラグ
 * @return 偏自己相関係数のベクトル
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator>
std::vector<double> pacf(Iterator first, Iterator last, std::size_t max_lag)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::pacf: empty range");
    }
    if (max_lag >= n) {
        max_lag = n - 1;
    }

    // まず ACF を計算
    auto r = acf(first, last, max_lag);

    std::vector<double> result;
    result.reserve(max_lag + 1);
    result.push_back(1.0);  // PACF(0) = 1

    if (max_lag == 0) return result;

    // Durbin-Levinson アルゴリズム
    std::vector<double> phi(max_lag + 1);
    std::vector<double> phi_prev(max_lag + 1);

    phi[1] = r[1];
    result.push_back(r[1]);
    phi_prev = phi;  // k=2 のループ開始前に phi_prev を phi[1] = r[1] で初期化

    for (std::size_t k = 2; k <= max_lag; ++k) {
        // φ_{k,k} を計算
        double num = r[k];
        double den = 1.0;

        for (std::size_t j = 1; j < k; ++j) {
            num -= phi_prev[j] * r[k - j];
            den -= phi_prev[j] * r[j];
        }

        if (std::abs(den) < 1e-15) {
            phi[k] = 0.0;
        } else {
            phi[k] = num / den;
        }

        // φ_{k,j} を更新
        for (std::size_t j = 1; j < k; ++j) {
            phi[j] = phi_prev[j] - phi[k] * phi_prev[k - j];
        }

        result.push_back(phi[k]);
        phi_prev = phi;
    }

    return result;
}

// ============================================================================
// Forecast Error Metrics
// ============================================================================

/**
 * @brief 平均絶対誤差 (MAE)
 *
 * 実測値と予測値の平均絶対誤差を計算します。
 *
 * @tparam Iterator1 実測値のRandomAccessIterator型
 * @tparam Iterator2 予測値のRandomAccessIterator型
 * @param first1 実測値の開始イテレータ
 * @param last1 実測値の終了イテレータ
 * @param first2 予測値の開始イテレータ
 * @return MAE
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator1, typename Iterator2>
double mae(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    auto n = static_cast<std::size_t>(std::distance(first1, last1));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mae: empty range");
    }

    double sum = 0.0;
    auto it2 = first2;
    for (auto it1 = first1; it1 != last1; ++it1, ++it2) {
        sum += std::abs(static_cast<double>(*it1) - static_cast<double>(*it2));
    }

    return sum / static_cast<double>(n);
}

/**
 * @brief 平均二乗誤差 (MSE)
 *
 * 実測値と予測値の平均二乗誤差を計算します。
 *
 * @tparam Iterator1 実測値のRandomAccessIterator型
 * @tparam Iterator2 予測値のRandomAccessIterator型
 * @param first1 実測値の開始イテレータ
 * @param last1 実測値の終了イテレータ
 * @param first2 予測値の開始イテレータ
 * @return MSE
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator1, typename Iterator2>
double mse(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    auto n = static_cast<std::size_t>(std::distance(first1, last1));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mse: empty range");
    }

    double sum = 0.0;
    auto it2 = first2;
    for (auto it1 = first1; it1 != last1; ++it1, ++it2) {
        double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
        sum += diff * diff;
    }

    return sum / static_cast<double>(n);
}

/**
 * @brief 平方根平均二乗誤差 (RMSE)
 *
 * @tparam Iterator1 実測値のRandomAccessIterator型
 * @tparam Iterator2 予測値のRandomAccessIterator型
 * @param first1 実測値の開始イテレータ
 * @param last1 実測値の終了イテレータ
 * @param first2 予測値の開始イテレータ
 * @return RMSE
 * @throws std::invalid_argument 空の範囲の場合
 */
template <typename Iterator1, typename Iterator2>
double rmse(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    return std::sqrt(mse(first1, last1, first2));
}

/**
 * @brief 平均絶対パーセント誤差 (MAPE)
 *
 * 実測値と予測値の平均絶対パーセント誤差を計算します。
 *
 * @tparam Iterator1 実測値のRandomAccessIterator型
 * @tparam Iterator2 予測値のRandomAccessIterator型
 * @param first1 実測値の開始イテレータ
 * @param last1 実測値の終了イテレータ
 * @param first2 予測値の開始イテレータ
 * @return MAPE（パーセント）
 * @throws std::invalid_argument 空の範囲の場合、またはすべての実測値が0の場合
 */
template <typename Iterator1, typename Iterator2>
double mape(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    auto n = static_cast<std::size_t>(std::distance(first1, last1));
    if (n == 0) {
        throw std::invalid_argument("statcpp::mape: empty range");
    }

    double sum = 0.0;
    std::size_t valid_count = 0;
    auto it2 = first2;
    for (auto it1 = first1; it1 != last1; ++it1, ++it2) {
        double actual = static_cast<double>(*it1);
        if (actual != 0.0) {
            double predicted = static_cast<double>(*it2);
            sum += std::abs((actual - predicted) / actual);
            valid_count++;
        }
    }

    if (valid_count == 0) {
        throw std::invalid_argument("statcpp::mape: all actual values are zero");
    }

    return sum / static_cast<double>(valid_count) * 100.0;
}

// ============================================================================
// Moving Average
// ============================================================================

/**
 * @brief 単純移動平均
 *
 * 指定されたウィンドウサイズで単純移動平均を計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param window ウィンドウサイズ
 * @return 移動平均のベクトル
 * @throws std::invalid_argument 空の範囲の場合、またはウィンドウサイズが無効な場合
 */
template <typename Iterator>
std::vector<double> moving_average(Iterator first, Iterator last, std::size_t window)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::moving_average: empty range");
    }
    if (window == 0 || window > n) {
        throw std::invalid_argument("statcpp::moving_average: invalid window size");
    }

    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }

    std::vector<double> result;
    result.reserve(n - window + 1);

    double sum = 0.0;
    for (std::size_t i = 0; i < window; ++i) {
        sum += data[i];
    }
    result.push_back(sum / static_cast<double>(window));

    for (std::size_t i = window; i < n; ++i) {
        sum += data[i] - data[i - window];
        result.push_back(sum / static_cast<double>(window));
    }

    return result;
}

/**
 * @brief 指数移動平均
 *
 * 平滑化パラメータalphaを使用して指数移動平均を計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param alpha 平滑化パラメータ（0 < alpha <= 1）
 * @return 指数移動平均のベクトル
 * @throws std::invalid_argument 空の範囲の場合、またはalphaが(0, 1]の範囲外の場合
 */
template <typename Iterator>
std::vector<double> exponential_moving_average(Iterator first, Iterator last, double alpha)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        throw std::invalid_argument("statcpp::exponential_moving_average: empty range");
    }
    if (alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("statcpp::exponential_moving_average: alpha must be in (0, 1]");
    }

    std::vector<double> result;
    result.reserve(n);

    auto it = first;
    double ema = static_cast<double>(*it);
    result.push_back(ema);
    ++it;

    for (; it != last; ++it) {
        ema = alpha * static_cast<double>(*it) + (1.0 - alpha) * ema;
        result.push_back(ema);
    }

    return result;
}

// ============================================================================
// Differencing
// ============================================================================

/**
 * @brief 差分系列（1次またはd次差分）
 *
 * 時系列データの差分を計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param order 差分の次数（デフォルト: 1）
 * @return 差分系列のベクトル
 * @throws std::invalid_argument データが差分次数に対して不十分な場合
 */
template <typename Iterator>
std::vector<double> diff(Iterator first, Iterator last, std::size_t order = 1)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= order) {
        throw std::invalid_argument("statcpp::diff: insufficient data for differencing order");
    }

    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }

    for (std::size_t d = 0; d < order; ++d) {
        std::vector<double> new_data;
        new_data.reserve(data.size() - 1);
        for (std::size_t i = 1; i < data.size(); ++i) {
            new_data.push_back(data[i] - data[i - 1]);
        }
        data = std::move(new_data);
    }

    return data;
}

/**
 * @brief 季節差分
 *
 * 季節性を持つ時系列データの季節差分を計算します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param period 季節周期
 * @return 季節差分系列のベクトル
 * @throws std::invalid_argument データが周期に対して不十分な場合
 */
template <typename Iterator>
std::vector<double> seasonal_diff(Iterator first, Iterator last, std::size_t period)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= period) {
        throw std::invalid_argument("statcpp::seasonal_diff: insufficient data for period");
    }

    std::vector<double> data;
    data.reserve(n);
    for (auto it = first; it != last; ++it) {
        data.push_back(static_cast<double>(*it));
    }

    std::vector<double> result;
    result.reserve(n - period);

    for (std::size_t i = period; i < n; ++i) {
        result.push_back(data[i] - data[i - period]);
    }

    return result;
}

// ============================================================================
// Lag
// ============================================================================

/**
 * @brief ラグ系列を生成
 *
 * 時系列データのラグ系列を作成します。
 *
 * @tparam Iterator RandomAccessIterator型
 * @param first 開始イテレータ
 * @param last 終了イテレータ
 * @param k ラグ
 * @return ラグ系列のベクトル
 * @throws std::invalid_argument ラグがデータ長以上の場合
 */
template <typename Iterator>
std::vector<double> lag(Iterator first, Iterator last, std::size_t k)
{
    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (k >= n) {
        throw std::invalid_argument("statcpp::lag: lag exceeds data length");
    }

    std::vector<double> result;
    result.reserve(n - k);

    auto it = first;
    for (std::size_t i = 0; i < n - k; ++i, ++it) {
        result.push_back(static_cast<double>(*it));
    }

    return result;
}

} // namespace statcpp
