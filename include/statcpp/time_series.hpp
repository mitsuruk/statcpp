/**
 * @file time_series.hpp
 * @brief Time series analysis
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
 * @brief Calculate autocorrelation coefficient (lag k)
 *
 * Calculates the autocorrelation coefficient at the specified lag for time series data.
 *
 * @note About normalization:
 *       This implementation normalizes by the variance of all data following Box-Jenkins (1976):
 *       rho(k) = gamma(k) / gamma(0)
 *       where gamma(k) = (1/n) * sum((x_t - mu)(x_{t+k} - mu)) is the autocovariance.
 *
 *       The denominator always uses gamma(0) (variance of all data), so rho(0) = 1
 *       and |rho(k)| <= 1 is guaranteed for all lags.
 *       This is the same method as R's acf() and Python statsmodels.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param lag Lag (time difference)
 * @return Autocorrelation coefficient [-1, 1]
 * @throws std::invalid_argument If range is empty, lag is >= data length, or variance is zero
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
        return 1.0;  // Autocorrelation at lag 0 is always 1
    }

    double m = statcpp::mean(first, last);

    // Variance (divided by N)
    double var = 0.0;
    for (auto it = first; it != last; ++it) {
        double diff = static_cast<double>(*it) - m;
        var += diff * diff;
    }

    if (var == 0.0) {
        throw std::invalid_argument("statcpp::autocorrelation: zero variance");
    }

    // Autocovariance (lag k)
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
 * @brief Calculate autocorrelation function (ACF) (from lag 0 to max_lag)
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param max_lag Maximum lag
 * @return Vector of autocorrelation coefficients
 * @throws std::invalid_argument If range is empty
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
 * @brief Calculate partial autocorrelation function (PACF) (Durbin-Levinson algorithm)
 *
 * Calculates PACF using the Durbin-Levinson algorithm.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param max_lag Maximum lag
 * @return Vector of partial autocorrelation coefficients
 * @throws std::invalid_argument If range is empty
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

    // First calculate ACF
    auto r = acf(first, last, max_lag);

    std::vector<double> result;
    result.reserve(max_lag + 1);
    result.push_back(1.0);  // PACF(0) = 1

    if (max_lag == 0) return result;

    // Durbin-Levinson algorithm
    std::vector<double> phi(max_lag + 1);
    std::vector<double> phi_prev(max_lag + 1);

    phi[1] = r[1];
    result.push_back(r[1]);
    phi_prev = phi;  // Initialize phi_prev with phi[1] = r[1] before k=2 iteration

    for (std::size_t k = 2; k <= max_lag; ++k) {
        // Calculate phi_{k,k}
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

        // Update phi_{k,j}
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
 * @brief Mean Absolute Error (MAE)
 *
 * Calculates the mean absolute error between actual and predicted values.
 *
 * @tparam Iterator1 RandomAccessIterator type for actual values
 * @tparam Iterator2 RandomAccessIterator type for predicted values
 * @param first1 Beginning iterator of actual values
 * @param last1 End iterator of actual values
 * @param first2 Beginning iterator of predicted values
 * @return MAE
 * @throws std::invalid_argument If range is empty
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
 * @brief Mean Squared Error (MSE)
 *
 * Calculates the mean squared error between actual and predicted values.
 *
 * @tparam Iterator1 RandomAccessIterator type for actual values
 * @tparam Iterator2 RandomAccessIterator type for predicted values
 * @param first1 Beginning iterator of actual values
 * @param last1 End iterator of actual values
 * @param first2 Beginning iterator of predicted values
 * @return MSE
 * @throws std::invalid_argument If range is empty
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
 * @brief Root Mean Squared Error (RMSE)
 *
 * @tparam Iterator1 RandomAccessIterator type for actual values
 * @tparam Iterator2 RandomAccessIterator type for predicted values
 * @param first1 Beginning iterator of actual values
 * @param last1 End iterator of actual values
 * @param first2 Beginning iterator of predicted values
 * @return RMSE
 * @throws std::invalid_argument If range is empty
 */
template <typename Iterator1, typename Iterator2>
double rmse(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    return std::sqrt(mse(first1, last1, first2));
}

/**
 * @brief Mean Absolute Percentage Error (MAPE)
 *
 * Calculates the mean absolute percentage error between actual and predicted values.
 *
 * @tparam Iterator1 RandomAccessIterator type for actual values
 * @tparam Iterator2 RandomAccessIterator type for predicted values
 * @param first1 Beginning iterator of actual values
 * @param last1 End iterator of actual values
 * @param first2 Beginning iterator of predicted values
 * @return MAPE (percentage)
 * @throws std::invalid_argument If range is empty or all actual values are zero
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
 * @brief Simple moving average
 *
 * Calculates the simple moving average with the specified window size.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param window Window size
 * @return Vector of moving averages
 * @throws std::invalid_argument If range is empty or window size is invalid
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
 * @brief Exponential moving average
 *
 * Calculates the exponential moving average using smoothing parameter alpha.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param alpha Smoothing parameter (0 < alpha <= 1)
 * @return Vector of exponential moving averages
 * @throws std::invalid_argument If range is empty or alpha is outside (0, 1]
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
 * @brief Difference series (first-order or d-th order differencing)
 *
 * Calculates the difference of time series data.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param order Differencing order (default: 1)
 * @return Vector of differenced series
 * @throws std::invalid_argument If data is insufficient for the differencing order
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
 * @brief Seasonal differencing
 *
 * Calculates the seasonal difference for time series data with seasonality.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param period Seasonal period
 * @return Vector of seasonal differenced series
 * @throws std::invalid_argument If data is insufficient for the period
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
 * @brief Generate lag series
 *
 * Creates a lag series from time series data.
 *
 * @tparam Iterator RandomAccessIterator type
 * @param first Beginning iterator
 * @param last End iterator
 * @param k Lag
 * @return Vector of lag series
 * @throws std::invalid_argument If lag exceeds data length
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
