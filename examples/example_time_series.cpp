/**
 * @file example_time_series.cpp
 * @brief Sample code for time series analysis
 *
 * Demonstrates usage examples of time series analysis methods including
 * autocorrelation function (ACF), partial autocorrelation function (PACF),
 * moving average, and exponential smoothing.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include "statcpp/time_series.hpp"

// ============================================================================
// Helper functions for displaying results
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Autocorrelation Function (ACF)
    // ============================================================================
    print_section("1. Autocorrelation Function (ACF)");

    std::cout << R"(
[Concept]
Correlation between time series values and their past values
Calculate correlation coefficient for each lag (time difference)

[Example: Detecting Seasonal Patterns]
For data with periodic patterns,
check which lags have high correlation
)";

    // Simple time series data (contains seasonal pattern)
    std::vector<double> ts_data;
    for (int i = 0; i < 40; ++i) {
        double value = 10.0 + 5.0 * std::sin(2.0 * 3.14159 * i / 12.0) +
                       ((i % 3 == 0) ? 2.0 : -0.5);
        ts_data.push_back(value);
    }

    std::size_t max_lag = 10;
    auto acf_values = statcpp::acf(ts_data.begin(), ts_data.end(), max_lag);

    print_subsection("Autocorrelation at Each Lag");
    std::cout << "  Lag    ACF Value\n";
    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        std::cout << std::setw(5) << lag << "  " << std::setw(8) << acf_values[lag];

        // Significance guideline (+/-2/sqrt(n))
        double significance_level = 2.0 / std::sqrt(static_cast<double>(ts_data.size()));
        if (std::abs(acf_values[lag]) > significance_level && lag > 0) {
            std::cout << "  *";
        }
        std::cout << std::endl;
    }

    std::cout << "\n* indicates significant autocorrelation\n";
    std::cout << "Significance boundary: +-" << (2.0 / std::sqrt(static_cast<double>(ts_data.size()))) << "\n";
    std::cout << "-> ACF values beyond this boundary are statistically significant\n";

    // ============================================================================
    // 2. Partial Autocorrelation Function (PACF)
    // ============================================================================
    print_section("2. Partial Autocorrelation Function (PACF)");

    std::cout << R"(
[Concept]
Direct correlation excluding effects of intermediate lags
Used for determining AR model order

[Example: Determining AR Model Order]
Identify which lags have direct influence
)";

    auto pacf_values = statcpp::pacf(ts_data.begin(), ts_data.end(), max_lag);

    print_subsection("Partial Autocorrelation at Each Lag");
    std::cout << "  Lag   PACF Value\n";
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
        std::cout << std::setw(5) << lag << "  " << std::setw(8) << pacf_values[lag - 1];

        double significance_level = 2.0 / std::sqrt(static_cast<double>(ts_data.size()));
        if (std::abs(pacf_values[lag - 1]) > significance_level) {
            std::cout << "  *";
        }
        std::cout << std::endl;
    }

    std::cout << "\n-> PACF is useful for identifying AR (autoregressive) model order\n";
    std::cout << "-> The lag where significant PACF cuts off is a candidate for AR order\n";

    // ============================================================================
    // 3. Moving Average
    // ============================================================================
    print_section("3. Moving Average");

    std::cout << R"(
[Concept]
Calculate average over a fixed period to remove noise
Smooth the trend for easier understanding

[Example: Smoothing Sales Data]
Remove daily fluctuations to see overall trend
)";

    std::vector<double> sales_data = {100, 110, 105, 115, 120, 118, 125, 130, 128, 135};
    std::size_t window = 3;

    auto sma = statcpp::moving_average(sales_data.begin(), sales_data.end(), window);

    print_subsection(std::to_string(window) + "-Period Moving Average");
    std::cout << "  Period  Sales   Moving Avg\n";

    for (std::size_t i = 0; i < sales_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(6) << sales_data[i];
        if (i >= window - 1) {
            std::cout << std::setw(10) << sma[i - window + 1];
        } else {
            std::cout << "        --";
        }
        std::cout << std::endl;
    }
    std::cout << "-> Short-term fluctuations are smoothed, making trend more visible\n";

    // ============================================================================
    // 4. Exponential Moving Average
    // ============================================================================
    print_section("4. Exponential Moving Average (EMA)");

    std::cout << R"(
[Concept]
Average that gives higher weight to recent data
More responsive to recent data than simple moving average

[Example: Emphasizing Recent Trends]
Alpha parameter adjusts responsiveness to recent data
)";

    double alpha = 0.3;  // Smoothing parameter
    auto ema = statcpp::exponential_moving_average(sales_data.begin(), sales_data.end(), alpha);

    print_subsection("Exponential Moving Average (alpha = " + std::to_string(alpha) + ")");
    std::cout << "  Period  Sales    EMA\n";

    for (std::size_t i = 0; i < sales_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1)
                  << std::setw(6) << sales_data[i]
                  << std::setw(8) << ema[i] << std::endl;
    }

    std::cout << "\n-> EMA gives higher weight to recent observations\n";
    std::cout << "-> Larger alpha means more responsive to changes\n";

    // ============================================================================
    // 5. Differencing
    // ============================================================================
    print_section("5. Differencing (for Stationarity)");

    std::cout << R"(
[Concept]
Calculate difference between current and past values
Remove trend to achieve stationarity

[Example: Trend Removal]
Remove trend component from data with
increasing tendency
)";

    // Data with trend
    std::vector<double> trend_data = {100, 102, 105, 109, 114, 120, 127, 135, 144, 154};

    auto diff1 = statcpp::diff(trend_data.begin(), trend_data.end(), 1);

    print_subsection("Comparison of Original Data and First Difference");
    std::cout << "  Period  Original   1st Diff\n";

    for (std::size_t i = 0; i < trend_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(10) << trend_data[i];
        if (i > 0) {
            std::cout << std::setw(10) << diff1[i - 1];
        } else {
            std::cout << "        --";
        }
        std::cout << std::endl;
    }

    std::cout << "\n-> Differencing removes the upward trend\n";
    std::cout << "-> Necessary processing to achieve stationarity\n";

    // ============================================================================
    // 6. Seasonal Differencing
    // ============================================================================
    print_section("6. Seasonal Differencing");

    std::cout << R"(
[Concept]
Calculate difference with values separated by seasonal period
Remove seasonal patterns

[Example: Removing Seasonality from Quarterly Data]
Remove seasonal variation from
data with period=4 (quarterly)
)";

    // Data with seasonality (period=4)
    std::vector<double> seasonal_data = {
        100, 80, 90, 110,  // Q1-Q4 Year 1
        105, 85, 95, 115,  // Q1-Q4 Year 2
        110, 90, 100, 120  // Q1-Q4 Year 3
    };

    std::size_t period = 4;
    auto seasonal_diff = statcpp::seasonal_diff(seasonal_data.begin(), seasonal_data.end(), period);

    print_subsection("Seasonal Data (period = " + std::to_string(period) + ")");
    std::cout << "  Quarter  Value   Seasonal Diff\n";

    for (std::size_t i = 0; i < seasonal_data.size(); ++i) {
        std::cout << std::setw(8) << (i + 1) << std::setw(6) << seasonal_data[i];
        if (i >= period) {
            std::cout << std::setw(11) << seasonal_diff[i - period];
        } else {
            std::cout << "         --";
        }
        std::cout << std::endl;
    }

    std::cout << "\n-> Seasonal differencing removes seasonal patterns\n";
    std::cout << "-> Compares same seasons (quarters) across years\n";

    // ============================================================================
    // 7. Lag Series
    // ============================================================================
    print_section("7. Lag Series");

    std::cout << R"(
[Concept]
Series shifted along the time axis
Align past values with current values for comparison

[Example: Comparison with Past Data]
Align prices from 2 periods ago with current prices for analysis
)";

    std::vector<double> price_data = {100, 102, 101, 103, 105, 104, 106};
    std::size_t lag = 2;

    auto lagged = statcpp::lag(price_data.begin(), price_data.end(), lag);

    print_subsection("Lag-" + std::to_string(lag) + " Series");
    std::cout << "     t   Price  Lag-" << lag << "\n";

    for (std::size_t i = 0; i < price_data.size(); ++i) {
        std::cout << std::setw(6) << (i + 1) << std::setw(7) << price_data[i];
        if (i >= lag) {
            std::cout << std::setw(8) << lagged[i - lag];
        } else {
            std::cout << "      --";
        }
        std::cout << std::endl;
    }
    std::cout << "-> Lag series can be used as explanatory variables in regression\n";

    // ============================================================================
    // 8. Forecast Evaluation Metrics
    // ============================================================================
    print_section("8. Forecast Evaluation Metrics");

    std::cout << R"(
[Concept]
Metrics for quantitatively evaluating forecast accuracy
Measure error between actual and predicted values

[Main Metrics]
- MAE: Mean Absolute Error (robust to outliers)
- RMSE: Root Mean Square Error (emphasizes large errors)
- MAPE: Mean Absolute Percentage Error (relative error)
)";

    std::vector<double> actual = {100, 105, 110, 115, 120};
    std::vector<double> forecast = {98, 107, 108, 116, 122};

    // Calculate error metrics manually
    double mae = 0.0, mse = 0.0, mape = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double error = actual[i] - forecast[i];
        mae += std::abs(error);
        mse += error * error;
        mape += std::abs(error / actual[i]) * 100.0;
    }
    mae /= actual.size();
    mse /= actual.size();
    double rmse = std::sqrt(mse);
    mape /= actual.size();

    print_subsection("Comparison of Forecast and Actual Values");
    std::cout << "  Period  Actual  Forecast   Error\n";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        std::cout << std::setw(6) << (i + 1)
                  << std::setw(8) << actual[i]
                  << std::setw(8) << forecast[i]
                  << std::setw(7) << (actual[i] - forecast[i]) << std::endl;
    }

    print_subsection("Error Metrics");
    std::cout << "  MAE (Mean Absolute Error): " << mae << "\n";
    std::cout << "  MSE (Mean Squared Error): " << mse << "\n";
    std::cout << "  RMSE (Root Mean Squared Error): " << rmse << "\n";
    std::cout << "  MAPE (Mean Absolute Percentage Error): " << mape << "%\n";
    std::cout << "\n-> Lower values indicate better forecast accuracy\n";

    // ============================================================================
    // 9. Practical Example: Sales Forecasting
    // ============================================================================
    print_section("9. Practical Example: Sales Data Analysis");

    std::cout << R"(
[Concept]
Combine time series analysis techniques to analyze real data
Comprehensively evaluate trend, seasonality, and autocorrelation

[Example: Monthly Sales Data]
Extract trend and patterns from
12 months of sales data
)";

    std::vector<double> monthly_sales = {
        120, 118, 125, 130, 128, 135, 140, 138, 145, 150, 148, 155
    };

    // 3-month moving average to understand trend
    auto trend = statcpp::moving_average(monthly_sales.begin(), monthly_sales.end(), 3);

    print_subsection("Monthly Sales Data (12 months) and Trend");
    std::cout << "  Month  Sales  3-Month MA  Trend\n";
    for (std::size_t i = 0; i < monthly_sales.size(); ++i) {
        std::cout << std::setw(4) << (i + 1) << std::setw(6) << monthly_sales[i];
        if (i >= 2) {
            std::cout << std::setw(9) << trend[i - 2];
            if (i >= 3 && trend[i - 2] > trend[i - 3]) {
                std::cout << "    Up";
            } else if (i >= 3 && trend[i - 2] < trend[i - 3]) {
                std::cout << "    Down";
            }
        }
        std::cout << std::endl;
    }

    // Check seasonality with autocorrelation
    auto sales_acf = statcpp::acf(monthly_sales.begin(), monthly_sales.end(), 6);
    print_subsection("Autocorrelation Analysis");
    bool has_pattern = false;
    for (std::size_t i = 1; i < sales_acf.size(); ++i) {
        if (std::abs(sales_acf[i]) > 0.5) {
            std::cout << "  Strong autocorrelation at lag " << i << "\n";
            has_pattern = true;
        }
    }
    if (!has_pattern) {
        std::cout << "  Weak autocorrelation\n";
    }
    std::cout << "-> Upward trend is confirmed\n";

    // ============================================================================
    // 10. Summary: Time Series Analysis Guidelines
    // ============================================================================
    print_section("Summary: Time Series Analysis Guidelines");

    std::cout << R"(
[Time Series Analysis Steps]

1. Data Visualization
   - Plot the series
   - Check for trend, seasonality, outliers

2. Stationarity Check
   - Use ACF/PACF plots
   - Apply differencing if needed

3. Pattern Identification
   - ACF: Determine MA order (q)
   - PACF: Determine AR order (p)
   - Identify seasonal patterns (period)

4. Model Selection
   - AR: PACF cuts off, ACF decays
   - MA: ACF cuts off, PACF decays
   - ARMA: Both decay gradually

5. Forecast Evaluation
   - Use MAE, RMSE, MAPE
   - Cross-validation
   - Out-of-sample testing

[Common Transformations]
+--------------+--------------------------------+
| Transform    | Purpose                        |
+--------------+--------------------------------+
| Differencing | Trend removal                  |
+--------------+--------------------------------+
| Seasonal     | Seasonality removal            |
| Differencing |                                |
+--------------+--------------------------------+
| Log          | Variance stabilization         |
| Transform    |                                |
+--------------+--------------------------------+
| Moving       | Noise smoothing                |
| Average      |                                |
+--------------+--------------------------------+

[Types of Time Series Models]
- AR (Autoregressive): Predict from past values
- MA (Moving Average): Predict from past errors
- ARMA: Combination of AR and MA
- ARIMA: ARMA with differencing
- SARIMA: Considers seasonality

[Practical Tips]
1. Start with simple models
2. Always perform residual diagnostics
3. Compare multiple models
4. Leverage domain knowledge
5. Watch for overfitting

[Application Fields]
- Finance: Stock price forecasting, risk management
- Economics: GDP forecasting, demand forecasting
- Weather: Weather forecasting, climate analysis
- Engineering: Signal processing, anomaly detection
- Business: Sales forecasting, inventory management
)";

    return 0;
}
