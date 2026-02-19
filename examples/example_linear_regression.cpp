/**
 * @file example_linear_regression.cpp
 * @brief Sample code for statcpp::linear_regression.hpp
 *
 * This file demonstrates the usage of linear regression analysis functions
 * provided in linear_regression.hpp through practical examples.
 *
 * [Provided Functions]
 * - simple_linear_regression()      : Simple linear regression
 * - multiple_linear_regression()    : Multiple linear regression
 * - predict()                       : Prediction
 * - prediction_interval_simple()    : Prediction interval (simple regression)
 * - confidence_interval_mean()      : Confidence interval for mean (simple regression)
 * - compute_residual_diagnostics()  : Residual diagnostics
 * - compute_vif()                   : Variance Inflation Factor (VIF)
 * - r_squared()                     : Coefficient of determination
 * - adjusted_r_squared()            : Adjusted coefficient of determination
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_linear_regression.cpp -o example_linear_regression
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// statcpp linear regression header
#include "statcpp/linear_regression.hpp"

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

template <typename T>
void print_data(const std::string& label, const std::vector<T>& data) {
    std::cout << label << ": ";
    for (const auto& d : data) std::cout << d << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. simple_linear_regression() - Simple Linear Regression
// ============================================================================

/**
 * @brief Example of simple_linear_regression() usage
 *
 * [Purpose]
 * Simple linear regression builds a model to predict the response variable (y)
 * from one explanatory variable (x).
 *
 * [Formula]
 * y = B0 + B1*x + e
 * - B0: intercept
 * - B1: slope
 * - e: error term
 *
 * [Return Value]
 * simple_regression_result structure:
 * - intercept, slope: regression coefficients
 * - intercept_se, slope_se: standard errors
 * - intercept_t, slope_t: t-statistics
 * - intercept_p, slope_p: p-values
 * - r_squared: coefficient of determination R^2
 * - adj_r_squared: adjusted R^2
 * - residual_se: residual standard error
 * - f_statistic, f_p_value: F-test results
 *
 * [Use Cases]
 * - Modeling relationship between two variables
 * - Building prediction models
 * - Causal relationship analysis (caution needed for proving causation)
 */
void example_simple_regression() {
    print_section("1. simple_linear_regression() - Simple Linear Regression");

    // Relationship between advertising spend and sales
    std::vector<double> ad_spend = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};  // 10k units
    std::vector<double> sales = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450}; // 10k units

    std::cout << "Data: Relationship between advertising spend and sales\n";
    print_data("Ad spend (10k units)", ad_spend);
    print_data("Sales (10k units)", sales);

    auto result = statcpp::simple_linear_regression(
        ad_spend.begin(), ad_spend.end(),
        sales.begin(), sales.end());

    print_subsection("Regression Results");
    std::cout << "Regression equation: Sales = " << result.intercept << " + "
              << result.slope << " * Ad spend\n\n";

    std::cout << "Coefficient Details:\n";
    std::cout << std::setw(12) << "Coef"
              << std::setw(12) << "Estimate"
              << std::setw(12) << "Std Error"
              << std::setw(10) << "t-value"
              << std::setw(12) << "p-value" << "\n";
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::setw(12) << "Intercept"
              << std::setw(12) << result.intercept
              << std::setw(12) << result.intercept_se
              << std::setw(10) << result.intercept_t
              << std::setw(12) << result.intercept_p << "\n";
    std::cout << std::setw(12) << "Slope"
              << std::setw(12) << result.slope
              << std::setw(12) << result.slope_se
              << std::setw(10) << result.slope_t
              << std::setw(12) << result.slope_p << "\n";

    print_subsection("Model Fit");
    std::cout << "R-squared:                   " << result.r_squared << "\n";
    std::cout << "Adjusted R-squared:          " << result.adj_r_squared << "\n";
    std::cout << "Residual standard error:     " << result.residual_se << " (10k units)\n";
    std::cout << "F-statistic:                 " << result.f_statistic << "\n";
    std::cout << "F-test p-value:              " << result.f_p_value << "\n";

    print_subsection("Interpretation");
    std::cout << "- For every 10k increase in ad spend, sales increase by about "
              << result.slope << " (10k units)\n";
    std::cout << "- R^2 = " << result.r_squared
              << " -> Ad spend explains " << (result.r_squared * 100)
              << "% of sales variation\n";
    std::cout << "- p-value < 0.05 -> Slope is statistically significant\n";
}

// ============================================================================
// 2. predict() and Prediction Intervals
// ============================================================================

/**
 * @brief Example of predict() and prediction intervals
 *
 * [Purpose]
 * Predict y for new x values using the model.
 * Prediction interval shows the range within which individual observations fall.
 * Confidence interval shows the range within which the mean y falls.
 */
void example_prediction() {
    print_section("2. predict() and Prediction Intervals");

    std::vector<double> x = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> y = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450};

    auto model = statcpp::simple_linear_regression(
        x.begin(), x.end(),
        y.begin(), y.end());

    // Point prediction
    print_subsection("Point Prediction");
    double x_new = 55.0;  // Ad spend of 55 (10k units)
    double y_pred = statcpp::predict(model, x_new);
    std::cout << "Predicted sales for ad spend of " << x_new << " (10k units): " << y_pred << " (10k units)\n";

    // Prediction interval
    print_subsection("Prediction Interval (95%)");
    auto pred_int = statcpp::prediction_interval_simple(
        model, x.begin(), x.end(), x_new, 0.95);
    std::cout << "Predicted value: " << pred_int.prediction << " (10k units)\n";
    std::cout << "95% Prediction interval: [" << pred_int.lower << ", " << pred_int.upper << "] (10k units)\n";
    std::cout << "Prediction standard error: " << pred_int.se_prediction << " (10k units)\n";
    std::cout << "-> A new observation falls within this interval with 95% probability\n";

    // Confidence interval for mean
    print_subsection("Confidence Interval for Mean (95%)");
    auto conf_int = statcpp::confidence_interval_mean(
        model, x.begin(), x.end(), x_new, 0.95);
    std::cout << "Predicted value: " << conf_int.prediction << " (10k units)\n";
    std::cout << "95% Confidence interval: [" << conf_int.lower << ", " << conf_int.upper << "] (10k units)\n";
    std::cout << "-> Mean sales at ad spend of " << x_new << " falls within this interval with 95% probability\n";

    print_subsection("Prediction Interval vs Confidence Interval");
    std::cout << "Prediction interval: Range for individual observations (wider)\n";
    std::cout << "Confidence interval: Range for mean value (narrower)\n";
}

// ============================================================================
// 3. multiple_linear_regression() - Multiple Linear Regression
// ============================================================================

/**
 * @brief Example of multiple_linear_regression() usage
 *
 * [Purpose]
 * Multiple linear regression builds a model to predict the response variable
 * from multiple explanatory variables.
 *
 * [Formula]
 * y = B0 + B1*x1 + B2*x2 + ... + Bp*xp + e
 *
 * [Notes]
 * - Intercept (B0) is automatically added
 * - X is an n x p matrix (n: observations, p: number of predictors)
 */
void example_multiple_regression() {
    print_section("3. multiple_linear_regression() - Multiple Linear Regression");

    // House price prediction: area, age, distance to station
    // X[i] = {area(sqm), age(years), station distance(min)}
    std::vector<std::vector<double>> X = {
        {60, 10, 5},   {70, 5, 8},   {80, 15, 3},  {65, 8, 10},
        {75, 12, 6},   {90, 3, 4},   {55, 20, 12}, {85, 7, 2},
        {72, 10, 7},   {68, 6, 9},   {95, 2, 5},   {58, 18, 15}
    };
    std::vector<double> price = {
        3200, 3800, 3500, 2900,
        3400, 4500, 2500, 4200,
        3300, 3100, 4800, 2200
    };  // 10k units

    std::cout << "Data: House Price Prediction\n";
    std::cout << "Explanatory variables: Area(sqm), Age(years), Station distance(min)\n";
    std::cout << "Response variable: Price (10k units)\n\n";

    for (std::size_t i = 0; i < X.size(); ++i) {
        std::cout << "  Property " << (i + 1) << ": Area=" << X[i][0]
                  << "sqm, Age=" << X[i][1] << "yrs, Station=" << X[i][2]
                  << "min -> " << price[i] << " (10k units)\n";
    }

    auto result = statcpp::multiple_linear_regression(X, price);

    print_subsection("Regression Results");
    std::cout << "Regression equation: Price = " << result.coefficients[0]
              << " + " << result.coefficients[1] << " * Area"
              << " + " << result.coefficients[2] << " * Age"
              << " + " << result.coefficients[3] << " * Station\n\n";

    std::cout << "Coefficient Details:\n";
    std::cout << std::setw(12) << "Variable"
              << std::setw(12) << "Coef"
              << std::setw(12) << "Std Error"
              << std::setw(10) << "t-value"
              << std::setw(12) << "p-value" << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<std::string> var_names = {"Intercept", "Area", "Age", "Station"};
    for (std::size_t i = 0; i < result.coefficients.size(); ++i) {
        std::cout << std::setw(12) << var_names[i]
                  << std::setw(12) << result.coefficients[i]
                  << std::setw(12) << result.coefficient_se[i]
                  << std::setw(10) << result.t_statistics[i]
                  << std::setw(12) << result.p_values[i] << "\n";
    }

    print_subsection("Model Fit");
    std::cout << "R-squared:                   " << result.r_squared << "\n";
    std::cout << "Adjusted R-squared:          " << result.adj_r_squared << "\n";
    std::cout << "F-statistic:                 " << result.f_statistic << "\n";
    std::cout << "F-test p-value:              " << result.f_p_value << "\n";

    // Prediction for new property
    print_subsection("Price Prediction for New Property");
    std::vector<double> new_property = {75, 8, 6};  // 75sqm, 8yrs old, 6min to station
    double pred_price = statcpp::predict(result, new_property);
    std::cout << "Property: Area=" << new_property[0] << "sqm, Age=" << new_property[1]
              << "yrs, Station=" << new_property[2] << "min\n";
    std::cout << "Predicted price: " << pred_price << " (10k units)\n";

    print_subsection("Coefficient Interpretation");
    std::cout << "- For every 1sqm increase in area, price increases by about " << result.coefficients[1] << " (10k units)\n";
    std::cout << "- For every 1yr increase in age, price changes by about " << result.coefficients[2] << " (10k units)\n";
    std::cout << "- For every 1min increase in station distance, price changes by about " << result.coefficients[3] << " (10k units)\n";
}

// ============================================================================
// 4. compute_residual_diagnostics() - Residual Diagnostics
// ============================================================================

/**
 * @brief Example of compute_residual_diagnostics() usage
 *
 * [Purpose]
 * Residual diagnostics verify whether regression model assumptions are met.
 *
 * [Diagnostic Items]
 * - Residuals: Observed - Predicted
 * - Standardized residuals: Residual / Residual standard error
 * - Studentized residuals: Standardized residuals computed excluding each point
 * - Leverage: Influence of each observation on the model
 * - Cook's distance: Influence measure for outliers
 * - Durbin-Watson statistic: Autocorrelation of residuals
 */
void example_residual_diagnostics() {
    print_section("4. compute_residual_diagnostics() - Residual Diagnostics");

    std::vector<double> x = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<double> y = {150, 180, 210, 250, 280, 310, 330, 370, 400, 450};

    auto model = statcpp::simple_linear_regression(
        x.begin(), x.end(),
        y.begin(), y.end());

    auto diag = statcpp::compute_residual_diagnostics(
        model, x.begin(), x.end(), y.begin(), y.end());

    print_subsection("Residual Details");
    std::cout << std::setw(8) << "X"
              << std::setw(10) << "Y"
              << std::setw(12) << "Residual"
              << std::setw(12) << "Std Resid"
              << std::setw(10) << "Leverage"
              << std::setw(12) << "Cook's D" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << std::setw(8) << x[i]
                  << std::setw(10) << y[i]
                  << std::setw(12) << diag.residuals[i]
                  << std::setw(12) << diag.standardized_residuals[i]
                  << std::setw(10) << diag.hat_values[i]
                  << std::setw(12) << diag.cooks_distance[i] << "\n";
    }

    print_subsection("Diagnostic Statistics");
    std::cout << "Durbin-Watson statistic: " << diag.durbin_watson << "\n";

    print_subsection("Diagnostic Interpretation");
    std::cout << R"(
[Standardized Residuals]
- |Standardized residual| > 2 may be an outlier
- |Standardized residual| > 3 is a serious outlier

[Leverage]
- High leverage indicates large influence on regression
- Guideline: Be cautious if > 2(p+1)/n
  (p: number of predictors, n: observations)

[Cook's Distance]
- Observations with Cook's D > 1 or > 4/n are highly influential
- Removing such observations may significantly change the model

[Durbin-Watson Statistic]
- DW ~ 2: No autocorrelation in residuals (ideal)
- DW < 2: Possible positive autocorrelation
- DW > 2: Possible negative autocorrelation
)";
}

// ============================================================================
// 5. compute_vif() - Variance Inflation Factor
// ============================================================================

/**
 * @brief Example of compute_vif() usage
 *
 * [Purpose]
 * VIF (Variance Inflation Factor) detects multicollinearity among predictors.
 * Multicollinearity makes coefficient estimates unstable.
 *
 * [Formula]
 * VIF_j = 1 / (1 - R^2_j)
 * R^2_j is the R-squared when regressing variable j on other variables
 *
 * [Guidelines]
 * - VIF < 5: No problem
 * - 5 <= VIF < 10: Slight problem
 * - VIF >= 10: Serious multicollinearity
 */
void example_vif() {
    print_section("5. compute_vif() - Variance Inflation Factor (Multicollinearity)");

    // Example with multicollinearity
    // Area (sqm) and number of rooms are strongly correlated
    std::vector<std::vector<double>> X = {
        {60, 2, 10},   {70, 3, 5},    {80, 3, 15},   {65, 2, 8},
        {75, 3, 12},   {90, 4, 3},    {55, 2, 20},   {85, 4, 7},
        {72, 3, 10},   {68, 3, 6},    {95, 4, 2},    {58, 2, 18}
    };  // Area, Rooms, Age

    std::cout << "Explanatory variables: Area(sqm), Rooms, Age(years)\n";
    std::cout << "(Area and Rooms may be strongly correlated)\n\n";

    auto vif = statcpp::compute_vif(X);

    std::vector<std::string> var_names = {"Area", "Rooms", "Age"};

    std::cout << "VIF (Variance Inflation Factor):\n";
    std::cout << std::setw(12) << "Variable" << std::setw(12) << "VIF" << "\n";
    std::cout << std::string(24, '-') << "\n";

    for (std::size_t i = 0; i < vif.size(); ++i) {
        std::cout << std::setw(12) << var_names[i]
                  << std::setw(12) << vif[i] << "\n";
    }

    std::cout << "\n[Criteria]\n";
    std::cout << "- VIF < 5: No problem\n";
    std::cout << "- 5 <= VIF < 10: Slight problem\n";
    std::cout << "- VIF >= 10: Serious multicollinearity\n";

    bool has_problem = false;
    for (std::size_t i = 0; i < vif.size(); ++i) {
        if (vif[i] >= 5.0) {
            std::cout << "\nWarning: " << var_names[i] << " has high VIF ("
                      << vif[i] << ")\n";
            has_problem = true;
        }
    }
    if (!has_problem) {
        std::cout << "\n-> No multicollinearity problems detected\n";
    }
}

// ============================================================================
// 6. R-squared Comparison and Adjustment
// ============================================================================

/**
 * @brief Calculation and comparison of coefficient of determination
 *
 * [R-squared (Coefficient of Determination)]
 * Shows how much of data variation the model explains
 * R^2 = 1 - SS_residual / SS_total
 *
 * [Adjusted R-squared]
 * Adjusts for number of variables
 * Does not necessarily increase when variables are added
 */
void example_r_squared() {
    print_section("6. R-squared and Adjusted R-squared");

    std::vector<double> y_actual = {100, 120, 140, 160, 180};
    std::vector<double> y_pred1 = {105, 115, 145, 155, 180};  // Good prediction
    std::vector<double> y_pred2 = {130, 130, 130, 130, 130};  // Poor prediction (mean only)

    print_data("Actual values", y_actual);
    print_data("Predictions (good model)", y_pred1);
    print_data("Predictions (poor model)", y_pred2);

    double r2_good = statcpp::r_squared(
        y_actual.begin(), y_actual.end(),
        y_pred1.begin(), y_pred1.end());
    double r2_bad = statcpp::r_squared(
        y_actual.begin(), y_actual.end(),
        y_pred2.begin(), y_pred2.end());

    std::cout << "\nR-squared:\n";
    std::cout << "  Good model: " << r2_good << "\n";
    std::cout << "  Poor model: " << r2_bad << "\n";

    std::cout << R"(
[R-squared Interpretation]
- R^2 = 1.0: Perfect prediction
- R^2 = 0.0: Same as predicting the mean
- R^2 < 0:   Worse than predicting the mean

[R-squared vs Adjusted R-squared]
- R^2 always increases when variables are added (overfitting risk)
- Adjusted R^2 includes penalty for unnecessary variables
- Use Adjusted R^2 for model comparison
)";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Linear Regression Analysis Functions");

    std::cout << R"(
+--------------------------------+-----------------------------------------+
| Function                       | Description                             |
+--------------------------------+-----------------------------------------+
| simple_linear_regression()     | Simple regression (y = B0 + B1*x)       |
| multiple_linear_regression()   | Multiple regression (y = B0 + B1*x1 +..)|
| predict()                      | Point prediction                        |
| prediction_interval_simple()   | Prediction interval (individual range)  |
| confidence_interval_mean()     | Confidence interval for mean            |
| compute_residual_diagnostics() | Residual diagnostics                    |
| compute_vif()                  | VIF (multicollinearity check)           |
| r_squared()                    | Coefficient of determination            |
| adjusted_r_squared()           | Adjusted coefficient of determination   |
+--------------------------------+-----------------------------------------+

[Return Value Structures]
- simple_regression_result: Detailed simple regression results
- multiple_regression_result: Detailed multiple regression results
- prediction_interval: Predicted value and interval
- residual_diagnostics: Residual diagnostic information

[Model Evaluation Points]
1. R^2: Model explanatory power (closer to 1 is better)
2. p-value: Statistical significance of coefficients (< 0.05 is significant)
3. Residual diagnostics: Whether model assumptions are met
4. VIF: Multicollinearity check (< 5 is desirable)

[Cautions]
- Correlation does not imply causation
- Extrapolation (prediction outside data range) is unreliable
- Multicollinearity makes coefficients unstable
- If residuals are not normally distributed, consider other methods
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_simple_regression();
    example_prediction();
    example_multiple_regression();
    example_residual_diagnostics();
    example_vif();
    example_r_squared();

    // Display summary
    print_summary();

    return 0;
}
