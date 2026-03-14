/**
 * @file example_dispersion_spread.cpp
 * @brief Sample code for statcpp::dispersion_spread.hpp
 *
 * This file explains the usage of functions that calculate measures of
 * dispersion (spread) provided in dispersion_spread.hpp through practical examples.
 *
 * [Provided Functions]
 * - range()                   : Range (maximum - minimum)
 * - var()                     : Variance (ddof specifiable)
 * - population_variance()     : Population variance (ddof=0)
 * - sample_variance()         : Sample variance (ddof=1)
 * - variance()                : Variance (alias for sample_variance)
 * - stdev()                   : Standard deviation (ddof specifiable)
 * - population_stddev()       : Population standard deviation (ddof=0)
 * - sample_stddev()           : Sample standard deviation (ddof=1)
 * - stddev()                  : Standard deviation (alias for sample_stddev)
 * - coefficient_of_variation(): Coefficient of variation
 * - iqr()                     : Interquartile range (*requires sorted data)
 * - mean_absolute_deviation() : Mean absolute deviation
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_dispersion_spread.cpp -o example_dispersion_spread
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// statcpp dispersion header
#include "statcpp/dispersion_spread.hpp"
#include "statcpp/basic_statistics.hpp"

// ============================================================================
// Helper Functions for Display
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
// 1. range() - Range
// ============================================================================

/**
 * @brief Example usage of range()
 *
 * [Purpose]
 * range() calculates the difference between the maximum and minimum values.
 * It is the simplest measure of data dispersion.
 *
 * [Formula]
 * range = max(x) - min(x)
 *
 * [Use Cases]
 * - Quick overview of overall data spread
 * - Checking acceptable range in quality control
 * - Measuring temperature fluctuation, stock price range
 *
 * [Notes]
 * - Very sensitive to outliers
 * - Does not provide detailed distribution information
 */
void example_range() {
    print_section("1. range() - Range");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("Test scores", scores);

    double r = statcpp::range(scores.begin(), scores.end());
    std::cout << "Range: " << r << " points (highest 95 - lowest 75)\n";

    // Check effect of outliers
    print_subsection("Effect of Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    print_data("With outlier", with_outlier);
    double r_outlier = statcpp::range(with_outlier.begin(), with_outlier.end());
    std::cout << "Range without outlier: " << r << "\n";
    std::cout << "Range with outlier: " << r_outlier << " (greatly expanded by 200)\n";
}

// ============================================================================
// 2. var() and population_variance(), sample_variance() - Variance
// ============================================================================

/**
 * @brief Example usage of var() and related functions
 *
 * [Purpose]
 * Variance measures how spread out the data is from the mean.
 * It is the average of squared deviations from the mean.
 *
 * [Formula]
 * Population variance: sigma^2 = Sum(xi - mu)^2 / N
 * Sample variance: s^2 = Sum(xi - x_bar)^2 / (N - 1)
 *
 * [ddof (Delta Degrees of Freedom)]
 * - ddof = 0: Divide by N (population variance)
 * - ddof = 1: Divide by N-1 (sample variance, unbiased variance)
 *
 * [Use Cases]
 * - Quantifying data variability
 * - Risk measurement (variance of investment returns)
 * - Intermediate calculations for statistical tests
 *
 * [Notes]
 * - Use ddof=1 (unbiased variance) when estimating population from sample
 * - Unit is the square of original data
 */
void example_variance() {
    print_section("2. var() - Variance");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("Test scores", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    std::cout << "Mean: " << mean_val << " points\n\n";

    // var() with ddof
    print_subsection("Difference in ddof");
    double var_population = statcpp::var(scores.begin(), scores.end(), 0);  // ddof=0
    double var_sample = statcpp::var(scores.begin(), scores.end(), 1);      // ddof=1
    std::cout << "Population variance (ddof=0, divide by N=" << scores.size() << "): " << var_population << "\n";
    std::cout << "Sample variance (ddof=1, divide by N-1=" << scores.size() - 1 << "): " << var_sample << "\n";

    // Alias functions
    print_subsection("Alias Functions");
    double pop_var = statcpp::population_variance(scores.begin(), scores.end());
    double samp_var = statcpp::sample_variance(scores.begin(), scores.end());
    double variance_val = statcpp::variance(scores.begin(), scores.end());  // = sample_variance
    std::cout << "population_variance(): " << pop_var << "\n";
    std::cout << "sample_variance():     " << samp_var << "\n";
    std::cout << "variance():            " << variance_val << " (= sample_variance)\n";

    // Using pre-calculated mean
    print_subsection("Using Pre-calculated Mean (Optimization)");
    double var_with_mean = statcpp::var(scores.begin(), scores.end(), mean_val, 1);
    std::cout << "Sample variance reusing mean: " << var_with_mean << "\n";
    std::cout << "-> Avoids recalculating mean when already computed\n";
}

// ============================================================================
// 3. stdev() and population_stddev(), sample_stddev() - Standard Deviation
// ============================================================================

/**
 * @brief Example usage of stdev() and related functions
 *
 * [Purpose]
 * Standard deviation is the square root of variance.
 * It has the same units as the original data, making it easier to interpret.
 *
 * [Formula]
 * Population std dev: sigma = sqrt(Sum(xi - mu)^2 / N)
 * Sample std dev: s = sqrt(Sum(xi - x_bar)^2 / (N - 1))
 *
 * [Use Cases]
 * - Typical magnitude of spread from the mean
 * - In normal distribution, about 68% of data falls within mean +/- 1 sigma
 * - Quality control, deviation score calculation
 *
 * [Notes]
 * - Distinguish between population and sample standard deviation as with variance
 */
void example_stddev() {
    print_section("3. stdev() - Standard Deviation");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("Test scores", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    std::cout << "Mean: " << mean_val << " points\n\n";

    // stdev() with ddof
    print_subsection("Difference in ddof");
    double stdev_pop = statcpp::stdev(scores.begin(), scores.end(), 0);  // ddof=0
    double stdev_samp = statcpp::stdev(scores.begin(), scores.end(), 1); // ddof=1
    std::cout << "Population std dev (ddof=0): " << stdev_pop << " points\n";
    std::cout << "Sample std dev (ddof=1): " << stdev_samp << " points\n";

    // Alias functions
    print_subsection("Alias Functions");
    double pop_sd = statcpp::population_stddev(scores.begin(), scores.end());
    double samp_sd = statcpp::sample_stddev(scores.begin(), scores.end());
    double stddev_val = statcpp::stddev(scores.begin(), scores.end());  // = sample_stddev
    std::cout << "population_stddev(): " << pop_sd << " points\n";
    std::cout << "sample_stddev():     " << samp_sd << " points\n";
    std::cout << "stddev():            " << stddev_val << " points (= sample_stddev)\n";

    // Interpretation example
    print_subsection("Interpreting Standard Deviation");
    std::cout << "Mean " << mean_val << " points +/- Std dev " << stddev_val << " points\n";
    std::cout << "-> About 68% of students fall within " << (mean_val - stddev_val)
              << " to " << (mean_val + stddev_val) << " points (for normal distribution)\n";
}

// ============================================================================
// 4. coefficient_of_variation() - Coefficient of Variation
// ============================================================================

/**
 * @brief Example usage of coefficient_of_variation()
 *
 * [Purpose]
 * Coefficient of Variation (CV) is the standard deviation divided by the mean.
 * It allows comparison of spread across data with different scales.
 *
 * [Formula]
 * CV = sigma / |mu| (returned as ratio; multiply by 100 for %)
 *
 * [Use Cases]
 * - Comparing data with different units or scales
 *   Example: Comparing spread of height (cm) and weight (kg)
 * - Evaluating measurement precision
 * - Comparing stock price volatility
 *
 * [Notes]
 * - Cannot be used when mean is close to 0 (division by zero)
 * - Suitable for ratio data or log-transformed data
 */
void example_coefficient_of_variation() {
    print_section("4. coefficient_of_variation() - Coefficient of Variation");

    // Comparing data with different scales
    std::vector<double> heights = {160, 165, 170, 175, 180};  // cm
    std::vector<double> weights = {50, 55, 60, 65, 70};       // kg

    print_data("Height (cm)", heights);
    print_data("Weight (kg)", weights);

    double height_mean = statcpp::mean(heights.begin(), heights.end());
    double height_sd = statcpp::stddev(heights.begin(), heights.end());
    double height_cv = statcpp::coefficient_of_variation(heights.begin(), heights.end());

    double weight_mean = statcpp::mean(weights.begin(), weights.end());
    double weight_sd = statcpp::stddev(weights.begin(), weights.end());
    double weight_cv = statcpp::coefficient_of_variation(weights.begin(), weights.end());

    std::cout << "\nHeight:\n";
    std::cout << "  Mean: " << height_mean << " cm\n";
    std::cout << "  Std dev: " << height_sd << " cm\n";
    std::cout << "  CV: " << height_cv << " (" << height_cv * 100 << "%)\n";

    std::cout << "\nWeight:\n";
    std::cout << "  Mean: " << weight_mean << " kg\n";
    std::cout << "  Std dev: " << weight_sd << " kg\n";
    std::cout << "  CV: " << weight_cv << " (" << weight_cv * 100 << "%)\n";

    std::cout << "\n-> Cannot compare using std dev due to different units,\n";
    std::cout << "  but CV allows comparison of relative spread\n";
    if (height_cv < weight_cv) {
        std::cout << "  Weight has greater relative spread\n";
    } else {
        std::cout << "  Height has greater relative spread\n";
    }
}

// ============================================================================
// 5. iqr() - Interquartile Range
// ============================================================================

/**
 * @brief Example usage of iqr()
 *
 * [Purpose]
 * Interquartile Range (IQR) is the difference between the third quartile (Q3)
 * and first quartile (Q1). It shows the spread of the middle 50% of data.
 *
 * [Formula]
 * IQR = Q3 - Q1
 *
 * [Important]
 * * Input data must be pre-sorted!
 *
 * [Use Cases]
 * - Measure of spread resistant to outliers
 * - Creating box plots
 * - Outlier detection (1.5 x IQR rule)
 *
 * [Notes]
 * - Only looks at middle 50%, ignores extreme values
 * - Information about distribution tails is lost
 */
void example_iqr() {
    print_section("5. iqr() - Interquartile Range");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // Sort before calculation (important)
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("Sorted data", sorted_scores);

    double iqr_val = statcpp::iqr(sorted_scores.begin(), sorted_scores.end());
    std::cout << "Interquartile Range (IQR): " << iqr_val << " points\n";

    // Check effect of outliers
    print_subsection("Robustness Against Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_outlier = with_outlier;
    std::sort(sorted_outlier.begin(), sorted_outlier.end());
    print_data("Data with outlier (sorted)", sorted_outlier);

    double iqr_outlier = statcpp::iqr(sorted_outlier.begin(), sorted_outlier.end());
    double range_outlier = statcpp::range(sorted_outlier.begin(), sorted_outlier.end());

    std::cout << "IQR without outlier: " << iqr_val << "\n";
    std::cout << "IQR with outlier: " << iqr_outlier << " (Q1-Q3 less affected)\n";
    std::cout << "Range with outlier: " << range_outlier << " (large change)\n";
    std::cout << "-> IQR is robust against outliers\n";

    // Application to outlier detection
    print_subsection("Outlier Detection (1.5 x IQR Rule)");
    double q1 = statcpp::percentile(sorted_outlier.begin(), sorted_outlier.end(), 0.25);
    double q3 = statcpp::percentile(sorted_outlier.begin(), sorted_outlier.end(), 0.75);
    double lower_fence = q1 - 1.5 * iqr_outlier;
    double upper_fence = q3 + 1.5 * iqr_outlier;
    std::cout << "Q1: " << q1 << ", Q3: " << q3 << ", IQR: " << iqr_outlier << "\n";
    std::cout << "Lower fence: " << lower_fence << "\n";
    std::cout << "Upper fence: " << upper_fence << "\n";
    std::cout << "-> 200 points is outside the fence, identified as outlier\n";
}

// ============================================================================
// 6. mean_absolute_deviation() - Mean Absolute Deviation
// ============================================================================

/**
 * @brief Example usage of mean_absolute_deviation()
 *
 * [Purpose]
 * Mean Absolute Deviation (MAD) is the average of absolute differences
 * between each data point and the mean.
 * It is a measure of spread more resistant to outliers than standard deviation.
 *
 * [Formula]
 * MAD = Sum|xi - x_bar| / n
 *
 * [Use Cases]
 * - When a robust measure of spread is needed
 * - Evaluating prediction error (MAE: Mean Absolute Error)
 * - Same units as original data, easy to interpret
 *
 * [Notes]
 * - Standard deviation has more convenient mathematical properties
 * - Statistical tests like ANOVA use standard deviation
 */
void example_mean_absolute_deviation() {
    print_section("6. mean_absolute_deviation() - Mean Absolute Deviation");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("Test scores", scores);

    double mean_val = statcpp::mean(scores.begin(), scores.end());
    double mad = statcpp::mean_absolute_deviation(scores.begin(), scores.end());
    double sd = statcpp::stddev(scores.begin(), scores.end());

    std::cout << "Mean: " << mean_val << " points\n";
    std::cout << "Mean Absolute Deviation (MAD): " << mad << " points\n";
    std::cout << "Standard Deviation (SD): " << sd << " points\n";
    std::cout << "-> MAD ~ 0.8 x SD (for normal distribution)\n";

    // Compare effect of outliers
    print_subsection("Robustness Against Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    print_data("Data with outlier", with_outlier);

    double mad_outlier = statcpp::mean_absolute_deviation(with_outlier.begin(), with_outlier.end());
    double sd_outlier = statcpp::stddev(with_outlier.begin(), with_outlier.end());

    std::cout << "\nWithout outlier:\n";
    std::cout << "  MAD: " << mad << ", SD: " << sd << "\n";
    std::cout << "With outlier:\n";
    std::cout << "  MAD: " << mad_outlier << ", SD: " << sd_outlier << "\n";
    std::cout << "\n-> Due to outlier:\n";
    std::cout << "  SD increase: " << (sd_outlier / sd - 1) * 100 << "%\n";
    std::cout << "  MAD increase: " << (mad_outlier / mad - 1) * 100 << "%\n";
    std::cout << "  -> MAD is less affected by outliers\n";
}

// ============================================================================
// 7. Example Using Lambda Expressions (Projection)
// ============================================================================

/**
 * @brief Advanced usage example with lambda expressions (projection)
 *
 * [Purpose]
 * All variance functions have overloads that accept lambda expressions
 * (projection functions). This allows calculating spread on struct members
 * or transformed values.
 */
void example_projection() {
    print_section("7. Example Using Lambda Expressions (Projection)");

    // Struct example
    struct Product {
        std::string name;
        double price;
        int quantity;
    };

    std::vector<Product> products = {
        {"Product A", 1000, 50},
        {"Product B", 1500, 30},
        {"Product C", 800, 70},
        {"Product D", 2000, 20},
        {"Product E", 1200, 40}
    };

    std::cout << "Product data:\n";
    for (const auto& p : products) {
        std::cout << "  " << p.name << ": Price=" << p.price
                  << " yen, Quantity=" << p.quantity << " units\n";
    }

    // Price statistics
    double price_mean = statcpp::mean(products.begin(), products.end(),
                                      [](const Product& p) { return p.price; });
    double price_sd = statcpp::stddev(products.begin(), products.end(),
                                      [](const Product& p) { return p.price; });
    double price_cv = statcpp::coefficient_of_variation(products.begin(), products.end(),
                                                        [](const Product& p) { return p.price; });

    std::cout << "\nPrice statistics:\n";
    std::cout << "  Mean: " << price_mean << " yen\n";
    std::cout << "  Std dev: " << price_sd << " yen\n";
    std::cout << "  CV: " << price_cv * 100 << "%\n";

    // Quantity statistics
    double qty_mean = statcpp::mean(products.begin(), products.end(),
                                    [](const Product& p) { return p.quantity; });
    double qty_sd = statcpp::stddev(products.begin(), products.end(),
                                    [](const Product& p) { return p.quantity; });
    double qty_cv = statcpp::coefficient_of_variation(products.begin(), products.end(),
                                                      [](const Product& p) { return p.quantity; });

    std::cout << "\nQuantity statistics:\n";
    std::cout << "  Mean: " << qty_mean << " units\n";
    std::cout << "  Std dev: " << qty_sd << " units\n";
    std::cout << "  CV: " << qty_cv * 100 << "%\n";

    std::cout << "\n-> Comparison using CV: ";
    if (price_cv > qty_cv) {
        std::cout << "Price has greater relative spread\n";
    } else {
        std::cout << "Quantity has greater relative spread\n";
    }
}

// ============================================================================
// 8. Population vs Sample: Which to Use
// ============================================================================

/**
 * @brief Choosing between population and sample variance
 *
 * [Purpose]
 * Explains the difference and usage of population variance (ddof=0)
 * and sample variance (ddof=1).
 */
void example_population_vs_sample() {
    print_section("8. Population vs Sample: Which to Use");

    std::vector<double> data = {10, 12, 14, 16, 18};
    print_data("Data", data);

    double pop_var = statcpp::population_variance(data.begin(), data.end());
    double samp_var = statcpp::sample_variance(data.begin(), data.end());

    std::cout << "Population variance (divide by N=" << data.size() << "): " << pop_var << "\n";
    std::cout << "Sample variance (divide by N-1=" << data.size() - 1 << "): " << samp_var << "\n";

    std::cout << R"(
[Guidelines for Choosing]

+---------------------+-----------------------------------+
| Situation           | Function to Use                   |
+---------------------+-----------------------------------+
| Data is entire      | population_variance()             |
| population          | population_stddev()               |
| Ex: Analyzing all   | var(first, last, 0)               |
| students in a class |                                   |
+---------------------+-----------------------------------+
| Data is a sample    | sample_variance()                 |
| (drawn from         | sample_stddev()                   |
| population)         | variance()                        |
| Ex: Survey,         | stddev()                          |
| clinical trial      | var(first, last, 1)               |
+---------------------+-----------------------------------+

[Reason]
Dividing by N-1 in sample variance yields an "unbiased estimator."
Using the sample mean reduces degrees of freedom by 1, so dividing
by N-1 gives an estimator whose expected value equals the population
variance (unbiased estimator).
)";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Choosing Measures of Spread");

    std::cout << R"(
+--------------------------+----------------------------------------+
| Function                 | Use Case                               |
+--------------------------+----------------------------------------+
| range()                  | Simplest spread. Sensitive to outliers |
| variance() / stddev()    | General measure of spread              |
| population_*()           | When data is entire population         |
| sample_*()               | When data is a sample (estimation)     |
| coefficient_of_variation()| Comparing data with different scales  |
| iqr()                    | Robust to outliers. For box plots      |
| mean_absolute_deviation()| Somewhat robust to outliers            |
+--------------------------+----------------------------------------+

[Notes]
- iqr() requires pre-sorting
- coefficient_of_variation() cannot be used when mean is 0
- Use sample_variance() / sample_stddev() for estimation
- Consider iqr() or MAD when outliers are present
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_range();
    example_variance();
    example_stddev();
    example_coefficient_of_variation();
    example_iqr();
    example_mean_absolute_deviation();
    example_projection();
    example_population_vs_sample();

    // Display summary
    print_summary();

    return 0;
}
