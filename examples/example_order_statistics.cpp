/**
 * @file example_order_statistics.cpp
 * @brief Sample code for statcpp::order_statistics.hpp
 *
 * This file explains how to use the order statistics functions
 * provided in order_statistics.hpp through practical examples.
 *
 * [Provided Functions]
 * - minimum()             : Minimum value
 * - maximum()             : Maximum value
 * - quartiles()           : Quartiles (Q1, Q2, Q3) *Requires sorted data
 * - percentile()          : Percentile *Requires sorted data
 * - five_number_summary() : Five-number summary *Requires sorted data
 * - interpolate_at()      : Percentile calculation by linear interpolation (internal function)
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_order_statistics.cpp -o example_order_statistics
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

// statcpp order statistics header
#include "statcpp/order_statistics.hpp"
#include "statcpp/basic_statistics.hpp"

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
// 1. minimum() / maximum() - Minimum and Maximum Values
// ============================================================================

/**
 * @brief Usage example of minimum() / maximum()
 *
 * [Purpose]
 * minimum() returns the minimum value, maximum() returns the maximum value.
 * Sorting is not required.
 *
 * [Use Cases]
 * - Checking the range of data
 * - Detecting anomalies
 * - Basic data validation
 */
void example_min_max() {
    print_section("1. minimum() / maximum() - Minimum and Maximum Values");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    print_data("Test scores", scores);

    double min_val = statcpp::minimum(scores.begin(), scores.end());
    double max_val = statcpp::maximum(scores.begin(), scores.end());

    std::cout << "Minimum value: " << min_val << " points\n";
    std::cout << "Maximum value: " << max_val << " points\n";
    std::cout << "Range: " << (max_val - min_val) << " points\n";

    // Works with integer data as well
    print_subsection("Integer data case");
    std::vector<int> integers = {3, 1, 4, 1, 5, 9, 2, 6};
    print_data("Integer data", integers);
    std::cout << "Minimum value: " << statcpp::minimum(integers.begin(), integers.end()) << "\n";
    std::cout << "Maximum value: " << statcpp::maximum(integers.begin(), integers.end()) << "\n";
}

// ============================================================================
// 2. quartiles() - Quartiles
// ============================================================================

/**
 * @brief Usage example of quartiles()
 *
 * [Purpose]
 * quartiles() returns three values (Q1, Q2, Q3) that divide the data into four parts.
 *
 * [Formulas]
 * Q1 (First quartile): 25th percentile of data
 * Q2 (Second quartile): 50th percentile of data (= median)
 * Q3 (Third quartile): 75th percentile of data
 *
 * [Important]
 * * Input data must be sorted in advance!
 *
 * [Use Cases]
 * - Understanding data distribution
 * - Creating box plots
 * - Outlier detection (IQR rule)
 *
 * [Interpolation Method]
 * Uses linear interpolation equivalent to R type=7 / Excel QUARTILE.INC
 */
void example_quartiles() {
    print_section("2. quartiles() - Quartiles");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // Sorting is required
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("Sorted data", sorted_scores);

    statcpp::quartile_result q = statcpp::quartiles(sorted_scores.begin(), sorted_scores.end());

    std::cout << "\nQuartiles:\n";
    std::cout << "  Q1 (25th percentile): " << q.q1 << " points\n";
    std::cout << "  Q2 (50th percentile): " << q.q2 << " points (= median)\n";
    std::cout << "  Q3 (75th percentile): " << q.q3 << " points\n";
    std::cout << "  IQR (Interquartile Range): " << (q.q3 - q.q1) << " points\n";

    // Interpretation
    std::cout << "\n[Interpretation]\n";
    std::cout << "- Bottom 25% of students scored " << q.q1 << " points or less\n";
    std::cout << "- Middle 50% of students scored between " << q.q1 << " and " << q.q3 << " points\n";
    std::cout << "- Top 25% of students scored " << q.q3 << " points or higher\n";
}

// ============================================================================
// 3. percentile() - Percentile
// ============================================================================

/**
 * @brief Usage example of percentile()
 *
 * [Purpose]
 * percentile() returns the data value corresponding to a specified proportion.
 * Percentiles are specified as proportions from 0.0 to 1.0, not 0 to 100.
 *
 * [Important]
 * * Input data must be sorted in advance!
 *
 * [Use Cases]
 * - Calculating deviation values
 * - Ranking performance
 * - Calculating p99 of response times
 */
void example_percentile() {
    print_section("3. percentile() - Percentile");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("Sorted data", sorted_scores);

    std::cout << "\nKey percentiles:\n";
    std::cout << "  0th percentile (minimum): "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.00) << " points\n";
    std::cout << "  10th percentile: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.10) << " points\n";
    std::cout << "  25th percentile (Q1): "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.25) << " points\n";
    std::cout << "  50th percentile (median): "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.50) << " points\n";
    std::cout << "  75th percentile (Q3): "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.75) << " points\n";
    std::cout << "  90th percentile: "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 0.90) << " points\n";
    std::cout << "  100th percentile (maximum): "
              << statcpp::percentile(sorted_scores.begin(), sorted_scores.end(), 1.00) << " points\n";

    // Practical example: Response time analysis
    print_subsection("Practical example: API response time analysis");
    std::vector<double> response_times = {
        45, 52, 48, 51, 49, 47, 55, 120, 46, 50,
        48, 53, 47, 49, 150, 51, 48, 52, 46, 200
    };
    std::sort(response_times.begin(), response_times.end());

    std::cout << "Response times (ms): ";
    for (double t : response_times) std::cout << t << " ";
    std::cout << "\n\n";

    std::cout << "p50 (median): "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.50) << " ms\n";
    std::cout << "p95: "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.95) << " ms\n";
    std::cout << "p99: "
              << statcpp::percentile(response_times.begin(), response_times.end(), 0.99) << " ms\n";
    std::cout << "\n-> p50 is good, but p99 is high (due to outliers)\n";
}

// ============================================================================
// 4. five_number_summary() - Five-Number Summary
// ============================================================================

/**
 * @brief Usage example of five_number_summary()
 *
 * [Purpose]
 * five_number_summary() returns five values that summarize the data:
 * - Minimum (min)
 * - First quartile (Q1)
 * - Median
 * - Third quartile (Q3)
 * - Maximum (max)
 *
 * [Important]
 * * Input data must be sorted in advance!
 *
 * [Use Cases]
 * - Quick overview of data
 * - Creating box plots
 * - Data summary for reports
 */
void example_five_number_summary() {
    print_section("4. five_number_summary() - Five-Number Summary");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    print_data("Sorted data", sorted_scores);

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(sorted_scores.begin(), sorted_scores.end());

    std::cout << "\nFive-number summary:\n";
    std::cout << "  Minimum: " << summary.min << " points\n";
    std::cout << "  Q1:      " << summary.q1 << " points\n";
    std::cout << "  Median:  " << summary.median << " points\n";
    std::cout << "  Q3:      " << summary.q3 << " points\n";
    std::cout << "  Maximum: " << summary.max << " points\n";

    // Text-based box plot (conceptual diagram)
    print_subsection("Box plot (conceptual diagram)");
    std::cout << R"(
    Min     Q1    Median   Q3    Max
      |     +-----+-----+     |
      +-----+     |     +-----+
      |     +-----+-----+     |
     )" << summary.min << "   " << summary.q1 << "   "
              << summary.median << "   " << summary.q3 << "   " << summary.max << "\n";
}

// ============================================================================
// 5. Outlier Detection (IQR Rule)
// ============================================================================

/**
 * @brief Application to outlier detection
 *
 * [IQR Rule]
 * Outlier definition:
 * - Mild outlier: Less than Q1 - 1.5*IQR, or greater than Q3 + 1.5*IQR
 * - Extreme outlier: Less than Q1 - 3.0*IQR, or greater than Q3 + 3.0*IQR
 */
void example_outlier_detection() {
    print_section("5. Outlier Detection (IQR Rule)");

    std::vector<double> data = {2, 3, 4, 5, 6, 7, 8, 9, 10, 50};  // 50 is an outlier

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    print_data("Data", sorted_data);

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(sorted_data.begin(), sorted_data.end());

    double iqr = summary.q3 - summary.q1;
    double lower_fence = summary.q1 - 1.5 * iqr;
    double upper_fence = summary.q3 + 1.5 * iqr;
    double lower_extreme = summary.q1 - 3.0 * iqr;
    double upper_extreme = summary.q3 + 3.0 * iqr;

    std::cout << "\nFive-number summary:\n";
    std::cout << "  Minimum: " << summary.min << "\n";
    std::cout << "  Q1: " << summary.q1 << "\n";
    std::cout << "  Median: " << summary.median << "\n";
    std::cout << "  Q3: " << summary.q3 << "\n";
    std::cout << "  Maximum: " << summary.max << "\n";

    std::cout << "\nIQR: " << iqr << "\n";
    std::cout << "\nOutlier criteria:\n";
    std::cout << "  Mild outlier: < " << lower_fence << " or > " << upper_fence << "\n";
    std::cout << "  Extreme outlier: < " << lower_extreme << " or > " << upper_extreme << "\n";

    std::cout << "\nDetected outliers:\n";
    for (double val : sorted_data) {
        if (val < lower_extreme || val > upper_extreme) {
            std::cout << "  " << val << " -> Extreme outlier\n";
        } else if (val < lower_fence || val > upper_fence) {
            std::cout << "  " << val << " -> Mild outlier\n";
        }
    }
}

// ============================================================================
// 6. Usage Example with Lambda (Projection)
// ============================================================================

/**
 * @brief Advanced usage example with lambda (projection)
 *
 * Calculates order statistics on struct members.
 */
void example_projection() {
    print_section("6. Usage Example with Lambda (Projection)");

    struct Employee {
        std::string name;
        int salary;
        int years;
    };

    std::vector<Employee> employees = {
        {"Alice", 450, 3},
        {"Bob", 520, 5},
        {"Charlie", 380, 2},
        {"Diana", 680, 8},
        {"Eve", 420, 4},
        {"Frank", 550, 6},
        {"Grace", 490, 5},
        {"Henry", 350, 1}
    };

    std::cout << "Employee data:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": salary=" << e.salary
                  << " (10k yen), tenure=" << e.years << " years\n";
    }

    // Minimum and maximum salary
    auto salary_proj = [](const Employee& e) { return e.salary; };

    int min_salary = statcpp::minimum(employees.begin(), employees.end(), salary_proj);
    int max_salary = statcpp::maximum(employees.begin(), employees.end(), salary_proj);

    std::cout << "\nSalary range:\n";
    std::cout << "  Minimum: " << min_salary << " (10k yen)\n";
    std::cout << "  Maximum: " << max_salary << " (10k yen)\n";

    // Sort by salary then calculate quartiles
    std::sort(employees.begin(), employees.end(),
              [](const Employee& a, const Employee& b) { return a.salary < b.salary; });

    std::cout << "\nSorted by salary:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": " << e.salary << " (10k yen)\n";
    }

    statcpp::five_number_summary_result summary =
        statcpp::five_number_summary(employees.begin(), employees.end(), salary_proj);

    std::cout << "\nFive-number summary of salaries:\n";
    std::cout << "  Minimum: " << summary.min << " (10k yen)\n";
    std::cout << "  Q1: " << summary.q1 << " (10k yen)\n";
    std::cout << "  Median: " << summary.median << " (10k yen)\n";
    std::cout << "  Q3: " << summary.q3 << " (10k yen)\n";
    std::cout << "  Maximum: " << summary.max << " (10k yen)\n";
}

// ============================================================================
// 7. Interpolation Method Explanation
// ============================================================================

/**
 * @brief Explanation of interpolation method
 *
 * This library uses linear interpolation equivalent to
 * R type=7 / Excel QUARTILE.INC.
 */
void example_interpolation() {
    print_section("7. Interpolation Method Explanation");

    std::cout << R"(
[Percentile Interpolation Method]

This library uses the same linear interpolation method as
R's quantile() function type=7 and Excel's QUARTILE.INC / PERCENTILE.INC.

[Calculation Method]
1. Index = p * (n - 1)  (p: percentile proportion, n: data count)
2. Split index into integer part lo and fractional part frac
3. Result = data[lo] * (1 - frac) + data[lo + 1] * frac

[Example: Calculate 25th percentile (Q1) for n=10 data]
Index = 0.25 * (10 - 1) = 2.25
lo = 2, frac = 0.25
Result = data[2] * 0.75 + data[3] * 0.25

)";

    // Concrete example
    std::vector<double> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    print_data("Data (n=10)", data);

    std::cout << "\nQ1 calculation:\n";
    std::cout << "  Index = 0.25 * 9 = 2.25\n";
    std::cout << "  lo = 2, frac = 0.25\n";
    std::cout << "  Result = data[2] * 0.75 + data[3] * 0.25\n";
    std::cout << "         = 30 * 0.75 + 40 * 0.25 = 22.5 + 10 = 32.5\n";
    std::cout << "  Actual calculation result: "
              << statcpp::percentile(data.begin(), data.end(), 0.25) << "\n";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Order Statistics Functions");

    std::cout << R"(
+------------------------+-------------------------------------------+
| Function               | Description                               |
+------------------------+-------------------------------------------+
| minimum()              | Minimum value (no sorting required)       |
| maximum()              | Maximum value (no sorting required)       |
| quartiles()            | Quartiles Q1, Q2, Q3 (requires sorting)   |
| percentile(first,last,p)| Arbitrary percentile (requires sorting)  |
| five_number_summary()  | Five-number summary (requires sorting)    |
+------------------------+-------------------------------------------+

[Return Value Structs]
- quartile_result: q1, q2, q3
- five_number_summary_result: min, q1, median, q3, max

[Important Notes]
- quartiles(), percentile(), five_number_summary() require
  pre-sorted data
- percentile() argument p is specified as proportion from 0.0 to 1.0
  (Example: 90th percentile -> p = 0.90)
- Interpolation method is equivalent to R type=7 / Excel QUARTILE.INC

[Outlier Detection (IQR Rule)]
- IQR = Q3 - Q1
- Mild outlier: Less than Q1 - 1.5*IQR, or greater than Q3 + 1.5*IQR
- Extreme outlier: Less than Q1 - 3.0*IQR, or greater than Q3 + 3.0*IQR
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_min_max();
    example_quartiles();
    example_percentile();
    example_five_number_summary();
    example_outlier_detection();
    example_projection();
    example_interpolation();

    // Display summary
    print_summary();

    return 0;
}
