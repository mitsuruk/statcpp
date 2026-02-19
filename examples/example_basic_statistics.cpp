/**
 * @file example_basic_statistics.cpp
 * @brief Sample code for statcpp::basic_statistics.hpp
 *
 * This file explains the usage of basic statistical functions provided
 * in basic_statistics.hpp through practical examples.
 *
 * [Provided Functions]
 * - sum()            : Sum
 * - count()          : Data count
 * - mean()           : Arithmetic mean
 * - median()         : Median (*requires sorted data)
 * - mode()           : Mode (single)
 * - modes()          : Mode (multiple)
 * - geometric_mean() : Geometric mean
 * - harmonic_mean()  : Harmonic mean
 * - trimmed_mean()   : Trimmed mean (*requires sorted data)
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_basic_statistics.cpp -o example_basic_statistics
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

// statcpp basic statistics header
#include "statcpp/basic_statistics.hpp"

// ============================================================================
// Helper Functions for Display
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// Sample Data Display
// ============================================================================

/**
 * @brief Display sample data
 */
void example_sample_data() {
    print_section("Sample Data");

    // Basic numeric data (assuming test scores)
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::cout << "Test score data: ";
    for (double s : scores) std::cout << s << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. sum() - Sum
// ============================================================================

/**
 * @brief Example usage of sum()
 *
 * [Purpose]
 * sum() calculates the total sum of the data.
 * It returns the result of adding all values together.
 *
 * [Use Cases]
 * - Calculate total sales
 * - Find total score
 * - Calculate cumulative values
 */
void example_sum() {
    print_section("1. sum() - Sum");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    double total = statcpp::sum(scores.begin(), scores.end());
    std::cout << "Total score: " << total << "\n";

    // Also works with integer data (return type depends on element type)
    std::vector<int> integers = {1, 2, 3, 4, 5};
    int int_sum = statcpp::sum(integers.begin(), integers.end());
    std::cout << "Integer sum (1+2+3+4+5): " << int_sum << "\n";
}

// ============================================================================
// 2. count() - Data Count
// ============================================================================

/**
 * @brief Example usage of count()
 *
 * [Purpose]
 * count() returns the number of data points (sample size).
 *
 * [Use Cases]
 * - Check sample size
 * - Use as denominator for mean calculation
 * - Data validation
 */
void example_count() {
    print_section("2. count() - Data Count");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    std::size_t n = statcpp::count(scores.begin(), scores.end());
    std::cout << "Data count: " << n << " people\n";
}

// ============================================================================
// 3. mean() - Arithmetic Mean
// ============================================================================

/**
 * @brief Example usage of mean()
 *
 * [Purpose]
 * mean() calculates the arithmetic mean of the data.
 * It is the sum of all data divided by the count.
 *
 * [Formula]
 * mean = (x1 + x2 + ... + xn) / n
 *
 * [Use Cases]
 * - Calculate class average score
 * - Calculate average temperature, average income, etc.
 * - Most common measure of "center"
 *
 * [Notes]
 * - Sensitive to outliers
 * - Throws exception for empty data
 */
void example_mean() {
    print_section("3. mean() - Arithmetic Mean");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};
    double avg = statcpp::mean(scores.begin(), scores.end());
    std::cout << "Average score: " << avg << " points\n";

    // Check effect of outliers
    print_subsection("Effect of Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200}; // 200 is an outlier
    double avg_with_outlier = statcpp::mean(with_outlier.begin(), with_outlier.end());
    std::cout << "Mean without outlier: " << avg << "\n";
    std::cout << "Mean with outlier: " << avg_with_outlier << " (increased due to 200)\n";
}

// ============================================================================
// 4. median() - Median
// ============================================================================

/**
 * @brief Example usage of median()
 *
 * [Purpose]
 * median() returns the middle value when data is sorted in ascending order.
 * - For odd count: the middle value
 * - For even count: average of the two middle values
 *
 * [Important]
 * * Input data must be pre-sorted!
 *
 * [Use Cases]
 * - When you want a center measure not affected by outliers
 * - Median income (more representative than mean)
 * - Median real estate price
 *
 * [Characteristics]
 * - Robust against outliers
 * - Applicable to ordinal scale data
 */
void example_median() {
    print_section("4. median() - Median");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // Sort before calculating median
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    std::cout << "Sorted data: ";
    for (double s : sorted_scores) std::cout << s << " ";
    std::cout << "\n";

    double med = statcpp::median(sorted_scores.begin(), sorted_scores.end());
    std::cout << "Median: " << med << " points\n";

    // Median is less affected by outliers
    print_subsection("Robustness Against Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_with_outlier = with_outlier;
    std::sort(sorted_with_outlier.begin(), sorted_with_outlier.end());
    double med_with_outlier = statcpp::median(sorted_with_outlier.begin(), sorted_with_outlier.end());
    std::cout << "Median without outlier: " << med << "\n";
    std::cout << "Median with outlier: " << med_with_outlier << " (limited effect from outlier)\n";

    // Even number of data points
    print_subsection("Even Number of Data Points");
    std::vector<double> even_data = {10, 20, 30, 40};
    double med_even = statcpp::median(even_data.begin(), even_data.end());
    std::cout << "Data: 10, 20, 30, 40\n";
    std::cout << "Median: " << med_even << " (average of 20 and 30)\n";
}

// ============================================================================
// 5. mode() - Mode
// ============================================================================

/**
 * @brief Example usage of mode()
 *
 * [Purpose]
 * mode() returns the most frequently occurring value in the data.
 *
 * [Use Cases]
 * - Most popular product size
 * - Most common response
 * - Representative value for categorical data
 *
 * [Notes]
 * - When there are multiple modes, returns the smallest value (deterministic behavior)
 * - Use modes() if you need all modes
 */
void example_mode() {
    print_section("5. mode() - Mode");

    // Data where 88 and 90 each appear twice
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::cout << "Original data: ";
    for (double s : scores) std::cout << s << " ";
    std::cout << "\n";

    double mode_val = statcpp::mode(scores.begin(), scores.end());
    std::cout << "Mode: " << mode_val << " (returns smallest when multiple modes exist)\n";

    // Example with categorical data
    print_subsection("Use with Categorical Data");
    std::vector<std::string> sizes = {"M", "L", "M", "S", "M", "L", "M", "XL"};
    std::string most_popular = statcpp::mode(sizes.begin(), sizes.end());
    std::cout << "Size data: M, L, M, S, M, L, M, XL\n";
    std::cout << "Most common size: " << most_popular << "\n";
}

// ============================================================================
// 6. modes() - Multiple Modes
// ============================================================================

/**
 * @brief Example usage of modes()
 *
 * [Purpose]
 * modes() returns all modes (sorted in ascending order as a vector).
 *
 * [Use Cases]
 * - Detecting bimodal distribution
 * - When you want to know all tied-for-first products
 * - Analysis of multiple-choice surveys
 */
void example_modes() {
    print_section("6. modes() - Multiple Modes");

    // 88 and 90 occur with same frequency (2 times each)
    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    std::vector<double> all_modes = statcpp::modes(scores.begin(), scores.end());
    std::cout << "All modes: ";
    for (double m : all_modes) std::cout << m << " ";
    std::cout << "\n";
    std::cout << "(88 and 90 each appear 2 times, tied for first)\n";
}

// ============================================================================
// 7. geometric_mean() - Geometric Mean
// ============================================================================

/**
 * @brief Example usage of geometric_mean()
 *
 * [Purpose]
 * geometric_mean() calculates the geometric mean of the data.
 * It is the n-th root of the product of n values.
 *
 * [Formula]
 * geometric_mean = (x1 * x2 * ... * xn)^(1/n)
 * = exp((log(x1) + log(x2) + ... + log(xn)) / n)
 *
 * [Use Cases]
 * - Average of growth rates (compound annual growth rate, etc.)
 * - Average of ratios and percentages
 * - Center of log-normally distributed data
 *
 * [Notes]
 * - All values must be positive (zero or negative not allowed)
 * - Always less than or equal to arithmetic mean (AM-GM inequality)
 */
void example_geometric_mean() {
    print_section("7. geometric_mean() - Geometric Mean");

    // Annual growth rate example (1.05 = 5% growth, 0.95 = 5% decline)
    std::vector<double> growth_rates = {1.05, 1.10, 0.95, 1.08, 1.03};
    std::cout << "Annual growth rates: 1.05, 1.10, 0.95, 1.08, 1.03\n";

    double arith_mean = statcpp::mean(growth_rates.begin(), growth_rates.end());
    double geom_mean = statcpp::geometric_mean(growth_rates.begin(), growth_rates.end());

    std::cout << "Arithmetic mean: " << arith_mean << "\n";
    std::cout << "Geometric mean: " << geom_mean << "\n";
    std::cout << "-> Geometric mean is appropriate for averaging growth rates\n";
    std::cout << "  Cumulative after 5 years: " << std::pow(geom_mean, 5)
              << " (multiplier from initial 1.0)\n";
}

// ============================================================================
// 8. harmonic_mean() - Harmonic Mean
// ============================================================================

/**
 * @brief Example usage of harmonic_mean()
 *
 * [Purpose]
 * harmonic_mean() calculates the harmonic mean of the data.
 * It is the reciprocal of the arithmetic mean of reciprocals.
 *
 * [Formula]
 * harmonic_mean = n / (1/x1 + 1/x2 + ... + 1/xn)
 *
 * [Use Cases]
 * - Average speed (average speed for round trip, etc.)
 * - Average of ratios
 * - Average of P/E ratios (price-earnings ratio)
 *
 * [Notes]
 * - Cannot be used with data containing zero
 * - Always smaller than arithmetic and geometric means
 */
void example_harmonic_mean() {
    print_section("8. harmonic_mean() - Harmonic Mean");

    // Average speed for round trip example
    // Going: 60 km/h, Returning: 40 km/h, traveling same distance
    std::vector<double> speeds = {60.0, 40.0};

    double arith_speed = statcpp::mean(speeds.begin(), speeds.end());
    double harm_speed = statcpp::harmonic_mean(speeds.begin(), speeds.end());

    std::cout << "Going speed: 60 km/h\n";
    std::cout << "Returning speed: 40 km/h\n";
    std::cout << "Arithmetic mean: " << arith_speed << " km/h (incorrect)\n";
    std::cout << "Harmonic mean: " << harm_speed << " km/h (correct average speed)\n";
    std::cout << "\n[Explanation]\n";
    std::cout << "Let distance be d:\n";
    std::cout << "  Going: d/60 hours, Returning: d/40 hours\n";
    std::cout << "  Total distance: 2d, Total time: d/60 + d/40 = d(2+3)/120 = 5d/120\n";
    std::cout << "  Average speed: 2d / (5d/120) = 240/5 = 48 km/h\n";

    // Verify relationship among three means
    print_subsection("Relationship Among Three Means (AM >= GM >= HM)");
    std::vector<double> positive_data = {2.0, 8.0};
    double am = statcpp::mean(positive_data.begin(), positive_data.end());
    double gm = statcpp::geometric_mean(positive_data.begin(), positive_data.end());
    double hm = statcpp::harmonic_mean(positive_data.begin(), positive_data.end());
    std::cout << "Data: 2, 8\n";
    std::cout << "Arithmetic mean (AM): " << am << "\n";
    std::cout << "Geometric mean (GM): " << gm << "\n";
    std::cout << "Harmonic mean (HM): " << hm << "\n";
    std::cout << "Always AM >= GM >= HM holds\n";
}

// ============================================================================
// 9. trimmed_mean() - Trimmed Mean
// ============================================================================

/**
 * @brief Example usage of trimmed_mean()
 *
 * [Purpose]
 * trimmed_mean() calculates the mean after removing a certain proportion
 * from both ends of the data. It reduces the effect of outliers while
 * using more information than the median.
 *
 * [Important]
 * * Input data must be pre-sorted!
 *
 * [Arguments]
 * proportion: Proportion to exclude from one side (0.0 to less than 0.5)
 *   - 0.1 -> Exclude bottom 10% and top 10%
 *   - 0.25 -> Exclude bottom 25% and top 25% (interquartile mean)
 *
 * [Use Cases]
 * - When you want to reduce the effect of outliers
 * - Olympic scoring (exclude highest and lowest scores)
 * - Reliable estimate of center
 */
void example_trimmed_mean() {
    print_section("9. trimmed_mean() - Trimmed Mean");

    std::vector<double> scores = {85, 90, 78, 92, 88, 75, 95, 82, 88, 90};

    // Use sorted data
    std::vector<double> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    std::cout << "Sorted data: ";
    for (double s : sorted_scores) std::cout << s << " ";
    std::cout << "\n\n";

    double mean_normal = statcpp::mean(sorted_scores.begin(), sorted_scores.end());
    double trimmed_10 = statcpp::trimmed_mean(sorted_scores.begin(), sorted_scores.end(), 0.1);
    double trimmed_20 = statcpp::trimmed_mean(sorted_scores.begin(), sorted_scores.end(), 0.2);

    std::cout << "Regular mean:                     " << mean_normal << "\n";
    std::cout << "10% trimmed mean (exclude 1 each end): " << trimmed_10 << "\n";
    std::cout << "20% trimmed mean (exclude 2 each end): " << trimmed_20 << "\n";

    // Effect of trimmed mean when outliers exist
    print_subsection("Effect of Trimmed Mean with Outliers");
    std::vector<double> with_outlier = {85, 90, 78, 92, 88, 75, 95, 82, 88, 200};
    std::vector<double> sorted_with_outlier = with_outlier;
    std::sort(sorted_with_outlier.begin(), sorted_with_outlier.end());

    std::cout << "Data with outlier (sorted): ";
    for (double s : sorted_with_outlier) std::cout << s << " ";
    std::cout << "\n";

    double mean_outlier = statcpp::mean(sorted_with_outlier.begin(), sorted_with_outlier.end());
    double trimmed_outlier = statcpp::trimmed_mean(sorted_with_outlier.begin(), sorted_with_outlier.end(), 0.1);

    std::cout << "Regular mean: " << mean_outlier << " (large effect from outlier 200)\n";
    std::cout << "10% trimmed mean: " << trimmed_outlier << " (outlier excluded)\n";
}

// ============================================================================
// 10. Advanced Usage with Lambda Expressions (Projection)
// ============================================================================

/**
 * @brief Advanced usage example with lambda expressions (projection)
 *
 * [Purpose]
 * Many functions have overloads that accept lambda expressions (projection functions).
 * This allows calculating statistics on struct members or transformed values.
 *
 * [Use Cases]
 * - Statistics on specific fields from a vector of structs
 * - Statistics after data transformation (log transformation, etc.)
 * - Processing complex data structures
 */
void example_projection() {
    print_section("10. Advanced Usage with Lambda Expressions (Projection)");

    // Struct example
    struct Student {
        std::string name;
        int math_score;
        int english_score;
    };

    std::vector<Student> students = {
        {"Alice", 85, 90},
        {"Bob", 78, 82},
        {"Charlie", 92, 88},
        {"Diana", 88, 95},
        {"Eve", 75, 80}
    };

    // Average of math scores
    double math_avg = statcpp::mean(students.begin(), students.end(),
                                    [](const Student& s) { return s.math_score; });

    // Average of English scores
    double english_avg = statcpp::mean(students.begin(), students.end(),
                                       [](const Student& s) { return s.english_score; });

    // Average of total scores
    double total_avg = statcpp::mean(students.begin(), students.end(),
                                     [](const Student& s) { return s.math_score + s.english_score; });

    std::cout << "Student data:\n";
    for (const auto& s : students) {
        std::cout << "  " << s.name << ": Math=" << s.math_score
                  << ", English=" << s.english_score << "\n";
    }
    std::cout << "\n";
    std::cout << "Math average: " << math_avg << " points\n";
    std::cout << "English average: " << english_avg << " points\n";
    std::cout << "Total average: " << total_avg << " points\n";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Which Mean to Use?");

    std::cout << R"(
+-----------------+----------------------------------------+
| Function        | Use Case                               |
+-----------------+----------------------------------------+
| mean()          | General average. When no outliers      |
| median()        | When outliers exist. Income, prices    |
| mode()          | Categorical data. Most popular choice  |
| geometric_mean()| Average of growth rates, ratios        |
| harmonic_mean() | Average speeds, P/E ratios             |
| trimmed_mean()  | Partial exclusion of outliers          |
+-----------------+----------------------------------------+

[Notes]
- median() and trimmed_mean() require pre-sorting
- geometric_mean() only for positive values
- harmonic_mean() cannot include zero
- Exceptions thrown for empty data
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_sample_data();
    example_sum();
    example_count();
    example_mean();
    example_median();
    example_mode();
    example_modes();
    example_geometric_mean();
    example_harmonic_mean();
    example_trimmed_mean();
    example_projection();

    // Display summary
    print_summary();

    return 0;
}
