/**
 * @file example_shape_of_distribution.cpp
 * @brief Sample code for statcpp::shape_of_distribution.hpp
 *
 * This file explains how to use the functions for measuring distribution
 * shape provided in shape_of_distribution.hpp through practical examples.
 *
 * [Provided Functions]
 * - population_skewness() : Population skewness
 * - sample_skewness()     : Sample skewness (bias-corrected)
 * - skewness()            : Skewness (alias for sample_skewness)
 * - population_kurtosis() : Population kurtosis (excess kurtosis)
 * - sample_kurtosis()     : Sample kurtosis (bias-corrected)
 * - kurtosis()            : Kurtosis (alias for sample_kurtosis)
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_shape_of_distribution.cpp -o example_shape_of_distribution
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>

// statcpp distribution shape header
#include "statcpp/shape_of_distribution.hpp"
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
// 1. skewness() - Skewness (Concept Explanation)
// ============================================================================

/**
 * @brief Conceptual explanation of skewness()
 *
 * [Purpose]
 * Skewness is a measure of the asymmetry of a distribution.
 * For symmetric distributions like the normal distribution, skewness is 0.
 *
 * [Formula]
 * Population skewness: gamma_1 = E[(X - mu)^3] / sigma^3
 * Sample skewness: Bias-corrected estimator
 *
 * [Interpretation]
 * - skewness = 0 : Symmetric distribution
 * - skewness > 0 : Right-skewed distribution (positive skew, longer right tail)
 * - skewness < 0 : Left-skewed distribution (negative skew, longer left tail)
 *
 * [Guidelines]
 * |skewness| < 0.5  : Approximately symmetric
 * 0.5 <= |skewness| < 1.0 : Moderate skewness
 * |skewness| >= 1.0 : High skewness
 */
void example_skewness_concept() {
    print_section("1. Skewness Concept");

    std::cout << R"(
[What is Skewness]
A measure of distribution asymmetry. Quantifies whether data is
symmetric around the mean.

[Visual Image]

Positive skewness (right-skewed):
    |
  ##|#
 ######______
 <- Mean is to the right of median ->
 Income distribution, house prices, etc.

Negative skewness (left-skewed):
           |
         ##|##
______######
 <- Mean is to the left of median ->
 Test scores (when near perfect), etc.

Symmetric (skewness ~ 0):
       |
     ##|##
   ########
 Mean ~ Median
 Normal distribution, etc.
)";
}

// ============================================================================
// 2. Skewness Calculation with Actual Data
// ============================================================================

/**
 * @brief Skewness calculation examples
 *
 * [Use Cases]
 * - Checking distribution shape
 * - Simple normality check
 * - Determining need for data transformation (e.g., log transform)
 *
 * [Notes]
 * - sample_skewness() requires at least 3 data points
 * - Sensitive to outliers
 */
void example_skewness_calculation() {
    print_section("2. Skewness Calculation Examples");

    // Nearly symmetric data
    std::vector<double> symmetric = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Positively skewed data (right-skewed)
    std::vector<double> right_skewed = {1, 2, 2, 3, 3, 3, 4, 4, 5, 10};

    // Negatively skewed data (left-skewed)
    std::vector<double> left_skewed = {1, 6, 7, 7, 7, 8, 8, 9, 9, 10};

    print_subsection("Nearly symmetric data");
    print_data("Data", symmetric);
    double skew_sym = statcpp::skewness(symmetric.begin(), symmetric.end());
    std::cout << "Skewness: " << skew_sym << " (~0: symmetric)\n";
    std::cout << "Mean: " << statcpp::mean(symmetric.begin(), symmetric.end()) << "\n";

    print_subsection("Positively skewed data (longer right tail)");
    print_data("Data", right_skewed);
    double skew_right = statcpp::skewness(right_skewed.begin(), right_skewed.end());
    std::cout << "Skewness: " << skew_right << " (>0: right-skewed)\n";
    std::cout << "Mean: " << statcpp::mean(right_skewed.begin(), right_skewed.end()) << "\n";
    std::cout << "-> Outlier 10 pulls the mean to the right\n";

    print_subsection("Negatively skewed data (longer left tail)");
    print_data("Data", left_skewed);
    double skew_left = statcpp::skewness(left_skewed.begin(), left_skewed.end());
    std::cout << "Skewness: " << skew_left << " (<0: left-skewed)\n";
    std::cout << "Mean: " << statcpp::mean(left_skewed.begin(), left_skewed.end()) << "\n";
    std::cout << "-> Outlier 1 pulls the mean to the left\n";
}

// ============================================================================
// 3. population_skewness() vs sample_skewness()
// ============================================================================

/**
 * @brief Difference between population and sample skewness
 *
 * [Purpose]
 * Explains the difference between population_skewness() and sample_skewness().
 * Use sample_skewness() when estimating population parameters from a sample.
 */
void example_population_vs_sample_skewness() {
    print_section("3. Population Skewness vs Sample Skewness");

    std::vector<double> data = {2, 3, 5, 7, 8, 9, 10, 12, 15, 20};
    print_data("Data (n=10)", data);

    double pop_skew = statcpp::population_skewness(data.begin(), data.end());
    double samp_skew = statcpp::sample_skewness(data.begin(), data.end());

    std::cout << "\nPopulation skewness (population_skewness): " << pop_skew << "\n";
    std::cout << "Sample skewness (sample_skewness):          " << samp_skew << "\n";
    std::cout << "skewness():                                 " << statcpp::skewness(data.begin(), data.end())
              << " (= sample_skewness)\n";

    std::cout << R"(
[When to Use Which]
- population_skewness(): When data is the entire population
  Example: Distribution of tenure for all company employees

- sample_skewness(): When data is a sample (extracted from population)
  Example: Survey responses
  Bias correction is applied for more accurate estimation
)";
}

// ============================================================================
// 4. kurtosis() - Kurtosis (Concept Explanation)
// ============================================================================

/**
 * @brief Conceptual explanation of kurtosis()
 *
 * [Purpose]
 * Kurtosis measures the "tail heaviness" or "peakedness" of a distribution.
 * It indicates how heavy the tails are compared to a normal distribution
 * (i.e., how likely outliers are).
 *
 * [Formula]
 * Population kurtosis (excess): gamma_2 = E[(X - mu)^4] / sigma^4 - 3
 * Subtracting 3 makes normal distribution kurtosis equal to 0 (excess kurtosis)
 *
 * [Interpretation]
 * - kurtosis = 0  : Same as normal distribution (mesokurtic)
 * - kurtosis > 0  : Heavier tails than normal (leptokurtic)
 *                   More likely to have outliers, more peaked distribution
 * - kurtosis < 0  : Lighter tails than normal (platykurtic)
 *                   Less likely to have outliers, flatter distribution
 */
void example_kurtosis_concept() {
    print_section("4. Kurtosis Concept");

    std::cout << R"(
[What is Kurtosis]
A measure of tail heaviness (likelihood of outliers).
Compared to normal distribution as baseline (0).

[Excess Kurtosis]
This library uses excess kurtosis (adjusted so normal distribution = 0)

[Visual Image]

Positive kurtosis (leptokurtic):
      ^
     ###
     ###   <- Peaked center, heavy tails
   #######
__########__#__
   Outliers more likely
   Example: t-distribution, financial returns

Negative kurtosis (platykurtic):
   _________
  ##########
 ############  <- Flat center, light tails
################
   Outliers less likely
   Example: Uniform distribution

Normal distribution (mesokurtic, kurtosis ~ 0):
     ##
   ######
 ##########
################
)";
}

// ============================================================================
// 5. Kurtosis Calculation with Actual Data
// ============================================================================

/**
 * @brief Kurtosis calculation examples
 *
 * [Use Cases]
 * - Evaluating tail heaviness
 * - Risk assessment (frequency of outliers)
 * - Simple normality check
 *
 * [Notes]
 * - sample_kurtosis() requires at least 4 data points
 * - Highly sensitive to outliers (uses 4th power)
 */
void example_kurtosis_calculation() {
    print_section("5. Kurtosis Calculation Examples");

    // Uniform-like data (negative kurtosis)
    std::vector<double> uniform_like = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Normal-like data
    std::vector<double> normal_like = {2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8};

    // Data with outliers (positive kurtosis)
    std::vector<double> heavy_tails = {1, 4, 5, 5, 5, 5, 5, 5, 6, 10};

    print_subsection("Uniform-like data");
    print_data("Data", uniform_like);
    double kurt_uniform = statcpp::kurtosis(uniform_like.begin(), uniform_like.end());
    std::cout << "Kurtosis: " << kurt_uniform << " (<0: light tails)\n";

    print_subsection("Normal-like data");
    print_data("Data", normal_like);
    double kurt_normal = statcpp::kurtosis(normal_like.begin(), normal_like.end());
    std::cout << "Kurtosis: " << kurt_normal << " (~0: same as normal distribution)\n";

    print_subsection("Data with outliers");
    print_data("Data", heavy_tails);
    double kurt_heavy = statcpp::kurtosis(heavy_tails.begin(), heavy_tails.end());
    std::cout << "Kurtosis: " << kurt_heavy << " (>0: heavy tails)\n";
    std::cout << "-> Values 1 and 10 at both ends increase kurtosis\n";
}

// ============================================================================
// 6. population_kurtosis() vs sample_kurtosis()
// ============================================================================

/**
 * @brief Difference between population and sample kurtosis
 */
void example_population_vs_sample_kurtosis() {
    print_section("6. Population Kurtosis vs Sample Kurtosis");

    std::vector<double> data = {2, 3, 5, 7, 8, 9, 10, 12, 15, 20};
    print_data("Data (n=10)", data);

    double pop_kurt = statcpp::population_kurtosis(data.begin(), data.end());
    double samp_kurt = statcpp::sample_kurtosis(data.begin(), data.end());

    std::cout << "\nPopulation kurtosis (population_kurtosis): " << pop_kurt << "\n";
    std::cout << "Sample kurtosis (sample_kurtosis):          " << samp_kurt << "\n";
    std::cout << "kurtosis():                                 " << statcpp::kurtosis(data.begin(), data.end())
              << " (= sample_kurtosis)\n";

    std::cout << R"(
[When to Use Which]
- population_kurtosis(): When data is the entire population

- sample_kurtosis(): When data is a sample (extracted from population)
  Bias correction is applied
)";
}

// ============================================================================
// 7. Practical Example: Income Distribution Analysis
// ============================================================================

/**
 * @brief Practical usage example: Income distribution
 *
 * Income distributions typically show right-skewed distribution (positive skewness).
 */
void example_income_distribution() {
    print_section("7. Practical Example: Income Distribution Analysis");

    // Hypothetical income data (in 10k units)
    std::vector<double> income = {
        300, 320, 350, 380, 400, 420, 450, 480, 500,
        520, 550, 600, 650, 700, 800, 1000, 1500, 2000, 5000
    };

    std::cout << "Income data (10k units): ";
    for (size_t i = 0; i < income.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << income[i];
    }
    std::cout << "\n\n";

    double mean_val = statcpp::mean(income.begin(), income.end());
    std::vector<double> sorted_income = income;
    std::sort(sorted_income.begin(), sorted_income.end());
    double median_val = statcpp::median(sorted_income.begin(), sorted_income.end());
    double skew = statcpp::skewness(income.begin(), income.end());
    double kurt = statcpp::kurtosis(income.begin(), income.end());

    std::cout << "Mean: " << mean_val << " (10k)\n";
    std::cout << "Median: " << median_val << " (10k)\n";
    std::cout << "Skewness: " << skew << "\n";
    std::cout << "Kurtosis: " << kurt << "\n";

    std::cout << "\n[Analysis Results]\n";
    std::cout << "- Mean > Median -> Mean pulled up by high earners\n";
    std::cout << "- Skewness > 0 -> Right-skewed distribution (few high earners)\n";
    std::cout << "- Kurtosis > 0 -> Heavy tails (extreme high earners exist)\n";
    std::cout << "\n-> The median is more appropriate as a 'representative value' for income\n";
}

// ============================================================================
// 8. Simple Normality Check
// ============================================================================

/**
 * @brief Simple normality check
 *
 * Skewness and kurtosis can be used for a simple check of whether
 * data follows a normal distribution.
 */
void example_normality_check() {
    print_section("8. Simple Normality Check");

    // Generate normal-like data using random number generator
    std::mt19937 gen(42);  // Fix seed for reproducibility
    std::normal_distribution<> normal_dist(100, 15);

    std::vector<double> normal_data;
    normal_data.reserve(100);
    for (int i = 0; i < 100; ++i) {
        normal_data.push_back(normal_dist(gen));
    }

    double skew = statcpp::skewness(normal_data.begin(), normal_data.end());
    double kurt = statcpp::kurtosis(normal_data.begin(), normal_data.end());

    std::cout << "Data generated from normal distribution (n=100, mu=100, sigma=15)\n\n";
    std::cout << "Skewness: " << skew << "\n";
    std::cout << "Kurtosis: " << kurt << "\n";

    std::cout << R"(
[Normality Guidelines]
Skewness: |skewness| < 2 is acceptable
Kurtosis: |kurtosis| < 7 is acceptable
(Based on Kline, 2015 criteria)

For more rigorous checking, use Shapiro-Wilk test, etc.
)";

    // Assessment
    std::cout << "\n[Assessment]\n";
    bool skew_ok = std::abs(skew) < 2.0;
    bool kurt_ok = std::abs(kurt) < 7.0;
    std::cout << "Skewness criterion: " << (skew_ok ? "OK" : "NG") << "\n";
    std::cout << "Kurtosis criterion: " << (kurt_ok ? "OK" : "NG") << "\n";
    std::cout << "Overall assessment: " << ((skew_ok && kurt_ok) ? "Approximately normal" : "Deviates from normal") << "\n";
}

// ============================================================================
// 9. Usage Example with Lambda (Projection)
// ============================================================================

/**
 * @brief Advanced usage example with lambda (projection)
 */
void example_projection() {
    print_section("9. Usage Example with Lambda (Projection)");

    // Struct example
    struct ExamResult {
        std::string name;
        double score;
    };

    std::vector<ExamResult> results = {
        {"Alice", 85}, {"Bob", 78}, {"Charlie", 92}, {"Diana", 65},
        {"Eve", 88}, {"Frank", 72}, {"Grace", 95}, {"Henry", 80},
        {"Ivy", 58}, {"Jack", 90}, {"Kate", 82}, {"Leo", 75}
    };

    std::cout << "Exam results:\n";
    for (const auto& r : results) {
        std::cout << "  " << r.name << ": " << r.score << " points\n";
    }

    auto score_proj = [](const ExamResult& r) { return r.score; };

    double mean_score = statcpp::mean(results.begin(), results.end(), score_proj);
    double skew = statcpp::skewness(results.begin(), results.end(), score_proj);
    double kurt = statcpp::kurtosis(results.begin(), results.end(), score_proj);

    std::cout << "\nScore statistics:\n";
    std::cout << "  Mean: " << mean_score << " points\n";
    std::cout << "  Skewness: " << skew << "\n";
    std::cout << "  Kurtosis: " << kurt << "\n";

    // Interpretation
    std::cout << "\n[Distribution Interpretation]\n";
    if (skew < -0.5) {
        std::cout << "- Negative skewness: Distribution skewed toward high scores (exam was easy)\n";
    } else if (skew > 0.5) {
        std::cout << "- Positive skewness: Distribution skewed toward low scores (exam was difficult)\n";
    } else {
        std::cout << "- Skewness ~ 0: Approximately symmetric distribution (appropriate difficulty)\n";
    }
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Distribution Shape Measures");

    std::cout << R"(
+-------------------------+------------------------------------------+
| Function                | Description                              |
+-------------------------+------------------------------------------+
| skewness()              | Sample skewness (bias-corrected)         |
| population_skewness()   | Population skewness                      |
| sample_skewness()       | = skewness()                             |
| kurtosis()              | Sample kurtosis (excess, bias-corrected) |
| population_kurtosis()   | Population kurtosis (excess)             |
| sample_kurtosis()       | = kurtosis()                             |
+-------------------------+------------------------------------------+

[Skewness Interpretation]
+------------------+------------------------------------------------+
| Value            | Meaning                                        |
+------------------+------------------------------------------------+
| ~ 0              | Symmetric distribution                         |
| > 0              | Right-skewed (longer right tail) e.g., income  |
| < 0              | Left-skewed (longer left tail) e.g., easy exam |
+------------------+------------------------------------------------+

[Kurtosis Interpretation] (Excess Kurtosis)
+------------------+------------------------------------------------+
| Value            | Meaning                                        |
+------------------+------------------------------------------------+
| ~ 0              | Same as normal distribution                    |
| > 0              | Heavy tails (outliers more likely)             |
| < 0              | Light tails (outliers less likely)             |
+------------------+------------------------------------------------+

[Notes]
- sample_skewness() requires n>=3, sample_kurtosis() requires n>=4
- Sensitive to outliers (especially kurtosis uses 4th power)
- Use statistical tests for rigorous normality checking
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_skewness_concept();
    example_skewness_calculation();
    example_population_vs_sample_skewness();
    example_kurtosis_concept();
    example_kurtosis_calculation();
    example_population_vs_sample_kurtosis();
    example_income_distribution();
    example_normality_check();
    example_projection();

    // Display summary
    print_summary();

    return 0;
}
