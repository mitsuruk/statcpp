/**
 * @file example_correlation_covariance.cpp
 * @brief Sample code for statcpp::correlation_covariance.hpp
 *
 * This file explains the usage of covariance and correlation coefficient
 * functions provided in correlation_covariance.hpp through practical examples.
 *
 * [Provided Functions]
 * - population_covariance() : Population covariance
 * - sample_covariance()     : Sample covariance (unbiased covariance)
 * - covariance()            : Covariance (alias for sample_covariance)
 * - pearson_correlation()   : Pearson correlation coefficient
 * - spearman_correlation()  : Spearman rank correlation coefficient
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_correlation_covariance.cpp -o example_correlation_covariance
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// statcpp correlation/covariance header
#include "statcpp/correlation_covariance.hpp"
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

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
// 1. Covariance Concept Explanation
// ============================================================================

/**
 * @brief Covariance concept explanation
 *
 * [Purpose]
 * Covariance is a measure of how much two variables vary together.
 *
 * [Formula]
 * Population covariance: Cov(X,Y) = E[(X-mu_x)(Y-mu_y)] = (1/n)Sum(xi-x_bar)(yi-y_bar)
 * Sample covariance: s_xy = (1/(n-1))Sum(xi-x_bar)(yi-y_bar)
 *
 * [Interpretation]
 * - Covariance > 0: When X is large, Y tends to be large (positive association)
 * - Covariance < 0: When X is large, Y tends to be small (negative association)
 * - Covariance ~ 0: No linear association between X and Y
 *
 * [Notes]
 * - Covariance depends on units (affected by scale)
 * - Cannot judge strength of association by magnitude alone
 */
void example_covariance_concept() {
    print_section("1. Covariance Concept");

    std::cout << R"(
[What is Covariance?]
A measure of the degree to which two variables "vary together."

[Visual Image]

Positive covariance:
Y
|     *  *
|   *  *
|  *
|*
+---------- X
When X increases, Y increases

Negative covariance:
Y
|*
|  *
|    *
|      *  *
+---------- X
When X increases, Y decreases

Covariance ~ 0:
Y
|  *    *
|*  *
|    *
|  *    *
+---------- X
No linear relationship between X and Y

)";
}

// ============================================================================
// 2. covariance() - Covariance Calculation
// ============================================================================

/**
 * @brief Example usage of covariance()
 *
 * [Use Cases]
 * - Examine direction of association (positive/negative) between two variables
 * - Intermediate calculation for variance analysis and regression
 * - Portfolio risk calculation
 */
void example_covariance() {
    print_section("2. covariance() - Covariance");

    // Study hours and test scores
    std::vector<double> study_hours = {2, 4, 6, 8, 10};
    std::vector<double> test_scores = {50, 60, 70, 80, 90};

    print_data("Study hours (h)", study_hours);
    print_data("Test scores", test_scores);

    double pop_cov = statcpp::population_covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double samp_cov = statcpp::sample_covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double cov = statcpp::covariance(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());

    std::cout << "\nPopulation covariance (population_covariance): " << pop_cov << "\n";
    std::cout << "Sample covariance (sample_covariance):         " << samp_cov << "\n";
    std::cout << "covariance():                                  " << cov << " (= sample_covariance)\n";

    std::cout << "\n-> Positive covariance: Longer study hours tend to result in higher scores\n";

    // Example of negative covariance
    print_subsection("Example of Negative Covariance");
    std::vector<double> absences = {0, 2, 4, 6, 8};    // Absence days
    std::vector<double> grades = {90, 80, 70, 60, 50};  // Grades

    print_data("Absence days", absences);
    print_data("Grades", grades);

    double neg_cov = statcpp::covariance(
        absences.begin(), absences.end(),
        grades.begin(), grades.end());
    std::cout << "Covariance: " << neg_cov << "\n";
    std::cout << "-> Negative covariance: More absences tend to result in lower grades\n";
}

// ============================================================================
// 3. pearson_correlation() - Pearson Correlation Coefficient
// ============================================================================

/**
 * @brief Example usage of pearson_correlation()
 *
 * [Purpose]
 * Pearson correlation coefficient expresses the strength and direction
 * of linear association between two variables in the range -1 to +1.
 * It is the standardized covariance, independent of scale.
 *
 * [Formula]
 * r = Cov(X,Y) / (sigma_x * sigma_y) = Sum(xi-x_bar)(yi-y_bar) / sqrt[Sum(xi-x_bar)^2 * Sum(yi-y_bar)^2]
 *
 * [Interpretation]
 * - r = +1 : Perfect positive linear relationship
 * - r = -1 : Perfect negative linear relationship
 * - r = 0  : No linear relationship
 *
 * [Guidelines]
 * |r| < 0.2 : Almost no correlation
 * 0.2 <= |r| < 0.4 : Weak correlation
 * 0.4 <= |r| < 0.6 : Moderate correlation
 * 0.6 <= |r| < 0.8 : Strong correlation
 * |r| >= 0.8 : Very strong correlation
 *
 * [Use Cases]
 * - Quantify strength of association between two variables
 * - Evaluate prediction models
 * - Variable selection
 *
 * [Notes]
 * - Only measures linear relationships (cannot detect nonlinear)
 * - Sensitive to outliers
 * - Correlation is not causation (beware of spurious correlations)
 */
void example_pearson_correlation() {
    print_section("3. pearson_correlation() - Pearson Correlation Coefficient");

    // Strong positive correlation
    print_subsection("Strong Positive Correlation");
    std::vector<double> height = {160, 165, 170, 175, 180};
    std::vector<double> weight = {50, 55, 62, 68, 75};

    print_data("Height (cm)", height);
    print_data("Weight (kg)", weight);

    double r_positive = statcpp::pearson_correlation(
        height.begin(), height.end(),
        weight.begin(), weight.end());
    std::cout << "Pearson correlation coefficient: " << r_positive << "\n";
    std::cout << "-> Strong positive correlation: Taller people tend to weigh more\n";

    // Strong negative correlation
    print_subsection("Strong Negative Correlation");
    std::vector<double> distance = {1, 2, 3, 4, 5};      // Commute distance
    std::vector<double> satisfaction = {90, 75, 60, 45, 30};  // Commute satisfaction

    print_data("Commute distance (km)", distance);
    print_data("Commute satisfaction", satisfaction);

    double r_negative = statcpp::pearson_correlation(
        distance.begin(), distance.end(),
        satisfaction.begin(), satisfaction.end());
    std::cout << "Pearson correlation coefficient: " << r_negative << "\n";
    std::cout << "-> Strong negative correlation: Longer distance means lower satisfaction\n";

    // Almost no correlation
    print_subsection("Almost No Correlation");
    std::vector<double> shoe_size = {24, 26, 25, 27, 24};
    std::vector<double> iq = {100, 95, 110, 100, 105};

    print_data("Shoe size", shoe_size);
    print_data("IQ", iq);

    double r_none = statcpp::pearson_correlation(
        shoe_size.begin(), shoe_size.end(),
        iq.begin(), iq.end());
    std::cout << "Pearson correlation coefficient: " << r_none << "\n";
    std::cout << "-> Almost no correlation: No relationship between shoe size and IQ\n";
}

// ============================================================================
// 4. spearman_correlation() - Spearman Rank Correlation Coefficient
// ============================================================================

/**
 * @brief Example usage of spearman_correlation()
 *
 * [Purpose]
 * Spearman rank correlation coefficient is the Pearson correlation
 * between the ranks of two variables. It can detect monotonic
 * relationships, not just linear ones.
 *
 * [Formula]
 * rho = Pearson(rank(X), rank(Y))
 *
 * [Characteristics]
 * - Less sensitive to outliers
 * - Applicable to ordinal scale data
 * - Detects monotonic relationships (always increasing/decreasing)
 *
 * [Use Cases]
 * - Correlation of ranked data
 * - When outliers are present
 * - Detecting nonlinear but monotonic relationships
 * - Ordinal scale data (satisfaction 1-5, etc.)
 */
void example_spearman_correlation() {
    print_section("4. spearman_correlation() - Spearman Rank Correlation Coefficient");

    // Near-linear relationship (Pearson and Spearman are close)
    print_subsection("Near-Linear Relationship");
    std::vector<double> study_hours = {2, 4, 6, 8, 10};
    std::vector<double> test_scores = {50, 60, 70, 80, 90};

    print_data("Study hours", study_hours);
    print_data("Test scores", test_scores);

    double r_pearson = statcpp::pearson_correlation(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());
    double r_spearman = statcpp::spearman_correlation(
        study_hours.begin(), study_hours.end(),
        test_scores.begin(), test_scores.end());

    std::cout << "Pearson correlation:  " << r_pearson << "\n";
    std::cout << "Spearman correlation: " << r_spearman << "\n";
    std::cout << "-> For linear relationships, both are nearly identical\n";

    // When outliers are present
    print_subsection("When Outliers Are Present");
    std::vector<double> x_outlier = {1, 2, 3, 4, 5, 100};  // 100 is an outlier
    std::vector<double> y_outlier = {10, 20, 30, 40, 50, 55};

    print_data("X (with outlier)", x_outlier);
    print_data("Y", y_outlier);

    double r_pearson_outlier = statcpp::pearson_correlation(
        x_outlier.begin(), x_outlier.end(),
        y_outlier.begin(), y_outlier.end());
    double r_spearman_outlier = statcpp::spearman_correlation(
        x_outlier.begin(), x_outlier.end(),
        y_outlier.begin(), y_outlier.end());

    std::cout << "Pearson correlation:  " << r_pearson_outlier << "\n";
    std::cout << "Spearman correlation: " << r_spearman_outlier << "\n";
    std::cout << "-> Spearman is less affected by outliers\n";

    // Monotonic but nonlinear relationship
    print_subsection("Monotonic but Nonlinear Relationship");
    std::vector<double> x_exp = {1, 2, 3, 4, 5};
    std::vector<double> y_exp = {2, 4, 8, 16, 32};  // Exponential increase

    print_data("X", x_exp);
    print_data("Y (exponential increase)", y_exp);

    double r_pearson_exp = statcpp::pearson_correlation(
        x_exp.begin(), x_exp.end(),
        y_exp.begin(), y_exp.end());
    double r_spearman_exp = statcpp::spearman_correlation(
        x_exp.begin(), x_exp.end(),
        y_exp.begin(), y_exp.end());

    std::cout << "Pearson correlation:  " << r_pearson_exp << "\n";
    std::cout << "Spearman correlation: " << r_spearman_exp << "\n";
    std::cout << "-> Spearman correctly detects the monotonic relationship\n";
}

// ============================================================================
// 5. Pearson vs Spearman - Choosing Between Them
// ============================================================================

/**
 * @brief Choosing between Pearson and Spearman
 */
void example_correlation_comparison() {
    print_section("5. Pearson vs Spearman - Choosing Between Them");

    std::cout << R"(
[Guidelines for Choosing]

+----------------------+-------------------------------------------+
| Situation            | Recommended Correlation                   |
+----------------------+-------------------------------------------+
| Examining linear     | pearson_correlation()                     |
| relationship         |                                           |
| Data near normal     |                                           |
| distribution         |                                           |
| No outliers          |                                           |
+----------------------+-------------------------------------------+
| Examining monotonic  | spearman_correlation()                    |
| relationship         |                                           |
| Outliers present     |                                           |
| Ordinal scale data   |                                           |
| Skewed distribution  |                                           |
+----------------------+-------------------------------------------+

[When Calculating Both]
- Both are close -> Linear relationship
- Spearman > Pearson -> Nonlinear but monotonic relationship
- Large difference -> Effect of outliers or complex relationship
)";
}

// ============================================================================
// 6. Example Using Lambda Expressions (Projection)
// ============================================================================

/**
 * @brief Advanced usage example with lambda expressions (projection)
 *
 * Calculates correlation between struct members.
 */
void example_projection() {
    print_section("6. Example Using Lambda Expressions (Projection)");

    struct Student {
        std::string name;
        double math_score;
        double english_score;
        double study_hours;
    };

    std::vector<Student> students = {
        {"Alice", 85, 90, 5},
        {"Bob", 70, 75, 3},
        {"Charlie", 95, 85, 7},
        {"Diana", 60, 70, 2},
        {"Eve", 80, 80, 4}
    };

    std::cout << "Student data:\n";
    for (const auto& s : students) {
        std::cout << "  " << s.name << ": Math=" << s.math_score
                  << ", English=" << s.english_score
                  << ", Study hours=" << s.study_hours << "h\n";
    }

    // Correlation between math and English
    auto math_proj = [](const Student& s) { return s.math_score; };
    auto eng_proj = [](const Student& s) { return s.english_score; };
    auto hours_proj = [](const Student& s) { return s.study_hours; };

    double r_math_eng = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        math_proj, eng_proj);

    double r_math_hours = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        math_proj, hours_proj);

    double r_eng_hours = statcpp::pearson_correlation(
        students.begin(), students.end(),
        students.begin(), students.end(),
        eng_proj, hours_proj);

    std::cout << "\nCorrelation coefficients:\n";
    std::cout << "  Math - English:     " << r_math_eng << "\n";
    std::cout << "  Math - Study hours: " << r_math_hours << "\n";
    std::cout << "  English - Study hours: " << r_eng_hours << "\n";

    // Correlation matrix (conceptual display)
    std::cout << "\nCorrelation matrix:\n";
    std::cout << "           Math    English  Study hrs\n";
    std::cout << "  Math     1.0000  " << std::setw(7) << r_math_eng
              << "  " << std::setw(7) << r_math_hours << "\n";
    std::cout << "  English  " << std::setw(7) << r_math_eng << "  1.0000  "
              << std::setw(7) << r_eng_hours << "\n";
    std::cout << "  Study hrs " << std::setw(7) << r_math_hours
              << "  " << std::setw(7) << r_eng_hours << "  1.0000\n";
}

// ============================================================================
// 7. Cautions When Interpreting Correlation Coefficients
// ============================================================================

/**
 * @brief Cautions when interpreting correlation coefficients
 */
void example_correlation_caveats() {
    print_section("7. Cautions When Interpreting Correlation Coefficients");

    std::cout << R"(
[Caution 1: Correlation is not Causation]

Example: There is a positive correlation between ice cream sales and drowning deaths
-> Ice cream does not cause drowning
-> Both are influenced by a third variable: "temperature" (spurious correlation)

[Caution 2: Missing Nonlinear Relationships]

Example: U-shaped relationship
  Y
  |
  |*        *
  |  *    *
  |    **
  +---------- X

Pearson correlation ~ 0, but there is clearly a relationship

[Caution 3: Effect of Outliers]

A single outlier can significantly change the correlation coefficient
-> It is important to draw a scatter plot and check

[Caution 4: Sample Size]

Small samples can show high correlations by chance
-> Conduct statistical tests to confirm significance

[Caution 5: Restricted Range]

When the range of a variable is restricted, correlation is underestimated
Example: When only tall people are studied, correlation between height and weight appears weak
)";

    // Practical example: Nonlinear relationship
    print_subsection("Practical Example: U-shaped Relationship");
    std::vector<double> age = {20, 30, 40, 50, 60, 70};
    std::vector<double> happiness = {80, 70, 60, 65, 75, 85};  // U-shaped

    print_data("Age", age);
    print_data("Happiness", happiness);

    double r = statcpp::pearson_correlation(
        age.begin(), age.end(),
        happiness.begin(), happiness.end());
    std::cout << "Pearson correlation coefficient: " << r << "\n";
    std::cout << "-> Correlation is small, but a U-shaped relationship exists\n";
    std::cout << "  (Happiness is lowest in middle age, high in youth and old age)\n";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Correlation and Covariance Functions");

    std::cout << R"(
+-------------------------+------------------------------------------+
| Function                | Description                              |
+-------------------------+------------------------------------------+
| covariance()            | Sample covariance (unbiased)             |
| population_covariance() | Population covariance                    |
| sample_covariance()     | = covariance()                           |
| pearson_correlation()   | Pearson correlation (linear)             |
| spearman_correlation()  | Spearman rank correlation (monotonic)    |
+-------------------------+------------------------------------------+

[Interpretation of Correlation Coefficient]
+-------------+-------------------------------------------------------+
| |r| Range   | Interpretation                                        |
+-------------+-------------------------------------------------------+
| 0.0 - 0.2   | Almost no correlation                                 |
| 0.2 - 0.4   | Weak correlation                                      |
| 0.4 - 0.6   | Moderate correlation                                  |
| 0.6 - 0.8   | Strong correlation                                    |
| 0.8 - 1.0   | Very strong correlation                               |
+-------------+-------------------------------------------------------+

[Important Notes]
- Correlation does not imply causation
- Pearson measures linear relationship, Spearman measures monotonic
- Use Spearman when outliers are present
- It is important to visually confirm with a scatter plot
- Statistical significance tests should also be performed
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_covariance_concept();
    example_covariance();
    example_pearson_correlation();
    example_spearman_correlation();
    example_correlation_comparison();
    example_projection();
    example_correlation_caveats();

    // Display summary
    print_summary();

    return 0;
}
