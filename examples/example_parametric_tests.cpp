/**
 * @file example_parametric_tests.cpp
 * @brief Sample code for statcpp::parametric_tests.hpp
 *
 * This file explains how to use the parametric statistical tests
 * provided in parametric_tests.hpp through practical examples.
 *
 * [Provided Functions]
 * - z_test()                     : One-sample z-test (known variance)
 * - z_test_proportion()          : One-sample proportion z-test
 * - z_test_proportion_two_sample(): Two-sample proportion z-test
 * - t_test()                     : One-sample t-test
 * - t_test_two_sample()          : Two-sample t-test (pooled variance)
 * - t_test_welch()               : Welch's t-test (unequal variances)
 * - t_test_paired()              : Paired t-test
 * - chisq_test_gof()             : Chi-square goodness-of-fit test
 * - chisq_test_independence()    : Chi-square independence test
 * - f_test()                     : F-test (variance comparison)
 * - bonferroni_correction()      : Bonferroni correction
 * - benjamini_hochberg_correction(): BH correction (FDR control)
 * - holm_correction()            : Holm correction
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_parametric_tests.cpp -o example_parametric_tests
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

// statcpp parametric tests header
#include "statcpp/parametric_tests.hpp"
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

// Display test results
void print_test_result(const statcpp::test_result& result, const std::string& stat_name = "Statistic") {
    std::cout << stat_name << ": " << result.statistic << "\n";
    std::cout << "Degrees of freedom: " << result.df << "\n";
    std::cout << "p-value: " << result.p_value << "\n";

    std::string alt_str;
    switch (result.alternative) {
        case statcpp::alternative_hypothesis::less:
            alt_str = "One-sided (less)";
            break;
        case statcpp::alternative_hypothesis::greater:
            alt_str = "One-sided (greater)";
            break;
        case statcpp::alternative_hypothesis::two_sided:
        default:
            alt_str = "Two-sided";
            break;
    }
    std::cout << "Alternative hypothesis: " << alt_str << "\n";

    std::cout << "Decision (alpha=0.05): ";
    if (result.p_value < 0.05) {
        std::cout << "Significant (reject null hypothesis)\n";
    } else {
        std::cout << "Not significant (fail to reject null hypothesis)\n";
    }
}

// ============================================================================
// 1. t_test() - One-Sample t-Test
// ============================================================================

/**
 * @brief Usage example of t_test()
 *
 * [Purpose]
 * One-sample t-test tests whether the sample mean is
 * statistically significantly different from a specific value (null hypothesis value).
 *
 * [Null hypothesis] H0: mu = mu0
 * [Alternative hypothesis] H1: mu != mu0 (two-sided), mu < mu0 (one-sided less), mu > mu0 (one-sided greater)
 *
 * [Use Cases]
 * - Whether a new treatment effect differs from a baseline
 * - Whether product quality meets specifications
 * - Whether experimental results match theoretical values
 *
 * [Notes]
 * - Assumes data follows a normal distribution
 * - This assumption is important for small sample sizes (n < 30)
 */
void example_t_test() {
    print_section("1. t_test() - One-Sample t-Test");

    // Example: Does the average height of students at a high school differ from the national average of 170cm?
    std::vector<double> heights = {168, 172, 175, 165, 170, 178, 169, 173, 167, 174};

    print_data("Height data (cm)", heights);
    double mu0 = 170.0;  // Null hypothesis: mean = 170cm
    std::cout << "Null hypothesis mean: " << mu0 << " cm\n\n";

    std::cout << "Sample statistics:\n";
    std::cout << "  Sample size: " << statcpp::count(heights.begin(), heights.end()) << "\n";
    std::cout << "  Sample mean: " << statcpp::mean(heights.begin(), heights.end()) << " cm\n";
    std::cout << "  Sample standard deviation: " << statcpp::sample_stddev(heights.begin(), heights.end()) << " cm\n\n";

    // Two-sided test
    print_subsection("Two-sided test: H1: mu != 170");
    auto result_two = statcpp::t_test(heights.begin(), heights.end(), mu0,
                                      statcpp::alternative_hypothesis::two_sided);
    print_test_result(result_two, "t-statistic");

    // One-sided test (greater)
    print_subsection("One-sided test: H1: mu > 170");
    auto result_greater = statcpp::t_test(heights.begin(), heights.end(), mu0,
                                          statcpp::alternative_hypothesis::greater);
    print_test_result(result_greater, "t-statistic");
}

// ============================================================================
// 2. t_test_two_sample() and t_test_welch() - Two-Sample t-Test
// ============================================================================

/**
 * @brief Usage example of two-sample t-test
 *
 * [Purpose]
 * Tests whether the means of two independent samples are statistically significantly different.
 *
 * [Two Methods]
 * - t_test_two_sample(): Uses pooled variance (assumes equal variances)
 * - t_test_welch(): Welch method (does not assume equal variances, more robust)
 *
 * [Use Cases]
 * - A/B test effect comparison
 * - Treatment vs. control group comparison
 * - Comparing scores between two groups
 */
void example_two_sample_t_test() {
    print_section("2. Two-Sample t-Test");

    // Example: Comparing test scores between a group using new materials and a group using traditional materials
    std::vector<double> group_new = {85, 90, 78, 92, 88, 95, 82, 91, 87, 89};
    std::vector<double> group_old = {75, 82, 70, 85, 78, 80, 72, 83, 76, 79};

    std::cout << "New materials group (n=" << group_new.size() << "):\n";
    print_data("  Data", group_new);
    std::cout << "  Mean: " << statcpp::mean(group_new.begin(), group_new.end()) << " points\n\n";

    std::cout << "Old materials group (n=" << group_old.size() << "):\n";
    print_data("  Data", group_old);
    std::cout << "  Mean: " << statcpp::mean(group_old.begin(), group_old.end()) << " points\n\n";

    // Pooled variance version
    print_subsection("t-test assuming equal variances (t_test_two_sample)");
    auto result_pooled = statcpp::t_test_two_sample(
        group_new.begin(), group_new.end(),
        group_old.begin(), group_old.end());
    print_test_result(result_pooled, "t-statistic");

    // Welch version
    print_subsection("Welch's t-test (t_test_welch) - does not assume equal variances");
    auto result_welch = statcpp::t_test_welch(
        group_new.begin(), group_new.end(),
        group_old.begin(), group_old.end());
    print_test_result(result_welch, "t-statistic");

    std::cout << "\n[Which one to use]\n";
    std::cout << "- When unsure about equal variances: Welch method is recommended\n";
    std::cout << "- Welch method is more conservative and controls Type I error better\n";
}

// ============================================================================
// 3. t_test_paired() - Paired t-Test
// ============================================================================

/**
 * @brief Usage example of t_test_paired()
 *
 * [Purpose]
 * Tests whether the difference in paired data (e.g., before/after comparison of same subjects)
 * is significantly different from 0.
 *
 * [Use Cases]
 * - Before/after treatment comparison
 * - Comparing two conditions on the same person
 * - Twin studies
 */
void example_paired_t_test() {
    print_section("3. t_test_paired() - Paired t-Test");

    // Example: Weight comparison before and after a diet program
    std::vector<double> weight_before = {70, 75, 68, 82, 78, 65, 72, 80, 74, 77};
    std::vector<double> weight_after =  {68, 72, 67, 78, 75, 64, 70, 76, 72, 74};

    std::cout << "Evaluating diet program effectiveness\n\n";

    std::cout << "Subject data:\n";
    std::cout << std::setw(10) << "Subject" << std::setw(12) << "Before(kg)"
              << std::setw(12) << "After(kg)" << std::setw(12) << "Diff(kg)" << "\n";
    std::cout << std::string(46, '-') << "\n";

    double sum_diff = 0;
    for (std::size_t i = 0; i < weight_before.size(); ++i) {
        double diff = weight_before[i] - weight_after[i];
        sum_diff += diff;
        std::cout << std::setw(10) << (i + 1)
                  << std::setw(12) << weight_before[i]
                  << std::setw(12) << weight_after[i]
                  << std::setw(12) << diff << "\n";
    }

    double mean_diff = sum_diff / weight_before.size();
    std::cout << std::string(46, '-') << "\n";
    std::cout << "Average change: " << mean_diff << " kg\n\n";

    print_subsection("Paired t-test: H1: difference != 0");
    auto result = statcpp::t_test_paired(
        weight_before.begin(), weight_before.end(),
        weight_after.begin(), weight_after.end());
    print_test_result(result, "t-statistic");

    std::cout << "\n-> Diet program effect is statistically "
              << (result.p_value < 0.05 ? "significant" : "not significant") << "\n";
}

// ============================================================================
// 4. z_test_proportion() - Proportion Test
// ============================================================================

/**
 * @brief Usage example of proportion z-test
 *
 * [Purpose]
 * Tests whether the sample proportion is significantly different from a specific value.
 *
 * [Use Cases]
 * - Whether defect rate exceeds a standard
 * - Whether election approval rating exceeds 50%
 * - Validating pass rates
 */
void example_proportion_test() {
    print_section("4. z_test_proportion() - Proportion z-Test");

    // Example: Does a party's approval rating exceed 50%?
    print_subsection("One-sample proportion test");
    std::size_t supporters = 54;   // Number of supporters
    std::size_t sample_size = 100; // Survey sample size
    double p0 = 0.5;               // Null hypothesis: approval rate = 50%

    std::cout << "Survey results:\n";
    std::cout << "  Survey sample size: " << sample_size << " people\n";
    std::cout << "  Number of supporters: " << supporters << " people\n";
    std::cout << "  Approval rate: " << (100.0 * supporters / sample_size) << "%\n";
    std::cout << "  Null hypothesis: approval rate = " << (p0 * 100) << "%\n\n";

    auto result_prop = statcpp::z_test_proportion(supporters, sample_size, p0,
                                                  statcpp::alternative_hypothesis::greater);
    print_test_result(result_prop, "z-statistic");

    // Two-sample proportion test
    print_subsection("Two-sample proportion test");
    std::size_t success_A = 60, n_A = 100;  // Ad A
    std::size_t success_B = 45, n_B = 100;  // Ad B

    std::cout << "A/B test results:\n";
    std::cout << "  Ad A: " << success_A << "/" << n_A << " = " << (100.0 * success_A / n_A) << "%\n";
    std::cout << "  Ad B: " << success_B << "/" << n_B << " = " << (100.0 * success_B / n_B) << "%\n\n";

    auto result_two_prop = statcpp::z_test_proportion_two_sample(
        success_A, n_A, success_B, n_B);
    print_test_result(result_two_prop, "z-statistic");
}

// ============================================================================
// 5. chisq_test_gof() - Chi-Square Goodness-of-Fit Test
// ============================================================================

/**
 * @brief Usage example of chi-square goodness-of-fit test
 *
 * [Purpose]
 * Tests whether the observed frequency distribution matches the expected distribution.
 *
 * [Use Cases]
 * - Whether a die is fair
 * - Whether observed data follows a theoretical distribution
 * - Validating Mendel's laws of genetics
 */
void example_chisq_gof() {
    print_section("5. chisq_test_gof() - Chi-Square Goodness-of-Fit Test");

    // Example: Testing die fairness
    std::vector<double> observed = {18, 20, 16, 22, 14, 30};  // Observed frequencies
    std::vector<double> expected = {20, 20, 20, 20, 20, 20};  // Expected frequencies (if fair)

    std::cout << "Results of rolling a die 120 times:\n";
    std::cout << std::setw(8) << "Face" << std::setw(12) << "Observed"
              << std::setw(12) << "Expected" << "\n";
    std::cout << std::string(32, '-') << "\n";
    for (int i = 0; i < 6; ++i) {
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << observed[i]
                  << std::setw(12) << expected[i] << "\n";
    }
    std::cout << "\n";

    auto result = statcpp::chisq_test_gof(
        observed.begin(), observed.end(),
        expected.begin(), expected.end());
    print_test_result(result, "Chi-square statistic");

    std::cout << "\n-> The die is " << (result.p_value < 0.05 ? "possibly unfair" : "fair") << "\n";
}

// ============================================================================
// 6. chisq_test_independence() - Chi-Square Independence Test
// ============================================================================

/**
 * @brief Usage example of chi-square independence test
 *
 * [Purpose]
 * Tests whether the row and column variables in a contingency table are independent.
 *
 * [Use Cases]
 * - Relationship between gender and product preference
 * - Relationship between treatment and recovery status
 * - Relationship between education and income
 */
void example_chisq_independence() {
    print_section("6. chisq_test_independence() - Chi-Square Independence Test");

    // Example: Relationship between gender and product preference
    std::vector<std::vector<double>> contingency_table = {
        {30, 20, 10},  // Male: A, B, C
        {20, 25, 35}   // Female: A, B, C
    };

    std::cout << "Contingency table of gender and product preference:\n";
    std::cout << std::setw(10) << "" << std::setw(12) << "Product A"
              << std::setw(12) << "Product B" << std::setw(12) << "Product C"
              << std::setw(12) << "Total" << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<std::string> rows = {"Male", "Female"};
    for (std::size_t i = 0; i < contingency_table.size(); ++i) {
        double row_total = 0;
        std::cout << std::setw(10) << rows[i];
        for (double val : contingency_table[i]) {
            std::cout << std::setw(12) << val;
            row_total += val;
        }
        std::cout << std::setw(12) << row_total << "\n";
    }

    std::cout << std::string(58, '-') << "\n";
    std::cout << std::setw(10) << "Total";
    double grand_total = 0;
    for (std::size_t j = 0; j < contingency_table[0].size(); ++j) {
        double col_total = 0;
        for (std::size_t i = 0; i < contingency_table.size(); ++i) {
            col_total += contingency_table[i][j];
        }
        std::cout << std::setw(12) << col_total;
        grand_total += col_total;
    }
    std::cout << std::setw(12) << grand_total << "\n\n";

    auto result = statcpp::chisq_test_independence(contingency_table);
    print_test_result(result, "Chi-square statistic");

    std::cout << "\n-> Gender and product preference are "
              << (result.p_value < 0.05 ? "related" : "independent") << "\n";
}

// ============================================================================
// 7. f_test() - F-Test (Variance Comparison)
// ============================================================================

/**
 * @brief Usage example of F-test
 *
 * [Purpose]
 * Tests whether the variances of two samples are equal.
 * Testing for equal variances is used to verify assumptions before applying t-tests.
 *
 * [Notes]
 * - Strongly assumes normality
 * - More robust methods include Levene's test and Bartlett's test
 */
void example_f_test() {
    print_section("7. f_test() - F-Test (Variance Comparison)");

    // Example: Comparing quality variation between two production lines
    std::vector<double> line_A = {10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1, 10.4};
    std::vector<double> line_B = {10.0, 10.5, 9.5, 10.3, 9.7, 10.2, 9.8, 10.6};

    std::cout << "Production line quality data:\n";
    print_data("Line A", line_A);
    std::cout << "  Variance: " << statcpp::sample_variance(line_A.begin(), line_A.end()) << "\n";
    print_data("Line B", line_B);
    std::cout << "  Variance: " << statcpp::sample_variance(line_B.begin(), line_B.end()) << "\n\n";

    auto result = statcpp::f_test(
        line_A.begin(), line_A.end(),
        line_B.begin(), line_B.end());
    print_test_result(result, "F-statistic");

    std::cout << "\n-> The variation between the two lines is "
              << (result.p_value < 0.05 ? "different" : "similar") << "\n";
}

// ============================================================================
// 8. Multiple Testing Correction
// ============================================================================

/**
 * @brief Usage example of multiple testing correction
 *
 * [Purpose]
 * When performing multiple tests simultaneously, the probability of false positives
 * (Type I error rate) increases. Multiple testing correction addresses this problem.
 *
 * [Correction Methods]
 * - Bonferroni: Most conservative, adjusts by alpha/n
 * - Holm: Stepwise version of Bonferroni, more power
 * - Benjamini-Hochberg: Controls FDR (False Discovery Rate)
 */
void example_multiple_testing_correction() {
    print_section("8. Multiple Testing Correction");

    // Example: Testing multiple gene expressions
    std::vector<double> p_values = {0.01, 0.04, 0.03, 0.005, 0.12, 0.02};

    std::cout << "When performing multiple tests (e.g., gene expression analysis)\n\n";
    std::cout << "Original p-values: ";
    for (double p : p_values) std::cout << p << " ";
    std::cout << "\n\n";

    auto bonf = statcpp::bonferroni_correction(p_values);
    auto holm = statcpp::holm_correction(p_values);
    auto bh = statcpp::benjamini_hochberg_correction(p_values);

    std::cout << std::setw(8) << "Test"
              << std::setw(12) << "Original"
              << std::setw(12) << "Bonferroni"
              << std::setw(12) << "Holm"
              << std::setw(12) << "BH (FDR)" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (std::size_t i = 0; i < p_values.size(); ++i) {
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << p_values[i]
                  << std::setw(12) << bonf[i]
                  << std::setw(12) << holm[i]
                  << std::setw(12) << bh[i] << "\n";
    }

    std::cout << R"(
[Choosing a Correction Method]
- Bonferroni: Most conservative, for strict control of false positives
- Holm: More power than Bonferroni, recommended method
- Benjamini-Hochberg: Controls FDR, suitable for exploratory research
)";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Parametric Tests");

    std::cout << R"(
+----------------------------+--------------------------------------------+
| Function                   | Use Case                                   |
+----------------------------+--------------------------------------------+
| t_test()                   | Whether one sample mean differs from value |
| t_test_two_sample()        | Two-group mean comparison (equal variance) |
| t_test_welch()             | Two-group mean comparison (unequal var) *  |
| t_test_paired()            | Comparison of paired data                  |
| z_test()                   | One-sample mean (known variance)           |
| z_test_proportion()        | Whether proportion differs from value      |
| chisq_test_gof()           | Whether frequency matches expected         |
| chisq_test_independence()  | Whether two variables are independent      |
| f_test()                   | Whether two group variances are equal      |
+----------------------------+--------------------------------------------+
* Recommended

[Specifying Alternative Hypothesis]
- alternative_hypothesis::two_sided  -> Two-sided test (default)
- alternative_hypothesis::less       -> One-sided test (less)
- alternative_hypothesis::greater    -> One-sided test (greater)

[Multiple Testing Correction]
- bonferroni_correction()        -> Most conservative
- holm_correction()              -> Improved Bonferroni
- benjamini_hochberg_correction()-> FDR control

[Assumptions for Parametric Tests]
- Data follows a normal distribution (or sufficient sample size)
- Homogeneity of variance (required for some tests)
- Consider nonparametric tests if assumptions are not met
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_t_test();
    example_two_sample_t_test();
    example_paired_t_test();
    example_proportion_test();
    example_chisq_gof();
    example_chisq_independence();
    example_f_test();
    example_multiple_testing_correction();

    // Display summary
    print_summary();

    return 0;
}
