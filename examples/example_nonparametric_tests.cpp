/**
 * @file example_nonparametric_tests.cpp
 * @brief statcpp Nonparametric Test Functions Sample Code
 *
 * This file demonstrates the following functions from nonparametric_tests.hpp:
 * - shapiro_wilk_test(): Shapiro-Wilk normality test
 * - ks_test_normal(): Kolmogorov-Smirnov normality test
 * - levene_test(): Levene's test for homogeneity of variance (Brown-Forsythe version)
 * - bartlett_test(): Bartlett's test for homogeneity of variance
 * - wilcoxon_signed_rank_test(): Wilcoxon signed-rank test
 * - mann_whitney_u_test(): Mann-Whitney U test
 * - kruskal_wallis_test(): Kruskal-Wallis test
 * - fisher_exact_test(): Fisher's exact test
 *
 * Compilation:
 *   g++ -std=c++17 -I../statcpp/include example_nonparametric_tests.cpp -o example_nonparametric_tests
 *
 * Execution:
 *   ./example_nonparametric_tests
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#include "statcpp/nonparametric_tests.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

void print_section(const std::string& title)
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << " " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title)
{
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// Examples for Each Function
// ============================================================================

/**
 * @brief shapiro_wilk_test() example
 *
 * [Purpose]
 * The Shapiro-Wilk test tests whether data follows a normal distribution.
 * It is considered the most powerful normality test and is widely used.
 *
 * [Formula]
 * W = (Sum of a_i * x_(i))^2 / Sum of (x_i - x_bar)^2
 * where x_(i) are order statistics, a_i are coefficients (computed from normal distribution expectations)
 * W ranges from 0 to 1, closer to 1 indicates more normal
 *
 * [Use Cases]
 * - Checking prerequisites (normality) for parametric tests
 * - Data check before t-tests or ANOVA
 * - Confirming distribution of product characteristics in quality control
 *
 * [Notes]
 * - Optimized for sample size n <= 50
 * - n > 5000 is not supported
 * - Null hypothesis H0: Data follows normal distribution
 * - Small p-value (typically < 0.05) rejects normality
 */
void example_shapiro_wilk_test()
{
    print_section("shapiro_wilk_test() - Shapiro-Wilk Normality Test");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Data close to normal distribution (test scores)
    print_subsection("Case 1: Data Close to Normal Distribution");
    std::cout << "Scenario: Test scores of 100 students (mean ~70, std dev ~10)\n";

    std::mt19937 gen(42);
    std::normal_distribution<> normal(70.0, 10.0);

    std::vector<double> normal_data;
    for (int i = 0; i < 30; ++i) {
        normal_data.push_back(normal(gen));
    }

    std::cout << "Data (first 10): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << normal_data[i] << " ";
    }
    std::cout << "...\n";

    auto result1 = statcpp::shapiro_wilk_test(normal_data.begin(), normal_data.end());
    std::cout << "W statistic: " << result1.statistic << "\n";
    std::cout << "p-value: " << result1.p_value << "\n";
    std::cout << "Decision at alpha=0.05: "
              << (result1.p_value > 0.05 ? "Cannot reject normality (can assume normal distribution)"
                                          : "Reject normality (not normal distribution)") << "\n";

    // Case 2: Non-normal distribution data (exponential-like waiting times)
    print_subsection("Case 2: Non-Normal Distribution Data");
    std::cout << "Scenario: Call center waiting times (seconds) - exponential-like\n";

    std::exponential_distribution<> expo(0.1);
    std::vector<double> expo_data;
    for (int i = 0; i < 30; ++i) {
        expo_data.push_back(expo(gen));
    }

    std::cout << "Data (first 10): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << expo_data[i] << " ";
    }
    std::cout << "...\n";

    auto result2 = statcpp::shapiro_wilk_test(expo_data.begin(), expo_data.end());
    std::cout << "W statistic: " << result2.statistic << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision at alpha=0.05: "
              << (result2.p_value > 0.05 ? "Cannot reject normality"
                                          : "Reject normality (not normal distribution)") << "\n";

    // Case 3: Uniform distribution data
    print_subsection("Case 3: Uniform Distribution Data");
    std::cout << "Scenario: Results of rolling a die 30 times\n";

    std::uniform_int_distribution<> uniform(1, 6);
    std::vector<double> uniform_data;
    for (int i = 0; i < 30; ++i) {
        uniform_data.push_back(static_cast<double>(uniform(gen)));
    }

    auto result3 = statcpp::shapiro_wilk_test(uniform_data.begin(), uniform_data.end());
    std::cout << "W statistic: " << result3.statistic << "\n";
    std::cout << "p-value: " << result3.p_value << "\n";
    std::cout << "Decision at alpha=0.05: "
              << (result3.p_value > 0.05 ? "Cannot reject normality"
                                          : "Reject normality (not normal distribution)") << "\n";
}

/**
 * @brief ks_test_normal() example
 *
 * [Purpose]
 * The Kolmogorov-Smirnov test (KS test) tests the maximum deviation between
 * the empirical distribution function and theoretical normal distribution function.
 *
 * [Formula]
 * D = max|F_n(x) - F(x)|
 * where F_n(x) is empirical distribution function, F(x) is standard normal CDF
 * Data is standardized before testing
 *
 * [Use Cases]
 * - Normality test for large datasets
 * - When Shapiro-Wilk test cannot be used (n > 5000)
 * - Rough check of distribution shape
 *
 * [Notes]
 * - Tends to have lower power than Shapiro-Wilk test
 * - More sensitive to tails than center of distribution
 * - Uses estimated mean and variance (Lilliefors correction)
 */
void example_ks_test_normal()
{
    print_section("ks_test_normal() - Kolmogorov-Smirnov Normality Test");

    std::cout << std::fixed << std::setprecision(4);

    std::mt19937 gen(123);

    // Case 1: Normal distribution data
    print_subsection("Case 1: Normal Distribution Data");
    std::cout << "Scenario: Product weight measurements (100 samples)\n";

    std::normal_distribution<> normal(500.0, 5.0);
    std::vector<double> normal_data;
    for (int i = 0; i < 100; ++i) {
        normal_data.push_back(normal(gen));
    }

    auto result1 = statcpp::ks_test_normal(normal_data.begin(), normal_data.end());
    std::cout << "D statistic (max deviation): " << result1.statistic << "\n";
    std::cout << "p-value: " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value > 0.05 ? "Cannot reject normality" : "Reject normality") << "\n";

    // Case 2: Bimodal distribution
    print_subsection("Case 2: Bimodal Distribution");
    std::cout << "Scenario: Mixed male/female height data\n";

    std::normal_distribution<> male(170.0, 6.0);
    std::normal_distribution<> female(158.0, 5.0);
    std::bernoulli_distribution gender(0.5);

    std::vector<double> bimodal_data;
    for (int i = 0; i < 100; ++i) {
        if (gender(gen)) {
            bimodal_data.push_back(male(gen));
        } else {
            bimodal_data.push_back(female(gen));
        }
    }

    auto result2 = statcpp::ks_test_normal(bimodal_data.begin(), bimodal_data.end());
    std::cout << "D statistic: " << result2.statistic << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "Cannot reject normality" : "Reject normality") << "\n";

    // Comparison with Shapiro-Wilk
    print_subsection("Comparison with Shapiro-Wilk Test");
    std::vector<double> test_data;
    for (int i = 0; i < 50; ++i) {
        test_data.push_back(normal(gen));
    }

    auto sw_result = statcpp::shapiro_wilk_test(test_data.begin(), test_data.end());
    auto ks_result = statcpp::ks_test_normal(test_data.begin(), test_data.end());

    std::cout << "Results for same normal distribution data:\n";
    std::cout << "  Shapiro-Wilk: W = " << sw_result.statistic
              << ", p = " << sw_result.p_value << "\n";
    std::cout << "  KS test:      D = " << ks_result.statistic
              << ", p = " << ks_result.p_value << "\n";
}

/**
 * @brief levene_test() example
 *
 * [Purpose]
 * Levene's test tests whether variances of multiple groups are equal (homogeneity of variance).
 * Uses Brown-Forsythe version with deviations from median, robust to outliers.
 *
 * [Formula]
 * W = [(N-k) / (k-1)] * [Sum of n_i*(Z_bar_i - Z_bar)^2 / Sum of Sum of (Z_ij - Z_bar_i)^2]
 * where Z_ij = |X_ij - Median_i|
 * W follows F distribution
 *
 * [Use Cases]
 * - Checking prerequisites (homogeneity of variance) for ANOVA
 * - Comparing variance of 2+ groups
 * - Testing homogeneity of variance when normality is questionable
 *
 * [Notes]
 * - More robust to non-normality than Bartlett's test
 * - Each group needs at least 2 elements
 * - Null hypothesis H0: All group variances are equal
 */
void example_levene_test()
{
    print_section("levene_test() - Levene's Test for Homogeneity of Variance");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Groups with equal variance
    print_subsection("Case 1: Groups with Equal Variance");
    std::cout << "Scenario: Comparing product weight variability across 3 factories\n";

    std::vector<std::vector<double>> equal_var_groups = {
        {100.2, 99.8, 100.5, 99.5, 100.1, 99.9, 100.3, 99.7},  // Factory A
        {100.1, 99.9, 100.4, 99.6, 100.0, 99.8, 100.2, 99.8},  // Factory B
        {100.3, 99.7, 100.2, 99.8, 100.1, 99.9, 100.0, 100.0}  // Factory C
    };

    std::cout << "Factory A: ";
    for (double v : equal_var_groups[0]) std::cout << v << " ";
    std::cout << "\nFactory B: ";
    for (double v : equal_var_groups[1]) std::cout << v << " ";
    std::cout << "\nFactory C: ";
    for (double v : equal_var_groups[2]) std::cout << v << " ";
    std::cout << "\n\n";

    auto result1 = statcpp::levene_test(equal_var_groups);
    std::cout << "F statistic: " << result1.statistic << "\n";
    std::cout << "p-value: " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value > 0.05 ? "Cannot reject homogeneity (variances can be assumed equal)"
                                          : "Reject homogeneity (variances differ)") << "\n";

    // Case 2: Groups with unequal variance
    print_subsection("Case 2: Groups with Unequal Variance");
    std::cout << "Scenario: Task completion time variability by experience level\n";

    std::vector<std::vector<double>> unequal_var_groups = {
        {45, 52, 48, 55, 42, 58, 40, 60, 38, 62},  // Novice (high variability)
        {30, 32, 31, 33, 29, 34, 28, 35, 30, 31},  // Intermediate (medium variability)
        {25, 26, 25, 27, 24, 26, 25, 26, 25, 26}   // Expert (low variability)
    };

    std::cout << "Novice:       ";
    for (double v : unequal_var_groups[0]) std::cout << v << " ";
    std::cout << "\nIntermediate: ";
    for (double v : unequal_var_groups[1]) std::cout << v << " ";
    std::cout << "\nExpert:       ";
    for (double v : unequal_var_groups[2]) std::cout << v << " ";
    std::cout << "\n\n";

    // Calculate variance for each group
    for (std::size_t i = 0; i < unequal_var_groups.size(); ++i) {
        double var = statcpp::sample_variance(unequal_var_groups[i].begin(),
                                               unequal_var_groups[i].end());
        std::cout << "Group " << i + 1 << " variance: " << var << "\n";
    }

    auto result2 = statcpp::levene_test(unequal_var_groups);
    std::cout << "\nF statistic: " << result2.statistic << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "Cannot reject homogeneity"
                                          : "Reject homogeneity (variances differ)") << "\n";
}

/**
 * @brief bartlett_test() example
 *
 * [Purpose]
 * Bartlett's test tests whether variances of multiple groups are equal.
 * Assumes normal distribution, higher power than Levene's test but
 * sensitive to non-normality.
 *
 * [Formula]
 * chi^2 = [(N-k)*ln(S^2_p) - Sum of (n_i-1)*ln(S^2_i)] / C
 * where S^2_p is pooled variance, C is correction factor
 * Test statistic follows chi-square distribution
 *
 * [Use Cases]
 * - Testing homogeneity of variance when data normality is confirmed
 * - ANOVA prerequisite check (for normal data)
 *
 * [Notes]
 * - Very sensitive to non-normality
 * - Levene's test is recommended for non-normal data
 * - Error if any group has zero or negative variance
 */
void example_bartlett_test()
{
    print_section("bartlett_test() - Bartlett's Test for Homogeneity of Variance");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Equal variance (normal distribution data)
    print_subsection("Case 1: Equal Variance Normal Distribution Data");
    std::cout << "Scenario: Plant growth (cm) with 3 types of fertilizer\n";

    std::mt19937 gen(456);
    std::normal_distribution<> dist(15.0, 2.0);

    std::vector<std::vector<double>> equal_groups(3);
    for (int g = 0; g < 3; ++g) {
        for (int i = 0; i < 15; ++i) {
            equal_groups[g].push_back(dist(gen) + g * 0.5);  // Slightly different means
        }
    }

    for (int g = 0; g < 3; ++g) {
        std::cout << "Fertilizer " << static_cast<char>('A' + g) << " variance: "
                  << statcpp::sample_variance(equal_groups[g].begin(), equal_groups[g].end())
                  << "\n";
    }

    auto result1 = statcpp::bartlett_test(equal_groups);
    std::cout << "\nChi-square statistic: " << result1.statistic << "\n";
    std::cout << "Degrees of freedom: " << result1.df << "\n";
    std::cout << "p-value: " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value > 0.05 ? "Cannot reject homogeneity" : "Reject homogeneity") << "\n";

    // Case 2: Unequal variance
    print_subsection("Case 2: Clearly Unequal Variance Data");
    std::cout << "Scenario: Output from different machines in quality control\n";

    std::normal_distribution<> dist1(100.0, 1.0);   // Variance 1
    std::normal_distribution<> dist2(100.0, 3.0);   // Variance 9
    std::normal_distribution<> dist3(100.0, 5.0);   // Variance 25

    std::vector<std::vector<double>> unequal_groups(3);
    for (int i = 0; i < 20; ++i) {
        unequal_groups[0].push_back(dist1(gen));
        unequal_groups[1].push_back(dist2(gen));
        unequal_groups[2].push_back(dist3(gen));
    }

    for (int g = 0; g < 3; ++g) {
        std::cout << "Machine " << g + 1 << " variance: "
                  << statcpp::sample_variance(unequal_groups[g].begin(), unequal_groups[g].end())
                  << "\n";
    }

    auto result2 = statcpp::bartlett_test(unequal_groups);
    std::cout << "\nChi-square statistic: " << result2.statistic << "\n";
    std::cout << "Degrees of freedom: " << result2.df << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "Cannot reject homogeneity" : "Reject homogeneity") << "\n";

    // Comparison with Levene
    print_subsection("Comparison with Levene's Test");
    auto levene_result = statcpp::levene_test(unequal_groups);
    std::cout << "Results for same unequal variance data:\n";
    std::cout << "  Bartlett's test: chi^2 = " << result2.statistic
              << ", p = " << result2.p_value << "\n";
    std::cout << "  Levene's test:   F     = " << levene_result.statistic
              << ", p = " << levene_result.p_value << "\n";
}

/**
 * @brief wilcoxon_signed_rank_test() example
 *
 * [Purpose]
 * The Wilcoxon signed-rank test is a nonparametric test for paired two-group
 * differences or testing if a one-group median equals a specific value.
 * It is the nonparametric alternative to paired t-test.
 *
 * [Formula]
 * W+ = Sum of R_i (sum of ranks where difference is positive)
 * z = (W+ - E[W+]) / sqrt(Var(W+))  (normal approximation)
 *
 * [Use Cases]
 * - Comparing paired data when normality cannot be assumed
 * - Measuring before/after treatment effects
 * - Analyzing ordinal scale data
 *
 * [Notes]
 * - Data with zero differences are excluded
 * - At least 2 non-zero differences required
 * - Default is two-sided test
 */
void example_wilcoxon_signed_rank_test()
{
    print_section("wilcoxon_signed_rank_test() - Wilcoxon Signed-Rank Test");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Paired data (before/after)
    print_subsection("Case 1: Diet Program Effectiveness");
    std::cout << "Scenario: Weight change for 10 participants before/after diet\n";

    std::vector<double> before = {75.2, 82.1, 68.5, 90.3, 77.8, 85.6, 72.4, 88.9, 79.1, 83.2};
    std::vector<double> after  = {73.1, 79.5, 67.8, 86.2, 75.9, 82.3, 71.0, 85.4, 76.8, 80.1};

    // Calculate differences
    std::vector<double> diff;
    std::cout << "Participant: Before  After  Difference\n";
    for (std::size_t i = 0; i < before.size(); ++i) {
        double d = after[i] - before[i];
        diff.push_back(d);
        std::cout << "   " << i + 1 << ":       " << before[i] << "  " << after[i]
                  << "   " << (d >= 0 ? "+" : "") << d << "\n";
    }

    // H0: median = 0 (no difference)
    auto result1 = statcpp::wilcoxon_signed_rank_test(diff.begin(), diff.end(), 0.0);
    std::cout << "\nW+ statistic: " << result1.statistic << "\n";
    std::cout << "p-value (two-sided): " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value < 0.05 ? "Significant weight loss" : "No significant difference") << "\n";

    // One-sided test
    print_subsection("Case 1b: One-Sided Test (decrease direction)");
    auto result1_less = statcpp::wilcoxon_signed_rank_test(
        diff.begin(), diff.end(), 0.0, statcpp::alternative_hypothesis::less);
    std::cout << "p-value (one-sided, less): " << result1_less.p_value << "\n";
    std::cout << "Decision: "
              << (result1_less.p_value < 0.05 ? "Significant weight loss" : "No significant decrease") << "\n";

    // Case 2: One-sample median test
    print_subsection("Case 2: One-Sample Median Test");
    std::cout << "Scenario: Does product weight meet the specification of 500g?\n";

    std::vector<double> weights = {498.5, 501.2, 499.8, 502.1, 497.6, 500.5, 499.1, 501.8, 498.2, 500.9};

    std::cout << "Measurements: ";
    for (double w : weights) std::cout << w << " ";
    std::cout << "\n";

    // H0: median = 500
    auto result2 = statcpp::wilcoxon_signed_rank_test(weights.begin(), weights.end(), 500.0);
    std::cout << "\nW+ statistic: " << result2.statistic << "\n";
    std::cout << "p-value (two-sided): " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "No significant difference from 500g specification" : "Significant difference from 500g specification") << "\n";

    // Case 3: No effect case
    print_subsection("Case 3: No Effect Case");
    std::cout << "Scenario: Placebo group changes\n";

    std::vector<double> placebo_diff = {-0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, 0.0, -0.2};
    std::cout << "Differences: ";
    for (double d : placebo_diff) std::cout << d << " ";
    std::cout << "\n";

    auto result3 = statcpp::wilcoxon_signed_rank_test(placebo_diff.begin(), placebo_diff.end(), 0.0);
    std::cout << "W+ statistic: " << result3.statistic << "\n";
    std::cout << "p-value: " << result3.p_value << "\n";
    std::cout << "Decision: "
              << (result3.p_value > 0.05 ? "No significant change (as expected)" : "Significant change") << "\n";
}

/**
 * @brief mann_whitney_u_test() example
 *
 * [Purpose]
 * The Mann-Whitney U test is a nonparametric test for whether two independent
 * groups have the same distribution.
 * It is the nonparametric alternative to independent t-test.
 *
 * [Formula]
 * U1 = R1 - n1*(n1+1)/2
 * where R1 is rank sum of group 1, n1 is sample size of group 1
 * z = (U1 - E[U1]) / sqrt(Var(U1))  (normal approximation)
 *
 * [Use Cases]
 * - Comparing two groups when normality cannot be assumed
 * - Comparing ordinal scale data
 * - Small sample sizes
 *
 * [Notes]
 * - Each group needs at least 2 elements
 * - Includes tie correction
 * - Distribution shapes should be similar
 */
void example_mann_whitney_u_test()
{
    print_section("mann_whitney_u_test() - Mann-Whitney U Test");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Difference exists
    print_subsection("Case 1: Comparing Two Treatment Effectiveness");
    std::cout << "Scenario: Pain reduction scores (0-10) for new vs conventional drug\n";

    std::vector<double> new_drug  = {8, 7, 9, 6, 8, 7, 9, 8, 7, 8};
    std::vector<double> old_drug  = {5, 4, 6, 5, 4, 5, 6, 4, 5, 5};

    std::cout << "New drug:         ";
    for (double v : new_drug) std::cout << v << " ";
    std::cout << "\nConventional drug: ";
    for (double v : old_drug) std::cout << v << " ";
    std::cout << "\n\n";

    std::cout << "New drug median: " << statcpp::median(new_drug.begin(), new_drug.end()) << "\n";
    std::cout << "Conventional drug median: " << statcpp::median(old_drug.begin(), old_drug.end()) << "\n";

    auto result1 = statcpp::mann_whitney_u_test(
        new_drug.begin(), new_drug.end(),
        old_drug.begin(), old_drug.end());

    std::cout << "\nU statistic: " << result1.statistic << "\n";
    std::cout << "p-value (two-sided): " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value < 0.05 ? "Significant difference (new drug is more effective)" : "No significant difference") << "\n";

    // One-sided test
    print_subsection("Case 1b: One-Sided Test (new drug > conventional)");
    auto result1_greater = statcpp::mann_whitney_u_test(
        new_drug.begin(), new_drug.end(),
        old_drug.begin(), old_drug.end(),
        statcpp::alternative_hypothesis::greater);
    std::cout << "p-value (one-sided, greater): " << result1_greater.p_value << "\n";

    // Case 2: No difference
    print_subsection("Case 2: No Difference Case");
    std::cout << "Scenario: Test scores of two classes\n";

    std::vector<double> class_a = {72, 85, 78, 90, 82, 75, 88, 79};
    std::vector<double> class_b = {74, 81, 76, 89, 84, 77, 86, 80};

    std::cout << "Class A: ";
    for (double v : class_a) std::cout << v << " ";
    std::cout << "\nClass B: ";
    for (double v : class_b) std::cout << v << " ";
    std::cout << "\n";

    auto result2 = statcpp::mann_whitney_u_test(
        class_a.begin(), class_a.end(),
        class_b.begin(), class_b.end());

    std::cout << "\nU statistic: " << result2.statistic << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "No significant difference" : "Significant difference") << "\n";

    // Case 3: Ordinal scale data
    print_subsection("Case 3: Ordinal Scale Data");
    std::cout << "Scenario: Customer satisfaction survey (1=very dissatisfied to 5=very satisfied)\n";

    std::vector<double> store_a = {4, 5, 4, 3, 5, 4, 4, 5, 3, 4};
    std::vector<double> store_b = {3, 2, 3, 4, 2, 3, 3, 2, 4, 3};

    std::cout << "Store A: ";
    for (double v : store_a) std::cout << v << " ";
    std::cout << "\nStore B: ";
    for (double v : store_b) std::cout << v << " ";
    std::cout << "\n";

    auto result3 = statcpp::mann_whitney_u_test(
        store_a.begin(), store_a.end(),
        store_b.begin(), store_b.end());

    std::cout << "\nU statistic: " << result3.statistic << "\n";
    std::cout << "p-value: " << result3.p_value << "\n";
    std::cout << "Decision: "
              << (result3.p_value < 0.05 ? "Significant difference in satisfaction between stores" : "No significant difference") << "\n";
}

/**
 * @brief kruskal_wallis_test() example
 *
 * [Purpose]
 * The Kruskal-Wallis test is a nonparametric test for whether 3 or more
 * independent groups have the same distribution.
 * It is the nonparametric alternative to one-way ANOVA.
 *
 * [Formula]
 * H = [12 / (N*(N+1))] * Sum of (n_i * R_bar_i^2) - 3*(N+1)
 * where R_bar_i is mean rank of group i
 * H approximately follows chi^2(k-1) distribution
 *
 * [Use Cases]
 * - Comparing 3+ groups when normality cannot be assumed
 * - Comparing ordinal scale data across groups
 * - When ANOVA assumptions are not met
 *
 * [Notes]
 * - Rejecting null hypothesis indicates "difference exists somewhere"
 * - Multiple comparisons needed to identify which groups differ
 * - Each group must be non-empty
 */
void example_kruskal_wallis_test()
{
    print_section("kruskal_wallis_test() - Kruskal-Wallis Test");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Significant difference
    print_subsection("Case 1: Comparing 3 Teaching Methods");
    std::cout << "Scenario: Test scores for lecture, hands-on, and online methods\n";

    std::vector<std::vector<double>> teaching_methods = {
        {65, 70, 68, 72, 69, 71, 67, 73},  // Lecture
        {78, 82, 80, 85, 79, 83, 81, 84},  // Hands-on
        {72, 75, 74, 77, 73, 76, 74, 78}   // Online
    };

    std::vector<std::string> method_names = {"Lecture", "Hands-on", "Online"};
    for (std::size_t i = 0; i < teaching_methods.size(); ++i) {
        std::cout << method_names[i] << ": ";
        for (double v : teaching_methods[i]) std::cout << v << " ";
        std::cout << "\n  Median: " << statcpp::median(teaching_methods[i].begin(),
                                                        teaching_methods[i].end()) << "\n";
    }

    auto result1 = statcpp::kruskal_wallis_test(teaching_methods);
    std::cout << "\nH statistic: " << result1.statistic << "\n";
    std::cout << "Degrees of freedom: " << result1.df << "\n";
    std::cout << "p-value: " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value < 0.05 ? "Significant difference between teaching methods" : "No significant difference") << "\n";

    // Case 2: No significant difference
    print_subsection("Case 2: No Difference Case");
    std::cout << "Scenario: Sales (10k units) at 3 stores\n";

    std::vector<std::vector<double>> stores = {
        {120, 135, 128, 142, 131, 138, 125, 140},
        {125, 132, 130, 138, 128, 135, 127, 136},
        {122, 137, 126, 140, 133, 134, 129, 139}
    };

    for (std::size_t i = 0; i < stores.size(); ++i) {
        std::cout << "Store " << static_cast<char>('A' + i) << ": ";
        for (double v : stores[i]) std::cout << v << " ";
        std::cout << "\n";
    }

    auto result2 = statcpp::kruskal_wallis_test(stores);
    std::cout << "\nH statistic: " << result2.statistic << "\n";
    std::cout << "Degrees of freedom: " << result2.df << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "No significant difference between stores" : "Significant difference") << "\n";

    // Case 3: 4+ groups
    print_subsection("Case 3: Comparing 4 Groups");
    std::cout << "Scenario: Exercise habits (hours/week) by age group\n";

    std::vector<std::vector<double>> age_groups = {
        {5.5, 6.0, 4.5, 7.0, 5.0, 6.5, 4.0, 5.5},  // 20s
        {4.0, 3.5, 5.0, 3.0, 4.5, 3.5, 4.0, 4.5},  // 30s
        {2.5, 3.0, 2.0, 3.5, 2.5, 2.0, 3.0, 2.5},  // 40s
        {1.5, 2.0, 1.0, 2.5, 1.5, 2.0, 1.0, 1.5}   // 50s
    };

    std::vector<std::string> age_names = {"20s", "30s", "40s", "50s"};
    for (std::size_t i = 0; i < age_groups.size(); ++i) {
        std::cout << age_names[i] << " median: "
                  << statcpp::median(age_groups[i].begin(), age_groups[i].end())
                  << " hours/week\n";
    }

    auto result3 = statcpp::kruskal_wallis_test(age_groups);
    std::cout << "\nH statistic: " << result3.statistic << "\n";
    std::cout << "Degrees of freedom: " << result3.df << "\n";
    std::cout << "p-value: " << result3.p_value << "\n";
    std::cout << "Decision: "
              << (result3.p_value < 0.05 ? "Significant difference in exercise time between age groups"
                                          : "No significant difference") << "\n";
}

/**
 * @brief fisher_exact_test() example
 *
 * [Purpose]
 * Fisher's exact test tests whether two categorical variables in a 2x2
 * contingency table are independent.
 * Unlike chi-square test, applicable even with small expected frequencies.
 *
 * [Formula]
 * P(a) = C(a+c,a) * C(b+d,b) / C(n,a+b)
 * Based on hypergeometric distribution probability
 * Odds ratio = (a*d) / (b*c)
 *
 * [Use Cases]
 * - 2x2 contingency tables with small sample size
 * - When expected frequencies < 5 in some cells
 * - Testing association between two categorical variables
 *
 * [Notes]
 * - For 2x2 tables only
 * - Computationally expensive for large numbers
 * - Returns odds ratio as test statistic
 */
void example_fisher_exact_test()
{
    print_section("fisher_exact_test() - Fisher's Exact Test");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Association exists
    print_subsection("Case 1: Treatment and Recovery Association");
    std::cout << "Scenario: New drug effectiveness (recovered/not recovered)\n";

    std::cout << "Contingency table:\n";
    std::cout << "              Recovered  Not Recovered  Total\n";
    std::cout << "New drug         12           3          15\n";
    std::cout << "Control           5          10          15\n";
    std::cout << "Total            17          13          30\n\n";

    // a=12, b=3, c=5, d=10
    auto result1 = statcpp::fisher_exact_test(12, 3, 5, 10);
    std::cout << "Odds ratio: " << result1.statistic << "\n";
    std::cout << "p-value (two-sided): " << result1.p_value << "\n";
    std::cout << "Decision: "
              << (result1.p_value < 0.05 ? "Significant association between treatment and recovery"
                                          : "No significant association") << "\n";

    std::cout << "\nInterpretation: Odds ratio of " << result1.statistic
              << " means new drug group's recovery odds are about "
              << result1.statistic << " times the control group\n";

    // One-sided test
    print_subsection("Case 1b: One-Sided Test");
    auto result1_greater = statcpp::fisher_exact_test(12, 3, 5, 10,
                                                       statcpp::alternative_hypothesis::greater);
    std::cout << "p-value (one-sided, greater): " << result1_greater.p_value << "\n";
    std::cout << "H1: New drug is more effective than control\n";

    // Case 2: No association
    print_subsection("Case 2: No Association Case");
    std::cout << "Scenario: Gender and product preference\n";

    std::cout << "Contingency table:\n";
    std::cout << "            Product A  Product B  Total\n";
    std::cout << "Male            8          7        15\n";
    std::cout << "Female          7          8        15\n";
    std::cout << "Total          15         15        30\n\n";

    auto result2 = statcpp::fisher_exact_test(8, 7, 7, 8);
    std::cout << "Odds ratio: " << result2.statistic << "\n";
    std::cout << "p-value: " << result2.p_value << "\n";
    std::cout << "Decision: "
              << (result2.p_value > 0.05 ? "No significant association between gender and product preference"
                                          : "Significant association") << "\n";

    // Case 3: Extreme case
    print_subsection("Case 3: Extremely Skewed Data");
    std::cout << "Scenario: Rare adverse event occurrence\n";

    std::cout << "Contingency table:\n";
    std::cout << "              With AE  Without AE  Total\n";
    std::cout << "Drug group       5         95       100\n";
    std::cout << "Control          0        100       100\n";
    std::cout << "Total            5        195       200\n\n";

    auto result3 = statcpp::fisher_exact_test(5, 95, 0, 100);
    std::cout << "Odds ratio: " << result3.statistic << " (infinity: zero in control group)\n";
    std::cout << "p-value: " << result3.p_value << "\n";
    std::cout << "Decision: "
              << (result3.p_value < 0.05 ? "Significant association between drug and adverse events"
                                          : "No significant association") << "\n";

    // Case 4: Small sample size
    print_subsection("Case 4: Small Sample");
    std::cout << "Scenario: Pilot study results\n";

    std::cout << "Contingency table:\n";
    std::cout << "              Success  Failure  Total\n";
    std::cout << "Experimental      4        1       5\n";
    std::cout << "Control           1        4       5\n";
    std::cout << "Total             5        5      10\n\n";

    auto result4 = statcpp::fisher_exact_test(4, 1, 1, 4);
    std::cout << "Odds ratio: " << result4.statistic << "\n";
    std::cout << "p-value: " << result4.p_value << "\n";
    std::cout << "Decision: "
              << (result4.p_value < 0.05 ? "Significant association"
                                          : "Small sample size makes significance detection difficult") << "\n";
}

/**
 * @brief Composite example: Parametric vs Nonparametric test selection
 */
void example_test_selection()
{
    print_section("Test Selection: Parametric vs Nonparametric");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Test Selection Based on Data Characteristics");
    std::cout << "Flow for selecting test when comparing effectiveness of two treatments\n\n";

    // Generate skewed distribution data
    std::mt19937 gen(789);
    std::exponential_distribution<> expo1(0.1);
    std::exponential_distribution<> expo2(0.15);

    std::vector<double> treatment_a, treatment_b;
    for (int i = 0; i < 20; ++i) {
        treatment_a.push_back(expo1(gen));
        treatment_b.push_back(expo2(gen));
    }

    std::cout << "Step 1: Check Normality\n";
    auto sw_a = statcpp::shapiro_wilk_test(treatment_a.begin(), treatment_a.end());
    auto sw_b = statcpp::shapiro_wilk_test(treatment_b.begin(), treatment_b.end());

    std::cout << "  Treatment A: W = " << sw_a.statistic << ", p = " << sw_a.p_value << "\n";
    std::cout << "  Treatment B: W = " << sw_b.statistic << ", p = " << sw_b.p_value << "\n";

    bool normal_a = sw_a.p_value > 0.05;
    bool normal_b = sw_b.p_value > 0.05;

    std::cout << "  Result: " << (normal_a ? "Group A is normal" : "Group A is non-normal")
              << ", " << (normal_b ? "Group B is normal" : "Group B is non-normal") << "\n\n";

    if (normal_a && normal_b) {
        std::cout << "Step 2: Both groups are normal, so check homogeneity of variance\n";
        std::vector<std::vector<double>> groups = {treatment_a, treatment_b};
        auto bartlett = statcpp::bartlett_test(groups);
        std::cout << "  Bartlett's test: chi^2 = " << bartlett.statistic
                  << ", p = " << bartlett.p_value << "\n";

        if (bartlett.p_value > 0.05) {
            std::cout << "  -> Equal variance: use Student's t-test\n";
        } else {
            std::cout << "  -> Unequal variance: use Welch's t-test\n";
        }
    } else {
        std::cout << "Step 2: Normality not satisfied, use nonparametric test\n";

        auto mw = statcpp::mann_whitney_u_test(
            treatment_a.begin(), treatment_a.end(),
            treatment_b.begin(), treatment_b.end());

        std::cout << "  Mann-Whitney U test: U = " << mw.statistic
                  << ", p = " << mw.p_value << "\n";
        std::cout << "  Decision: "
                  << (mw.p_value < 0.05 ? "Significant difference between groups" : "No significant difference") << "\n";
    }

    print_subsection("Test Selection Guidelines");
    std::cout << "+-------------------------------------------------------------+\n";
    std::cout << "| Situation                        Recommended Test           |\n";
    std::cout << "+-------------------------------------------------------------+\n";
    std::cout << "| Normality test (1 group)         Shapiro-Wilk test          |\n";
    std::cout << "| Homogeneity of var (normal)      Bartlett's test            |\n";
    std::cout << "| Homogeneity of var (non-normal)  Levene's test              |\n";
    std::cout << "| 2-group comparison (normal, EV)  Student's t-test           |\n";
    std::cout << "| 2-group comparison (normal, UV)  Welch's t-test             |\n";
    std::cout << "| 2-group comparison (non-normal)  Mann-Whitney U test        |\n";
    std::cout << "| Paired 2-group (non-normal)      Wilcoxon signed-rank test  |\n";
    std::cout << "| 3+ groups (non-normal)           Kruskal-Wallis test        |\n";
    std::cout << "| 2x2 table (small sample)         Fisher's exact test        |\n";
    std::cout << "+-------------------------------------------------------------+\n";
}

// ============================================================================
// Summary Output
// ============================================================================

void print_summary()
{
    print_section("Summary: nonparametric_tests.hpp Function List");

    std::cout << R"(
+----------------------------------------------------------------------------+
| Function                          Purpose                                   |
+----------------------------------------------------------------------------+
| shapiro_wilk_test()               Normality test (high power, n<=5000)      |
| ks_test_normal()                  Normality test (KS test)                  |
| levene_test()                     Homogeneity of variance (robust)          |
| bartlett_test()                   Homogeneity of variance (assumes normal)  |
| wilcoxon_signed_rank_test()       1-sample/paired location test             |
| mann_whitney_u_test()             Independent 2-sample location test        |
| kruskal_wallis_test()             Independent k-sample location test        |
| fisher_exact_test()               2x2 table independence test               |
+----------------------------------------------------------------------------+

[Advantages of Nonparametric Tests]
  - No distribution assumptions like normality needed
  - Robust to outliers
  - Applicable to ordinal scale data
  - Applicable to small samples

[Disadvantages of Nonparametric Tests]
  - Generally lower power than parametric tests
  - Confidence interval calculation is more complex

[Interpreting Test Results]
  - test_result structure contains statistic, p_value, df, alternative
  - p-value < significance level -> Reject null hypothesis
  - p-value >= significance level -> Cannot reject null hypothesis (not acceptance)
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::cout << "==========================================================\n";
    std::cout << " statcpp Nonparametric Test Functions Sample Code\n";
    std::cout << "==========================================================\n";

    example_shapiro_wilk_test();
    example_ks_test_normal();
    example_levene_test();
    example_bartlett_test();
    example_wilcoxon_signed_rank_test();
    example_mann_whitney_u_test();
    example_kruskal_wallis_test();
    example_fisher_exact_test();
    example_test_selection();
    print_summary();

    return 0;
}
