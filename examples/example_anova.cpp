/**
 * @file example_anova.cpp
 * @brief Sample code for statcpp Analysis of Variance (ANOVA) functions
 *
 * This file explains the following functions provided in anova.hpp:
 * - one_way_anova(): One-way analysis of variance
 * - two_way_anova(): Two-way analysis of variance
 * - tukey_hsd(): Tukey HSD test using studentized range distribution (post-hoc comparison)
 * - bonferroni_posthoc(): Bonferroni method (post-hoc comparison)
 * - dunnett_posthoc(): Dunnett method (comparison with control group)
 * - scheffe_posthoc(): Scheffe method (post-hoc comparison)
 * - one_way_ancova(): One-way analysis of covariance
 * - eta_squared(), omega_squared(), cohens_f(): Effect sizes
 *
 * Compilation:
 *   g++ -std=c++17 -I../statcpp/include example_anova.cpp -o example_anova
 *
 * Execution:
 *   ./example_anova
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "statcpp/anova.hpp"
#include "statcpp/basic_statistics.hpp"

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

void print_anova_table_oneway(const statcpp::one_way_anova_result& result)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nANOVA Table:\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Source             SS        df      MS        F       p-value\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Between Groups  " << std::setw(8) << result.between.ss
              << "  " << std::setw(4) << result.between.df
              << "  " << std::setw(8) << result.between.ms
              << "  " << std::setw(6) << result.between.f_statistic
              << "  " << std::setw(6) << result.between.p_value << "\n";
    std::cout << "Within Groups   " << std::setw(8) << result.within.ss
              << "  " << std::setw(4) << result.within.df
              << "  " << std::setw(8) << result.within.ms << "\n";
    std::cout << "Total           " << std::setw(8) << result.ss_total
              << "  " << std::setw(4) << result.df_total << "\n";
    std::cout << "--------------------------------------------------------------------\n";
}

void print_anova_table_twoway(const statcpp::two_way_anova_result& result)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nANOVA Table:\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Source           SS        df      MS        F       p-value\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Factor A      " << std::setw(8) << result.factor_a.ss
              << "  " << std::setw(4) << result.factor_a.df
              << "  " << std::setw(8) << result.factor_a.ms
              << "  " << std::setw(6) << result.factor_a.f_statistic
              << "  " << std::setw(6) << result.factor_a.p_value << "\n";
    std::cout << "Factor B      " << std::setw(8) << result.factor_b.ss
              << "  " << std::setw(4) << result.factor_b.df
              << "  " << std::setw(8) << result.factor_b.ms
              << "  " << std::setw(6) << result.factor_b.f_statistic
              << "  " << std::setw(6) << result.factor_b.p_value << "\n";
    std::cout << "Interaction(AxB) " << std::setw(8) << result.interaction.ss
              << "  " << std::setw(4) << result.interaction.df
              << "  " << std::setw(8) << result.interaction.ms
              << "  " << std::setw(6) << result.interaction.f_statistic
              << "  " << std::setw(6) << result.interaction.p_value << "\n";
    std::cout << "Error         " << std::setw(8) << result.error.ss
              << "  " << std::setw(4) << result.error.df
              << "  " << std::setw(8) << result.error.ms << "\n";
    std::cout << "Total         " << std::setw(8) << result.ss_total
              << "  " << std::setw(4) << result.df_total << "\n";
    std::cout << "--------------------------------------------------------------------\n";
}

// ============================================================================
// Sample Code for Each Function
// ============================================================================

/**
 * @brief Sample for one_way_anova()
 *
 * [Purpose]
 * One-way ANOVA tests whether there are differences among the means
 * of three or more groups. It analyzes the effect of one independent
 * variable (factor) on a dependent variable.
 *
 * [Formula]
 * F = MS_between / MS_within
 * SS_between = sum of ni(xi_bar - x_bar)^2  (between-group variation)
 * SS_within = sum of sum of (xij - xi_bar)^2 (within-group variation)
 *
 * [Use Cases]
 * - Comparing effectiveness of multiple treatments/teaching methods
 * - Comparing means across multiple groups
 * - Testing factor effects in experimental design
 *
 * [Notes]
 * - Requires assumptions of normality and homogeneity of variance
 * - A significant result only indicates "there is a difference somewhere"
 * - Post-hoc comparisons are needed to identify which groups differ
 */
void example_one_way_anova()
{
    print_section("one_way_anova() - One-Way Analysis of Variance");

    std::cout << std::fixed << std::setprecision(4);

    // Case 1: Significant difference
    print_subsection("Case 1: Comparing Three Teaching Methods");
    std::cout << "Scenario: Test scores (out of 100) for Lecture, Exercise, and PBL methods\n";

    std::vector<std::vector<double>> teaching_methods = {
        {65, 70, 68, 72, 69, 71, 67, 73, 66, 70},  // Lecture
        {78, 82, 80, 85, 79, 83, 81, 84, 77, 80},  // Exercise
        {72, 75, 74, 77, 73, 76, 74, 78, 71, 75}   // PBL
    };

    std::vector<std::string> method_names = {"Lecture", "Exercise", "PBL"};
    for (std::size_t i = 0; i < teaching_methods.size(); ++i) {
        std::cout << method_names[i] << " mean: "
                  << statcpp::mean(teaching_methods[i].begin(), teaching_methods[i].end())
                  << "\n";
    }

    auto result1 = statcpp::one_way_anova(teaching_methods);
    print_anova_table_oneway(result1);

    std::cout << "\nConclusion: "
              << (result1.between.p_value < 0.05 ? "Significant difference among teaching methods (reject H0)"
                                                  : "No significant difference") << "\n";
    std::cout << "Grand mean: " << result1.grand_mean << "\n";

    // Case 2: No significant difference
    print_subsection("Case 2: No Difference");
    std::cout << "Scenario: Sales (in 10,000 yen) from 3 stores\n";

    std::vector<std::vector<double>> stores = {
        {120, 135, 128, 142, 131},
        {125, 132, 130, 138, 128},
        {122, 137, 126, 140, 133}
    };

    for (std::size_t i = 0; i < stores.size(); ++i) {
        std::cout << "Store " << static_cast<char>('A' + i) << " mean: "
                  << statcpp::mean(stores[i].begin(), stores[i].end()) << "\n";
    }

    auto result2 = statcpp::one_way_anova(stores);
    print_anova_table_oneway(result2);

    std::cout << "\nConclusion: "
              << (result2.between.p_value > 0.05 ? "No significant difference among stores" : "Significant difference") << "\n";

    // Case 3: Four-group comparison
    print_subsection("Case 3: Comparing Four Fertilizers");
    std::cout << "Scenario: Plant growth (cm) with 4 types of fertilizer\n";

    std::vector<std::vector<double>> fertilizers = {
        {15.2, 16.1, 14.8, 15.9, 15.5, 16.0},  // Fertilizer A
        {18.3, 19.1, 17.8, 18.5, 18.9, 18.2},  // Fertilizer B
        {16.5, 17.2, 16.8, 17.0, 16.7, 17.1},  // Fertilizer C
        {14.1, 15.0, 14.5, 14.8, 14.3, 14.7}   // Fertilizer D (control)
    };

    std::vector<std::string> fert_names = {"A", "B", "C", "D(control)"};
    for (std::size_t i = 0; i < fertilizers.size(); ++i) {
        std::cout << "Fertilizer " << fert_names[i] << " mean: "
                  << statcpp::mean(fertilizers[i].begin(), fertilizers[i].end()) << "\n";
    }

    auto result3 = statcpp::one_way_anova(fertilizers);
    print_anova_table_oneway(result3);

    std::cout << "\nConclusion: "
              << (result3.between.p_value < 0.05 ? "Significant difference among fertilizers" : "No significant difference") << "\n";

    // Effect size calculation
    print_subsection("Effect Sizes for Case 1");
    double eta_sq = statcpp::eta_squared(result1);
    double omega_sq = statcpp::omega_squared(result1);
    double cohens_f_val = statcpp::cohens_f(result1);

    std::cout << "Eta-squared:      " << eta_sq << "\n";
    std::cout << "Omega-squared:    " << omega_sq << "\n";
    std::cout << "Cohen's f:        " << cohens_f_val << "\n";
    std::cout << "\nEffect size interpretation:\n";
    std::cout << "  Eta-squared - Proportion of variance explained (" << (eta_sq * 100) << "%)\n";
    if (cohens_f_val < 0.10) std::cout << "  Cohen's f: Small effect\n";
    else if (cohens_f_val < 0.25) std::cout << "  Cohen's f: Medium effect\n";
    else if (cohens_f_val < 0.40) std::cout << "  Cohen's f: Large effect\n";
    else std::cout << "  Cohen's f: Very large effect\n";
}

/**
 * @brief Sample for two_way_anova()
 *
 * [Purpose]
 * Two-way ANOVA analyzes the effects of two independent variables (factors)
 * and their interaction on a dependent variable simultaneously.
 *
 * [Formula]
 * F_A = MS_A / MS_error  (main effect of Factor A)
 * F_B = MS_B / MS_error  (main effect of Factor B)
 * F_AB = MS_AB / MS_error (interaction effect)
 *
 * [Use Cases]
 * - When you want to test effects of two factors simultaneously
 * - When you want to examine interaction between factors
 * - Multi-factor experiments in experimental design
 *
 * [Notes]
 * - Interpret main effects cautiously when interaction is significant
 * - Equal sample sizes per cell required (balanced design)
 * - Assumes normality and homogeneity of variance
 */
void example_two_way_anova()
{
    print_section("two_way_anova() - Two-Way Analysis of Variance");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Case: Temperature x Humidity Effect on Product Quality");
    std::cout << "Scenario: Effect of temperature (low/high) and humidity (low/medium/high) on product quality\n";
    std::cout << "4 measurements per condition\n\n";

    // data[temperature][humidity][replication]
    // Temperature: 0=low, 1=high
    // Humidity: 0=low, 1=medium, 2=high
    std::vector<std::vector<std::vector<double>>> data = {
        {   // Low temperature
            {85, 87, 86, 88},  // Low humidity
            {82, 84, 83, 85},  // Medium humidity
            {78, 80, 79, 81}   // High humidity
        },
        {   // High temperature
            {88, 90, 89, 91},  // Low humidity
            {90, 92, 91, 93},  // Medium humidity
            {87, 89, 88, 90}   // High humidity
        }
    };

    // Display cell means
    std::cout << "Cell means:\n";
    std::cout << "             Low Hum  Med Hum  High Hum\n";
    for (std::size_t temp = 0; temp < 2; ++temp) {
        std::cout << (temp == 0 ? "Low Temp    " : "High Temp   ");
        for (std::size_t hum = 0; hum < 3; ++hum) {
            double mean = statcpp::mean(data[temp][hum].begin(), data[temp][hum].end());
            std::cout << std::setw(7) << mean << " ";
        }
        std::cout << "\n";
    }

    auto result = statcpp::two_way_anova(data);
    print_anova_table_twoway(result);

    std::cout << "\nConclusions:\n";
    std::cout << "  Factor A (Temperature): "
              << (result.factor_a.p_value < 0.05 ? "Significant (p < 0.05)" : "Not significant") << "\n";
    std::cout << "  Factor B (Humidity): "
              << (result.factor_b.p_value < 0.05 ? "Significant (p < 0.05)" : "Not significant") << "\n";
    std::cout << "  Interaction (Temp x Humidity): "
              << (result.interaction.p_value < 0.05 ? "Significant (p < 0.05)" : "Not significant") << "\n";

    // Effect sizes (partial eta-squared)
    print_subsection("Effect Sizes (Partial Eta-squared)");
    double partial_eta_a = statcpp::partial_eta_squared_a(result);
    double partial_eta_b = statcpp::partial_eta_squared_b(result);
    double partial_eta_ab = statcpp::partial_eta_squared_interaction(result);

    std::cout << "Temperature partial eta-sq:   " << partial_eta_a << "\n";
    std::cout << "Humidity partial eta-sq:      " << partial_eta_b << "\n";
    std::cout << "Interaction partial eta-sq:   " << partial_eta_ab << "\n";

    // Case without interaction
    print_subsection("Case 2: No Interaction");
    std::cout << "Scenario: Drug A x Drug B combination effect (additive)\n";

    std::vector<std::vector<std::vector<double>>> data2 = {
        {   // Drug A: absent
            {10, 11, 10, 11},  // Drug B: absent
            {15, 16, 15, 16}   // Drug B: present
        },
        {   // Drug A: present
            {20, 21, 20, 21},  // Drug B: absent
            {25, 26, 25, 26}   // Drug B: present
        }
    };

    std::cout << "\nCell means:\n";
    std::cout << "            Drug B:No  Drug B:Yes\n";
    for (std::size_t a_level = 0; a_level < 2; ++a_level) {
        std::cout << (a_level == 0 ? "Drug A:No   " : "Drug A:Yes  ");
        for (std::size_t b_level = 0; b_level < 2; ++b_level) {
            double mean = statcpp::mean(data2[a_level][b_level].begin(),
                                        data2[a_level][b_level].end());
            std::cout << std::setw(10) << mean << "  ";
        }
        std::cout << "\n";
    }

    auto result2 = statcpp::two_way_anova(data2);
    print_anova_table_twoway(result2);

    std::cout << "\nInterpretation: When interaction is not significant, effects of each factor are independent (additive)\n";
}

/**
 * @brief Sample for tukey_hsd()
 *
 * [Purpose]
 * Performs Tukey's Honestly Significant Difference (HSD) test as a post-hoc
 * comparison to determine which pairs of groups differ after ANOVA
 * shows a significant difference. Uses the studentized range distribution
 * (Tukey-Kramer method for unequal sample sizes).
 *
 * [Formula]
 * q = |mean_i - mean_j| / SE
 * where SE = sqrt(MSE/2 * (1/n_i + 1/n_j)),
 * p-values from the studentized range distribution with k groups and df_error
 *
 * [Use Cases]
 * - All pairwise comparisons after ANOVA
 * - Works with both equal and unequal sample sizes (Tukey-Kramer)
 * - Exact control of familywise Type I error rate
 *
 * [Notes]
 * - Only perform when ANOVA is significant
 * - Tests all possible pairwise comparisons
 * - More powerful than Bonferroni for all-pairwise comparisons
 */
void example_tukey_hsd()
{
    print_section("tukey_hsd() - Tukey HSD (Post-hoc Comparison)");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Comparing Four Diet Methods");
    std::cout << "Weight loss (kg) comparison\n";

    std::vector<std::vector<double>> diets = {
        {5.2, 6.1, 4.8, 5.9, 5.5, 6.0},  // Diet A
        {8.3, 9.1, 7.8, 8.5, 8.9, 8.2},  // Diet B
        {6.5, 7.2, 6.8, 7.0, 6.7, 7.1},  // Diet C
        {4.1, 5.0, 4.5, 4.8, 4.3, 4.7}   // Diet D
    };

    std::vector<std::string> diet_names = {"A", "B", "C", "D"};
    for (std::size_t i = 0; i < diets.size(); ++i) {
        std::cout << "Diet " << diet_names[i] << " mean: "
                  << statcpp::mean(diets[i].begin(), diets[i].end()) << " kg\n";
    }

    // First perform ANOVA
    auto anova_result = statcpp::one_way_anova(diets);
    print_anova_table_oneway(anova_result);

    if (anova_result.between.p_value < 0.05) {
        std::cout << "\nANOVA detected significant difference. Proceeding with post-hoc comparison.\n";

        auto tukey_result = statcpp::tukey_hsd(anova_result, diets, 0.05);

        std::cout << "\n" << tukey_result.method << " Results:\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "Comparison   Mean Diff    SE    Statistic  p-value   95% CI        Decision\n";
        std::cout << "------------------------------------------------------------\n";

        for (const auto& comp : tukey_result.comparisons) {
            std::cout << diet_names[comp.group1] << " vs " << diet_names[comp.group2]
                      << "  " << std::setw(7) << comp.mean_diff
                      << "  " << std::setw(6) << comp.se
                      << "  " << std::setw(7) << comp.statistic
                      << "  " << std::setw(6) << comp.p_value
                      << "  [" << std::setw(6) << comp.lower << ", " << std::setw(6) << comp.upper << "]"
                      << "  " << (comp.significant ? "Sig.*" : "n.s.") << "\n";
        }
        std::cout << "------------------------------------------------------------\n";
        std::cout << "* Significance level alpha = " << tukey_result.alpha << "\n";
    }
}

/**
 * @brief Sample for bonferroni_posthoc()
 *
 * [Purpose]
 * Bonferroni method is a conservative post-hoc comparison method
 * that adjusts the significance level by the number of comparisons.
 *
 * [Formula]
 * Adjusted alpha = alpha / m  (where m is the number of pairs)
 * or p_adjusted = p x m
 *
 * [Use Cases]
 * - When there are few groups to compare
 * - When conservative decisions are desired
 * - Simple and easy to understand
 *
 * [Notes]
 * - Very conservative (low power)
 * - Becomes overly conservative with many groups
 */
void example_bonferroni_posthoc()
{
    print_section("bonferroni_posthoc() - Bonferroni Method (Post-hoc Comparison)");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Comparing Three Treatments");

    std::vector<std::vector<double>> treatments = {
        {72, 75, 74, 77, 73, 76, 74, 78},  // Treatment A
        {65, 68, 67, 70, 66, 69, 67, 71},  // Treatment B (control)
        {78, 81, 80, 83, 79, 82, 80, 84}   // Treatment C
    };

    auto anova_result = statcpp::one_way_anova(treatments);
    auto bonf_result = statcpp::bonferroni_posthoc(anova_result, 0.05);

    std::cout << "\n" << bonf_result.method << " Results:\n";
    std::cout << "Number of comparisons: " << bonf_result.comparisons.size() << "\n";
    std::cout << "Adjusted alpha: " << (0.05 / bonf_result.comparisons.size()) << "\n\n";

    for (const auto& comp : bonf_result.comparisons) {
        std::cout << "Group " << comp.group1 + 1 << " vs Group " << comp.group2 + 1
                  << ": Mean diff = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[Significant]" : "[n.s.]") << "\n";
    }
}

/**
 * @brief Sample for dunnett_posthoc()
 *
 * [Purpose]
 * Dunnett method is a specialized post-hoc comparison method for
 * comparing each treatment group against a control group.
 * Higher power than all-pairwise comparisons.
 *
 * [Formula]
 * t = (x_bar_i - x_bar_control) / SE
 * Critical value from Dunnett distribution
 *
 * [Use Cases]
 * - Comparison with control group (placebo, standard treatment)
 * - Multiple treatment groups vs one control group
 * - Dose-response studies in drug development
 *
 * [Notes]
 * - Control group must be clearly defined
 * - Does not compare between non-control groups
 */
void example_dunnett_posthoc()
{
    print_section("dunnett_posthoc() - Dunnett Method (Comparison with Control)");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Three New Drugs vs Placebo");
    std::cout << "Pain relief score (0-100)\n";

    std::vector<std::vector<double>> drugs = {
        {35, 38, 36, 40, 37, 39, 36, 38},  // Placebo (control)
        {52, 55, 54, 57, 53, 56, 54, 58},  // Drug A
        {48, 51, 50, 53, 49, 52, 50, 54},  // Drug B
        {45, 48, 47, 50, 46, 49, 47, 51}   // Drug C
    };

    std::vector<std::string> drug_names = {"Placebo", "Drug A", "Drug B", "Drug C"};
    for (std::size_t i = 0; i < drugs.size(); ++i) {
        std::cout << drug_names[i] << " mean: "
                  << statcpp::mean(drugs[i].begin(), drugs[i].end()) << "\n";
    }

    auto anova_result = statcpp::one_way_anova(drugs);
    auto dunnett_result = statcpp::dunnett_posthoc(anova_result, 0, 0.05);  // 0 = Placebo

    std::cout << "\n" << dunnett_result.method << " Results (Control = Placebo):\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto& comp : dunnett_result.comparisons) {
        std::cout << drug_names[comp.group1] << " vs " << drug_names[comp.group2]
                  << ": Diff = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[Significantly more effective than placebo]" : "[No significant difference]")
                  << "\n";
    }
}

/**
 * @brief Sample for scheffe_posthoc()
 *
 * [Purpose]
 * Scheffe method is the most flexible but conservative post-hoc
 * comparison method that can test all possible contrasts (linear combinations).
 *
 * [Formula]
 * S = sqrt[(k-1) x F_critical]
 * Contrast C is significant if |C/SE(C)| > S
 *
 * [Use Cases]
 * - When you want to test complex contrasts (e.g., A vs (B+C)/2)
 * - When contrasts are decided after looking at the data
 * - When maximum flexibility in post-hoc comparison is needed
 *
 * [Notes]
 * - Very conservative (low power)
 * - Not suitable for simple pairwise comparisons
 */
void example_scheffe_posthoc()
{
    print_section("scheffe_posthoc() - Scheffe Method (Post-hoc Comparison)");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Comparing Four Advertising Methods");

    std::vector<std::vector<double>> ads = {
        {45, 48, 47, 50, 46, 49},  // TV ads
        {52, 55, 54, 57, 53, 56},  // Web ads
        {42, 45, 44, 47, 43, 46},  // Newspaper ads
        {38, 41, 40, 43, 39, 42}   // Radio ads
    };

    auto anova_result = statcpp::one_way_anova(ads);
    auto scheffe_result = statcpp::scheffe_posthoc(anova_result, 0.05);

    std::cout << "\n" << scheffe_result.method << " Results:\n";
    std::cout << "Scheffe method protects against all contrasts\n\n";

    std::vector<std::string> ad_names = {"TV", "Web", "Newspaper", "Radio"};
    for (const auto& comp : scheffe_result.comparisons) {
        std::cout << ad_names[comp.group1] << " vs " << ad_names[comp.group2]
                  << ": Diff = " << comp.mean_diff
                  << ", p = " << comp.p_value
                  << " " << (comp.significant ? "[Sig.]" : "[n.s.]") << "\n";
    }

    std::cout << "\nAdvantage of Scheffe method: Can test complex contrasts post-hoc\n";
    std::cout << "Example: (TV + Web) / 2 vs (Newspaper + Radio) / 2 comparison is also possible\n";
}

/**
 * @brief Sample for one_way_ancova()
 *
 * [Purpose]
 * Analysis of Covariance (ANCOVA) tests group differences while
 * controlling for the effect of a covariate (confounding variable).
 *
 * [Formula]
 * Adjusted mean = y_bar_i - b(x_bar_i - x_bar_grand)
 * where b is the common regression coefficient and x is the covariate
 *
 * [Use Cases]
 * - Controlling for baseline values before treatment
 * - Removing effects of covariates like age, gender
 * - Correction when experimental control is insufficient
 *
 * [Notes]
 * - Assumes equal regression slopes across groups
 * - Assumes linear relationship between covariate and dependent variable
 * - Covariate should be independent of treatment
 */
void example_one_way_ancova()
{
    print_section("one_way_ancova() - One-Way Analysis of Covariance");

    std::cout << std::fixed << std::setprecision(4);

    print_subsection("Scenario: Comparing Three Learning Methods (Adjusted for Pretest)");
    std::cout << "Dependent variable: Posttest score\n";
    std::cout << "Covariate:          Pretest score\n\n";

    // (posttest score, pretest score) pairs
    std::vector<std::vector<std::pair<double, double>>> learning_methods = {
        {   // Learning Method A
            {75, 60}, {80, 65}, {78, 62}, {82, 68}, {77, 63}, {79, 64}
        },
        {   // Learning Method B
            {85, 70}, {90, 75}, {88, 72}, {92, 78}, {87, 73}, {89, 74}
        },
        {   // Learning Method C
            {78, 65}, {83, 70}, {81, 67}, {85, 73}, {80, 68}, {82, 69}
        }
    };

    std::vector<std::string> method_labels = {"Method A", "Method B", "Method C"};

    // Display raw means for each group
    std::cout << "Group means:\n";
    for (std::size_t i = 0; i < learning_methods.size(); ++i) {
        double sum_post = 0.0, sum_pre = 0.0;
        for (const auto& [post, pre] : learning_methods[i]) {
            sum_post += post;
            sum_pre += pre;
        }
        std::cout << method_labels[i]
                  << " - Pretest: " << (sum_pre / learning_methods[i].size())
                  << ", Posttest: " << (sum_post / learning_methods[i].size()) << "\n";
    }

    auto ancova_result = statcpp::one_way_ancova(learning_methods);

    std::cout << "\nANCOVA Table:\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Source            SS       df      MS       F      p-value\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Covariate(Pre) " << std::setw(8) << ancova_result.ss_covariate
              << "  " << std::setw(4) << ancova_result.df_covariate
              << "  " << std::setw(8) << ancova_result.ms_covariate
              << "  " << std::setw(6) << ancova_result.f_covariate
              << "  " << std::setw(6) << ancova_result.p_covariate << "\n";
    std::cout << "Treatment      " << std::setw(8) << ancova_result.ss_treatment
              << "  " << std::setw(4) << ancova_result.df_treatment
              << "  " << std::setw(8) << ancova_result.ms_treatment
              << "  " << std::setw(6) << ancova_result.f_treatment
              << "  " << std::setw(6) << ancova_result.p_treatment << "\n";
    std::cout << "Error          " << std::setw(8) << ancova_result.ss_error
              << "  " << std::setw(4) << ancova_result.df_error
              << "  " << std::setw(8) << ancova_result.ms_error << "\n";
    std::cout << "------------------------------------------------------------\n";

    std::cout << "\nConclusions:\n";
    std::cout << "  Covariate (Pretest): "
              << (ancova_result.p_covariate < 0.05 ? "Significant (pretest affects posttest)" : "Not significant")
              << "\n";
    std::cout << "  Treatment (Learning Method): "
              << (ancova_result.p_treatment < 0.05 ? "Significant (difference among methods)" : "Not significant")
              << "\n";

    std::cout << "\nAdjusted means (effect of pretest removed):\n";
    for (std::size_t i = 0; i < ancova_result.adjusted_means.size(); ++i) {
        std::cout << "  " << method_labels[i] << ": " << ancova_result.adjusted_means[i] << "\n";
    }

    std::cout << "\nInterpretation: Evaluating learning method effects after controlling for pretest differences\n";
}

// ============================================================================
// Summary Output
// ============================================================================

void print_summary()
{
    print_section("Summary: Functions in anova.hpp");

    std::cout << R"(
+----------------------------------------------------------------------+
| Function                      Purpose                                |
+----------------------------------------------------------------------+
| one_way_anova()               One-way ANOVA (1 factor)               |
| two_way_anova()               Two-way ANOVA (2 factors + interaction)|
| tukey_hsd()                   Tukey HSD (all pairwise comparisons)   |
| bonferroni_posthoc()          Bonferroni (conservative multiple comp)|
| dunnett_posthoc()             Dunnett (comparison with control)      |
| scheffe_posthoc()             Scheffe (most flexible post-hoc)       |
| one_way_ancova()              One-way ANCOVA (covariate adjustment)  |
| eta_squared()                 Eta-squared (effect size)              |
| omega_squared()               Omega-squared (unbiased effect size)   |
| cohens_f()                    Cohen's f (effect size)                |
+----------------------------------------------------------------------+

[ANOVA Workflow]
  1. Check normality and homogeneity of variance
  2. Perform ANOVA (F-test)
  3. If F-test is significant -> Post-hoc to identify specific differences
  4. Report effect sizes

[Choosing Post-hoc Methods]
  +---------------------------------------------+
  | Purpose                   Recommended       |
  +---------------------------------------------+
  | All pairwise              Tukey HSD         |
  | Few comparisons           Bonferroni        |
  | Control vs treatments     Dunnett           |
  | Complex contrasts         Scheffe           |
  +---------------------------------------------+

[Effect Size Interpretation]
  Eta-squared:
    - 0.01: Small effect
    - 0.06: Medium effect
    - 0.14: Large effect

  Cohen's f:
    - 0.10: Small effect
    - 0.25: Medium effect
    - 0.40: Large effect

[Important Notes]
  - ANOVA assumptions: normality, homogeneity of variance, independence
  - Post-hoc only when ANOVA is significant
  - Interpret main effects cautiously when interaction is significant
  - Always report effect sizes
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::cout << "==========================================================\n";
    std::cout << " statcpp ANOVA Functions - Sample Code\n";
    std::cout << "==========================================================\n";

    example_one_way_anova();
    example_two_way_anova();
    example_tukey_hsd();
    example_bonferroni_posthoc();
    example_dunnett_posthoc();
    example_scheffe_posthoc();
    example_one_way_ancova();
    print_summary();

    return 0;
}
