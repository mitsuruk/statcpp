/**
 * @file example_power_analysis.cpp
 * @brief Comprehensive sample code for power analysis
 *
 * This file demonstrates practical usage examples of power analysis
 * in statistical testing. Power analysis is an essential tool for
 * determining appropriate sample sizes in research design and
 * evaluating the power of existing data.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "statcpp/power_analysis.hpp"
#include "statcpp/effect_size.hpp"

// Helper function: Display section header
void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n\n";
}

// Helper function: Display subsection header
void print_subsection(const std::string& title) {
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '-') << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================
    // 1. Basic Concepts of Power Analysis
    // ============================================================
    print_section("1. Basic Concepts of Power Analysis");

    std::cout << "Power analysis deals with the following four elements:\n\n";
    std::cout << "1. Sample size (n): Number of observations needed\n";
    std::cout << "2. Effect size: Magnitude of the effect to detect\n";
    std::cout << "   - Cohen's d: Standardized mean difference\n";
    std::cout << "   - Small: 0.2, Medium: 0.5, Large: 0.8 (Cohen, 1988)\n";
    std::cout << "3. Significance level (alpha): Probability of Type I error (usually 0.05)\n";
    std::cout << "4. Power (1-beta): Probability of detecting a true effect (aim for 0.80+)\n\n";

    std::cout << "If three of these four elements are known, the remaining one can be calculated.\n";
    std::cout << "The most common use is determining required sample size from effect size, alpha, and power.\n";

    // ============================================================
    // 2. Power Analysis for One-Sample t-Test
    // ============================================================
    print_section("2. Power Analysis for One-Sample t-Test");

    print_subsection("2.1 Basic Power Calculation");
    std::cout << "Calculate power from known sample size and effect size.\n\n";

    std::cout << "Example: Verifying the effect of a new educational program\n";
    std::cout << "    - Existing standard score mean: mu0 = 100\n";
    std::cout << "    - Expected improved mean: mu1 = 105\n";
    std::cout << "    - Standard deviation: sigma = 15\n";
    std::cout << "    - Available sample size: n = 30\n\n";

    double cohen_d_education = (105.0 - 100.0) / 15.0;
    std::cout << "Effect size (Cohen's d) = (105 - 100) / 15 = " << cohen_d_education << "\n";
    std::cout << "This is classified as a 'small' effect size.\n\n";

    double power_30 = statcpp::power_t_test_one_sample(cohen_d_education, 30);
    std::cout << "Power with n=30: " << power_30 << "\n";
    std::cout << "-> Approximately " << (power_30 * 100) << "% probability of detecting the effect\n\n";

    std::cout << "If power is insufficient, more samples are needed:\n";
    double power_50 = statcpp::power_t_test_one_sample(cohen_d_education, 50);
    double power_100 = statcpp::power_t_test_one_sample(cohen_d_education, 100);
    std::cout << "Power with n=50:  " << power_50 << " (" << (power_50 * 100) << "%)\n";
    std::cout << "Power with n=100: " << power_100 << " (" << (power_100 * 100) << "%)\n";

    print_subsection("2.2 Required Sample Size Calculation");
    std::cout << "Calculate the sample size needed to achieve a target power.\n\n";

    std::cout << "Example: Clinical trial design for a new drug\n";
    std::cout << "    - Expected effect size: d = 0.5 (medium)\n";
    std::cout << "    - Target power: 80%\n";
    std::cout << "    - Significance level: alpha = 0.05\n\n";

    std::size_t n_required = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05);
    std::cout << "Required sample size: n = " << n_required << "\n\n";

    std::cout << "Required sample sizes at different power levels:\n";
    std::size_t n_70 = statcpp::sample_size_t_test_one_sample(0.5, 0.70);
    std::size_t n_80 = statcpp::sample_size_t_test_one_sample(0.5, 0.80);
    std::size_t n_90 = statcpp::sample_size_t_test_one_sample(0.5, 0.90);
    std::cout << "Power 70%: n = " << n_70 << "\n";
    std::cout << "Power 80%: n = " << n_80 << "\n";
    std::cout << "Power 90%: n = " << n_90 << "\n";
    std::cout << "-> Higher power requires more samples\n";

    print_subsection("2.3 Comparison of One-Sided and Two-Sided Tests");
    std::cout << "One-sided tests are used when the direction of effect is known in advance.\n";
    std::cout << "Fewer samples are needed to achieve the same power.\n\n";

    std::cout << "For effect size d = 0.5, power 80%:\n";
    std::size_t n_two = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "two.sided");
    std::size_t n_greater = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "greater");
    std::size_t n_less = statcpp::sample_size_t_test_one_sample(0.5, 0.80, 0.05, "less");

    std::cout << "Two-sided test: n = " << n_two << "\n";
    std::cout << "One-sided test (greater): n = " << n_greater << "\n";
    std::cout << "One-sided test (less): n = " << n_less << "\n\n";

    std::cout << "Note: Only use one-sided tests when the direction of effect can be predicted in advance.\n";
    std::cout << "      When direction is unknown, two-sided tests are safer.\n";

    // ============================================================
    // 3. Power Analysis for Two-Sample t-Test
    // ============================================================
    print_section("3. Power Analysis for Two-Sample t-Test");

    print_subsection("3.1 Sample Size Design for A/B Testing");
    std::cout << "Verifying whether new website design (B) is better than existing design (A)\n\n";

    std::cout << "Background information:\n";
    std::cout << "  - Current conversion rate: 10%\n";
    std::cout << "  - Desired improvement: 20% relative increase (10% -> 12%)\n";
    std::cout << "  - However, assuming effect size in continuous values: d = 0.3\n\n";

    std::size_t n_ab = statcpp::sample_size_t_test_two_sample(0.3, 0.80, 0.05);
    std::cout << "Required sample size per group: n = " << n_ab << "\n";
    std::cout << "Total sample size: " << (n_ab * 2) << " people\n\n";

    std::cout << "Confirming power in actual test:\n";
    double power_ab = statcpp::power_t_test_two_sample(0.3, n_ab, n_ab);
    std::cout << "Power with n1 = n2 = " << n_ab << ": " << power_ab << "\n";

    print_subsection("3.2 Unequal Sample Sizes");
    std::cout << "There may be cases where experimental and control group sample sizes differ.\n";
    std::cout << "Example: Control group data is abundant, but experimental group is costly\n\n";

    std::cout << "When allocating samples at 1:2 ratio (experimental:control = 1:2):\n";
    std::size_t n_unequal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 2.0);
    std::cout << "Experimental group sample size: n1 = " << n_unequal << "\n";
    std::cout << "Control group sample size: n2 = " << (n_unequal * 2) << "\n\n";

    std::cout << "Comparison of equal (1:1) and unequal (1:2) allocation:\n";
    std::size_t n_equal = statcpp::sample_size_t_test_two_sample(0.5, 0.80, 0.05, 1.0);
    std::cout << "Equal (1:1): n = " << n_equal << " per group, total = " << (n_equal * 2) << "\n";
    std::cout << "Unequal (1:2): n1 = " << n_unequal << ", n2 = " << (n_unequal * 2)
              << ", total = " << (n_unequal * 3) << "\n\n";
    std::cout << "-> Unequal allocation increases total sample size, so equal allocation is more efficient when possible.\n";

    print_subsection("3.3 Estimating Effect Size from Actual Data");
    std::cout << "Estimate effect size from existing pilot study data and\n";
    std::cout << "design sample size for the main study.\n\n";

    std::vector<double> pilot_control = {23.1, 25.3, 22.8, 24.5, 26.1, 23.9, 25.7, 24.2};
    std::vector<double> pilot_treatment = {26.4, 28.2, 27.1, 29.3, 26.8, 28.9, 27.5, 28.0};

    std::cout << "Pilot study data:\n";
    std::cout << "Control (n=8): ";
    for (double x : pilot_control) std::cout << x << " ";
    std::cout << "\nTreatment (n=8): ";
    for (double x : pilot_treatment) std::cout << x << " ";
    std::cout << "\n\n";

    double d_pilot = statcpp::cohens_d_two_sample(pilot_treatment.begin(), pilot_treatment.end(),
                                                  pilot_control.begin(), pilot_control.end());
    std::cout << "Estimated effect size (Cohen's d): " << d_pilot << "\n";
    std::cout << "This corresponds to a 'large' effect size (d > 0.8).\n\n";

    std::cout << "Required sample size for main study (power 80%):\n";
    std::size_t n_main = statcpp::sample_size_t_test_two_sample(d_pilot, 0.80);
    std::cout << "Sample size per group: n = " << n_main << "\n\n";

    std::cout << "Note: Effect size estimates from pilot studies can be unstable.\n";
    std::cout << "      Conservative estimation (setting effect size smaller) is recommended.\n";

    // ============================================================
    // 4. Power Analysis for Proportion Tests
    // ============================================================
    print_section("4. Power Analysis for Proportion Tests");

    print_subsection("4.1 Comparing Treatment Success Rates in Clinical Trials");
    std::cout << "Comparing success rates of new treatment (experimental) vs standard treatment (control).\n\n";

    std::cout << "Study design:\n";
    std::cout << "  - Standard treatment success rate: p1 = 0.60 (60%)\n";
    std::cout << "  - Expected new treatment success rate: p2 = 0.75 (75%)\n";
    std::cout << "  - Difference to detect: 15 percentage points\n";
    std::cout << "  - Target power: 80%\n";
    std::cout << "  - Significance level: alpha = 0.05\n\n";

    std::size_t n_clinical = statcpp::sample_size_prop_test(0.60, 0.75, 0.80, 0.05);
    std::cout << "Required sample size per group: n = " << n_clinical << "\n";
    std::cout << "Total sample size: " << (n_clinical * 2) << " people\n\n";

    std::cout << "Confirming actual power:\n";
    double power_clinical = statcpp::power_prop_test(0.60, 0.75, n_clinical);
    std::cout << "Calculated power: " << power_clinical << " (" << (power_clinical * 100) << "%)\n";

    print_subsection("4.2 Sample Sizes for Different Effect Sizes");
    std::cout << "Detecting improvement from baseline success rate p1 = 0.50\n\n";

    std::cout << "Magnitude of improvement to detect and sample size:\n";
    double base_p = 0.50;
    std::vector<double> improvements = {0.05, 0.10, 0.15, 0.20};

    std::cout << std::setw(20) << "Improvement"
              << std::setw(15) << "p2"
              << std::setw(20) << "Required n (per group)" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (double imp : improvements) {
        double p2 = base_p + imp;
        std::size_t n = statcpp::sample_size_prop_test(base_p, p2, 0.80, 0.05);
        std::cout << std::setw(15) << (imp * 100) << "%"
                  << std::setw(15) << p2
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\n-> Smaller effects require more samples to detect.\n";

    print_subsection("4.3 When Baseline Rate is Extreme");
    std::cout << "When baseline rate (p1) is extreme (close to 0 or 1),\n";
    std::cout << "required sample size changes even for the same absolute difference.\n\n";

    std::cout << "Detecting a 10 percentage point improvement:\n\n";

    std::vector<std::pair<double, double>> scenarios = {
        {0.10, 0.20},  // Low baseline
        {0.30, 0.40},  // Medium-low baseline
        {0.50, 0.60},  // Medium baseline
        {0.70, 0.80},  // Medium-high baseline
        {0.85, 0.95}   // High baseline
    };

    std::cout << std::setw(15) << "p1"
              << std::setw(15) << "p2"
              << std::setw(20) << "Required n" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& scenario : scenarios) {
        std::size_t n = statcpp::sample_size_prop_test(scenario.first, scenario.second, 0.80, 0.05);
        std::cout << std::setw(15) << scenario.first
                  << std::setw(15) << scenario.second
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\n-> Most samples are needed when baseline rate is near 0.5.\n";
    std::cout << "   Extreme rates (close to 0 or 1) require relatively fewer samples.\n";

    // ============================================================
    // 5. Practical Examples of Sample Size Design
    // ============================================================
    print_section("5. Practical Examples of Sample Size Design");

    print_subsection("5.1 Psychology Research: Verifying Cognitive Training Program Effect");
    std::cout << "Research objective: Verify whether cognitive training program improves working memory\n\n";

    std::cout << "Information from prior research:\n";
    std::cout << "  - Effect sizes reported in similar intervention studies: d = 0.40 ~ 0.60\n";
    std::cout << "  - Conservatively assume d = 0.45\n\n";

    std::cout << "Research design options:\n\n";

    std::cout << "Option 1: Two-sided test, power 80%\n";
    std::size_t n_psych_80 = statcpp::sample_size_t_test_two_sample(0.45, 0.80);
    std::cout << "  Sample size per group: " << n_psych_80 << "\n";
    std::cout << "  Total sample size: " << (n_psych_80 * 2) << "\n\n";

    std::cout << "Option 2: Two-sided test, power 90% (more conservative)\n";
    std::size_t n_psych_90 = statcpp::sample_size_t_test_two_sample(0.45, 0.90);
    std::cout << "  Sample size per group: " << n_psych_90 << "\n";
    std::cout << "  Total sample size: " << (n_psych_90 * 2) << "\n\n";

    std::cout << "Option 3: One-sided test (detect improvement only), power 80%\n";
    std::size_t n_psych_one = statcpp::sample_size_t_test_two_sample(0.45, 0.80, 0.05, 1.0, "greater");
    std::cout << "  Sample size per group: " << n_psych_one << "\n";
    std::cout << "  Total sample size: " << (n_psych_one * 2) << "\n\n";

    std::cout << "Recommendation: Option 1 (two-sided, 80%) is the standard choice.\n";
    std::cout << "      Option 2 is worth considering if resources allow.\n";

    print_subsection("5.2 Medical Research: Verifying New Drug Efficacy");
    std::cout << "Research objective: Verify whether new drug lowers blood pressure (placebo-controlled trial)\n\n";

    std::cout << "Setting clinically meaningful effect:\n";
    std::cout << "  - Systolic blood pressure reduction: 5 mmHg\n";
    std::cout << "  - Population standard deviation: 12 mmHg\n";
    std::cout << "  - Effect size: d = 5/12 = " << (5.0/12.0) << "\n\n";

    double d_medical = 5.0 / 12.0;
    std::size_t n_medical = statcpp::sample_size_t_test_two_sample(d_medical, 0.80, 0.05);
    std::cout << "Required sample size (per group): " << n_medical << "\n";
    std::cout << "Total sample size: " << (n_medical * 2) << "\n\n";

    std::cout << "Adjustment for dropout rate:\n";
    double dropout_rate = 0.15;  // Assume 15% dropout
    std::size_t n_adjusted = static_cast<std::size_t>(std::ceil(n_medical / (1.0 - dropout_rate)));
    std::cout << "  Dropout rate: " << (dropout_rate * 100) << "%\n";
    std::cout << "  Adjusted sample size (per group): " << n_adjusted << "\n";
    std::cout << "  Adjusted total sample size: " << (n_adjusted * 2) << "\n\n";

    std::cout << "Important: In clinical trials, it's common to recruit more than\n";
    std::cout << "        the calculated sample size to account for dropouts and non-compliance.\n";

    print_subsection("5.3 Marketing Research: Measuring Email Campaign Effectiveness");
    std::cout << "Research objective: Verify whether new email design improves click rate\n\n";

    std::cout << "Current analysis:\n";
    std::cout << "  - Current click rate: 3%\n";
    std::cout << "  - Target click rate: 4% (33% relative improvement)\n";
    std::cout << "  - Significance level: alpha = 0.05\n";
    std::cout << "  - Target power: 80%\n\n";

    std::size_t n_marketing = statcpp::sample_size_prop_test(0.03, 0.04, 0.80, 0.05);
    std::cout << "Required sends per version: " << n_marketing << "\n";
    std::cout << "Total sends: " << (n_marketing * 2) << "\n\n";

    std::cout << "Recalculation with more realistic target:\n";
    std::cout << "  Target click rate: 3.5% (17% relative improvement)\n";
    std::size_t n_realistic = statcpp::sample_size_prop_test(0.03, 0.035, 0.80, 0.05);
    std::cout << "  Required sends per version: " << n_realistic << "\n";
    std::cout << "  Total sends: " << (n_realistic * 2) << "\n\n";

    std::cout << "Insight: Detecting small improvements from a low baseline rate\n";
    std::cout << "      requires very large sample sizes.\n";

    // ============================================================
    // 6. Relationship Between Effect Size and Power
    // ============================================================
    print_section("6. Relationship Between Effect Size and Power");

    print_subsection("6.1 Effect Size Magnitude and Sample Size");
    std::cout << "Effect size classification based on Cohen's criteria and required sample sizes:\n\n";

    std::vector<std::pair<std::string, double>> effect_sizes = {
        {"Small", 0.2},
        {"Medium", 0.5},
        {"Large", 0.8}
    };

    std::cout << std::setw(20) << "Effect Size"
              << std::setw(10) << "d"
              << std::setw(20) << "Sample (n, 80%)"
              << std::setw(20) << "Sample (n, 90%)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& es : effect_sizes) {
        std::size_t n_80 = statcpp::sample_size_t_test_two_sample(es.second, 0.80);
        std::size_t n_90 = statcpp::sample_size_t_test_two_sample(es.second, 0.90);
        std::cout << std::setw(20) << es.first
                  << std::setw(10) << es.second
                  << std::setw(20) << n_80
                  << std::setw(20) << n_90 << "\n";
    }

    std::cout << "\nKey insights:\n";
    std::cout << "  - When effect size is halved, required sample size roughly quadruples\n";
    std::cout << "  - Detecting small effects requires very large samples\n";
    std::cout << "  - Minimum detectable effect size should be carefully set during research design\n";

    print_subsection("6.2 Power Curve for Fixed Sample Size");
    std::cout << "Change in power by effect size when n=50 (per group):\n\n";

    std::size_t fixed_n = 50;
    std::vector<double> d_values = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << std::setw(15) << "Effect size (d)"
              << std::setw(15) << "Power"
              << "  " << "Visualization" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (double d : d_values) {
        double power = statcpp::power_t_test_two_sample(d, fixed_n, fixed_n);
        int bars = static_cast<int>(power * 40);  // Scale adjustment
        std::cout << std::setw(15) << d
                  << std::setw(15) << power
                  << "  " << std::string(bars, '#') << "\n";
    }

    std::cout << "\n-> Power increases rapidly as effect size increases.\n";
    std::cout << "   With n=50, d=0.5 (medium effect) yields about 70% power.\n";

    // ============================================================
    // 7. Trade-offs Between alpha, beta, n, and Effect Size
    // ============================================================
    print_section("7. Trade-offs Between alpha, beta, n, and Effect Size");

    print_subsection("7.1 Impact of Significance Level (alpha)");
    std::cout << "Setting a stricter significance level (smaller alpha) requires more samples.\n\n";

    double effect = 0.5;
    double power_target = 0.80;
    std::vector<double> alpha_levels = {0.10, 0.05, 0.01, 0.001};

    std::cout << std::setw(15) << "Alpha"
              << std::setw(20) << "Required n (per group)"
              << std::setw(15) << "Increase" << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::size_t baseline_n = 0;
    for (size_t i = 0; i < alpha_levels.size(); ++i) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(effect, power_target, alpha_levels[i]);
        if (i == 0) baseline_n = n;
        double increase = (static_cast<double>(n) / baseline_n - 1.0) * 100;

        std::cout << std::setw(15) << alpha_levels[i]
                  << std::setw(20) << n
                  << std::setw(14) << (i == 0 ? "-" : "+" + std::to_string(static_cast<int>(increase)) + "%") << "\n";
    }

    std::cout << "\nConsiderations:\n";
    std::cout << "  - Consider stricter alpha (e.g., 0.01) for multiple comparisons\n";
    std::cout << "  - alpha = 0.10 may be acceptable for exploratory research\n";
    std::cout << "  - However, alpha = 0.05 is the most common standard\n";

    print_subsection("7.2 Choosing Power (1-beta)");
    std::cout << "Comparing the impact of changing target power.\n\n";

    std::vector<double> power_levels = {0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

    std::cout << std::setw(12) << "Power"
              << std::setw(10) << "beta"
              << std::setw(20) << "Required n (per group)"
              << std::setw(15) << "Increase" << "\n";
    std::cout << std::string(57, '-') << "\n";

    std::size_t n_baseline = 0;
    for (size_t i = 0; i < power_levels.size(); ++i) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(0.5, power_levels[i]);
        if (i == 2) n_baseline = n;  // 80% as baseline
        double increase = (static_cast<double>(n) / n_baseline - 1.0) * 100;
        double beta = 1.0 - power_levels[i];

        std::cout << std::setw(12) << power_levels[i]
                  << std::setw(10) << beta
                  << std::setw(20) << n
                  << std::setw(14) << (i == 2 ? "-" : (increase >= 0 ? "+" : "") + std::to_string(static_cast<int>(increase)) + "%") << "\n";
    }

    std::cout << "\nGuidelines:\n";
    std::cout << "  - 80% (beta=0.20): Most common standard\n";
    std::cout << "  - 90% (beta=0.10): Recommended for important decision-making research\n";
    std::cout << "  - 95% (beta=0.05): Very conservative, used in clinical trials\n";

    print_subsection("7.3 Dealing with Effect Size Uncertainty");
    std::cout << "Effect size estimates come with uncertainty.\n";
    std::cout << "A conservative approach assumes smaller effect sizes than expected.\n\n";

    std::cout << "Example: When prior research reports d = 0.6\n\n";

    std::vector<std::pair<std::string, double>> scenarios_es = {
        {"Optimistic (reported)", 0.60},
        {"Realistic (80%)", 0.48},
        {"Conservative (70%)", 0.42},
        {"Very conservative (60%)", 0.36}
    };

    std::cout << std::setw(25) << "Scenario"
              << std::setw(15) << "Assumed d"
              << std::setw(20) << "Required n" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& scenario : scenarios_es) {
        std::size_t n = statcpp::sample_size_t_test_two_sample(scenario.second, 0.80);
        std::cout << std::setw(25) << scenario.first
                  << std::setw(15) << scenario.second
                  << std::setw(20) << n << "\n";
    }

    std::cout << "\nRecommended approach:\n";
    std::cout << "  1. Don't over-rely on prior research effect sizes (publication bias possible)\n";
    std::cout << "  2. If multiple prior studies exist, use mean or lower bound\n";
    std::cout << "  3. When estimating from pilot study, use lower bound of confidence interval\n";
    std::cout << "  4. When uncertainty is high, assume conservative (smaller) effect size\n";

    // ============================================================
    // 8. Practical Guidelines and Best Practices
    // ============================================================
    print_section("8. Practical Guidelines and Best Practices");

    print_subsection("8.1 Research Design Checklist");
    std::cout << "Standard procedure when conducting power analysis:\n\n";

    std::cout << "[ ] Step 1: Clarify research hypothesis\n";
    std::cout << "    - Explicitly state null and alternative hypotheses\n";
    std::cout << "    - Decide between one-sided and two-sided tests\n\n";

    std::cout << "[ ] Step 2: Set effect size\n";
    std::cout << "    - Gather information from prior research\n";
    std::cout << "    - Identify minimum clinically/practically meaningful effect size\n";
    std::cout << "    - Make conservative estimates\n\n";

    std::cout << "[ ] Step 3: Determine statistical parameters\n";
    std::cout << "    - Significance level alpha (usually 0.05)\n";
    std::cout << "    - Target power (usually 0.80 or higher)\n\n";

    std::cout << "[ ] Step 4: Calculate sample size\n";
    std::cout << "    - Select appropriate test method\n";
    std::cout << "    - Perform power analysis\n\n";

    std::cout << "[ ] Step 5: Evaluate feasibility\n";
    std::cout << "    - Check against resources (budget, time, personnel)\n";
    std::cout << "    - Adjust parameters as needed\n\n";

    std::cout << "[ ] Step 6: Account for dropouts\n";
    std::cout << "    - Adjust for expected dropout rate\n";
    std::cout << "    - Set final recruitment target\n\n";

    std::cout << "[ ] Step 7: Document\n";
    std::cout << "    - Record all assumptions and calculations in study protocol\n";
    std::cout << "    - Consider pre-registration\n";

    print_subsection("8.2 Strategies When Sample Size is Constrained");
    std::cout << "Strategies when ideal sample size cannot be secured:\n\n";

    std::cout << "Strategy 1: Report achievable power\n";
    std::size_t available_n = 30;
    std::vector<double> detectable_effects = {0.3, 0.5, 0.7, 0.9};

    std::cout << "  Available sample: n = " << available_n << " per group\n";
    std::cout << "  Detectable effect sizes (80% power):\n\n";

    for (double d : detectable_effects) {
        double achieved_power = statcpp::power_t_test_two_sample(d, available_n, available_n);
        std::cout << "    d = " << d << ": power = " << achieved_power
                  << " (" << (achieved_power * 100) << "%)"
                  << (achieved_power >= 0.80 ? " OK" : "") << "\n";
    }

    std::cout << "\nStrategy 2: Increase effect size\n";
    std::cout << "  - Design stronger interventions\n";
    std::cout << "  - Improve measurement precision (reduce variance)\n";
    std::cout << "  - Target more homogeneous participant groups\n\n";

    std::cout << "Strategy 3: Adopt more efficient designs\n";
    std::cout << "  - Consider within-subjects design (paired)\n";
    std::cout << "  - Adjusted analysis using covariates\n";
    std::cout << "  - Adopt adaptive designs\n\n";

    std::cout << "Strategy 4: Honest reporting\n";
    std::cout << "  - Report achieved power\n";
    std::cout << "  - Specify range of undetectable effects\n";
    std::cout << "  - Note limitations in interpreting results\n";

    print_subsection("8.3 General Recommendations");
    std::cout << "Standard parameter settings for various research fields:\n\n";

    std::cout << "* Basic/Exploratory Research:\n";
    std::cout << "    alpha = 0.05 (two-sided), power = 0.80\n";
    std::cout << "    Effect size: Medium (d = 0.5) or based on prior research\n\n";

    std::cout << "* Clinical Trials (Confirmatory Research):\n";
    std::cout << "    alpha = 0.05 (two-sided), power = 0.90\n";
    std::cout << "    Effect size: Based on minimum clinically meaningful difference\n\n";

    std::cout << "* Superiority Trial:\n";
    std::cout << "    alpha = 0.05 (two-sided), power = 0.80-0.90\n";
    std::cout << "    Effect size: Based on expected effect\n\n";

    std::cout << "* Non-inferiority Trial:\n";
    std::cout << "    alpha = 0.025 (one-sided), power = 0.80-0.90\n";
    std::cout << "    Effect size: Based on non-inferiority margin\n\n";

    std::cout << "* Pilot Study:\n";
    std::cout << "    alpha = 0.10, power = 0.70-0.80\n";
    std::cout << "    Effect size: Large (d = 0.8) detectable level\n";

    // ============================================================
    // 9. Common Mistakes and Correct Interpretations
    // ============================================================
    print_section("9. Common Mistakes and Correct Interpretations");

    print_subsection("9.1 Mistake: Post-hoc Power Analysis");
    std::cout << "X Incorrect approach:\n";
    std::cout << "   'After conducting research, calculate power using\n";
    std::cout << "    observed effect size for non-significant results'\n\n";

    std::cout << "Problems:\n";
    std::cout << "  - Post-hoc power is perfectly correlated with p-value from significance test\n";
    std::cout << "  - Provides no additional information\n";
    std::cout << "  - Differs from original purpose of power (research planning)\n\n";

    std::cout << "O Correct approach:\n";
    std::cout << "   Report confidence intervals showing effect size estimate and uncertainty\n";
    std::cout << "   Example: 'Mean difference = 2.3, 95% CI [-0.5, 5.1]'\n";

    print_subsection("9.2 Mistake: Misunderstanding Power");
    std::cout << "X Common misunderstanding:\n";
    std::cout << "   '80% power means there's an 80% probability the effect exists'\n\n";

    std::cout << "O Correct understanding:\n";
    std::cout << "   What 80% power means:\n";
    std::cout << "   'If a true effect exists, the probability of detecting it statistically significantly is 80%'\n\n";

    std::cout << "Important distinction:\n";
    std::cout << "  - Power is a conditional probability assuming true effect exists\n";
    std::cout << "  - Different concept from probability of effect existence\n";
    std::cout << "  - Bayesian statistics can estimate posterior probability\n";

    print_subsection("9.3 Mistake: Over-relying on Effect Size Estimates");
    std::cout << "X Common mistake:\n";
    std::cout << "   'Pilot study showed d = 0.8, so I'll use that for main study'\n\n";

    std::cout << "Illustration of the problem:\n";
    std::vector<double> pilot_effect_estimates = {0.8, 0.7, 0.9, 0.75, 0.85};
    std::cout << "  Hypothetical effect size estimates from 5 small studies:\n";
    std::cout << "  ";
    for (double e : pilot_effect_estimates) std::cout << e << " ";
    double mean_effect = 0.0;
    for (double e : pilot_effect_estimates) mean_effect += e;
    mean_effect /= pilot_effect_estimates.size();
    std::cout << "\n  Mean: " << mean_effect << "\n\n";

    std::cout << "  These estimates vary around the true effect size (e.g., 0.5).\n";
    std::cout << "  Small samples tend to produce large values by chance (Winner's curse)\n\n";

    std::cout << "O Correct approach:\n";
    std::cout << "  1. If multiple studies exist, use meta-analysis results\n";
    std::cout << "  2. Use lower bound of confidence interval\n";
    std::cout << "  3. Adopt conservative estimate (70-80% of reported value)\n";
    std::cout << "  4. Base on minimum clinically/practically meaningful effect size\n";

    print_subsection("9.4 Mistake: Linear Thinking About Sample Size");
    std::cout << "X Common mistake:\n";
    std::cout << "   'If effect size is halved, just double the sample size'\n\n";

    std::cout << "Actual relationship:\n";
    double d_base = 0.4;
    std::size_t n_base = statcpp::sample_size_t_test_two_sample(d_base, 0.80);
    double d_half = d_base / 2.0;
    std::size_t n_half = statcpp::sample_size_t_test_two_sample(d_half, 0.80);

    std::cout << "  Effect size d = " << d_base << ": n = " << n_base << "\n";
    std::cout << "  Effect size d = " << d_half << ": n = " << n_half << "\n";
    std::cout << "  Increase factor: " << (static_cast<double>(n_half) / n_base) << " times\n\n";

    std::cout << "O Correct understanding:\n";
    std::cout << "   Required sample size is inversely proportional to effect size squared.\n";
    std::cout << "   When effect size is halved, sample size increases roughly 4 times.\n";

    print_subsection("9.5 Mistake: Confusing alpha and beta");
    std::cout << "Important distinction:\n\n";

    std::cout << "alpha (Type I error rate):\n";
    std::cout << "  - Probability of concluding 'effect exists' when it actually doesn't\n";
    std::cout << "  - Directly set by researcher (usually 0.05)\n";
    std::cout << "  - Threshold compared against p-value\n\n";

    std::cout << "beta (Type II error rate):\n";
    std::cout << "  - Probability of 'failing to detect' when effect actually exists\n";
    std::cout << "  - Power = 1 - beta\n";
    std::cout << "  - Depends on sample size, effect size, and alpha\n\n";

    std::cout << "Balance example:\n";
    std::cout << "  Standard setting: alpha = 0.05, beta = 0.20 (power = 0.80)\n";
    std::cout << "  -> Type I error is prioritized over Type II error (1:4 ratio)\n";

    // ============================================================
    // 10. Summary and Application to Practice
    // ============================================================
    print_section("10. Summary and Application to Practice");

    std::cout << "* Essential Purpose of Power Analysis:\n\n";
    std::cout << "1. Optimal allocation of research resources\n";
    std::cout << "   - Avoid meaningless results from insufficient samples\n";
    std::cout << "   - Don't waste resources on excessive samples\n\n";

    std::cout << "2. Improve transparency and reproducibility\n";
    std::cout << "   - Pre-registration of research plan\n";
    std::cout << "   - Justification of hypothesis and sample size\n\n";

    std::cout << "3. Ethical research practice\n";
    std::cout << "   - Minimize burden on participants\n";
    std::cout << "   - Ensure scientifically valuable research\n\n";

    std::cout << "* Recommended Workflow for Implementation:\n\n";

    std::cout << "Phase 1: Planning\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "  | 1. Clarify research hypothesis    |\n";
    std::cout << "  | 2. Estimate effect size (conserv.)|\n";
    std::cout << "  | 3. Conduct power analysis         |\n";
    std::cout << "  | 4. Evaluate feasibility           |\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "           |\n";
    std::cout << "Phase 2: Documentation\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "  | 5. Document in study protocol     |\n";
    std::cout << "  | 6. Pre-registration (recommended) |\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "           |\n";
    std::cout << "Phase 3: Execution\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "  | 7. Collect data as planned        |\n";
    std::cout << "  | 8. Record any deviations          |\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "           |\n";
    std::cout << "Phase 4: Reporting\n";
    std::cout << "  +-----------------------------------+\n";
    std::cout << "  | 9. Report effect size and CI      |\n";
    std::cout << "  | 10. Explain any deviations        |\n";
    std::cout << "  +-----------------------------------+\n\n";

    std::cout << "* Final Checklist:\n\n";
    std::cout << "Items to confirm before starting research:\n";
    std::cout << "  [x] Is effect size setting rationale clear?\n";
    std::cout << "  [x] Is sample size calculation documented?\n";
    std::cout << "  [x] Is dropout rate accounted for?\n";
    std::cout << "  [x] Is statistical test selection appropriate?\n";
    std::cout << "  [x] Is one-sided/two-sided choice justified?\n";
    std::cout << "  [x] Is target sample size achievable?\n";
    std::cout << "  [x] Is there an alternative plan (if sample insufficient)?\n\n";

    std::cout << "* For Further Learning:\n\n";
    std::cout << "Recommended literature:\n";
    std::cout << "  - Cohen, J. (1988). Statistical Power Analysis for the\n";
    std::cout << "    Behavioral Sciences (2nd ed.)\n";
    std::cout << "  - Faul et al. (2007). G*Power 3: A flexible statistical\n";
    std::cout << "    power analysis program. Behavior Research Methods.\n\n";

    std::cout << "Online resources:\n";
    std::cout << "  - G*Power (free software)\n";
    std::cout << "  - R packages: pwr, WebPower\n";
    std::cout << "  - Python: statsmodels.stats.power\n\n";

    std::cout << "=====================================\n";
    std::cout << "Power analysis demonstration program complete\n";
    std::cout << "=====================================\n";

    return 0;
}
