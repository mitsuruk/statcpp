/**
 * @file example_effect_size.cpp
 * @brief Effect Size Sample Code
 *
 * Demonstrates the usage of effect size indicators including Cohen's d,
 * Hedges' g, Glass's delta, Eta-squared, Omega-squared,
 * and correlation coefficient transformations.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/effect_size.hpp"

int main() {
    std::cout << "=== Effect Size Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Cohen's d (One-sample)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "1. Cohen's d (One-sample)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Effect size standardized by dividing the difference between" << std::endl;
    std::cout << "sample mean and theoretical mean by the standard deviation" << std::endl;
    std::cout << "d = (X_bar - mu) / s" << std::endl;

    std::vector<double> sample1 = {85, 90, 88, 92, 87, 89, 91, 86, 88, 90};
    double mu0 = 80.0;  // Theoretical mean

    std::cout << "\n[Example: Test Score Improvement]" << std::endl;
    std::cout << "Test scores after a new learning program (n=10)" << std::endl;
    std::cout << "Previous average score: " << mu0 << " points" << std::endl;

    // Using sample standard deviation
    double cohens_d_value = statcpp::cohens_d(sample1.begin(), sample1.end(), mu0);
    std::cout << "\n--- Effect Size Calculation ---" << std::endl;
    std::cout << "Sample mean: " << statcpp::mean(sample1.begin(), sample1.end()) << " points" << std::endl;
    std::cout << "Theoretical mean (H0): " << mu0 << " points" << std::endl;
    std::cout << "Cohen's d: " << cohens_d_value << std::endl;

    auto magnitude = statcpp::interpret_cohens_d(cohens_d_value);
    std::cout << "Effect size magnitude: ";
    switch (magnitude) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "Small"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "Medium"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "Large"; break;
    }
    std::cout << std::endl;

    std::cout << "\nInterpretation guidelines for Cohen's d:" << std::endl;
    std::cout << "  < 0.2: Negligible" << std::endl;
    std::cout << "  0.2 - 0.5: Small" << std::endl;
    std::cout << "  0.5 - 0.8: Medium" << std::endl;
    std::cout << "  >= 0.8: Large" << std::endl;

    // ============================================================================
    // 2. Cohen's d (Two-sample)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "2. Cohen's d (Two-sample)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Effect size standardized by dividing the difference between" << std::endl;
    std::cout << "two group means by the pooled standard deviation" << std::endl;
    std::cout << "d = (X_bar1 - X_bar2) / s_pooled" << std::endl;

    std::vector<double> control_group = {75, 78, 76, 80, 77, 79, 78, 76};
    std::vector<double> treatment_group = {85, 88, 90, 87, 89, 86, 88, 91};

    std::cout << "\n[Example: Drug Effectiveness]" << std::endl;
    std::cout << "Comparing control group (conventional drug) vs treatment group (new drug)" << std::endl;

    double d_two = statcpp::cohens_d_two_sample(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );

    std::cout << "\n--- Effect Size Calculation ---" << std::endl;
    std::cout << "Control group mean: "
              << statcpp::mean(control_group.begin(), control_group.end()) << " points" << std::endl;
    std::cout << "Treatment group mean: "
              << statcpp::mean(treatment_group.begin(), treatment_group.end()) << " points" << std::endl;
    std::cout << "Cohen's d (two-sample): " << d_two << std::endl;
    std::cout << "  -> Standardized magnitude of the difference between groups" << std::endl;

    std::cout << "\nEffect size evaluation: ";
    switch (statcpp::interpret_cohens_d(d_two)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "Small"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "Medium"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "Large"; break;
    }
    std::cout << std::endl;
    std::cout << "  -> The new drug is more effective than the conventional drug" << std::endl;

    // ============================================================================
    // 3. Hedges' g (Bias-corrected Cohen's d)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "3. Hedges' g (Bias-corrected Cohen's d)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Cohen's d corrected for small sample size bias" << std::endl;
    std::cout << "The smaller the sample size, the greater the correction effect" << std::endl;

    std::cout << "\n[Example: Bias Correction Comparison]" << std::endl;

    double g_one = statcpp::hedges_g(sample1.begin(), sample1.end(), mu0);
    std::cout << "\n--- One-sample case ---" << std::endl;
    std::cout << "Hedges' g: " << g_one << std::endl;

    double g_two = statcpp::hedges_g_two_sample(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );
    std::cout << "\n--- Two-sample case ---" << std::endl;
    std::cout << "Hedges' g: " << g_two << std::endl;

    std::cout << "\n--- Comparison of Cohen's d and Hedges' g ---" << std::endl;
    std::cout << "Cohen's d:  " << d_two << std::endl;
    std::cout << "Hedges' g:  " << g_two << std::endl;
    std::cout << "Difference: " << std::abs(d_two - g_two) << std::endl;
    std::cout << "  -> Hedges' g corrects for small sample bias" << std::endl;
    std::cout << "  -> With larger sample sizes, both are nearly equal" << std::endl;

    // ============================================================================
    // 4. Glass's Delta
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "4. Glass's Delta (Using Control Group Standard Deviation)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Uses only the control group's standard deviation for standardization" << std::endl;
    std::cout << "Useful when treatment may affect variance" << std::endl;

    double delta = statcpp::glass_delta(
        control_group.begin(), control_group.end(),
        treatment_group.begin(), treatment_group.end()
    );

    std::cout << "\n[Example: Treatment Effect]" << std::endl;
    std::cout << "Evaluating effect using control group's standard deviation as reference" << std::endl;

    std::cout << "\n--- Glass's Delta Calculation ---" << std::endl;
    std::cout << "Glass's Delta: " << delta << std::endl;
    std::cout << "  -> Standardized by control group's standard deviation" << std::endl;

    std::cout << "\nControl group std dev: "
              << statcpp::sample_stddev(control_group.begin(), control_group.end()) << std::endl;
    std::cout << "Treatment group std dev: "
              << statcpp::sample_stddev(treatment_group.begin(), treatment_group.end()) << std::endl;
    std::cout << "  -> When group variances differ, Glass's Delta is appropriate" << std::endl;

    // ============================================================================
    // 5. Conversion Between Correlation Coefficient and Cohen's d
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "5. Conversion Between Correlation Coefficient and Cohen's d" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Different effect size indices can be converted between each other" << std::endl;
    std::cout << "Useful for unifying different scales in meta-analysis" << std::endl;

    double r_from_d = statcpp::d_to_r(d_two);
    double d_from_r = statcpp::r_to_d(r_from_d);

    std::cout << "\n[Example: Cohen's d <-> Correlation r]" << std::endl;
    std::cout << "\n--- Round-trip Conversion ---" << std::endl;
    std::cout << "Original Cohen's d: " << d_two << std::endl;
    std::cout << "-> Converted to correlation r: " << r_from_d << std::endl;
    std::cout << "-> Converted back to Cohen's d: " << d_from_r << std::endl;
    std::cout << "  -> Confirm that round-trip conversion returns original value" << std::endl;

    // Conversion from t-value to correlation coefficient
    double t_value = 3.5;
    double df = 28.0;
    double r_from_t = statcpp::t_to_r(t_value, df);
    std::cout << "\n[Example: t-statistic to Correlation Coefficient Conversion]" << std::endl;
    std::cout << "t-value: " << t_value << std::endl;
    std::cout << "Degrees of freedom: " << df << std::endl;
    std::cout << "Correlation r: " << r_from_t << std::endl;
    std::cout << "  -> Express t-test results as correlation coefficient" << std::endl;

    // ============================================================================
    // 6. Eta-squared and Partial Eta-squared
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "6. Eta-squared and Partial Eta-squared" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Effect size indicator for Analysis of Variance (ANOVA)" << std::endl;
    std::cout << "eta^2 = SS(effect) / SS(total)" << std::endl;
    std::cout << "Indicates the proportion of total variance explained by the factor" << std::endl;

    // ANOVA example (hypothetical data)
    double ss_effect = 150.0;
    double ss_total = 500.0;
    double eta2 = statcpp::eta_squared(ss_effect, ss_total);

    std::cout << "\n[Example: ANOVA Effect Size]" << std::endl;
    std::cout << "\n--- Eta-squared Calculation ---" << std::endl;
    std::cout << "Sum of squares for effect SS(effect): " << ss_effect << std::endl;
    std::cout << "Total sum of squares SS(total): " << ss_total << std::endl;
    std::cout << "eta^2: " << eta2 << std::endl;
    std::cout << "  -> The factor explains " << (eta2 * 100) << "% of total variance" << std::endl;

    // Calculate Partial eta^2 from F-value
    double f_value = 8.5;
    double df1 = 2.0;
    double df2 = 27.0;
    double partial_eta2 = statcpp::partial_eta_squared(f_value, df1, df2);

    std::cout << "\n--- Calculating Partial eta^2 from F-value ---" << std::endl;
    std::cout << "F(" << df1 << ", " << df2 << ") = " << f_value << std::endl;
    std::cout << "Partial eta^2: " << partial_eta2 << std::endl;
    std::cout << "  -> Effect size after removing other factors" << std::endl;

    std::cout << "\nEffect size evaluation: ";
    switch (statcpp::interpret_eta_squared(partial_eta2)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "Small"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "Medium"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "Large"; break;
    }
    std::cout << std::endl;

    std::cout << "\n[Interpretation Guidelines for eta^2]" << std::endl;
    std::cout << "  < 0.01: Negligible" << std::endl;
    std::cout << "  0.01 - 0.06: Small" << std::endl;
    std::cout << "  0.06 - 0.14: Medium" << std::endl;
    std::cout << "  >= 0.14: Large" << std::endl;

    // ============================================================================
    // 7. Omega-squared (Bias-corrected Eta-squared)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "7. Omega-squared (Bias-corrected Version)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Eta-squared corrected for small sample bias" << std::endl;
    std::cout << "eta^2 tends to overestimate effects, omega^2 corrects for this" << std::endl;

    double ms_error = 12.5;
    double df_effect = 2.0;
    double omega2 = statcpp::omega_squared(ss_effect, ss_total, ms_error, df_effect);

    std::cout << "\n[Example: Bias Correction Comparison]" << std::endl;
    std::cout << "\n--- Omega-squared Calculation ---" << std::endl;
    std::cout << "Sum of squares for effect SS(effect): " << ss_effect << std::endl;
    std::cout << "Total sum of squares SS(total): " << ss_total << std::endl;
    std::cout << "Mean square error MS(error): " << ms_error << std::endl;
    std::cout << "Effect degrees of freedom df(effect): " << df_effect << std::endl;

    std::cout << "\n--- Comparison of eta^2 and omega^2 ---" << std::endl;
    std::cout << "eta^2 (Eta-squared):   " << eta2 << std::endl;
    std::cout << "omega^2 (Omega-squared): " << omega2 << std::endl;
    std::cout << "Difference:             " << (eta2 - omega2) << std::endl;
    std::cout << "\n  -> omega^2 corrects for small sample bias" << std::endl;
    std::cout << "  -> Usually omega^2 < eta^2 (more conservative estimate)" << std::endl;

    // ============================================================================
    // 8. Cohen's h (Effect Size for Proportions)
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "8. Cohen's h (Effect Size for Proportions)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Measures the difference between two proportions" << std::endl;
    std::cout << "Uses arcsine transformation" << std::endl;

    double p1 = 0.65;  // Group 1 success rate
    double p2 = 0.45;  // Group 2 success rate

    double h = statcpp::cohens_h(p1, p2);

    std::cout << "\n[Example: Conversion Rate Comparison]" << std::endl;
    std::cout << "A/B test comparing two landing pages" << std::endl;

    std::cout << "\n--- Cohen's h Calculation ---" << std::endl;
    std::cout << "Page A conversion rate: " << (p1 * 100) << "%" << std::endl;
    std::cout << "Page B conversion rate: " << (p2 * 100) << "%" << std::endl;
    std::cout << "Cohen's h: " << h << std::endl;
    std::cout << "  -> Standardized value of the difference in proportions" << std::endl;

    std::cout << "\n[Interpretation Guidelines for Cohen's h]" << std::endl;
    std::cout << "  < 0.2: Small" << std::endl;
    std::cout << "  0.2 - 0.5: Medium" << std::endl;
    std::cout << "  >= 0.5: Large" << std::endl;

    // ============================================================================
    // 9. Odds Ratio and Relative Risk
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "9. Odds Ratio and Risk Ratio (Relative Risk)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Effect size indicators for 2x2 contingency tables" << std::endl;
    std::cout << "Odds Ratio: (a/b) / (c/d) = ad/bc" << std::endl;
    std::cout << "Risk Ratio: [a/(a+b)] / [c/(c+d)]" << std::endl;

    // 2x2 contingency table example:
    //        Disease+  Disease-
    // Exposed    30        70      (a=30, b=70)
    // Control    10        90      (c=10, d=90)

    double a = 30, b = 70, c = 10, d = 90;

    std::cout << "\n[Example: Epidemiological Study]" << std::endl;
    std::cout << "Investigating the relationship between smoking and lung disease" << std::endl;

    std::cout << "\n--- 2x2 Contingency Table ---" << std::endl;
    std::cout << "              Disease+  Disease-" << std::endl;
    std::cout << "  Exposed        " << a << "        " << b << std::endl;
    std::cout << "  Unexposed      " << c << "        " << d << std::endl;

    double or_val = statcpp::odds_ratio(a, b, c, d);
    double rr_val = statcpp::risk_ratio(a, b, c, d);

    std::cout << "\n--- Effect Size Calculation ---" << std::endl;
    std::cout << "Odds Ratio: " << or_val << std::endl;
    std::cout << "  -> Exposed group's odds are " << or_val << " times the unexposed group" << std::endl;

    std::cout << "\nRisk Ratio: " << rr_val << std::endl;
    std::cout << "  -> Exposed group's risk is " << rr_val << " times the unexposed group" << std::endl;

    double risk_exposed = a / (a + b);
    double risk_control = c / (c + d);
    std::cout << "\n--- Risk Details ---" << std::endl;
    std::cout << "Exposed group risk: " << risk_exposed << " (" << (risk_exposed * 100) << "%)" << std::endl;
    std::cout << "Unexposed group risk: " << risk_control << " (" << (risk_control * 100) << "%)" << std::endl;
    std::cout << "Risk difference: " << (risk_exposed - risk_control) << std::endl;

    // ============================================================================
    // 10. Interpretation of Correlation Coefficients
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "10. Interpretation of Correlation Coefficients" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Concept]" << std::endl;
    std::cout << "Evaluate the strength of relationship by the absolute value of r" << std::endl;
    std::cout << "r^2 can be interpreted as coefficient of determination (proportion of variance explained)" << std::endl;

    std::vector<double> correlations = {0.05, 0.15, 0.35, 0.55, 0.85};

    std::cout << "\n[Example: Correlation Strength]" << std::endl;
    std::cout << "\n--- Interpretation of Various Correlation Coefficients ---" << std::endl;
    for (double r : correlations) {
        std::cout << "  r = " << std::setw(4) << r << ": ";
        switch (statcpp::interpret_correlation(r)) {
            case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
            case statcpp::effect_size_magnitude::small: std::cout << "Weak (Small)"; break;
            case statcpp::effect_size_magnitude::medium: std::cout << "Moderate (Medium)"; break;
            case statcpp::effect_size_magnitude::large: std::cout << "Strong (Large)"; break;
        }
        std::cout << " (r^2 = " << (r * r) << ", variance explained: " << (r * r * 100) << "%)" << std::endl;
    }

    std::cout << "\n[Interpretation Guidelines for Correlation]" << std::endl;
    std::cout << "  < 0.1: Negligible" << std::endl;
    std::cout << "  0.1 - 0.3: Weak (Small)" << std::endl;
    std::cout << "  0.3 - 0.5: Moderate (Medium)" << std::endl;
    std::cout << "  >= 0.5: Strong (Large)" << std::endl;

    // ============================================================================
    // 11. Comparison and Summary of Effect Sizes
    // ============================================================================
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "11. Comparison and Summary of Effect Sizes" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::cout << "\n[Summary of Two-sample Effect Sizes]" << std::endl;
    std::cout << "\n--- Various Effect Size Indicators ---" << std::endl;
    std::cout << "Cohen's d:       " << d_two << std::endl;
    std::cout << "  -> Standardized by pooled standard deviation" << std::endl;
    std::cout << "Hedges' g:       " << g_two << std::endl;
    std::cout << "  -> Bias-corrected version for small samples" << std::endl;
    std::cout << "Glass's Delta:   " << delta << std::endl;
    std::cout << "  -> Standardized by control group's standard deviation" << std::endl;
    std::cout << "Converted to r:  " << r_from_d << std::endl;
    std::cout << "  -> Expressed as strength of association" << std::endl;

    std::cout << "\n[Overall Evaluation]" << std::endl;
    std::cout << "Effect magnitude indicated by all indices: ";
    switch (statcpp::interpret_cohens_d(d_two)) {
        case statcpp::effect_size_magnitude::negligible: std::cout << "Negligible"; break;
        case statcpp::effect_size_magnitude::small: std::cout << "Small"; break;
        case statcpp::effect_size_magnitude::medium: std::cout << "Medium"; break;
        case statcpp::effect_size_magnitude::large: std::cout << "Large"; break;
    }
    std::cout << std::endl;

    std::cout << "\n[Guidelines for Using Effect Sizes]" << std::endl;
    std::cout << "1. Report effect size in addition to statistical significance" << std::endl;
    std::cout << "2. Use bias-corrected versions (Hedges' g, omega^2) for small samples" << std::endl;
    std::cout << "3. Convert to unified effect size indices for meta-analysis" << std::endl;
    std::cout << "4. Choose appropriate effect size indicator based on context" << std::endl;

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
