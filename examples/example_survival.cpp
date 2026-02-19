/**
 * @file example_survival.cpp
 * @brief Sample code for survival analysis
 *
 * Demonstrates usage examples of survival analysis methods including
 * Kaplan-Meier survival curves, log-rank test, and Nelson-Aalen
 * cumulative hazard estimation.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include "statcpp/survival.hpp"

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

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Kaplan-Meier Survival Curve
    // ============================================================================
    print_section("1. Kaplan-Meier Survival Curve");

    std::cout << R"(
[Concept]
Estimation of survival function accounting for censored data
Shows how survival probability changes over time

[Example: Patient Survival Analysis]
Tracking survival time (in months) for 10 patients
-> Includes censored data
)";

    // Survival time data (months)
    std::vector<double> times = {1, 3, 4, 5, 8, 10, 12, 15, 18, 20};
    // Event occurrence (1=death, 0=censored)
    std::vector<bool> events = {1, 1, 0, 1, 1, 0, 1, 1, 0, 1};

    auto km_result = statcpp::kaplan_meier(times, events);

    print_subsection("Data Overview");
    std::cout << "  Sample size: " << times.size() << " patients\n";
    std::cout << "  Events: " << std::count(events.begin(), events.end(), 1) << "\n";
    std::cout << "  Censored: " << std::count(events.begin(), events.end(), 0) << "\n";

    print_subsection("Kaplan-Meier Survival Table");
    std::cout << "  Time   At Risk   Events   Survival    95% CI\n";

    for (std::size_t i = 0; i < km_result.times.size(); ++i) {
        std::cout << std::setw(6) << km_result.times[i]
                  << std::setw(9) << km_result.n_at_risk[i]
                  << std::setw(9) << km_result.n_events[i]
                  << std::setw(10) << km_result.survival[i];
        if (km_result.ci_lower[i] > 0) {
            std::cout << "  [" << km_result.ci_lower[i] << ", " << km_result.ci_upper[i] << "]";
        }
        std::cout << std::endl;
    }
    std::cout << "-> Survival rate decreases over time\n";

    // ============================================================================
    // 2. Median Survival Time
    // ============================================================================
    print_section("2. Median Survival Time");

    std::cout << R"(
[Concept]
The time point where the survival function equals 0.5
50% of subjects survive beyond this point

[Example: Median Survival]
Time when half of patients are still alive
-> More robust indicator than mean
)";

    double median_time = statcpp::median_survival_time(km_result);

    print_subsection("Median Survival Time");
    std::cout << "  Median: " << median_time << " months\n";
    std::cout << "\nInterpretation: 50% of subjects survive beyond " << median_time << " months\n";

    // ============================================================================
    // 3. Survival Probability at Specific Time Points
    // ============================================================================
    print_section("3. Survival Probability at Specific Time Points");

    std::cout << R"(
[Concept]
Estimate survival probability at specified time points
Evaluation at clinically important points (e.g., 6 months, 1 year, 5 years)

[Example: Survival Rates at Key Time Points]
Evaluate survival probability at 6, 12, and 18 months
)";

    std::vector<double> time_points = {6, 12, 18};

    print_subsection("Survival Probability at Each Time Point");
    for (double t : time_points) {
        // Find the last observation at or before the specified time point
        double survival_prob = 1.0;
        for (std::size_t i = 0; i < km_result.times.size(); ++i) {
            if (km_result.times[i] <= t) {
                survival_prob = km_result.survival[i];
            } else {
                break;
            }
        }
        std::cout << "  S(" << t << " months) = " << survival_prob
                  << " (" << (survival_prob * 100) << "%)" << std::endl;
    }
    std::cout << "-> Survival probability decreases over time\n";

    // ============================================================================
    // 4. Log-rank Test (Two-Group Comparison)
    // ============================================================================
    print_section("4. Log-rank Test (Comparing Two Survival Curves)");

    std::cout << R"(
[Concept]
Compare entire survival curves between two groups
Null hypothesis: Both survival curves are identical

[Example: Treatment Effect Evaluation]
Compare survival time between treatment and control groups
-> Test if treatment improves survival
)";

    // Group 1 (Treatment group)
    std::vector<double> times1 = {1, 3, 5, 8, 12, 15, 20};
    std::vector<bool> events1 = {1, 1, 1, 1, 0, 1, 0};

    // Group 2 (Control group)
    std::vector<double> times2 = {2, 4, 6, 7, 9, 10, 11};
    std::vector<bool> events2 = {1, 1, 1, 1, 1, 1, 1};

    auto logrank_result = statcpp::logrank_test(times1, events1, times2, events2);

    print_subsection("Group Information");
    std::cout << "  Treatment group:\n";
    std::cout << "    n = " << times1.size() << ", Events = "
              << std::count(events1.begin(), events1.end(), 1) << "\n";

    std::cout << "\n  Control group:\n";
    std::cout << "    n = " << times2.size() << ", Events = "
              << std::count(events2.begin(), events2.end(), 1) << "\n";

    print_subsection("Log-rank Test Results");
    std::cout << "  Chi-square statistic: " << logrank_result.statistic << "\n";
    std::cout << "  Degrees of freedom: " << logrank_result.df << "\n";
    std::cout << "  P-value: " << logrank_result.p_value << "\n";

    std::cout << "\nInterpretation (alpha = 0.05):\n";
    if (logrank_result.p_value < 0.05) {
        std::cout << "  -> Significant difference between survival curves\n";
    } else {
        std::cout << "  -> No significant difference between survival curves\n";
    }

    // ============================================================================
    // 5. Comparison of Kaplan-Meier Curves
    // ============================================================================
    print_section("5. Comparing Survival Curves");

    std::cout << R"(
[Concept]
Compare median survival times between groups
Evaluate magnitude of treatment effect

[Example: Median Survival Difference]
Quantify survival time extension from treatment
)";

    auto km1 = statcpp::kaplan_meier(times1, events1);
    auto km2 = statcpp::kaplan_meier(times2, events2);

    print_subsection("Comparison of Median Survival Times");
    std::cout << "  Treatment group median survival: " << statcpp::median_survival_time(km1) << " months\n";
    std::cout << "  Control group median survival: " << statcpp::median_survival_time(km2) << " months\n";

    double diff = statcpp::median_survival_time(km1) - statcpp::median_survival_time(km2);
    std::cout << "  Difference: " << diff << " months\n";
    if (diff > 0) {
        std::cout << "  -> Treatment group has longer survival\n";
    }

    // ============================================================================
    // 6. Nelson-Aalen Cumulative Hazard Estimator
    // ============================================================================
    print_section("6. Nelson-Aalen Cumulative Hazard Estimation");

    std::cout << R"(
[Concept]
Estimate cumulative risk (hazard) over time
Relationship with survival function: S(t) = exp(-H(t))

[Example: Cumulative Risk Evaluation]
Analyze how risk accumulates over time
)";

    auto na_result = statcpp::nelson_aalen(times, events);

    print_subsection("Cumulative Hazard Estimates");
    std::cout << "  Time    Cumulative Hazard\n";

    for (std::size_t i = 0; i < std::min(std::size_t(5), na_result.times.size()); ++i) {
        std::cout << std::setw(6) << na_result.times[i]
                  << std::setw(16) << na_result.cumulative_hazard[i] << std::endl;
    }

    std::cout << "\n-> Cumulative hazard increases over time\n";
    std::cout << "-> Higher values indicate higher risk\n";

    // ============================================================================
    // 7. Practical Example: Clinical Trial Data
    // ============================================================================
    print_section("7. Practical Example: Clinical Trial Data Analysis");

    std::cout << R"(
[Concept]
Compare survival between new drug and standard treatment
Statistically evaluate clinical utility

[Example: New Drug Efficacy Evaluation]
Survival analysis of new drug vs standard treatment groups
-> Quantitatively demonstrate treatment effect
)";

    // New drug group
    std::vector<double> new_drug_times = {6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    std::vector<bool> new_drug_events = {0, 1, 0, 1, 0, 1, 0, 0, 1, 0};

    // Standard treatment group
    std::vector<double> standard_times = {4, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    std::vector<bool> standard_events = {1, 1, 1, 1, 1, 0, 1, 1, 1, 0};

    print_subsection("Survival Times for Each Treatment Group");
    auto km_new = statcpp::kaplan_meier(new_drug_times, new_drug_events);
    std::cout << "  New drug group:\n";
    std::cout << "    Median survival: " << statcpp::median_survival_time(km_new) << " months\n";

    auto km_std = statcpp::kaplan_meier(standard_times, standard_events);
    std::cout << "\n  Standard treatment group:\n";
    std::cout << "    Median survival: " << statcpp::median_survival_time(km_std) << " months\n";

    auto trial_logrank = statcpp::logrank_test(new_drug_times, new_drug_events,
                                                standard_times, standard_events);

    print_subsection("Statistical Testing");
    std::cout << "  Log-rank test p-value: " << trial_logrank.p_value << "\n";

    if (trial_logrank.p_value < 0.05) {
        double improvement = statcpp::median_survival_time(km_new) - statcpp::median_survival_time(km_std);
        std::cout << "\nConclusion: New drug significantly improves survival\n";
        std::cout << "      Median survival improvement: " << improvement << " months\n";
    } else {
        std::cout << "\nConclusion: No significant difference between treatments\n";
    }

    // ============================================================================
    // 8. Summary: Survival Analysis Interpretation Guide
    // ============================================================================
    print_section("Summary: Survival Analysis Interpretation Guide");

    std::cout << R"(
[Key Concepts]

Survival Function S(t):
  - Probability of surviving beyond time t
  - Starts at 1.0 (100%) and decreases over time
  - Higher curve indicates better survival

Censoring:
  - Lost to follow-up (dropout)
  - Study ends before event
  - Competing risks
  -> Censored data is still informative

Median Survival Time:
  - Time point where S(t) = 0.5
  - 50% of subjects survive beyond this point
  - Robust to outliers

Log-rank Test:
  - Compares entire survival curves
  - Null hypothesis: Both curves are equal
  - Assumes proportional hazards

Cumulative Hazard:
  - Accumulated total risk
  - Relationship with survival: S(t) = exp(-H(t))
  - Useful for hazard ratio estimation

[Applications of Survival Analysis]
+------------------+------------------------------------+
| Field            | Applications                       |
+------------------+------------------------------------+
| Medical/Clinical | Treatment efficacy, patient        |
|                  | prognosis                          |
+------------------+------------------------------------+
| Engineering      | Product lifetime, reliability      |
|                  | analysis                           |
+------------------+------------------------------------+
| Business         | Customer churn, subscription       |
|                  | cancellation                       |
+------------------+------------------------------------+
| Social Sciences  | Unemployment duration, marriage    |
|                  | duration                           |
+------------------+------------------------------------+

[Analysis Steps]
1. Prepare data (survival time, event, censoring)
2. Create Kaplan-Meier curves
3. Calculate median survival time
4. Compare groups (Log-rank test)
5. Cox proportional hazards model (advanced analysis)

[Important Notes]
- Censoring should be non-informative (random)
- Verify proportional hazards assumption
- Consider sample size
- Apply multiple comparison correction
)";

    return 0;
}
