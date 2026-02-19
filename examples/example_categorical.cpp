/**
 * @file example_categorical.cpp
 * @brief Sample code for categorical data analysis
 *
 * Demonstrates usage of contingency tables (cross-tabulation), odds ratio,
 * relative risk, risk difference, number needed to treat (NNT), and other
 * categorical data analysis methods.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/categorical.hpp"

int main() {
    std::cout << "=== Categorical Data Analysis Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Contingency Table (Cross-tabulation)
    // ============================================================================
    std::cout << "\n1. Creating a Contingency Table (Cross-tabulation)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Example: Gender (0=Male, 1=Female) and Product Preference (0=Product A, 1=Product B, 2=Product C)
    std::vector<std::size_t> gender = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                        0, 0, 1, 1, 0, 1, 0, 1, 0, 1};
    std::vector<std::size_t> product = {0, 0, 1, 1, 2, 0, 1, 1, 2, 2,
                                         0, 1, 0, 1, 2, 2, 1, 0, 2, 1};

    auto ct = statcpp::contingency_table(gender, product);

    std::cout << "Gender vs Product Preference:" << std::endl;
    std::cout << "           Product A  Product B  Product C  Total" << std::endl;

    const char* row_names[] = {"Male  ", "Female"};
    for (std::size_t i = 0; i < ct.n_rows; ++i) {
        std::cout << row_names[i] << "     ";
        for (std::size_t j = 0; j < ct.n_cols; ++j) {
            std::cout << std::setw(9) << ct.table[i][j] << "  ";
        }
        std::cout << std::setw(5) << ct.row_totals[i] << std::endl;
    }

    std::cout << "Total        ";
    for (std::size_t j = 0; j < ct.n_cols; ++j) {
        std::cout << std::setw(9) << ct.col_totals[j] << "  ";
    }
    std::cout << std::setw(5) << ct.total << std::endl;

    // ============================================================================
    // 2. Odds Ratio
    // ============================================================================
    std::cout << "\n2. Odds Ratio Analysis" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Example 2x2 contingency table: Treatment efficacy study
    //              Success  Failure
    // Treatment       50       10
    // Control         30       20

    std::vector<std::vector<std::size_t>> treatment_table = {
        {50, 10},  // Treatment: 50 successes, 10 failures
        {30, 20}   // Control:   30 successes, 20 failures
    };

    auto or_result = statcpp::odds_ratio(treatment_table);

    std::cout << "Treatment Efficacy Study:" << std::endl;
    std::cout << "             Success  Failure" << std::endl;
    std::cout << "Treatment        50       10" << std::endl;
    std::cout << "Control          30       20" << std::endl;

    std::cout << "\nOdds Ratio Results:" << std::endl;
    std::cout << "  Odds Ratio: " << or_result.odds_ratio << std::endl;
    std::cout << "  Log Odds Ratio: " << or_result.log_odds_ratio << std::endl;
    std::cout << "  SE(log OR): " << or_result.se_log_odds_ratio << std::endl;
    std::cout << "  95% CI: [" << or_result.ci_lower << ", " << or_result.ci_upper << "]" << std::endl;

    std::cout << "\nInterpretation:" << std::endl;
    if (or_result.odds_ratio > 1.0) {
        std::cout << "  Treatment group has " << or_result.odds_ratio
                  << " times higher odds of success than control" << std::endl;
    } else if (or_result.odds_ratio < 1.0) {
        std::cout << "  Treatment group has lower odds of success than control" << std::endl;
    } else {
        std::cout << "  No difference in odds between groups" << std::endl;
    }

    // Method with direct cell values
    auto or_direct = statcpp::odds_ratio(50, 10, 30, 20);
    std::cout << "  (Verification: OR = " << or_direct.odds_ratio << ")" << std::endl;

    // ============================================================================
    // 3. Relative Risk (Risk Ratio)
    // ============================================================================
    std::cout << "\n3. Relative Risk (Risk Ratio)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto rr_result = statcpp::relative_risk(treatment_table);

    std::cout << "Relative Risk Results:" << std::endl;
    std::cout << "  Relative Risk: " << rr_result.relative_risk << std::endl;
    std::cout << "  Log RR: " << rr_result.log_relative_risk << std::endl;
    std::cout << "  SE(log RR): " << rr_result.se_log_relative_risk << std::endl;
    std::cout << "  95% CI: [" << rr_result.ci_lower << ", " << rr_result.ci_upper << "]" << std::endl;

    double risk_treatment = 50.0 / (50.0 + 10.0);
    double risk_control = 30.0 / (30.0 + 20.0);

    std::cout << "\nRisk Calculation:" << std::endl;
    std::cout << "  Treatment risk: " << risk_treatment << " (" << (risk_treatment * 100) << "%)" << std::endl;
    std::cout << "  Control risk: " << risk_control << " (" << (risk_control * 100) << "%)" << std::endl;

    std::cout << "\nInterpretation:" << std::endl;
    if (rr_result.relative_risk > 1.0) {
        std::cout << "  Treatment group has " << rr_result.relative_risk
                  << " times higher risk of success than control" << std::endl;
    } else if (rr_result.relative_risk < 1.0) {
        std::cout << "  Treatment group has lower risk of success than control" << std::endl;
    }

    // ============================================================================
    // 4. Risk Difference (Attributable Risk)
    // ============================================================================
    std::cout << "\n4. Risk Difference (Attributable Risk)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    auto rd_result = statcpp::risk_difference(treatment_table);

    std::cout << "Risk Difference Results:" << std::endl;
    std::cout << "  Risk Difference: " << rd_result.risk_difference << std::endl;
    std::cout << "  Standard Error: " << rd_result.se << std::endl;
    std::cout << "  95% CI: [" << rd_result.ci_lower << ", " << rd_result.ci_upper << "]" << std::endl;

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  Absolute risk difference: " << (rd_result.risk_difference * 100) << "%" << std::endl;
    if (rd_result.risk_difference > 0) {
        std::cout << "  Treatment increases success by " << (rd_result.risk_difference * 100)
                  << " percentage points" << std::endl;
    }

    // ============================================================================
    // 5. Number Needed to Treat (NNT)
    // ============================================================================
    std::cout << "\n5. Number Needed to Treat (NNT)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double nnt = statcpp::number_needed_to_treat(treatment_table);

    std::cout << "NNT: " << nnt << std::endl;
    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  Approximately " << std::ceil(nnt) << " patients need to be treated" << std::endl;
    std::cout << "  to achieve one additional success compared to control" << std::endl;

    // ============================================================================
    // 6. Disease-Exposure Study Example
    // ============================================================================
    std::cout << "\n6. Disease-Exposure Study Example" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Smoking and lung cancer association (hypothetical data)
    //              Lung Cancer+  Lung Cancer-
    // Smoker            80              120
    // Non-smoker        20              180

    std::vector<std::vector<std::size_t>> smoking_table = {
        {80, 120},   // Smokers
        {20, 180}    // Non-smokers
    };

    std::cout << "Smoking and Lung Cancer Study:" << std::endl;
    std::cout << "             Cancer+  Cancer-" << std::endl;
    std::cout << "Smokers          80      120" << std::endl;
    std::cout << "Non-smokers      20      180" << std::endl;

    auto smoking_or = statcpp::odds_ratio(smoking_table);
    auto smoking_rr = statcpp::relative_risk(smoking_table);
    auto smoking_rd = statcpp::risk_difference(smoking_table);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Odds Ratio: " << smoking_or.odds_ratio
              << " (95% CI: [" << smoking_or.ci_lower << ", " << smoking_or.ci_upper << "])" << std::endl;
    std::cout << "  Relative Risk: " << smoking_rr.relative_risk
              << " (95% CI: [" << smoking_rr.ci_lower << ", " << smoking_rr.ci_upper << "])" << std::endl;
    std::cout << "  Risk Difference: " << smoking_rd.risk_difference
              << " (95% CI: [" << smoking_rd.ci_lower << ", " << smoking_rd.ci_upper << "])" << std::endl;

    double smoker_risk = 80.0 / (80.0 + 120.0);
    double nonsmoker_risk = 20.0 / (20.0 + 180.0);

    std::cout << "\nCancer Incidence Rate:" << std::endl;
    std::cout << "  Smokers: " << (smoker_risk * 100) << "%" << std::endl;
    std::cout << "  Non-smokers: " << (nonsmoker_risk * 100) << "%" << std::endl;

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  Smokers have " << smoking_or.odds_ratio << " times the odds of lung cancer compared to non-smokers" << std::endl;
    std::cout << "  Smokers have " << smoking_rr.relative_risk << " times the risk of lung cancer compared to non-smokers" << std::endl;

    // ============================================================================
    // 7. Vaccine Efficacy Example
    // ============================================================================
    std::cout << "\n7. Vaccine Efficacy Study" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    //              Infected  Not Infected
    // Vaccinated       5          495
    // Unvaccinated    50          450

    std::vector<std::vector<std::size_t>> vaccine_table = {
        {5, 495},     // Vaccinated
        {50, 450}     // Unvaccinated
    };

    std::cout << "Vaccine Trial Results:" << std::endl;
    std::cout << "             Infected  Not Infected" << std::endl;
    std::cout << "Vaccinated        5          495" << std::endl;
    std::cout << "Unvaccinated     50          450" << std::endl;

    auto vaccine_rr = statcpp::relative_risk(vaccine_table);

    double attack_rate_vaccinated = 5.0 / 500.0;
    double attack_rate_unvaccinated = 50.0 / 500.0;
    double vaccine_efficacy = 1.0 - vaccine_rr.relative_risk;

    std::cout << "\nAttack Rate:" << std::endl;
    std::cout << "  Vaccinated: " << (attack_rate_vaccinated * 100) << "%" << std::endl;
    std::cout << "  Unvaccinated: " << (attack_rate_unvaccinated * 100) << "%" << std::endl;

    std::cout << "\nRelative Risk: " << vaccine_rr.relative_risk << std::endl;
    std::cout << "Vaccine Efficacy: " << (vaccine_efficacy * 100) << "%" << std::endl;

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  The vaccine reduces infection risk by " << (vaccine_efficacy * 100) << "%" << std::endl;

    try {
        double vaccine_nnt = statcpp::number_needed_to_treat(vaccine_table);
        std::cout << "  NNT: " << std::ceil(vaccine_nnt)
                  << " (this many need to be vaccinated to prevent one infection)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  (NNT calculation requires positive risk difference)" << std::endl;
    }

    // ============================================================================
    // 8. Comparison of Odds Ratio and Risk Ratio
    // ============================================================================
    std::cout << "\n8. Comparison of Odds Ratio and Risk Ratio" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Which measure to use:" << std::endl;
    std::cout << "  - Relative Risk (RR): Used in cohort studies and randomized trials" << std::endl;
    std::cout << "    More intuitive interpretation" << std::endl;
    std::cout << "  - Odds Ratio (OR): Used in case-control studies" << std::endl;
    std::cout << "    Approximates RR when outcome is rare (<10%)" << std::endl;
    std::cout << "  - Risk Difference (RD): Shows absolute effect size" << std::endl;
    std::cout << "    Useful for clinical decision-making" << std::endl;

    std::cout << "\nFor the treatment study:" << std::endl;
    std::cout << "  Odds Ratio: " << or_result.odds_ratio << std::endl;
    std::cout << "  Risk Ratio: " << rr_result.relative_risk << std::endl;
    std::cout << "  Note: OR and RR differ because success rate is not rare" << std::endl;

    std::cout << "\n=== Examples completed successfully ===" << std::endl;

    return 0;
}
