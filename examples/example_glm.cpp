/**
 * @file example_glm.cpp
 * @brief Generalized Linear Model (GLM) Sample Code
 *
 * Demonstrates the usage of logistic regression, Poisson regression,
 * link functions, residual calculations, overdispersion tests,
 * and other GLM techniques.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include "statcpp/glm.hpp"

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
    // 1. Logistic Regression
    // ============================================================================
    print_section("1. Logistic Regression");

    std::cout << R"(
[Concept]
Models binary response variables (0/1)
Applies linear model via logit transformation

[Example: Exam Pass/Fail Prediction]
Predict probability of passing based on study hours
-> P(pass) = 1 / (1 + exp(-(B0 + B1 * hours)))
)";

    // Example: Exam pass/fail prediction (explanatory variable: study hours)
    std::vector<std::vector<double>> X_logistic = {
        {1.0},  // study hours
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0},
        {7.0},
        {8.0},
        {9.0},
        {10.0}
    };

    std::vector<double> y_binary = {0, 0, 0, 1, 0, 1, 1, 1, 1, 1};  // pass/fail

    auto logistic_model = statcpp::logistic_regression(X_logistic, y_binary);

    print_subsection("Logistic Regression Model");
    std::cout << "  Coefficients:\n";
    std::cout << "    Intercept: " << logistic_model.coefficients[0] << "\n";
    std::cout << "    Study hours: " << logistic_model.coefficients[1] << "\n";

    print_subsection("Model Fit");
    std::cout << "  Residual deviance: " << logistic_model.residual_deviance << "\n";
    std::cout << "  Null deviance: " << logistic_model.null_deviance << "\n";
    std::cout << "  Converged: " << (logistic_model.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << logistic_model.iterations << "\n";

    // McFadden's Pseudo R-squared
    double pseudo_r2 = statcpp::pseudo_r_squared_mcfadden(logistic_model);
    std::cout << "  McFadden's Pseudo R-squared: " << pseudo_r2 << "\n";
    std::cout << "  -> Closer to 1 indicates better fit\n";

    // Calculate odds ratios
    auto odds_ratios = statcpp::odds_ratios(logistic_model);
    print_subsection("Odds Ratios");
    for (std::size_t i = 0; i < odds_ratios.size(); ++i) {
        std::cout << "  Coefficient " << (i + 1) << ": OR = " << odds_ratios[i];
        if (i == 0) {
            std::cout << " (per 1-hour increase in study time)";
        }
        std::cout << std::endl;
    }
    std::cout << "  -> OR > 1: positive effect, OR < 1: negative effect\n";

    // Predicted probabilities
    print_subsection("Predicted Probabilities");
    for (double hours = 3.0; hours <= 8.0; hours += 1.0) {
        std::vector<double> x_new = {hours};
        double prob = statcpp::predict_probability(logistic_model, x_new);
        std::cout << "  Study time " << hours << " hours: P(pass) = " << prob << std::endl;
    }
    std::cout << "  -> More study hours increases probability of passing\n";

    // ============================================================================
    // 2. Poisson Regression
    // ============================================================================
    print_section("2. Poisson Regression");

    std::cout << R"(
[Concept]
Models count data (non-negative integers)
Applies linear model via log link function

[Example: Website Visit Prediction]
Predict visitor count from advertising budget
-> log(E[visitors]) = B0 + B1 * budget
)";

    // Example: Website visit prediction (explanatory variable: ad budget)
    std::vector<std::vector<double>> X_poisson = {
        {10.0},  // ad budget (in thousands)
        {15.0},
        {20.0},
        {25.0},
        {30.0},
        {35.0},
        {40.0},
        {45.0}
    };

    std::vector<double> y_count = {12, 18, 25, 30, 38, 42, 50, 55};  // visitor counts

    auto poisson_model = statcpp::poisson_regression(X_poisson, y_count);

    print_subsection("Poisson Regression Model");
    std::cout << "  Coefficients:\n";
    std::cout << "    Intercept: " << poisson_model.coefficients[0] << "\n";
    std::cout << "    Ad budget: " << poisson_model.coefficients[1] << "\n";

    print_subsection("Model Fit");
    std::cout << "  Residual deviance: " << poisson_model.residual_deviance << "\n";
    std::cout << "  Converged: " << (poisson_model.converged ? "Yes" : "No") << "\n";

    // Incidence Rate Ratios
    auto irr = statcpp::incidence_rate_ratios(poisson_model);
    print_subsection("Incidence Rate Ratios (IRR)");
    for (std::size_t i = 0; i < irr.size(); ++i) {
        std::cout << "  Coefficient " << (i + 1) << ": IRR = " << irr[i];
        if (i == 0) {
            std::cout << " (per $1000 increase in ad budget)";
        }
        std::cout << std::endl;
    }
    std::cout << "  -> IRR > 1: positive effect (count increases)\n";

    // Predicted counts
    print_subsection("Predicted Counts");
    for (double budget = 20.0; budget <= 40.0; budget += 5.0) {
        std::vector<double> x_new = {budget};
        double count = statcpp::predict_count(poisson_model, x_new);
        std::cout << "  Ad budget $" << budget << "k: E[visitors] = " << count << "\n";
    }
    std::cout << "  -> Higher ad budget increases visitor count\n";

    // ============================================================================
    // 3. Link Function Comparison
    // ============================================================================
    print_section("3. Link Functions");

    std::cout << R"(
[Concept]
Functions that connect expected value of response variable to linear predictor
Different link functions model different assumptions

[Main Link Functions]
- Logit: Optimal for binomial distribution
- Probit: Cumulative density function of normal distribution
- Identity: Regular linear regression
- Log: Poisson regression
)";

    double mu_value = 0.7;

    print_subsection("Forward Transform (mu -> eta)");
    std::cout << "When mu = " << mu_value << ":\n";
    std::cout << "  Logit: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::logit) << "\n";
    std::cout << "  Probit: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::probit) << "\n";
    std::cout << "  Identity: " << statcpp::detail::link_transform(mu_value, statcpp::link_function::identity) << "\n";

    double eta_value = 0.5;
    print_subsection("Inverse Transform (eta -> mu)");
    std::cout << "When eta = " << eta_value << ":\n";
    std::cout << "  Inverse Logit: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::logit) << "\n";
    std::cout << "  Inverse Probit: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::probit) << "\n";
    std::cout << "  Identity: " << statcpp::detail::inverse_link(eta_value, statcpp::link_function::identity) << "\n";

    // ============================================================================
    // 4. GLM Residual Calculation
    // ============================================================================
    print_section("4. GLM Residuals");

    std::cout << R"(
[Concept]
GLM has multiple types of residuals
Each type provides different diagnostic information

[Residual Types]
- Response residuals: Difference between observed and predicted values
- Pearson residuals: Standardized residuals
- Deviance residuals: Contribution to deviance
)";

    auto residuals = statcpp::compute_glm_residuals(logistic_model, X_logistic, y_binary);

    print_subsection("Logistic Regression Residuals");
    std::cout << "  Obs   Response  Pearson   Deviance\n";
    for (std::size_t i = 0; i < std::min(std::size_t(5), residuals.response.size()); ++i) {
        std::cout << "   " << std::setw(2) << i
                  << "    " << std::setw(8) << residuals.response[i]
                  << "  " << std::setw(8) << residuals.pearson[i]
                  << "  " << std::setw(8) << residuals.deviance[i]
                  << std::endl;
    }

    std::cout << "\nResidual Interpretation:\n";
    std::cout << "  - Response residuals: y - predicted value\n";
    std::cout << "  - Pearson residuals: Standardized residuals\n";
    std::cout << "  - Deviance residuals: Contribution to model deviance\n";

    // ============================================================================
    // 5. Overdispersion Test
    // ============================================================================
    print_section("5. Overdispersion Test");

    std::cout << R"(
[Concept]
Evaluates whether Poisson distribution assumption is appropriate
Overdispersion occurs when variance exceeds expected value

[Example: Poisson Model Diagnostics]
Check if dispersion parameter is close to 1
-> If > 1, consider negative binomial distribution
)";

    double disp_stat = statcpp::overdispersion_test(poisson_model, X_poisson, y_count);

    print_subsection("Dispersion Statistic");
    std::cout << "  Dispersion parameter: " << disp_stat << "\n";

    std::cout << "\nInterpretation:\n";
    std::cout << "  ~ 1.0: No overdispersion (Poisson is appropriate)\n";
    std::cout << "  > 1.0: Overdispersion present (consider negative binomial)\n";
    std::cout << "  < 1.0: Underdispersion (rare)\n";

    if (disp_stat > 1.5) {
        std::cout << "\nResult: Significant overdispersion detected\n";
    } else {
        std::cout << "\nResult: No significant overdispersion\n";
    }

    // ============================================================================
    // 6. Summary: GLM Family Selection
    // ============================================================================
    print_section("Summary: GLM Family and Link Function Selection");

    std::cout << R"(
[GLM Selection by Response Variable Type]

Binary/Binomial data (0/1, success/failure):
  - Family: Binomial
  - Link function: Logit (most common) or Probit
  - Examples: Disease presence, loan default, click occurrence

Count data (0, 1, 2, ...):
  - Family: Poisson
  - Link function: Log
  - Examples: Event occurrences, store visitors, error counts
  - Note: Check for overdispersion

Continuous data (positive values only):
  - Family: Gamma
  - Link function: Log or Inverse
  - Examples: Insurance claims, survival time, waiting time

Continuous data (any real number):
  - Family: Gaussian/Normal
  - Link function: Identity
  - Note: This is ordinary linear regression

[Model Diagnostic Checklist]

1. Convergence check:
   - Did the algorithm converge?
   - Logistic: )" << (logistic_model.converged ? "Converged" : "Not converged") << R"(
   - Poisson: )" << (poisson_model.converged ? "Converged" : "Not converged") << R"(

2. Residual validation:
   - Check if residual plots show patterns
   - Check for outliers

3. Overdispersion test (count models):
   - Is dispersion parameter ~ 1?

4. Model fit assessment:
   - Compare deviance with degrees of freedom
   - Use pseudo R-squared for binomial models

5. Prediction validation:
   - Check if predictions are reasonable
   - Perform cross-validation if data is sufficient

[Advantages of GLM]
- Appropriately models response variable distribution
- Flexible extension of linear models
- Statistical inference via maximum likelihood estimation
- Applicable to diverse fields
)";

    return 0;
}
