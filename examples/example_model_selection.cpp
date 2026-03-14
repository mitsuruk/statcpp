/**
 * @file example_model_selection.cpp
 * @brief Model Selection Sample Code
 *
 * Demonstrates the usage of AIC, BIC, cross-validation, LOOCV,
 * ridge regression, Lasso regression, Elastic Net regression,
 * and other model selection and regularization techniques.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/model_selection.hpp"
#include "statcpp/linear_regression.hpp"

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
    // 1. Information Criteria (AIC, BIC)
    // ============================================================================
    print_section("1. Information Criteria (AIC, BIC)");

    std::cout << R"(
[Concept]
AIC (Akaike Information Criterion): Evaluates balance between model fit and complexity
BIC (Bayesian Information Criterion): Penalizes complexity more than AIC

[Example: Simple Regression Model Evaluation]
Evaluates trade-off between data fit and number of parameters
-> Lower values indicate better models
)";

    // Sample data
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {2.1, 4.2, 5.9, 8.1, 10.3, 12.0, 14.2, 16.1, 18.0, 20.1};

    // 2D matrix format for cross-validation (with intercept column)
    std::vector<std::vector<double>> X_simple;
    for (double val : x) {
        X_simple.push_back({1.0, val});  // intercept, x
    }

    // Simple regression model
    auto model = statcpp::simple_linear_regression(x.begin(), x.end(), y.begin(), y.end());

    print_subsection("Simple Regression Model");
    std::cout << "  Fitted equation: y = " << model.intercept << " + " << model.slope << "x\n";
    std::cout << "  R-squared = " << model.r_squared << "\n";

    double aic_value = statcpp::aic_linear(model, x.size());
    double bic_value = statcpp::bic_linear(model, x.size());

    print_subsection("Information Criteria");
    std::cout << "  AIC: " << aic_value << "\n";
    std::cout << "  BIC: " << bic_value << "\n";

    std::cout << "\nInterpretation:\n";
    std::cout << "  - Lower values indicate better models\n";
    std::cout << "  - BIC penalizes complexity more than AIC\n";
    std::cout << "  - Used for comparing multiple models\n";

    // ============================================================================
    // 2. Cross-Validation (K-Fold)
    // ============================================================================
    print_section("2. K-Fold Cross-Validation");

    std::cout << R"(
[Concept]
Split data into K parts, train on K-1 parts and validate on 1 part repeatedly
Evaluates predictive performance and detects overfitting

[Example: 5-Fold Cross-Validation]
Split data into 5 parts, use each part once for validation
-> Evaluates model's generalization performance
)";

    std::size_t k_folds = 5;
    auto cv_result = statcpp::cross_validate_linear(X_simple, y, k_folds);

    print_subsection(std::to_string(k_folds) + "-Fold Cross-Validation Results");
    std::cout << "  Mean MSE: " << cv_result.mean_error << "\n";
    std::cout << "  Standard Error: " << cv_result.se_error << "\n";

    std::cout << "\nScores for each fold:\n";
    for (std::size_t i = 0; i < cv_result.fold_errors.size(); ++i) {
        std::cout << "  Fold " << (i + 1) << ": MSE = " << cv_result.fold_errors[i] << std::endl;
    }

    std::cout << "\n-> Lower MSE indicates better predictive performance\n";

    // ============================================================================
    // 3. Leave-One-Out Cross-Validation (LOOCV)
    // ============================================================================
    print_section("3. Leave-One-Out Cross-Validation (LOOCV)");

    std::cout << R"(
[Concept]
Train on all data except one point, validate on that one point, repeat for all data
Produces nearly unbiased estimates but is computationally expensive

[Example: Number of splits equals data size]
For n data points, perform n-fold cross-validation
)";

    auto loocv_result = statcpp::loocv_linear(X_simple, y);

    print_subsection("LOOCV Results");
    std::cout << "  Mean MSE: " << loocv_result.mean_error << "\n";
    std::cout << "  Standard Error: " << loocv_result.se_error << "\n";

    std::cout << "\n-> LOOCV = " << x.size() << "-fold cross-validation (each observation used once for validation)\n";
    std::cout << "-> Nearly unbiased but computationally expensive\n";

    // ============================================================================
    // 4. Ridge Regression (L2 Regularization)
    // ============================================================================
    print_section("4. Ridge Regression (L2 Regularization)");

    std::cout << R"(
[Concept]
Penalizes sum of squared coefficients to prevent overfitting
Enables stable estimation even with multicollinear variables

[Example: Data with Multicollinearity]
Data with highly correlated variables x2 and x3
Shrinks coefficients for stabilization
)";

    // Data with multicollinearity (intercept is automatically added)
    std::vector<std::vector<double>> X = {
        {2.0, 2.1},  // x1 and x2 are highly correlated
        {3.0, 3.2},
        {4.0, 4.1},
        {5.0, 5.0},
        {6.0, 6.2},
        {7.0, 7.1},
        {8.0, 8.0},
        {9.0, 9.1}
    };
    std::vector<double> y_multi = {5, 8, 11, 14, 17, 20, 23, 26};

    double lambda = 1.0;
    auto ridge_model = statcpp::ridge_regression(X, y_multi, lambda);

    print_subsection("Ridge Regression Model (lambda = " + std::to_string(lambda) + ")");
    std::cout << "  Coefficients: [";
    for (std::size_t i = 0; i < ridge_model.coefficients.size(); ++i) {
        std::cout << ridge_model.coefficients[i];
        if (i + 1 < ridge_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Comparing different lambda values
    print_subsection("Effect of lambda on Coefficients");
    std::cout << "   lambda       B0       B1       B2\n";
    for (double test_lambda : {0.0, 0.5, 1.0, 5.0, 10.0}) {
        auto test_model = statcpp::ridge_regression(X, y_multi, test_lambda);
        std::cout << std::setw(4) << test_lambda << "  ";
        for (std::size_t i = 0; i < test_model.coefficients.size(); ++i) {
            std::cout << std::setw(8) << test_model.coefficients[i];
        }
        std::cout << std::endl;
    }

    std::cout << "\n-> Larger lambda shrinks coefficients toward 0\n";

    // ============================================================================
    // 5. Lasso Regression (L1 Regularization)
    // ============================================================================
    print_section("5. Lasso Regression (L1 Regularization)");

    std::cout << R"(
[Concept]
Penalizes sum of absolute coefficients
Sets some coefficients exactly to 0 (variable selection)

[Example: Automatic Variable Selection]
Sets coefficients of unimportant variables to 0
Creates sparse models
)";

    auto lasso_model = statcpp::lasso_regression(X, y_multi, lambda);

    print_subsection("Lasso Regression Model (lambda = " + std::to_string(lambda) + ")");
    std::cout << "  Coefficients: [";
    for (std::size_t i = 0; i < lasso_model.coefficients.size(); ++i) {
        std::cout << lasso_model.coefficients[i];
        if (i + 1 < lasso_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    print_subsection("Effect of lambda on Coefficients (Variable Selection)");
    std::cout << "   lambda       B0       B1       B2\n";
    for (double test_lambda : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        auto test_model = statcpp::lasso_regression(X, y_multi, test_lambda);
        std::cout << std::setw(4) << test_lambda << "  ";
        for (std::size_t i = 0; i < test_model.coefficients.size(); ++i) {
            std::cout << std::setw(8) << test_model.coefficients[i];
        }
        std::cout << std::endl;
    }

    std::cout << "\n-> Lasso can set coefficients exactly to 0 (variable selection)\n";

    // ============================================================================
    // 6. Elastic Net Regression (L1 + L2 Regularization)
    // ============================================================================
    print_section("6. Elastic Net Regression (L1 + L2 Regularization)");

    std::cout << R"(
[Concept]
Combines strengths of Ridge and Lasso
Uses both L1 and L2 penalties simultaneously

[Example: Hybrid Approach]
Balances variable selection and stability
Selects/excludes correlated variable groups together
)";

    double alpha = 0.5;  // Balance between L1 and L2
    auto enet_model = statcpp::elastic_net_regression(X, y_multi, lambda, alpha);

    print_subsection("Elastic Net Model (lambda = " + std::to_string(lambda) + ", alpha = " + std::to_string(alpha) + ")");
    std::cout << "  Coefficients: [";
    for (std::size_t i = 0; i < enet_model.coefficients.size(); ++i) {
        std::cout << enet_model.coefficients[i];
        if (i + 1 < enet_model.coefficients.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\nMeaning of alpha parameter:\n";
    std::cout << "  alpha = 0: Pure Ridge (L2 only)\n";
    std::cout << "  alpha = 1: Pure Lasso (L1 only)\n";
    std::cout << "  0 < alpha < 1: Mixture of both\n";

    // ============================================================================
    // 7. Regularization Parameter Selection (Cross-Validation)
    // ============================================================================
    print_section("7. Regularization Parameter lambda Selection (Cross-Validation)");

    std::cout << R"(
[Concept]
Automatically select optimal lambda using cross-validation
Search for lambda that minimizes prediction error

[Example: Grid Search]
Select the best from multiple lambda candidates
)";

    std::vector<double> lambda_grid = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    auto ridge_cv_result = statcpp::cv_ridge(X, y_multi, lambda_grid, 5);

    print_subsection("Ridge Regression Cross-Validation Results");
    std::cout << "  Optimal lambda: " << ridge_cv_result.first << "\n";
    std::cout << "  Optimal coefficients: [";
    for (std::size_t i = 0; i < ridge_cv_result.second.size(); ++i) {
        std::cout << ridge_cv_result.second[i];
        if (i + 1 < ridge_cv_result.second.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\n-> This lambda minimizes cross-validation error\n";

    // ============================================================================
    // 8. Summary: Model Selection Guidelines
    // ============================================================================
    print_section("Summary: Model Selection Guidelines");

    std::cout << R"(
[When to Use Each Regularization Method]

Ridge Regression:
  - Use case: Multicollinearity, all variables are important
  - Effect: Shrinks coefficients, retains all variables
  - Example: Prediction with many correlated variables

Lasso Regression:
  - Use case: Variable selection needed, sparse models
  - Effect: Sets some coefficients exactly to 0
  - Example: High-dimensional data, interpretability priority

Elastic Net:
  - Use case: Group selection, many variables
  - Effect: Combines advantages of both
  - Example: p >> n, correlated variable groups

[Validation Strategies]
- K-Fold Cross-Validation: General purpose (k=5 or 10)
- LOOCV: Small datasets, unbiased estimation
- Train/Test Split: Large datasets, fast

[Model Comparison Criteria]
- AIC: Prediction-focused, smaller penalty
- BIC: Consistency, larger penalty for complexity
- Cross-Validation: Direct measurement of predictive performance

[Practical Advice]
1. Apply standardization first
2. Select lambda via cross-validation
3. Compare multiple methods
4. Consider interpretability
5. Utilize domain knowledge
)";

    return 0;
}
