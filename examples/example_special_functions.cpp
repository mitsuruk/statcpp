/**
 * @file example_special_functions.cpp
 * @brief Sample code for special functions
 *
 * Demonstrates usage examples of special functions including
 * gamma function, beta function, error function, regularized
 * incomplete gamma function, and regularized incomplete beta function.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/special_functions.hpp"

int main() {
    std::cout << "=== Special Functions Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // ============================================================================
    // 1. Gamma Function and Its Logarithm
    // ============================================================================
    std::cout << "\n1. Gamma Function and Its Logarithm" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> gamma_values = {1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5};

    std::cout << "Gamma function values:" << std::endl;
    for (double x : gamma_values) {
        double gamma_x = statcpp::tgamma(x);
        double lgamma_x = statcpp::lgamma(x);
        std::cout << "  Gamma(" << std::setw(4) << x << ") = " << std::setw(12) << gamma_x
                  << ",  log Gamma(" << std::setw(4) << x << ") = " << std::setw(12) << lgamma_x
                  << std::endl;
    }

    std::cout << "\nNote: For positive integers, Gamma(n) = (n-1)!" << std::endl;
    std::cout << "  Gamma(1) = 0! = " << statcpp::tgamma(1.0) << std::endl;
    std::cout << "  Gamma(2) = 1! = " << statcpp::tgamma(2.0) << std::endl;
    std::cout << "  Gamma(3) = 2! = " << statcpp::tgamma(3.0) << std::endl;
    std::cout << "  Gamma(4) = 3! = " << statcpp::tgamma(4.0) << std::endl;
    std::cout << "  Gamma(5) = 4! = " << statcpp::tgamma(5.0) << std::endl;

    // ============================================================================
    // 2. Beta Function
    // ============================================================================
    std::cout << "\n2. Beta Function" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::pair<double, double>> beta_params = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
        {1.0, 2.0}, {2.0, 3.0}, {0.5, 0.5}
    };

    std::cout << "Beta function values:" << std::endl;
    for (const auto& pair : beta_params) {
        double a = pair.first;
        double b = pair.second;
        double beta_ab = statcpp::beta(a, b);
        double lbeta_ab = statcpp::lbeta(a, b);
        std::cout << "  B(" << std::setw(4) << a << ", " << std::setw(4) << b << ") = "
                  << std::setw(12) << beta_ab
                  << ",  log B(" << std::setw(4) << a << ", " << std::setw(4) << b << ") = "
                  << std::setw(12) << lbeta_ab << std::endl;
    }

    std::cout << "\nNote: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)" << std::endl;

    // ============================================================================
    // 3. Regularized Incomplete Beta Function
    // ============================================================================
    std::cout << "\n3. Regularized Incomplete Beta Function" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "I_x(a, b) for various values:" << std::endl;
    double a = 2.0, b = 3.0;
    std::vector<double> x_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << "\nFor a = " << a << ", b = " << b << ":" << std::endl;
    for (double x : x_values) {
        double betainc_val = statcpp::betainc(a, b, x);
        std::cout << "  I_{" << std::setw(3) << x << "}(" << a << ", " << b << ") = "
                  << std::setw(10) << betainc_val << std::endl;
    }

    // Inverse function test
    std::cout << "\nInverse function test (betaincinv):" << std::endl;
    std::vector<double> p_values = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (double p : p_values) {
        double x_inv = statcpp::betaincinv(a, b, p);
        double p_check = statcpp::betainc(a, b, x_inv);
        std::cout << "  p = " << std::setw(4) << p << " -> x = " << std::setw(10) << x_inv
                  << " -> I_x = " << std::setw(10) << p_check << std::endl;
    }

    // ============================================================================
    // 4. Error Function
    // ============================================================================
    std::cout << "\n4. Error Function" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> erf_values = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

    std::cout << "Error function values:" << std::endl;
    for (double x : erf_values) {
        double erf_x = statcpp::erf(x);
        double erfc_x = statcpp::erfc(x);
        std::cout << "  erf(" << std::setw(5) << x << ") = " << std::setw(10) << erf_x
                  << ",  erfc(" << std::setw(5) << x << ") = " << std::setw(10) << erfc_x
                  << std::endl;
    }

    std::cout << "\nNote: erf(x) + erfc(x) = 1" << std::endl;
    std::cout << "  erf(1) + erfc(1) = " << (statcpp::erf(1.0) + statcpp::erfc(1.0)) << std::endl;

    // ============================================================================
    // 5. Standard Normal Distribution CDF
    // ============================================================================
    std::cout << "\n5. Standard Normal Distribution CDF" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Phi(x) values:" << std::endl;
    std::vector<double> norm_values = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

    for (double x : norm_values) {
        double phi_x = statcpp::norm_cdf(x);
        std::cout << "  Phi(" << std::setw(5) << x << ") = " << std::setw(10) << phi_x << std::endl;
    }

    std::cout << "\nNote: Phi(0) = 0.5, Phi(-x) = 1 - Phi(x)" << std::endl;
    std::cout << "  Phi(-1.96) + Phi(1.96) = "
              << (statcpp::norm_cdf(-1.96) + statcpp::norm_cdf(1.96)) << std::endl;

    // ============================================================================
    // 6. Standard Normal Distribution Inverse CDF (Quantile)
    // ============================================================================
    std::cout << "\n6. Standard Normal Distribution Inverse CDF (Quantile)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> quantile_probs = {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99};

    std::cout << "Phi^{-1}(p) values (quantiles):" << std::endl;
    for (double p : quantile_probs) {
        double q = statcpp::norm_quantile(p);
        double p_check = statcpp::norm_cdf(q);
        std::cout << "  Phi^{-1}(" << std::setw(5) << p << ") = " << std::setw(10) << q
                  << " -> Phi(" << std::setw(10) << q << ") = " << std::setw(10) << p_check
                  << std::endl;
    }

    std::cout << "\nCommon quantiles:" << std::endl;
    std::cout << "  90% confidence interval: +-" << statcpp::norm_quantile(0.95) << std::endl;
    std::cout << "  95% confidence interval: +-" << statcpp::norm_quantile(0.975) << std::endl;
    std::cout << "  99% confidence interval: +-" << statcpp::norm_quantile(0.995) << std::endl;

    // ============================================================================
    // 7. Regularized Incomplete Gamma Function
    // ============================================================================
    std::cout << "\n7. Regularized Incomplete Gamma Function" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double gamma_a = 2.5;
    std::vector<double> gamma_x = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::cout << "P(a, x) and Q(a, x) for a = " << gamma_a << ":" << std::endl;
    for (double x : gamma_x) {
        double p_val = statcpp::gammainc_lower(gamma_a, x);
        double q_val = statcpp::gammainc_upper(gamma_a, x);
        std::cout << "  P(" << gamma_a << ", " << std::setw(4) << x << ") = " << std::setw(10) << p_val
                  << ",  Q(" << gamma_a << ", " << std::setw(4) << x << ") = " << std::setw(10) << q_val
                  << ",  P+Q = " << std::setw(10) << (p_val + q_val) << std::endl;
    }

    std::cout << "\nNote: P(a,x) + Q(a,x) = 1" << std::endl;

    // Inverse function test
    std::cout << "\nInverse function test (gammainc_lower_inv):" << std::endl;
    std::vector<double> gamma_p = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (double p : gamma_p) {
        double x_inv = statcpp::gammainc_lower_inv(gamma_a, p);
        double p_check = statcpp::gammainc_lower(gamma_a, x_inv);
        std::cout << "  p = " << std::setw(4) << p << " -> x = " << std::setw(10) << x_inv
                  << " -> P(a,x) = " << std::setw(10) << p_check << std::endl;
    }

    // ============================================================================
    // 8. Mathematical Constants
    // ============================================================================
    std::cout << "\n8. Mathematical Constants" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << std::setprecision(15);
    std::cout << "pi          = " << statcpp::pi << std::endl;
    std::cout << "sqrt(2)     = " << statcpp::sqrt_2 << std::endl;
    std::cout << "sqrt(2*pi)  = " << statcpp::sqrt_2_pi << std::endl;
    std::cout << "log(sqrt(2*pi)) = " << statcpp::log_sqrt_2_pi << std::endl;

    // ============================================================================
    // 9. Practical Example: Chi-Square Distribution CDF
    // ============================================================================
    std::cout << std::setprecision(6);
    std::cout << "\n9. Practical Example: Chi-Square Distribution CDF" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double df = 5.0;  // Degrees of freedom
    std::vector<double> chi2_values = {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0};

    std::cout << "Chi-square distribution CDF with df = " << df << ":" << std::endl;
    std::cout << "(Using formula P(df/2, x/2))" << std::endl;
    for (double x : chi2_values) {
        // Chi-square CDF = P(df/2, x/2)
        double cdf = statcpp::gammainc_lower(df / 2.0, x / 2.0);
        std::cout << "  chi^2(" << std::setw(5) << x << "; df=" << df << ") = " << std::setw(10) << cdf
                  << std::endl;
    }

    // ============================================================================
    // 10. Practical Example: Beta Distribution CDF
    // ============================================================================
    std::cout << "\n10. Practical Example: Beta Distribution CDF" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    double beta_a = 2.0, beta_b = 5.0;
    std::vector<double> beta_x = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::cout << "Beta distribution CDF with alpha = " << beta_a << ", beta = " << beta_b << ":" << std::endl;
    for (double x : beta_x) {
        double cdf = statcpp::betainc(beta_a, beta_b, x);
        std::cout << "  F(" << std::setw(3) << x << "; alpha=" << beta_a << ", beta=" << beta_b << ") = "
                  << std::setw(10) << cdf << std::endl;
    }

    std::cout << "\n=== Examples completed successfully ===" << std::endl;

    return 0;
}
