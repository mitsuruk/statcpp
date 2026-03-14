/**
 * @file example_continuous_distributions.cpp
 * @brief Sample code for continuous probability distributions
 *
 * Demonstrates usage of normal distribution, log-normal distribution,
 * t-distribution, chi-squared distribution, F-distribution,
 * exponential distribution, Weibull distribution, and other continuous
 * probability distributions.
 */

#include <iostream>
#include <iomanip>
#include "statcpp/continuous_distributions.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=== Continuous Distribution Examples ===\n\n";

    // Normal Distribution
    std::cout << "1. Normal Distribution (mu=0, sigma=1)\n";
    std::cout << "   PDF at x=0: " << statcpp::normal_pdf(0.0) << "\n";
    std::cout << "   CDF at x=0: " << statcpp::normal_cdf(0.0) << "\n";
    std::cout << "   95th percentile: " << statcpp::normal_quantile(0.95) << "\n";
    std::cout << "   Random sample: " << statcpp::normal_rand() << "\n\n";

    // Log-normal Distribution
    std::cout << "2. Log-normal Distribution (mu=0, sigma=1)\n";
    std::cout << "   PDF at x=1: " << statcpp::lognormal_pdf(1.0) << "\n";
    std::cout << "   CDF at x=1: " << statcpp::lognormal_cdf(1.0) << "\n";
    std::cout << "   Median: " << statcpp::lognormal_quantile(0.5) << "\n";
    std::cout << "   Random sample: " << statcpp::lognormal_rand() << "\n\n";

    // t-Distribution
    std::cout << "3. Student's t-distribution (df=10)\n";
    std::cout << "   PDF at t=0: " << statcpp::t_pdf(0.0, 10.0) << "\n";
    std::cout << "   CDF at t=2: " << statcpp::t_cdf(2.0, 10.0) << "\n";
    std::cout << "   97.5th percentile: " << statcpp::t_quantile(0.975, 10.0) << "\n";
    std::cout << "   Random sample: " << statcpp::t_rand(10.0) << "\n\n";

    // Chi-square Distribution
    std::cout << "4. Chi-squared Distribution (df=5)\n";
    std::cout << "   PDF at x=5: " << statcpp::chisq_pdf(5.0, 5.0) << "\n";
    std::cout << "   CDF at x=5: " << statcpp::chisq_cdf(5.0, 5.0) << "\n";
    std::cout << "   95th percentile: " << statcpp::chisq_quantile(0.95, 5.0) << "\n";
    std::cout << "   Random sample: " << statcpp::chisq_rand(5.0) << "\n\n";

    // F-Distribution
    std::cout << "5. F-distribution (df1=5, df2=10)\n";
    std::cout << "   PDF at x=1: " << statcpp::f_pdf(1.0, 5.0, 10.0) << "\n";
    std::cout << "   CDF at x=2: " << statcpp::f_cdf(2.0, 5.0, 10.0) << "\n";
    std::cout << "   95th percentile: " << statcpp::f_quantile(0.95, 5.0, 10.0) << "\n";
    std::cout << "   Random sample: " << statcpp::f_rand(5.0, 10.0) << "\n\n";

    // Exponential Distribution
    std::cout << "6. Exponential Distribution (lambda=2.0)\n";
    std::cout << "   PDF at x=0.5: " << statcpp::exponential_pdf(0.5, 2.0) << "\n";
    std::cout << "   CDF at x=0.5: " << statcpp::exponential_cdf(0.5, 2.0) << "\n";
    std::cout << "   Median: " << statcpp::exponential_quantile(0.5, 2.0) << "\n";
    std::cout << "   Random sample: " << statcpp::exponential_rand(2.0) << "\n\n";

    // Weibull Distribution
    std::cout << "7. Weibull Distribution (shape=2.0, scale=1.0)\n";
    std::cout << "   PDF at x=1: " << statcpp::weibull_pdf(1.0, 2.0, 1.0) << "\n";
    std::cout << "   CDF at x=1: " << statcpp::weibull_cdf(1.0, 2.0, 1.0) << "\n";
    std::cout << "   Median: " << statcpp::weibull_quantile(0.5, 2.0, 1.0) << "\n";
    std::cout << "   Random sample: " << statcpp::weibull_rand(2.0, 1.0) << "\n\n";

    return 0;
}
