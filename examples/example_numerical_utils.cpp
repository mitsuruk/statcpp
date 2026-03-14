/**
 * @file example_numerical_utils.cpp
 * @brief Sample code for statcpp::numerical_utils.hpp
 *
 * This file demonstrates the usage of numerical utility functions
 * provided in numerical_utils.hpp through practical examples.
 *
 * [Provided Functions]
 * - approx_equal()            : Approximate equality for floating-point numbers
 * - has_converged()           : Convergence check
 * - log1p_safe(), expm1_safe(): Safe mathematical functions
 * - kahan_sum()               : High-precision summation (Kahan summation)
 * - safe_divide()             : Safe division (avoid division by zero)
 * - clamp()                   : Value range limiting
 * - in_range()                : Range check
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_numerical_utils.cpp -o example_numerical_utils
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include "statcpp/numerical_utils.hpp"

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

// ============================================================================
// 1. Floating-Point Problems and Approximate Equality
// ============================================================================

/**
 * @brief Example of floating-point comparison and approximate equality
 *
 * [Problem]
 * Computers cannot represent floating-point numbers exactly,
 * so mathematically equal calculation results may differ
 *
 * [Causes]
 * - Binary representation limits
 * - Rounding error accumulation
 * - Catastrophic cancellation
 *
 * [Solution]
 * Use approx_equal() to check "approximately equal"
 */
void example_floating_point_comparison() {
    print_section("1. Floating-Point Number Comparison");

    std::cout << R"(
[Problem: Floating-Point Rounding Errors]

Mathematically 0.1 + 0.2 = 0.3, but on computers...
)";

    std::cout << std::setprecision(17);  // High precision display
    double a = 0.1 + 0.2;
    double b = 0.3;

    std::cout << "Result of 0.1 + 0.2: " << a << "\n";
    std::cout << "Value of 0.3:        " << b << "\n";
    std::cout << "Difference:          " << (a - b) << "\n";

    print_subsection("Normal Comparison Operator (==)");
    std::cout << "0.1 + 0.2 == 0.3 ? " << (a == b ? "true" : "false") << "\n";
    std::cout << "-> Returns false!\n";

    print_subsection("Using approx_equal()");
    std::cout << std::setprecision(4);
    bool approx = statcpp::approx_equal(a, b);
    std::cout << "approx_equal(0.1+0.2, 0.3) ? " << (approx ? "true" : "false") << "\n";
    std::cout << "-> Correctly returns true\n";

    print_subsection("Practical Example: Loop Termination");
    std::cout << R"(
Bad example (using ==):
    double x = 0.0;
    while (x != 1.0) {
        x += 0.1;
    }
    // May become infinite loop!

Good example (using approx_equal):
    double x = 0.0;
    while (!approx_equal(x, 1.0)) {
        x += 0.1;
    }
    // Terminates correctly
)";

    std::cout << "\nActually running the loop:\n";
    double x = 0.0;
    int count = 0;
    std::cout << std::setprecision(10);
    while (!statcpp::approx_equal(x, 1.0) && count < 15) {
        std::cout << "  x = " << x << "\n";
        x += 0.1;
        count++;
    }
    std::cout << "  Final x = " << x << "\n";
    std::cout << "  Loop iterations: " << count << "\n";
    std::cout << std::setprecision(4);
}

// ============================================================================
// 2. Convergence Check
// ============================================================================

/**
 * @brief Example of convergence check
 *
 * [Concept]
 * In iterative calculations, terminate when values stop changing significantly
 *
 * [Use Cases]
 * - Iterative algorithms like Newton's method
 * - Optimization algorithms
 * - Fixed-point iteration
 * - EM algorithm
 */
void example_convergence() {
    print_section("2. Convergence Check (has_converged)");

    std::cout << R"(
[Concept]
In iterative calculations, determine "convergence" when
current value is almost unchanged from previous value
and terminate calculation

[Example: Square Root Calculation (Babylonian Method)]
Calculate sqrt(2) iteratively
)";

    print_subsection("Iteration Process");
    double target = 2.0;
    double x = 1.0;  // Initial value
    const int max_iter = 20;

    std::cout << std::setprecision(10);
    std::cout << "  Iteration 0: x = " << x << "\n";

    for (int i = 1; i <= max_iter; ++i) {
        double x_new = 0.5 * (x + target / x);  // Babylonian method update formula

        std::cout << "  Iteration " << i << ": x = " << x_new;

        if (statcpp::has_converged(x_new, x)) {
            std::cout << " <- Converged!\n";
            std::cout << "\nTrue value: " << std::sqrt(target) << "\n";
            std::cout << "Calculated: " << x_new << "\n";
            std::cout << "Error:      " << std::abs(x_new - std::sqrt(target)) << "\n";
            std::cout << "Iterations to converge: " << i << "\n";
            break;
        } else {
            std::cout << "\n";
        }

        x = x_new;
    }
    std::cout << std::setprecision(4);

    print_subsection("Practical Notes");
    std::cout << R"(
Without convergence check:
- Continue unnecessary calculations (wasted computation time)
- Cannot determine when to stop

With convergence check:
- Automatically terminate when sufficient precision reached
- Improved computational efficiency
)";
}

// ============================================================================
// 3. High-Precision Summation (Kahan Summation)
// ============================================================================

/**
 * @brief Example of Kahan summation
 *
 * [Problem]
 * When adding large and small numbers, small numbers can disappear (catastrophic cancellation)
 *
 * [Solution]
 * Kahan summation algorithm corrects rounding errors
 *
 * [Use Cases]
 * - Summing large quantities of numbers
 * - Adding numbers with large magnitude differences
 * - High-precision statistical calculations
 */
void example_kahan_sum() {
    print_section("3. High-Precision Summation (Kahan Summation)");

    std::cout << R"(
[Problem: Catastrophic Cancellation]

When adding large and small numbers, small numbers "disappear"

[Example]
Calculate 10^10 + 1 + 1 - 10^10
Mathematically should be 2, but...
)";

    std::vector<double> values = {1e10, 1.0, 1.0, -1e10};

    print_subsection("Normal Summation");
    double naive_sum = 0.0;
    for (double v : values) {
        naive_sum += v;
    }
    std::cout << "Calculated: " << naive_sum << "\n";
    std::cout << "Expected:   2.0\n";
    std::cout << "Error:      " << std::abs(naive_sum - 2.0) << "\n";
    std::cout << "-> Rounding error prevents correct result!\n";

    print_subsection("Kahan Summation");
    double kahan = statcpp::kahan_sum(values.begin(), values.end());
    std::cout << "Calculated: " << kahan << "\n";
    std::cout << "Expected:   2.0\n";
    std::cout << "Error:      " << std::abs(kahan - 2.0) << "\n";
    std::cout << "-> Correct result obtained!\n";

    print_subsection("Practical Example: Summing Large Data");
    std::cout << R"(
Sum of 1000 values of 0.1 (mathematically 100)
)";

    std::vector<double> many_small_values(1000, 0.1);

    double naive_many = 0.0;
    for (double v : many_small_values) {
        naive_many += v;
    }

    double kahan_many = statcpp::kahan_sum(many_small_values.begin(),
                                            many_small_values.end());

    std::cout << std::setprecision(10);
    std::cout << "Normal sum:   " << naive_many << "\n";
    std::cout << "Kahan sum:    " << kahan_many << "\n";
    std::cout << "Expected:     100.0\n";
    std::cout << "Normal error: " << std::abs(naive_many - 100.0) << "\n";
    std::cout << "Kahan error:  " << std::abs(kahan_many - 100.0) << "\n";
    std::cout << std::setprecision(4);

    std::cout << "\n-> Kahan summation is more precise\n";

    print_subsection("When to Use Kahan Summation");
    std::cout << R"(
Use when:
- Summing large quantities of numbers
- Dealing with numbers of very different magnitudes
- High-precision statistical calculations (mean, variance)
- Numerical integration

Not needed when:
- Summing few numbers
- Numbers are of similar magnitude
- Speed is top priority
)";
}

// ============================================================================
// 4. Safe Division
// ============================================================================

/**
 * @brief Example of safe division
 *
 * [Problem]
 * Division by zero crashes the program
 *
 * [Solution]
 * safe_divide() detects division by zero and returns default value
 *
 * [Use Cases]
 * - Division with user input
 * - Data analysis (possible missing or abnormal values)
 * - Ratio calculations
 */
void example_safe_division() {
    print_section("4. Safe Division (safe_divide)");

    std::cout << R"(
[Problem: Division by Zero]

Normal division causes error when denominator is 0

[Example]
)";

    print_subsection("Using safe_divide()");

    double a = 10.0;
    double b = 2.0;
    double c = 0.0;

    std::cout << "10 / 2 = " << statcpp::safe_divide(a, b) << "\n";
    std::cout << "-> Calculates normally\n\n";

    std::cout << "10 / 0 (default: NaN) = " << statcpp::safe_divide(a, c) << "\n";
    std::cout << "-> Returns NaN (safe)\n\n";

    std::cout << "10 / 0 (default: 0) = " << statcpp::safe_divide(a, c, 0.0) << "\n";
    std::cout << "-> Returns specified default value (0)\n\n";

    std::cout << "10 / 0 (default: infinity) = "
              << statcpp::safe_divide(a, c, std::numeric_limits<double>::infinity()) << "\n";
    std::cout << "-> Returns infinity\n";

    print_subsection("Practical Example: Ratio Calculation");
    std::cout << R"(
Click-through rate (CTR) calculation: clicks / impressions
)";

    std::vector<int> impressions = {1000, 500, 0, 200};  // Impressions
    std::vector<int> clicks = {50, 30, 0, 15};           // Clicks

    std::cout << "\nAd campaign click-through rates:\n";
    for (size_t i = 0; i < impressions.size(); ++i) {
        double ctr = statcpp::safe_divide(
            static_cast<double>(clicks[i]),
            static_cast<double>(impressions[i]),
            0.0  // If impressions is 0, CTR is also 0
        );
        std::cout << "  Campaign " << (i+1) << ": "
                  << clicks[i] << "/" << impressions[i]
                  << " = " << ctr * 100 << "%\n";
    }

    print_subsection("Notes");
    std::cout << R"(
Default value choice depends on use case:
- Statistical calculations -> NaN (treat as missing value)
- Business metrics -> 0 (no applicable data)
- Mathematical limits -> infinity

Choose appropriate default value based on purpose!
)";
}

// ============================================================================
// 5. Value Range Limiting (Clamp)
// ============================================================================

/**
 * @brief Example of clamp() and in_range()
 *
 * [Concept]
 * clamp: Constrain value within specified range
 * in_range: Check if value is within range
 *
 * [Use Cases]
 * - User input validation
 * - Probability value range limiting (0-1)
 * - Image processing (pixel value range)
 * - Parameter validity check
 */
void example_clamp() {
    print_section("5. Value Range Limiting (clamp / in_range)");

    std::cout << R"(
[Concept]
clamp(x, min, max): Constrain x within [min, max] range
in_range(x, min, max): Check if x is within [min, max] range
)";

    print_subsection("clamp() Behavior");
    std::cout << "Range: [0, 10]\n";
    std::cout << "  clamp(5, 0, 10)  = " << statcpp::clamp(5.0, 0.0, 10.0) << " (within range)\n";
    std::cout << "  clamp(-5, 0, 10) = " << statcpp::clamp(-5.0, 0.0, 10.0) << " (limited to lower bound)\n";
    std::cout << "  clamp(15, 0, 10) = " << statcpp::clamp(15.0, 0.0, 10.0) << " (limited to upper bound)\n";

    print_subsection("in_range() Behavior");
    std::cout << "Range: [0, 10]\n";
    std::cout << "  in_range(5, 0, 10)  = " << (statcpp::in_range(5.0, 0.0, 10.0) ? "true" : "false") << "\n";
    std::cout << "  in_range(-5, 0, 10) = " << (statcpp::in_range(-5.0, 0.0, 10.0) ? "true" : "false") << "\n";
    std::cout << "  in_range(15, 0, 10) = " << (statcpp::in_range(15.0, 0.0, 10.0) ? "true" : "false") << "\n";

    print_subsection("Practical Example 1: Probability Value Range Limiting");
    std::cout << R"(
Probabilities must be within [0, 1] range
)";

    std::vector<double> probabilities = {0.5, -0.1, 1.2, 0.0, 1.0};
    std::cout << "\nLimiting calculated probability values to [0, 1]:\n";
    for (double p : probabilities) {
        double clamped = statcpp::clamp(p, 0.0, 1.0);
        std::cout << "  " << p << " -> " << clamped;
        if (p != clamped) {
            std::cout << " (limited)";
        }
        std::cout << "\n";
    }

    print_subsection("Practical Example 2: User Input Validation");
    std::cout << R"(
Validating age input (0-120 is reasonable range)
)";

    std::vector<int> ages = {25, -5, 150, 0, 120};
    std::cout << "\nValidating entered ages:\n";
    for (int age : ages) {
        bool valid = statcpp::in_range(static_cast<double>(age), 0.0, 120.0);
        std::cout << "  Age " << age << ": "
                  << (valid ? "Valid" : "Invalid") << "\n";
    }

    print_subsection("Practical Example 3: Score Normalization");
    std::cout << R"(
Limiting test scores to 0-100 range
)";

    std::vector<double> raw_scores = {85, 120, -10, 0, 100, 105};
    std::cout << "\nRaw score -> Normalized score:\n";
    for (double score : raw_scores) {
        double normalized = statcpp::clamp(score, 0.0, 100.0);
        std::cout << "  " << score << " -> " << normalized;
        if (score != normalized) {
            std::cout << " (out-of-range corrected)";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// 6. Safe Mathematical Functions (log1p, expm1)
// ============================================================================

/**
 * @brief Example of log1p_safe() and expm1_safe()
 *
 * [Problem]
 * When x is close to 0, precision of log(1+x) and exp(x)-1 calculations degrades
 *
 * [Solution]
 * log1p(x) = log(1+x) calculated with high precision
 * expm1(x) = exp(x)-1 calculated with high precision
 *
 * [Use Cases]
 * - Calculating small change rates
 * - Financial calculations (compound interest)
 * - Statistical calculations (log transformations)
 */
void example_safe_math() {
    print_section("6. Safe Mathematical Functions (log1p / expm1)");

    std::cout << R"(
[Problem: Catastrophic Cancellation]

When x is close to 0, precision of log(1+x) and exp(x)-1 degrades

[Example: Compound Interest Calculation with Small Rates]
)";

    double small_x = 1e-10;

    print_subsection("log1p (High-Precision log(1+x))");
    std::cout << std::setprecision(15);
    std::cout << "When x = " << small_x << ":\n";
    std::cout << "  log(1 + x)  = " << std::log(1.0 + small_x) << " (normal)\n";
    std::cout << "  log1p(x)    = " << statcpp::log1p_safe(small_x) << " (high-precision)\n";
    std::cout << "  True value  ~ " << small_x << " (when x is small)\n";

    print_subsection("expm1 (High-Precision exp(x)-1)");
    std::cout << "When x = " << small_x << ":\n";
    std::cout << "  exp(x) - 1  = " << (std::exp(small_x) - 1.0) << " (normal)\n";
    std::cout << "  expm1(x)    = " << statcpp::expm1_safe(small_x) << " (high-precision)\n";
    std::cout << "  True value  ~ " << small_x << " (when x is small)\n";
    std::cout << std::setprecision(4);

    print_subsection("Practical Example: Compound Interest Calculation");
    std::cout << R"(
Interest earned from investing for 1 year at 0.01% annual rate (= 0.0001)
Principal: 1,000,000 units
)";

    double principal = 1000000.0;
    double rate = 0.0001;  // 0.01%

    double interest_naive = principal * (std::exp(rate) - 1.0);
    double interest_accurate = principal * statcpp::expm1_safe(rate);

    std::cout << "\nNormal calculation:        " << interest_naive << " units\n";
    std::cout << "High-precision calculation: " << interest_accurate << " units\n";
    std::cout << "Theoretical value:          " << principal * rate << " units\n";
    std::cout << "\n-> High-precision calculation is more accurate\n";
}

// ============================================================================
// Summary
// ============================================================================

void print_summary() {
    print_section("Summary: Numerical Utilities");

    std::cout << R"(
+---------------------+------------------------------------------+
| Function            | Purpose                                  |
+---------------------+------------------------------------------+
| approx_equal()      | Approximate equality for floating-point  |
| has_converged()     | Convergence check for iterations         |
| kahan_sum()         | High-precision summation (reduce rounding|
|                     | errors)                                  |
| safe_divide()       | Safe division avoiding division by zero  |
| clamp()             | Constrain value within specified range   |
| in_range()          | Check if value is within range           |
| log1p_safe()        | High-precision log(1+x) calculation      |
| expm1_safe()        | High-precision exp(x)-1 calculation      |
+---------------------+------------------------------------------+

[Floating-Point Considerations]
1. Avoid comparison with == -> Use approx_equal()
2. Large quantity summations accumulate rounding errors -> Use kahan_sum()
3. Small value log, exp calculations lose precision -> Use log1p, expm1

[Practical Best Practices]
DO:
- Use approx_equal() for floating-point comparison
- Implement convergence checks in iterative calculations
- Use clamp() to constrain user input values
- Use safe_divide() when division by zero is possible

DON'T:
- Use == with floating-point numbers
- Ignore infinite loop risks
- Ignore rounding errors

[Performance Trade-offs]
- High-precision functions (Kahan sum, etc.) are slightly slower
- Use only when precision is important
- Most effective for large data or iterative calculations
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_floating_point_comparison();
    example_convergence();
    example_kahan_sum();
    example_safe_division();
    example_clamp();
    example_safe_math();

    // Display summary
    print_summary();

    return 0;
}
