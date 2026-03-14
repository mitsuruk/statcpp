/**
 * @file example_estimation.cpp
 * @brief Sample code for statcpp::estimation.hpp
 *
 * This file demonstrates the usage of statistical estimation functions
 * (confidence intervals, margin of error) provided in estimation.hpp
 * through practical examples.
 *
 * [Provided Functions]
 * - standard_error()              : Standard error
 * - ci_mean()                     : Confidence interval for mean
 * - margin_of_error_mean()        : Margin of error for mean
 * - ci_proportion()               : Confidence interval for proportion
 * - margin_of_error_proportion()  : Margin of error for proportion
 * - ci_mean_diff_*()              : Confidence interval for two-sample mean difference
 * - ci_variance()                 : Confidence interval for variance
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_estimation.cpp -o example_estimation
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include "statcpp/estimation.hpp"
#include "statcpp/basic_statistics.hpp"

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

template <typename T>
void print_data(const std::string& label, const std::vector<T>& data) {
    std::cout << label << ": ";
    for (const auto& d : data) std::cout << d << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. Concept of Confidence Intervals
// ============================================================================

/**
 * @brief Explanation of confidence interval concepts
 *
 * [Purpose]
 * A Confidence Interval (CI) is an interval that expresses the
 * uncertainty of a population parameter estimate.
 *
 * [Interpretation]
 * Meaning of a 95% confidence interval [L, U]:
 * - NOT "the probability that the parameter is in [L, U] is 95%" (common misconception)
 * - Correct: "If we sample 100 times using the same method and create intervals,
 *   about 95 of them will contain the true parameter"
 *
 * [Confidence Levels]
 * - 90% CI: Narrower interval, lower precision
 * - 95% CI: Standard, good balance
 * - 99% CI: Wider interval, more conservative
 */
void example_ci_concept() {
    print_section("1. Confidence Interval Concepts");

    std::cout << R"(
[What is a Confidence Interval?]
An indicator showing the range where the true value of a
population parameter (mean, proportion, etc.) is expected to be contained

[Visual Image]

True population mean mu
      |
------+---------------------
    [----]  <- Sample 1's 95% CI (contains mu)
       [----]  <- Sample 2's 95% CI (contains mu)
 [----]        <- Sample 3's 95% CI (does NOT contain mu!)
     [----]    <- Sample 4's 95% CI (contains mu)
        [----] <- Sample 5's 95% CI (contains mu)

When sampling 100 times, about 95 intervals will contain mu
(About 5 will not contain it)

[Choosing Confidence Levels]
+----------------+-------------+---------------------------+
| Confidence     | Interval    | Use Case                  |
| Level          | Width       |                           |
+----------------+-------------+---------------------------+
| 90%            | Narrow      | Exploratory analysis      |
| 95%            | Standard    | General research (common) |
| 99%            | Wide        | Pharmaceuticals, safety   |
+----------------+-------------+---------------------------+
)";
}

// ============================================================================
// 2. Standard Error
// ============================================================================

/**
 * @brief Example of standard error usage
 *
 * [Concept]
 * Standard Error (SE) represents the variability of a statistic
 * (such as the mean) due to sampling.
 *
 * [Formula]
 * SE = s / sqrt(n)
 * s: sample standard deviation, n: sample size
 *
 * [Interpretation]
 * - Small SE -> Precise estimate
 * - Large SE -> Imprecise estimate
 * - Increasing n decreases SE (decreases proportionally to sqrt(n))
 *
 * [Use Cases]
 * - Calculating confidence intervals
 * - Computing test statistics for hypothesis tests
 * - Evaluating precision of estimates
 */
void example_standard_error() {
    print_section("2. Standard Error");

    std::cout << R"(
[Concept]
The variability of a statistic (such as the mean) due to sampling

[Example: Height Measurement]
Measuring height (cm) of 20 adult males
)";

    std::vector<double> heights = {
        172, 168, 175, 171, 169, 173, 170, 174,
        168, 172, 171, 169, 173, 170, 172, 171,
        169, 174, 170, 172
    };

    print_data("Height data", heights);

    double mean_height = statcpp::mean(heights.begin(), heights.end());
    double sd = statcpp::sample_stddev(heights.begin(), heights.end());
    double se = statcpp::standard_error(heights.begin(), heights.end());

    std::cout << "\nSample mean: " << mean_height << " cm\n";
    std::cout << "Sample standard deviation: " << sd << " cm\n";
    std::cout << "Standard error: " << se << " cm\n";

    std::cout << "\n-> The smaller the standard error, the more precise the population mean estimate\n";

    print_subsection("Effect of Sample Size");
    std::cout << R"(
Since SE = s / sqrt(n), increasing n decreases SE:

n=20  -> SE = )";
    std::cout << se << " cm\n";
    std::cout << "n=80  -> SE = " << sd / std::sqrt(80) << " cm (decreased to 1/2)\n";
    std::cout << "n=320 -> SE = " << sd / std::sqrt(320) << " cm (decreased to 1/4)\n";

    std::cout << "\n-> Quadrupling n halves SE\n";
}

// ============================================================================
// 3. Confidence Interval for Mean
// ============================================================================

/**
 * @brief Example of confidence interval for mean
 *
 * [Concept]
 * Shows the range of estimated values for population mean mu
 *
 * [Formula]
 * CI = x_bar +/- t(alpha/2, n-1) * SE
 * x_bar: sample mean, t: t-distribution quantile, SE: standard error
 *
 * [Use Cases]
 * - Estimating mean height, mean income, mean scores
 * - Estimating effect sizes in experiments
 * - Evaluating mean differences in A/B tests
 */
void example_ci_mean() {
    print_section("3. Confidence Interval for Mean");

    std::cout << R"(
[Example: Product Lifetime Testing]
Testing battery lifetime (hours) of 8 units
)";

    std::vector<double> lifetimes = {23.1, 25.3, 22.8, 24.5, 26.1, 23.7, 24.9, 25.5};
    print_data("Lifetime data", lifetimes);

    double mean_life = statcpp::mean(lifetimes.begin(), lifetimes.end());
    double se_life = statcpp::standard_error(lifetimes.begin(), lifetimes.end());

    std::cout << "\nSample mean: " << mean_life << " hours\n";
    std::cout << "Standard error: " << se_life << " hours\n";

    // 95% CI
    print_subsection("95% Confidence Interval");
    auto ci_95 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.95);
    std::cout << "95% CI: [" << ci_95.lower << ", " << ci_95.upper << "]\n";
    std::cout << "Point estimate: " << ci_95.point_estimate << "\n";
    std::cout << "\nInterpretation: The population mean lifetime is expected to be\n";
    std::cout << "      between " << ci_95.lower << " and " << ci_95.upper << " hours with 95% confidence\n";

    // 99% CI
    print_subsection("99% Confidence Interval (More Conservative)");
    auto ci_99 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.99);
    std::cout << "99% CI: [" << ci_99.lower << ", " << ci_99.upper << "]\n";
    std::cout << "\n-> 99% CI is wider than 95% CI (more conservative)\n";

    // 90% CI
    print_subsection("90% Confidence Interval (More Aggressive)");
    auto ci_90 = statcpp::ci_mean(lifetimes.begin(), lifetimes.end(), 0.90);
    std::cout << "90% CI: [" << ci_90.lower << ", " << ci_90.upper << "]\n";
    std::cout << "\n-> 90% CI is narrower than 95% CI (used in exploratory analysis)\n";

    print_subsection("Margin of Error");
    double moe_95 = statcpp::margin_of_error_mean(lifetimes.begin(), lifetimes.end(), 0.95);
    std::cout << "Margin of Error (95%): +/-" << moe_95 << " hours\n";
    std::cout << "-> Mean = " << mean_life << " +/- " << moe_95 << " hours\n";
}

// ============================================================================
// 4. Confidence Interval for Proportion
// ============================================================================

/**
 * @brief Example of confidence interval for proportion
 *
 * [Concept]
 * Shows the range of estimated values for population proportion p
 *
 * [Formula]
 * CI = p_hat +/- z(alpha/2) * sqrt[p_hat(1-p_hat)/n]
 * p_hat: sample proportion, z: standard normal quantile
 *
 * [Use Cases]
 * - Approval rating surveys
 * - Estimating defect rates
 * - Estimating conversion rates
 * - Estimating prevalence rates
 */
void example_ci_proportion() {
    print_section("4. Confidence Interval for Proportion");

    std::cout << R"(
[Example: Product Defect Rate Survey]
Out of 1000 products inspected, 65 were defective
-> What is the overall defect rate?
)";

    std::size_t defects = 65;
    std::size_t n = 1000;
    double p_hat = static_cast<double>(defects) / n;

    std::cout << "\nSample size: " << n << " units\n";
    std::cout << "Number of defects: " << defects << " units\n";
    std::cout << "Sample proportion: " << p_hat << " (" << p_hat * 100 << "%)\n";

    // 95% CI
    print_subsection("95% Confidence Interval");
    auto ci_prop = statcpp::ci_proportion(defects, n, 0.95);
    std::cout << "95% CI: [" << ci_prop.lower << ", " << ci_prop.upper << "]\n";
    std::cout << "       ([" << ci_prop.lower * 100 << "%, "
              << ci_prop.upper * 100 << "%])\n";
    std::cout << "\nInterpretation: The population defect rate is expected to be\n";
    std::cout << "      between " << ci_prop.lower * 100 << " and "
              << ci_prop.upper * 100 << "% with 95% confidence\n";

    // Margin of Error
    print_subsection("Margin of Error");
    double moe_prop = statcpp::margin_of_error_proportion(defects, n, 0.95);
    std::cout << "Margin of Error (95%): +/-" << moe_prop << " (+/-"
              << moe_prop * 100 << "%)\n";
    std::cout << "-> Defect rate = " << p_hat * 100 << "% +/- "
              << moe_prop * 100 << "%\n";

    // Worst case margin of error
    print_subsection("Worst Case Margin of Error");
    double moe_worst = statcpp::margin_of_error_proportion_worst_case(n, 0.95);
    std::cout << R"(
Worst case (when p=0.5) margin of error: +/-)" << moe_worst * 100 << "%\n";
    std::cout << "\n-> Use this value for sample size design when proportion is unknown\n";

    print_subsection("Practical Example: Opinion Poll");
    std::cout << R"(
Surveying n=1000 people in approval rating poll
People who answered "yes": 520 (52%)

What is the 95% CI?
)";
    auto poll_ci = statcpp::ci_proportion(520, 1000, 0.95);
    std::cout << "95% CI: [" << poll_ci.lower * 100 << "%, "
              << poll_ci.upper * 100 << "%]\n";
    std::cout << "\n-> True approval rate is between " << poll_ci.lower * 100 << " and "
              << poll_ci.upper * 100 << "%\n";
    std::cout << "   (Can we say it exceeds the majority?)\n";
}

// ============================================================================
// 5. Confidence Interval for Two-Sample Mean Difference
// ============================================================================

/**
 * @brief Example of confidence interval for two-sample mean difference
 *
 * [Concept]
 * Range of estimated values for difference in population means mu1 - mu2
 *
 * [Types]
 * - Pooled (equal variance assumption): Assumes equal variances in both groups
 * - Welch (unequal variance): For potentially different variances
 * - Paired: For same subjects before/after comparisons
 *
 * [Use Cases]
 * - A/B tests (comparing old vs new versions)
 * - Clinical trials (treatment vs control groups)
 * - Educational intervention effectiveness
 */
void example_ci_mean_diff() {
    print_section("5. Confidence Interval for Two-Sample Mean Difference");

    std::cout << R"(
[Example: A/B Test]
Comparing page dwell time (seconds) between new (B) and old (A) website designs
)";

    std::vector<double> design_a = {45, 52, 48, 50, 47, 49, 51, 46, 48, 50};  // Old design
    std::vector<double> design_b = {52, 58, 55, 57, 53, 56, 54, 55, 57, 56};  // New design

    print_data("Old design (A)", design_a);
    print_data("New design (B)", design_b);

    double mean_a = statcpp::mean(design_a.begin(), design_a.end());
    double mean_b = statcpp::mean(design_b.begin(), design_b.end());
    double observed_diff = mean_b - mean_a;

    std::cout << "\nMean dwell time:\n";
    std::cout << "  Old design (A): " << mean_a << " seconds\n";
    std::cout << "  New design (B): " << mean_b << " seconds\n";
    std::cout << "  Difference (B - A): " << observed_diff << " seconds\n";

    print_subsection("Assuming Equal Variance (Pooled t-test)");
    auto ci_pooled = statcpp::ci_mean_diff_pooled(
        design_b.begin(), design_b.end(),
        design_a.begin(), design_a.end(),
        0.95
    );
    std::cout << "95% CI: [" << ci_pooled.lower << ", " << ci_pooled.upper << "]\n";
    std::cout << "Point estimate: " << ci_pooled.point_estimate << "\n";
    std::cout << "\nInterpretation: New design keeps visitors\n";
    std::cout << "      " << ci_pooled.lower << " to " << ci_pooled.upper
              << " seconds longer (95% CI)\n";

    if (ci_pooled.lower > 0) {
        std::cout << "\n-> Since CI is entirely positive, new design's effect is statistically significant\n";
    }

    print_subsection("Assuming Unequal Variance (Welch's t-test)");
    auto ci_welch = statcpp::ci_mean_diff_welch(
        design_b.begin(), design_b.end(),
        design_a.begin(), design_a.end(),
        0.95
    );
    std::cout << "95% CI: [" << ci_welch.lower << ", " << ci_welch.upper << "]\n";
    std::cout << "\n-> When equal variance is uncertain, using Welch is safer\n";

    print_subsection("Practical Example: Paired Data");
    std::cout << R"(
Blood pressure measured before and after treatment for same 10 patients
)";
    std::vector<double> before = {140, 135, 142, 138, 145, 137, 141, 139, 143, 136};
    std::vector<double> after =  {132, 130, 135, 133, 138, 131, 134, 132, 136, 130};

    print_data("Before treatment", before);
    print_data("After treatment", after);

    // For paired data, calculate differences then use ci_mean()
    std::vector<double> differences;
    for (size_t i = 0; i < before.size(); ++i) {
        differences.push_back(before[i] - after[i]);
    }

    print_data("Difference (Before - After)", differences);

    auto ci_paired = statcpp::ci_mean(differences.begin(), differences.end(), 0.95);

    std::cout << "\n95% CI for treatment effect (Before - After): ["
              << ci_paired.lower << ", " << ci_paired.upper << "]\n";
    std::cout << "Point estimate: " << ci_paired.point_estimate << " mmHg reduction\n";

    if (ci_paired.lower > 0) {
        std::cout << "\n-> Treatment significantly reduced blood pressure\n";
    }
}

// ============================================================================
// 6. Confidence Interval for Variance
// ============================================================================

/**
 * @brief Example of confidence interval for variance
 *
 * [Concept]
 * Shows the range of estimated values for population variance sigma^2
 *
 * [Formula]
 * Uses chi-squared distribution
 * CI = [(n-1)s^2 / chi^2(alpha/2), (n-1)s^2 / chi^2(1-alpha/2)]
 *
 * [Use Cases]
 * - Manufacturing process variability assessment
 * - Quality control
 * - Risk assessment
 * - Measurement instrument precision evaluation
 */
void example_ci_variance() {
    print_section("6. Confidence Interval for Variance");

    std::cout << R"(
[Example: Manufacturing Process Variability Assessment]
Evaluating variability in product weight (g)
)";

    std::vector<double> weights = {
        100.2, 99.8, 100.1, 99.9, 100.3,
        99.7, 100.0, 100.2, 99.8, 100.1
    };

    print_data("Weight data", weights);

    double mean_weight = statcpp::mean(weights.begin(), weights.end());
    double var_weight = statcpp::var(weights.begin(), weights.end(), 1);
    double sd_weight = std::sqrt(var_weight);

    std::cout << "\nSample mean: " << mean_weight << " g\n";
    std::cout << "Sample variance: " << var_weight << " g^2\n";
    std::cout << "Sample standard deviation: " << sd_weight << " g\n";

    print_subsection("95% Confidence Interval for Variance");
    auto ci_var = statcpp::ci_variance(weights.begin(), weights.end(), 0.95);
    std::cout << "95% CI: [" << ci_var.lower << ", " << ci_var.upper << "]\n";
    std::cout << "\nInterpretation: Population variance is expected to be\n";
    std::cout << "      between " << ci_var.lower << " and " << ci_var.upper << " g^2 with 95% confidence\n";

    print_subsection("95% Confidence Interval for Standard Deviation");
    std::cout << "95% CI: [" << std::sqrt(ci_var.lower) << ", "
              << std::sqrt(ci_var.upper) << "] g\n";
    std::cout << "\n-> Standard deviation CI is the square root of variance CI\n";

    print_subsection("Practical Example: Quality Control");
    std::cout << R"(
In quality control, variability (variance) assessment is important:
- Large variance -> Unstable product quality
- Small variance -> Stable product quality

Target: Standard deviation < 0.3 g

Current 95% CI: [)" << std::sqrt(ci_var.lower) << ", "
              << std::sqrt(ci_var.upper) << "] g\n";

    if (std::sqrt(ci_var.upper) < 0.3) {
        std::cout << "\n-> Target met (entire CI is below 0.3)\n";
    } else if (std::sqrt(ci_var.lower) > 0.3) {
        std::cout << "\n-> Target not met (entire CI exceeds 0.3)\n";
    } else {
        std::cout << "\n-> Inconclusive (CI straddles 0.3)\n";
    }
}

// ============================================================================
// 7. Sample Size Design
// ============================================================================

/**
 * @brief Sample size design example
 *
 * [Concept]
 * Calculate the required sample size to achieve desired margin of error
 *
 * [Formula (for proportions)]
 * n = (z(alpha/2) / MOE)^2 * p(1-p)
 *
 * When p is unknown, use p=0.5 (maximum variance)
 */
void example_sample_size() {
    print_section("7. Sample Size Design (MOE Method)");

    std::cout << R"(
[Example 1: Exit Poll for National Election]
Want to estimate candidate A's vote share with +/-3% precision (95% CI)
What sample size is needed?
)";

    double desired_moe = 0.03;  // +/-3%

    print_subsection("Without Prior Information (Conservative Estimate)");
    std::cout << "When vote share is unknown, assume p=0.5 (maximum variance)\n";

    // Using new function
    std::size_t n_conservative = statcpp::sample_size_for_moe_proportion(
        desired_moe, 0.95, 0.5);

    std::cout << "Required sample size: " << n_conservative << " people\n";
    std::cout << "\n-> Need to survey about 1068 people in exit poll\n";
    std::cout << "  (Most conservative estimate without prior information)\n";

    print_subsection("With Prior Poll Data");
    std::cout << R"(
If prior poll shows candidate A's support is about 40%:
)";
    double p_prior = 0.40;
    std::size_t n_with_prior = statcpp::sample_size_for_moe_proportion(
        desired_moe, 0.95, p_prior);

    std::cout << "Required sample size: " << n_with_prior << " people\n";
    std::cout << "\n-> About 1025 (slightly less than conservative estimate)\n";
    std::cout << "  Using prior information can reduce survey costs\n";

    // New example: Local election
    print_subsection("[Example 2: Local Election / Mayoral Race]");
    std::cout << R"(
Want to estimate incumbent's approval with +/-5% precision (95% CI)
Budget is limited for local elections, so relaxing precision
)";

    std::size_t n_local = statcpp::sample_size_for_moe_proportion(
        0.05, 0.95, 0.5);

    std::cout << "Required sample size: " << n_local << " people\n";
    std::cout << "\n-> About 385 people (achievable scale with +/-5% precision)\n";

    // New example: Close election
    print_subsection("[Example 3: Close Election Requiring Higher Precision]");
    std::cout << R"(
Two candidates are neck-and-neck, want +/-2% precision (95% CI)
Close races require high precision
)";

    std::size_t n_tight = statcpp::sample_size_for_moe_proportion(
        0.02, 0.95, 0.5);

    std::cout << "Required sample size: " << n_tight << " people\n";
    std::cout << "\n-> About 2401 people (large-scale survey needed for high precision)\n";

    // New example: Changing confidence level
    print_subsection("[Example 4: Changing Confidence Level]");
    std::cout << R"(
Want +/-3% precision with 99% CI (more certain estimate)
)";

    std::size_t n_high_conf = statcpp::sample_size_for_moe_proportion(
        0.03, 0.99, 0.5);

    std::cout << "Required sample size (99% CI): " << n_high_conf << " people\n";

    std::size_t n_normal_conf = statcpp::sample_size_for_moe_proportion(
        0.03, 0.95, 0.5);

    std::cout << "Required sample size (95% CI): " << n_normal_conf << " people\n";
    std::cout << "\n-> Higher confidence level requires larger sample size\n";

    // New example: Referendum
    print_subsection("[Example 5: Referendum / Plebiscite Support Survey]");
    std::cout << R"(
Surveying approval for constitutional amendment or base relocation
Want to estimate approval rate with +/-2.5% precision (95% CI)
)";

    std::size_t n_referendum = statcpp::sample_size_for_moe_proportion(
        0.025, 0.95, 0.5);

    std::cout << "Required sample size: " << n_referendum << " people\n";
    std::cout << "\n-> About 1537 people (important decisions require sufficient precision)\n";

    // Relationship table between MOE and sample size
    print_subsection("Relationship Between MOE and Sample Size (95% CI)");
    std::cout << "\nTypical precision levels for election/opinion surveys:\n";
    std::cout << "+------------+-------------+-----------------------+\n";
    std::cout << "| MOE        | Sample Size | Use Case              |\n";
    std::cout << "+------------+-------------+-----------------------+\n";

    struct MOEExample {
        double moe;
        const char* use_case;
    };

    std::vector<MOEExample> moe_examples = {
        {0.01, "Large national survey"},
        {0.02, "National/close race"},
        {0.03, "Standard exit poll"},
        {0.04, "Medium poll"},
        {0.05, "Local/preliminary"},
        {0.10, "Small exploratory"}
    };

    for (const auto& ex : moe_examples) {
        std::size_t n = statcpp::sample_size_for_moe_proportion(ex.moe, 0.95, 0.5);
        std::cout << "| +/-" << std::setw(5) << ex.moe * 100 << "% | "
                  << std::setw(11) << n << " | "
                  << std::setw(21) << ex.use_case << " |\n";
    }
    std::cout << "+------------+-------------+-----------------------+\n";

    std::cout << "\n-> Key rule: Doubling precision requires quadrupling sample size\n";
    std::cout << "  Example: Going from +/-6% to +/-3% costs 4 times as much\n";
}

// ============================================================================
// Summary
// ============================================================================

void print_summary() {
    print_section("Summary: Statistical Estimation Functions");

    std::cout << R"(
+--------------------------------+-------------------------------------+
| Function                       | Description                         |
+--------------------------------+-------------------------------------+
| standard_error()               | Standard error (estimation          |
|                                | precision)                          |
| ci_mean()                      | Confidence interval for mean        |
| margin_of_error_mean()         | Margin of error for mean            |
| ci_proportion()                | Confidence interval for proportion  |
| margin_of_error_*()            | Margin of error for proportion      |
| sample_size_for_moe_proportion | Sample size for proportion          |
|                                | estimation                          |
| sample_size_for_moe_mean       | Sample size for mean estimation     |
| ci_mean_diff_pooled()          | Two-sample mean diff (equal var)    |
| ci_mean_diff_welch()           | Two-sample mean diff (unequal var)  |
| ci_mean_diff_paired()          | Paired mean difference              |
| ci_variance()                  | Confidence interval for variance    |
+--------------------------------+-------------------------------------+

[Confidence Interval Interpretation]
Correct interpretation:
   "If we sample 100 times using the same method, about 95 of the
   resulting intervals will contain the true parameter"

WRONG interpretation:
   "The probability that the parameter is in this interval is 95%"
   (The parameter is a fixed value, not probabilistic)

[Practical Tips]
1. Choosing confidence level:
   - Exploratory analysis -> 90%
   - General research -> 95% (standard)
   - Pharmaceuticals/safety -> 99%

2. Mean difference estimation:
   - Equal variance -> ci_mean_diff_pooled()
   - Unknown/unequal variance -> ci_mean_diff_welch() (safer)
   - Paired data -> ci_mean_diff_paired()

3. Sample size design (MOE method):
   - Used in exit polls, opinion surveys
   - Proportion: sample_size_for_moe_proportion(moe, conf, p)
   - Mean: sample_size_for_moe_mean(moe, sigma, conf)
   - Doubling precision -> 4x sample size needed
   - For proportions, assuming p=0.5 is conservative

   Typical election survey precision:
   - +/-1%: ~9604 people (large national survey)
   - +/-2%: ~2401 people (national/close race)
   - +/-3%: ~1068 people (standard exit poll)
   - +/-5%: ~385 people (local election)

[Statistical Significance Using CI]
Determination using confidence intervals:
- CI does NOT contain 0 -> Statistically significant
- CI contains 0 -> Not statistically significant

Example: Mean diff 95% CI [2.3, 5.7] -> Significant (doesn't contain 0)
         Mean diff 95% CI [-1.2, 3.4] -> Not significant (contains 0)
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_ci_concept();
    example_standard_error();
    example_ci_mean();
    example_ci_proportion();
    example_ci_mean_diff();
    example_ci_variance();
    example_sample_size();

    // Display summary
    print_summary();

    return 0;
}
