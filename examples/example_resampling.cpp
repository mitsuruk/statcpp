/**
 * @file example_resampling.cpp
 * @brief Sample code for resampling methods
 *
 * Demonstrates usage examples of resampling methods including
 * bootstrap, jackknife, permutation tests, and cross-validation.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include "statcpp/resampling.hpp"
#include "statcpp/basic_statistics.hpp"
#include "statcpp/dispersion_spread.hpp"

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=============================================================\n";
    std::cout << "          statcpp Resampling Methods Examples               \n";
    std::cout << "=============================================================\n\n";

    std::cout << "What are Resampling Methods?\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Resampling methods generate repeated samples from observed data\n";
    std::cout << "to estimate the distribution of statistics.\n\n";
    std::cout << "Key advantages:\n";
    std::cout << "  * No theoretical distribution assumptions required\n";
    std::cout << "  * Applicable to complex statistics\n";
    std::cout << "  * Enables inference with finite samples\n\n";

    std::cout << "=============================================================\n";
    std::cout << "1. Bootstrap Method - Basic Concept\n";
    std::cout << "=============================================================\n\n";

    std::cout << "How Bootstrap Works:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Original data: [12, 13, 11, 13, 14]  (n=5)\n\n";
    std::cout << "Generate samples of the same size with replacement:\n";
    std::cout << "  Bootstrap Sample 1: [12, 12, 13, 11, 14]  -> Mean: 12.4\n";
    std::cout << "  Bootstrap Sample 2: [13, 14, 11, 13, 13]  -> Mean: 12.8\n";
    std::cout << "  Bootstrap Sample 3: [11, 12, 12, 13, 14]  -> Mean: 12.4\n";
    std::cout << "  ...(repeat)\n";
    std::cout << "  Bootstrap Sample B: [13, 13, 14, 12, 13]  -> Mean: 13.0\n\n";
    std::cout << "Estimate standard error and confidence intervals from\n";
    std::cout << "the distribution of these statistics\n\n";

    // Sample data
    std::vector<double> data = {12.5, 13.2, 11.8, 12.9, 13.5, 12.1, 13.8, 12.7};

    std::cout << "=============================================================\n";
    std::cout << "2. Bootstrap Method - Confidence Interval for Mean\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Real data example: Battery lifetime (hours)\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Measurements: ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    statcpp::set_seed(123);  // For reproducibility

    auto boot_mean = statcpp::bootstrap_mean(
        data.begin(), data.end(),
        1000,  // number of bootstrap samples
        0.95   // confidence level
    );

    double original_mean = statcpp::mean(data.begin(), data.end());

    std::cout << "Original data mean:     " << original_mean << " hours\n";
    std::cout << "Bootstrap estimate:     " << boot_mean.estimate << " hours\n";
    std::cout << "Bootstrap std. error:   " << boot_mean.standard_error << " hours\n";
    std::cout << "95% confidence interval: [" << boot_mean.ci_lower << ", " << boot_mean.ci_upper << "] hours\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "  The true population mean is estimated to be in the range [" << boot_mean.ci_lower << ", "
              << boot_mean.ci_upper << "] with 95% confidence.\n\n";

    std::cout << "Meaning of Bootstrap Standard Error:\n";
    std::cout << "  It is a measure of the variability of the sample mean.\n";
    std::cout << "  If the same experiment were repeated, the mean would vary\n";
    std::cout << "  within approximately +-" << boot_mean.standard_error << " hours.\n\n";

    std::cout << "=============================================================\n";
    std::cout << "3. Bootstrap Method - Confidence Interval for Median\n";
    std::cout << "=============================================================\n\n";

    std::cout << "The median is a statistic robust to outliers.\n";
    std::cout << "Traditional theoretical methods make it difficult to compute\n";
    std::cout << "confidence intervals for the median, but bootstrap makes it easy.\n\n";

    std::cout << "Data example with outlier: Monthly overtime hours\n";
    std::cout << "-------------------------------------------------------------\n";
    std::vector<double> overtime = {5.2, 6.1, 4.8, 5.9, 6.5, 5.3, 32.7, 6.2};
    std::cout << "Overtime hours: ";
    for (size_t i = 0; i < overtime.size(); ++i) {
        std::cout << overtime[i];
        if (i < overtime.size() - 1) std::cout << ", ";
    }
    std::cout << " hours\n";
    std::cout << "  * 32.7 hours is an outlier (unusual case)\n\n";

    statcpp::set_seed(456);
    auto boot_median = statcpp::bootstrap_median(
        overtime.begin(), overtime.end(),
        1000,
        0.95
    );

    double original_median = statcpp::median(overtime.begin(), overtime.end());
    double overtime_mean = statcpp::mean(overtime.begin(), overtime.end());

    std::cout << "Mean:                  " << overtime_mean << " hours (heavily affected by outlier)\n";
    std::cout << "Median:                " << original_median << " hours (robust)\n\n";
    std::cout << "Bootstrap median:      " << boot_median.estimate << " hours\n";
    std::cout << "Bootstrap std. error:  " << boot_median.standard_error << " hours\n";
    std::cout << "95% confidence interval: [" << boot_median.ci_lower << ", " << boot_median.ci_upper << "] hours\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "  Even with the outlier (32.7 hours), the median remains stable\n";
    std::cout << "  at approximately " << original_median << " hours.\n";
    std::cout << "  The mean (" << overtime_mean << " hours) is heavily influenced by the outlier.\n";
    std::cout << "  In such cases, the median is more appropriate as a representative value.\n\n";

    std::cout << "=============================================================\n";
    std::cout << "4. Bootstrap Method - Confidence Interval for Standard Deviation\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Quality control example: Product weight variability\n";
    std::cout << "-------------------------------------------------------------\n";
    std::vector<double> weights = {100.2, 99.8, 100.5, 99.9, 100.3, 100.1, 99.7, 100.4};
    std::cout << "Product weights: ";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i];
        if (i < weights.size() - 1) std::cout << ", ";
    }
    std::cout << " g\n\n";

    statcpp::set_seed(789);
    auto boot_sd = statcpp::bootstrap_stddev(
        weights.begin(), weights.end(),
        1000,
        0.95
    );

    double original_sd = statcpp::stddev(weights.begin(), weights.end());

    std::cout << "Sample std. deviation: " << original_sd << " g\n";
    std::cout << "Bootstrap estimate:    " << boot_sd.estimate << " g\n";
    std::cout << "Bootstrap std. error:  " << boot_sd.standard_error << " g\n";
    std::cout << "95% confidence interval: [" << boot_sd.ci_lower << ", " << boot_sd.ci_upper << "] g\n\n";

    std::cout << "Application to quality control:\n";
    std::cout << "  Tolerance range: +-0.5g (99.5g ~ 100.5g)\n";
    std::cout << "  Current variability (sigma): " << boot_sd.estimate << " g\n";
    std::cout << "  If 99.5% should be within tolerance, required sigma is about 0.15g\n";
    std::cout << "  -> Current process is within acceptable range\n\n";

    std::cout << "=============================================================\n";
    std::cout << "5. Permutation Test - Basic Concept\n";
    std::cout << "=============================================================\n\n";

    std::cout << "What is a Permutation Test?\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "If the null hypothesis 'no difference between groups' is true,\n";
    std::cout << "swapping data labels should not change the distribution of statistics.\n\n";
    std::cout << "Example: Testing new drug efficacy\n";
    std::cout << "  Placebo group: [12, 13, 11, 13]  Mean: 12.25\n";
    std::cout << "  Drug group:    [14, 15, 15, 14]  Mean: 14.50\n";
    std::cout << "  Observed diff: 14.50 - 12.25 = 2.25\n\n";
    std::cout << "Permutation examples:\n";
    std::cout << "  Perm 1: [12,13,15,14] vs [11,13,14,15]  Diff: 0.00\n";
    std::cout << "  Perm 2: [12,15,11,14] vs [13,13,14,15]  Diff: 0.75\n";
    std::cout << "  Perm 3: [14,13,15,13] vs [12,11,14,15]  Diff: -0.25\n";
    std::cout << "  ...\n\n";
    std::cout << "Compare the observed difference with the distribution of\n";
    std::cout << "differences generated by permutations to calculate the p-value.\n\n";

    std::cout << "=============================================================\n";
    std::cout << "6. Permutation Test - Testing Mean Difference Between Two Groups\n";
    std::cout << "=============================================================\n\n";

    std::cout << "A/B test example: Testing new UI effectiveness\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Question: Did the new UI design change user session duration?\n\n";

    std::vector<double> old_ui = {12.5, 13.2, 11.8, 12.9, 13.1, 12.3};
    std::vector<double> new_ui = {14.1, 15.3, 14.8, 15.0, 14.5, 14.9};

    std::cout << "Old UI session time (min): ";
    for (size_t i = 0; i < old_ui.size(); ++i) {
        std::cout << old_ui[i];
        if (i < old_ui.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "New UI session time (min): ";
    for (size_t i = 0; i < new_ui.size(); ++i) {
        std::cout << new_ui[i];
        if (i < new_ui.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    double mean_old = statcpp::mean(old_ui.begin(), old_ui.end());
    double mean_new = statcpp::mean(new_ui.begin(), new_ui.end());
    double observed_diff = mean_new - mean_old;

    std::cout << "Old UI mean:        " << mean_old << " min\n";
    std::cout << "New UI mean:        " << mean_new << " min\n";
    std::cout << "Observed difference: " << observed_diff << " min\n\n";

    statcpp::set_seed(321);
    auto perm_result = statcpp::permutation_test_two_sample(
        old_ui.begin(), old_ui.end(),
        new_ui.begin(), new_ui.end(),
        1000
    );

    std::cout << "Permutation test results:\n";
    std::cout << "  p-value: " << perm_result.p_value << "\n\n";

    std::cout << "Interpretation:\n";
    if (perm_result.p_value < 0.05) {
        std::cout << "  p-value < 0.05 -> Statistically significant\n";
        std::cout << "  The new UI shows a significant difference in session time\n";
        std::cout << "  compared to the old UI. Recommend implementing the new UI.\n\n";
    } else {
        std::cout << "  p-value >= 0.05 -> Not statistically significant\n";
        std::cout << "  While there appears to be a difference, we cannot rule out\n";
        std::cout << "  that it occurred by chance. More data collection is needed.\n\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "7. Permutation Test - Verification with Stricter Criteria\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Pharmaceutical clinical trial example\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Pharmaceuticals often use strict criteria (alpha = 0.01)\n\n";

    std::vector<double> placebo = {78.2, 79.5, 77.8, 78.9, 79.2, 78.5, 79.1};
    std::vector<double> drug = {82.1, 83.4, 81.9, 82.7, 83.0, 82.3, 82.8};

    std::cout << "Placebo group recovery rate (%): ";
    for (size_t i = 0; i < placebo.size(); ++i) {
        std::cout << placebo[i];
        if (i < placebo.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "Drug group recovery rate (%):    ";
    for (size_t i = 0; i < drug.size(); ++i) {
        std::cout << drug[i];
        if (i < drug.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    double mean_placebo = statcpp::mean(placebo.begin(), placebo.end());
    double mean_drug = statcpp::mean(drug.begin(), drug.end());
    double drug_diff = mean_drug - mean_placebo;

    std::cout << "Placebo group mean: " << mean_placebo << " %\n";
    std::cout << "Drug group mean:    " << mean_drug << " %\n";
    std::cout << "Observed difference: " << drug_diff << " %\n\n";

    statcpp::set_seed(654);
    auto perm_drug = statcpp::permutation_test_two_sample(
        placebo.begin(), placebo.end(),
        drug.begin(), drug.end(),
        2000  // More permutations for better precision
    );

    std::cout << "Permutation test results (2000 permutations):\n";
    std::cout << "  p-value: " << perm_drug.p_value << "\n\n";

    std::cout << "Interpretation (alpha = 0.01):\n";
    if (perm_drug.p_value < 0.01) {
        std::cout << "  p-value < 0.01 -> Very strong statistical evidence\n";
        std::cout << "  The drug effect is statistically highly significant.\n";
        std::cout << "  Proceeding to the next phase of clinical trials is recommended.\n\n";
    } else if (perm_drug.p_value < 0.05) {
        std::cout << "  0.01 <= p-value < 0.05\n";
        std::cout << "  Significant by normal standards, but does not meet the strict\n";
        std::cout << "  pharmaceutical criterion (alpha=0.01). Additional trials needed.\n\n";
    } else {
        std::cout << "  p-value >= 0.05 -> Not statistically significant\n";
        std::cout << "  Could not confirm drug effectiveness.\n\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "8. When to Use Bootstrap vs Permutation Test\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Bootstrap Method (Confidence Interval Estimation):\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Purpose: Quantify uncertainty of statistics (mean, median, etc.)\n";
    std::cout << "Use cases:\n";
    std::cout << "  * Want confidence intervals for parameters\n";
    std::cout << "  * Want to know 'what range does the true value fall in?'\n";
    std::cout << "  * Want to estimate standard error\n";
    std::cout << "Examples:\n";
    std::cout << "  'Average satisfaction with new product is 7.2-8.5 points (95% CI)'\n";
    std::cout << "  'Median income is estimated between 45,000-52,000'\n\n";

    std::cout << "Permutation Test (Hypothesis Testing):\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "Purpose: Test whether there is a statistically significant difference\n";
    std::cout << "Use cases:\n";
    std::cout << "  * Want to verify 'is there a difference between A and B?'\n";
    std::cout << "  * Want to statistically determine presence of effect\n";
    std::cout << "  * Want to calculate p-value\n";
    std::cout << "Examples:\n";
    std::cout << "  'The new drug has a statistically significant effect (p<0.01)'\n";
    std::cout << "  'No significant difference observed between genders (p=0.34)'\n\n";

    std::cout << "=============================================================\n";
    std::cout << "9. Advantages and Disadvantages of Resampling Methods\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Advantages:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "[+] No distribution assumptions required\n";
    std::cout << "    -> No need to assume normality, etc.\n\n";
    std::cout << "[+] Applicable to complex statistics\n";
    std::cout << "    -> Can handle median, IQR, correlation coefficient, etc.\n\n";
    std::cout << "[+] Intuitive to understand\n";
    std::cout << "    -> Simple concept of 'repeatedly sampling from data'\n\n";
    std::cout << "[+] Enables inference with finite samples\n";
    std::cout << "    -> Does not rely on theoretical asymptotic properties\n\n";

    std::cout << "Disadvantages:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "[-] High computational cost\n";
    std::cout << "    -> Thousands of resamplings needed (usually not a problem with modern computers)\n\n";
    std::cout << "[-] Depends on original data quality\n";
    std::cout << "    -> If original data is biased, results will be biased\n\n";
    std::cout << "[-] Unstable with very small samples\n";
    std::cout << "    -> Reliability may decrease with n < 10\n\n";

    std::cout << "=============================================================\n";
    std::cout << "10. Practical Guidelines\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Choosing Number of Bootstrap Samples:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  * Exploratory analysis:     B = 200 - 500\n";
    std::cout << "  * Standard CI:              B = 1000 - 2000\n";
    std::cout << "  * Precise estimation:       B = 5000 - 10000\n";
    std::cout << "  * Papers/official reports:  B = 10000+\n\n";

    std::cout << "Choosing Number of Permutations:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  * Exploratory analysis:     N = 500 - 1000\n";
    std::cout << "  * Standard testing:         N = 1000 - 5000\n";
    std::cout << "  * Precise p-values:         N = 10000+\n";
    std::cout << "  * Use more for small p-values (p<0.01)\n\n";

    std::cout << "Choosing Confidence Level:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  * 90% CI: Preliminary analysis, screening\n";
    std::cout << "  * 95% CI: Standard scientific research (most common)\n";
    std::cout << "  * 99% CI: Important decisions, medical/safety related\n\n";

    std::cout << "Choosing Significance Level:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  * alpha = 0.10: Exploratory research\n";
    std::cout << "  * alpha = 0.05: Standard scientific research (most common)\n";
    std::cout << "  * alpha = 0.01: Pharmaceuticals, important policy decisions\n";
    std::cout << "  * alpha = 0.001: Extremely important decisions\n\n";

    std::cout << "Setting Random Seed:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  statcpp::set_seed(123);  // Essential for reproducibility\n";
    std::cout << "  Always document the seed value in papers and reports.\n\n";

    std::cout << "Sample Size Guidelines:\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "  * n < 10:    Resampling methods unstable, use with caution\n";
    std::cout << "  * n = 10-30: Bootstrap usable, but interpret carefully\n";
    std::cout << "  * n > 30:    Bootstrap works stably\n";
    std::cout << "  * n > 100:   Theoretical and bootstrap results nearly match\n\n";

    std::cout << "=============================================================\n";
    std::cout << "11. Common Mistakes and Correct Interpretations\n";
    std::cout << "=============================================================\n\n";

    std::cout << "[X] Mistake 1: '95% CI contains the true value with 95% probability'\n";
    std::cout << "[O] Correct:   'Repeating the procedure, 95% of intervals contain the true value'\n";
    std::cout << "    Note: The true value is fixed; it's the interval that varies.\n\n";

    std::cout << "[X] Mistake 2: 'p-value is the probability the hypothesis is correct'\n";
    std::cout << "[O] Correct:   'Probability of observing results as extreme or more,\n";
    std::cout << "               given the null hypothesis is true'\n";
    std::cout << "    Note: p-value is about data probability, not hypothesis probability.\n\n";

    std::cout << "[X] Mistake 3: 'p > 0.05 means no difference exists'\n";
    std::cout << "[O] Correct:   'p > 0.05 means insufficient evidence to show difference'\n";
    std::cout << "    Note: 'No difference' and 'cannot detect difference' are different.\n\n";

    std::cout << "[X] Mistake 4: 'Statistical significance = practical importance'\n";
    std::cout << "[O] Correct:   'Statistical significance and practical importance are different'\n";
    std::cout << "    Note: Large n can make small differences significant.\n";
    std::cout << "          Separate judgment needed for practical meaningfulness.\n\n";

    std::cout << "=============================================================\n";
    std::cout << "12. Summary - When to Use Which Method\n";
    std::cout << "=============================================================\n\n";

    std::cout << "Want confidence interval for the mean:\n";
    std::cout << "  -> Use bootstrap_mean()\n";
    std::cout << "    Example: 'What is the 95% CI for average satisfaction with new product?'\n\n";

    std::cout << "Want confidence interval for median (robust to outliers):\n";
    std::cout << "  -> Use bootstrap_median()\n";
    std::cout << "    Example: 'What is the confidence interval for median income?'\n\n";

    std::cout << "Want confidence interval for variability (standard deviation):\n";
    std::cout << "  -> Use bootstrap_stddev()\n";
    std::cout << "    Example: 'How much does product quality vary?'\n\n";

    std::cout << "Want to test for significant difference between two groups:\n";
    std::cout << "  -> Use permutation_test_two_sample()\n";
    std::cout << "    Example: 'Is there a difference in efficacy between new and old drugs?'\n\n";

    std::cout << "Don't want to make distribution assumptions:\n";
    std::cout << "  -> Use resampling methods (Bootstrap, Permutation test)\n";
    std::cout << "    No need to assume normality, etc.\n\n";

    std::cout << "Want to handle complex statistics (median, IQR, etc.):\n";
    std::cout << "  -> Use Bootstrap\n";
    std::cout << "    Effective when theoretical methods don't exist or are complex\n\n";

    std::cout << "=============================================================\n";
    std::cout << "All examples completed!\n";
    std::cout << "=============================================================\n";

    return 0;
}
