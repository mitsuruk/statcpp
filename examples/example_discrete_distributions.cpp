/**
 * @file example_discrete_distributions.cpp
 * @brief Sample code for statcpp::discrete_distributions.hpp
 *
 * This file explains the usage of discrete probability distribution
 * functions provided in discrete_distributions.hpp through practical examples.
 *
 * [Provided Distributions]
 * - Binomial Distribution    : Binomial distribution
 * - Poisson Distribution     : Poisson distribution
 * - Bernoulli Distribution   : Bernoulli distribution
 * - Discrete Uniform         : Discrete uniform distribution
 * - Geometric Distribution   : Geometric distribution
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_discrete_distributions.cpp -o example_discrete_distributions
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/discrete_distributions.hpp"

// ============================================================================
// Helper Functions for Display
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
// 1. Binomial Distribution
// ============================================================================

/**
 * @brief Example usage of Binomial distribution
 *
 * [Concept]
 * Distribution of the number of successes in n independent trials,
 * each with success probability p
 *
 * [Formula]
 * P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
 *
 * [Parameters]
 * - n: Number of trials (positive integer)
 * - p: Success probability for each trial (0 <= p <= 1)
 * - k: Number of successes (0 <= k <= n)
 *
 * [Use Cases]
 * - Coin flips (probability of getting k heads in n flips)
 * - Quality control (probability of k defectives in n items)
 * - A/B testing (probability of k users clicking out of n)
 * - Election prediction (k voters out of n voting for a candidate)
 */
void example_binomial() {
    print_section("1. Binomial Distribution");

    std::cout << R"(
[Concept]
Distribution of the number of successes in n independent trials,
each with success probability p

[Example: Coin Flips]
Flip a fair coin (probability of heads p=0.5) 10 times
-> How many heads will appear?
)";

    int n = 10;
    double p = 0.5;

    print_subsection("Probability Mass Function (PMF)");
    std::cout << "P(X=5 | n=10, p=0.5) = " << statcpp::binomial_pmf(5, n, p) << "\n";
    std::cout << "-> Probability of exactly 5 heads in 10 flips\n";

    print_subsection("Cumulative Distribution Function (CDF)");
    std::cout << "P(X<=5 | n=10, p=0.5) = " << statcpp::binomial_cdf(5, n, p) << "\n";
    std::cout << "-> Probability of 5 or fewer heads in 10 flips\n";

    print_subsection("Quantile");
    int median = statcpp::binomial_quantile(0.5, n, p);
    std::cout << "Median (0.5 quantile) = " << median << "\n";
    std::cout << "-> Minimum number of successes where probability exceeds 50%\n";

    print_subsection("Random Number Generation");
    statcpp::set_seed(42);
    std::cout << "Random samples: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << statcpp::binomial_rand(n, p) << " ";
    }
    std::cout << "\n-> Simulating number of heads when flipping 10 times\n";

    print_subsection("Practical Example: Quality Control");
    std::cout << R"(
Manufacturing line produces 1000 products. Defect rate is 2%.
When sampling 100 items, what is the probability of 3 or more defectives?
)";
    int sample_size = 100;
    double defect_rate = 0.02;
    double prob_3_or_more = 1.0 - statcpp::binomial_cdf(2, sample_size, defect_rate);
    std::cout << "P(X>=3) = 1 - P(X<=2) = " << prob_3_or_more << "\n";
    std::cout << "-> About " << prob_3_or_more * 100 << "% probability of 3 or more defectives\n";
}

// ============================================================================
// 2. Poisson Distribution
// ============================================================================

/**
 * @brief Example usage of Poisson distribution
 *
 * [Concept]
 * Distribution of the number of random events occurring in
 * a fixed interval of time or space
 *
 * [Formula]
 * P(X = k) = (lambda^k * e^(-lambda)) / k!
 *
 * [Parameters]
 * - lambda: Average number of occurrences (lambda > 0)
 *
 * [Use Cases]
 * - Number of customers per hour
 * - Number of system failures per day
 * - Number of website accesses per minute
 * - Annual occurrence of natural disasters
 */
void example_poisson() {
    print_section("2. Poisson Distribution");

    std::cout << R"(
[Concept]
Distribution of the number of random events occurring in
a fixed interval of time or space

[Example: Call Center]
A call center receives an average of 3 inquiries per hour
-> How many calls will come in the next hour?
)";

    double lambda = 3.0;

    print_subsection("Probability Mass Function (PMF)");
    std::cout << "P(X=3 | lambda=3.0) = " << statcpp::poisson_pmf(3, lambda) << "\n";
    std::cout << "-> Probability of exactly 3 calls\n";

    std::cout << "\nP(X=0 | lambda=3.0) = " << statcpp::poisson_pmf(0, lambda) << "\n";
    std::cout << "-> Probability of zero calls\n";

    print_subsection("Cumulative Distribution Function (CDF)");
    std::cout << "P(X<=5 | lambda=3.0) = " << statcpp::poisson_cdf(5, lambda) << "\n";
    std::cout << "-> Probability of 5 or fewer calls\n";

    print_subsection("Quantile");
    int p95 = statcpp::poisson_quantile(0.95, lambda);
    std::cout << "95th percentile = " << p95 << "\n";
    std::cout << "-> 95% probability of this many or fewer calls\n";

    print_subsection("Random Number Generation");
    statcpp::set_seed(42);
    std::cout << "Calls per hour (5 hours): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << statcpp::poisson_rand(lambda) << " ";
    }
    std::cout << "\n";

    print_subsection("Practical Example: Server Load Prediction");
    std::cout << R"(
Web server receives average 10 requests per minute (lambda=10)
What is the probability of 15 or more requests in one minute?
)";
    double server_lambda = 10.0;
    double prob_15_or_more = 1.0 - statcpp::poisson_cdf(14, server_lambda);
    std::cout << "P(X>=15) = 1 - P(X<=14) = " << prob_15_or_more << "\n";
    std::cout << "-> About " << prob_15_or_more * 100 << "% probability of high load\n";
}

// ============================================================================
// 3. Bernoulli Distribution
// ============================================================================

/**
 * @brief Example usage of Bernoulli distribution
 *
 * [Concept]
 * Distribution where only success (1) or failure (0) occurs in a single trial
 * Special case of Binomial distribution with n=1
 *
 * [Formula]
 * P(X = 1) = p
 * P(X = 0) = 1-p
 *
 * [Parameters]
 * - p: Success probability (0 <= p <= 1)
 *
 * [Use Cases]
 * - Single coin flip
 * - Whether a user clicks a button or not
 * - Whether a single product is good or defective
 * - Yes/No binary choices
 */
void example_bernoulli() {
    print_section("3. Bernoulli Distribution");

    std::cout << R"(
[Concept]
Distribution where only success (1) or failure (0) occurs in a single trial

[Example: Ad Click]
Ad click rate is 70% (p=0.7)
-> Will the next user click?
)";

    double p = 0.7;

    print_subsection("Probability Mass Function (PMF)");
    std::cout << "P(X=1 | p=0.7) = " << statcpp::bernoulli_pmf(1, p) << " (clicks)\n";
    std::cout << "P(X=0 | p=0.7) = " << statcpp::bernoulli_pmf(0, p) << " (does not click)\n";

    print_subsection("Cumulative Distribution Function (CDF)");
    std::cout << "P(X<=0 | p=0.7) = " << statcpp::bernoulli_cdf(0, p) << "\n";
    std::cout << "-> Probability of not clicking\n";

    print_subsection("Quantile");
    int median = statcpp::bernoulli_quantile(0.5, p);
    std::cout << "Median (0.5 quantile) = " << median << "\n";

    print_subsection("Random Number Generation");
    statcpp::set_seed(42);
    std::cout << "Click results for 10 users (1=click, 0=no click):\n   ";
    int click_count = 0;
    for (int i = 0; i < 10; ++i) {
        int result = statcpp::bernoulli_rand(p);
        std::cout << result << " ";
        click_count += result;
    }
    std::cout << "\n-> " << click_count << " out of 10 users clicked\n";
}

// ============================================================================
// 4. Discrete Uniform Distribution
// ============================================================================

/**
 * @brief Example usage of Discrete Uniform distribution
 *
 * [Concept]
 * Distribution where all integers from a to b occur with equal probability
 *
 * [Formula]
 * P(X = k) = 1 / (b - a + 1)  (a <= k <= b)
 *
 * [Parameters]
 * - a: Minimum value (integer)
 * - b: Maximum value (integer, a <= b)
 *
 * [Use Cases]
 * - Rolling a die (1-6)
 * - Selecting a random integer
 * - Lottery drawing
 * - Random sampling
 */
void example_discrete_uniform() {
    print_section("4. Discrete Uniform Distribution");

    std::cout << R"(
[Concept]
Distribution where all values occur with equal probability

[Example: Rolling a Die]
A fair 6-sided die (1-6 with equal probability)
)";

    int a = 1, b = 6;

    print_subsection("Probability Mass Function (PMF)");
    std::cout << "P(X=3 | a=1, b=6) = " << statcpp::discrete_uniform_pmf(3, a, b) << "\n";
    std::cout << "-> All faces have equal probability (1/6 = " << 1.0/6.0 << ")\n";

    print_subsection("Cumulative Distribution Function (CDF)");
    std::cout << "P(X<=3 | a=1, b=6) = " << statcpp::discrete_uniform_cdf(3, a, b) << "\n";
    std::cout << "-> Probability of rolling 3 or less (3/6 = 0.5)\n";

    print_subsection("Quantile");
    int median = statcpp::discrete_uniform_quantile(0.5, a, b);
    std::cout << "Median (0.5 quantile) = " << median << "\n";

    print_subsection("Random Number Generation");
    statcpp::set_seed(42);
    std::cout << "Rolling die 10 times: ";
    std::vector<int> counts(7, 0);
    for (int i = 0; i < 10; ++i) {
        int result = statcpp::discrete_uniform_rand(a, b);
        std::cout << result << " ";
        counts[result]++;
    }
    std::cout << "\n\nFrequency distribution:\n";
    for (int i = 1; i <= 6; ++i) {
        std::cout << "   Face " << i << ": " << counts[i] << " times\n";
    }
}

// ============================================================================
// 5. Geometric Distribution
// ============================================================================

/**
 * @brief Example usage of Geometric distribution
 *
 * [Concept]
 * Distribution of the number of trials needed until the first success
 *
 * [Formula]
 * P(X = k) = (1-p)^(k-1) * p
 *
 * [Parameters]
 * - p: Success probability for each trial (0 < p <= 1)
 * - k: Number of trials (k >= 0 or k >= 1, depending on implementation)
 *
 * [Use Cases]
 * - Number of trials until first success
 * - Number of uses until machine failure
 * - Number of visits until first purchase
 * - Time until system error occurs
 *
 * [Memoryless Property]
 * Geometric distribution has the "memoryless property":
 * Past failures do not change the probability of success on the next trial
 */
void example_geometric() {
    print_section("5. Geometric Distribution");

    std::cout << R"(
[Concept]
Distribution of the number of trials needed until the first success

[Example: Basketball Free Throw]
A shooter with success probability p=0.3,
how many throws are needed until the first success?
)";

    double p = 0.3;

    print_subsection("Probability Mass Function (PMF)");
    std::cout << "P(X=2 | p=0.3) = " << statcpp::geometric_pmf(2, p) << "\n";
    std::cout << "-> Probability of first success on 2nd trial\n";
    std::cout << "   (1st miss, 2nd success)\n";

    std::cout << "\nP(X=1 | p=0.3) = " << statcpp::geometric_pmf(1, p) << "\n";
    std::cout << "-> Probability of success on 1st trial\n";

    print_subsection("Cumulative Distribution Function (CDF)");
    std::cout << "P(X<=5 | p=0.3) = " << statcpp::geometric_cdf(5, p) << "\n";
    std::cout << "-> Probability of success within 5 trials\n";

    print_subsection("Quantile");
    int median = statcpp::geometric_quantile(0.5, p);
    std::cout << "Median (0.5 quantile) = " << median << "\n";
    std::cout << "-> 50% probability of success within this many trials\n";

    print_subsection("Random Number Generation");
    statcpp::set_seed(42);
    std::cout << "5 simulations (trials until first success): ";
    double total = 0;
    for (int i = 0; i < 5; ++i) {
        int trials = statcpp::geometric_rand(p);
        std::cout << trials << " ";
        total += trials;
    }
    std::cout << "\nMean: " << total / 5.0 << " trials\n";
    std::cout << "Theoretical expected value: " << 1.0 / p << " trials\n";

    print_subsection("Practical Example: Customer Support");
    std::cout << R"(
Support center phone connection probability is 20% (p=0.2)
What is the probability of connecting within 10 calls?
)";
    double support_p = 0.2;
    double prob_within_10 = statcpp::geometric_cdf(10, support_p);
    std::cout << "P(X<=10) = " << prob_within_10 << "\n";
    std::cout << "-> About " << prob_within_10 * 100 << "% probability of connecting within 10 calls\n";
}

// ============================================================================
// 6. Distribution Comparison and Selection
// ============================================================================

void example_comparison() {
    print_section("6. Discrete Distribution Comparison and Selection");

    std::cout << R"(
+-----------------+------------------------------------------------+
| Distribution    | Use Case                                       |
+-----------------+------------------------------------------------+
| Bernoulli       | Single trial (success/failure)                 |
|                 | Ex: Single coin flip, single click decision    |
+-----------------+------------------------------------------------+
| Binomial        | Number of successes in n independent trials    |
|                 | Ex: Number of heads in n coin flips            |
+-----------------+------------------------------------------------+
| Geometric       | Number of trials until first success           |
|                 | Ex: Coin flips until first head appears        |
+-----------------+------------------------------------------------+
| Poisson         | Random event count in fixed time/space         |
|                 | Ex: Customers per hour, errors per day         |
+-----------------+------------------------------------------------+
| Discrete        | All values with equal probability              |
| Uniform         | Ex: Rolling a die, lottery drawing             |
+-----------------+------------------------------------------------+

[Choosing Between Binomial and Poisson]

Use Binomial when:
- Number of trials n is clear
- Success probability p for each trial is known
- n is small to moderate

Use Poisson when:
- Number of trials is very large (n -> infinity)
- Success probability is very small (p -> 0)
- np = lambda is constant
- Modeling "rare events"

[Approximation Relationship]
When n is large and p is small, Binomial(n, p) ~ Poisson(lambda=np)
)";

    print_subsection("Example: Binomial -> Poisson Approximation");
    int n = 1000;
    double p = 0.003;  // Small probability
    double lambda = n * p;  // lambda = 3.0

    int k = 5;
    double binomial_prob = statcpp::binomial_pmf(k, n, p);
    double poisson_prob = statcpp::poisson_pmf(k, lambda);

    std::cout << "For n=1000, p=0.003, k=5:\n";
    std::cout << "  Binomial PMF: " << binomial_prob << "\n";
    std::cout << "  Poisson PMF (lambda=3): " << poisson_prob << "\n";
    std::cout << "  Relative error: " << std::abs(binomial_prob - poisson_prob) / binomial_prob * 100 << "%\n";
    std::cout << "\n-> When n is large and p is small, Poisson is a good approximation\n";
}

// ============================================================================
// Summary
// ============================================================================

void print_summary() {
    print_section("Summary: Discrete Distribution Functions");

    std::cout << R"(
[Functions Common to Each Distribution]
- XXX_pmf(k, params...)      : Probability Mass Function P(X=k)
- XXX_cdf(k, params...)      : Cumulative Distribution Function P(X<=k)
- XXX_quantile(p, params...) : Quantile (inverse CDF)
- XXX_rand(params...)        : Random number generation

[Important Concepts]
1. PMF (Probability Mass Function): Probability of specific value k
2. CDF (Cumulative Distribution Function): Cumulative probability up to k
3. Quantile: k value corresponding to specified probability p
4. Expected value E[X] and Variance Var[X] differ by distribution

[Practical Tips]
- Small data, clear trial count -> Binomial
- Large scale, rare events -> Poisson
- Trials until first success -> Geometric
- Equal probability discrete selection -> Discrete Uniform
- Single trial -> Bernoulli

[Applications to Statistical Inference]
These distributions are the foundation for estimation and testing:
- Binomial test
- Poisson test
- Chi-squared goodness of fit test

See estimation.hpp, parametric_tests.hpp for details.
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_binomial();
    example_poisson();
    example_bernoulli();
    example_discrete_uniform();
    example_geometric();
    example_comparison();

    // Display summary
    print_summary();

    return 0;
}
