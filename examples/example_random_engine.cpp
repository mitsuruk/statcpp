/**
 * @file example_random_engine.cpp
 * @brief Sample code for random engines
 *
 * Demonstrates usage examples of the statcpp library's random engine
 * management features (seed setting, thread-local engine, reproducible
 * random number generation).
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <map>
#include "statcpp/random_engine.hpp"

int main() {
    std::cout << "=== Random Engine Examples ===" << std::endl;

    // ============================================================================
    // 1. Using the Default Random Engine
    // ============================================================================
    std::cout << "\n1. Using the Default Random Engine" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Default engine type: Mersenne Twister (mt19937_64)" << std::endl;

    // Generate uniform random numbers from the default engine
    auto& engine = statcpp::get_random_engine();
    std::cout << "\nGenerate 10 random numbers from the engine:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << (i + 1) << ": " << engine() << std::endl;
    }

    // ============================================================================
    // 2. Setting the Seed
    // ============================================================================
    std::cout << "\n2. Setting and Using the Seed" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Fix the seed for reproducibility
    statcpp::set_seed(42);
    std::cout << "Seed set to 42" << std::endl;

    std::cout << "\nFirst sequence (seed=42):" << std::endl;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // Generate again with the same seed
    statcpp::set_seed(42);
    std::cout << "\nSecond sequence (seed=42 reset):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    std::cout << "\nNote: Both sequences are identical because of the same seed." << std::endl;

    // ============================================================================
    // 3. Generation with Different Seeds
    // ============================================================================
    std::cout << "\n3. Different Seeds Produce Different Sequences" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(100);
    std::cout << "Sequence with seed=100:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    statcpp::set_seed(200);
    std::cout << "\nSequence with seed=200:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // ============================================================================
    // 4. Using Random Seed
    // ============================================================================
    std::cout << "\n4. Randomizing the Seed" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::randomize_seed();
    std::cout << "Seed randomized (non-reproducible):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(6) << uniform(engine) << std::endl;
    }

    // ============================================================================
    // 5. Using with Various Distributions
    // ============================================================================
    std::cout << "\n5. Using Random Engine with Various Distributions" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);  // Set seed for reproducibility

    // Uniform distribution [0, 10)
    std::cout << "\nUniform distribution [0, 10):" << std::endl;
    std::uniform_real_distribution<double> uniform_10(0.0, 10.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(4) << uniform_10(engine) << std::endl;
    }

    // Normal distribution N(0, 1)
    std::cout << "\nNormal distribution N(0, 1):" << std::endl;
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::fixed << std::setprecision(4) << normal(engine) << std::endl;
    }

    // Integer uniform distribution [1, 6] (dice)
    std::cout << "\nInteger uniform distribution [1, 6] (dice):" << std::endl;
    std::uniform_int_distribution<int> dice(1, 6);
    for (int i = 0; i < 10; ++i) {
        std::cout << "  Roll " << (i + 1) << ": " << dice(engine) << std::endl;
    }

    // Bernoulli distribution (coin toss)
    std::cout << "\nBernoulli distribution (coin flip, p=0.5):" << std::endl;
    std::bernoulli_distribution coin(0.5);
    int heads = 0, tails = 0;
    for (int i = 0; i < 100; ++i) {
        if (coin(engine)) {
            ++heads;
        } else {
            ++tails;
        }
    }
    std::cout << "  100 flips: Heads = " << heads << ", Tails = " << tails << std::endl;

    // ============================================================================
    // 6. Comparing Different Random Engines
    // ============================================================================
    std::cout << "\n6. Comparing Different Random Engines" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // mt19937 (32-bit Mersenne Twister)
    std::mt19937 mt32(42);
    std::cout << "mt19937 (32-bit): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << mt32() << " ";
    }
    std::cout << std::endl;

    // mt19937_64 (64-bit Mersenne Twister)
    std::mt19937_64 mt64(42);
    std::cout << "mt19937_64 (64-bit): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << mt64() << " ";
    }
    std::cout << std::endl;

    // minstd_rand (Linear Congruential)
    std::minstd_rand lcg(42);
    std::cout << "minstd_rand (LCG): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << lcg() << " ";
    }
    std::cout << std::endl;

    // ============================================================================
    // 7. Simple Random Quality Test
    // ============================================================================
    std::cout << "\n7. Simple Random Quality Test" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(12345);

    // Generate many uniform random numbers in [0, 1) and compute statistics
    const int n_samples = 100000;
    std::vector<double> samples;
    samples.reserve(n_samples);

    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    for (int i = 0; i < n_samples; ++i) {
        samples.push_back(unit_uniform(engine));
    }

    // Calculate mean (theoretical: 0.5)
    double sum = 0.0;
    for (double x : samples) {
        sum += x;
    }
    double mean = sum / n_samples;

    // Calculate variance (theoretical: 1/12 = 0.0833)
    double sum_sq = 0.0;
    for (double x : samples) {
        double diff = x - mean;
        sum_sq += diff * diff;
    }
    double variance = sum_sq / n_samples;

    std::cout << "Generated " << n_samples << " uniform [0,1) samples:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Mean (expected 0.5):      " << mean << std::endl;
    std::cout << "  Variance (expected 0.083): " << variance << std::endl;

    // Distribution uniformity check (divide into 10 bins)
    std::cout << "\nDistribution uniformity (10 bins):" << std::endl;
    std::map<int, int> bins;
    for (double x : samples) {
        int bin = static_cast<int>(x * 10.0);
        if (bin >= 10) bin = 9;  // Handle 1.0 case
        ++bins[bin];
    }

    for (int i = 0; i < 10; ++i) {
        double ratio = static_cast<double>(bins[i]) / n_samples;
        std::cout << "  [" << std::setw(3) << (i * 10) << "%-" << std::setw(3) << ((i + 1) * 10) << "%): "
                  << std::setw(6) << bins[i] << " (" << std::setw(5) << std::setprecision(2)
                  << (ratio * 100.0) << "%)" << std::endl;
    }
    std::cout << "  Expected: approximately 10% per bin" << std::endl;

    // ============================================================================
    // 8. Thread Safety Confirmation
    // ============================================================================
    std::cout << "\n8. Thread-Local Storage" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "The random engine is thread-local for thread safety." << std::endl;
    std::cout << "Each thread has its own independent random engine instance." << std::endl;

    // ============================================================================
    // 9. Using Type Traits
    // ============================================================================
    std::cout << "\n9. Random Engine Type Traits" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::cout << "Type trait checks:" << std::endl;
    std::cout << "  std::mt19937 is random engine: "
              << (statcpp::is_random_engine_v<std::mt19937> ? "Yes" : "No") << std::endl;
    std::cout << "  std::mt19937_64 is random engine: "
              << (statcpp::is_random_engine_v<std::mt19937_64> ? "Yes" : "No") << std::endl;
    std::cout << "  std::minstd_rand is random engine: "
              << (statcpp::is_random_engine_v<std::minstd_rand> ? "Yes" : "No") << std::endl;
    std::cout << "  int is random engine: "
              << (statcpp::is_random_engine_v<int> ? "Yes" : "No") << std::endl;

    // ============================================================================
    // 10. Practical Example: Estimating Pi with Monte Carlo Method
    // ============================================================================
    std::cout << "\n10. Practical Example: Estimating Pi with Monte Carlo Method" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);

    const int n_trials = 1000000;
    int inside_circle = 0;

    std::uniform_real_distribution<double> coord(-1.0, 1.0);

    for (int i = 0; i < n_trials; ++i) {
        double x = coord(engine);
        double y = coord(engine);
        if (x * x + y * y <= 1.0) {
            ++inside_circle;
        }
    }

    double pi_estimate = 4.0 * static_cast<double>(inside_circle) / n_trials;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Number of trials: " << n_trials << std::endl;
    std::cout << "Points inside circle: " << inside_circle << std::endl;
    std::cout << "Estimated pi: " << pi_estimate << std::endl;
    std::cout << "Actual pi:    " << 3.141593 << std::endl;
    std::cout << "Error:        " << std::abs(pi_estimate - 3.141593) << std::endl;

    std::cout << "\n=== Examples completed successfully ===" << std::endl;

    return 0;
}
