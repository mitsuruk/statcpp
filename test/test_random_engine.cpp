#include <gtest/gtest.h>
#include "statcpp/random_engine.hpp"
#include <vector>

// ============================================================================
// Random Engine Tests
// ============================================================================

/**
 * @brief Tests random engine reproducibility with same seed
 * @test Verifies that setting the same seed produces identical random number sequences
 */
TEST(RandomEngineTest, Reproducibility) {
    // Setting the same seed should produce the same sequence
    statcpp::set_seed(12345);
    std::vector<std::uint64_t> seq1;
    auto& engine = statcpp::get_random_engine();
    for (int i = 0; i < 10; ++i) {
        seq1.push_back(engine());
    }

    statcpp::set_seed(12345);
    std::vector<std::uint64_t> seq2;
    for (int i = 0; i < 10; ++i) {
        seq2.push_back(engine());
    }

    EXPECT_EQ(seq1, seq2);
}

/**
 * @brief Tests that different seeds produce different random values
 * @test Verifies that different seeds produce different random number sequences
 */
TEST(RandomEngineTest, DifferentSeeds) {
    statcpp::set_seed(111);
    auto& engine = statcpp::get_random_engine();
    std::uint64_t val1 = engine();

    statcpp::set_seed(222);
    std::uint64_t val2 = engine();

    // Different seeds should (almost certainly) produce different values
    EXPECT_NE(val1, val2);
}

/**
 * @brief Tests random seed initialization
 * @test Verifies that randomize_seed() initializes the random engine and produces non-zero values
 */
TEST(RandomEngineTest, RandomizeSeed) {
    // Just make sure it doesn't crash
    statcpp::randomize_seed();
    auto& engine = statcpp::get_random_engine();
    std::uint64_t val = engine();
    EXPECT_NE(val, 0); // Very unlikely to be 0
}

/**
 * @brief Tests random engine type trait detection
 * @test Verifies that is_random_engine_v correctly identifies random engine types
 */
TEST(IsRandomEngineTest, TypeTraits) {
    EXPECT_TRUE(statcpp::is_random_engine_v<std::mt19937>);
    EXPECT_TRUE(statcpp::is_random_engine_v<std::mt19937_64>);
    EXPECT_TRUE(statcpp::is_random_engine_v<std::minstd_rand>);
    EXPECT_FALSE(statcpp::is_random_engine_v<int>);
    EXPECT_FALSE(statcpp::is_random_engine_v<double>);
}
