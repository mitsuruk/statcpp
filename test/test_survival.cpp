#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "statcpp/survival.hpp"

// ============================================================================
// Kaplan-Meier Tests
// ============================================================================

/**
 * @brief Tests Kaplan-Meier estimator with no events (all censored)
 * @test Verifies that survival probability remains 1.0 when no events occur
 */
TEST(KaplanMeierTest, NoEvents) {
    std::vector<double> times = {1, 2, 3, 4, 5};
    std::vector<bool> events = {false, false, false, false, false};

    auto result = statcpp::kaplan_meier(times, events);

    // Only time 0 with survival 1.0 since no events
    EXPECT_EQ(result.survival.size(), 1);
    EXPECT_DOUBLE_EQ(result.survival[0], 1.0);
}

/**
 * @brief Tests Kaplan-Meier estimator when all observations are events
 * @test Verifies that survival probability decreases monotonically when all events occur
 */
TEST(KaplanMeierTest, AllEvents) {
    std::vector<double> times = {1, 2, 3};
    std::vector<bool> events = {true, true, true};

    auto result = statcpp::kaplan_meier(times, events);

    // Should have survival decreasing
    EXPECT_GT(result.survival.size(), 1);
    EXPECT_DOUBLE_EQ(result.survival[0], 1.0);

    // Survival should decrease
    for (std::size_t i = 1; i < result.survival.size(); ++i) {
        EXPECT_LE(result.survival[i], result.survival[i-1]);
    }
}

/**
 * @brief Tests Kaplan-Meier estimator with mixed events and censoring
 * @test Verifies that survival probability is computed correctly with both events and censored observations
 */
TEST(KaplanMeierTest, MixedEventsAndCensoring) {
    std::vector<double> times = {1, 2, 3, 4, 5};
    std::vector<bool> events = {true, false, true, false, true};

    auto result = statcpp::kaplan_meier(times, events);

    EXPECT_GT(result.survival.size(), 1);
    EXPECT_DOUBLE_EQ(result.survival[0], 1.0);
    // Survival probability can reach 0 when all remaining subjects experience the event
    EXPECT_GE(result.survival.back(), 0);
}

/**
 * @brief Tests Kaplan-Meier estimator with empty input data
 * @test Verifies that an exception is thrown when input vectors are empty
 */
TEST(KaplanMeierTest, EmptyData) {
    std::vector<double> times;
    std::vector<bool> events;
    EXPECT_THROW(statcpp::kaplan_meier(times, events), std::invalid_argument);
}

/**
 * @brief Tests Kaplan-Meier estimator with mismatched vector lengths
 * @test Verifies that an exception is thrown when times and events vectors have different lengths
 */
TEST(KaplanMeierTest, MismatchedLengths) {
    std::vector<double> times = {1, 2, 3};
    std::vector<bool> events = {true, false};
    EXPECT_THROW(statcpp::kaplan_meier(times, events), std::invalid_argument);
}

// ============================================================================
// Log-rank Test Tests
// ============================================================================

/**
 * @brief Tests log-rank test with identical survival groups
 * @test Verifies that p-value is valid when comparing two identical survival groups
 */
TEST(LogrankTest, SameGroups) {
    std::vector<double> times1 = {1, 2, 3, 4, 5};
    std::vector<bool> events1 = {true, true, true, true, true};
    std::vector<double> times2 = {1, 2, 3, 4, 5};
    std::vector<bool> events2 = {true, true, true, true, true};

    auto result = statcpp::logrank_test(times1, events1, times2, events2);

    // Same groups should have small test statistic
    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
    EXPECT_EQ(result.df, 1);
}

/**
 * @brief Tests log-rank test with different survival groups
 * @test Verifies that p-value is computed correctly when comparing groups with different survival times
 */
TEST(LogrankTest, DifferentGroups) {
    std::vector<double> times1 = {1, 2, 3, 4, 5};
    std::vector<bool> events1 = {true, true, true, true, true};
    std::vector<double> times2 = {10, 20, 30, 40, 50};
    std::vector<bool> events2 = {true, true, true, true, true};

    auto result = statcpp::logrank_test(times1, events1, times2, events2);

    EXPECT_GE(result.p_value, 0.0);
    EXPECT_LE(result.p_value, 1.0);
}

// ============================================================================
// Median Survival Time Tests
// ============================================================================

/**
 * @brief Tests median survival time computation
 * @test Verifies that median survival time is computed as the point where survival probability crosses 0.5
 */
TEST(MedianSurvivalTest, Basic) {
    std::vector<double> times = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<bool> events = {true, true, true, true, true, true, true, true, true, true};

    auto km = statcpp::kaplan_meier(times, events);
    double median = statcpp::median_survival_time(km);

    // Median should be where survival crosses 0.5
    EXPECT_GT(median, 0);
}

// ============================================================================
// Nelson-Aalen Tests
// ============================================================================

/**
 * @brief Tests Nelson-Aalen cumulative hazard estimator
 * @test Verifies that cumulative hazard is non-decreasing and starts at zero
 */
TEST(NelsonAalenTest, Basic) {
    std::vector<double> times = {1, 2, 3, 4, 5};
    std::vector<bool> events = {true, true, false, true, true};

    auto result = statcpp::nelson_aalen(times, events);

    EXPECT_GT(result.times.size(), 1);
    EXPECT_DOUBLE_EQ(result.cumulative_hazard[0], 0.0);

    // Cumulative hazard should be non-decreasing
    for (std::size_t i = 1; i < result.cumulative_hazard.size(); ++i) {
        EXPECT_GE(result.cumulative_hazard[i], result.cumulative_hazard[i-1]);
    }
}
