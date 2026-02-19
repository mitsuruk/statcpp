#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "statcpp/categorical.hpp"

// ============================================================================
// Contingency Table Tests
// ============================================================================

/**
 * @brief Tests basic 2x2 contingency table construction
 * @test Verifies that contingency table correctly computes cell counts, row totals, and column totals
 */
TEST(ContingencyTableTest, Basic2x2) {
    std::vector<std::size_t> row = {0, 0, 0, 1, 1, 1};
    std::vector<std::size_t> col = {0, 0, 1, 0, 1, 1};

    auto result = statcpp::contingency_table(row, col);

    EXPECT_EQ(result.n_rows, 2);
    EXPECT_EQ(result.n_cols, 2);
    EXPECT_EQ(result.total, 6);

    // Expected table:
    //     col0  col1
    // row0  2    1
    // row1  1    2
    EXPECT_EQ(result.table[0][0], 2);
    EXPECT_EQ(result.table[0][1], 1);
    EXPECT_EQ(result.table[1][0], 1);
    EXPECT_EQ(result.table[1][1], 2);

    EXPECT_EQ(result.row_totals[0], 3);
    EXPECT_EQ(result.row_totals[1], 3);
    EXPECT_EQ(result.col_totals[0], 3);
    EXPECT_EQ(result.col_totals[1], 3);
}

/**
 * @brief Tests error handling for empty data in contingency table
 * @test Verifies that contingency table construction throws exception for empty input
 */
TEST(ContingencyTableTest, EmptyData) {
    std::vector<std::size_t> row;
    std::vector<std::size_t> col;
    EXPECT_THROW(statcpp::contingency_table(row, col), std::invalid_argument);
}

/**
 * @brief Tests error handling for mismatched vector lengths
 * @test Verifies that contingency table construction throws exception when row and column vectors have different lengths
 */
TEST(ContingencyTableTest, MismatchedLengths) {
    std::vector<std::size_t> row = {0, 1, 0};
    std::vector<std::size_t> col = {0, 1};
    EXPECT_THROW(statcpp::contingency_table(row, col), std::invalid_argument);
}

// ============================================================================
// Odds Ratio Tests
// ============================================================================

/**
 * @brief Tests basic odds ratio calculation from 2x2 table
 * @test Verifies that odds ratio, log odds ratio, and confidence intervals are correctly computed
 */
TEST(CategoricalOddsRatioTest, Basic) {
    // Standard 2x2 table
    // [[10, 20], [30, 40]]
    std::vector<std::vector<std::size_t>> table = {{10, 20}, {30, 40}};
    auto result = statcpp::odds_ratio(table);

    // OR = (10*40)/(20*30) = 400/600 = 2/3
    EXPECT_NEAR(result.odds_ratio, 2.0/3.0, 1e-10);
    EXPECT_NEAR(result.log_odds_ratio, std::log(2.0/3.0), 1e-10);
    EXPECT_GT(result.se_log_odds_ratio, 0);
    EXPECT_LT(result.ci_lower, result.odds_ratio);
    EXPECT_GT(result.ci_upper, result.odds_ratio);
}

/**
 * @brief Tests odds ratio calculation using direct cell values
 * @test Verifies that odds ratio can be computed directly from four cell counts
 */
TEST(CategoricalOddsRatioTest, DirectValues) {
    auto result = statcpp::odds_ratio(10, 20, 30, 40);
    EXPECT_NEAR(result.odds_ratio, 2.0/3.0, 1e-10);
}

/**
 * @brief Tests error handling for zero cells in odds ratio calculation
 * @test Verifies that odds ratio throws exception when any cell count is zero
 */
TEST(CategoricalOddsRatioTest, ZeroCell) {
    std::vector<std::vector<std::size_t>> table = {{0, 20}, {30, 40}};
    EXPECT_THROW(statcpp::odds_ratio(table), std::invalid_argument);
}

// ============================================================================
// Relative Risk Tests
// ============================================================================

/**
 * @brief Tests basic relative risk calculation
 * @test Verifies that relative risk and its confidence interval are correctly computed from 2x2 table
 */
TEST(RelativeRiskTest, Basic) {
    // Exposed: 20 disease, 80 no disease (n=100)
    // Unexposed: 10 disease, 90 no disease (n=100)
    std::vector<std::vector<std::size_t>> table = {{20, 80}, {10, 90}};
    auto result = statcpp::relative_risk(table);

    // RR = (20/100) / (10/100) = 2.0
    EXPECT_NEAR(result.relative_risk, 2.0, 1e-10);
    EXPECT_GT(result.se_log_relative_risk, 0);
    EXPECT_LT(result.ci_lower, result.relative_risk);
    EXPECT_GT(result.ci_upper, result.relative_risk);
}

/**
 * @brief Tests relative risk calculation using direct cell values
 * @test Verifies that relative risk can be computed directly from four cell counts
 */
TEST(RelativeRiskTest, DirectValues) {
    auto result = statcpp::relative_risk(20, 80, 10, 90);
    EXPECT_NEAR(result.relative_risk, 2.0, 1e-10);
}

// ============================================================================
// Risk Difference Tests
// ============================================================================

/**
 * @brief Tests basic risk difference calculation
 * @test Verifies that risk difference and standard error are correctly computed from 2x2 table
 */
TEST(RiskDifferenceTest, Basic) {
    // Exposed: 20/100 = 0.2
    // Unexposed: 10/100 = 0.1
    std::vector<std::vector<std::size_t>> table = {{20, 80}, {10, 90}};
    auto result = statcpp::risk_difference(table);

    // RD = 0.2 - 0.1 = 0.1
    EXPECT_NEAR(result.risk_difference, 0.1, 1e-10);
    EXPECT_GT(result.se, 0);
}

// ============================================================================
// Number Needed to Treat Tests
// ============================================================================

/**
 * @brief Tests number needed to treat calculation
 * @test Verifies that NNT is correctly computed as reciprocal of risk difference
 */
TEST(NNTTest, Basic) {
    // RD = 0.1 -> NNT = 10
    std::vector<std::vector<std::size_t>> table = {{20, 80}, {10, 90}};
    double nnt = statcpp::number_needed_to_treat(table);
    EXPECT_NEAR(nnt, 10.0, 1e-10);
}
