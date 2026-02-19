#include <gtest/gtest.h>
#include "statcpp/frequency_distribution.hpp"
#include <vector>
#include <string>

// ============================================================================
// Frequency Table Tests
// ============================================================================

/**
 * @brief Tests basic frequency table construction and calculations
 * @test Verifies that frequency table correctly computes counts, relative frequencies, and cumulative values
 */
TEST(FrequencyTableTest, BasicTable) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    auto result = statcpp::frequency_table(data.begin(), data.end());

    EXPECT_EQ(result.total_count, 10u);
    EXPECT_EQ(result.entries.size(), 4u);

    // Check entries (should be sorted by value)
    EXPECT_EQ(result.entries[0].value, 1);
    EXPECT_EQ(result.entries[0].count, 1u);
    EXPECT_DOUBLE_EQ(result.entries[0].relative_frequency, 0.1);
    EXPECT_EQ(result.entries[0].cumulative_count, 1u);
    EXPECT_DOUBLE_EQ(result.entries[0].cumulative_relative_frequency, 0.1);

    EXPECT_EQ(result.entries[1].value, 2);
    EXPECT_EQ(result.entries[1].count, 2u);
    EXPECT_DOUBLE_EQ(result.entries[1].relative_frequency, 0.2);
    EXPECT_EQ(result.entries[1].cumulative_count, 3u);
    EXPECT_DOUBLE_EQ(result.entries[1].cumulative_relative_frequency, 0.3);

    EXPECT_EQ(result.entries[2].value, 3);
    EXPECT_EQ(result.entries[2].count, 3u);
    EXPECT_DOUBLE_EQ(result.entries[2].relative_frequency, 0.3);
    EXPECT_EQ(result.entries[2].cumulative_count, 6u);
    EXPECT_DOUBLE_EQ(result.entries[2].cumulative_relative_frequency, 0.6);

    EXPECT_EQ(result.entries[3].value, 4);
    EXPECT_EQ(result.entries[3].count, 4u);
    EXPECT_DOUBLE_EQ(result.entries[3].relative_frequency, 0.4);
    EXPECT_EQ(result.entries[3].cumulative_count, 10u);
    EXPECT_DOUBLE_EQ(result.entries[3].cumulative_relative_frequency, 1.0);
}

/**
 * @brief Tests frequency table construction with empty data range
 * @test Verifies that empty range produces empty frequency table
 */
TEST(FrequencyTableTest, EmptyRange) {
    std::vector<int> data;
    auto result = statcpp::frequency_table(data.begin(), data.end());
    EXPECT_EQ(result.total_count, 0u);
    EXPECT_TRUE(result.entries.empty());
}

/**
 * @brief Tests frequency table construction with single element
 * @test Verifies that single element produces correct frequency table
 */
TEST(FrequencyTableTest, SingleElement) {
    std::vector<int> data = {42};
    auto result = statcpp::frequency_table(data.begin(), data.end());
    EXPECT_EQ(result.total_count, 1u);
    EXPECT_EQ(result.entries.size(), 1u);
    EXPECT_EQ(result.entries[0].value, 42);
    EXPECT_EQ(result.entries[0].count, 1u);
}

/**
 * @brief Tests frequency table construction with all identical values
 * @test Verifies that all identical values produce single entry with relative frequency of 1.0
 */
TEST(FrequencyTableTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5, 5};
    auto result = statcpp::frequency_table(data.begin(), data.end());
    EXPECT_EQ(result.total_count, 5u);
    EXPECT_EQ(result.entries.size(), 1u);
    EXPECT_EQ(result.entries[0].value, 5);
    EXPECT_EQ(result.entries[0].count, 5u);
    EXPECT_DOUBLE_EQ(result.entries[0].relative_frequency, 1.0);
}

/**
 * @brief Tests frequency table construction with string data
 * @test Verifies that frequency table works correctly with non-numeric string data
 */
TEST(FrequencyTableTest, StringData) {
    std::vector<std::string> data = {"a", "b", "b", "c", "c", "c"};
    auto result = statcpp::frequency_table(data.begin(), data.end());
    EXPECT_EQ(result.total_count, 6u);
    EXPECT_EQ(result.entries.size(), 3u);
    EXPECT_EQ(result.entries[0].value, "a");
    EXPECT_EQ(result.entries[1].value, "b");
    EXPECT_EQ(result.entries[2].value, "c");
}

// ============================================================================
// Frequency Count Tests
// ============================================================================

/**
 * @brief Tests basic frequency count calculation
 * @test Verifies that frequency count produces correct counts for each unique value
 */
TEST(FrequencyCountTest, BasicCount) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3};
    auto result = statcpp::frequency_count(data.begin(), data.end());
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[1], 1u);
    EXPECT_EQ(result[2], 2u);
    EXPECT_EQ(result[3], 3u);
}

/**
 * @brief Tests frequency count calculation with empty data range
 * @test Verifies that empty range produces empty frequency count map
 */
TEST(FrequencyCountTest, EmptyRange) {
    std::vector<int> data;
    auto result = statcpp::frequency_count(data.begin(), data.end());
    EXPECT_TRUE(result.empty());
}

/**
 * @brief Tests frequency count calculation with projection function
 * @test Verifies that frequency count works correctly with custom projection function
 */
TEST(FrequencyCountTest, Projection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {2}, {2}, {3}, {3}, {3}};
    auto result = statcpp::frequency_count(data.begin(), data.end(),
                                           [](const Item& i) { return i.category; });
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[1], 1u);
    EXPECT_EQ(result[2], 2u);
    EXPECT_EQ(result[3], 3u);
}

// ============================================================================
// Relative Frequency Tests
// ============================================================================

/**
 * @brief Tests basic relative frequency calculation
 * @test Verifies that relative frequencies are correctly computed as proportions
 */
TEST(RelativeFrequencyTest, BasicRelative) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    auto result = statcpp::relative_frequency(data.begin(), data.end());
    EXPECT_EQ(result.size(), 4u);
    EXPECT_DOUBLE_EQ(result[1], 0.1);
    EXPECT_DOUBLE_EQ(result[2], 0.2);
    EXPECT_DOUBLE_EQ(result[3], 0.3);
    EXPECT_DOUBLE_EQ(result[4], 0.4);
}

/**
 * @brief Tests relative frequency calculation with empty data range
 * @test Verifies that empty range produces empty relative frequency map
 */
TEST(RelativeFrequencyTest, EmptyRange) {
    std::vector<int> data;
    auto result = statcpp::relative_frequency(data.begin(), data.end());
    EXPECT_TRUE(result.empty());
}

/**
 * @brief Tests that relative frequencies sum to one
 * @test Verifies that all relative frequencies sum to 1.0
 */
TEST(RelativeFrequencyTest, SumToOne) {
    std::vector<int> data = {1, 1, 2, 2, 2, 3};
    auto result = statcpp::relative_frequency(data.begin(), data.end());
    double sum = 0.0;
    for (const auto& [val, freq] : result) {
        sum += freq;
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ============================================================================
// Cumulative Frequency Tests
// ============================================================================

/**
 * @brief Tests basic cumulative frequency calculation
 * @test Verifies that cumulative frequencies are correctly accumulated in sorted order
 */
TEST(CumulativeFrequencyTest, BasicCumulative) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3};
    auto result = statcpp::cumulative_frequency(data.begin(), data.end());
    EXPECT_EQ(result.size(), 3u);

    // Should be sorted by value
    EXPECT_EQ(result[0].first, 1);
    EXPECT_EQ(result[0].second, 1u);

    EXPECT_EQ(result[1].first, 2);
    EXPECT_EQ(result[1].second, 3u);

    EXPECT_EQ(result[2].first, 3);
    EXPECT_EQ(result[2].second, 6u);
}

/**
 * @brief Tests cumulative frequency calculation with empty data range
 * @test Verifies that empty range produces empty cumulative frequency vector
 */
TEST(CumulativeFrequencyTest, EmptyRange) {
    std::vector<int> data;
    auto result = statcpp::cumulative_frequency(data.begin(), data.end());
    EXPECT_TRUE(result.empty());
}

/**
 * @brief Tests that final cumulative frequency equals total count
 * @test Verifies that the last cumulative frequency equals the total number of elements
 */
TEST(CumulativeFrequencyTest, FinalEqualsTotal) {
    std::vector<int> data = {1, 1, 2, 3, 3, 3, 4};
    auto result = statcpp::cumulative_frequency(data.begin(), data.end());
    EXPECT_EQ(result.back().second, 7u);
}

// ============================================================================
// Cumulative Relative Frequency Tests
// ============================================================================

/**
 * @brief Tests basic cumulative relative frequency calculation
 * @test Verifies that cumulative relative frequencies are correctly accumulated as proportions
 */
TEST(CumulativeRelativeFrequencyTest, BasicCumulative) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    auto result = statcpp::cumulative_relative_frequency(data.begin(), data.end());
    EXPECT_EQ(result.size(), 4u);

    EXPECT_EQ(result[0].first, 1);
    EXPECT_DOUBLE_EQ(result[0].second, 0.1);

    EXPECT_EQ(result[1].first, 2);
    EXPECT_DOUBLE_EQ(result[1].second, 0.3);

    EXPECT_EQ(result[2].first, 3);
    EXPECT_DOUBLE_EQ(result[2].second, 0.6);

    EXPECT_EQ(result[3].first, 4);
    EXPECT_DOUBLE_EQ(result[3].second, 1.0);
}

/**
 * @brief Tests cumulative relative frequency calculation with empty data range
 * @test Verifies that empty range produces empty cumulative relative frequency vector
 */
TEST(CumulativeRelativeFrequencyTest, EmptyRange) {
    std::vector<int> data;
    auto result = statcpp::cumulative_relative_frequency(data.begin(), data.end());
    EXPECT_TRUE(result.empty());
}

/**
 * @brief Tests that final cumulative relative frequency equals one
 * @test Verifies that the last cumulative relative frequency equals 1.0
 */
TEST(CumulativeRelativeFrequencyTest, FinalEqualsOne) {
    std::vector<int> data = {1, 1, 2, 3, 3, 3, 4};
    auto result = statcpp::cumulative_relative_frequency(data.begin(), data.end());
    EXPECT_NEAR(result.back().second, 1.0, 1e-10);
}

// ============================================================================
// Projection Tests
// ============================================================================

/**
 * @brief Tests frequency table construction with projection function
 * @test Verifies that frequency table works correctly with custom projection function
 */
TEST(FrequencyProjectionTest, TableWithProjection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {2}, {2}, {3}, {3}, {3}};
    auto result = statcpp::frequency_table(data.begin(), data.end(),
                                           [](const Item& i) { return i.category; });
    EXPECT_EQ(result.total_count, 6u);
    EXPECT_EQ(result.entries.size(), 3u);
    EXPECT_EQ(result.entries[0].value, 1);
    EXPECT_EQ(result.entries[1].value, 2);
    EXPECT_EQ(result.entries[2].value, 3);
}

/**
 * @brief Tests relative frequency calculation with projection function
 * @test Verifies that relative frequency works correctly with custom projection function
 */
TEST(FrequencyProjectionTest, RelativeWithProjection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {1}, {2}, {2}, {2}};
    auto result = statcpp::relative_frequency(data.begin(), data.end(),
                                              [](const Item& i) { return i.category; });
    EXPECT_DOUBLE_EQ(result[1], 0.4);
    EXPECT_DOUBLE_EQ(result[2], 0.6);
}

/**
 * @brief Tests cumulative frequency calculation with projection function
 * @test Verifies that cumulative frequency works correctly with custom projection function
 */
TEST(FrequencyProjectionTest, CumulativeWithProjection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {2}, {2}, {3}, {3}, {3}};
    auto result = statcpp::cumulative_frequency(data.begin(), data.end(),
                                                [](const Item& i) { return i.category; });
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].second, 1u);
    EXPECT_EQ(result[1].second, 3u);
    EXPECT_EQ(result[2].second, 6u);
}

/**
 * @brief Tests cumulative relative frequency calculation with projection function
 * @test Verifies that cumulative relative frequency works correctly with custom projection function
 */
TEST(FrequencyProjectionTest, CumulativeRelativeWithProjection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {2}, {2}, {3}, {3}, {3}};
    auto result = statcpp::cumulative_relative_frequency(data.begin(), data.end(),
                                                         [](const Item& i) { return i.category; });
    EXPECT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0].second, 1.0 / 6.0, 1e-10);
    EXPECT_NEAR(result[1].second, 3.0 / 6.0, 1e-10);
    EXPECT_NEAR(result[2].second, 1.0, 1e-10);
}
