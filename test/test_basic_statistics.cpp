#include <gtest/gtest.h>
#include "statcpp/basic_statistics.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// ============================================================================
// Sum Tests
// ============================================================================

/**
 * @brief Test sum calculation with integer vector
 * @test Verify sum function returns correct total for integer values
 */
TEST(SumTest, IntegerVector) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_EQ(statcpp::sum(data.begin(), data.end()), 15);
}

/**
 * @brief Test sum calculation with double vector
 * @test Verify sum function returns correct total for floating-point values
 */
TEST(SumTest, DoubleVector) {
    std::vector<double> data = {1.5, 2.5, 3.0};
    EXPECT_DOUBLE_EQ(statcpp::sum(data.begin(), data.end()), 7.0);
}

/**
 * @brief Test sum calculation with empty range
 * @test Verify sum function returns 0 for empty input
 */
TEST(SumTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_EQ(statcpp::sum(data.begin(), data.end()), 0);
}

/**
 * @brief Test sum calculation with projection function
 * @test Verify sum function works with custom projection to extract values
 */
TEST(SumTest, Projection) {
    struct Item { int value; };
    std::vector<Item> data = {{1}, {2}, {3}};
    auto result = statcpp::sum(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_EQ(result, 6);
}

// ============================================================================
// Count Tests
// ============================================================================

/**
 * @brief Test count function with non-empty range
 * @test Verify count function returns correct number of elements
 */
TEST(CountTest, NonEmpty) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_EQ(statcpp::count(data.begin(), data.end()), 5u);
}

/**
 * @brief Test count function with empty range
 * @test Verify count function returns 0 for empty input
 */
TEST(CountTest, Empty) {
    std::vector<int> data;
    EXPECT_EQ(statcpp::count(data.begin(), data.end()), 0u);
}

// ============================================================================
// Mean Tests
// ============================================================================

/**
 * @brief Test mean calculation with integer vector
 * @test Verify mean function returns correct arithmetic mean for integers
 */
TEST(MeanTest, IntegerVector) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(statcpp::mean(data.begin(), data.end()), 3.0);
}

/**
 * @brief Test mean calculation with double vector
 * @test Verify mean function returns correct arithmetic mean for floating-point values
 */
TEST(MeanTest, DoubleVector) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(statcpp::mean(data.begin(), data.end()), 2.5);
}

/**
 * @brief Test mean calculation with empty range
 * @test Verify mean function throws exception for empty input
 */
TEST(MeanTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::mean(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test mean calculation with projection function
 * @test Verify mean function works with custom projection to extract values
 */
TEST(MeanTest, Projection) {
    struct Item { double value; };
    std::vector<Item> data = {{2.0}, {4.0}, {6.0}};
    auto result = statcpp::mean(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 4.0);
}

// ============================================================================
// Median Tests
// ============================================================================

/**
 * @brief Test median calculation with odd number of elements
 * @test Verify median function returns middle value for odd-sized dataset
 */
TEST(MedianTest, OddCount) {
    std::vector<int> data = {1, 2, 3, 4, 5};  // sorted
    EXPECT_DOUBLE_EQ(statcpp::median(data.begin(), data.end()), 3.0);
}

/**
 * @brief Test median calculation with even number of elements
 * @test Verify median function returns average of two middle values for even-sized dataset
 */
TEST(MedianTest, EvenCount) {
    std::vector<int> data = {1, 2, 3, 4};  // sorted
    EXPECT_DOUBLE_EQ(statcpp::median(data.begin(), data.end()), 2.5);
}

/**
 * @brief Test median calculation with single element
 * @test Verify median function returns the element itself for single-element dataset
 */
TEST(MedianTest, SingleElement) {
    std::vector<int> data = {42};
    EXPECT_DOUBLE_EQ(statcpp::median(data.begin(), data.end()), 42.0);
}

/**
 * @brief Test median calculation with empty range
 * @test Verify median function throws exception for empty input
 */
TEST(MedianTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::median(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test median calculation with projection function
 * @test Verify median function works with custom projection to extract values
 */
TEST(MedianTest, Projection) {
    struct Item { int value; };
    std::vector<Item> data = {{1}, {2}, {3}, {4}, {5}};
    auto result = statcpp::median(data.begin(), data.end(), [](const Item& i) { return i.value; });
    EXPECT_DOUBLE_EQ(result, 3.0);
}

// ============================================================================
// Mode Tests
// ============================================================================

/**
 * @brief Test mode calculation with single most frequent value
 * @test Verify mode function returns the most frequent value in dataset
 */
TEST(ModeTest, SingleMode) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4};
    EXPECT_EQ(statcpp::mode(data.begin(), data.end()), 3);
}

/**
 * @brief Test mode calculation when all values are identical
 * @test Verify mode function returns the repeated value when all elements are the same
 */
TEST(ModeTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5};
    EXPECT_EQ(statcpp::mode(data.begin(), data.end()), 5);
}

/**
 * @brief Test mode calculation with empty range
 * @test Verify mode function throws exception for empty input
 */
TEST(ModeTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::mode(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test mode calculation with projection function
 * @test Verify mode function works with custom projection to extract values
 */
TEST(ModeTest, Projection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {2}, {2}, {3}};
    auto result = statcpp::mode(data.begin(), data.end(), [](const Item& i) { return i.category; });
    EXPECT_EQ(result, 2);
}

/**
 * @brief Test mode calculation when multiple values have same frequency
 * @test Verify mode function returns the smallest value when multiple modes exist
 */
TEST(ModeTest, MultipleModes) {
    // When frequencies are equal, return the minimum value
    std::vector<int> data = {1, 1, 2, 2, 3, 3};
    EXPECT_EQ(statcpp::mode(data.begin(), data.end()), 1);
}

// ============================================================================
// Modes Tests (Multiple most frequent values)
// ============================================================================

/**
 * @brief Test modes calculation with single most frequent value
 * @test Verify modes function returns vector with single mode
 */
TEST(ModesTest, SingleMode) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4};
    auto result = statcpp::modes(data.begin(), data.end());
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 3);
}

/**
 * @brief Test modes calculation with multiple most frequent values
 * @test Verify modes function returns all values with highest frequency
 */
TEST(ModesTest, MultipleModes) {
    std::vector<int> data = {1, 1, 2, 2, 3, 3};
    auto result = statcpp::modes(data.begin(), data.end());
    EXPECT_EQ(result.size(), 3u);
    // Sorted in ascending order
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
}

/**
 * @brief Test modes calculation with exactly two modes
 * @test Verify modes function returns both values when two have equal highest frequency
 */
TEST(ModesTest, TwoModes) {
    std::vector<int> data = {1, 2, 2, 3, 3, 4};
    auto result = statcpp::modes(data.begin(), data.end());
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
}

/**
 * @brief Test modes calculation when all values are identical
 * @test Verify modes function returns single value when all elements are the same
 */
TEST(ModesTest, AllSame) {
    std::vector<int> data = {5, 5, 5, 5};
    auto result = statcpp::modes(data.begin(), data.end());
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 5);
}

/**
 * @brief Test modes calculation when all values are unique
 * @test Verify modes function returns all elements when each has frequency 1
 */
TEST(ModesTest, AllUnique) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto result = statcpp::modes(data.begin(), data.end());
    // All have frequency 1, so all elements are returned
    EXPECT_EQ(result.size(), 5u);
}

/**
 * @brief Test modes calculation with empty range
 * @test Verify modes function throws exception for empty input
 */
TEST(ModesTest, EmptyRange) {
    std::vector<int> data;
    EXPECT_THROW(statcpp::modes(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test modes calculation with projection function
 * @test Verify modes function works with custom projection to extract values
 */
TEST(ModesTest, Projection) {
    struct Item { int category; };
    std::vector<Item> data = {{1}, {1}, {2}, {2}, {3}};
    auto result = statcpp::modes(data.begin(), data.end(), [](const Item& i) { return i.category; });
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
}

// ============================================================================
// Geometric Mean Tests
// ============================================================================

/**
 * @brief Test geometric mean calculation with positive values
 * @test Verify geometric mean function computes nth root of product correctly
 */
TEST(GeometricMeanTest, PositiveValues) {
    std::vector<double> data = {1.0, 2.0, 4.0, 8.0};
    // geometric mean = (1 * 2 * 4 * 8)^(1/4) = 64^0.25 = 2.828...
    double expected = std::pow(64.0, 0.25);
    EXPECT_NEAR(statcpp::geometric_mean(data.begin(), data.end()), expected, 1e-10);
}

/**
 * @brief Test geometric mean calculation when all values are 1
 * @test Verify geometric mean function returns 1.0 when all elements are 1
 */
TEST(GeometricMeanTest, AllOnes) {
    std::vector<double> data = {1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(statcpp::geometric_mean(data.begin(), data.end()), 1.0);
}

/**
 * @brief Test geometric mean calculation with empty range
 * @test Verify geometric mean function throws exception for empty input
 */
TEST(GeometricMeanTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::geometric_mean(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test geometric mean calculation with zero value
 * @test Verify geometric mean function throws exception when dataset contains zero
 */
TEST(GeometricMeanTest, ZeroValue) {
    std::vector<double> data = {1.0, 0.0, 2.0};
    EXPECT_THROW(statcpp::geometric_mean(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test geometric mean calculation with negative value
 * @test Verify geometric mean function throws exception when dataset contains negative values
 */
TEST(GeometricMeanTest, NegativeValue) {
    std::vector<double> data = {1.0, -2.0, 3.0};
    EXPECT_THROW(statcpp::geometric_mean(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Harmonic Mean Tests
// ============================================================================

/**
 * @brief Test harmonic mean calculation with positive values
 * @test Verify harmonic mean function computes reciprocal of arithmetic mean of reciprocals
 */
TEST(HarmonicMeanTest, PositiveValues) {
    std::vector<double> data = {1.0, 2.0, 4.0};
    // harmonic mean = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 = 1.714...
    double expected = 3.0 / 1.75;
    EXPECT_NEAR(statcpp::harmonic_mean(data.begin(), data.end()), expected, 1e-10);
}

/**
 * @brief Test harmonic mean calculation when all values are identical
 * @test Verify harmonic mean function returns the repeated value when all elements are the same
 */
TEST(HarmonicMeanTest, AllSame) {
    std::vector<double> data = {5.0, 5.0, 5.0};
    EXPECT_DOUBLE_EQ(statcpp::harmonic_mean(data.begin(), data.end()), 5.0);
}

/**
 * @brief Test harmonic mean calculation with empty range
 * @test Verify harmonic mean function throws exception for empty input
 */
TEST(HarmonicMeanTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::harmonic_mean(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test harmonic mean calculation with zero value
 * @test Verify harmonic mean function throws exception when dataset contains zero
 */
TEST(HarmonicMeanTest, ZeroValue) {
    std::vector<double> data = {1.0, 0.0, 2.0};
    EXPECT_THROW(statcpp::harmonic_mean(data.begin(), data.end()), std::invalid_argument);
}

/**
 * @brief Test harmonic mean calculation with a near-zero value (subnormal)
 * @test Verify harmonic mean function throws exception for subnormal values
 *       that would cause 1/val to overflow to infinity
 */
TEST(HarmonicMeanTest, NearZeroValue) {
    // std::numeric_limits<double>::min() / 2 is a subnormal, smaller than the
    // smallest positive normal double, so 1/val would overflow.
    std::vector<double> data = {1.0, std::numeric_limits<double>::min() / 2.0, 2.0};
    EXPECT_THROW(statcpp::harmonic_mean(data.begin(), data.end()), std::invalid_argument);
}

// ============================================================================
// Logarithmic Mean Tests
// ============================================================================

/**
 * @brief Test logarithmic mean calculation for distinct positive values
 * @test Verify logarithmic mean formula (y - x) / (ln(y) - ln(x))
 */
TEST(LogarithmicMeanTest, DistinctValues) {
    // L(1, e) = (e - 1) / (ln(e) - ln(1)) = (e - 1) / 1 = e - 1
    EXPECT_NEAR(statcpp::logarithmic_mean(1.0, std::exp(1.0)), std::exp(1.0) - 1.0, 1e-10);
}

/**
 * @brief Test logarithmic mean calculation when both values are equal
 * @test Verify logarithmic mean returns the value itself when x == y
 */
TEST(LogarithmicMeanTest, EqualValues) {
    EXPECT_DOUBLE_EQ(statcpp::logarithmic_mean(3.0, 3.0), 3.0);
}

/**
 * @brief Test logarithmic mean for large equal values (relative near-equality check)
 * @test Verify that values very close in relative terms are handled without 0/0 computation.
 *       Previously, an absolute threshold would fail for large magnitudes.
 */
TEST(LogarithmicMeanTest, LargeNearlyEqualValues) {
    // x and y differ by a relative amount smaller than 1e-10, so the function
    // should return a value very close to x without computing (y-x)/(ln y - ln x).
    double x = 1e15;
    double y = x + x * 5e-11;  // relative difference ~5e-11 < 1e-10
    double result = statcpp::logarithmic_mean(x, y);
    EXPECT_NEAR(result, x, x * 1e-9);
}

/**
 * @brief Test logarithmic mean throws for non-positive arguments
 * @test Verify that non-positive arguments raise std::invalid_argument
 */
TEST(LogarithmicMeanTest, NonPositiveArgument) {
    EXPECT_THROW(statcpp::logarithmic_mean(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(statcpp::logarithmic_mean(1.0, -1.0), std::invalid_argument);
}

// ============================================================================
// Trimmed Mean Tests
// ============================================================================

/**
 * @brief Test trimmed mean calculation with 10% trimming
 * @test Verify trimmed mean function removes specified proportion from each end
 */
TEST(TrimmedMeanTest, TenPercent) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // sorted
    // 10% trim: remove 1 element from each end -> [2,3,4,5,6,7,8,9]
    // mean = (2+3+4+5+6+7+8+9) / 8 = 44/8 = 5.5
    EXPECT_DOUBLE_EQ(statcpp::trimmed_mean(data.begin(), data.end(), 0.1), 5.5);
}

/**
 * @brief Test trimmed mean calculation with zero trimming
 * @test Verify trimmed mean function equals regular mean when trim proportion is 0
 */
TEST(TrimmedMeanTest, ZeroTrim) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(statcpp::trimmed_mean(data.begin(), data.end(), 0.0), 3.0);
}

/**
 * @brief Test trimmed mean calculation with empty range
 * @test Verify trimmed mean function throws exception for empty input
 */
TEST(TrimmedMeanTest, EmptyRange) {
    std::vector<double> data;
    EXPECT_THROW(statcpp::trimmed_mean(data.begin(), data.end(), 0.1), std::invalid_argument);
}

/**
 * @brief Test trimmed mean calculation with invalid proportion
 * @test Verify trimmed mean function throws exception for invalid trim proportion
 */
TEST(TrimmedMeanTest, InvalidProportion) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    EXPECT_THROW(statcpp::trimmed_mean(data.begin(), data.end(), 0.5), std::invalid_argument);
    EXPECT_THROW(statcpp::trimmed_mean(data.begin(), data.end(), -0.1), std::invalid_argument);
}

/**
 * @brief Test trimmed mean calculation when trimming would remove all elements
 * @test Verify trimmed mean function handles edge case near 50% trimming
 */
TEST(TrimmedMeanTest, AllElementsTrimmed) {
    std::vector<double> data = {1, 2};
    // 50% would trim all elements, but 0.5 is invalid
    // With proportion = 0.49, trim_count = floor(2 * 0.49) = 0
    EXPECT_NO_THROW(statcpp::trimmed_mean(data.begin(), data.end(), 0.49));
}

// ============================================================================
// Array Support Tests
// ============================================================================

/**
 * @brief Test statistical functions with std::array
 * @test Verify functions work correctly with standard library array container
 */
TEST(ArraySupportTest, StdArray) {
    std::array<int, 5> data = {1, 2, 3, 4, 5};
    EXPECT_EQ(statcpp::sum(data.begin(), data.end()), 15);
    EXPECT_DOUBLE_EQ(statcpp::mean(data.begin(), data.end()), 3.0);
}

/**
 * @brief Test statistical functions with C-style array
 * @test Verify functions work correctly with C-style arrays
 */
TEST(ArraySupportTest, CArray) {
    int data[] = {1, 2, 3, 4, 5};
    EXPECT_EQ(statcpp::sum(std::begin(data), std::end(data)), 15);
    EXPECT_DOUBLE_EQ(statcpp::mean(std::begin(data), std::end(data)), 3.0);
}

/**
 * @brief Test statistical functions with pointer range
 * @test Verify functions work correctly with raw pointer iterators
 */
TEST(ArraySupportTest, PointerRange) {
    int data[] = {1, 2, 3, 4, 5};
    int* first = data;
    int* last = data + 5;
    EXPECT_EQ(statcpp::sum(first, last), 15);
    EXPECT_DOUBLE_EQ(statcpp::mean(first, last), 3.0);
}
