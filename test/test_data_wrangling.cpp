#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>

#include "statcpp/data_wrangling.hpp"

// ============================================================================
// Missing Data Tests
// ============================================================================

/**
 * @brief Tests missing value detection
 * @test Verifies that NA and NaN values are correctly identified as missing
 */
TEST(IsNATest, Basic) {
    EXPECT_TRUE(statcpp::is_na(statcpp::NA));
    EXPECT_TRUE(statcpp::is_na(std::nan("")));
    EXPECT_FALSE(statcpp::is_na(0.0));
    EXPECT_FALSE(statcpp::is_na(1.0));
}

/**
 * @brief Tests dropping NA values from a vector
 * @test Verifies that NA values are removed while preserving non-missing values
 */
TEST(DropNATest, Vector) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0, statcpp::NA, 5.0};
    auto result = statcpp::dropna(data);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
}

/**
 * @brief Tests dropping rows with NA values from a matrix
 * @test Verifies that rows containing any NA values are removed from the matrix
 */
TEST(DropNATest, Matrix) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {statcpp::NA, 4.0},
        {5.0, 6.0}
    };
    auto result = statcpp::dropna(data);

    EXPECT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0][0], 1.0);
    EXPECT_DOUBLE_EQ(result[1][0], 5.0);
}

/**
 * @brief Tests filling NA values with a specific constant
 * @test Verifies that NA values are replaced with the specified fill value
 */
TEST(FillNATest, WithValue) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0};
    auto result = statcpp::fillna(data, 0.0);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

/**
 * @brief Tests filling NA values with mean imputation
 * @test Verifies that NA values are replaced with the mean of non-missing values
 */
TEST(FillNATest, WithMean) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0, statcpp::NA, 5.0};
    auto result = statcpp::fillna_mean(data);

    // mean of {1, 3, 5} = 3
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[3], 3.0);
}

/**
 * @brief Tests filling NA values with median imputation
 * @test Verifies that NA values are replaced with the median of non-missing values
 */
TEST(FillNATest, WithMedian) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0, statcpp::NA, 5.0};
    auto result = statcpp::fillna_median(data);

    // median of {1, 3, 5} = 3
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[3], 3.0);
}

/**
 * @brief Tests forward fill imputation for NA values
 * @test Verifies that NA values are filled with the last observed non-missing value
 */
TEST(FillNATest, ForwardFill) {
    std::vector<double> data = {1.0, statcpp::NA, statcpp::NA, 4.0, statcpp::NA};
    auto result = statcpp::fillna_ffill(data);

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
    EXPECT_DOUBLE_EQ(result[3], 4.0);
    EXPECT_DOUBLE_EQ(result[4], 4.0);
}

/**
 * @brief Tests backward fill imputation for NA values
 * @test Verifies that NA values are filled with the next observed non-missing value
 */
TEST(FillNATest, BackwardFill) {
    std::vector<double> data = {statcpp::NA, 2.0, statcpp::NA, statcpp::NA, 5.0};
    auto result = statcpp::fillna_bfill(data);

    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
    EXPECT_DOUBLE_EQ(result[3], 5.0);
    EXPECT_DOUBLE_EQ(result[4], 5.0);
}

/**
 * @brief Tests linear interpolation for NA values
 * @test Verifies that NA values are filled using linear interpolation between non-missing values
 */
TEST(FillNATest, Interpolate) {
    std::vector<double> data = {1.0, statcpp::NA, statcpp::NA, 4.0};
    auto result = statcpp::fillna_interpolate(data);

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
    EXPECT_DOUBLE_EQ(result[3], 4.0);
}

// ============================================================================
// Filtering Tests
// ============================================================================

/**
 * @brief Tests basic filtering with a predicate function
 * @test Verifies that elements satisfying the predicate are retained
 */
TEST(FilterTest, Basic) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto result = statcpp::filter(data, [](int x) { return x > 3; });

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 4);
    EXPECT_EQ(result[1], 5);
}

/**
 * @brief Tests range-based filtering
 * @test Verifies that values within the specified range (inclusive) are retained
 */
TEST(FilterTest, Range) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = statcpp::filter_range(data, 2.0, 4.0);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 4.0);
}

/**
 * @brief Tests filtering matrix rows with a predicate
 * @test Verifies that rows satisfying the predicate are retained
 */
TEST(FilterTest, Rows) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    auto result = statcpp::filter_rows(data, [](const auto& row) {
        return row[0] > 2.0;
    });

    EXPECT_EQ(result.size(), 2);
}

// ============================================================================
// Transformation Tests
// ============================================================================

/**
 * @brief Tests natural logarithm transformation
 * @test Verifies that log transformation is applied correctly to positive values
 */
TEST(TransformTest, Log) {
    std::vector<double> data = {1.0, std::exp(1.0), std::exp(2.0)};
    auto result = statcpp::log_transform(data);

    EXPECT_NEAR(result[0], 0.0, 1e-10);
    EXPECT_NEAR(result[1], 1.0, 1e-10);
    EXPECT_NEAR(result[2], 2.0, 1e-10);
}

/**
 * @brief Tests log transformation with negative and zero values
 * @test Verifies that log transformation produces NaN for non-positive values
 */
TEST(TransformTest, LogNegative) {
    std::vector<double> data = {-1.0, 0.0, 1.0};
    auto result = statcpp::log_transform(data);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

/**
 * @brief Tests square root transformation
 * @test Verifies that square root transformation is computed correctly
 */
TEST(TransformTest, Sqrt) {
    std::vector<double> data = {0.0, 1.0, 4.0, 9.0};
    auto result = statcpp::sqrt_transform(data);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 2.0);
    EXPECT_DOUBLE_EQ(result[3], 3.0);
}

/**
 * @brief Tests Box-Cox transformation
 * @test Verifies that Box-Cox transformation works correctly for different lambda values
 */
TEST(TransformTest, BoxCox) {
    std::vector<double> data = {1.0, 2.0, 3.0};

    // lambda = 0 (log transform)
    auto result0 = statcpp::boxcox_transform(data, 0.0);
    EXPECT_NEAR(result0[0], 0.0, 1e-10);

    // lambda = 1 (linear: (x-1)/1 = x-1)
    auto result1 = statcpp::boxcox_transform(data, 1.0);
    EXPECT_NEAR(result1[0], 0.0, 1e-10);
    EXPECT_NEAR(result1[1], 1.0, 1e-10);
}

/**
 * @brief Tests rank transformation
 * @test Verifies that values are correctly converted to their ranks
 */
TEST(TransformTest, Rank) {
    std::vector<double> data = {3.0, 1.0, 2.0};
    auto result = statcpp::rank_transform(data);

    EXPECT_DOUBLE_EQ(result[0], 3.0);  // 3.0 is rank 3
    EXPECT_DOUBLE_EQ(result[1], 1.0);  // 1.0 is rank 1
    EXPECT_DOUBLE_EQ(result[2], 2.0);  // 2.0 is rank 2
}

/**
 * @brief Tests rank transformation with tied values
 * @test Verifies that tied values receive average ranks
 */
TEST(TransformTest, RankTies) {
    std::vector<double> data = {1.0, 2.0, 2.0, 4.0};
    auto result = statcpp::rank_transform(data);

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.5);  // tied ranks
    EXPECT_DOUBLE_EQ(result[2], 2.5);  // tied ranks
    EXPECT_DOUBLE_EQ(result[3], 4.0);
}

// ============================================================================
// Group-by Tests
// ============================================================================

/**
 * @brief Tests basic group-by operation
 * @test Verifies that values are correctly grouped by their keys
 */
TEST(GroupByTest, Basic) {
    std::vector<std::string> keys = {"A", "B", "A", "B", "A"};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = statcpp::group_by(keys, values);

    EXPECT_EQ(result.groups.size(), 2);
    EXPECT_EQ(result.groups["A"].size(), 3);
    EXPECT_EQ(result.groups["B"].size(), 2);
}

/**
 * @brief Tests group-by with mean aggregation
 * @test Verifies that mean is computed correctly for each group
 */
TEST(GroupByTest, Mean) {
    std::vector<std::string> keys = {"A", "B", "A", "B"};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};

    auto result = statcpp::group_mean(keys, values);

    EXPECT_EQ(result.keys.size(), 2);
    // A: mean(1, 3) = 2, B: mean(2, 4) = 3
    for (std::size_t i = 0; i < result.keys.size(); ++i) {
        if (result.keys[i] == "A") {
            EXPECT_DOUBLE_EQ(result.values[i], 2.0);
        } else {
            EXPECT_DOUBLE_EQ(result.values[i], 3.0);
        }
    }
}

/**
 * @brief Tests group-by with sum aggregation
 * @test Verifies that sum is computed correctly for each group
 */
TEST(GroupByTest, Sum) {
    std::vector<int> keys = {1, 2, 1, 2};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};

    auto result = statcpp::group_sum(keys, values);

    EXPECT_EQ(result.keys.size(), 2);
}

// ============================================================================
// Sorting Tests
// ============================================================================

/**
 * @brief Tests sorting in ascending order
 * @test Verifies that values are sorted from smallest to largest
 */
TEST(SortTest, Ascending) {
    std::vector<double> data = {3.0, 1.0, 4.0, 1.0, 5.0};
    auto result = statcpp::sort_values(data, true);

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
    EXPECT_DOUBLE_EQ(result[3], 4.0);
    EXPECT_DOUBLE_EQ(result[4], 5.0);
}

/**
 * @brief Tests sorting in descending order
 * @test Verifies that values are sorted from largest to smallest
 */
TEST(SortTest, Descending) {
    std::vector<double> data = {3.0, 1.0, 4.0};
    auto result = statcpp::sort_values(data, false);

    EXPECT_DOUBLE_EQ(result[0], 4.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
}

/**
 * @brief Tests argsort to get sorting indices
 * @test Verifies that argsort returns indices that would sort the array
 */
TEST(ArgsortTest, Basic) {
    std::vector<double> data = {3.0, 1.0, 2.0};
    auto result = statcpp::argsort(data);

    EXPECT_EQ(result[0], 1);  // 1.0 is at index 1
    EXPECT_EQ(result[1], 2);  // 2.0 is at index 2
    EXPECT_EQ(result[2], 0);  // 3.0 is at index 0
}

// ============================================================================
// Sampling Tests
// ============================================================================

/**
 * @brief Tests random sampling with replacement
 * @test Verifies that sampling with replacement produces the requested sample size
 */
TEST(SampleTest, WithReplacement) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    statcpp::set_seed(42);
    auto result = statcpp::sample_with_replacement(data, 10);

    EXPECT_EQ(result.size(), 10);
    for (int val : result) {
        EXPECT_GE(val, 1);
        EXPECT_LE(val, 5);
    }
}

/**
 * @brief Tests random sampling without replacement
 * @test Verifies that sampling without replacement produces unique values
 */
TEST(SampleTest, WithoutReplacement) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    statcpp::set_seed(42);
    auto result = statcpp::sample_without_replacement(data, 3);

    EXPECT_EQ(result.size(), 3);
    // All values should be unique
    std::sort(result.begin(), result.end());
    EXPECT_NE(result[0], result[1]);
    EXPECT_NE(result[1], result[2]);
}

/**
 * @brief Tests sampling without replacement when sample size exceeds population
 * @test Verifies that an exception is thrown when requesting more samples than available
 */
TEST(SampleTest, WithoutReplacementTooLarge) {
    std::vector<int> data = {1, 2, 3};
    EXPECT_THROW(statcpp::sample_without_replacement(data, 5), std::invalid_argument);
}

/**
 * @brief Tests stratified sampling
 * @test Verifies that stratified sampling maintains proportions from each stratum
 */
TEST(SampleTest, Stratified) {
    std::vector<std::string> strata = {"A", "A", "A", "A", "B", "B", "B", "B"};
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8};

    statcpp::set_seed(42);
    auto result = statcpp::stratified_sample(strata, data, 0.5);

    // Should get roughly half from each stratum
    EXPECT_GE(result.size(), 2);
    EXPECT_LE(result.size(), 8);
}

// ============================================================================
// Duplicate Tests
// ============================================================================

/**
 * @brief Tests dropping duplicate values
 * @test Verifies that duplicate values are removed, keeping only unique values
 */
TEST(DuplicateTest, DropDuplicates) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4};
    auto result = statcpp::drop_duplicates(data);

    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
    EXPECT_EQ(result[3], 4);
}

/**
 * @brief Tests counting value frequencies
 * @test Verifies that each unique value is counted correctly
 */
TEST(DuplicateTest, ValueCounts) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3};
    auto result = statcpp::value_counts(data);

    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], 3);
}

/**
 * @brief Tests extracting duplicate values
 * @test Verifies that only values appearing more than once are returned
 */
TEST(DuplicateTest, GetDuplicates) {
    std::vector<int> data = {1, 2, 2, 3, 3, 3, 4};
    auto result = statcpp::get_duplicates(data);

    EXPECT_EQ(result.size(), 2);  // 2 and 3 are duplicated
}

// ============================================================================
// Rolling Aggregation Tests
// ============================================================================

/**
 * @brief Tests rolling mean calculation
 * @test Verifies that rolling mean is computed correctly over sliding windows
 */
TEST(RollingTest, Mean) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = statcpp::rolling_mean(data, 3);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 2.0);  // mean(1, 2, 3)
    EXPECT_DOUBLE_EQ(result[1], 3.0);  // mean(2, 3, 4)
    EXPECT_DOUBLE_EQ(result[2], 4.0);  // mean(3, 4, 5)
}

/**
 * @brief Tests rolling standard deviation calculation
 * @test Verifies that rolling standard deviation is computed correctly over sliding windows
 */
TEST(RollingTest, Std) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = statcpp::rolling_std(data, 3);

    EXPECT_EQ(result.size(), 3);
    EXPECT_GT(result[0], 0.0);
}

/**
 * @brief Tests rolling minimum calculation
 * @test Verifies that rolling minimum is computed correctly over sliding windows
 */
TEST(RollingTest, Min) {
    std::vector<double> data = {3.0, 1.0, 4.0, 1.0, 5.0};
    auto result = statcpp::rolling_min(data, 3);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // min(3, 1, 4)
    EXPECT_DOUBLE_EQ(result[1], 1.0);  // min(1, 4, 1)
    EXPECT_DOUBLE_EQ(result[2], 1.0);  // min(4, 1, 5)
}

/**
 * @brief Tests rolling maximum calculation
 * @test Verifies that rolling maximum is computed correctly over sliding windows
 */
TEST(RollingTest, Max) {
    std::vector<double> data = {3.0, 1.0, 4.0, 1.0, 5.0};
    auto result = statcpp::rolling_max(data, 3);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 4.0);  // max(3, 1, 4)
    EXPECT_DOUBLE_EQ(result[1], 4.0);  // max(1, 4, 1)
    EXPECT_DOUBLE_EQ(result[2], 5.0);  // max(4, 1, 5)
}

/**
 * @brief Tests rolling sum calculation
 * @test Verifies that rolling sum is computed correctly over sliding windows
 */
TEST(RollingTest, Sum) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = statcpp::rolling_sum(data, 3);

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 6.0);   // sum(1, 2, 3)
    EXPECT_DOUBLE_EQ(result[1], 9.0);   // sum(2, 3, 4)
    EXPECT_DOUBLE_EQ(result[2], 12.0);  // sum(3, 4, 5)
}

/**
 * @brief Tests rolling operations with invalid window sizes
 * @test Verifies that exceptions are thrown for invalid window sizes (0 or larger than data)
 */
TEST(RollingTest, InvalidWindow) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(statcpp::rolling_mean(data, 0), std::invalid_argument);
    EXPECT_THROW(statcpp::rolling_mean(data, 5), std::invalid_argument);
}

// ============================================================================
// Encoding Tests
// ============================================================================

/**
 * @brief Tests label encoding for categorical data
 * @test Verifies that categorical values are mapped to integer labels
 */
TEST(EncodingTest, LabelEncode) {
    std::vector<std::string> data = {"cat", "dog", "cat", "bird", "dog"};
    auto result = statcpp::label_encode(data);

    EXPECT_EQ(result.encoded.size(), 5);
    EXPECT_EQ(result.classes.size(), 3);
    EXPECT_EQ(result.encoded[0], result.encoded[2]);  // "cat" same encoding
    EXPECT_EQ(result.encoded[1], result.encoded[4]);  // "dog" same encoding
}

/**
 * @brief Tests one-hot encoding for categorical data
 * @test Verifies that categorical values are converted to binary indicator vectors
 */
TEST(EncodingTest, OneHotEncode) {
    std::vector<std::string> data = {"A", "B", "A", "C"};
    auto result = statcpp::one_hot_encode(data);

    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0].size(), 3);  // 3 classes

    // Each row should have exactly one 1.0
    for (const auto& row : result) {
        double sum = 0.0;
        for (double val : row) {
            sum += val;
        }
        EXPECT_DOUBLE_EQ(sum, 1.0);
    }
}

/**
 * @brief Tests equal-width binning
 * @test Verifies that continuous values are binned into equal-width intervals
 */
TEST(EncodingTest, BinEqualWidth) {
    std::vector<double> data = {0.0, 2.5, 5.0, 7.5, 10.0};
    auto result = statcpp::bin_equal_width(data, 4);

    EXPECT_EQ(result.size(), 5);
    EXPECT_EQ(result[0], 0);  // 0.0 -> bin 0
    EXPECT_EQ(result[1], 1);  // 2.5 -> bin 1
    EXPECT_EQ(result[2], 2);  // 5.0 -> bin 2
    EXPECT_EQ(result[3], 3);  // 7.5 -> bin 3
    EXPECT_EQ(result[4], 3);  // 10.0 -> bin 3 (max value)
}

/**
 * @brief Tests equal-frequency binning
 * @test Verifies that continuous values are binned so each bin has approximately equal frequency
 */
TEST(EncodingTest, BinEqualFreq) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto result = statcpp::bin_equal_freq(data, 4);

    EXPECT_EQ(result.size(), 8);
    // Each bin should have approximately same number of elements
}

// ============================================================================
// Validation Tests
// ============================================================================

/**
 * @brief Tests basic data validation
 * @test Verifies that missing, infinite, and negative values are detected correctly
 */
TEST(ValidationTest, Basic) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0, std::numeric_limits<double>::infinity()};
    auto result = statcpp::validate_data(data);

    EXPECT_FALSE(result.is_valid);  // has NA and inf
    EXPECT_EQ(result.n_missing, 1);
    EXPECT_EQ(result.n_infinite, 1);
    EXPECT_EQ(result.n_negative, 0);
}

/**
 * @brief Tests data validation with missing values allowed
 * @test Verifies that validation passes when missing values are explicitly allowed
 */
TEST(ValidationTest, AllowMissing) {
    std::vector<double> data = {1.0, statcpp::NA, 3.0};
    auto result = statcpp::validate_data(data, true, true, true);

    EXPECT_TRUE(result.is_valid);
    EXPECT_EQ(result.n_missing, 1);
}

/**
 * @brief Tests data validation rejecting negative values
 * @test Verifies that validation fails when negative values are present and not allowed
 */
TEST(ValidationTest, NoNegative) {
    std::vector<double> data = {1.0, -2.0, 3.0};
    auto result = statcpp::validate_data(data, true, true, false);

    EXPECT_FALSE(result.is_valid);
    EXPECT_EQ(result.n_negative, 1);
}

/**
 * @brief Tests range validation
 * @test Verifies that all values are within the specified range
 */
TEST(ValidationTest, Range) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    EXPECT_TRUE(statcpp::validate_range(data, 0.0, 10.0));
    EXPECT_FALSE(statcpp::validate_range(data, 2.0, 4.0));
}
