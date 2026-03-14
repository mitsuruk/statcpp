/**
 * @file example_data_wrangling.cpp
 * @brief Sample code for data wrangling
 *
 * Demonstrates usage of missing value handling, data transformation,
 * filtering, aggregation, and other data preprocessing functions.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include "statcpp/data_wrangling.hpp"

int main() {
    std::cout << "=== Data Wrangling Examples ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Missing Value Handling
    // ============================================================================
    std::cout << "\n1. Missing Value Handling" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> data_with_na = {1.0, 2.0, statcpp::NA, 4.0, 5.0, statcpp::NA, 7.0};

    std::cout << "Original data: [";
    for (std::size_t i = 0; i < data_with_na.size(); ++i) {
        if (statcpp::is_na(data_with_na[i])) {
            std::cout << "NA";
        } else {
            std::cout << data_with_na[i];
        }
        if (i + 1 < data_with_na.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Drop NA
    auto dropped = statcpp::dropna(data_with_na);
    std::cout << "\nAfter dropping NA: [";
    for (std::size_t i = 0; i < dropped.size(); ++i) {
        std::cout << dropped[i];
        if (i + 1 < dropped.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Fill NA with 0
    auto filled_zero = statcpp::fillna(data_with_na, 0.0);
    std::cout << "Fill with 0: [";
    for (std::size_t i = 0; i < filled_zero.size(); ++i) {
        std::cout << filled_zero[i];
        if (i + 1 < filled_zero.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Fill NA with mean
    auto filled_mean = statcpp::fillna_mean(data_with_na);
    std::cout << "Fill with mean: [";
    for (std::size_t i = 0; i < filled_mean.size(); ++i) {
        std::cout << filled_mean[i];
        if (i + 1 < filled_mean.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Linear interpolation
    auto interpolated = statcpp::fillna_interpolate(data_with_na);
    std::cout << "Linear interpolation: [";
    for (std::size_t i = 0; i < interpolated.size(); ++i) {
        std::cout << interpolated[i];
        if (i + 1 < interpolated.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 2. Filtering
    // ============================================================================
    std::cout << "\n2. Filtering" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Filter even numbers only
    auto evens = statcpp::filter(numbers, [](double x) { return static_cast<int>(x) % 2 == 0; });
    std::cout << "Even numbers: [";
    for (std::size_t i = 0; i < evens.size(); ++i) {
        std::cout << evens[i];
        if (i + 1 < evens.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Filter by range (3 to 7 inclusive)
    auto range_filtered = statcpp::filter_range(numbers, 3.0, 7.0);
    std::cout << "Range [3, 7]: [";
    for (std::size_t i = 0; i < range_filtered.size(); ++i) {
        std::cout << range_filtered[i];
        if (i + 1 < range_filtered.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 3. Transformations
    // ============================================================================
    std::cout << "\n3. Transformations" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> transform_data = {1.0, 4.0, 9.0, 16.0, 25.0};

    // Square root transformation
    auto sqrt_data = statcpp::sqrt_transform(transform_data);
    std::cout << "Square root transform: [";
    for (std::size_t i = 0; i < sqrt_data.size(); ++i) {
        std::cout << sqrt_data[i];
        if (i + 1 < sqrt_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Log transformation
    auto log_data = statcpp::log_transform(transform_data);
    std::cout << "Log transform: [";
    for (std::size_t i = 0; i < log_data.size(); ++i) {
        std::cout << log_data[i];
        if (i + 1 < log_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Box-Cox transformation (lambda = 0.5)
    auto boxcox_data = statcpp::boxcox_transform(transform_data, 0.5);
    std::cout << "Box-Cox transform (lambda=0.5): [";
    for (std::size_t i = 0; i < boxcox_data.size(); ++i) {
        std::cout << boxcox_data[i];
        if (i + 1 < boxcox_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Rank transformation
    std::vector<double> rank_data = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};
    auto ranks = statcpp::rank_transform(rank_data);
    std::cout << "\nRank transformation:" << std::endl;
    std::cout << "Original data: [";
    for (std::size_t i = 0; i < rank_data.size(); ++i) {
        std::cout << rank_data[i];
        if (i + 1 < rank_data.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Ranks:         [";
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        std::cout << ranks[i];
        if (i + 1 < ranks.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 4. Grouping and Aggregation
    // ============================================================================
    std::cout << "\n4. Grouping and Aggregation" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::string> groups = {"A", "B", "A", "B", "A", "B", "C", "C"};
    std::vector<double> values = {10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0};

    auto mean_result = statcpp::group_mean(groups, values);
    std::cout << "Group means:" << std::endl;
    for (std::size_t i = 0; i < mean_result.keys.size(); ++i) {
        std::cout << "  Group " << mean_result.keys[i] << ": " << mean_result.values[i] << std::endl;
    }

    auto sum_result = statcpp::group_sum(groups, values);
    std::cout << "\nGroup sums:" << std::endl;
    for (std::size_t i = 0; i < sum_result.keys.size(); ++i) {
        std::cout << "  Group " << sum_result.keys[i] << ": " << sum_result.values[i] << std::endl;
    }

    auto count_result = statcpp::group_count(groups, values);
    std::cout << "\nGroup counts:" << std::endl;
    for (std::size_t i = 0; i < count_result.keys.size(); ++i) {
        std::cout << "  Group " << count_result.keys[i] << ": " << count_result.values[i] << std::endl;
    }

    // ============================================================================
    // 5. Sorting
    // ============================================================================
    std::cout << "\n5. Sorting" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> unsorted = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0};

    auto sorted_asc = statcpp::sort_values(unsorted, true);
    std::cout << "Ascending sort: [";
    for (std::size_t i = 0; i < sorted_asc.size(); ++i) {
        std::cout << sorted_asc[i];
        if (i + 1 < sorted_asc.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto indices = statcpp::argsort(unsorted, true);
    std::cout << "Sort indices: [";
    for (std::size_t i = 0; i < indices.size(); ++i) {
        std::cout << indices[i];
        if (i + 1 < indices.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 6. Sampling
    // ============================================================================
    std::cout << "\n6. Sampling" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    statcpp::set_seed(42);

    std::vector<int> population = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Sampling with replacement
    auto sample_with = statcpp::sample_with_replacement(population, 5);
    std::cout << "With replacement (n=5): [";
    for (std::size_t i = 0; i < sample_with.size(); ++i) {
        std::cout << sample_with[i];
        if (i + 1 < sample_with.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Sampling without replacement
    auto sample_without = statcpp::sample_without_replacement(population, 5);
    std::cout << "Without replacement (n=5): [";
    for (std::size_t i = 0; i < sample_without.size(); ++i) {
        std::cout << sample_without[i];
        if (i + 1 < sample_without.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Stratified sampling
    std::vector<std::string> strata = {"A", "A", "A", "A", "B", "B", "B", "C", "C", "C"};
    std::vector<int> strata_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto stratified = statcpp::stratified_sample(strata, strata_values, 0.5);
    std::cout << "Stratified sampling (50%): [";
    for (std::size_t i = 0; i < stratified.size(); ++i) {
        std::cout << stratified[i];
        if (i + 1 < stratified.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 7. Duplicate Handling
    // ============================================================================
    std::cout << "\n7. Duplicate Handling" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<int> duplicates = {1, 2, 2, 3, 3, 3, 4, 5, 5};

    auto unique = statcpp::drop_duplicates(duplicates);
    std::cout << "Drop duplicates: [";
    for (std::size_t i = 0; i < unique.size(); ++i) {
        std::cout << unique[i];
        if (i + 1 < unique.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto counts = statcpp::value_counts(duplicates);
    std::cout << "\nValue counts:" << std::endl;
    for (const auto& pair : counts) {
        std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }

    auto dup_values = statcpp::get_duplicates(duplicates);
    std::cout << "\nDuplicated values: [";
    for (std::size_t i = 0; i < dup_values.size(); ++i) {
        std::cout << dup_values[i];
        if (i + 1 < dup_values.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 8. Rolling Statistics
    // ============================================================================
    std::cout << "\n8. Rolling Statistics" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> time_series = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    auto rolling_mean_3 = statcpp::rolling_mean(time_series, 3);
    std::cout << "Rolling mean (window=3): [";
    for (std::size_t i = 0; i < rolling_mean_3.size(); ++i) {
        std::cout << rolling_mean_3[i];
        if (i + 1 < rolling_mean_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto rolling_std_3 = statcpp::rolling_std(time_series, 3);
    std::cout << "Rolling std dev (window=3): [";
    for (std::size_t i = 0; i < rolling_std_3.size(); ++i) {
        std::cout << rolling_std_3[i];
        if (i + 1 < rolling_std_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto rolling_sum_3 = statcpp::rolling_sum(time_series, 3);
    std::cout << "Rolling sum (window=3): [";
    for (std::size_t i = 0; i < rolling_sum_3.size(); ++i) {
        std::cout << rolling_sum_3[i];
        if (i + 1 < rolling_sum_3.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 9. Categorical Encoding
    // ============================================================================
    std::cout << "\n9. Categorical Encoding" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<std::string> categories = {"red", "blue", "green", "red", "green", "blue"};

    auto encoded = statcpp::label_encode(categories);
    std::cout << "Label encoding:" << std::endl;
    std::cout << "  Encoded: [";
    for (std::size_t i = 0; i < encoded.encoded.size(); ++i) {
        std::cout << encoded.encoded[i];
        if (i + 1 < encoded.encoded.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Classes: [";
    for (std::size_t i = 0; i < encoded.classes.size(); ++i) {
        std::cout << encoded.classes[i];
        if (i + 1 < encoded.classes.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Binning (equal width)
    std::vector<double> continuous = {1.5, 2.3, 4.7, 5.8, 7.2, 9.1, 10.5};
    auto bins_width = statcpp::bin_equal_width(continuous, 3);
    std::cout << "\nBinning (equal width, n=3):" << std::endl;
    std::cout << "  Original data: [";
    for (std::size_t i = 0; i < continuous.size(); ++i) {
        std::cout << continuous[i];
        if (i + 1 < continuous.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Bins:          [";
    for (std::size_t i = 0; i < bins_width.size(); ++i) {
        std::cout << bins_width[i];
        if (i + 1 < bins_width.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // ============================================================================
    // 10. Data Validation
    // ============================================================================
    std::cout << "\n10. Data Validation" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> test_data = {1.0, 2.0, statcpp::NA, 4.0, -1.0,
                                      std::numeric_limits<double>::infinity()};

    auto validation = statcpp::validate_data(test_data, true, true, true);
    std::cout << "Validation (all allowed):" << std::endl;
    std::cout << "  Valid: " << (validation.is_valid ? "Yes" : "No") << std::endl;
    std::cout << "  Missing values: " << validation.n_missing << std::endl;
    std::cout << "  Infinite values: " << validation.n_infinite << std::endl;
    std::cout << "  Negative values: " << validation.n_negative << std::endl;

    auto strict_validation = statcpp::validate_data(test_data, false, false, false);
    std::cout << "\nValidation (strict):" << std::endl;
    std::cout << "  Valid: " << (strict_validation.is_valid ? "Yes" : "No") << std::endl;

    std::vector<double> range_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    bool in_range = statcpp::validate_range(range_data, 0.0, 10.0);
    std::cout << "\nRange validation [0, 10]: " << (in_range ? "Pass" : "Fail") << std::endl;

    std::cout << "\n=== Examples completed successfully ===" << std::endl;

    return 0;
}
