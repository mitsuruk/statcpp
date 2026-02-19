/**
 * @file example_frequency_distribution.cpp
 * @brief Sample code for statcpp::frequency_distribution.hpp
 *
 * This file demonstrates the usage of frequency distribution functions
 * provided in frequency_distribution.hpp through practical examples.
 *
 * [Provided Functions]
 * - frequency_table()               : Frequency table (sorted, all info included)
 * - frequency_count()               : Frequency count (fast, unordered_map version)
 * - relative_frequency()            : Relative frequency
 * - cumulative_frequency()          : Cumulative frequency
 * - cumulative_relative_frequency() : Cumulative relative frequency
 *
 * [Compilation]
 * g++ -std=c++17 -I/path/to/statcpp/include example_frequency_distribution.cpp -o example_frequency_distribution
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

// statcpp frequency distribution header
#include "statcpp/frequency_distribution.hpp"

// ============================================================================
// Helper functions for displaying results
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

template <typename T>
void print_data(const std::string& label, const std::vector<T>& data) {
    std::cout << label << ": ";
    for (const auto& d : data) std::cout << d << " ";
    std::cout << "\n";
}

// ============================================================================
// 1. frequency_table() - Frequency Table
// ============================================================================

/**
 * @brief Example of frequency_table() usage
 *
 * [Purpose]
 * frequency_table() calculates frequency, relative frequency, cumulative frequency,
 * and cumulative relative frequency for each value all at once,
 * returning a sorted frequency table.
 *
 * [Return Value Structure]
 * frequency_table_result<T>:
 *   - entries: vector<frequency_entry<T>>
 *     - value: Value
 *     - count: Frequency (occurrence count)
 *     - relative_frequency: Relative frequency (proportion)
 *     - cumulative_count: Cumulative frequency
 *     - cumulative_relative_frequency: Cumulative relative frequency
 *   - total_count: Total data count
 *
 * [Use Cases]
 * - Understanding data distribution
 * - Creating histograms
 * - Summarizing categorical data
 */
void example_frequency_table() {
    print_section("1. frequency_table() - Frequency Table");

    // Test scores (discrete values)
    std::vector<int> scores = {80, 85, 90, 80, 75, 85, 90, 95, 80, 85,
                               85, 90, 80, 75, 85, 90, 80, 85, 80, 90};

    std::cout << "Test score data:\n";
    print_data("Data", scores);
    std::cout << "\n";

    auto table = statcpp::frequency_table(scores.begin(), scores.end());

    std::cout << "Frequency Table:\n";
    std::cout << std::setw(8) << "Value"
              << std::setw(10) << "Freq"
              << std::setw(12) << "Rel Freq"
              << std::setw(12) << "Cum Freq"
              << std::setw(15) << "Cum Rel Freq" << "\n";
    std::cout << std::string(57, '-') << "\n";

    for (const auto& entry : table.entries) {
        std::cout << std::setw(8) << entry.value
                  << std::setw(10) << entry.count
                  << std::setw(12) << entry.relative_frequency
                  << std::setw(12) << entry.cumulative_count
                  << std::setw(15) << entry.cumulative_relative_frequency << "\n";
    }
    std::cout << std::string(57, '-') << "\n";
    std::cout << "Total: " << table.total_count << " records\n";
}

// ============================================================================
// 2. frequency_count() - Frequency Count (Fast Version)
// ============================================================================

/**
 * @brief Example of frequency_count() usage
 *
 * [Purpose]
 * frequency_count() quickly calculates only the occurrence count for each value.
 * Results are returned as unordered_map, so they are not sorted.
 *
 * [Features]
 * - Faster than frequency_table() (no relative frequency calculations)
 * - Results are not sorted
 * - Appropriate when only frequency is needed
 */
void example_frequency_count() {
    print_section("2. frequency_count() - Frequency Count (Fast Version)");

    std::vector<std::string> responses = {
        "Agree", "Disagree", "Agree", "Neutral", "Agree",
        "Disagree", "Agree", "Agree", "Disagree", "Neutral"
    };

    std::cout << "Survey response data:\n";
    for (const auto& r : responses) std::cout << r << " ";
    std::cout << "\n\n";

    auto freq = statcpp::frequency_count(responses.begin(), responses.end());

    std::cout << "Frequency:\n";
    // Sort for display (unordered_map has undefined order)
    std::map<std::string, std::size_t> sorted_freq(freq.begin(), freq.end());
    for (const auto& [value, count] : sorted_freq) {
        std::cout << "  " << value << ": " << count << " times\n";
    }
}

// ============================================================================
// 3. relative_frequency() - Relative Frequency
// ============================================================================

/**
 * @brief Example of relative_frequency() usage
 *
 * [Purpose]
 * relative_frequency() calculates the relative frequency (proportion) for each value.
 * Relative frequency = frequency of value / total data count
 *
 * [Formula]
 * Relative frequency(x) = count(x) / n
 *
 * [Use Cases]
 * - Comparing data as proportions
 * - Probability distribution estimation
 * - When percentage display is needed
 */
void example_relative_frequency() {
    print_section("3. relative_frequency() - Relative Frequency");

    std::vector<int> dice_rolls = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4,
                                   5, 6, 1, 2, 3, 4, 5, 6, 6, 6};

    print_data("Dice roll results", dice_rolls);
    std::cout << "Number of trials: " << dice_rolls.size() << "\n\n";

    auto rel_freq = statcpp::relative_frequency(dice_rolls.begin(), dice_rolls.end());

    // Sort for display
    std::map<int, double> sorted_freq(rel_freq.begin(), rel_freq.end());

    std::cout << "Relative Frequency:\n";
    std::cout << std::setw(8) << "Face" << std::setw(15) << "Rel Freq"
              << std::setw(15) << "Percent" << "\n";
    std::cout << std::string(38, '-') << "\n";

    for (const auto& [value, freq] : sorted_freq) {
        std::cout << std::setw(8) << value
                  << std::setw(15) << freq
                  << std::setw(14) << (freq * 100) << "%\n";
    }

    std::cout << "\nTheoretical value (fair die): 16.67% each\n";
    std::cout << "-> 6 appears more often, suggesting possible bias\n";
}

// ============================================================================
// 4. cumulative_frequency() - Cumulative Frequency
// ============================================================================

/**
 * @brief Example of cumulative_frequency() usage
 *
 * [Purpose]
 * cumulative_frequency() calculates the cumulative frequency up to each value.
 * Results are sorted in ascending order of values.
 *
 * [Formula]
 * Cumulative frequency(x) = Sum of count(y) for all y <= x
 *
 * [Use Cases]
 * - Finding "how many data points are <= x"
 * - Approximating cumulative distribution function
 * - Percentile calculations
 */
void example_cumulative_frequency() {
    print_section("4. cumulative_frequency() - Cumulative Frequency");

    std::vector<int> ages = {20, 25, 30, 35, 40, 25, 30, 35, 30, 30,
                             25, 35, 40, 45, 30, 35, 25, 30, 35, 40};

    print_data("Participant ages", ages);
    std::cout << "\n";

    auto cum_freq = statcpp::cumulative_frequency(ages.begin(), ages.end());

    std::cout << "Cumulative Frequency:\n";
    std::cout << std::setw(10) << "Age <=" << std::setw(15) << "Cum Count" << "\n";
    std::cout << std::string(25, '-') << "\n";

    for (const auto& [value, cum] : cum_freq) {
        std::cout << std::setw(8) << value << " yrs"
                  << std::setw(12) << cum << " people\n";
    }

    std::cout << "\n-> Participants aged 35 or younger: " << cum_freq[4].second << " people\n";
}

// ============================================================================
// 5. cumulative_relative_frequency() - Cumulative Relative Frequency
// ============================================================================

/**
 * @brief Example of cumulative_relative_frequency() usage
 *
 * [Purpose]
 * cumulative_relative_frequency() calculates the cumulative relative frequency
 * up to each value.
 * Returns values equivalent to empirical cumulative distribution function (ECDF).
 *
 * [Formula]
 * Cumulative relative frequency(x) = Cumulative frequency(x) / n
 *
 * [Use Cases]
 * - Finding "what percentage of data is <= x"
 * - Creating empirical cumulative distribution function (ECDF)
 * - Percentile rank calculations
 */
void example_cumulative_relative_frequency() {
    print_section("5. cumulative_relative_frequency() - Cumulative Relative Frequency");

    std::vector<int> test_scores = {60, 65, 70, 75, 80, 85, 90, 95,
                                    70, 75, 80, 80, 85, 85, 85, 90};

    print_data("Test scores", test_scores);
    std::cout << "Number of examinees: " << test_scores.size() << "\n\n";

    auto cum_rel = statcpp::cumulative_relative_frequency(
        test_scores.begin(), test_scores.end());

    std::cout << "Cumulative Relative Frequency:\n";
    std::cout << std::setw(10) << "Score <="
              << std::setw(15) << "Cum Ratio"
              << std::setw(15) << "Percent" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& [value, cum_rel_freq] : cum_rel) {
        std::cout << std::setw(8) << value << " pts"
                  << std::setw(15) << cum_rel_freq
                  << std::setw(14) << (cum_rel_freq * 100) << "%\n";
    }

    // Display specific percentile positions
    std::cout << "\n[Interpretation Examples]\n";
    for (const auto& [value, cum_rel_freq] : cum_rel) {
        if (cum_rel_freq >= 0.5 && cum_rel_freq < 0.7) {
            std::cout << "- Scores <= " << value << " represent "
                      << (cum_rel_freq * 100) << "% of total (top "
                      << ((1.0 - cum_rel_freq) * 100) << "%)\n";
        }
    }
}

// ============================================================================
// 6. Example Using Lambda (Projection)
// ============================================================================

/**
 * @brief Advanced example using lambda (projection)
 *
 * Calculate frequency distribution for struct members.
 */
void example_projection() {
    print_section("6. Example Using Lambda (Projection)");

    struct Employee {
        std::string name;
        std::string department;
        int years;
    };

    std::vector<Employee> employees = {
        {"Alice", "Sales", 3},
        {"Bob", "Development", 5},
        {"Charlie", "Sales", 2},
        {"Diana", "Development", 8},
        {"Eve", "HR", 4},
        {"Frank", "Sales", 5},
        {"Grace", "Development", 5},
        {"Henry", "HR", 1},
        {"Ivy", "Development", 3},
        {"Jack", "Sales", 7}
    };

    std::cout << "Employee data:\n";
    for (const auto& e : employees) {
        std::cout << "  " << e.name << ": " << e.department
                  << ", " << e.years << " years\n";
    }

    // Frequency by department
    print_subsection("Frequency by Department");
    auto dept_proj = [](const Employee& e) { return e.department; };
    auto dept_freq = statcpp::frequency_count(
        employees.begin(), employees.end(), dept_proj);

    std::map<std::string, std::size_t> sorted_dept(dept_freq.begin(), dept_freq.end());
    for (const auto& [dept, count] : sorted_dept) {
        std::cout << "  " << dept << ": " << count << " people\n";
    }

    // Relative frequency of years of service
    print_subsection("Relative Frequency of Years of Service");
    auto years_proj = [](const Employee& e) { return e.years; };
    auto years_rel = statcpp::relative_frequency(
        employees.begin(), employees.end(), years_proj);

    std::map<int, double> sorted_years(years_rel.begin(), years_rel.end());
    for (const auto& [years, freq] : sorted_years) {
        std::cout << "  " << years << " years: "
                  << (freq * 100) << "%\n";
    }

    // Frequency table of years of service
    print_subsection("Frequency Table of Years of Service");
    auto years_table = statcpp::frequency_table(
        employees.begin(), employees.end(), years_proj);

    std::cout << std::setw(10) << "Years"
              << std::setw(10) << "Count"
              << std::setw(15) << "Cum Ratio" << "\n";
    std::cout << std::string(35, '-') << "\n";

    for (const auto& entry : years_table.entries) {
        std::cout << std::setw(8) << entry.value << " yrs"
                  << std::setw(10) << entry.count
                  << std::setw(14) << (entry.cumulative_relative_frequency * 100) << "%\n";
    }
}

// ============================================================================
// 7. Frequency Distribution for String Data
// ============================================================================

/**
 * @brief Example with string data
 */
void example_string_data() {
    print_section("7. Frequency Distribution for String Data");

    std::vector<std::string> blood_types = {
        "A", "B", "O", "AB", "A", "A", "O", "B", "A", "O",
        "A", "B", "A", "O", "A", "AB", "A", "O", "B", "A"
    };

    std::cout << "Blood type data:\n";
    for (const auto& bt : blood_types) std::cout << bt << " ";
    std::cout << "\n\n";

    auto table = statcpp::frequency_table(blood_types.begin(), blood_types.end());

    std::cout << "Blood Type Distribution:\n";
    std::cout << std::setw(8) << "Type"
              << std::setw(10) << "Count"
              << std::setw(15) << "Proportion" << "\n";
    std::cout << std::string(33, '-') << "\n";

    for (const auto& entry : table.entries) {
        std::cout << std::setw(8) << entry.value
                  << std::setw(10) << entry.count
                  << std::setw(14) << (entry.relative_frequency * 100) << "%\n";
    }

    std::cout << "\nBlood type distribution in Japan (reference):\n";
    std::cout << "  Type A: ~40%, Type O: ~30%, Type B: ~20%, Type AB: ~10%\n";
}

// ============================================================================
// Summary
// ============================================================================

/**
 * @brief Display summary
 */
void print_summary() {
    print_section("Summary: Frequency Distribution Functions");

    std::cout << R"(
+-----------------------------------+------------------------------------+
| Function                          | Description                        |
+-----------------------------------+------------------------------------+
| frequency_table()                 | Complete frequency table (sorted)  |
|                                   | Includes freq, rel freq, cum freq  |
+-----------------------------------+------------------------------------+
| frequency_count()                 | Frequency only (fast,              |
|                                   | unordered_map)                     |
+-----------------------------------+------------------------------------+
| relative_frequency()              | Relative frequency (proportions)   |
+-----------------------------------+------------------------------------+
| cumulative_frequency()            | Cumulative frequency               |
+-----------------------------------+------------------------------------+
| cumulative_relative_frequency()   | Cumulative relative frequency      |
|                                   | (ECDF)                             |
+-----------------------------------+------------------------------------+

[Return Value Structures]
- frequency_table_result<T>:
    - entries: vector<frequency_entry<T>>
    - total_count: Total data count

- frequency_entry<T>:
    - value: Value
    - count: Frequency
    - relative_frequency: Relative frequency
    - cumulative_count: Cumulative frequency
    - cumulative_relative_frequency: Cumulative relative frequency

[When to Use Which]
- Need all information -> frequency_table()
- Need only frequency (fast) -> frequency_count()
- Need proportions -> relative_frequency()
- Need cumulative info -> cumulative_* functions

[Supported Data Types]
- Numeric types (int, double, etc.)
- String type (std::string)
- Any type with comparison operator < defined
)";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // Run each example
    example_frequency_table();
    example_frequency_count();
    example_relative_frequency();
    example_cumulative_frequency();
    example_cumulative_relative_frequency();
    example_projection();
    example_string_data();

    // Display summary
    print_summary();

    return 0;
}
