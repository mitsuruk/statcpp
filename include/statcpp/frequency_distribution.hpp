/**
 * @file frequency_distribution.hpp
 * @brief Frequency distribution table creation and frequency-related calculations
 *
 * Provides functions for frequency distribution including frequency tables,
 * relative frequency, cumulative frequency, etc.
 * Useful for analyzing categorical and discrete data.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace statcpp {

// ============================================================================
// Return value structures
// ============================================================================

/**
 * @brief Frequency table entry
 *
 * Holds the frequency, relative frequency, cumulative frequency,
 * and cumulative relative frequency for each value.
 *
 * @tparam T Value type
 */
template <typename T>
struct frequency_entry {
    T value;                                ///< Value
    std::size_t count;                      ///< Frequency
    double relative_frequency;              ///< Relative frequency
    std::size_t cumulative_count;           ///< Cumulative frequency
    double cumulative_relative_frequency;   ///< Cumulative relative frequency
};

/**
 * @brief Frequency table result
 *
 * Holds overall information about the frequency table.
 *
 * @tparam T Value type
 */
template <typename T>
struct frequency_table_result {
    std::vector<frequency_entry<T>> entries;  ///< Frequency table entries
    std::size_t total_count;                   ///< Total frequency
};

// ============================================================================
// Frequency Table
// ============================================================================

/**
 * @brief Create frequency table (returned with sorted keys)
 *
 * Calculates frequency, relative frequency, cumulative frequency,
 * and cumulative relative frequency for each value.
 * Results are sorted in ascending order by key.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @return Frequency table result
 */
template <typename Iterator>
auto frequency_table(Iterator first, Iterator last)
    -> frequency_table_result<typename std::iterator_traits<Iterator>::value_type>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    if (first == last) {
        return {{}, 0};
    }

    // Count frequencies (std::map maintains sorted order)
    std::map<value_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[*it];
    }

    std::size_t total = static_cast<std::size_t>(std::distance(first, last));
    double total_d = static_cast<double>(total);

    // Build result
    std::vector<frequency_entry<value_type>> entries;
    entries.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        entries.push_back({
            pair.first,
            pair.second,
            static_cast<double>(pair.second) / total_d,
            cumulative,
            static_cast<double>(cumulative) / total_d
        });
    }

    return {std::move(entries), total};
}

/**
 * @brief Frequency table with projection
 *
 * Creates a frequency table of the results after applying a projection function to each element.
 * Results are sorted in ascending order by key.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param proj Projection function
 * @return Frequency table result
 */
template <typename Iterator, typename Projection>
auto frequency_table(Iterator first, Iterator last, Projection proj)
    -> frequency_table_result<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    if (first == last) {
        return {{}, 0};
    }

    std::map<result_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[std::invoke(proj, *it)];
    }

    std::size_t total = static_cast<std::size_t>(std::distance(first, last));
    double total_d = static_cast<double>(total);

    std::vector<frequency_entry<result_type>> entries;
    entries.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        entries.push_back({
            pair.first,
            pair.second,
            static_cast<double>(pair.second) / total_d,
            cumulative,
            static_cast<double>(cumulative) / total_d
        });
    }

    return {std::move(entries), total};
}

// ============================================================================
// Frequency Count
// ============================================================================

/**
 * @brief Return frequency for each value (unordered_map version, fast)
 *
 * Counts the number of occurrences of each value.
 * Sort order is not guaranteed, but operates faster.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @return Map of values and their frequencies
 */
template <typename Iterator>
auto frequency_count(Iterator first, Iterator last)
    -> std::unordered_map<typename std::iterator_traits<Iterator>::value_type, std::size_t>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    std::unordered_map<value_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[*it];
    }
    return freq;
}

/**
 * @brief Frequency count with projection
 *
 * Returns the frequency of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param proj Projection function
 * @return Map of values and their frequencies
 */
template <typename Iterator, typename Projection>
auto frequency_count(Iterator first, Iterator last, Projection proj)
    -> std::unordered_map<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, std::size_t>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    std::unordered_map<result_type, std::size_t> freq;
    for (auto it = first; it != last; ++it) {
        ++freq[std::invoke(proj, *it)];
    }
    return freq;
}

// ============================================================================
// Relative Frequency
// ============================================================================

/**
 * @brief Return relative frequency for each value
 *
 * Calculates the relative frequency by dividing each value's frequency by the total count.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @return Map of values and their relative frequencies
 */
template <typename Iterator>
auto relative_frequency(Iterator first, Iterator last)
    -> std::unordered_map<typename std::iterator_traits<Iterator>::value_type, double>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto freq = frequency_count(first, last);
    double total = static_cast<double>(n);

    std::unordered_map<value_type, double> result;
    result.reserve(freq.size());
    for (const auto& pair : freq) {
        result[pair.first] = static_cast<double>(pair.second) / total;
    }
    return result;
}

/**
 * @brief Relative frequency with projection
 *
 * Returns the relative frequency of results after applying a projection function to each element.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param proj Projection function
 * @return Map of values and their relative frequencies
 */
template <typename Iterator, typename Projection>
auto relative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::unordered_map<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, double>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto freq = frequency_count(first, last, proj);
    double total = static_cast<double>(n);

    std::unordered_map<result_type, double> result;
    result.reserve(freq.size());
    for (const auto& pair : freq) {
        result[pair.first] = static_cast<double>(pair.second) / total;
    }
    return result;
}

// ============================================================================
// Cumulative Frequency
// ============================================================================

/**
 * @brief Return cumulative frequency (for sorted values)
 *
 * Calculates the cumulative frequency up to each value.
 * Results are sorted in ascending order by value.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @return Vector of (value, cumulative frequency) pairs
 */
template <typename Iterator>
auto cumulative_frequency(Iterator first, Iterator last)
    -> std::vector<std::pair<typename std::iterator_traits<Iterator>::value_type, std::size_t>>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    if (first == last) {
        return {};
    }

    // Get frequency in sorted order
    std::map<value_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[*it];
    }

    std::vector<std::pair<value_type, std::size_t>> result;
    result.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        result.emplace_back(pair.first, cumulative);
    }

    return result;
}

/**
 * @brief Cumulative frequency with projection
 *
 * Returns the cumulative frequency of results after applying a projection function to each element.
 * Results are sorted in ascending order by value.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param proj Projection function
 * @return Vector of (value, cumulative frequency) pairs
 */
template <typename Iterator, typename Projection>
auto cumulative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::vector<std::pair<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, std::size_t>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    if (first == last) {
        return {};
    }

    std::map<result_type, std::size_t> freq_map;
    for (auto it = first; it != last; ++it) {
        ++freq_map[std::invoke(proj, *it)];
    }

    std::vector<std::pair<result_type, std::size_t>> result;
    result.reserve(freq_map.size());

    std::size_t cumulative = 0;
    for (const auto& pair : freq_map) {
        cumulative += pair.second;
        result.emplace_back(pair.first, cumulative);
    }

    return result;
}

// ============================================================================
// Cumulative Relative Frequency
// ============================================================================

/**
 * @brief Return cumulative relative frequency (for sorted values)
 *
 * Calculates the cumulative relative frequency up to each value.
 * Results are sorted in ascending order by value.
 *
 * @tparam Iterator Iterator type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @return Vector of (value, cumulative relative frequency) pairs
 */
template <typename Iterator>
auto cumulative_relative_frequency(Iterator first, Iterator last)
    -> std::vector<std::pair<typename std::iterator_traits<Iterator>::value_type, double>>
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto cum_freq = cumulative_frequency(first, last);
    double total = static_cast<double>(n);

    std::vector<std::pair<value_type, double>> result;
    result.reserve(cum_freq.size());

    for (const auto& pair : cum_freq) {
        result.emplace_back(pair.first, static_cast<double>(pair.second) / total);
    }

    return result;
}

/**
 * @brief Cumulative relative frequency with projection
 *
 * Returns the cumulative relative frequency of results after applying a projection function to each element.
 * Results are sorted in ascending order by value.
 *
 * @tparam Iterator Iterator type
 * @tparam Projection Projection function type
 * @param first Beginning iterator
 * @param last Ending iterator
 * @param proj Projection function
 * @return Vector of (value, cumulative relative frequency) pairs
 */
template <typename Iterator, typename Projection>
auto cumulative_relative_frequency(Iterator first, Iterator last, Projection proj)
    -> std::vector<std::pair<std::invoke_result_t<Projection,
           typename std::iterator_traits<Iterator>::value_type>, double>>
{
    using result_type = std::invoke_result_t<Projection,
        typename std::iterator_traits<Iterator>::value_type>;

    auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return {};
    }

    auto cum_freq = cumulative_frequency(first, last, proj);
    double total = static_cast<double>(n);

    std::vector<std::pair<result_type, double>> result;
    result.reserve(cum_freq.size());

    for (const auto& pair : cum_freq) {
        result.emplace_back(pair.first, static_cast<double>(pair.second) / total);
    }

    return result;
}

} // namespace statcpp
